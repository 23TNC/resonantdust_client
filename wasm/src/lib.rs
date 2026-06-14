//! Synchronous wasm surface — what the view's worker hosts.
//!
//! The view never talks to a gate directly; this is the bridge. It owns a
//! `web-sys` WebSocket (whose `send` is synchronous and whose `onmessage` is a
//! callback draining into a queue) plus the sans-IO [`Client`] core. The JS side
//! drives it: `connect` → poll `is_open` → `load_content` (JS fetched the bundle)
//! → `login` → `set_anchor` → `pump` on an interval; `render_region` reads the
//! world snapshot. No async runtime is pulled — the one async step (content fetch)
//! lives in JS.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, WebSocket};

use resonantdust_codec::card_model::Micro;
use resonantdust_codec::packed::{
    pack_definition, pack_macro_zone_full, surface_of, tile_full, tile_slot, unpack_macro_zone,
    unpack_zone_definition, world_tile, zone_local, INVENTORY_LAYER, ZONE_SIZE,
};
use resonantdust_protocol::protocol::{ClientMsg, GateMsg};

use resonantdust_client_core::client::{Client, Command, Event};
use resonantdust_client_core::content;
use resonantdust_client_core::zones::AnchorRadii;

/// Margin (in TILES) the viewport anchor's `active` disk extends past the visible
/// region. Just enough to pull in the macro_zones bordering the viewport so they
/// stream as a pan reaches them — NOT a deep prefetch ring (a 2-zone ring is the
/// perimeter, which more than doubles the requested zones). The buffered clock
/// masks the rest.
const ANCHOR_MARGIN_TILES: i32 = 2;

/// One reducer's running tally for the debug HUD's "calls" tab.
#[derive(Default)]
struct CmdStat {
    /// Outgoing `Call` frames sent (a fresh send each time — so an outbox retry
    /// counts as another request).
    requests: u32,
    ok: u32,
    err: u32,
    promise: u32,
    /// Serialized JSON byte size of the request/reply frames. An ESTIMATE of
    /// wire bytes: it ignores WebSocket framing + any transport compression.
    tx_bytes: u64,
    rx_bytes: u64,
}

/// Per-reducer call accounting for the debug HUD. Every outgoing `Call` is
/// tallied by reducer name in [`note_send`](CallStats::note_send); the matching
/// `CallOk`/`CallErr`/`CallPromise` reply (which carries only the cid) is
/// attributed back via the `cid → reducer` map in
/// [`note_reply`](CallStats::note_reply). A promised call is resolved later by a
/// terminal reply on the SAME cid, so its mapping is kept on `CallPromise` and
/// dropped only on the terminal `CallOk`/`CallErr`.
#[derive(Default)]
struct CallStats {
    by_reducer: BTreeMap<String, CmdStat>,
    cid_reducer: HashMap<u32, String>,
}

impl CallStats {
    fn note_send(&mut self, cid: u32, reducer: &str, bytes: usize) {
        let e = self.by_reducer.entry(reducer.to_string()).or_default();
        e.requests += 1;
        e.tx_bytes += bytes as u64;
        self.cid_reducer.insert(cid, reducer.to_string());
    }

    /// Tally an inbound frame IF it's a call reply; rows/time/etc. aren't
    /// per-command and are ignored.
    fn note_reply(&mut self, msg: &GateMsg, bytes: usize) {
        let (cid, terminal) = match msg {
            GateMsg::CallOk { cid, .. } => (*cid, true),
            GateMsg::CallErr { cid, .. } => (*cid, true),
            GateMsg::CallPromise { cid, .. } => (*cid, false),
            _ => return,
        };
        // Terminal reply consumes the mapping; a promise keeps it (its ok/err
        // lands later on the same cid).
        let reducer = if terminal {
            self.cid_reducer.remove(&cid)
        } else {
            self.cid_reducer.get(&cid).cloned()
        };
        let Some(reducer) = reducer else { return };
        let e = self.by_reducer.entry(reducer).or_default();
        e.rx_bytes += bytes as u64;
        match msg {
            GateMsg::CallOk { .. } => e.ok += 1,
            GateMsg::CallErr { .. } => e.err += 1,
            GateMsg::CallPromise { .. } => e.promise += 1,
            _ => {}
        }
    }

    /// The tally as a JSON array (one object per reducer, sorted by name), the
    /// shape the worker forwards to the debug panel.
    fn to_json(&self) -> String {
        let arr: Vec<serde_json::Value> = self
            .by_reducer
            .iter()
            .map(|(name, s)| {
                serde_json::json!({
                    "command": name,
                    "requests": s.requests,
                    "ok": s.ok,
                    "err": s.err,
                    "promise": s.promise,
                    "tx": s.tx_bytes,
                    "rx": s.rx_bytes,
                })
            })
            .collect();
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
    }
}

/// One table's running subscription tally for the debug HUD's "subs" tab.
#[derive(Default)]
struct TableStat {
    /// Currently-open subscriptions on this table (`Sub` − `Unsub`). Zone/card
    /// subs are one-per-zone, so this climbs with the anchor's coverage.
    active: i32,
    /// Every `Sub` ever issued for this table — additive, never decremented.
    /// Divided by `active` it reads the churn (re-subscribe) the anchor drives.
    total: u32,
    /// Serialized JSON byte size of `Sub`/`Unsub` frames sent (tx) and the
    /// `Row`/`Applied` frames received (rx). ESTIMATES — they ignore WebSocket
    /// framing + any transport compression.
    tx_bytes: u64,
    rx_bytes: u64,
}

/// Per-table subscription accounting for the debug HUD. Outgoing `Sub`/`Unsub`
/// frames adjust the table's `active` count + tx in [`note_send`](SubStats::note_send);
/// inbound `Row` frames (which carry the table) and `Applied` acks (cid→table
/// via the sid map) add rx in [`note_inbound`](SubStats::note_inbound). `Unsub`
/// and `Applied` carry only a sid, so a `sid → table` map attributes them back.
#[derive(Default)]
struct SubStats {
    by_table: BTreeMap<String, TableStat>,
    sid_table: HashMap<u32, String>,
}

impl SubStats {
    fn note_send(&mut self, frame: &ClientMsg, bytes: usize) {
        match frame {
            ClientMsg::Sub { sid, table, .. } => {
                let e = self.by_table.entry(table.clone()).or_default();
                e.active += 1;
                e.total += 1;
                e.tx_bytes += bytes as u64;
                self.sid_table.insert(*sid, table.clone());
            }
            ClientMsg::Unsub { sid } => {
                // Attribute the teardown's tx to the table the sid subscribed.
                if let Some(table) = self.sid_table.remove(sid) {
                    let e = self.by_table.entry(table).or_default();
                    e.active -= 1;
                    e.tx_bytes += bytes as u64;
                }
            }
            ClientMsg::Call { .. } => {}
        }
    }

    /// Tally an inbound frame IF it streams subscription data (`Row`) or acks a
    /// subscription (`Applied`); call replies + others are ignored here.
    fn note_inbound(&mut self, msg: &GateMsg, bytes: usize) {
        match msg {
            GateMsg::Row { table, .. } => {
                self.by_table.entry(table.clone()).or_default().rx_bytes += bytes as u64;
            }
            GateMsg::Applied { sid } => {
                if let Some(table) = self.sid_table.get(sid) {
                    self.by_table.entry(table.clone()).or_default().rx_bytes += bytes as u64;
                }
            }
            _ => {}
        }
    }

    /// The tally as a JSON array (one object per table, sorted by name), the
    /// shape the worker forwards to the debug panel.
    fn to_json(&self) -> String {
        let arr: Vec<serde_json::Value> = self
            .by_table
            .iter()
            .map(|(table, s)| {
                serde_json::json!({
                    "table": table,
                    "subs": s.active.max(0),
                    "total": s.total,
                    "tx": s.tx_bytes,
                    "rx": s.rx_bytes,
                })
            })
            .collect();
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
    }
}

#[wasm_bindgen]
pub struct WasmClient {
    core: Client,
    ws: Option<WebSocket>,
    /// Inbound frames the WS `onmessage` callback parks until the next `pump`.
    incoming: Rc<RefCell<VecDeque<String>>>,
    /// Flipped true by the WS `onopen` callback.
    open: Rc<RefCell<bool>>,
    // Closures must outlive the WS — dropping them detaches the handlers.
    _onmessage: Option<Closure<dyn FnMut(MessageEvent)>>,
    _onopen: Option<Closure<dyn FnMut()>>,
    /// Set when a card/zone row changed since the last `pump`, so the worker
    /// re-emits the active render region.
    changed: bool,
    /// The new content version if the gate broadcast a `content_changed` since
    /// the last drain — the worker reloads the bundle + tells the main thread.
    content_changed: Option<String>,
    /// Per-reducer call tally for the debug HUD's "calls" tab.
    stats: CallStats,
    /// Per-table subscription tally for the debug HUD's "subs" tab.
    subs: SubStats,
}

#[wasm_bindgen]
impl WasmClient {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmClient {
        WasmClient {
            core: Client::new(),
            ws: None,
            incoming: Rc::new(RefCell::new(VecDeque::new())),
            open: Rc::new(RefCell::new(false)),
            _onmessage: None,
            _onopen: None,
            changed: false,
            content_changed: None,
            stats: CallStats::default(),
            subs: SubStats::default(),
        }
    }

    /// Open the gate WebSocket and wire the inbound queue + open flag. Returns once
    /// the socket is *created*; poll [`is_open`](Self::is_open) for the handshake.
    pub fn connect(&mut self, ws_url: &str) -> Result<(), JsValue> {
        let ws = WebSocket::new(ws_url)?;

        let incoming = self.incoming.clone();
        let onmessage = Closure::<dyn FnMut(MessageEvent)>::new(move |e: MessageEvent| {
            if let Some(txt) = e.data().as_string() {
                incoming.borrow_mut().push_back(txt);
            }
        });
        ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        let open = self.open.clone();
        let onopen = Closure::<dyn FnMut()>::new(move || {
            *open.borrow_mut() = true;
        });
        ws.set_onopen(Some(onopen.as_ref().unchecked_ref()));

        self.ws = Some(ws);
        self._onmessage = Some(onmessage);
        self._onopen = Some(onopen);
        Ok(())
    }

    /// True once the WebSocket handshake completed.
    pub fn is_open(&self) -> bool {
        *self.open.borrow()
    }

    /// Load the DSL content the JS side fetched from the gate's `/content`.
    /// `rd_json` is the `[[name, src], …]` array (the payload's `rd` field).
    pub fn load_content(&mut self, rd_json: &str) -> Result<(), JsValue> {
        let rd: Vec<(String, String)> = serde_json::from_str(rd_json)
            .map_err(|e| JsValue::from_str(&format!("rd parse: {e}")))?;
        let bundle = content::parse_bundle(&rd)
            .map_err(|e| JsValue::from_str(&format!("content load: {e}")))?;
        self.core.set_bundle(bundle);
        Ok(())
    }

    /// Trust-on-first-use login. `player_id` lands later via `pump`.
    pub fn login(&mut self, name: &str) {
        let frames = self.core.dispatch(Command::Login { name: name.to_string() });
        self.send(&frames);
    }

    /// Aim a viewport anchor at hex `(q, r)` on `(surface, owner)`, subscribing the
    /// surrounding zones. `radius_tiles` is the VISIBLE half-extent in TILES (the
    /// `AnchorRadii` tiers are tile distances, not zone counts). The `active` disk
    /// is the visible region plus [`ANCHOR_MARGIN_TILES`], so the macro_zones
    /// bordering the viewport are requested (Card + Zone subs) and stream as a pan
    /// reaches them. No separate prefetch ring — `hot`/`warm`/`cold` are 0.
    /// `owner` is `0` for the world, or the soul card_id for an inventory surface.
    /// Each `(surface, owner)` is a distinct named anchor so multiple viewports
    /// don't clobber each other. Re-call on pan.
    pub fn set_anchor(&mut self, surface: u8, owner: u32, q: i32, r: i32, radius_tiles: i32) {
        let active = radius_tiles.max(1) + ANCHOR_MARGIN_TILES;
        let frames = self.core.dispatch(Command::SetAnchor {
            name: format!("viewport:{surface}:{owner}"),
            surface,
            owner,
            q,
            r,
            radii: AnchorRadii { active, hot: 0, warm: 0, cold: 0 },
        });
        self.send(&frames);
    }

    /// Drop a card loose at a GLOBAL world cell `(q, r)` on `(surface, owner)` —
    /// the view's drag-drop path. One-way: on success the new position arrives via
    /// the render stream; on rejection the data is unchanged, so the card simply
    /// tweens back to its origin (no ack, no prediction).
    pub fn place_loose(&mut self, card_id: u32, surface: u8, owner: u32, q: i32, r: i32) {
        if let Ok(frames) = self.core.place_loose(card_id, surface, owner, q, r) {
            self.send(&frames);
        }
    }

    /// Drop a card onto `parent_id`'s stack in `direction` (drop-on-a-card).
    pub fn place_stack(&mut self, card_id: u32, parent_id: u32, direction: u8) {
        if let Ok(frames) = self.core.place_stack(card_id, parent_id, direction) {
            self.send(&frames);
        }
    }

    /// Create a new card via `create_card` — the chat `/give` path. `owner` owns
    /// the new card; `card_key` is the content def id (e.g. `"corpus"`); the new
    /// card lands in `zone_owner`'s `surface` zone (the gate resolves the def +
    /// stock; the shard places it).
    ///
    /// Placement: when `world_q/world_r` are `0,0` AND `zone_owner == owner`, send
    /// `macro_zone = 0` so the SHARD auto-places into the default bucket
    /// (`first_free_cell` — collision-free; this is `/give 1025 corpus` → first
    /// empty inventory slot). Otherwise resolve the global cell to an explicit
    /// `macro_zone` + local cell (world zones are owned by `0`, inventory zones by
    /// the container card) and place there (exact snap, no collision avoidance).
    pub fn give(
        &mut self,
        owner: u32,
        card_key: &str,
        zone_owner: u32,
        surface: u8,
        world_q: i32,
        world_r: i32,
    ) {
        let (macro_zone, q, r) = if zone_owner == owner && world_q == 0 && world_r == 0 {
            (0u64, 0u8, 0u8)
        } else {
            // World surfaces are one shared grid owned by `0`; inventory/pocket
            // zones are owned by the container card.
            let zo = if surface == INVENTORY_LAYER { zone_owner } else { 0 };
            let (zq, lq) = zone_local(world_q);
            let (zr, lr) = zone_local(world_r);
            (pack_macro_zone_full(zo, surface, zq, zr), lq, lr)
        };
        let frames = self.core.dispatch(Command::CreateCard {
            owner,
            card_key: card_key.to_string(),
            surface,
            macro_zone,
            q,
            r,
        });
        self.send(&frames);
    }

    /// Upload an edited master texture channel to the gate (art editor "save
    /// master"). `data_b64` is the standard-base64 PNG; the gate writes it to the
    /// texture R2 bucket at `textures/master/<aspect>/<faction>/<variant>.<channel>.png`.
    /// Fire-and-forget — gated server-side on the content-author capability;
    /// success/failure is logged gate-side.
    pub fn upload_master(
        &mut self,
        aspect: &str,
        faction: &str,
        variant: &str,
        channel: &str,
        data_b64: &str,
    ) {
        let frames = self.core.dispatch(Command::Call {
            reducer: "upload_master".to_string(),
            args: serde_json::json!({
                "aspect": aspect,
                "faction": faction,
                "variant": variant,
                "channel": channel,
                "data": data_b64,
            }),
        });
        self.send(&frames);
    }

    /// Author a NEW version of an existing `.rd` source (art editor "save DSL").
    /// `lineage` is the source name the gate tracks; `text` is the full file. The
    /// authority validates + hot-swaps + persists to R2, then broadcasts
    /// `content_changed`. Fire-and-forget — gated on the content-author capability.
    pub fn modify_content(&mut self, lineage: &str, text: &str) {
        let frames = self.core.dispatch(Command::Call {
            reducer: "modify_content".to_string(),
            args: serde_json::json!({ "lineage": lineage, "text": text }),
        });
        self.send(&frames);
    }

    /// Author a brand-new `.rd` source named `name` with `text` (a card that
    /// shipped no source for this facet). Same authority validate + hot-swap +
    /// persist + `content_changed` as `modify_content`.
    pub fn add_content(&mut self, name: &str, text: &str) {
        let frames = self.core.dispatch(Command::Call {
            reducer: "add_content".to_string(),
            args: serde_json::json!({ "name": name, "text": text }),
        });
        self.send(&frames);
    }

    /// Replace locale `domain`'s JSON (art editor "save locale"). The authority
    /// validates + hot-swaps + persists + broadcasts `content_changed`. Fire-and-
    /// forget — gated on the content-author capability.
    pub fn modify_locale(&mut self, domain: &str, json: &str) {
        let frames = self.core.dispatch(Command::Call {
            reducer: "modify_locale".to_string(),
            args: serde_json::json!({ "domain": domain, "json": json }),
        });
        self.send(&frames);
    }

    /// Replace visuals source `name` (`visuals/…`) with `text` (art editor "save
    /// visuals"). The authority validates + hot-swaps + persists + broadcasts
    /// `content_changed`. Fire-and-forget — gated on the content-author capability.
    pub fn modify_visuals(&mut self, name: &str, text: &str) {
        let frames = self.core.dispatch(Command::Call {
            reducer: "modify_visuals".to_string(),
            args: serde_json::json!({ "name": name, "text": text }),
        });
        self.send(&frames);
    }

    /// Subscribe to the world chat feed. Idempotent — call once login resolved (so
    /// our sender id/name are known); inbound messages then accumulate for
    /// [`take_chat`](Self::take_chat). Safe to re-call.
    pub fn subscribe_chat(&mut self) {
        let frames = self.core.subscribe_chat();
        self.send(&frames);
    }

    /// Send a chat message to the world feed. Sender id/name come from the session;
    /// the shard trims/validates `body`. Fire-and-forget — it echoes back through
    /// our own subscription like any other message.
    pub fn send_chat(&mut self, body: &str) {
        let frames = self.core.send_chat(body.to_string());
        self.send(&frames);
    }

    /// Drain chat messages folded since the last call, as a JSON array of
    /// `{ sentAt: string, senderPlayerId: number, senderName: string, body: string }`
    /// (sorted by `sentAt`; `sentAt` is a string because the packed u64 exceeds
    /// JS's safe-integer range). Empty `[]` when nothing arrived. The worker calls
    /// this each pump and posts non-empty batches to the chat UI.
    pub fn take_chat(&mut self) -> String {
        let mut msgs = self.core.drain_chat();
        msgs.sort_by_key(|m| m.sent_at);
        let arr: Vec<serde_json::Value> = msgs
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "sentAt": m.sent_at.to_string(),
                    "senderPlayerId": m.sender_player_id,
                    "senderName": m.sender_name,
                    "body": m.body,
                })
            })
            .collect();
        serde_json::to_string(&arr).unwrap_or_else(|_| "[]".to_string())
    }

    /// The assigned player id, or `-1` before login resolves.
    pub fn player_id(&self) -> i32 {
        self.core.player_id().map(|x| x as i32).unwrap_or(-1)
    }

    /// Our player_soul card_id (the on-surface-0 card the player owns directly,
    /// whose inventory IS the player's), or `-1` until the discovery walk folds
    /// it in (`player_id → player_soul`, a pump or two after login). The view
    /// opens this card's inventory as the player's own.
    pub fn player_soul_id(&self) -> i32 {
        self.core.player_souls().next().map(|x| x as i32).unwrap_or(-1)
    }

    /// One drive step: tick the clock, fold inbound frames, flush outbound.
    /// Returns true if any card/zone row changed (the worker re-emits then).
    pub fn pump(&mut self) -> bool {
        let now = js_sys::Date::now();
        // Tick BEFORE folding frames so the clock's local time (`perf_ms`) is set
        // first. A server-time sample folded in `apply` records `note_sample(
        // server_ms, self.perf_ms)`, and `server_now_ms` later extrapolates with
        // the same `perf_ms`. If we folded first, the FIRST sample would anchor at
        // `perf_ms = 0.0` (the init value, tick not yet run) — then every estimate
        // adds a full wall-clock of "elapsed" local time (~1.78e12 ms), doubling
        // the clock to ~2× epoch. Every future-stamped Call (ensure_region /
        // request_zone) then gets `time_drift:client_ahead`-rejected by the gate,
        // so neighbour regions never materialize. Tick-first → samples anchor
        // against real local time.
        self.core.tick(now);
        // Drain into a Vec first so the queue borrow is released before `apply`.
        let frames: Vec<String> = self.incoming.borrow_mut().drain(..).collect();
        for raw in frames {
            if let Ok(msg) = serde_json::from_str::<GateMsg>(&raw) {
                // Tally call replies (by cid → reducer) + subscription data (by
                // table) before `apply` consumes the frame; `raw.len()` is the
                // rx-bytes estimate.
                self.stats.note_reply(&msg, raw.len());
                self.subs.note_inbound(&msg, raw.len());
                for ev in self.core.apply(msg) {
                    match ev {
                        Event::CardUpserted { .. }
                        | Event::CardRemoved { .. }
                        | Event::ZoneUpserted { .. } => self.changed = true,
                        Event::ContentChanged { version } => self.content_changed = Some(version),
                        _ => {}
                    }
                }
            }
        }
        // Re-tick: run the promotion kick + recompute the clock against any
        // samples just folded (same `now`, so `perf_ms` is stable).
        self.core.tick(now);
        let out = self.core.drain_outbound();
        self.send(&out);
        // A zone that loaded ahead of the buffered clock promotes with no row
        // event; fold that render kick in so the worker re-emits the view (else a
        // stationary viewport stays blank until the next pan).
        let kicked = self.core.take_render_kick();
        std::mem::take(&mut self.changed) || kicked
    }

    /// The new content version if the gate hot-swapped its corpus since the last
    /// call, else `undefined`. Drains the flag — the worker calls this each pump
    /// and, on `Some`, re-fetches `/content`, reloads the matching bundle, and
    /// tells the main thread to refresh its render-side `Content`/`Locales`.
    pub fn take_content_changed(&mut self) -> Option<String> {
        self.content_changed.take()
    }

    /// The clock-discipline + RTT diagnostics as a JSON object (the view's
    /// `SyncStats` shape, camelCase) for the debug HUD's "sync" tab. The view
    /// adds the `Date.now()`-relative fields itself. Cheap — call each pump.
    pub fn clock_stats(&self) -> String {
        serde_json::to_string(&self.core.clock_stats()).unwrap_or_else(|_| "{}".to_string())
    }

    /// Per-reducer gateway-call tally as a JSON array (the debug HUD's "calls"
    /// tab): `[{ command, requests, ok, err, promise, tx, rx }]`, sorted by
    /// command name. `tx`/`rx` are serialized-frame byte ESTIMATES. Cheap —
    /// drained each pump.
    pub fn call_stats(&self) -> String {
        self.stats.to_json()
    }

    /// Per-table subscription tally as a JSON array (the debug HUD's "subs"
    /// tab): `[{ table, subs, tx, rx }]`, sorted by table name. `subs` is the
    /// currently-open count; `tx`/`rx` are serialized-frame byte ESTIMATES (tx =
    /// `Sub`/`Unsub` frames, rx = the `Row`/`Applied` frames they stream back).
    /// Cheap — drained each pump.
    pub fn sub_stats(&self) -> String {
        self.subs.to_json()
    }

    /// The renderables in a region of `surface` centred on hex `(center_q,
    /// center_r)`, as a JSON `Renderable[]` (the view's render-feed shape):
    /// the zone tile grid first (under everything), then cards. Loose cards sit
    /// at their cell; stacked members resolve to their root's cell (the DSL fans
    /// the stack). All within the region's half-extents on the named surface.
    pub fn render_region(
        &self,
        surface: u8,
        owner: u32,
        center_q: i32,
        center_r: i32,
        half_cols: i32,
        half_rows: i32,
    ) -> String {
        let now = self.core.clock_ms();
        let world = self.core.world();
        let mut out: Vec<serde_json::Value> = Vec::new();

        // ── Tiles ──────────────────────────────────────────────────────────
        // Each subscribed zone carries an 8×8 packed tile grid (16 u64 words).
        // Decode every non-empty cell (`def_id == 0` ⇒ no tile) to its packed
        // definition (the zone's tile card_type + the cell's def_id) + stock,
        // and emit at its absolute hex. Same surface/owner filter as cards so an
        // inventory viewport shows only its own surface's tiles. The view turns
        // each into a `HexTileVisual` + DSL `tilePrims` scatter.
        for zone in world.zones.current_all(now) {
            let zone_owner = ((zone.macro_zone >> 32) & 0xffff_ffff) as u32;
            if surface_of(zone.macro_zone) != surface || zone_owner != owner {
                continue;
            }
            let (cq, cr) = unpack_macro_zone(zone.macro_zone);
            let card_type = unpack_zone_definition(zone.packed_definition);
            let words = zone.tile_words();
            // Iterate the LOGICAL 7×7 cells; read each from its (8-wide) storage
            // slot and fold to a global tile via the centred mapping.
            for idx in 0..(ZONE_SIZE * ZONE_SIZE) {
                let lc = (idx % ZONE_SIZE) as u8;
                let lr = (idx / ZONE_SIZE) as u8;
                let (def_id, stock0, stock1) = tile_full(&words, tile_slot(lc, lr));
                if def_id == 0 {
                    continue;
                }
                let q = world_tile(cq, lc);
                let r = world_tile(cr, lr);
                if (q - center_q).abs() > half_cols + 1 || (r - center_r).abs() > half_rows + 1 {
                    continue;
                }
                out.push(serde_json::json!({
                    "layer": "tile",
                    "q": q,
                    "r": r,
                    "packed": pack_definition(card_type, def_id),
                    "stock0": stock0,
                    "stock1": stock1,
                }));
            }
        }

        for row in world.cards.current_all(now) {
            // `macro_zone` = [owner:u32 | surface:u8 | qr]; match both bands so an
            // inventory viewport (owner = soul card_id) shows only that inventory.
            let row_owner = ((row.macro_zone >> 32) & 0xffff_ffff) as u32;
            if row.is_dead() || surface_of(row.macro_zone) != surface || row_owner != owner {
                continue;
            }
            // Resolve the card's CELL: a loose card sits at its own cell; a stacked
            // card sits at its ROOT's cell (a stacked card's `micro_location` IS the
            // root card_id; it shares the root's zone). The view fans it from there
            // via the DSL `card_data.stack` (decoded from `flags`) — we just dumb-
            // draw every card at a resolved cell.
            let (lq, lr) = match row.micro() {
                Micro::Loose { local_q, local_r, .. } => (local_q, local_r),
                Micro::Stacked { root, .. } => match world.cards.current(root, now).map(|r| r.micro()) {
                    Some(Micro::Loose { local_q, local_r, .. }) => (local_q, local_r),
                    // Root unknown (not subscribed) or itself stacked (no nested
                    // stacks yet) — skip; it'll resolve once the root streams in.
                    _ => continue,
                },
            };
            let (cq, cr) = unpack_macro_zone(row.macro_zone);
            let q = world_tile(cq, lq);
            let r = world_tile(cr, lr);
            if (q - center_q).abs() > half_cols + 1 || (r - center_r).abs() > half_rows + 1 {
                continue;
            }
            out.push(serde_json::json!({
                "layer": "card",
                "cardId": row.card_id,
                "q": q,
                "r": r,
                // Snapped — no within-cell offset; the DSL applies any stack fan.
                "offsetX": 0,
                "offsetY": 0,
                "packed": row.packed_definition,
                "stock": row.stock,
                "flags": row.flags,
            }));
        }
        serde_json::to_string(&out).unwrap_or_else(|_| "[]".to_string())
    }
}

impl WasmClient {
    /// Serialize + send each frame on the WebSocket (no-op before `connect`).
    /// Tallies every outgoing frame: `Call`s into [`CallStats`] (request count +
    /// tx estimate + cid→reducer), `Sub`/`Unsub` into [`SubStats`] (active count
    /// + tx estimate + sid→table).
    fn send(&mut self, frames: &[ClientMsg]) {
        if let Some(ws) = &self.ws {
            for m in frames {
                if let Ok(s) = serde_json::to_string(m) {
                    match m {
                        ClientMsg::Call { cid, reducer, .. } => {
                            self.stats.note_send(*cid, reducer, s.len());
                        }
                        ClientMsg::Sub { .. } | ClientMsg::Unsub { .. } => {
                            self.subs.note_send(m, s.len());
                        }
                    }
                    let _ = ws.send_with_str(&s);
                }
            }
        }
    }
}

impl Default for WasmClient {
    fn default() -> Self {
        Self::new()
    }
}
