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
use std::collections::VecDeque;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{MessageEvent, WebSocket};

use resonantdust_codec::card_model::Micro;
use resonantdust_codec::packed::{
    pack_definition, surface_of, tile_full, tile_slot, unpack_macro_zone, unpack_zone_definition,
    world_tile, ZONE_SIZE,
};
use resonantdust_protocol::protocol::{ClientMsg, GateMsg};

use resonantdust_client_core::client::{Client, Command, Event};
use resonantdust_client_core::content;
use resonantdust_client_core::zones::AnchorRadii;

/// Lookahead ring (in tiles) the viewport anchor subscribes beyond the visible
/// region, so a zone's tiles stream in before a pan brings it on screen. Two
/// zones of margin — enough to cover the subscribe→materialize→stream round-trip
/// at normal pan speeds.
const PREFETCH_TILES: i32 = 2 * ZONE_SIZE;

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
    /// `AnchorRadii` tiers are tile distances, not zone counts) — the `active`
    /// tier covers exactly what's on screen so its zones stream. The `cold` tier
    /// extends one [`PREFETCH_TILES`] ring further so the next zones' tiles
    /// materialize BEFORE they scroll into view — without it a pan reveals a
    /// blank row/col while the just-entered zone is still being requested.
    /// `owner` is `0` for the world, or the soul card_id for an inventory
    /// surface. Each `(surface, owner)` is a distinct named anchor so multiple
    /// viewports don't clobber each other. Re-call on pan.
    pub fn set_anchor(&mut self, surface: u8, owner: u32, q: i32, r: i32, radius_tiles: i32) {
        let active = radius_tiles.max(1);
        let frames = self.core.dispatch(Command::SetAnchor {
            name: format!("viewport:{surface}:{owner}"),
            surface,
            owner,
            q,
            r,
            // active = visible (Card + Zone subs); cold = prefetch ring (Zone sub
            // ⇒ tiles materialize ahead). hot/warm unused (0 ⇒ skipped).
            radii: AnchorRadii { active, hot: 0, warm: 0, cold: active + PREFETCH_TILES },
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
        std::mem::take(&mut self.changed)
    }

    /// The new content version if the gate hot-swapped its corpus since the last
    /// call, else `undefined`. Drains the flag — the worker calls this each pump
    /// and, on `Some`, re-fetches `/content`, reloads the matching bundle, and
    /// tells the main thread to refresh its render-side `Content`/`Locales`.
    pub fn take_content_changed(&mut self) -> Option<String> {
        self.content_changed.take()
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
    fn send(&self, frames: &[ClientMsg]) {
        if let Some(ws) = &self.ws {
            for m in frames {
                if let Ok(s) = serde_json::to_string(m) {
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
