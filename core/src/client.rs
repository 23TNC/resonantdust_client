//! The sans-IO client core.
//!
//! Pure logic — **no socket, no async, no clock of its own**. It maps intents to
//! outbound frames and inbound frames to world updates + semantic events:
//!
//! - [`Client::dispatch`] takes a [`Command`] and returns the [`ClientMsg`]s to
//!   put on the wire (allocating subscription / call ids). It never blocks and
//!   never waits for a result — fire as many as you like; results arrive later as
//!   events. (This is the property that lets NPCs / players pipeline freely.)
//! - [`Client::apply`] takes one inbound [`GateMsg`], folds it into the
//!   [`World`], and returns the [`Event`]s it produced.
//!
//! Both NPC (native) and player (wasm) front-ends drive the *same* core: issue
//! `Command`s, observe `Event`s + [`Client::world`]. Only the run-loop that
//! shuttles frames to/from the socket is per-target. The core is exhaustively
//! testable by feeding frames — no transport needed.

use std::collections::{BTreeMap, HashMap, HashSet};

use resonantdust_dsl::bridge::Card;
use resonantdust_codec::card_model::{stack_branch, stack_index};
use resonantdust_dsl::loader::Bundle;
use resonantdust_protocol::protocol::{ClientMsg, GateMsg, RowOp};
use resonantdust_dsl::recipe::{build_frame, iterators};
use resonantdust_state::recipe_state::CardStore;
use resonantdust_state::stack::{self, plan_place, StackStore};
use resonantdust_dsl::vm::match_recipe;

use crate::actions::{QueuedAction, DEFAULT_DELAY_MS, MAX_TIME_DRIFT_RETRIES, TIME_DRIFT_RETRY_PAD_MS};
use crate::clock::ClockSync;
use crate::rows::CardRow;
use crate::world::World;
use crate::zones::{DataType, ZoneIntent, ZoneManager};

/// An intent issued to the client. Expands to one or more outbound frames via
/// [`Client::dispatch`].
#[derive(Debug, Clone)]
pub enum Command {
    /// Trust-on-first-use login by name. Also subscribes `players` (filtered to
    /// this name) so the core learns our `player_id` and emits [`Event::PlayerId`].
    Login { name: String },
    /// Mint a card via the generic `create_card` primitive. The entity tree is
    /// built by **chaining** this — `player → player_soul → world soul → cards` —
    /// each step using the previous card's discovered id as `owner`:
    /// - `owner`    — a `player_id` (for a player_soul) or a `card_id`.
    /// - `card_key` — content name; the gate resolves it to a packed def. A
    ///                player_soul (`card_key = "player_soul"`) is identified by
    ///                its definition (`>= 0xFFF0`) — no flag needed.
    /// - `surface`  — band: `0` player_soul / `64` world / `1` inventory.
    CreateCard {
        owner: u32,
        card_key: String,
        surface: u8,
        /// Placement override: `macro_zone != 0` spawns at that zone, loose cell
        /// `(q, r)`; `0` = default (inventory bucket / surface's (0,0) cell).
        macro_zone: u64,
        q: u8,
        r: u8,
    },
    /// Subscribe to a table, optionally SQL-filtered.
    Subscribe { table: String, filter: Option<String> },
    /// Place (or move) a named zone anchor in hex `(q, r)` on `(surface, owner)`
    /// with its tier `radii`. Drives the active/hot/warm/cold tiers; the
    /// resulting subscription / region / spawn frames come back via
    /// [`zone_frames`]. Soul anchors are normally placed by `discover` (reading
    /// the soul def's `anchor_*` aspects); this command is the explicit entry.
    #[allow(dead_code)]
    SetAnchor { name: String, surface: u8, owner: u32, q: i32, r: i32, radii: crate::zones::AnchorRadii },
    /// Move `soul` ONE tile to the adjacent world cell `(dest_q, dest_r)`. The
    /// client computes the arrival time from the current + destination tile
    /// `cost`s and the soul's `speed` (`travel = (cost_cur+cost_dst)·1000 /
    /// (2·speed)` ms) and future-stamps the move via `move_soul`: the soul stays
    /// at its current cell and "arrives" when the clock reaches `arrival_ms`.
    MoveStep { soul: u32, dest_q: i32, dest_r: i32 },
    /// Walk `soul` to the world cell `(target_q, target_r)`: the client computes a
    /// hex path and PIPELINES the steps (one future-stamped `move_soul` per tile,
    /// the next requested as each prior step's row arrives). A no-op if already
    /// there or no path. Re-issuing replaces any in-flight movement (path change).
    MoveTo { soul: u32, target_q: i32, target_r: i32 },
    /// Escape hatch: call an arbitrary gate/shard reducer. (Typed commands for
    /// place_card / propose_action land as the action path grows.)
    #[allow(dead_code)]
    Call { reducer: String, args: serde_json::Value },
}

/// An in-flight pipelined movement for one soul. The client requests step N+1 the
/// moment step N's future-stamped row arrives, future-dating each at its
/// cumulative arrival, so the soul walks the `path` on schedule.
#[derive(Debug, Clone)]
struct MovePlan {
    /// Cells to visit in order; `path[0]` is adjacent to the start cell.
    path: Vec<(i32, i32)>,
    /// How many steps have been REQUESTED (`move_soul` sent for `path[0..requested]`).
    requested: usize,
    /// The cell the last-requested step lands on (to match its inbound row).
    last_dest: (i32, i32),
    /// `arrival_ms` (future `valid_at`) of the last-requested step — the departure
    /// time for the next step.
    last_arrival: u64,
}

/// Something the core observed while folding an inbound frame. The front-end's
/// react surface — match on these to drive rendering / NPC decisions.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Server clock advanced (ms since epoch). Sampled from `time` keepalives and
    /// call replies; the core stamps outbound `client_time_ms` from it.
    Clock { ms: u64 },
    /// Our `player_id` was learned from the `players` subscription.
    PlayerId { id: u32 },
    /// A subscription's initial batch has been delivered.
    Applied { sid: u32 },
    /// A card row was inserted or updated in the world.
    CardUpserted { card_id: u32 },
    /// A card version row was deleted (GC reap or removal).
    CardRemoved { card_id: u32 },
    /// A zone row landed (by its `macro_zone` location key).
    ZoneUpserted { macro_zone: u64 },
    /// A reducer call succeeded.
    CallOk { cid: u32 },
    /// A reducer call failed.
    CallErr { cid: u32, error: String },
    /// A protocol-level error, or a row we couldn't decode.
    Error { error: String },
}

/// A recipe the matcher found applicable to a board state, with the `bindings`
/// (card_ids per iterator/offset) that satisfy it — ready to pass to
/// [`Client::propose`]. `root` is the card to propose as the action root: the
/// original candidate root for a root-anchored recipe, or `0` when the matcher
/// folded the root into a stack branch (the promotion passes — see
/// [`Client::match_recipes`]). The action *location* (surface/macro_zone/
/// micro_location) is always the original candidate root's cell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeMatch {
    pub recipe: String,
    pub root: u32,
    pub bindings: Vec<Vec<u32>>,
    /// `true` when matched against **live** data (the soul is present at the
    /// root's zone — execute now); `false` for a **memory** match (read from the
    /// soul's frozen watermark — a goal to verify on arrival, not to propose
    /// blind).
    pub live: bool,
    /// `@input` statement count — `≤ 1` bypasses the action-queue debounce
    /// (lifecycle/single-input recipes fire immediately; there's nothing to
    /// cancel).
    pub input_count: usize,
}

impl RecipeMatch {
    /// Priority for the triggered path: the **most specific** recipe wins — the
    /// one requiring the most `@input` slots. On a 3-corpus stack both
    /// `triple_corpus` (3 inputs) and `corpus_b_top` (2 inputs) match (bindings
    /// carry all 3 branch members either way, so a bound-card count can't tell
    /// them apart); `input_count` does — `triple_corpus` (3) beats `corpus_b_top`
    /// (2). Ties break toward more bound cards.
    pub fn priority(&self) -> (usize, usize) {
        let bound = self.bindings.iter().flatten().filter(|&&id| id != 0).count();
        (self.input_count, bound)
    }
}

/// The client core: world model + id/clock/session state, driven by
/// [`Client::dispatch`] / [`Client::apply`].
pub struct Client {
    world: World,
    next_sid: u32,
    next_cid: u32,
    /// Adaptive server-time estimator (runs behind by `client_delay`).
    clock: ClockSync,
    /// Local monotonic time (ms) the driver supplies via [`tick`](Self::tick) —
    /// extrapolates the clock between samples. `0` in tests (no driver).
    perf_ms: f64,
    /// Cached server-time estimate (ms) = `clock.server_now_ms(perf_ms)`, the
    /// value stamped onto outbound `client_time_ms`. Refreshed on each sample +
    /// `tick`. `0` until the first sample.
    clock_ms: u64,
    /// Our login name, set by [`Command::Login`], used to match our `players` row.
    name: Option<String>,
    /// Our `player_id`, learned from the `players` subscription.
    player_id: Option<u32>,
    /// The DSL content bundle (recipe ids, defs, matching). Loaded from the gate
    /// via [`crate::content::fetch_bundle`]; needed to `propose` recipes.
    bundle: Option<Bundle>,
    /// The zone subsystem — anchors → active/hot/cold tiers → subscription /
    /// region / spawn intents. Mutated by [`Command::SetAnchor`] and by inbound
    /// region/zone rows (`note_region` / `note_zone_arrived`); its intents are
    /// drained into frames by [`Client::zone_frames`].
    zones: ZoneManager,
    /// Live scoped subscription sid per `(macro_zone, data type)` — so a sub can
    /// be `Unsub`'d by its sid when the zone manager closes it.
    zone_subs: HashMap<(u64, DataType), u32>,
    /// Live region subscription sid per `macro_region`.
    region_subs: HashMap<u64, u32>,
    /// Outbound frames queued while folding inbound rows (the soul-discovery
    /// roster subscriptions) — drained alongside the zone intents by
    /// [`Client::drain_outbound`]. `dispatch` returns its frames directly; this
    /// is the channel for frames the core *originates* in response to inbound
    /// state (it has no socket of its own).
    pending_out: Vec<ClientMsg>,
    /// Our player_soul card_ids we've subscribed a soul-roster for (`cards WHERE
    /// owner_id = <player_soul>`). Walk: player_id → player_soul → souls.
    player_souls: HashSet<u32>,
    /// Souls (world actors) we control, each mapped to its last-anchored world
    /// hex `(q, r)` — so we only re-anchor on a real move, and `ensure_inventory`
    /// / the anchor fire once on first sight. Keyed by soul card_id.
    souls: HashMap<u32, (i32, i32)>,
    /// Triggered action queue, keyed by chain root: a matched recipe waits here
    /// for its debounce window before being proposed. See [`crate::actions`].
    actions: HashMap<u32, QueuedAction>,
    /// In-flight reducer calls → `(send_perf_ms, send_client_time_ms)` captured
    /// when the `cid` was minted. A `time_drift` reject names the rejected stamp
    /// (`send_client_time`) and lets [`note_drift`](Self::note_drift) re-seat the
    /// clock at the call's true send instant via the RTT midpoint — not the
    /// current clock. Cleared on the call's reply.
    inflight_calls: HashMap<u32, (f64, u64)>,
    /// Chain roots whose board state changed since the last match pass — the
    /// dirty set. Re-identification is **budgeted** ([`MATCH_BUDGET_MS`]): card
    /// rows mark their root dirty here and [`tick`](Self::tick) drains the set at
    /// most ~1/s, collapsing a burst of rows into one `evaluate_root` per root.
    dirty_roots: HashSet<u32>,
    /// Local monotonic time (ms) the dirty set was last flushed — the budget gate.
    last_match_perf: f64,
    /// Magnetic cards we own → their resolution deadline (server ms): first-seen
    /// + the def's `magnetic.duration`. Drives [`Client::magnetic_pass`].
    magnet_deadlines: HashMap<u32, u64>,
    /// Magnets whose overdue `magnetic.failure` recipe we've already proposed —
    /// so the deadline fires the fallback exactly once.
    magnet_failed: HashSet<u32>,
    /// Local monotonic time (ms) the magnetic pass last ran — its budget gate.
    last_magnet_perf: f64,
    /// Cards we've already ensured an inventory zone for (any container we
    /// transitively own that carries `aspect.inventory`) — so the recursive
    /// ensure fires once per container. See [`ensure_owned_inventory`].
    inventoried: HashSet<u32>,
    /// Precomputed per-recipe matcher metadata, built once in [`set_bundle`] — the
    /// recipe index. Avoids re-walking every recipe's AST on each match pass.
    recipe_meta: Vec<RecipeMeta>,
    /// Union of all aspect names that gate some recipe (its strictly-required
    /// top-level `aspect.X` guards). The only aspects the pre-filter reads off a
    /// card — bounds the per-pass aspect probing to the handful that matter.
    indexed_aspects: HashSet<String>,
    /// Protocol / row-decode errors the core has observed (the `Event::Error`
    /// stream, retained). A driver/harness drains this to fail loudly on silent
    /// corruption rather than only logging it. See [`drain_errors`](Self::drain_errors).
    seen_errors: Vec<String>,
    /// Final outcomes of fired queued actions: `(recipe, Err(reason)?)` — `None`
    /// = accepted, `Some(reason)` = dropped on a non-retry rejection. Lets a
    /// driver/harness tell a real accept from a silent drop (the queue empties on
    /// both). A time-drift *reschedule* is not final, so it's not recorded here.
    action_outcomes: Vec<(String, Option<String>)>,
    /// Cards moved LOCALLY whose position hasn't been synced to the server —
    /// commit-based position ([[project_card_position_sync]]). A move applies to
    /// the local world (position is client-authoritative) and lands here; the set
    /// is flushed via the `move_cards` batch reducer before a recipe proposal (and
    /// later on an observer 1→>1 transition). Cleared on flush.
    dirty_positions: HashSet<u32>,
    /// Monotonic u16 for stamping LOCAL position rows' `valid_at` seq — keeps
    /// same-ms local moves distinct in the bitemporal store. Server rows use the
    /// shard's global sequence; this is purely client-local.
    local_seq: u16,
    /// Future-stamped rows awaiting promotion: `(promote_at_ms, card_id)`. A
    /// future row ARRIVES once (marking its root dirty while still not-current, so
    /// the match finds nothing), but its promotion when the clock reaches
    /// `promote_at_ms` brings no new arrival — so without this the matcher never
    /// re-evaluates a card a recipe created/changed in the future (lifecycles,
    /// self-advancing recipes, progress). [`tick`](Self::tick) drains the due
    /// entries and re-marks their chain roots dirty — the promotion kick. See
    /// [[project_future_row_progress_kick]].
    pending_promotions: Vec<(u64, u32)>,
    /// In-flight movements, keyed by soul card_id. The client PIPELINES tile steps:
    /// it requests step N+1 the instant it receives step N's future-stamped row
    /// ([`advance_movement`](Self::advance_movement)), so the server has step N's
    /// whole traversal to validate the next one. See [`Command::MoveTo`].
    movements: HashMap<u32, MovePlan>,
    /// Live per-zone OBSERVER counts from the gate (`GateMsg::ZoneObservers`):
    /// `macro_zone -> distinct connections watching`. A move in a zone with
    /// `observers > 1` is in shared space and must sync immediately; `≤ 1` stays
    /// client-local. Commit-based position, Phase 2 ([[project_card_position_sync]]).
    zone_observers: HashMap<u64, u32>,
}

/// Precomputed matcher metadata for one recipe — the recipe index entry. Built
/// once per bundle ([`Client::set_bundle`]) so [`Client::match_recipes`] doesn't
/// re-parse iterators / scan tokens for every recipe on every (budgeted) pass.
struct RecipeMeta {
    name: String,
    iters: Vec<resonantdust_dsl::recipe::Iter>,
    /// The recipe reads/writes the `root` slot (matched root-anchored, no promotion).
    uses_root: bool,
    /// Bitmask of top-level branches the recipe references.
    want_branches: u32,
    /// Per-`stack_id` window span = (max top-level offset referenced) + 1, for the
    /// sliding top/bottom stacks (indices 2/3). A recipe reading `slot.2.0`+`slot.2.1`
    /// has `branch_spans[2] == 2`; the matcher then slides that 2-wide window over the
    /// top stack so the pair can match cards at ANY contiguous offset, not just 0,1.
    /// Stacks 0 (root) and 1 (hex/tile) don't slide → their spans are unused.
    branch_spans: [u8; 4],
    /// `@input` statement count — the action-queue debounce discriminator.
    input_count: usize,
    /// Aspect names this recipe **strictly requires** on a top-level slot (a
    /// `*slot.<top>.<off>.aspect.X K (ge|gt) … if`-guarded read): absence from the
    /// candidate stack guarantees no match, so the recipe is skipped. Conservative
    /// — compound/non-threshold guards contribute nothing (never drops a match).
    req_aspects: Vec<String>,
}

impl Default for Client {
    fn default() -> Self {
        Self {
            world: World::default(),
            next_sid: 1,
            next_cid: 1,
            clock: ClockSync::new(),
            perf_ms: 0.0,
            clock_ms: 0,
            name: None,
            player_id: None,
            bundle: None,
            zones: ZoneManager::new(),
            zone_subs: HashMap::new(),
            region_subs: HashMap::new(),
            pending_out: Vec::new(),
            actions: HashMap::new(),
            player_souls: HashSet::new(),
            souls: HashMap::new(),
            inflight_calls: HashMap::new(),
            dirty_roots: HashSet::new(),
            last_match_perf: 0.0,
            magnet_deadlines: HashMap::new(),
            magnet_failed: HashSet::new(),
            last_magnet_perf: 0.0,
            inventoried: HashSet::new(),
            recipe_meta: Vec::new(),
            indexed_aspects: HashSet::new(),
            seen_errors: Vec::new(),
            action_outcomes: Vec::new(),
            dirty_positions: HashSet::new(),
            local_seq: 0,
            pending_promotions: Vec::new(),
            movements: HashMap::new(),
            zone_observers: HashMap::new(),
        }
    }
}

/// Re-identification budget: the dirty-root set is matched at most once per this
/// many local ms (see [`Client::tick`]). Collapses a burst of card rows into one
/// `evaluate_root` per root — the action-queue debounce (seconds) absorbs the
/// sub-second batching, so matches still resolve well within a window.
const MATCH_BUDGET_MS: f64 = 1000.0;

/// Magnetic-pass budget: scan owned magnets + gather/deadline at most once per
/// this many local ms (see [`Client::magnetic_pass`]). 1s granularity is ample
/// against multi-second magnetic windows.
const MAGNET_BUDGET_MS: f64 = 1000.0;

impl Client {
    pub fn new() -> Self {
        Self::default()
    }

    /// The local world model (read-only). Front-ends render / decide from this.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Our `player_id`, once learned.
    pub fn player_id(&self) -> Option<u32> {
        self.player_id
    }

    /// Resolve a content card name to its packed definition via the loaded
    /// bundle (`None` if the bundle isn't loaded or the name is unknown). For
    /// drivers/tests that scan the world by card kind.
    pub fn packed_def(&self, name: &str) -> Option<u16> {
        self.bundle.as_ref().and_then(|b| b.packed_def(name))
    }

    /// The last-sampled server clock (ms).
    pub fn clock_ms(&self) -> u64 {
        self.clock_ms
    }

    /// The zone subsystem (read-only) — anchors, tiers, the active search scope.
    pub fn zones(&self) -> &ZoneManager {
        &self.zones
    }

    /// The triggered action queue's current entries: `(root, recipe, in_flight)`
    /// — `in_flight` is true once proposed and awaiting its reply. For the test
    /// harness to observe debounce / supersede / fire.
    #[allow(dead_code)]
    pub fn queued(&self) -> Vec<(u32, String, bool)> {
        self.actions.iter().map(|(r, a)| (*r, a.recipe.clone(), a.submit_cid.is_some())).collect()
    }

    /// Anchor-aware garbage collection: reap card/zone version rows no soul
    /// remembers — keeping each id's future rows, its current-at-now row, and the
    /// rows pinned by every soul's frozen memory watermark for its zone. The
    /// per-soul memory-LRU (in the zone manager) bounds how much is pinned. Cheap
    /// to call on the budget tick; safe any time.
    #[allow(dead_code)] // driver calls this periodically
    pub fn gc(&mut self) {
        let now = self.clock_ms;
        let zones = &self.zones;
        self.world.cards.gc(now, |zone| zones.zone_card_pins(zone));
        self.world.zones.gc(now, |zone| zones.zone_tile_pins(zone));
    }

    /// The souls (world actors) we control, by card_id. Populated by the
    /// player→player_soul→soul discovery walk; each is anchored at its world
    /// position and has its inventory container subscribed.
    #[allow(dead_code)] // Phase 4 iterates souls to match per-soul scopes
    pub fn souls(&self) -> impl Iterator<Item = u32> + '_ {
        self.souls.keys().copied()
    }

    /// Our discovered player_soul card_ids (the cards owned directly by our
    /// `player_id`). Used to hang a world soul off a player_soul during seeding.
    pub fn player_souls(&self) -> impl Iterator<Item = u32> + '_ {
        self.player_souls.iter().copied()
    }

    /// The gate's last-reported distinct-observer count for `macro_zone` (cards
    /// subscribers across connections). `0` if the gate never reported it / the
    /// zone is private — we only store `> 1` (see [`Self::apply`]). Lets the
    /// harness assert a zone is genuinely shared (two clients anchored in it).
    pub fn observers(&self, macro_zone: u64) -> u32 {
        self.zone_observers.get(&macro_zone).copied().unwrap_or(0)
    }

    /// The world hex `(q, r)` `card` currently sits at (CURRENT row at `now`, so a
    /// future-stamped move row is excluded until it promotes), or `None` if it
    /// isn't on the world surface. Used by the harness to track a moving soul.
    pub fn soul_cell(&self, card: u32) -> Option<(i32, i32)> {
        world_hex(self.world.cards.current(card, self.clock_ms)?)
    }

    /// Drain every outbound frame the core has queued in response to inbound
    /// state — the soul-discovery roster subscriptions plus the zone subsystem's
    /// subscription / region / spawn frames. The driver calls this after each
    /// [`apply`](Self::apply) (and it's safe to call after `dispatch` too); the
    /// core has no socket, so this is how state-driven frames reach the wire.
    pub fn drain_outbound(&mut self) -> Vec<ClientMsg> {
        let mut out = std::mem::take(&mut self.pending_out);
        out.extend(self.zone_frames());
        out
    }

    fn sid(&mut self) -> u32 {
        let s = self.next_sid;
        self.next_sid += 1;
        s
    }

    fn cid(&mut self) -> u32 {
        let c = self.next_cid;
        self.next_cid += 1;
        // Record the send instant (local perf + the clock we're about to stamp)
        // so a `time_drift` reply can re-seat the clock at the real send time.
        self.inflight_calls.insert(c, (self.perf_ms, self.clock_ms));
        c
    }

    /// Translate a [`Command`] into the frames to send. Synchronous and
    /// non-blocking; results (if any) arrive later via [`Client::apply`].
    pub fn dispatch(&mut self, cmd: Command) -> Vec<ClientMsg> {
        match cmd {
            Command::Login { name } => {
                let cid = self.cid();
                let login = ClientMsg::Call {
                    cid,
                    reducer: "claim_or_login".to_string(),
                    args: serde_json::json!({ "client_time_ms": self.clock_ms, "name": name }),
                };
                let watch = ClientMsg::Sub {
                    sid: self.sid(),
                    table: "players".to_string(),
                    filter: Some(format!("name = '{name}'")),
                };
                self.name = Some(name);
                // SUBSCRIBE BEFORE LOGIN. The gate handles messages serially and
                // its `login_relay` blocks up to 2s polling for the player row in
                // its upstream `players` cache — which is only populated once this
                // `players WHERE name=…` sub is active on the gate's connection. If
                // `claim_or_login` were processed first, the poll watches an
                // inactive subscription, times out, and the gate session is never
                // set (the player_id-keyed gate features silently no-op).
                vec![watch, login]
            }
            Command::CreateCard { owner, card_key, surface, macro_zone, q, r } => {
                vec![ClientMsg::Call {
                    cid: self.cid(),
                    reducer: "create_card".to_string(),
                    args: serde_json::json!({
                        "client_time_ms": self.clock_ms,
                        "owner_id": owner,
                        "surface": surface,
                        "card_key": card_key,
                        "macro_zone": macro_zone,
                        "q": q,
                        "r": r,
                    }),
                }]
            }
            Command::Subscribe { table, filter } => {
                vec![ClientMsg::Sub { sid: self.sid(), table, filter }]
            }
            Command::SetAnchor { name, surface, owner, q, r, radii } => {
                // soul 0 — a manually-set anchor carries no per-soul memory.
                self.zones.set_anchor(&name, q, r, surface, owner, radii, 0, self.clock_ms);
                self.zone_frames()
            }
            Command::MoveStep { soul, dest_q, dest_r } => {
                self.build_move_soul(soul, dest_q, dest_r).into_iter().collect()
            }
            Command::MoveTo { soul, target_q, target_r } => {
                self.start_move_to(soul, target_q, target_r)
            }
            Command::Call { reducer, args } => {
                vec![ClientMsg::Call { cid: self.cid(), reducer, args }]
            }
        }
    }

    /// Drain the zone subsystem's pending intents into outbound frames, tracking
    /// the sids so a subscription-class change can tear down the old scoped subs
    /// first. Call this after any [`Command::SetAnchor`] (dispatch does it for
    /// you) and after [`apply`](Self::apply) folds a region/zone row — those
    /// inbound rows feed the region gate, which may queue `request_zone` /
    /// `ensure_region` calls and subscription changes.
    ///
    /// Each `Subscribe` intent is one scoped table subscription: `Card` → the
    /// `cards` table, `Zone` → the `zones` table, both filtered to the
    /// `macro_zone`. `on = false` tears that one down (by its tracked sid). The
    /// headless world models cards + zones; `souls` / `tile_cards` aren't here.
    pub fn zone_frames(&mut self) -> Vec<ClientMsg> {
        let intents = self.zones.take_intents();
        let mut out = Vec::new();
        for intent in intents {
            match intent {
                ZoneIntent::Subscribe { zone, data, on } => {
                    let key = (zone, data);
                    if on {
                        // Idempotent: skip if already subscribed.
                        if self.zone_subs.contains_key(&key) {
                            continue;
                        }
                        let sid = self.sid();
                        self.zone_subs.insert(key, sid);
                        let table = match data {
                            DataType::Card => "cards",
                            DataType::Zone => "zones",
                        };
                        out.push(ClientMsg::Sub {
                            sid,
                            table: table.to_string(),
                            filter: Some(format!("macro_zone = {zone}")),
                        });
                    } else if let Some(sid) = self.zone_subs.remove(&key) {
                        out.push(ClientMsg::Unsub { sid });
                    }
                }
                ZoneIntent::Region { region, subscribed } => {
                    if subscribed {
                        let sid = self.sid();
                        self.region_subs.insert(region, sid);
                        out.push(ClientMsg::Sub {
                            sid,
                            table: "regions".to_string(),
                            filter: Some(format!("macro_region = {region}")),
                        });
                    } else if let Some(sid) = self.region_subs.remove(&region) {
                        out.push(ClientMsg::Unsub { sid });
                    }
                }
                ZoneIntent::RequestZone { zone } => {
                    let cid = self.cid();
                    out.push(ClientMsg::Call {
                        cid,
                        reducer: "request_zone".to_string(),
                        args: serde_json::json!({
                            "client_time_ms": self.clock_ms,
                            "macro_zone": zone,
                        }),
                    });
                }
                ZoneIntent::EnsureRegion { zone } => {
                    let cid = self.cid();
                    out.push(ClientMsg::Call {
                        cid,
                        reducer: "ensure_region".to_string(),
                        args: serde_json::json!({
                            "client_time_ms": self.clock_ms,
                            "macro_zone": zone,
                        }),
                    });
                }
            }
        }
        out
    }

    /// One step of the player → player_soul → soul → cards discovery walk, run
    /// when a `cards` row lands. Recognizes two roles by ownership and reacts:
    ///
    /// - **Our player_soul** (`owner_id == player_id`, `player_owned`): subscribe
    ///   its soul roster (`cards WHERE owner_id = <this card>`) so the souls it
    ///   owns stream in.
    /// - **A soul we own** (`owner_id` is a tracked player_soul): anchor it at its
    ///   world hex (the recipe search scope) and, on first sight, subscribe its
    ///   inventory container (`ensure_inventory`) so its cards stream in.
    ///
    /// Idempotent: roster subs fire once per player_soul; `ensure_inventory` once
    /// per soul; the anchor re-fires only when the soul actually moves.
    fn discover(&mut self, card_id: u32) {
        let now = self.clock_ms;
        // Snapshot what we need so the world borrow ends before the &mut work.
        let Some((owner_id, packed, hex)) = self
            .world
            .cards
            .current(card_id, now)
            .map(|c| (c.owner_id, c.packed_definition, world_hex(c)))
        else {
            return;
        };
        let me = self.player_id;
        // player_soul identity is by DEFINITION now (reserved 0xFFF0..=0xFFFF),
        // not the old `player_owned` flag.
        let is_player_soul = resonantdust_codec::packed::is_player_soul(packed);

        // Role 1: our player_soul → subscribe the souls it owns.
        if me == Some(owner_id) && is_player_soul {
            if self.player_souls.insert(card_id) {
                let sid = self.sid();
                self.pending_out.push(ClientMsg::Sub {
                    sid,
                    table: "cards".to_string(),
                    filter: Some(format!("owner_id = {card_id}")),
                });
                // Re-entrant sweep: souls this player_soul owns may have already
                // arrived (when `player_id == player_soul card_id` the roster sub
                // `owner_id = N` returns both, in unspecified order). Re-run
                // discovery for every current card it owns so out-of-order
                // arrivals aren't missed.
                let owned: Vec<u32> = self
                    .world
                    .cards
                    .current_all(now)
                    .filter(|c| c.owner_id == card_id && c.card_id != card_id)
                    .map(|c| c.card_id)
                    .collect();
                for soul in owned {
                    self.discover(soul);
                }
            }
            return;
        }

        // Role 2: a soul owned by one of our player_souls → anchor + inventory.
        if self.player_souls.contains(&owner_id) {
            let Some((q, r)) = hex else { return }; // must be world-placed
            let first_sight = !self.souls.contains_key(&card_id);
            let moved = self.souls.get(&card_id) != Some(&(q, r));
            if first_sight {
                // The soul's own inventory is always its action container — ensure
                // it directly (don't depend on the soul def carrying
                // aspect.inventory). Mark it so the generic recursive ensure
                // doesn't double-hold the same zone.
                self.inventoried.insert(card_id);
                self.zones.ensure_inventory(card_id, now);
            }
            if moved {
                self.souls.insert(card_id, (q, r));
                // Tier radii come from the soul def's `anchor_*` aspects.
                let radii = self.anchor_radii(packed);
                self.zones.set_anchor(
                    &format!("soul:{card_id}"),
                    q,
                    r,
                    resonantdust_codec::packed::WORLD_LAYER,
                    0,
                    radii,
                    card_id, // the soul this anchor represents (per-soul memory)
                    now,
                );
            }
        }
    }

    /// Read a card def's anchor tier radii (`anchor_active`/`_hot`/`_warm`/
    /// `_cold` aspects) from the content bundle — the same `card_view` + aspect
    /// read the gate uses (`def_aspect_total`). Zero/absent aspects → that tier
    /// is not provided. Empty radii if the bundle isn't loaded or the def is
    /// unknown.
    fn anchor_radii(&self, packed_definition: u16) -> crate::zones::AnchorRadii {
        let read = |asp: &str| self.def_aspect(packed_definition, asp).unwrap_or(0) as i32;
        crate::zones::AnchorRadii {
            active: read("anchor_active"),
            hot: read("anchor_hot"),
            warm: read("anchor_warm"),
            cold: read("anchor_cold"),
        }
    }

    /// Whether a card def carries `aspect.inventory` (it's a container). Read off
    /// the bundle like [`anchor_radii`](Self::anchor_radii). `false` if the bundle
    /// isn't loaded or the def is unknown.
    fn has_inventory(&self, packed_definition: u16) -> bool {
        self.def_aspect(packed_definition, "inventory").is_some_and(|v| v != 0)
    }

    /// Read one `aspect.<name>` int off a card def via the content bundle (the
    /// same `card_view` + aspect read the gate uses). `None` if the bundle isn't
    /// loaded, the def is unknown, or the aspect is absent.
    fn def_aspect(&self, packed_definition: u16, aspect: &str) -> Option<i64> {
        let bundle = self.bundle.as_ref()?;
        let def_id =
            bundle.name_for_packed(packed_definition).and_then(|name| bundle.card_def_id(name))?;
        self.aspect_of_def_id(def_id, aspect)
    }

    /// Read one `aspect.<name>` int off a bundle `def_id` directly (the synthetic
    /// tile already carries its `def_id`, so this avoids a packed round-trip).
    fn aspect_of_def_id(&self, def_id: u16, aspect: &str) -> Option<i64> {
        use resonantdust_dsl::bridge::{card_view, Card};
        use resonantdust_dsl::vm::{Cell, Store};
        let bundle = self.bundle.as_ref()?;
        let store = Store::with_root(card_view(bundle, &Card { def_id, stock: Vec::new() }));
        store.read(&format!("aspect.{aspect}")).map(Cell::as_int)
    }

    /// Ensure the inventory zone for any container we **transitively own** that
    /// carries `aspect.inventory` — once per container (tracked in `inventoried`).
    /// Recursion is row-driven: subscribing a container's inventory streams its
    /// children, whose `cards` rows re-enter here, so a chest-in-a-chest ensures
    /// down the whole owned tree without an explicit walk.
    fn ensure_owned_inventory(&mut self, card_id: u32) {
        if self.inventoried.contains(&card_id) || self.owning_soul(card_id).is_none() {
            return;
        }
        let Some(packed) = self.world.cards.current(card_id, self.clock_ms).map(|c| c.packed_definition)
        else {
            return;
        };
        if self.has_inventory(packed) {
            self.inventoried.insert(card_id);
            self.zones.ensure_inventory(card_id, self.clock_ms);
        }
    }

    /// Place (stack / move loose) `card_id` per `placement`. Runs the shared
    /// [`stack::plan_place`] over the local world (the SAME resolution the shard
    /// runs), then applies the resolved writes to the **local world** and marks
    /// them dirty — **no server call**. Position is client-authoritative, so this
    /// is the source of truth, not prediction of server state; the dirty set is
    /// flushed via [`flush_positions`](Self::flush_positions) before a recipe
    /// proposal (commit-based sync — see [[project_card_position_sync]]). The
    /// moved card's chain root is marked dirty so the matcher re-evaluates.
    ///
    /// Place a card loose at a GLOBAL world cell `(q, r)` on `(surface, owner)` —
    /// folds the global coord to `(macro_zone, local)` and builds the Loose
    /// placement. The view's drop path calls this with the resolved drop cell.
    pub fn place_loose(
        &mut self,
        card_id: u32,
        surface: u8,
        owner: u32,
        q: i32,
        r: i32,
    ) -> Result<Vec<ClientMsg>, String> {
        use resonantdust_codec::packed::{pack_macro_zone_full, zone_local};
        let (zq, lq) = zone_local(q);
        let (zr, lr) = zone_local(r);
        let macro_zone = pack_macro_zone_full(owner, surface, zq, zr);
        self.place(card_id, stack::Placement::Loose { surface, macro_zone, q: lq, r: lr, x: 0, y: 0 })
    }

    /// Stack a card onto `parent_id` in `direction` (the drop-on-a-card path).
    pub fn place_stack(
        &mut self,
        card_id: u32,
        parent_id: u32,
        direction: u8,
    ) -> Result<Vec<ClientMsg>, String> {
        self.place(card_id, stack::Placement::Stack { parent_id, direction })
    }

    /// Returns `Err` (with reason) if the move is infeasible; an empty frame vec
    /// on success (nothing goes on the wire until a commit).
    pub fn place(
        &mut self,
        card_id: u32,
        placement: stack::Placement,
    ) -> Result<Vec<ClientMsg>, String> {
        let caller = self.player_id.unwrap_or(0);
        let now = self.clock_ms;
        // Stacking eligibility reads each card's content bit-fields; resolve them
        // from our bundle (fall back to the regular-card default if it isn't
        // loaded yet). See [`resonantdust_codec::stacking`].
        let bundle = self.bundle.as_ref();
        let bits = |p: u16| {
            bundle
                .map(|b| resonantdust_dsl::defs::stack_bits(b, p))
                .unwrap_or(resonantdust_codec::stacking::DEFAULT_BITS)
        };
        // The source's chain root BEFORE the move. If a member leaves a chain (a
        // drag-carry or unstack), that old root's queue must be re-evaluated even
        // though its own row never changes — it just lost members. Marking the new
        // roots (below) alone would leave a stale recipe queued on the old chain.
        let old_root = self.chain_root(card_id);
        let plan = plan_place(&self.world, card_id, placement, caller, now, &bits)?;
        let mut moved: Vec<u32> = Vec::with_capacity(plan.writes.len());
        for w in plan.writes {
            let Some(mut row) = self.world.cards.current(w.card_id, now).cloned() else {
                continue;
            };
            let (micro_location, flags) = w.micro.apply(row.flags);
            row.macro_zone = w.macro_zone;
            row.micro_location = micro_location;
            row.flags = flags;
            row.valid_at = resonantdust_codec::packed::pack_valid_at(now, self.next_local_seq());
            self.world.cards.apply(RowOp::Update, row);
            self.dirty_positions.insert(w.card_id);
            moved.push(w.card_id);
            if let Some(root) = self.chain_root(w.card_id) {
                self.dirty_roots.insert(root);
            }
        }
        // Re-evaluate the chain the source departed (it lost members → a queued
        // recipe there may no longer match).
        if let Some(root) = old_root {
            self.dirty_roots.insert(root);
        }
        // Anchor each moved card's position across its WHOLE version history,
        // including any future-stamped server row already in the store (a recipe
        // completion finalize). Without this, that future row promotes when the
        // clock reaches it and clobbers our local prediction. New rows arriving
        // after this are handled by the dirty_position check in `apply_row`.
        let pins: Vec<(u32, u64, u32, u32)> = moved
            .iter()
            .filter_map(|&id| {
                self.world.cards.current(id, now).map(|c| (id, c.macro_zone, c.micro_location, c.flags))
            })
            .collect();
        for (id, mz, ml, fl) in pins {
            self.world.cards.pin_position(id, mz, ml, fl);
        }
        // Live-sync gate: a move into/within SHARED space (the destination zone has
        // observers > 1) or of an anchor-carrier syncs immediately; otherwise it
        // stays client-local+dirty until a commit (recipe proposal). The moved
        // cards' positions go out now; other dirty cards (private zones) wait.
        let sync_now =
            moved.iter().any(|&id| self.is_anchor_carrier(id) || self.zone_observed_by_others(id));
        if sync_now {
            if let Some(frame) = self.build_move_cards(&moved) {
                self.pending_out.push(frame);
            }
        }
        Ok(Vec::new())
    }

    /// Monotonic u16 for stamping local position rows' `valid_at` seq.
    fn next_local_seq(&mut self) -> u16 {
        let s = self.local_seq;
        self.local_seq = self.local_seq.wrapping_add(1);
        s
    }

    /// Build ONE batched `move_cards` call for `ids`, reading each card's resolved
    /// `(macro_zone, micro_location, stack_state)` from the local world (verbatim —
    /// the client already resolved the move) and clearing them from the dirty set.
    /// `None` if none resolve. The single point that emits `move_cards`.
    fn build_move_cards(&mut self, ids: &[u32]) -> Option<ClientMsg> {
        let now = self.clock_ms;
        let pmask = resonantdust_codec::card_model::placement_mask();
        let mut card_ids = Vec::new();
        let mut macro_zones = Vec::new();
        let mut micros = Vec::new();
        let mut stacks = Vec::new();
        for &id in ids {
            if let Some(c) = self.world.cards.current(id, now) {
                card_ids.push(id);
                macro_zones.push(c.macro_zone);
                micros.push(c.micro_location);
                stacks.push((c.flags & pmask) as u8);
            }
            self.dirty_positions.remove(&id);
        }
        if card_ids.is_empty() {
            return None;
        }
        let caller = self.player_id.unwrap_or(0);
        let cid = self.cid();
        Some(ClientMsg::Call {
            cid,
            reducer: "move_cards".to_string(),
            args: serde_json::json!({
                "client_time_ms": now,
                "caller_player_id": caller,
                "card_ids": card_ids,
                "macro_zones": macro_zones,
                "micro_locations": micros,
                "stack_states": stacks,
            }),
        })
    }

    /// The `cost` aspect of the tile at world cell `(macro_zone, local lq, lr)`,
    /// read from the zone's stored tile grid (same path as [`Self::synthetic_tile`]).
    /// `None` if the zone tiles aren't loaded or the cell is empty.
    fn tile_cost(&self, macro_zone: u64, lq: usize, lr: usize) -> Option<i64> {
        use resonantdust_codec::packed::{pack_definition, tile_def_id, tile_slot};
        let now = self.clock_ms;
        let zone = self.world.zones.current(macro_zone, now)?;
        let words = zone.tile_words();
        let def_id_tile = tile_def_id(&words, tile_slot(lq as u8, lr as u8));
        if def_id_tile == 0 {
            return None;
        }
        self.def_aspect(pack_definition(zone.tile_card_type(), def_id_tile), "cost")
    }

    /// The `cost` of the world tile at `(wq, wr)` (decomposes world hex → zone +
    /// local), via [`Self::tile_cost`].
    fn world_tile_cost(&self, wq: i32, wr: i32) -> Option<i64> {
        use resonantdust_codec::packed::{pack_macro_zone_full, zone_local, WORLD_LAYER};
        let (zq, lq) = zone_local(wq);
        let (zr, lr) = zone_local(wr);
        let mz = pack_macro_zone_full(0, WORLD_LAYER, zq, zr);
        self.tile_cost(mz, lq as usize, lr as usize)
    }

    /// Build ONE future-stamped `move_soul` step: `soul` departs `from` at
    /// `depart_ms` and arrives at the adjacent cell `dest` at `depart + travel`,
    /// `travel = (cost_from + cost_dest)·1000 / (2·speed)` ms. Returns
    /// `(frame, arrival_ms)`. `None` if the soul/tiles/speed don't resolve (the
    /// cells must be in a loaded/active zone). The single point that emits
    /// `move_soul`; the single-step and pipelined paths both go through it.
    fn build_move_soul_step(
        &mut self,
        soul: u32,
        from: (i32, i32),
        dest: (i32, i32),
        depart_ms: u64,
    ) -> Option<(ClientMsg, u64)> {
        use resonantdust_codec::packed::{pack_macro_zone_full, pack_micro_loose, zone_local, WORLD_LAYER};
        let soul_def = self.world.cards.current(soul, self.clock_ms)?.packed_definition;
        let speed = self.def_aspect(soul_def, "speed")?;
        if speed <= 0 {
            return None;
        }
        let cost_from = self.world_tile_cost(from.0, from.1)?;
        let cost_dest = self.world_tile_cost(dest.0, dest.1)?;
        let travel_ms = (((cost_from + cost_dest) * 1000) / (2 * speed)).max(1) as u64;
        let arrival_ms = depart_ms + travel_ms;

        let (dzq, dlq) = zone_local(dest.0);
        let (dzr, dlr) = zone_local(dest.1);
        let dest_macro = pack_macro_zone_full(0, WORLD_LAYER, dzq, dzr);
        let dest_micro = pack_micro_loose(dlq, dlr, 0, 0);
        let caller = self.player_id.unwrap_or(0);
        let cid = self.cid();
        // The gate RE-DERIVES `arrival_ms` from `soul_def`/`from`/`dest`/`depart_ms`
        // (content cost + speed) and overrides ours; the shard verifies `soul_def`
        // and the soul's actual cell at `depart_ms` == `from`. So our values are a
        // prediction the server validates — a lie just gets the move rejected.
        let frame = ClientMsg::Call {
            cid,
            reducer: "move_soul".to_string(),
            args: serde_json::json!({
                "client_time_ms": self.clock_ms,
                "caller_player_id": caller,
                "soul_id": soul,
                "soul_def": soul_def,
                "from_q": from.0,
                "from_r": from.1,
                "dest": { "surface": WORLD_LAYER, "macro_zone": dest_macro, "micro_location": dest_micro },
                "depart_ms": depart_ms,
                "arrival_ms": arrival_ms,
            }),
        };
        Some((frame, arrival_ms))
    }

    /// A single immediate step from `soul`'s current cell to the adjacent
    /// `(dest_q, dest_r)`, departing now ([`Command::MoveStep`]).
    fn build_move_soul(&mut self, soul: u32, dest_q: i32, dest_r: i32) -> Option<ClientMsg> {
        let now = self.clock_ms;
        let from = world_hex(self.world.cards.current(soul, now)?)?;
        self.build_move_soul_step(soul, from, (dest_q, dest_r), now).map(|(f, _)| f)
    }

    /// Plan + start a pipelined walk of `soul` to `(target_q, target_r)`: compute a
    /// hex path, fire step 0 now, and record a [`MovePlan`] so each later step
    /// fires as its predecessor's row arrives ([`advance_movement`](Self::advance_movement)).
    fn start_move_to(&mut self, soul: u32, target_q: i32, target_r: i32) -> Vec<ClientMsg> {
        let now = self.clock_ms;
        let Some(start) = self.world.cards.current(soul, now).and_then(world_hex) else {
            return Vec::new();
        };
        let path = hex_path(start, (target_q, target_r));
        if path.is_empty() {
            self.movements.remove(&soul);
            return Vec::new();
        }
        match self.build_move_soul_step(soul, start, path[0], now) {
            Some((frame, arrival)) => {
                self.movements.insert(
                    soul,
                    MovePlan { last_dest: path[0], last_arrival: arrival, requested: 1, path },
                );
                vec![frame]
            }
            None => Vec::new(),
        }
    }

    /// Pipeline hook: a future-stamped row for `soul` arrived at `arrival_ms`. If
    /// it confirms the last-requested movement step, fire the next one (departing
    /// at `arrival_ms`, the moment the soul reaches that cell), pushing it onto
    /// the outbound queue. No-op if `soul` isn't moving or the path is complete.
    fn advance_movement(&mut self, soul: u32, arrival_ms: u64) {
        let Some((next, from)) = self.movements.get(&soul).and_then(|p| {
            (p.requested < p.path.len()).then(|| (p.path[p.requested], p.last_dest))
        }) else {
            return;
        };
        // Confirm: the row at `arrival_ms` is the cell the last step landed on.
        if self.world.cards.current(soul, arrival_ms).and_then(world_hex) != Some(from) {
            return;
        }
        if let Some((frame, new_arrival)) = self.build_move_soul_step(soul, from, next, arrival_ms) {
            self.pending_out.push(frame);
            if let Some(p) = self.movements.get_mut(&soul) {
                p.requested += 1;
                p.last_dest = next;
                p.last_arrival = new_arrival;
            }
        }
    }

    /// Flush EVERY locally-moved (dirty) card position to the server as one batched
    /// `move_cards` — the commit point of the commit-based position model (called
    /// before a recipe proposal). Empty dirty set → no frame.
    pub fn flush_positions(&mut self) -> Vec<ClientMsg> {
        let ids: Vec<u32> = self.dirty_positions.iter().copied().collect();
        self.build_move_cards(&ids).into_iter().collect()
    }

    /// Distinct connections (other than us) the gate reports watching `card_id`'s
    /// current zone — a `> 1` count means shared space, so a move there must sync.
    fn zone_observed_by_others(&self, card_id: u32) -> bool {
        self.world
            .cards
            .current(card_id, self.clock_ms)
            .and_then(|c| self.zone_observers.get(&c.macro_zone).copied())
            .is_some_and(|n| n > 1)
    }

    /// Whether `card_id` is an anchor-carrier whose position drives observation —
    /// a soul, or a container carrying `aspect.inventory` (an anchor at its own
    /// inventory). Its moves ALWAYS sync (can't defer — others' observation of the
    /// zones it anchors depends on it).
    fn is_anchor_carrier(&self, card_id: u32) -> bool {
        use resonantdust_codec::packed::{unpack_definition, SOUL_CARD_TYPE};
        self.world.cards.current(card_id, self.clock_ms).is_some_and(|c| {
            let (card_type, _) = unpack_definition(c.packed_definition);
            card_type == SOUL_CARD_TYPE || self.has_inventory(c.packed_definition)
        })
    }

    /// Install the content bundle (recipe ids + defs) fetched from the gate, then
    /// build the recipe index ([`RecipeMeta`] + `indexed_aspects`) over it.
    pub fn set_bundle(&mut self, bundle: Bundle) {
        self.bundle = Some(bundle);
        self.build_recipe_index();
    }

    /// Precompute per-recipe matcher metadata once: parse iterators, root usage,
    /// branch mask, input count, and the strictly-required top-level aspects. The
    /// match pass then iterates this table instead of re-walking every recipe's
    /// AST. A recipe with neither iterators nor a `root` reference has no anchor
    /// to operate on and is dropped; root-only recipes (no iterators but
    /// referencing `root`) are kept — they fire ON their root.
    fn build_recipe_index(&mut self) {
        use resonantdust_dsl::parser::Stmt;
        let mut meta: Vec<RecipeMeta> = Vec::new();
        let mut indexed: HashSet<String> = HashSet::new();
        if let Some(bundle) = self.bundle.as_ref() {
            let mut rid = 1u16;
            while let Some(name) = bundle.recipe_name(rid) {
                rid += 1;
                let Some(recipe) = bundle.recipe(name) else { continue };
                let iters = iterators(recipe);
                let uses_root = recipe_references_root(recipe);
                if iters.is_empty() && !uses_root {
                    continue; // no anchor at all — nothing to operate on
                }
                let input = recipe.hook("input").map(|h| h.body.as_slice()).unwrap_or(&[]);
                let input_count = input.iter().filter(|s| matches!(s, Stmt::Instr(_))).count();
                let req_aspects = required_top_aspects(recipe);
                for a in &req_aspects {
                    indexed.insert(a.clone());
                }
                meta.push(RecipeMeta {
                    name: name.to_string(),
                    uses_root,
                    want_branches: top_branch_mask(&iters),
                    branch_spans: top_branch_spans(recipe),
                    input_count,
                    req_aspects,
                    iters,
                });
            }
        }
        self.recipe_meta = meta;
        self.indexed_aspects = indexed;
    }

    /// Collect into `out` the `indexed_aspects` present on `card` — the
    /// per-candidate-card probe behind the recipe aspect pre-filter. Reads the
    /// card WITH its stock (via [`card_view`]), so a stock-sourced aspect — a
    /// tile's `pine`/`flora` rolling up to `wood`, build progress, etc. — counts.
    /// Probing the bare def (empty stock) would read those as 0 and UNSOUNDLY skip
    /// any recipe that requires them (e.g. `cut_tree` needs the tile's `wood`).
    fn add_indexed_aspects_card(&self, card: &Card, out: &mut HashSet<String>) {
        use resonantdust_dsl::bridge::card_view;
        use resonantdust_dsl::vm::{Cell, Store};
        let Some(bundle) = self.bundle.as_ref() else { return };
        let store = Store::with_root(card_view(bundle, card));
        for asp in &self.indexed_aspects {
            if !out.contains(asp)
                && store.read(&format!("aspect.{asp}")).map(Cell::as_int).is_some_and(|v| v != 0)
            {
                out.insert(asp.clone());
            }
        }
    }

    /// Propose a recipe action: resolve `recipe_name` → id via the bundle and
    /// build the `propose_action` frame. `bindings[iterator][offset]` are the
    /// card_ids filling the recipe's slots (the client matcher's output);
    /// `(surface, macro_zone, micro_location)` locate the action cell. The gate
    /// is the authority — it gathers, validates the bindings, plans, and applies;
    /// effects stream back as card rows. `Err` if the bundle is missing or the
    /// recipe is unknown.
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &mut self,
        recipe_name: &str,
        root: u32,
        bindings: Vec<Vec<u32>>,
        surface: u8,
        macro_zone: u64,
        micro_location: u32,
    ) -> Result<Vec<ClientMsg>, String> {
        let recipe_id = self
            .bundle
            .as_ref()
            .ok_or("propose: content bundle not loaded")?
            .recipe_def_id(recipe_name)
            .ok_or_else(|| format!("propose: unknown recipe {recipe_name:?}"))?;
        let caller = self.player_id.unwrap_or(0);
        // Commit point: sync any locally-moved card positions FIRST, so the gate
        // gathers the operating set at its real (just-moved) positions before it
        // validates + applies this recipe. The propose carries only bindings.
        let mut out = self.flush_positions();
        let cid = self.cid();
        out.push(ClientMsg::Call {
            cid,
            reducer: "propose_action".to_string(),
            args: serde_json::json!({
                "recipe_id": recipe_id,
                "surface": surface,
                "macro_zone": macro_zone,
                "micro_location": micro_location,
                "root": root,
                "bindings": bindings,
                "caller_player_id": caller,
                "client_time_ms": self.clock_ms,
            }),
        });
        Ok(out)
    }

    /// Discover which recipes apply to `root`'s current board state, with the
    /// bindings that satisfy them — the client matcher. For each recipe in the
    /// bundle, lays `root`'s real stack members into candidate bindings (by
    /// branch/index, per the recipe's iterators) and runs the shared
    /// `match_recipe` @input check; the matches are returned ready to
    /// [`propose`](Self::propose).
    ///
    /// Reads through `soul`'s **memory view**: each card is read at the soul's
    /// watermark for its zone — live (`now`) where the soul is present
    /// (active/hot), or the frozen remembered moment for warm/cold zones. A match
    /// over a remembered (stale) root is tagged `live = false` (a goal: verify on
    /// arrival, don't propose blind); the soul's own inventory is always live.
    ///
    /// First cut: top-level iterators + nested owner-chain (inventory) + the
    /// branch-0 synthetic tile. Candidate-root enumeration over the soul's pool
    /// is the NPC loop's job.
    pub fn match_recipes(&self, soul: u32, root: u32) -> Vec<RecipeMatch> {
        self.match_recipes_inner(soul, root, true)
    }

    /// [`match_recipes`](Self::match_recipes) with the index aspect pre-filter
    /// **disabled** — the exhaustive oracle. The pre-filter must only ever skip
    /// recipes that provably can't match, so this returns the *same* set as the
    /// filtered path; a harness diffs the two to catch an unsound skip. (No-op vs
    /// the filtered path until aspect-guarded recipes exist — `indexed_aspects`
    /// is empty for `def_id eq` recipes.)
    pub fn match_recipes_unfiltered(&self, soul: u32, root: u32) -> Vec<RecipeMatch> {
        self.match_recipes_inner(soul, root, false)
    }

    /// Whether `card_id` is ineligible to be bound into a NEW action. Uses the
    /// SHARED [`bind_blocked`] predicate — the exact verb-independent baseline the
    /// gate's `check_card` gates on — so the matcher never proposes a binding the
    /// gate would reject (dead, or exclusively `slot_claim`-held). (We mirror the
    /// gate's eligibility rather than permanently claiming dead cards — a permanent
    /// claim would defeat GC's `slot_claim`-gated dead-row reaping.)
    fn is_held(&self, card_id: u32, now_ms: u64) -> bool {
        self.world
            .cards
            .current(card_id, now_ms)
            .is_some_and(|c| resonantdust_codec::card_model::bind_blocked(c.flags))
    }

    fn match_recipes_inner(&self, soul: u32, root: u32, use_filter: bool) -> Vec<RecipeMatch> {
        let Some(bundle) = self.bundle.as_ref() else {
            return Vec::new();
        };
        let now = self.clock_ms;

        // Don't propose a NEW action on a root that's exclusively held (by an
        // in-flight action — including this recipe's own holds while it runs) or
        // dead+claimed (a destroyed card stays claimed; see apply_action). The gate
        // would reject it anyway; skipping here stops the wasteful re-propose +
        // "card held/dead" rejection. Mirrors the gate's `check_card`. Self-
        // advancing recipes still re-fire AFTER their hold releases at completion.
        if root != 0 && self.is_held(root, now) {
            return Vec::new();
        }

        // The soul's knowledge-time for the root's zone: `now` if it's present
        // there (live), else the frozen watermark (memory). Drives the whole
        // match's freshness. Inventory/unknown zones aren't remembered → `now`.
        let view_t = self
            .world
            .cards
            .zone_of(root)
            .and_then(|z| self.zones.card_view_time(soul, z, now))
            .unwrap_or(now);
        let live = view_t == now;

        // root's stack members (as the soul knows them, at `view_t`) grouped by
        // `stack_id` (= stack_branch + 1: 1 hex, 2 top, 3 bottom; 0 is the loose
        // root, never a member), each ordered by stack index. Recipe slot numbers
        // now equal the stack_id they address.
        let mut by_branch: BTreeMap<u8, Vec<(u8, u32)>> = BTreeMap::new();
        for m in self.world.members_of(root, view_t) {
            // Skip a member that's exclusively held (in-flight) or dead+claimed —
            // same reason as the held-root guard above.
            if self.is_held(m.card_id, now) {
                continue;
            }
            by_branch.entry(stack_branch(m.flags) + 1).or_default().push((stack_index(m.flags), m.card_id));
        }
        for v in by_branch.values_mut() {
            v.sort_by_key(|(idx, _)| *idx);
        }
        let branch = |b: u8| -> Vec<u32> {
            by_branch.get(&b).map(|v| v.iter().map(|(_, id)| *id).collect()).unwrap_or_default()
        };
        // Indexed by stack_id: [0] is the root slot (`slot.0.0`, filled by the
        // `root` param — never a member list), [1] hex/tile, [2] top, [3] bottom.
        let base_branches: [Vec<u32>; 4] = [Vec::new(), branch(1), branch(2), branch(3)];

        // card_id → typed dsl Card, read at the soul's PER-CARD view time (a
        // world card at the root's view_t, an owned inventory item live at `now`).
        let lookup = |id: u32| -> Option<Card> {
            let card_t = self
                .world
                .cards
                .zone_of(id)
                .and_then(|z| self.zones.card_view_time(soul, z, now))
                .unwrap_or(now);
            let c = self.world.card_at(id, card_t)?;
            let name = bundle.name_for_packed(c.packed_definition)?;
            // Decode the row's per-instance stock u32 → positional slot values so
            // the recipe reads this card's actual stock aspects (build progress,
            // etc.), not just the def defaults.
            let stock = resonantdust_dsl::bridge::stock_to_vec(bundle, name, c.stock);
            Some(Card { def_id: bundle.card_def_id(name)?, stock })
        };

        // Synthetic hex tile: the tile under the root (stack 1), as the soul
        // remembers it (read at `view_t`).
        let synthetic = (root != 0).then(|| self.synthetic_tile(root, &base_branches[1], view_t)).flatten();

        // Aspect pre-filter set: the indexed aspects actually present on the cards
        // that fill this candidate's top-level slots (read at each card's own view
        // time, mirroring `lookup`). A recipe whose strictly-required top aspects
        // aren't all in here is skipped without building a frame.
        let mut available: HashSet<String> = HashSet::new();
        if use_filter && !self.indexed_aspects.is_empty() {
            let mut probe: Vec<u32> = base_branches.iter().flatten().copied().collect();
            if root != 0 {
                probe.push(root);
            }
            for id in probe {
                if let Some(c) = lookup(id) {
                    self.add_indexed_aspects_card(&c, &mut available);
                }
            }
            if let Some(s) = synthetic.as_ref() {
                self.add_indexed_aspects_card(s, &mut available);
            }
        }

        let mut matches = Vec::new();
        for m in &self.recipe_meta {
            // Index pre-filter: a required top-level aspect absent from the
            // candidate's slot-fillers means no @input line can hold → no match.
            if use_filter && !m.req_aspects.iter().all(|a| available.contains(a)) {
                continue;
            }
            let Some(recipe) = bundle.recipe(&m.name) else { continue };

            // The pass list mirrors recipeMatcher.ts: a root-anchored recipe is
            // tried with the root in place (no promotion); a branch recipe is
            // tried with the root promoted into stack 2 (top), then stack 3
            // (bottom) — so a loose root acts as the first member of its chain.
            let passes: Vec<(u32, [Vec<u32>; 4])> = if m.uses_root {
                if root == 0 { vec![] } else { vec![(root, base_branches.clone())] }
            } else if root == 0 {
                vec![(0, base_branches.clone())]
            } else {
                vec![
                    (0, promote_root(root, &base_branches, 2)),
                    (0, promote_root(root, &base_branches, 3)),
                ]
            };

            'passes: for (pass_root, branches) in passes {
                if !anchors_fit(m.uses_root, m.want_branches, pass_root, &branches, synthetic.is_some()) {
                    continue;
                }
                // Slide the top/bottom (stack 2/3) windows over the branch lists so
                // the recipe's slots can match cards at ANY contiguous offset, not
                // just index 0 (`corpus_b_top` = two adjacent corpus anywhere in the
                // top stack; `corpus_dust` = a dust anywhere in it). Each `windowed`
                // narrows stacks 2/3 to one candidate window; lowest offset wins.
                for windowed in window_views(&branches, &m.branch_spans) {
                    let bindings = self.build_bindings(&m.iters, &windowed, synthetic.is_some(), now);
                    // A root-only recipe (no iterators) operates on `pass_root`, which
                    // isn't in `bindings`; only skip the no-cards case when the recipe
                    // actually has slot iterators to fill.
                    if !m.iters.is_empty() && bindings.iter().all(|b| b.is_empty()) {
                        continue;
                    }
                    let mut frame =
                        build_frame(bundle, recipe, pass_root, &bindings, synthetic.as_ref(), &lookup);
                    let input = recipe.hook("input").map(|h| h.body.as_slice()).unwrap_or(&[]);
                    let matched = match_recipe(input, &mut frame.store, &bundle.catalog, &bundle.functions)
                        .map(|p| p.matched)
                        .unwrap_or(false);
                    if matched {
                        matches.push(RecipeMatch {
                            recipe: m.name.clone(),
                            root: pass_root,
                            bindings,
                            live,
                            input_count: m.input_count,
                        });
                        break 'passes; // first matching (pass, window) wins for this recipe
                    }
                }
            }
        }
        matches
    }

    /// Per-iterator card_id bindings for a candidate, given the (possibly
    /// promoted) stack lists indexed by stack_id. Mirrors `buildBindings` in
    /// recipeMatcher.ts: a top-level iterator takes its stack's cards (an empty
    /// hex stack (1) with a synthetic tile takes the `[0]` sentinel); a nested
    /// iterator walks its resolved parent's chain in the iterator's branch.
    fn build_bindings(
        &self,
        iters: &[resonantdust_dsl::recipe::Iter],
        branches: &[Vec<u32>; 4],
        has_synthetic: bool,
        now: u64,
    ) -> Vec<Vec<u32>> {
        iters
            .iter()
            .map(|it| {
                if !it.parent.is_empty() {
                    let parent = self.nested_parent(&it.parent, branches, now);
                    return match parent {
                        0 => Vec::new(),
                        p => self.branch_members(p, it.branch, now),
                    };
                }
                let cards = branches.get(it.branch as usize).cloned().unwrap_or_default();
                if cards.is_empty() && it.branch == 1 && has_synthetic {
                    return vec![0]; // synthetic-tile sentinel (stack 1, the hex member)
                }
                cards
            })
            .collect()
    }

    /// Resolve a nested iterator's parent path (e.g. `slot.1.0.owner`) to a
    /// card_id by walking `slot.B.O` (top-level branch lookup) then `owner`
    /// (owner_id) / `parent` (micro_location root) steps. `0` on any miss.
    /// Mirrors `nestedParent` in recipeMatcher.ts.
    fn nested_parent(&self, parent: &str, branches: &[Vec<u32>; 4], now: u64) -> u32 {
        let segs: Vec<&str> = parent.split('.').collect();
        let mut card_id = 0u32;
        let mut i = 0;
        while i < segs.len() {
            if segs[i] == "slot" && i + 2 < segs.len() {
                let (Ok(b), Ok(o)) = (segs[i + 1].parse::<usize>(), segs[i + 2].parse::<usize>())
                else {
                    return 0;
                };
                card_id = branches.get(b).and_then(|c| c.get(o)).copied().unwrap_or(0);
                if card_id == 0 {
                    return 0;
                }
                i += 3;
            } else if segs[i] == "owner" {
                card_id = match self.world.cards.current(card_id, now) {
                    Some(c) => c.owner_id,
                    None => return 0,
                };
                if card_id == 0 {
                    return 0;
                }
                i += 1;
            } else if segs[i] == "parent" {
                card_id = match self.world.cards.current(card_id, now) {
                    Some(c) => c.micro_location,
                    None => return 0,
                };
                if card_id == 0 {
                    return 0;
                }
                i += 1;
            } else {
                i += 1;
            }
        }
        card_id
    }

    /// A card's stack members in one stack, ordered by stack index (the chain
    /// walker for nested iterators). `branch` is the iterator's `stack_id`
    /// (1/2/3); the codec stores `stack_branch = stack_id − 1`, so compare on
    /// `stack_branch + 1`. Mirrors `buildChain` / `branchWalker`.
    fn branch_members(&self, parent: u32, branch: u8, now: u64) -> Vec<u32> {
        let mut members: Vec<(u8, u32)> = self
            .world
            .members_of(parent, now)
            .into_iter()
            .filter(|m| stack_branch(m.flags) + 1 == branch)
            .map(|m| (stack_index(m.flags), m.card_id))
            .collect();
        members.sort_by_key(|(idx, _)| *idx);
        members.into_iter().map(|(_, id)| id).collect()
    }

    /// The synthetic tile under `root` for branch-0 matching: `Some` only when
    /// `root` sits on a world-tile surface, has no card on its hex branch, and a
    /// non-empty tile occupies its cell. Reads the zone's packed tile grid for
    /// the cell's def + stock (mirrors `getZoneTileSlot` + ActionManager's
    /// synthetic-tile branch). Card-card tile promotion is not modelled yet.
    fn synthetic_tile(&self, root: u32, hex_branch: &[u32], now: u64) -> Option<Card> {
        use resonantdust_codec::card_model::Micro;
        use resonantdust_codec::packed::{pack_definition, surface_of, tile_def_id, tile_slot, tile_stock, WORLD_LAYER};
        if !hex_branch.is_empty() {
            return None; // a card occupies the hex branch — no synthetic tile
        }
        let bundle = self.bundle.as_ref()?;
        let card = self.world.cards.current(root, now)?;
        if surface_of(card.macro_zone) < WORLD_LAYER {
            return None;
        }
        let (lq, lr) = match card.micro() {
            Micro::Loose { local_q, local_r, .. } => (local_q as usize, local_r as usize),
            Micro::Stacked { .. } => return None,
        };
        let zone = self.world.zones.current(card.macro_zone, now)?;
        let words = zone.tile_words();
        let idx = tile_slot(lq as u8, lr as u8);
        let def_id_tile = tile_def_id(&words, idx);
        if def_id_tile == 0 {
            return None;
        }
        let packed = pack_definition(zone.tile_card_type(), def_id_tile);
        let name = bundle.name_for_packed(packed)?;
        Some(Card {
            def_id: bundle.card_def_id(name)?,
            stock: vec![tile_stock(&words, idx, 0) as i64, tile_stock(&words, idx, 1) as i64],
        })
    }

    /// The distinct tile definitions (by name) present in the zone at
    /// `macro_zone`, current now — the "unique tile attributes" search keys the
    /// recipe-availability pre-filter will work over. Empty if the zone or bundle
    /// isn't loaded.
    pub fn zone_tile_names(&self, macro_zone: u64) -> Vec<String> {
        let Some(bundle) = self.bundle.as_ref() else {
            return Vec::new();
        };
        let Some(zone) = self.world.zones.current(macro_zone, self.clock_ms) else {
            return Vec::new();
        };
        let tile_type = zone.tile_card_type();
        zone.unique_tile_def_ids()
            .iter()
            .filter_map(|&def_id| {
                let packed = resonantdust_codec::packed::pack_definition(tile_type, def_id);
                bundle.name_for_packed(packed).map(String::from)
            })
            .collect()
    }

    /// Errors (`Event::Error`) the core has seen since the last drain — protocol
    /// errors and undecodable rows. A driver/harness checks this to fail on
    /// silent corruption. Drains the buffer.
    pub fn drain_errors(&mut self) -> Vec<String> {
        std::mem::take(&mut self.seen_errors)
    }

    /// Fold one inbound frame into the world, returning the events it produced.
    /// Any `Event::Error` is also stashed in `seen_errors` for [`drain_errors`].
    pub fn apply(&mut self, msg: GateMsg) -> Vec<Event> {
        let events = self.apply_inner(msg);
        for ev in &events {
            if let Event::Error { error } = ev {
                self.seen_errors.push(error.clone());
            }
        }
        events
    }

    fn apply_inner(&mut self, msg: GateMsg) -> Vec<Event> {
        match msg {
            GateMsg::Time { server_micros } => self.sample_clock(&server_micros),
            GateMsg::Applied { sid } => vec![Event::Applied { sid }],
            GateMsg::Error { error } => vec![Event::Error { error }],
            GateMsg::CallOk { cid, server_micros } => {
                let mut out = self.sample_clock(&server_micros);
                self.resolve_action(cid, None);
                self.inflight_calls.remove(&cid);
                out.push(Event::CallOk { cid });
                out
            }
            GateMsg::CallErr { cid, error, server_micros } => {
                // A gate `time_drift` rejection re-seats the clock toward server,
                // using this call's recorded send instant (RTT midpoint).
                self.note_drift(cid, &error);
                let mut out = self.sample_clock(&server_micros);
                self.resolve_action(cid, Some(&error));
                self.inflight_calls.remove(&cid);
                out.push(Event::CallErr { cid, error });
                out
            }
            GateMsg::Row { table, op, row, .. } => self.apply_row(&table, op, row),
            // Content hot-swap broadcast — definitions changed gate-side. No card
            // state to fold; a future definition cache would refresh here.
            GateMsg::ContentChanged { .. } => vec![],
            // Live observer count for a zone. Store it (drives the move-sync gate),
            // and on a ≤1→>1 transition flush any dirty cards in that zone so the
            // newly-arrived observer sees their true positions.
            GateMsg::ZoneObservers { macro_zone, observers } => {
                if let Ok(zone) = macro_zone.parse::<u64>() {
                    let prev = self.zone_observers.get(&zone).copied().unwrap_or(0);
                    if observers <= 1 {
                        self.zone_observers.remove(&zone);
                    } else {
                        self.zone_observers.insert(zone, observers);
                    }
                    if prev <= 1 && observers > 1 {
                        let now = self.clock_ms;
                        let ids: Vec<u32> = self
                            .dirty_positions
                            .iter()
                            .copied()
                            .filter(|&id| {
                                self.world.cards.current(id, now).is_some_and(|c| c.macro_zone == zone)
                            })
                            .collect();
                        if let Some(frame) = self.build_move_cards(&ids) {
                            self.pending_out.push(frame);
                        }
                    }
                }
                vec![]
            }
        }
    }

    /// Advance the local monotonic clock (driver-supplied `perf_ms`) and refresh
    /// the cached server-time estimate between frames. The driver calls this each
    /// loop iteration so outbound stamps stay fresh without a new sample.
    pub fn tick(&mut self, perf_ms: f64) {
        self.perf_ms = perf_ms;
        if self.clock.is_synced() {
            self.clock_ms = self.clock.server_now_ms(perf_ms).max(0.0) as u64;
        }
        // Promotion kick: future rows that have come due re-mark their chain root
        // dirty (no Insert fires on promotion, so the matcher would otherwise never
        // re-evaluate a recipe-created/changed card). See `pending_promotions`.
        if !self.pending_promotions.is_empty() {
            let now = self.clock_ms;
            let mut due: Vec<u32> = Vec::new();
            self.pending_promotions.retain(|(t, id)| {
                if *t <= now {
                    due.push(*id);
                    false
                } else {
                    true
                }
            });
            for id in due {
                if let Some(root) = self.chain_root(id) {
                    self.dirty_roots.insert(root);
                }
            }
        }
        // Budgeted re-identification: drain the dirty-root set at most ~1/s.
        if !self.dirty_roots.is_empty() && perf_ms - self.last_match_perf >= MATCH_BUDGET_MS {
            self.last_match_perf = perf_ms;
            self.flush_dirty();
        }
        self.fire_ready();
        // Magnetic pass (budgeted): drive the magnets we own — gather in-range
        // cards onto them to satisfy their recipe, or fire the overdue fallback.
        if perf_ms - self.last_magnet_perf >= MAGNET_BUDGET_MS {
            self.last_magnet_perf = perf_ms;
            self.magnetic_pass();
        }
    }

    /// One tick of the "magnetic player": drive every magnetic card we own. For
    /// each magnet, pull in-range same-zone loose cards onto it to satisfy its
    /// `magnetic.recipe` (the ordinary matcher then queues + proposes success),
    /// and propose its `magnetic.failure` once the `magnetic.duration` window
    /// lapses. Reuses [`Client::place`] (stacking marks the magnet dirty →
    /// `evaluate_root`) and [`Client::propose`] (root-only failure). Staged so the
    /// world/bundle borrows are released before the `&mut self` mutations.
    fn magnetic_pass(&mut self) {
        use resonantdust_codec::card_model::Micro;
        use resonantdust_codec::packed::{owner_of, surface_of, unpack_macro_zone, world_tile};
        if self.player_souls.is_empty() {
            return;
        }
        let Some(bundle) = self.bundle.as_ref() else { return };
        let now = self.clock_ms;

        struct Plan {
            magnet: u32,
            surface: u8,
            macro_zone: u64,
            micro_location: u32,
            cell: (i32, i32),
            radius: i32,
            duration_ms: u64,
            failure: Option<String>,
            /// Outstanding (branch, packed_def) the magnet still needs gathered.
            need: Vec<(u8, u16)>,
        }

        // Phase A — read each owned magnet into an owned Plan.
        let mut plans: Vec<Plan> = Vec::new();
        for c in self.world.cards.current_all(now) {
            if !self.player_souls.contains(&c.owner_id) {
                continue;
            }
            let Some(recipe_id) = bundle.magnetic_recipe_id(c.packed_definition) else { continue };
            // Only a loose card is a gatherable root.
            let Micro::Loose { local_q, local_r, .. } = Micro::of(c.micro_location, c.flags) else {
                continue;
            };
            let Some(success) = bundle.recipe_name(recipe_id).map(str::to_string) else { continue };
            let Some(node) = bundle.recipe(&success) else { continue };
            // Required (branch, def) counts, minus what's already stacked.
            let mut want: HashMap<(u8, u16), u32> = HashMap::new();
            for (branch, _off, def) in resonantdust_dsl::recipe::required_input_defs(bundle, node) {
                *want.entry((branch, def)).or_insert(0) += 1;
            }
            let mut have: HashMap<(u8, u16), u32> = HashMap::new();
            for m in self.world.members_of(c.card_id, now) {
                if let Micro::Stacked { branch, .. } = Micro::of(m.micro_location, m.flags) {
                    *have.entry((branch, m.packed_definition)).or_insert(0) += 1;
                }
            }
            let mut need: Vec<(u8, u16)> = Vec::new();
            for ((branch, def), n) in want {
                for _ in have.get(&(branch, def)).copied().unwrap_or(0)..n {
                    need.push((branch, def));
                }
            }
            plans.push(Plan {
                magnet: c.card_id,
                surface: surface_of(c.macro_zone),
                macro_zone: c.macro_zone,
                micro_location: c.micro_location,
                cell: (local_q as i32, local_r as i32),
                radius: bundle.magnetic_radius(c.packed_definition).unwrap_or(0),
                duration_ms: bundle.magnetic_duration_ms(c.packed_definition).unwrap_or(0),
                failure: bundle
                    .magnetic_failure_recipe_id(c.packed_definition)
                    .and_then(|id| bundle.recipe_name(id))
                    .map(str::to_string),
                need,
            });
        }
        if plans.is_empty() {
            let known: Vec<u32> = self.magnet_deadlines.keys().copied().collect();
            for m in known {
                self.zones.clear_anchor(&format!("mag:{m}"), now);
            }
            self.magnet_deadlines.clear();
            self.magnet_failed.clear();
            let frames = self.zone_frames();
            self.pending_out.extend(frames);
            return;
        }

        // Phase B — pick in-range loose candidates for each shortfall. A card is
        // claimed once across the whole pass (`used`), so two magnets / two slots
        // never grab the same card.
        let mut used: HashSet<u32> = HashSet::new();
        let mut stack_ops: Vec<(u32, u32, u8)> = Vec::new(); // (candidate, magnet, direction)
        for p in &plans {
            for &(branch, def) in &p.need {
                let pick = self
                    .world
                    .cards
                    .current_all(now)
                    .filter(|c| {
                        c.card_id != p.magnet
                            && c.macro_zone == p.macro_zone
                            && c.packed_definition == def
                            && !used.contains(&c.card_id)
                            && matches!(Micro::of(c.micro_location, c.flags), Micro::Loose { .. })
                    })
                    .filter_map(|c| match Micro::of(c.micro_location, c.flags) {
                        Micro::Loose { local_q, local_r, .. } => {
                            let d = hex_distance(p.cell, (local_q as i32, local_r as i32));
                            (d <= p.radius).then_some((d, c.card_id))
                        }
                        _ => None,
                    })
                    .min(); // nearest, then lowest id
                if let Some((_, cand)) = pick {
                    used.insert(cand);
                    // recipe branch (1 hex / 2 top / 3 bottom) → STACK_DIR (0/1/2).
                    stack_ops.push((cand, p.magnet, branch.saturating_sub(1)));
                }
            }
        }

        // Phase C — mutate. Anchor each magnet so its neighbourhood (the cards it
        // attracts, which we may not own) is subscribed; idempotent per pass.
        for p in &plans {
            let (zq, zr) = unpack_macro_zone(p.macro_zone);
            self.zones.set_anchor(
                &format!("mag:{}", p.magnet),
                world_tile(zq, p.cell.0 as u8),
                world_tile(zr, p.cell.1 as u8),
                p.surface,
                owner_of(p.macro_zone),
                crate::zones::AnchorRadii { active: p.radius.max(1), ..Default::default() },
                0,
                now,
            );
        }
        // Stack gathered cards, fire overdue failures, prune.
        for (cand, magnet, direction) in stack_ops {
            if let Ok(frames) =
                self.place(cand, stack::Placement::Stack { parent_id: magnet, direction })
            {
                self.pending_out.extend(frames);
            }
            self.dirty_roots.insert(magnet);
        }
        for p in &plans {
            let deadline = *self.magnet_deadlines.entry(p.magnet).or_insert(now + p.duration_ms);
            if now >= deadline && !self.magnet_failed.contains(&p.magnet) {
                if let Some(fail) = &p.failure {
                    if let Ok(frames) = self.propose(
                        fail,
                        p.magnet,
                        Vec::new(),
                        p.surface,
                        p.macro_zone,
                        p.micro_location,
                    ) {
                        self.pending_out.extend(frames);
                        self.magnet_failed.insert(p.magnet);
                    }
                }
            }
        }
        // Drop anchors + state for magnets that are gone (consumed / not ours).
        let live: HashSet<u32> = plans.iter().map(|p| p.magnet).collect();
        let stale: Vec<u32> =
            self.magnet_deadlines.keys().copied().filter(|k| !live.contains(k)).collect();
        for m in stale {
            self.zones.clear_anchor(&format!("mag:{m}"), now);
        }
        self.magnet_deadlines.retain(|k, _| live.contains(k));
        self.magnet_failed.retain(|k| live.contains(k));
        let frames = self.zone_frames();
        self.pending_out.extend(frames);
    }

    /// Re-evaluate every dirty chain root and clear the set: resolve each root's
    /// acting soul and reconcile its queue entry. A root no longer reachable by
    /// any of our souls (moved out of scope / consumed) drops its pending action.
    fn flush_dirty(&mut self) {
        let roots: Vec<u32> = self.dirty_roots.drain().collect();
        for root in roots {
            match self.actor_soul_for_root(root) {
                Some(soul) => self.evaluate_root(soul, root),
                None => {
                    if self.actions.get(&root).is_some_and(|a| a.submit_cid.is_none()) {
                        self.actions.remove(&root);
                    }
                }
            }
        }
    }

    // ── triggered action queue ─────────────────────────────────────────────────

    /// Re-evaluate `root` for `soul` and reconcile the action queue: a new best
    /// LIVE match queues (or keeps a running timer if unchanged, or supersedes +
    /// resets on change); no match drops a non-in-flight entry. The best match is
    /// the one binding the most cards (e.g. `triple_corpus` over `corpus_b_top`).
    /// Single-input recipes get a zero debounce (fire-at-once).
    fn evaluate_root(&mut self, soul: u32, root: u32) {
        // An in-flight action (proposed, awaiting its reply) must be left alone:
        // re-queuing it would overwrite its `submit_cid`, so the reply could never
        // resolve the action — it would dangle and its outcome be lost (the
        // re-propose lands as a "already in flight" dup against the gate's dedup).
        // `resolve_action` clears it on reply; the next evaluate re-queues then.
        if self.actions.get(&root).is_some_and(|a| a.submit_cid.is_some()) {
            return;
        }
        let now = self.clock_ms;
        let best = self
            .match_recipes(soul, root)
            .into_iter()
            .filter(|m| m.live)
            .max_by_key(|m| m.priority());
        let Some(m) = best else {
            // No match → cancel a queued (not-yet-sent) entry for this root.
            if self.actions.get(&root).is_some_and(|a| a.submit_cid.is_none()) {
                self.actions.remove(&root);
            }
            return;
        };
        // Action location = the original root card's cell.
        let Some(rc) = self.world.cards.current(root, now) else { return };
        let (surface, macro_zone, micro_location) = (
            resonantdust_codec::packed::surface_of(rc.macro_zone),
            rc.macro_zone,
            rc.micro_location,
        );
        let delay_ms = if m.input_count <= 1 { 0.0 } else { DEFAULT_DELAY_MS };

        // Unchanged + still pending → keep the running debounce timer.
        if let Some(a) = self.actions.get_mut(&root) {
            if a.submit_cid.is_none() && a.same_as(&m.recipe, &m.bindings) {
                a.surface = surface;
                a.macro_zone = macro_zone;
                a.micro_location = micro_location;
                return;
            }
        }
        // New / superseding match → (re)queue, resetting the debounce timer.
        self.actions.insert(
            root,
            QueuedAction {
                soul,
                recipe: m.recipe,
                root: m.root,
                bindings: m.bindings,
                surface,
                macro_zone,
                micro_location,
                scheduled_at: self.perf_ms,
                delay_ms,
                retry_count: 0,
                submit_cid: None,
            },
        );
    }

    /// Propose every queued action whose debounce window has elapsed. Re-checks
    /// the match one last time (config may have changed during the window);
    /// drops it if it no longer holds, else emits `propose_action` (frames go to
    /// `pending_out`) and records its `cid` for reply matching.
    fn fire_ready(&mut self) {
        let perf = self.perf_ms;
        let ready: Vec<u32> =
            self.actions.iter().filter(|(_, a)| a.ready(perf)).map(|(r, _)| *r).collect();
        for root in ready {
            let Some(a) = self.actions.get(&root).cloned() else { continue };
            // Fire-time safety re-eval against the **chain root** (the map key) —
            // NOT `a.root`, which is the *propose* root and is `0` whenever the
            // matcher folded the loose root into a branch (e.g. triple_corpus).
            // Re-matching on root 0 finds nothing, so using it here would drop
            // every folded-root action without ever proposing it.
            let still = self
                .match_recipes(a.soul, root)
                .into_iter()
                .any(|m| m.live && m.recipe == a.recipe && m.bindings == a.bindings);
            if !still {
                self.actions.remove(&root);
                continue;
            }
            match self.propose(&a.recipe, a.root, a.bindings.clone(), a.surface, a.macro_zone, a.micro_location) {
                Ok(frames) => {
                    // The propose_action is the LAST frame — propose() may prepend a
                    // `move_cards` position flush. Track the propose cid, not the flush's.
                    if let Some(ClientMsg::Call { cid, .. }) = frames.last() {
                        if let Some(e) = self.actions.get_mut(&root) {
                            e.submit_cid = Some(*cid);
                        }
                    }
                    self.pending_out.extend(frames);
                }
                Err(_) => {
                    self.actions.remove(&root);
                }
            }
        }
    }

    /// A reducer reply for an in-flight queued action: `Ok` (or "already in
    /// flight" dedup) clears it; a `time_drift:client_ahead` reschedules it (up
    /// to [`MAX_TIME_DRIFT_RETRIES`]); any other error drops it.
    fn resolve_action(&mut self, cid: u32, error: Option<&str>) {
        let Some((&root, _)) = self.actions.iter().find(|(_, a)| a.submit_cid == Some(cid)) else {
            return;
        };
        let recipe = self.actions.get(&root).map(|a| a.recipe.clone()).unwrap_or_default();
        match error {
            None => {
                self.action_outcomes.push((recipe, None)); // accepted
                self.actions.remove(&root);
            }
            Some(e) if e.contains("already in flight") => {
                self.action_outcomes.push((recipe, None)); // a prior attempt landed — done
                self.actions.remove(&root);
            }
            Some(e) if e.contains("client_ahead_by") => {
                let gap = e
                    .split("client_ahead_by=")
                    .nth(1)
                    .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(0.0);
                if let Some(a) = self.actions.get_mut(&root) {
                    if a.retry_count < MAX_TIME_DRIFT_RETRIES {
                        a.retry_count += 1;
                        a.delay_ms = gap + TIME_DRIFT_RETRY_PAD_MS;
                        a.scheduled_at = self.perf_ms;
                        a.submit_cid = None; // re-arm to refire after the gap
                    } else {
                        self.actions.remove(&root);
                    }
                }
            }
            Some(e) => {
                self.action_outcomes.push((recipe, Some(e.to_string()))); // dropped on rejection
                self.actions.remove(&root);
            }
        }
    }

    /// Drain the final outcomes of fired queued actions: `(recipe, Err(reason)?)`.
    /// A driver/harness uses this to distinguish a real accept (`None`) from a
    /// silent drop on rejection (`Some(reason)`) — the queue empties on both.
    pub fn drain_action_outcomes(&mut self) -> Vec<(String, Option<String>)> {
        std::mem::take(&mut self.action_outcomes)
    }

    /// The chain root of a card: the card itself if loose, else its stack root.
    /// `None` if the card isn't currently known.
    fn chain_root(&self, card_id: u32) -> Option<u32> {
        use resonantdust_codec::card_model::Micro;
        let card = self.world.cards.current(card_id, self.clock_ms)?;
        Some(match card.micro() {
            Micro::Stacked { root, .. } => root,
            Micro::Loose { .. } => card_id,
        })
    }

    /// Walk a card's `owner_id` chain to a soul we control, or `None`. A soul
    /// owns the cards in its inventory (directly or via nested containers); this
    /// is how a root in a soul's pocket resolves to its acting soul. Depth-capped.
    fn owning_soul(&self, card_id: u32) -> Option<u32> {
        let now = self.clock_ms;
        let mut cur = card_id;
        for _ in 0..32 {
            if self.souls.contains_key(&cur) {
                return Some(cur);
            }
            let c = self.world.cards.current(cur, now)?;
            if c.owner_id == 0 || c.owner_id == cur {
                return None;
            }
            cur = c.owner_id;
        }
        None
    }

    /// Which of our souls should act on `root` — the actor for the triggered
    /// path. **Ownership-agnostic** (per the recipe-permission split): a soul acts
    /// on any valid recipe in its reach, not only on cards it owns. Resolution:
    /// 1. a soul **present** (live) at the root's zone via its anchor (world
    ///    recipes — chopping a tree it doesn't own); else
    /// 2. a soul that **owns** the root through the container chain (inventory
    ///    recipes — assembling cards in its own pocket).
    /// `None` if no soul of ours can reach it. (Coarse pick — the NPC loop /
    /// Permissions will refine which soul + whether it's allowed.)
    fn actor_soul_for_root(&self, root: u32) -> Option<u32> {
        let now = self.clock_ms;
        let zone = self.world.cards.current(root, now)?.macro_zone;
        if let Some(s) =
            self.souls.keys().copied().find(|&s| self.zones.card_view_time(s, zone, now) == Some(now))
        {
            return Some(s);
        }
        self.owning_soul(root)
    }

    /// Fold a server-time sample into the [`ClockSync`] discipline and refresh
    /// the cached estimate. Emits `Clock` with the (behind-true-server) estimate.
    fn sample_clock(&mut self, server_micros: &str) -> Vec<Event> {
        match server_micros.parse::<u64>() {
            Ok(us) => {
                self.clock.note_sample(us as f64 / 1_000.0, self.perf_ms);
                self.clock_ms = self.clock.server_now_ms(self.perf_ms).max(0.0) as u64;
                vec![Event::Clock { ms: self.clock_ms }]
            }
            Err(_) => vec![],
        }
    }

    /// Parse a gate `time_drift:client_(ahead|behind)_by=N` reply for call `cid`
    /// and re-seat the clock at that call's **true send instant**. The gate judged
    /// the drift against the `client_time_ms` we stamped when the call was sent;
    /// the recorded `(send_perf, send_client)` for `cid` is exactly that stamp and
    /// its local time. The corrected server time was true at the gate's receive
    /// moment ≈ `send_perf + RTT/2`, i.e. the midpoint between send and this reply
    /// — that's the perf we associate with the re-seated capture. (Falls back to
    /// the current clock if the entry was already reaped.)
    fn note_drift(&mut self, cid: u32, error: &str) {
        let Some(rest) = error.strip_prefix("time_drift:client_") else { return };
        let (ahead, rest) = if let Some(r) = rest.strip_prefix("ahead_by=") {
            (true, r)
        } else if let Some(r) = rest.strip_prefix("behind_by=") {
            (false, r)
        } else {
            return;
        };
        let gap: f64 = rest.split(|c: char| !c.is_ascii_digit()).next().and_then(|s| s.parse().ok()).unwrap_or(0.0);
        let (send_perf, send_client) =
            self.inflight_calls.get(&cid).copied().unwrap_or((self.perf_ms, self.clock_ms));
        let midpoint_perf = (send_perf + self.perf_ms) / 2.0;
        self.clock.correct_from_drift(ahead, gap, send_client as f64, midpoint_perf);
        self.clock_ms = self.clock.server_now_ms(self.perf_ms).max(0.0) as u64;
    }

    fn apply_row(&mut self, table: &str, op: RowOp, row: serde_json::Value) -> Vec<Event> {
        match table {
            "cards" => match serde_json::from_value::<CardRow>(row) {
                Ok(mut card) => {
                    let card_id = card.card_id;
                    // Position reconciliation: when we hold a pending LOCAL move for
                    // this card (`dirty_position`), the server's row still carries the
                    // pre-move position. Keep OURS — macro_zone + micro_location + the
                    // placement bits of flags — and take only the server's holds/state.
                    // The exception is `pos_need`: the server REQUIRES this position (a
                    // spawn / authoritative relocation), so adopt it and drop the dirty
                    // mark. Recipe OUTPUT writes carry `pos_want` (advisory), not
                    // `pos_need`, so a card we drag-spliced mid-action keeps its
                    // predicted index until our own move flushes.
                    if matches!(op, RowOp::Insert | RowOp::Update)
                        && self.dirty_positions.contains(&card_id)
                    {
                        if resonantdust_codec::card_model::pos_need(card.flags) {
                            self.dirty_positions.remove(&card_id);
                        } else if let Some(local) = self.world.cards.current(card_id, self.clock_ms) {
                            let pmask = resonantdust_codec::card_model::placement_mask();
                            card.macro_zone = local.macro_zone;
                            card.micro_location = local.micro_location;
                            card.flags = (card.flags & !pmask) | (local.flags & pmask);
                        }
                    }
                    let macro_zone = card.macro_zone;
                    let promote_at = card.time_ms();
                    let ev = match op {
                        RowOp::Insert | RowOp::Update => Event::CardUpserted { card_id },
                        RowOp::Delete => Event::CardRemoved { card_id },
                    };
                    self.world.cards.apply(op, card);
                    // Warmth: a card update in this zone feeds the zone's Card-sub
                    // close-candidate counter (no-op unless it's a candidate).
                    self.zones.note_update(macro_zone, DataType::Card, self.clock_ms);
                    // Soul-discovery walk: a player_soul row → subscribe its soul
                    // roster; a soul row → anchor it + subscribe its inventory.
                    if matches!(op, RowOp::Insert | RowOp::Update) {
                        self.discover(card_id);
                        // Recursive inventory: any container we own that carries
                        // aspect.inventory gets its inventory zone (once).
                        self.ensure_owned_inventory(card_id);
                        // Triggered path: mark the changed card's chain root dirty;
                        // the budgeted match pass in `tick` re-evaluates it (and
                        // resolves the acting soul) ~1/s, collapsing row bursts.
                        if let Some(root) = self.chain_root(card_id) {
                            self.dirty_roots.insert(root);
                        }
                        // Future-stamped row: its promotion brings no new arrival,
                        // so schedule a dirty re-mark when the clock reaches it (the
                        // promotion kick — lifecycles / created cards re-evaluate).
                        if promote_at > self.clock_ms {
                            self.pending_promotions.push((promote_at, card_id));
                            // Movement pipeline: a future-stamped soul row may be
                            // a move step's arrival — if so, fire the next step now
                            // (the server gets this step's whole traversal to OK it).
                            self.advance_movement(card_id, promote_at);
                        }
                    }
                    vec![ev]
                }
                Err(e) => vec![Event::Error { error: format!("cards row decode: {e}") }],
            },
            "zones" => match serde_json::from_value::<crate::rows::ZoneRow>(row) {
                Ok(zone) => {
                    let macro_zone = zone.macro_zone;
                    self.world.zones.apply(op, zone);
                    // The row's arrival is the authoritative "zone materialized"
                    // signal for the region gate (clears any pending request).
                    match op {
                        RowOp::Insert | RowOp::Update => self.zones.note_zone_arrived(macro_zone),
                        RowOp::Delete => self.zones.note_zone_departed(macro_zone),
                    }
                    // Warmth: a zone-tile update feeds the Zone-sub candidate counter.
                    self.zones.note_update(macro_zone, DataType::Zone, self.clock_ms);
                    vec![Event::ZoneUpserted { macro_zone }]
                }
                Err(e) => vec![Event::Error { error: format!("zones row decode: {e}") }],
            },
            "regions" => match serde_json::from_value::<crate::rows::RegionRow>(row) {
                Ok(region) => {
                    // Feed the region gate. Insert/Update → latest presence bits
                    // (newly-present zones reconcile into live subs / requests);
                    // Delete → forget the region.
                    match op {
                        RowOp::Insert | RowOp::Update => self.zones.note_region(
                            region.macro_region,
                            region.zone_presence,
                            region.zone_available,
                            region.distance,
                        ),
                        RowOp::Delete => self.zones.note_region_removed(region.macro_region),
                    }
                    vec![]
                }
                Err(e) => vec![Event::Error { error: format!("regions row decode: {e}") }],
            },
            "players" => {
                // Learn our player_id from our own row (gate camelCases keys +
                // stringifies numbers, so `playerId` is a string).
                let row_name = row.get("name").and_then(|v| v.as_str());
                let pid = row
                    .get("playerId")
                    .and_then(|v| v.as_str())
                    .and_then(|s| s.parse::<u32>().ok());
                match (row_name, pid) {
                    (Some(rn), Some(pid))
                        if self.name.as_deref() == Some(rn) && self.player_id != Some(pid) =>
                    {
                        self.player_id = Some(pid);
                        // Kick the discovery walk: subscribe our player_soul
                        // roster. `owner_id` is overloaded — a `player_id` for a
                        // player_soul, a `card_id` for everything else — and the
                        // two id spaces OVERLAP (both start at 1024). A bare
                        // `owner_id = {pid}` therefore also matches another
                        // player's inventory whose container *card_id* equals our
                        // `pid` (a cross-player leak the multi-client harness
                        // caught). player_souls are identified by DEFINITION — the
                        // reserved range `packed_definition >= 0xFFF0` — so pin
                        // that; it selects exactly our player_soul(s), independent
                        // of placement (player_souls carry real positions now).
                        let sid = self.sid();
                        let player_soul_min = resonantdust_codec::packed::PLAYER_SOUL_PACKED_MIN;
                        self.pending_out.push(ClientMsg::Sub {
                            sid,
                            table: "cards".to_string(),
                            filter: Some(format!(
                                "owner_id = {pid} AND packed_definition >= {player_soul_min}"
                            )),
                        });
                        vec![Event::PlayerId { id: pid }]
                    }
                    _ => vec![],
                }
            }
            _ => vec![],
        }
    }
}

/// Does the recipe read or write the chain root, addressed as `slot.0.<offset>`
/// (branch 0)? A root-anchored recipe is matched with the candidate root placed
/// (no promotion); a branch recipe folds the root into a stack instead. The
/// path must *start* with `slot.0.` — a nested `…owner.slot.0.0` references some
/// other card's root, not the action root. Mirrors `RecipeMeta.root`.
fn recipe_references_root(recipe: &resonantdust_dsl::parser::Node) -> bool {
    use resonantdust_dsl::parser::{Stmt, Token};
    let starts_at_root = |p: &str| {
        let segs: Vec<&str> = p.split('.').collect();
        segs.len() >= 3 && segs[0] == "slot" && segs[1] == "0" && segs[2].parse::<u32>().is_ok()
    };
    for hook in ["input", "output"] {
        let Some(h) = recipe.hook(hook) else { continue };
        for stmt in &h.body {
            let Stmt::Instr(toks) = stmt else { continue };
            for tok in toks {
                if let Token::Slot(p) | Token::Value(p) = tok {
                    if starts_at_root(p) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// The aspect names a recipe **strictly requires** on a top-level slot: those
/// read by a canonical positive-threshold guard `*slot.<b>.<o>.aspect.<NAME> K
/// (ge|gt) … if …`. If such an aspect is absent (reads 0) the comparison is
/// false, the guarded line acquires no hold, and `match_recipe`'s conjunction
/// fails — so the recipe provably cannot match. Sound and conservative: lines
/// with no `if`, with an `or` (the read may not gate the verb), or using other
/// comparisons contribute nothing; nested-slot reads (`…owner.slot…aspect`) are
/// excluded (the card isn't in the root's stack). Used to pre-filter the index.
fn required_top_aspects(recipe: &resonantdust_dsl::parser::Node) -> Vec<String> {
    use resonantdust_dsl::parser::{Stmt, Token};
    let mut out: Vec<String> = Vec::new();
    let Some(h) = recipe.hook("input") else { return out };
    for stmt in &h.body {
        let Stmt::Instr(toks) = stmt else { continue };
        // Only a guarded line whose predicate has no disjunction is sound to index.
        let is_word = |t: &Token, w: &str| matches!(t, Token::Word(x) if x == w);
        if !toks.iter().any(|t| is_word(t, "if")) || toks.iter().any(|t| is_word(t, "or")) {
            continue;
        }
        // Scan for `*slot.B.O.aspect.NAME  K  (ge|gt)` triples (top-level only).
        for w in toks.windows(3) {
            let (Token::Value(path), Token::Number(k), Token::Word(op)) = (&w[0], &w[1], &w[2]) else {
                continue;
            };
            let segs: Vec<&str> = path.split('.').collect();
            let top_aspect = segs.len() == 5
                && segs[0] == "slot"
                && segs[3] == "aspect"
                && segs[1].parse::<u32>().is_ok()
                && segs[2].parse::<u32>().is_ok();
            let requires_positive = (op == "ge" && *k >= 1) || (op == "gt" && *k >= 0);
            if top_aspect && requires_positive && !out.iter().any(|a| a == segs[4]) {
                out.push(segs[4].to_string());
            }
        }
    }
    out
}

/// Bitmask of the top-level (`parent == ""`) branches a recipe references —
/// `RecipeMeta.branches`. Nested iterators don't contribute (they're reached
/// through their parent's chain).
fn top_branch_mask(iters: &[resonantdust_dsl::recipe::Iter]) -> u32 {
    iters.iter().filter(|it| it.parent.is_empty()).fold(0u32, |m, it| m | (1 << it.branch))
}

/// Per-`stack_id` window span = (max top-level offset)+1, indexed by stack_id —
/// `RecipeMeta.branch_spans`. Scans every `@input`/`@output` path's FIRST
/// `slot.<b>.<o>` triple (the top-level iterator; nested triples come later in the
/// path and don't widen a top-level window). A branch the recipe never references
/// stays 0 (not slid). Only stacks 2/3 are slid by the matcher, but spans are
/// computed for all so the field is self-describing.
fn top_branch_spans(recipe: &resonantdust_dsl::parser::Node) -> [u8; 4] {
    use resonantdust_dsl::parser::{Stmt, Token};
    let mut spans = [0u8; 4];
    for hook in ["input", "output"] {
        let Some(h) = recipe.hook(hook) else { continue };
        for stmt in &h.body {
            let Stmt::Instr(toks) = stmt else { continue };
            for tok in toks {
                if let Token::Slot(p) | Token::Value(p) = tok {
                    let segs: Vec<&str> = p.split('.').collect();
                    if segs.first() == Some(&"slot") && segs.len() >= 3 {
                        if let (Ok(b), Ok(o)) =
                            (segs[1].parse::<usize>(), segs[2].parse::<u8>())
                        {
                            if b < 4 {
                                spans[b] = spans[b].max(o.saturating_add(1));
                            }
                        }
                    }
                }
            }
        }
    }
    spans
}

/// The candidate window layouts to try for one matching pass: the branch lists
/// with the sliding stacks (2 top / 3 bottom) narrowed to each contiguous window
/// of the recipe's span. Stacks 0/1 (root/hex-tile) NEVER slide — they pass
/// through unchanged, so the synthetic-tile sentinel and root are preserved; a
/// stack-2/3 branch with span 0 (unreferenced) also passes through. Lowest offset
/// first → deterministic, first matching window wins. An empty result means a
/// needed stack is shorter than its span — the recipe can't match this pass.
fn window_views(branches: &[Vec<u32>; 4], spans: &[u8; 4]) -> Vec<[Vec<u32>; 4]> {
    let starts = |len: usize, span: u8| -> Vec<usize> {
        let s = span as usize;
        if s == 0 {
            vec![0] // not slid — the branch passes through unchanged
        } else if len < s {
            vec![] // window can't fit → no candidate
        } else {
            (0..=len - s).collect()
        }
    };
    let s2 = starts(branches[2].len(), spans[2]);
    let s3 = starts(branches[3].len(), spans[3]);
    let mut out = Vec::with_capacity(s2.len() * s3.len());
    for &k2 in &s2 {
        for &k3 in &s3 {
            let mut w = branches.clone();
            if spans[2] > 0 {
                w[2] = branches[2][k2..k2 + spans[2] as usize].to_vec();
            }
            if spans[3] > 0 {
                w[3] = branches[3][k3..k3 + spans[3] as usize].to_vec();
            }
            out.push(w);
        }
    }
    out
}

/// Promote a loose root into a stack: prepend it to that stack's card list (so
/// it acts as the chain's first member, `slot.<stack_id>.0`) and leave the root
/// slot empty. `branch` is the stack_id (2 top / 3 bottom). Mirrors `promoteRoot`
/// in recipeMatcher.ts.
fn promote_root(root: u32, base: &[Vec<u32>; 4], branch: usize) -> [Vec<u32>; 4] {
    let mut b = base.clone();
    let mut promoted = Vec::with_capacity(b[branch].len() + 1);
    promoted.push(root);
    promoted.extend_from_slice(&b[branch]);
    b[branch] = promoted;
    b
}

/// Whether a candidate's required anchors are all present in this pass — its
/// root requirement and every top-level stack it references (a synthetic tile
/// satisfies stack 1, the hex member). `branches`/`want_branches` are indexed by
/// stack_id. Mirrors `anchorsFit` in recipeMatcher.ts.
fn anchors_fit(
    uses_root: bool,
    want_branches: u32,
    pass_root: u32,
    branches: &[Vec<u32>; 4],
    has_synthetic: bool,
) -> bool {
    if uses_root && pass_root == 0 {
        return false;
    }
    let mut have = 0u32;
    for (i, b) in branches.iter().enumerate() {
        if !b.is_empty() {
            have |= 1 << i;
        }
    }
    if has_synthetic {
        have |= 1 << 1;
    }
    (want_branches & !have) == 0
}

/// The world hex `(q, r)` a card sits at, or `None` if it isn't on the world
/// surface. `macro_zone` carries the chunk coords (`unpack_macro_zone`, in
/// chunk units); the loose `micro` carries the local cell within the chunk —
/// so `hex = chunk * ZONE_SIZE + local`. (A stacked soul has no own cell; it
/// would inherit its root's — not a world-soul case, so treated as cell 0.)
fn world_hex(card: &CardRow) -> Option<(i32, i32)> {
    use resonantdust_codec::card_model::Micro;
    use resonantdust_codec::packed::{surface_of, unpack_macro_zone, world_tile, WORLD_LAYER};
    if surface_of(card.macro_zone) != WORLD_LAYER {
        return None;
    }
    let (cq, cr) = unpack_macro_zone(card.macro_zone);
    let (lq, lr) = match card.micro() {
        Micro::Loose { local_q, local_r, .. } => (local_q, local_r),
        Micro::Stacked { .. } => (0, 0),
    };
    Some((world_tile(cq, lq), world_tile(cr, lr)))
}

/// Axial-hex distance between two world cells (`(|dq| + |dq+dr| + |dr|) / 2`).
fn hex_distance(a: (i32, i32), b: (i32, i32)) -> i32 {
    let (dq, dr) = (a.0 - b.0, a.1 - b.1);
    (dq.abs() + (dq + dr).abs() + dr.abs()) / 2
}

/// The 6 axial-hex neighbors of `(q, r)`.
fn hex_neighbors(q: i32, r: i32) -> [(i32, i32); 6] {
    [(q + 1, r), (q - 1, r), (q, r + 1), (q, r - 1), (q + 1, r - 1), (q - 1, r + 1)]
}

/// Longest path the greedy walker will produce (a sanity cap).
const MAX_PATH_STEPS: usize = 64;

/// A greedy hex path from `start` (exclusive) to `target` (inclusive): each step
/// takes the neighbor that most reduces the hex distance to `target`. Assumes all
/// tiles are traversable (no obstacles yet), so greedy always converges; stops if
/// no neighbor improves (already adjacent-blocked) or the cap is hit.
fn hex_path(start: (i32, i32), target: (i32, i32)) -> Vec<(i32, i32)> {
    let mut path = Vec::new();
    let mut cur = start;
    while cur != target && path.len() < MAX_PATH_STEPS {
        let best = hex_neighbors(cur.0, cur.1)
            .into_iter()
            .min_by_key(|n| hex_distance(*n, target))
            .unwrap();
        if hex_distance(best, target) >= hex_distance(cur, target) {
            break;
        }
        path.push(best);
        cur = best;
    }
    path
}

/// Map a [`stack::Placement`] to the wire `place_card` `placement` arg (the
/// shard's flat `Placement` struct: snake_case keys; `xy` packs `(x, y)`). Unused
/// by the commit-based move path (positions sync via `move_cards`); retained for
/// the genuine-sync placements (equip / `move_soul`) that still use `place_card`.
#[allow(dead_code)]
fn placement_json(p: &stack::Placement) -> serde_json::Value {
    match *p {
        stack::Placement::Stack { parent_id, direction } => serde_json::json!({
            "kind": 0, "parent_id": parent_id, "direction": direction,
            "surface": 0, "macro_zone": 0, "q": 0, "r": 0, "xy": 0,
        }),
        stack::Placement::Loose { surface, macro_zone, q, r, x, y } => serde_json::json!({
            "kind": 1, "parent_id": 0, "direction": 0,
            "surface": surface, "macro_zone": macro_zone, "q": q, "r": r,
            "xy": ((x as u16 as u32) << 16) | (y as u16 as u32),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resonantdust_codec::packed::pack_valid_at;

    /// A `cards` Row frame in the exact wire shape (camelCase keys, every number
    /// stringified) the gate emits.
    fn cards_row_frame(op: RowOp, card_id: u32, owner_id: u32, time_ms: u64) -> GateMsg {
        GateMsg::Row {
            sid: 0,
            table: "cards".to_string(),
            op,
            old: None,
            row: serde_json::json!({
                "validAt": pack_valid_at(time_ms, 1).to_string(),
                "cardId": card_id.to_string(),
                "macroZone": "0",
                "microLocation": "0",
                "ownerId": owner_id.to_string(),
                "packedDefinition": "0",
                "flags": "0",
                "flagsBk": "0",
                "stock": "0",
            }),
        }
    }

    #[test]
    fn login_dispatches_players_watch_before_call_with_monotonic_ids() {
        let mut c = Client::new();
        let frames = c.dispatch(Command::Login { name: "ann".to_string() });
        assert_eq!(frames.len(), 2);
        // The `players` subscription MUST come first so the gate's login read sees
        // the row (the gate handles messages serially + blocks on the read).
        match &frames[0] {
            ClientMsg::Sub { sid, table, filter } => {
                assert_eq!((*sid, table.as_str()), (1, "players"));
                assert_eq!(filter.as_deref(), Some("name = 'ann'"));
            }
            other => panic!("expected sub, got {other:?}"),
        }
        match &frames[1] {
            ClientMsg::Call { cid, reducer, args } => {
                assert_eq!((*cid, reducer.as_str()), (1, "claim_or_login"));
                assert_eq!(args["name"], "ann");
            }
            other => panic!("expected call, got {other:?}"),
        }
        // ids count up independently across subsequent dispatches.
        let more = c.dispatch(Command::Subscribe { table: "cards".to_string(), filter: None });
        assert!(matches!(&more[0], ClientMsg::Sub { sid: 2, .. }));
        let call = c.dispatch(Command::Call { reducer: "ping".to_string(), args: serde_json::json!({}) });
        assert!(matches!(&call[0], ClientMsg::Call { cid: 2, .. }));
    }

    #[test]
    fn learns_player_id_then_creates_player_soul_owned_by_it() {
        let mut c = Client::new();
        c.dispatch(Command::Login { name: "ann".to_string() });
        // A players row for someone else is ignored.
        let other = GateMsg::Row {
            sid: 0,
            table: "players".to_string(),
            op: RowOp::Insert,
            old: None,
            row: serde_json::json!({ "playerId": "999", "name": "bob" }),
        };
        assert_eq!(c.apply(other), vec![]);
        assert_eq!(c.player_id(), None);
        // Our row sets player_id + emits the event.
        let ours = GateMsg::Row {
            sid: 0,
            table: "players".to_string(),
            op: RowOp::Insert,
            old: None,
            row: serde_json::json!({ "playerId": "1024", "name": "ann" }),
        };
        assert_eq!(c.apply(ours), vec![Event::PlayerId { id: 1024 }]);
        assert_eq!(c.player_id(), Some(1024));
        // The explicit chain mints the player_soul as a card OWNED BY the learned
        // player_id (surface 0) — identified by its definition, no flag.
        let frames = c.dispatch(Command::CreateCard {
            owner: c.player_id().unwrap(),
            card_key: "player_soul".to_string(),
            surface: 0,
            macro_zone: 0,
            q: 0,
            r: 0,
        });
        match &frames[0] {
            ClientMsg::Call { reducer, args, .. } => {
                assert_eq!(reducer, "create_card");
                assert_eq!(args["owner_id"], 1024);
                assert_eq!(args["card_key"], "player_soul");
            }
            other => panic!("expected call, got {other:?}"),
        }
    }

    #[test]
    fn clock_sample_emits_event_and_stamps_outbound() {
        let mut c = Client::new();
        // A server-time sample emits a Clock event. The disciplined estimate runs
        // behind true server time by `client_delay` (the discipline is unit-tested
        // in clock.rs); here we only assert an event fires.
        assert!(matches!(
            c.apply(GateMsg::Time { server_micros: "5000000".to_string() }).as_slice(),
            [Event::Clock { .. }]
        ));
        // Outbound calls stamp `client_time_ms` from the cached estimate. Set it
        // directly for a determinate assertion (the sampling path is exercised
        // above + in clock.rs).
        c.clock_ms = 5000;
        let spawn = c.dispatch(Command::CreateCard {
            owner: 1,
            card_key: "player_soul".to_string(),
            surface: 0,
            macro_zone: 0,
            q: 0,
            r: 0,
        });
        match &spawn[0] {
            ClientMsg::Call { args, .. } => assert_eq!(args["client_time_ms"], 5000),
            other => panic!("expected call, got {other:?}"),
        }
    }

    #[test]
    fn cards_rows_fold_into_the_world_with_events() {
        let mut c = Client::new();
        assert_eq!(c.apply(cards_row_frame(RowOp::Insert, 1024, 7, 100)),
                   vec![Event::CardUpserted { card_id: 1024 }]);
        assert_eq!(c.apply(cards_row_frame(RowOp::Insert, 1025, 7, 100)),
                   vec![Event::CardUpserted { card_id: 1025 }]);
        // World reflects them at a now past their stamp.
        assert_eq!(c.world().cards.current_all(150).count(), 2);
        assert_eq!(c.world().cards.current(1024, 150).unwrap().owner_id, 7);
        // Delete reaps + emits.
        assert_eq!(c.apply(cards_row_frame(RowOp::Delete, 1025, 7, 100)),
                   vec![Event::CardRemoved { card_id: 1025 }]);
        assert_eq!(c.world().cards.current_all(150).count(), 1);
    }

    #[test]
    fn place_validates_and_emits_place_card_frame_without_mutating_world() {
        use resonantdust_codec::card_model::{micro_is_card, Micro};
        use resonantdust_codec::packed::{is_player_soul, with_surface, STACK_DIR_UP, WORLD_LAYER};

        fn row(card_id: u32, owner_id: u32, packed_definition: u16) -> CardRow {
            let (micro_location, flags) = Micro::snap(0, 0).apply(0);
            CardRow {
                valid_at: pack_valid_at(100, 1),
                card_id,
                macro_zone: with_surface(0, WORLD_LAYER),
                micro_location,
                owner_id,
                packed_definition,
                flags,
                flags_bk: 0,
                stock: 0,
            }
        }

        let mut c = Client::new();
        c.player_id = Some(7);
        c.clock_ms = 1000;
        // player_soul (def 0xFFFF, owner = player 7) — the owner-walk terminus;
        // two loose cards owned by it. owning_player(card) → 7 via the soul.
        c.world.cards.apply(RowOp::Insert, row(1024, 7, 0xFFFF));
        assert!(is_player_soul(c.world.cards.current(1024, 1000).unwrap().packed_definition));
        c.world.cards.apply(RowOp::Insert, row(2000, 1024, 0));
        c.world.cards.apply(RowOp::Insert, row(2001, 1024, 0));

        // Stack 2000 onto 2001 — commit-based: applies to the LOCAL world and
        // marks dirty, emits NO frame (nothing on the wire until a commit).
        let frames = c
            .place(2000, stack::Placement::Stack { parent_id: 2001, direction: STACK_DIR_UP })
            .unwrap();
        assert!(frames.is_empty(), "place is local — no immediate wire frame");
        // The local world reflects the move: 2000 is now a stack member of 2001.
        let r = c.world.cards.current(2000, 1000).unwrap();
        assert!(micro_is_card(r.flags), "place applies the move to the local world");
        assert!(matches!(Micro::of(r.micro_location, r.flags), Micro::Stacked { root: 2001, .. }));
        assert!(c.dirty_positions.contains(&2000), "moved card is dirty until flushed");

        // Flushing the dirty set emits ONE batched move_cards call, then clears.
        let flush = c.flush_positions();
        match &flush[0] {
            ClientMsg::Call { reducer, args, .. } => {
                assert_eq!(reducer, "move_cards");
                assert_eq!(args["card_ids"][0], 2000);
                assert_eq!(args["caller_player_id"], 7);
            }
            other => panic!("expected move_cards call, got {other:?}"),
        }
        assert!(c.dirty_positions.is_empty(), "flush clears the dirty set");

        // A self-stack is rejected (feasibility check fails → Err, no local apply).
        assert!(c.place(2001, stack::Placement::Stack { parent_id: 2001, direction: STACK_DIR_UP }).is_err());
    }

    #[test]
    fn place_syncs_in_shared_zone_stays_local_when_private() {
        use resonantdust_codec::card_model::Micro;
        use resonantdust_codec::packed::{with_surface, STACK_DIR_UP, WORLD_LAYER};

        fn loose_row(card_id: u32, owner_id: u32, macro_zone: u64) -> CardRow {
            let (micro_location, flags) = Micro::snap(0, 0).apply(0);
            CardRow {
                valid_at: pack_valid_at(100, 1),
                card_id,
                macro_zone,
                micro_location,
                owner_id,
                packed_definition: 0,
                flags,
                flags_bk: 0,
                stock: 0,
            }
        }

        let world = with_surface(0, WORLD_LAYER); // a shared world zone (0,0)
        let mut c = Client::new();
        c.player_id = Some(7);
        c.clock_ms = 1000;
        // A player_soul terminus so owning_player resolves; two loose world cards.
        c.world.cards.apply(RowOp::Insert, {
            let mut r = loose_row(1024, 7, world);
            r.packed_definition = 0xFFFF;
            r
        });
        c.world.cards.apply(RowOp::Insert, loose_row(3000, 1024, world));
        c.world.cards.apply(RowOp::Insert, loose_row(3001, 1024, world));

        // PRIVATE (no observers reported): a move stays local — no wire frame.
        c.place(3000, stack::Placement::Stack { parent_id: 3001, direction: STACK_DIR_UP })
            .unwrap();
        assert!(c.dirty_positions.contains(&3000));
        assert!(c.pending_out.is_empty(), "private-zone move stays local");

        // The gate reports the world zone is SHARED (2 observers).
        c.apply(GateMsg::ZoneObservers { macro_zone: world.to_string(), observers: 2 });
        assert_eq!(c.zone_observers.get(&world).copied(), Some(2));
        // The earlier dirty card 3000 is in that zone → the ≤1→>1 flush syncs it.
        assert!(matches!(c.pending_out.first(), Some(ClientMsg::Call { reducer, .. }) if reducer == "move_cards"));
        c.pending_out.clear();

        // A NEW move in the now-shared zone syncs immediately.
        c.place(3001, stack::Placement::Loose { surface: WORLD_LAYER, macro_zone: world, q: 1, r: 1, x: 0, y: 0 })
            .unwrap();
        assert!(
            matches!(c.pending_out.first(), Some(ClientMsg::Call { reducer, .. }) if reducer == "move_cards"),
            "move in a shared zone syncs immediately"
        );
    }

    /// A `regions` Row frame in the gate's wire shape (camelCase, stringified).
    fn region_row_frame(op: RowOp, macro_region: u64, presence: u64, available: u64) -> GateMsg {
        GateMsg::Row {
            sid: 0,
            table: "regions".to_string(),
            op,
            old: None,
            row: serde_json::json!({
                "validAt": pack_valid_at(100, 1).to_string(),
                "macroRegion": macro_region.to_string(),
                "zonePresence": presence.to_string(),
                "zoneAvailable": available.to_string(),
            }),
        }
    }

    fn count_sub<'a>(frames: &'a [ClientMsg], table: &str) -> usize {
        frames.iter().filter(|f| matches!(f, ClientMsg::Sub { table: t, .. } if t == table)).count()
    }
    fn count_call<'a>(frames: &'a [ClientMsg], reducer: &str) -> usize {
        frames.iter().filter(|f| matches!(f, ClientMsg::Call { reducer: r, .. } if r == reducer)).count()
    }

    #[test]
    fn set_anchor_scopes_subscriptions_and_gates_region() {
        use resonantdust_codec::packed::WORLD_LAYER;
        let mut c = Client::new();
        // Region-interior anchor (chunk 2,2): the 3×3 active ring stays in region 0.
        let frames = c.dispatch(Command::SetAnchor {
            name: "soul".to_string(),
            surface: WORLD_LAYER,
            owner: 0,
            q: 16,
            r: 16,
            // active radius 8 from the chunk-(2,2) corner spans chunks 1..=3 each
            // axis → the same 9 zones in region 0 the old 3×3 ring produced.
            radii: crate::zones::AnchorRadii { active: 8, hot: 0, warm: 0, cold: 0 },
        });
        // 9 active zones, each Full = zones + cards scoped to its macro_zone.
        assert_eq!(count_sub(&frames, "zones"), 9, "one scoped zones sub per active zone");
        assert_eq!(count_sub(&frames, "cards"), 9, "one scoped cards sub per active zone");
        // All 9 share one region → one region sub + one ensure_region (region unknown).
        assert_eq!(count_sub(&frames, "regions"), 1);
        assert_eq!(count_call(&frames, "ensure_region"), 1);
        // No request_zone yet — the region isn't mirrored.
        assert_eq!(count_call(&frames, "request_zone"), 0);
        // The scoped filters are macro_zone equality.
        assert!(frames.iter().any(|f| matches!(f,
            ClientMsg::Sub { table, filter: Some(fl), .. }
                if table == "cards" && fl.starts_with("macro_zone = "))));
    }

    #[test]
    fn region_row_arrival_requests_each_active_zone_once() {
        use resonantdust_codec::packed::{pack_macro_zone_full, region_of_zone, WORLD_LAYER};
        let mut c = Client::new();
        c.dispatch(Command::SetAnchor {
            name: "soul".to_string(),
            surface: WORLD_LAYER,
            owner: 0,
            q: 16,
            r: 16,
            // active radius 8 from the chunk-(2,2) corner spans chunks 1..=3 each
            // axis → the same 9 zones in region 0 the old 3×3 ring produced.
            radii: crate::zones::AnchorRadii { active: 8, hot: 0, warm: 0, cold: 0 },
        });
        let (region, _) = region_of_zone(pack_macro_zone_full(0, WORLD_LAYER, 2, 2));

        // The region declares all its zones present. apply() folds it (no event);
        // the follow-up frames carry the spawn requests.
        assert_eq!(c.apply(region_row_frame(RowOp::Insert, region, u64::MAX, 0)), vec![]);
        let frames = c.zone_frames();
        assert_eq!(count_call(&frames, "request_zone"), 9, "each active zone requested once");

        // A second identical region update does not re-request (requested set).
        c.apply(region_row_frame(RowOp::Update, region, u64::MAX, 0));
        assert_eq!(count_call(&c.zone_frames(), "request_zone"), 0, "no duplicate requests");
    }

    #[test]
    fn zone_arrival_clears_the_pending_request() {
        use resonantdust_codec::packed::{pack_macro_zone_full, region_of_zone, WORLD_LAYER};
        let mut c = Client::new();
        c.dispatch(Command::SetAnchor {
            name: "soul".to_string(),
            surface: WORLD_LAYER,
            owner: 0,
            q: 16,
            r: 16,
            // active radius 8 from the chunk-(2,2) corner spans chunks 1..=3 each
            // axis → the same 9 zones in region 0 the old 3×3 ring produced.
            radii: crate::zones::AnchorRadii { active: 8, hot: 0, warm: 0, cold: 0 },
        });
        let target = pack_macro_zone_full(0, WORLD_LAYER, 2, 2);
        let (region, _) = region_of_zone(target);
        c.apply(region_row_frame(RowOp::Insert, region, u64::MAX, 0));
        let _ = c.zone_frames(); // drains the 9 requests

        // The target zone's row lands → note_zone_arrived. A later region update
        // must not re-request it (already arrived), even though `requested` was
        // cleared on arrival.
        let zrow = GateMsg::Row {
            sid: 0,
            table: "zones".to_string(),
            op: RowOp::Insert,
            old: None,
            row: serde_json::json!({
                "validAt": pack_valid_at(100, 1).to_string(),
                "zoneId": "1",
                "macroZone": target.to_string(),
                "packedDefinition": "0",
                "ownerId": "0",
                "t0": "0", "t1": "0", "t2": "0", "t3": "0", "t4": "0", "t5": "0",
                "t6": "0", "t7": "0", "t8": "0", "t9": "0", "t10": "0", "t11": "0",
                "t12": "0", "t13": "0", "t14": "0", "t15": "0",
            }),
        };
        assert_eq!(c.apply(zrow), vec![Event::ZoneUpserted { macro_zone: target }]);
        let _ = c.zone_frames();
        c.apply(region_row_frame(RowOp::Update, region, u64::MAX, u64::MAX));
        let frames = c.zone_frames();
        assert!(
            !frames.iter().any(|f| matches!(f,
                ClientMsg::Call { reducer, args, .. }
                    if reducer == "request_zone" && args["macro_zone"] == target)),
            "an arrived zone is not re-requested"
        );
    }

    /// A `cards` Row frame in wire shape with a specific owner / macro_zone /
    /// packed_definition (placement is a world/inventory snap via the codec).
    fn card_frame_at(card_id: u32, owner_id: u32, macro_zone: u64, packed_definition: u16) -> GateMsg {
        use resonantdust_codec::card_model::Micro;
        let (micro_location, flags) = Micro::snap(0, 0).apply(0);
        GateMsg::Row {
            sid: 0,
            table: "cards".to_string(),
            op: RowOp::Insert,
            old: None,
            row: serde_json::json!({
                "validAt": pack_valid_at(100, 1).to_string(),
                "cardId": card_id.to_string(),
                "macroZone": macro_zone.to_string(),
                "microLocation": micro_location.to_string(),
                "ownerId": owner_id.to_string(),
                "packedDefinition": packed_definition.to_string(),
                "flags": flags.to_string(),
                "flagsBk": "0",
                "stock": "0",
            }),
        }
    }

    #[test]
    fn discovery_walks_player_to_soul_to_inventory() {
        use resonantdust_codec::packed::{
            pack_macro_zone_full, INVENTORY_LAYER, PLAYER_SOUL_PACKED, WORLD_LAYER,
        };

        let mut c = Client::new();
        c.dispatch(Command::Login { name: "ann".to_string() });
        // Set the clock directly so `discover` sees the t=100 rows as current
        // (the disciplined clock runs behind, so a small sampled value would go
        // to 0; the discipline is tested in clock.rs).
        c.clock_ms = 1000;
        // Learn our player_id → queues the player_soul roster sub.
        c.apply(GateMsg::Row {
            sid: 0,
            table: "players".to_string(),
            op: RowOp::Insert,
            old: None,
            row: serde_json::json!({ "playerId": "1000", "name": "ann" }),
        });
        let frames = c.drain_outbound();
        assert!(
            frames.iter().any(|f| matches!(f,
                ClientMsg::Sub { table, filter: Some(fl), .. }
                    if table == "cards" && fl == "owner_id = 1000 AND packed_definition >= 65520")),
            "learning player_id subscribes the player_soul roster (scoped to the reserved \
             player-soul def range so a colliding card_id can't leak another player's inventory)"
        );

        // Our player_soul (owner_id == player_id, def 0xFFFF, surface 0) →
        // subscribe the souls it owns.
        let psoul = pack_macro_zone_full(0, /* PLAYER_SOUL_SURFACE */ 0, 0, 0);
        c.apply(card_frame_at(5000, 1000, psoul, PLAYER_SOUL_PACKED));
        let frames = c.drain_outbound();
        assert!(
            frames.iter().any(|f| matches!(f,
                ClientMsg::Sub { table, filter: Some(fl), .. }
                    if table == "cards" && fl == "owner_id = 5000")),
            "the player_soul row subscribes its soul roster"
        );
        assert!(c.souls().next().is_none(), "the player_soul is not itself a soul");

        // A soul owned by that player_soul, standing at world (0,0) → anchored +
        // its inventory container subscribed.
        let world00 = pack_macro_zone_full(0, WORLD_LAYER, 0, 0);
        c.apply(card_frame_at(6000, 5000, world00, 0));
        assert_eq!(c.souls().collect::<Vec<_>>(), vec![6000], "the soul is discovered");
        let frames = c.drain_outbound();
        let inv = pack_macro_zone_full(6000, INVENTORY_LAYER, 0, 0);
        assert!(
            frames.iter().any(|f| matches!(f,
                ClientMsg::Sub { table, filter: Some(fl), .. }
                    if table == "cards" && *fl == format!("macro_zone = {inv}"))),
            "the soul's inventory container is subscribed (cards scoped to it)"
        );
        // The inventory container is active (ownership-driven, no bundle needed).
        // The soul's *world* anchor radii come from its def's `anchor_*` aspects,
        // which need a loaded content bundle — absent here, so the world ring is
        // empty and only the inventory is active. (The range-disk world ring is
        // covered by the zones.rs unit tests with explicit radii.)
        assert!(c.zones().active_zones().any(|z| z == inv), "inventory zone is active");
        assert_eq!(c.zones().active_zones().count(), 1, "inventory only (no bundle → zero anchor radii)");
    }

    #[test]
    fn matcher_helpers_mirror_recipematcher_ts() {
        use resonantdust_dsl::parser::parse;
        use resonantdust_dsl::recipe::iterators;

        // promote_root prepends the root into a stack (by stack_id) and leaves
        // the root slot empty.
        let base = [vec![], vec![], vec![10, 11], vec![]];
        let p1 = promote_root(99, &base, 2);
        assert_eq!(p1[2], vec![99, 10, 11]);
        let p2 = promote_root(99, &base, 3);
        assert_eq!(p2[3], vec![99]);
        assert_eq!(p2[2], vec![10, 11]);

        // anchors_fit: root requirement + stack mask (synthetic satisfies stack 1).
        assert!(anchors_fit(false, 0b10, 0, &[vec![], vec![1], vec![], vec![]], false));
        assert!(!anchors_fit(true, 0, 0, &[vec![], vec![], vec![], vec![]], false), "root needed but absent");
        assert!(anchors_fit(true, 0, 5, &[vec![], vec![], vec![], vec![]], false), "root present");
        assert!(anchors_fit(false, 0b10, 0, &[vec![], vec![], vec![], vec![]], true), "synthetic gives stack 1");
        assert!(!anchors_fit(false, 0b100, 0, &[vec![], vec![], vec![], vec![]], false), "stack 2 missing");

        // recipe_references_root + top_branch_mask over parsed recipes.
        let root_src = "<recipe>\n  ::r>\n    @input>\n      *slot.0.0.aspect.x 1 ge if &slot.0.0 use\n";
        let rn = parse(root_src).unwrap();
        let rr = rn.bucket("recipe").unwrap().def("r").unwrap();
        assert!(recipe_references_root(rr));
        assert_eq!(top_branch_mask(&iterators(rr)), 0, "root-only recipe has no branches");

        let cut_src = "<recipe>\n  ::c>\n    @input>\n\
            \x20     *slot.1.0.aspect.wood 1 ge if &slot.1.0 use\n\
            \x20     *slot.2.0.aspect.corpus_lit 1 ge if &slot.2.0 claim\n\
            \x20     $card::axe *slot.2.0.owner.slot.2.0.def_id eq if &slot.2.0.owner.slot.2.0 share\n";
        let cn = parse(cut_src).unwrap();
        let cr = cn.bucket("recipe").unwrap().def("c").unwrap();
        assert!(!recipe_references_root(cr), "cut_tree is tile/stack-rooted, not root-anchored");
        // top-level stacks 1 (tile) + 2 (actor); the nested axe iterator doesn't count.
        assert_eq!(top_branch_mask(&iterators(cr)), 0b110);
    }

    #[test]
    fn top_branch_spans_counts_max_top_offset_plus_one() {
        use resonantdust_dsl::parser::parse;
        // slot.2.0 + slot.2.1 → span 2; the nested slot.2.0.owner.slot.2.0's first
        // triple is also slot.2.0 (top-level), not a wider window.
        let src = "<recipe>\n  ::r>\n    @input>\n\
            \x20     $card::corpus *slot.2.0.def_id eq if &slot.2.0 use\n\
            \x20     $card::corpus *slot.2.1.def_id eq if &slot.2.1 claim\n\
            \x20     $card::axe *slot.2.0.owner.slot.2.0.def_id eq if &slot.2.0.owner.slot.2.0 share\n";
        let n = parse(src).unwrap();
        let r = n.bucket("recipe").unwrap().def("r").unwrap();
        assert_eq!(top_branch_spans(r), [0, 0, 2, 0]);

        // a root + single-dust recipe (corpus_dust shape) → top span 1.
        let dust = "<recipe>\n  ::d>\n    @input>\n\
            \x20     $card::corpus *slot.0.0.def_id eq if &slot.0.0 use\n\
            \x20     $card::dust *slot.2.0.def_id eq if &slot.2.0 claim\n";
        let dn = parse(dust).unwrap();
        let dr = dn.bucket("recipe").unwrap().def("d").unwrap();
        assert_eq!(top_branch_spans(dr), [1, 0, 1, 0]);
    }

    #[test]
    fn window_views_slides_top_and_bottom_only() {
        // span-2 window over the top stack [10,11,12,13] → 3 adjacent windows;
        // root(0)/hex(1)/bottom(3) pass through unchanged.
        let branches = [vec![1], vec![2], vec![10, 11, 12, 13], vec![9]];
        let views = window_views(&branches, &[0, 0, 2, 0]);
        let tops: Vec<Vec<u32>> = views.iter().map(|w| w[2].clone()).collect();
        assert_eq!(tops, vec![vec![10, 11], vec![11, 12], vec![12, 13]]);
        for w in &views {
            assert_eq!(w[0], vec![1]);
            assert_eq!(w[1], vec![2]);
            assert_eq!(w[3], vec![9]); // span-0 bottom unchanged
        }
    }

    #[test]
    fn window_views_span_zero_passthrough_and_too_short_empty() {
        // span 0 everywhere → one passthrough view.
        let b = [vec![], vec![], vec![10, 11], vec![20]];
        let v = window_views(&b, &[0, 0, 0, 0]);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0][2], vec![10, 11]);
        // span 3 over a 2-card top → no candidate window.
        assert!(window_views(&b, &[0, 0, 3, 0]).is_empty());
    }

    #[test]
    fn call_replies_and_protocol_errors_surface() {
        let mut c = Client::new();
        assert_eq!(c.apply(GateMsg::CallOk { cid: 3, server_micros: "0".to_string() }),
                   vec![Event::Clock { ms: 0 }, Event::CallOk { cid: 3 }]);
        assert_eq!(c.apply(GateMsg::CallErr { cid: 4, error: "nope".to_string(), server_micros: "".to_string() }),
                   vec![Event::CallErr { cid: 4, error: "nope".to_string() }]);
        assert_eq!(c.apply(GateMsg::Error { error: "bad".to_string() }),
                   vec![Event::Error { error: "bad".to_string() }]);
    }
}
