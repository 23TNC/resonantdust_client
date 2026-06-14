//! Multi-client test sessions.
//!
//! A [`Session`] bundles one gate connection + one sans-IO [`Client`] core + a
//! login name — i.e. **one player's client**. The harness opens several and
//! drives them *together* ([`pump_all`]) so each receives the others' broadcast
//! rows; that's how we exercise multiplayer (cross-client visibility, contended
//! actions, …). Each session is otherwise a thin IO wrapper over the core's
//! `dispatch`/`apply`, with the run-loop the binary needs.
//!
//! The pump primitive is [`Session::drain_step`]: tick the clock + action queue,
//! drain whatever inbound frames are ready (bounded, cancel-safe `timeout` over
//! the framed socket), flush outbound. Single-session waits ([`pump_until`] /
//! [`pump_for_state`]) and the multiplexed [`pump_all`] are all built on it, so
//! no session starves another while it blocks on its own socket.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use anyhow::bail;
use resonantdust_codec::card_model::Micro;
use resonantdust_codec::packed::{pack_macro_zone_full, INVENTORY_LAYER, STACK_DIR_UP, WORLD_LAYER};
use resonantdust_state::recipe_state::owning_player;
use resonantdust_state::stack::Placement;

use crate::client::{Client, Command, Event};
use crate::content;
use crate::gate::GateConnection;
use crate::transport::WsTransport;

/// Surface band the player_soul lives on (`0`, never rendered — it *is* the
/// player). World souls live on [`WORLD_LAYER`], inventories on [`INVENTORY_LAYER`].
pub const PLAYER_SOUL_SURFACE: u8 = 0;

/// Monotonic ms since first call — the shared `perf_ms` timeline the clock
/// discipline extrapolates between samples. ONE timeline across every session.
pub fn perf_ms() -> f64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// One player's client: a gate connection + a sans-IO core + the login name.
pub struct Session {
    pub name: String,
    pub core: Client,
    conn: GateConnection<WsTransport>,
}

impl Session {
    /// Connect to `url`, log in as `name` (creating the player if new), and load
    /// the content bundle from `http_base`. Ready to seed/act afterward.
    pub async fn start(url: &str, http_base: &str, name: &str) -> anyhow::Result<Session> {
        let conn = GateConnection::new(WsTransport::connect(url).await?);
        let mut s = Session { name: name.to_string(), core: Client::new(), conn };
        s.send(Command::Login { name: name.to_string() }).await?;
        if !s.pump_until(Duration::from_secs(5), |e| matches!(e, Event::PlayerId { .. })).await? {
            bail!("{name}: never learned player_id");
        }
        let bundle = content::fetch_bundle(http_base).await?;
        s.core.set_bundle(bundle);
        Ok(s)
    }

    pub fn player_id(&self) -> Option<u32> {
        self.core.player_id()
    }

    /// Dispatch a command and flush its frames.
    pub async fn send(&mut self, cmd: Command) -> anyhow::Result<()> {
        let frames = self.core.dispatch(cmd);
        self.conn.send_all(&frames).await
    }

    /// One pump step: tick the clock + action queue, drain whatever inbound
    /// frames are ready (bounded ~20ms; a per-recv `timeout` keeps it from
    /// blocking on a quiet socket — cancel-safe over the framed stream), flush
    /// outbound. `on_event` observes each folded event; `Event::Error` is logged.
    /// Returns `true` at EOF (socket closed). The single multiplexing primitive.
    async fn drain_step(&mut self, mut on_event: impl FnMut(&Event)) -> anyhow::Result<bool> {
        self.core.tick(perf_ms());
        self.conn.send_all(&self.core.drain_outbound()).await?;
        let deadline = Instant::now() + Duration::from_millis(20);
        let mut eof = false;
        while Instant::now() < deadline {
            match tokio::time::timeout(Duration::from_millis(5), self.conn.next()).await {
                Ok(Ok(Some(msg))) => {
                    for ev in self.core.apply(msg) {
                        if let Event::Error { error } = &ev {
                            eprintln!("{}: {error}", self.name);
                        }
                        on_event(&ev);
                    }
                    self.conn.send_all(&self.core.drain_outbound()).await?;
                }
                Ok(Ok(None)) => {
                    eof = true;
                    break;
                }
                Ok(Err(e)) => return Err(e),
                Err(_) => break, // nothing ready right now
            }
        }
        self.core.tick(perf_ms());
        self.conn.send_all(&self.core.drain_outbound()).await?;
        Ok(eof)
    }

    /// Pump this session until an event matches `done` or `budget` elapses.
    pub async fn pump_until(
        &mut self,
        budget: Duration,
        mut done: impl FnMut(&Event) -> bool,
    ) -> anyhow::Result<bool> {
        let deadline = Instant::now() + budget;
        let mut hit = false;
        while Instant::now() < deadline && !hit {
            let eof = self.drain_step(|e| {
                if done(e) {
                    hit = true;
                }
            }).await?;
            if eof {
                break;
            }
            if !hit {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        Ok(hit)
    }

    /// Pump until a predicate over the core state holds or `total` elapses.
    pub async fn pump_for_state(
        &mut self,
        total: Duration,
        ready: impl Fn(&Client) -> bool,
    ) -> anyhow::Result<bool> {
        let deadline = Instant::now() + total;
        while Instant::now() < deadline {
            if ready(&self.core) {
                return Ok(true);
            }
            if self.drain_step(|_| {}).await? {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        Ok(ready(&self.core))
    }

    /// Pump idle for `budget` — a settle / quiet wait.
    pub async fn pump(&mut self, budget: Duration) -> anyhow::Result<()> {
        self.pump_until(budget, |_| false).await?;
        Ok(())
    }

    /// Seed a character via the generic create-card chain: create the
    /// `player_soul` (only if this session has none yet), a world soul
    /// (`soul_key`, e.g. `"human"`) owned by it, and the `loadout` cards in that
    /// soul's inventory. Returns the world soul's `card_id`. Call repeatedly to
    /// give a player multiple characters. Each layer is created by id and
    /// discovered before the next — the cumbersome-but-generic chain.
    pub async fn seed_soul(
        &mut self,
        soul_key: &str,
        loadout: &[&str],
        at: Option<(i32, i32)>,
    ) -> anyhow::Result<u32> {
        let player_id = self.player_id().expect("logged in");

        // 1. player_soul (once per session) — a card owned by our player_id.
        if self.core.player_souls().next().is_none() {
            self.send(Command::CreateCard {
                owner: player_id,
                card_key: "player_soul".to_string(),
                surface: PLAYER_SOUL_SURFACE,
                macro_zone: 0,
                q: 0,
                r: 0,
            })
            .await?;
            self.await_call().await?;
            if !self
                .pump_for_state(Duration::from_secs(5), |c| c.player_souls().next().is_some())
                .await?
            {
                bail!("{}: player_soul never discovered", self.name);
            }
        }
        let player_soul = self.core.player_souls().next().unwrap();

        // 2. world soul owned by the player_soul; discover the NEW one (diff the
        //    soul set so this works for the 2nd/3rd character too).
        let before: HashSet<u32> = self.core.souls().collect();
        // Place the world soul at `at` (world tile coords) if given: decompose into
        // zone (chunk) + local cell via the centred 7×7 fold. `None` → origin.
        let (macro_zone, q, r) = match at {
            Some((wq, wr)) => {
                use resonantdust_codec::packed::{pack_macro_zone_full, zone_local};
                let (zq, lq) = zone_local(wq);
                let (zr, lr) = zone_local(wr);
                (pack_macro_zone_full(0, WORLD_LAYER, zq, zr), lq, lr)
            }
            None => (0, 0, 0),
        };
        self.send(Command::CreateCard {
            owner: player_soul,
            card_key: soul_key.to_string(),
            surface: WORLD_LAYER,
            macro_zone,
            q,
            r,
        })
        .await?;
        self.await_call().await?;
        if !self
            .pump_for_state(Duration::from_secs(5), |c| c.souls().any(|s| !before.contains(&s)))
            .await?
        {
            bail!("{}: world soul {soul_key:?} never discovered", self.name);
        }
        let soul = self.core.souls().find(|s| !before.contains(s)).unwrap();

        // 3. loadout into the world soul's inventory, one card each.
        for key in loadout {
            self.send(Command::CreateCard {
                owner: soul,
                card_key: key.to_string(),
                surface: INVENTORY_LAYER,
                macro_zone: 0,
                q: 0,
                r: 0,
            })
            .await?;
            self.await_call().await?;
        }
        self.pump(Duration::from_secs(2)).await?; // let the rows stream in
        println!(
            "[{}] seeded {soul_key} soul {soul} under player_soul {player_soul} + {} cards",
            self.name,
            loadout.len()
        );
        Ok(soul)
    }

    /// Validate per-card stock READ + WRITE end-to-end: create a fresh dust root
    /// + a `tally` (8-bit `progress` stock), stack the tally on the dust, then let
    /// the triggered `prime` recipe read the tally's progress (decoded from its
    /// stock u32) and `inc` it (→ SetCardStock) until it reaches 3. Asserts the
    /// per-card stock actually climbed 0→3 — proves stock_to_vec (read) and the
    /// SetCardStock write path work, not just compile.
    pub async fn run_stock_progress(&mut self) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_millis(500)).await?;
        let tally_def =
            self.core.packed_def("tally").ok_or_else(|| anyhow::anyhow!("[{tag}] no tally def"))?;
        let human = self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let me = self.player_id().unwrap_or(0);

        // Fresh tally, loose in the human's inventory. `prime` is a ROOT-ONLY
        // recipe — it fires on the tally itself (no stack), so a loose tally is
        // its own chain root.
        self.send(Command::CreateCard {
            owner: human,
            card_key: "tally".to_string(),
            surface: INVENTORY_LAYER,
            macro_zone: 0,
            q: 0,
            r: 0,
        })
        .await?;
        self.await_call().await?;

        let find_loose = |c: &Client, def: u16| -> Option<u32> {
            let now = c.clock_ms();
            c.world()
                .cards
                .current_all(now)
                .find(|r| {
                    r.packed_definition == def
                        && !r.is_dead()
                        && matches!(r.micro(), Micro::Loose { .. })
                        && owning_player(c.world(), r.card_id, now) == Some(me)
                })
                .map(|r| r.card_id)
        };
        if !self
            .pump_for_state(Duration::from_secs(5), |c| find_loose(c, tally_def).is_some())
            .await?
        {
            bail!("[{tag}] tally never discovered");
        }
        let tally = find_loose(&self.core, tally_def).unwrap();
        let progress = |c: &Client| -> u32 {
            let now = c.clock_ms();
            c.world().cards.current(tally, now).map(|r| r.stock & 0xFF).unwrap_or(0)
        };

        // STAGE 3: the freshly-spawned tally must carry its `@define` stock
        // default (progress = 1), proving the gate seeds stock at create time.
        let seeded = progress(&self.core);
        if seeded != 1 {
            bail!("[{tag}] spawn-default FAILED — expected fresh tally progress 1, got {seeded}");
        }
        println!("[{tag}] ✓ spawn-default verified — fresh tally progress = 1 (from @define)");
        println!("[{tag}] tally {tally} loose; root-only `prime` priming progress 1→3 ...");

        // The triggered `prime` recipe fires repeatedly (budget ~1/s), inc-ing the
        // tally's per-card stock until 3. Poll its decoded progress (bits 0-7).
        let mut last = seeded;
        for _ in 0..30 {
            self.pump(Duration::from_millis(500)).await?;
            let p = progress(&self.core);
            if p != last {
                println!("[{tag}]   progress = {p}");
                last = p;
            }
            if p >= 3 {
                break;
            }
        }
        let p = progress(&self.core);
        if p != 3 {
            bail!("[{tag}] stock read/write FAILED — expected progress 3, got {p}");
        }
        println!("[{tag}] ✓ stock read/write verified — progress climbed 1→3 via SetCardStock");
        Ok(())
    }

    /// Validate `as` handle-binding end-to-end: spawn an `anvil` (root-only
    /// marker), let the triggered `forge` recipe fire once — it creates a `widget`
    /// into the owner's inventory bound `&h as`, folds the widget's `progress`
    /// stock 1→4, and nests a `pip` into the widget. Asserts the widget appears
    /// owned by us with progress = 4, proving same-card stock FOLDS into the
    /// Create row (no SetCardStock). Returns the widget's card_id so the caller
    /// can confirm the nested pip's `owner_id` resolved to it (the shard tag → id
    /// path) by querying the shard DB — the pip lives in the widget's own
    /// inventory zone, which the client doesn't subscribe to.
    pub async fn run_forge_as(&mut self) -> anyhow::Result<u32> {
        let tag = self.name.clone();
        self.pump(Duration::from_millis(500)).await?;
        let anvil_def =
            self.core.packed_def("anvil").ok_or_else(|| anyhow::anyhow!("[{tag}] no anvil def"))?;
        let widget_def =
            self.core.packed_def("widget").ok_or_else(|| anyhow::anyhow!("[{tag}] no widget def"))?;
        let human = self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let me = self.player_id().unwrap_or(0);

        // Fresh anvil, loose in the human's inventory — its own chain root, so the
        // root-only `forge` fires on it.
        self.send(Command::CreateCard {
            owner: human,
            card_key: "anvil".to_string(),
            surface: INVENTORY_LAYER,
            macro_zone: 0,
            q: 0,
            r: 0,
        })
        .await?;
        self.await_call().await?;
        if !self
            .pump_for_state(Duration::from_secs(5), |c| {
                find_owned_loose(c, anvil_def, me).is_some()
            })
            .await?
        {
            bail!("[{tag}] anvil never discovered");
        }
        println!("[{tag}] anvil spawned; waiting for root-only `forge` to fire ...");

        // `forge` fires (budget ~1/s): a widget appears in our inventory with its
        // progress folded to 4.
        let widget_progress = |c: &Client| -> Option<(u32, u32)> {
            let now = c.clock_ms();
            find_owned_loose(c, widget_def, me)
                .and_then(|id| c.world().cards.current(id, now).map(|r| (id, r.stock & 0xFF)))
        };
        let mut found = None;
        for _ in 0..30 {
            self.pump(Duration::from_millis(500)).await?;
            if let Some((id, prog)) = widget_progress(&self.core) {
                found = Some((id, prog));
                break;
            }
        }
        let (widget, prog) = found.ok_or_else(|| anyhow::anyhow!("[{tag}] forge never created a widget"))?;
        if prog != 4 {
            bail!("[{tag}] fold FAILED — expected widget progress 4 (default 1 + abs set 4), got {prog}");
        }
        println!("[{tag}] ✓ fold verified — widget {widget} created with progress = 4 (folded into Create)");
        Ok(widget)
    }

    /// Pump until the next reducer reply lands (CallOk/CallErr).
    async fn await_call(&mut self) -> anyhow::Result<()> {
        self.pump_until(Duration::from_secs(5), |e| {
            matches!(e, Event::CallOk { .. } | Event::CallErr { .. })
        })
        .await?;
        Ok(())
    }

    /// Run the `corpus_b_top` → `corpus_dim` chain end-to-end on fresh cards.
    /// Two corpus stacked (branch 1) fire `corpus_b_top`: it destroys the root
    /// corpus, splices the stacked one down to a new loose root, and creates a
    /// `corpus_dim` in the owner's inventory. That `corpus_dim` is a loose root,
    /// so the **root-only** `corpus_dim` recipe then fires on it: destroy the
    /// corpus_dim, create a `corpus`. Asserts both legs (queue → proposal →
    /// effects). Self-contained: mints its own corpus and tracks them by id, so
    /// it's independent of the loadout / other scenarios. Run it BEFORE
    /// `run_triple_corpus` (which leaves a 2-corpus stack that also matches
    /// `corpus_b_top`) so the only `corpus_dim` in flight is this chain's.
    pub async fn run_corpus_dim_chain(&mut self) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_secs(3)).await?;
        let corpus_def =
            self.core.packed_def("corpus").ok_or_else(|| anyhow::anyhow!("[{tag}] no corpus def"))?;
        let corpus_dim_def = self
            .core
            .packed_def("corpus_dim")
            .ok_or_else(|| anyhow::anyhow!("[{tag}] no corpus_dim def"))?;
        let human =
            self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let me = self.player_id().unwrap_or(0);

        let loose_of = |c: &Client, def: u16| -> Vec<u32> {
            let now = c.clock_ms();
            c.world()
                .cards
                .current_all(now)
                .filter(|r| {
                    r.packed_definition == def
                        && !r.is_dead()
                        && matches!(r.micro(), Micro::Loose { .. })
                        && owning_player(c.world(), r.card_id, now) == Some(me)
                })
                .map(|r| r.card_id)
                .collect()
        };

        // Mint two fresh corpus, identify them by set-difference (the loadout has
        // its own loose corpus we must not touch).
        let before: HashSet<u32> = loose_of(&self.core, corpus_def).into_iter().collect();
        for _ in 0..2 {
            self.send(Command::CreateCard {
                owner: human,
                card_key: "corpus".to_string(),
                surface: INVENTORY_LAYER,
                macro_zone: 0,
                q: 0,
                r: 0,
            })
            .await?;
            self.await_call().await?;
        }
        if !self
            .pump_for_state(Duration::from_secs(5), |c| {
                loose_of(c, corpus_def).iter().filter(|id| !before.contains(id)).count() >= 2
            })
            .await?
        {
            bail!("[{tag}] fresh corpus never discovered");
        }
        let fresh: Vec<u32> =
            loose_of(&self.core, corpus_def).into_iter().filter(|id| !before.contains(id)).collect();
        let (base, top) = (fresh[0], fresh[1]);

        // Stack `top` onto `base` (UP = branch 1) — a LOCAL move (commit-based: no
        // wire traffic; positions flush when the proposal commits). The matcher
        // promotes the loose root `base` into branch 1, so slot.1.0=base, slot.1.1=top.
        self.core
            .place(top, Placement::Stack { parent_id: base, direction: STACK_DIR_UP })
            .map_err(|e| anyhow::anyhow!("[{tag}] stack {top}→{base} rejected (feasibility): {e}"))?;
        self.pump(Duration::from_secs(3)).await?; // settle to the match
        // Assert the stack is real (locally) BEFORE the destroy — so a never-stacked
        // card can't make the later splice assertion a false pass.
        match self.core.world().cards.current(top, self.core.clock_ms()).map(|c| c.micro()) {
            Some(Micro::Stacked { root, .. }) if root == base => {}
            other => bail!("[{tag}] {top} not stacked on {base} before corpus_b_top: {other:?}"),
        }
        println!("[{tag}] corpus {top} stacked on {base} (local); expecting corpus_b_top ...");

        // LEG 1 — corpus_b_top: queued + debounced (2 inputs → 5s window).
        let queued = self.core.queued();
        if !queued.iter().any(|(_, r, in_flight)| r == "corpus_b_top" && !in_flight) {
            bail!("[{tag}] expected corpus_b_top queued+debounced; got {queued:?}");
        }
        println!("[{tag}] ✓ corpus_b_top queued + debounced");

        // Fire → gate-accepted.
        let mut cleared = false;
        for _ in 0..24 {
            self.pump(Duration::from_millis(500)).await?;
            if self.core.queued().iter().all(|(_, r, _)| r != "corpus_b_top") {
                cleared = true;
                break;
            }
        }
        if !cleared {
            bail!("[{tag}] corpus_b_top never fired: {:?}", self.core.queued());
        }
        match self.core.drain_action_outcomes().into_iter().find(|(r, _)| r == "corpus_b_top") {
            Some((_, Some(err))) => bail!("[{tag}] corpus_b_top REJECTED by the gate: {err}"),
            None => bail!("[{tag}] queue cleared but no corpus_b_top outcome"),
            Some((_, None)) => {}
        }
        println!("[{tag}] ✓ corpus_b_top fired + gate-accepted");

        // Effect (duration 10 → +10s): base destroyed, top spliced to a loose
        // root (alive), a corpus_dim created in our inventory.
        self.pump(Duration::from_secs(14)).await?;
        {
            let now = self.core.clock_ms();
            let w = self.core.world();
            if w.cards.current(base, now).map(|c| c.is_dead()) != Some(true) {
                bail!("[{tag}] corpus_b_top: root corpus {base} was not destroyed");
            }
            match w.cards.current(top, now) {
                Some(c) if c.is_dead() => bail!("[{tag}] corpus_b_top consumed the stacked corpus {top}"),
                Some(c) if !matches!(c.micro(), Micro::Loose { .. }) => {
                    bail!("[{tag}] stacked corpus {top} did not splice to a loose root")
                }
                None => bail!("[{tag}] stacked corpus {top} vanished"),
                Some(_) => {}
            }
        }
        let dim = loose_of(&self.core, corpus_dim_def);
        if dim.len() != 1 {
            bail!("[{tag}] corpus_b_top: expected exactly 1 corpus_dim created, got {}", dim.len());
        }
        let corpus_dim = dim[0];
        println!("[{tag}] ✓ corpus_b_top effects — root destroyed, {top} spliced, corpus_dim {corpus_dim} created");

        // LEG 2 — corpus_dim (ROOT-ONLY recipe): it auto-fires on the loose
        // corpus_dim (1 input → fire-at-once). Queue + accept.
        self.pump(Duration::from_secs(3)).await?;
        let mut cleared = false;
        for _ in 0..24 {
            self.pump(Duration::from_millis(500)).await?;
            if self.core.queued().iter().all(|(_, r, _)| r != "corpus_dim") {
                if self.core.drain_action_outcomes().iter().any(|(r, _)| r == "corpus_dim") {
                    cleared = true;
                    break;
                }
            }
        }
        if !cleared {
            bail!("[{tag}] root-only corpus_dim recipe never fired on {corpus_dim}");
        }
        println!("[{tag}] ✓ corpus_dim (root-only) fired + gate-accepted");

        // Effect (duration 30 → +30s): corpus_dim destroyed, a fresh corpus made.
        let corpus_before: HashSet<u32> = loose_of(&self.core, corpus_def).into_iter().collect();
        self.pump(Duration::from_secs(34)).await?;
        if self.core.world().cards.current(corpus_dim, self.core.clock_ms()).map(|c| c.is_dead())
            != Some(true)
        {
            bail!("[{tag}] corpus_dim root {corpus_dim} was not destroyed");
        }
        let created = loose_of(&self.core, corpus_def).into_iter().any(|id| !corpus_before.contains(&id));
        if !created {
            bail!("[{tag}] corpus_dim recipe did not create a corpus");
        }
        println!("[{tag}] ✓ corpus_dim effects — corpus_dim destroyed, corpus created");

        let errors = self.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{tag}] {} protocol/decode error(s): {errors:?}", errors.len());
        }
        Ok(())
    }

    /// SPIKE — de-risk the one untested primitive the compressed test needs: the
    /// tile-context `cut_tree` recipe. Single client. Equip = stack the `axe` onto
    /// our world soul (branch UP → the soul's `slot.1.0`, which `cut_tree`'s
    /// `slot.1.0.owner.slot.1.0` axe binding reads); then move a `corpus` onto a
    /// known-wood forest tile in our anchor zone (3,3). Verify `cut_tree`
    /// auto-proposes, fires, and yields `corpus_dim` + `log` into our inventory
    /// with the source corpus destroyed. `tile` is the local (lq, lr) cell.
    /// Magnetic-recipe spike: place a `despair` magnet + a `dread` candidate
    /// adjacent on the world, BOTH owned by this client's player_soul — so this
    /// client acts as the magnetic player that owns + drives them. Asserts the
    /// magnetic pass gathers the dread onto the magnet and `despair_success`
    /// fires (the magnet is consumed). Proves [`Client::magnetic_pass`] end-to-end.
    pub async fn run_magnetic(&mut self) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_secs(3)).await?;
        let despair = self
            .core
            .packed_def("despair")
            .ok_or_else(|| anyhow::anyhow!("[{tag}] no despair def"))?;
        let psoul = self
            .core
            .player_souls()
            .next()
            .ok_or_else(|| anyhow::anyhow!("[{tag}] no player_soul"))?;
        // Adjacent world cells in alice's spawn zone (3,3) = tiles 24..31, clear of
        // her human at (26,26). hex_dist((28,28),(29,28)) = 1, well within radius 3.
        let zone = pack_macro_zone_full(0, WORLD_LAYER, 3, 3);
        for (key, q, r) in [("despair", 4u8, 4u8), ("dread", 5u8, 4u8)] {
            self.send(Command::CreateCard {
                owner: psoul,
                card_key: key.to_string(),
                surface: WORLD_LAYER,
                macro_zone: zone,
                q,
                r,
            })
            .await?;
            self.await_call().await?;
        }
        if !self
            .pump_for_state(Duration::from_secs(5), |c| {
                c.world()
                    .cards
                    .current_all(c.clock_ms())
                    .any(|r| r.packed_definition == despair && !r.is_dead())
            })
            .await?
        {
            bail!("[{tag}] despair magnet never discovered");
        }
        let magnet = self
            .core
            .world()
            .cards
            .current_all(self.core.clock_ms())
            .find(|r| r.packed_definition == despair && !r.is_dead())
            .map(|r| r.card_id)
            .unwrap();
        println!("[{tag}] despair magnet {magnet} placed; magnetic player should gather dread + fire despair_success ...");
        if !self
            .pump_for_state(Duration::from_secs(20), |c| {
                c.world().cards.current(magnet, c.clock_ms()).map(|r| r.is_dead()).unwrap_or(true)
            })
            .await?
        {
            bail!("[{tag}] despair magnet {magnet} never consumed — success did not fire");
        }
        println!("[{tag}] magnet {magnet} consumed — despair_success fired ✓");
        Ok(())
    }

    pub async fn run_cut_tree(&mut self, tile: (u8, u8)) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_secs(2)).await?;
        let def = |k: &str| self.core.packed_def(k).ok_or_else(|| anyhow::anyhow!("[{tag}] no {k} def"));
        let (axe_def, corpus_def) = (def("axe")?, def("corpus")?);
        let (corpus_dim_def, log_def) = (def("corpus_dim")?, def("log")?);
        let soul = self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let me = self.player_id().unwrap_or(0);

        let owned_of = |c: &Client, d: u16, loose_only: bool| -> Vec<u32> {
            let now = c.clock_ms();
            c.world()
                .cards
                .current_all(now)
                .filter(|r| {
                    r.packed_definition == d
                        && !r.is_dead()
                        && (!loose_only || matches!(r.micro(), Micro::Loose { .. }))
                        && owning_player(c.world(), r.card_id, now) == Some(me)
                })
                .map(|r| r.card_id)
                .collect()
        };
        let axe = *owned_of(&self.core, axe_def, false).first().ok_or_else(|| anyhow::anyhow!("[{tag}] no axe"))?;
        let corpus = *owned_of(&self.core, corpus_def, true).first().ok_or_else(|| anyhow::anyhow!("[{tag}] no loose corpus"))?;

        // 1. EQUIP — stack the axe onto the world soul (branch UP → soul.slot.1.0).
        self.core
            .place(axe, Placement::Stack { parent_id: soul, direction: STACK_DIR_UP })
            .map_err(|e| anyhow::anyhow!("[{tag}] stack axe {axe}→soul {soul} rejected: {e}"))?;
        self.pump(Duration::from_secs(1)).await?;
        match self.core.world().cards.current(axe, self.core.clock_ms()).map(|c| c.micro()) {
            Some(Micro::Stacked { root, branch, index }) if root == soul && branch == STACK_DIR_UP && index == 0 => {}
            other => bail!("[{tag}] axe {axe} not at soul.slot.1.0 (UP/0): {other:?}"),
        }
        println!("[{tag}] axe {axe} equipped on soul {soul}; moving corpus {corpus} onto tree {tile:?} ...");

        // 2. MOVE the corpus onto the known-wood forest tile in our anchor zone (3,3).
        let zone = pack_macro_zone_full(0, WORLD_LAYER, 3, 3);
        self.core
            .place(corpus, Placement::Loose { surface: WORLD_LAYER, macro_zone: zone, q: tile.0, r: tile.1, x: 0, y: 0 })
            .map_err(|e| anyhow::anyhow!("[{tag}] move corpus {corpus}→tree rejected: {e}"))?;
        self.pump(Duration::from_secs(3)).await?; // settle to the match

        // 3. cut_tree queued.
        let queued = self.core.queued();
        if !queued.iter().any(|(_, r, _)| r == "cut_tree") {
            bail!("[{tag}] expected cut_tree queued after corpus→tree; got {queued:?}");
        }
        println!("[{tag}] ✓ cut_tree queued");

        // 4. fire → gate-accepted.
        let mut cleared = false;
        for _ in 0..24 {
            self.pump(Duration::from_millis(500)).await?;
            if self.core.queued().iter().all(|(_, r, _)| r != "cut_tree") {
                cleared = true;
                break;
            }
        }
        if !cleared {
            bail!("[{tag}] cut_tree never fired: {:?}", self.core.queued());
        }
        match self.core.drain_action_outcomes().into_iter().find(|(r, _)| r == "cut_tree") {
            Some((_, Some(err))) => bail!("[{tag}] cut_tree REJECTED by the gate: {err}"),
            None => bail!("[{tag}] queue cleared but no cut_tree outcome"),
            Some((_, None)) => {}
        }
        println!("[{tag}] ✓ cut_tree fired + gate-accepted");

        // 5. effects (duration 10 → +10s): corpus destroyed; corpus_dim + log in
        //    our inventory.
        let dim_before: HashSet<u32> = owned_of(&self.core, corpus_dim_def, false).into_iter().collect();
        let log_before: HashSet<u32> = owned_of(&self.core, log_def, false).into_iter().collect();
        self.pump(Duration::from_secs(14)).await?;
        if self.core.world().cards.current(corpus, self.core.clock_ms()).map(|c| c.is_dead()) != Some(true) {
            bail!("[{tag}] cut_tree: corpus {corpus} was not destroyed");
        }
        let new_dim = owned_of(&self.core, corpus_dim_def, false).into_iter().any(|id| !dim_before.contains(&id));
        let new_log = owned_of(&self.core, log_def, false).into_iter().any(|id| !log_before.contains(&id));
        if !new_dim {
            bail!("[{tag}] cut_tree did not create a corpus_dim");
        }
        if !new_log {
            bail!("[{tag}] cut_tree did not create a log");
        }
        println!("[{tag}] ✓ cut_tree effects — corpus destroyed, corpus_dim + log created");

        let errors = self.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{tag}] {} protocol/decode error(s): {errors:?}", errors.len());
        }
        Ok(())
    }

    /// Movement S1: a single FUTURE-STAMPED step. Move our soul one tile east and
    /// verify (a) it stays at the start cell until the cost-derived arrival, then
    /// (b) promotes to the destination when the clock reaches the stamped row.
    /// Same zone (3,3) [forest, cost 30; soul speed 12 → ~2.5 s travel].
    pub async fn run_move_step(&mut self) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_secs(2)).await?;
        let soul = self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let (cq, cr) = self
            .core
            .soul_cell(soul)
            .ok_or_else(|| anyhow::anyhow!("[{tag}] soul {soul} not on the world surface"))?;
        let dest = (cq + 1, cr); // adjacent hex east, still in zone (3,3)
        println!("[{tag}] move soul {soul}: ({cq},{cr}) → {dest:?} (forest cost 30 / speed 12 → ~2.5s)");

        self.send(Command::MoveStep { soul, dest_q: dest.0, dest_r: dest.1 }).await?;
        self.await_call().await?;

        // The future-stamped row must NOT have promoted yet — the soul exists at
        // the destination only once the clock reaches `arrival_ms`.
        self.pump(Duration::from_millis(500)).await?;
        if self.core.soul_cell(soul) != Some((cq, cr)) {
            bail!("[{tag}] soul left start ({cq},{cr}) before arrival: now {:?}", self.core.soul_cell(soul));
        }
        println!("[{tag}] ✓ soul still at ({cq},{cr}) mid-move (future row pending, not yet arrived)");

        // Pump past the ~2.5 s arrival; the future row promotes via `current(now)`.
        self.pump(Duration::from_secs(4)).await?;
        match self.core.soul_cell(soul) {
            Some(p) if p == dest => {}
            other => bail!("[{tag}] soul did not arrive at {dest:?}; at {other:?}"),
        }
        println!("[{tag}] ✓ soul arrived at {dest:?} (future-stamped move promoted on schedule)");

        let errors = self.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{tag}] {} protocol/decode error(s): {errors:?}", errors.len());
        }
        Ok(())
    }

    /// Movement S2: a PIPELINED multi-tile walk. `MoveTo` a target N tiles away;
    /// the client computes a hex path and requests each step as the prior step's
    /// row arrives (the server gets each traversal as lead time). Verify the soul
    /// reaches the target on the cumulative schedule. `dx` tiles east, same zone.
    pub async fn run_move_to(&mut self, dx: i32) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_secs(2)).await?;
        let soul = self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let (cq, cr) = self
            .core
            .soul_cell(soul)
            .ok_or_else(|| anyhow::anyhow!("[{tag}] soul {soul} not on world"))?;
        let target = (cq + dx, cr);
        println!("[{tag}] walk soul {soul}: ({cq},{cr}) → {target:?} ({dx} tiles, pipelined)");

        self.send(Command::MoveTo { soul, target_q: target.0, target_r: target.1 }).await?;

        // Pump through the traversal (~2.5s/forest-tile; pump generously).
        let mut arrived = false;
        for _ in 0..40 {
            self.pump(Duration::from_millis(500)).await?;
            if self.core.soul_cell(soul) == Some(target) {
                arrived = true;
                break;
            }
        }
        if !arrived {
            bail!("[{tag}] soul did not reach {target:?}; stuck at {:?}", self.core.soul_cell(soul));
        }
        println!("[{tag}] ✓ soul walked {dx} tiles to {target:?} via pipelined future-stamped steps");

        let errors = self.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{tag}] {} protocol/decode error(s): {errors:?}", errors.len());
        }
        Ok(())
    }

    /// Movement S4 negative: an ILLEGAL non-adjacent jump (2 tiles in one step)
    /// must be rejected by the gate's adjacency check — the soul stays put.
    pub async fn run_move_reject(&mut self) -> anyhow::Result<()> {
        let tag = self.name.clone();
        self.pump(Duration::from_secs(1)).await?;
        let soul = self.core.souls().next().ok_or_else(|| anyhow::anyhow!("[{tag}] no soul"))?;
        let (cq, cr) = self
            .core
            .soul_cell(soul)
            .ok_or_else(|| anyhow::anyhow!("[{tag}] soul not on world"))?;
        let bad = (cq + 2, cr); // 2 tiles east — NOT a hex neighbor
        println!("[{tag}] illegal jump attempt {:?} → {bad:?} (expect gate reject)", (cq, cr));

        self.send(Command::MoveStep { soul, dest_q: bad.0, dest_r: bad.1 }).await?;
        // Pump; a rejected move stamps nothing, so the soul must NOT have moved.
        self.pump(Duration::from_secs(2)).await?;
        if self.core.soul_cell(soul) != Some((cq, cr)) {
            bail!("[{tag}] illegal 2-tile jump was NOT rejected — soul at {:?}", self.core.soul_cell(soul));
        }
        println!("[{tag}] ✓ gate rejected the illegal jump (soul held at {:?})", (cq, cr));
        Ok(())
    }

    /// Run the triple_corpus recipe scenario on THIS session's own cards: find
    /// our 3 loose corpus + a dust, stack the corpus onto the dust (real moves),
    /// then let the triggered action queue debounce → fire → the gate apply the
    /// `destroy` effect. Asserts the full chain: debounce/supersede, the matcher
    /// soundness oracle, a real CallOk (not a dropped rejection), the effect
    /// landing (corpus[0] destroyed, the rest intact), and no decode errors.
    /// Self-contained per session — multiple clients can each run it.
    pub async fn run_triple_corpus(&mut self) -> anyhow::Result<()> {
        let tag = self.name.clone();
        // Catch up + let the clock RE-STABILIZE first. A session idle while
        // another took its turn has a stale `clock_ms` + buffered frames; its
        // first ticks jump the clock ~forward, and during that transient the
        // future-stamped rows from our own moves can be evaluated a tick before
        // they promote (the trigger then clears the root and promotion doesn't
        // re-fire it). A few seconds of pumping settles the ClockSync re-anchor
        // so our moves' rows promote on the same timeline alice's do.
        self.pump(Duration::from_secs(3)).await?;
        let corpus_def = self.core.packed_def("corpus");
        let dust_def = self.core.packed_def("dust");
        let (Some(corpus_def), Some(dust_def)) = (corpus_def, dust_def) else {
            bail!("[{tag}] bundle missing corpus/dust defs ({corpus_def:?}, {dust_def:?})");
        };

        let now = self.core.clock_ms();
        let me = self.player_id().unwrap_or(0);
        let mut root: Option<u32> = None;
        let mut corpus: Vec<u32> = Vec::new();
        {
            let w = self.core.world();
            for c in w.cards.current_all(now) {
                if c.is_dead() || owning_player(w, c.card_id, now) != Some(me) {
                    continue;
                }
                if c.packed_definition == corpus_def && matches!(c.micro(), Micro::Loose { .. }) {
                    corpus.push(c.card_id);
                } else if c.packed_definition == dust_def && root.is_none() {
                    root = Some(c.card_id);
                }
            }
        }
        corpus.truncate(3);
        let Some(root) = root else {
            bail!("[{tag}] fixture: no loose dust owned by us to root the recipe on");
        };
        if corpus.len() < 3 {
            bail!("[{tag}] fixture: need 3 loose corpus, found {}", corpus.len());
        }

        // Stack the 3 corpus onto the dust — LOCAL moves (commit-based: no wire
        // traffic per move; positions flush when triple_corpus commits). Stack all
        // three BEFORE pumping so the matcher sees the full 3-stack, not an
        // intermediate 2 (which would match corpus_b_top first).
        println!("[{tag}] stacking 3 corpus onto dust {root} locally (commit-based) ...");
        for &c in &corpus {
            self.core
                .place(c, Placement::Stack { parent_id: root, direction: STACK_DIR_UP })
                .map_err(|e| anyhow::anyhow!("[{tag}] stack {c} rejected (feasibility): {e}"))?;
        }
        self.pump(Duration::from_secs(3)).await?; // settle to the match

        // Debounce + supersede: triple_corpus queued, nothing fired yet.
        let queued = self.core.queued();
        if !queued.iter().any(|(_, r, in_flight)| r == "triple_corpus" && !in_flight) {
            bail!("[{tag}] expected triple_corpus queued+debounced; got {queued:?}");
        }
        if queued.iter().any(|(_, _, in_flight)| *in_flight) {
            bail!("[{tag}] an action fired before the debounce window: {queued:?}");
        }
        println!("[{tag}] ✓ triple_corpus queued + debounced");

        // Soundness oracle: the index pre-filter must not drop a valid match.
        let soul = self.core.souls().next().expect("a soul for the matcher oracle");
        if self.core.match_recipes(soul, root) != self.core.match_recipes_unfiltered(soul, root) {
            bail!("[{tag}] aspect pre-filter dropped a valid match (filtered != exhaustive)");
        }

        // Fire: poll until the queue clears, then check the REAL outcome (the
        // queue empties on a dropped rejection too — empty ≠ accepted).
        let mut cleared = false;
        for _ in 0..24 {
            self.pump(Duration::from_millis(500)).await?;
            if self.core.queued().is_empty() {
                cleared = true;
                break;
            }
        }
        if !cleared {
            bail!("[{tag}] queue not cleared — triple_corpus didn't fire: {:?}", self.core.queued());
        }
        let outcomes = self.core.drain_action_outcomes();
        match outcomes.iter().find(|(r, _)| r == "triple_corpus") {
            Some((_, Some(err))) => bail!("[{tag}] triple_corpus REJECTED by the gate: {err}"),
            None => bail!("[{tag}] queue emptied but no triple_corpus outcome: {outcomes:?}"),
            Some((_, None)) => {}
        }
        println!("[{tag}] ✓ triple_corpus fired + gate-accepted");

        // Effect: completion is future-stamped at now + duration(10)·1000 = +10s.
        // Pump past it; assert corpus[0] destroyed, the other two + dust intact.
        self.pump(Duration::from_secs(14)).await?;
        let now = self.core.clock_ms();
        let is_dead = |id: u32| self.core.world().cards.current(id, now).map(|c| c.is_dead());
        if !is_dead(corpus[0]).unwrap_or(true) {
            bail!("[{tag}] effect: corpus[0]={} was not destroyed", corpus[0]);
        }
        let survivors =
            [corpus[1], corpus[2]].iter().filter(|&&c| is_dead(c) == Some(false)).count();
        if survivors != 2 {
            bail!("[{tag}] effect: over-consumed — expected 2 surviving corpus, got {survivors}");
        }
        if is_dead(root) != Some(false) {
            bail!("[{tag}] effect: destroyed the dust root {root} (should survive)");
        }
        println!("[{tag}] ✓ effect verified — corpus[0] destroyed, 2 corpus + dust intact");

        let errors = self.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{tag}] {} protocol/decode error(s): {errors:?}", errors.len());
        }
        Ok(())
    }
}

/// The card_id of a live, loose card of `def` owned (transitively) by player
/// `me`, if any. Shared by the stock/forge scenarios.
fn find_owned_loose(c: &Client, def: u16, me: u32) -> Option<u32> {
    let now = c.clock_ms();
    c.world()
        .cards
        .current_all(now)
        .find(|r| {
            r.packed_definition == def
                && !r.is_dead()
                && matches!(r.micro(), Micro::Loose { .. })
                && owning_player(c.world(), r.card_id, now) == Some(me)
        })
        .map(|r| r.card_id)
}

/// Drive several sessions together for `budget` so each receives the others'
/// broadcast rows — the multiplayer pump. Round-robins one [`Session::drain_step`]
/// per session per round, so a session blocked on its own quiet socket never
/// starves the rest.
pub async fn pump_all(sessions: &mut [Session], budget: Duration) -> anyhow::Result<()> {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        for s in sessions.iter_mut() {
            s.drain_step(|_| {}).await?;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    Ok(())
}
