//! Resonant Dust client core — entry point + the **multi-client** test harness.
//!
//! The reusable thing is the sans-IO core ([`client::Client`]); a [`session::Session`]
//! wraps it with one gate connection + login (one *player's* client). This `main`
//! opens SEVERAL sessions and drives them together ([`session::pump_all`]) to
//! exercise multiplayer — cross-client visibility, isolation, and concurrent
//! actions — over one live gate.

mod actions;
mod clock;
mod client;
mod content;
mod gate;
mod rows;
mod session;
mod transport;
mod world;
mod zones;

use std::time::Duration;

use resonantdust_data::card_model::Micro;
use resonantdust_data::packed::{pack_macro_zone_full, INVENTORY_LAYER, STACK_DIR_UP, WORLD_LAYER};
use resonantdust_data::recipe_state::owning_player;
use resonantdust_data::stack::Placement;
use session::{pump_all, Session};

/// Gate WS endpoint, selected by environment. `RD_ENV` picks the target gate by
/// its service name on the shared `resonantdust` docker network:
///   - `claude` (default) → `ws://gate-claude:8474/ws` (agent dev)
///   - `test`   → `ws://gate-test:8475/ws`   (the test harness — wiped/reseeded)
///   - `dev`    → `ws://gate:8473/ws`         (the user's gate; don't touch)
/// `GATE_URL` overrides everything (e.g. `ws://127.0.0.1:8475/ws` from the host).
/// Default is `claude` (not `dev`) — env isolation: the agent never drives dev.
fn gate_url() -> String {
    if let Ok(url) = std::env::var("GATE_URL") {
        return url;
    }
    match std::env::var("RD_ENV").as_deref() {
        Ok("dev") => "ws://gate:8473/ws".to_string(),
        Ok("test") => "ws://gate-test:8475/ws".to_string(),
        _ => "ws://gate-claude:8474/ws".to_string(), // claude default
    }
}

/// Derive the gate's HTTP base (`http://host:port`) from its WS url.
fn http_base(ws_url: &str) -> String {
    ws_url
        .replace("ws://", "http://")
        .replace("wss://", "https://")
        .trim_end_matches("/ws")
        .to_string()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = gate_url();
    let http = http_base(&url);
    // Client roster — `CLIENT_NAMES=alice,bob,carol` (default two). Each name is
    // one player's client; the scenario generalizes to N (cross-checks need ≥2).
    let names: Vec<String> = std::env::var("CLIENT_NAMES")
        .ok()
        .map(|s| s.split(',').map(|n| n.trim().to_string()).filter(|n| !n.is_empty()).collect())
        .filter(|v: &Vec<String>| !v.is_empty())
        .unwrap_or_else(|| vec!["alice".to_string(), "bob".to_string()]);
    println!("harness: connecting {} client(s) to {url}: {names:?}", names.len());

    // --- Open each client (own WS + sans-IO core) and seed its character ------
    // Every player gets a player_soul → human (world 0,0) → private loadout.
    let loadout = ["dust", "corpus", "corpus", "corpus", "axe"];
    let mut sessions: Vec<Session> = Vec::new();
    let mut humans: Vec<u32> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let mut s = Session::start(&url, &http, name).await?;
        // Spawn each soul in macro_zone (3,3) [plains], one tile apart so they
        // share a zone but not a cell: alice@(26,26), bob@(28,28), … (26+2i).
        let coord = 26 + (i as i32) * 2;
        let human = s.seed_soul("human", &loadout, Some((coord, coord))).await?;
        println!("harness: {name} player={:?} human={human}", s.player_id());
        sessions.push(s);
        humans.push(human);
    }

    // --- SPIKE hook: de-risk the tile-context cut_tree primitive in isolation.
    // `CUT_TREE_SPIKE=1` runs ONLY the single-client cut_tree spike on alice (it
    // consumes the axe + a corpus, so it can't share a run with the recipe tests)
    // and returns. alice's tree tile = zone (3,3) local (5,5) [world (29,29)].
    if std::env::var("CUT_TREE_SPIKE").is_ok() {
        if let Some(s) = sessions.first_mut() {
            s.run_cut_tree((5, 5)).await?;
        }
        println!("harness: CUT_TREE SPIKE PASS");
        return Ok(());
    }

    // --- DEFAULT: the compressed two-client integration test ------------------
    // One run exercises most of the game: concurrent corpus_b_top (different
    // acquisition order) + cut_tree on a shared world, cross-client visibility +
    // private-inventory isolation, and both corpus_dim chains back to corpus.
    // `INDIVIDUAL=1` falls back to the separate per-feature tests below (the
    // failure-triage path). `CLIENT_NAMES` must yield ≥2 clients for combined.
    if std::env::var("INDIVIDUAL").is_err() {
        if sessions.len() < 2 {
            anyhow::bail!("combined test needs ≥2 clients (set CLIENT_NAMES or INDIVIDUAL=1)");
        }
        run_combined(&mut sessions, &humans).await?;
        println!("harness: HARNESS PASS (combined, {} clients)", sessions.len());
        return Ok(());
    }

    // --- Multiplayer sync + cross-client checks (only with ≥2 clients) --------
    if sessions.len() >= 2 {
        // Distinct souls per player.
        for i in 0..humans.len() {
            for j in (i + 1)..humans.len() {
                if humans[i] == humans[j] {
                    anyhow::bail!("HARNESS FAIL: clients {i}/{j} share soul id {}", humans[i]);
                }
            }
        }
        // Drive ALL together so each receives the others' broadcast world rows.
        pump_all(&mut sessions, Duration::from_secs(4)).await?;

        // (1) CROSS-CLIENT VISIBILITY: every human shares macro_zone (3,3) (one
        //     tile apart — alice@(26,26), bob@(28,28)), so each client's world-
        //     zone sub must surface every OTHER client's soul, without claiming
        //     it as its own.
        for (i, s) in sessions.iter().enumerate() {
            let now = s.core.clock_ms();
            for (j, &other) in humans.iter().enumerate() {
                if i == j {
                    continue;
                }
                if s.core.world().cards.current(other, now).is_none() {
                    anyhow::bail!(
                        "HARNESS FAIL: {} can't see client {j}'s world soul {other} (visibility broken)",
                        s.name
                    );
                }
                if s.core.souls().any(|x| x == other) {
                    anyhow::bail!("HARNESS FAIL: {} claimed foreign soul {other} as its own", s.name);
                }
                // (2) INVENTORY ISOLATION: the world soul is shared, its inventory
                //     is NOT — see no cards in the other's inventory ZONE. Key on
                //     macro_zone (unambiguous) not owner_id — the player_id/card_id
                //     overlap means `owner_id == other` would also match this
                //     client's OWN player_soul (false positive).
                let other_inv = pack_macro_zone_full(other, INVENTORY_LAYER, 0, 0);
                let leaked =
                    s.core.world().cards.current_all(now).filter(|c| c.macro_zone == other_inv).count();
                if leaked != 0 {
                    anyhow::bail!(
                        "HARNESS FAIL: {} sees {leaked} cards in client {j}'s inventory zone (isolation broken)",
                        s.name
                    );
                }
            }
        }
        println!("harness: ✓ cross-client visibility + inventory isolation across {} clients", sessions.len());

        // (3) SHARED-ZONE OBSERVERS: both souls anchor in macro_zone (3,3), so
        //     each client subscribes `cards WHERE macro_zone=(3,3)`. The gate
        //     counts distinct connection-subscribers and broadcasts the count;
        //     with two clients anchored there it must read exactly 2 on BOTH
        //     (the observer-gated-sync trigger — Phase 2 — observed end-to-end).
        let spawn_zone = pack_macro_zone_full(0, WORLD_LAYER, 3, 3);
        // One more concurrent pump so any in-flight ZoneObservers broadcast is
        // drained on every socket (the 1→2 transition fires while the 2nd client
        // seeds, before this point).
        pump_all(&mut sessions, Duration::from_secs(2)).await?;
        for s in &sessions {
            let obs = s.core.observers(spawn_zone);
            if obs != sessions.len() as u32 {
                anyhow::bail!(
                    "HARNESS FAIL: {} reads {obs} observer(s) for spawn zone (3,3), expected {} (observer count broken)",
                    s.name,
                    sessions.len()
                );
            }
        }
        println!("harness: ✓ spawn zone (3,3) reads {} observers on every client", sessions.len());

        // (4) MOVE-SYNC (observer-gated, end-to-end): alice drops a NON-anchor card
        //     (her `axe`, untouched by the recipe tests) from her PRIVATE inventory
        //     into the shared zone (3,3), then moves it within the zone. Because the
        //     destination zone has observers > 1, each `place()` commits immediately
        //     via `move_cards` (no recipe proposal needed), so bob — subscribed to
        //     the zone — sees the synced position. Exercises the
        //     `zone_observed_by_others` branch of `place()` across two live clients.
        // (Toggle off with SKIP_MOVE_SYNC=1 to isolate it from later stages.)
        if std::env::var("SKIP_MOVE_SYNC").is_err() {
        let axe_def = sessions[0]
            .core
            .packed_def("axe")
            .ok_or_else(|| anyhow::anyhow!("no axe def in bundle"))?;
        // alice's store only holds HER cards (+ others' world souls), so the lone
        // axe-def card is alice's own.
        let an = sessions[0].core.clock_ms();
        let axe = sessions[0]
            .core
            .world()
            .cards
            .current_all(an)
            .find(|c| c.packed_definition == axe_def && !c.is_dead())
            .map(|c| c.card_id)
            .ok_or_else(|| anyhow::anyhow!("alice has no axe to move"))?;
        // It's private: bob must not see it yet.
        if sessions[1].core.world().cards.current(axe, sessions[1].core.clock_ms()).is_some() {
            anyhow::bail!("HARNESS FAIL: bob sees alice's private axe {axe} before it's world-placed");
        }

        // Drop it loose into the shared world at local cell (5,5).
        sessions[0]
            .core
            .place(axe, Placement::Loose { surface: WORLD_LAYER, macro_zone: spawn_zone, q: 5, r: 5, x: 0, y: 0 })
            .map_err(|e| anyhow::anyhow!("alice place axe→world failed: {e}"))?;
        let placed_micro = sessions[0]
            .core
            .world()
            .cards
            .current(axe, sessions[0].core.clock_ms())
            .map(|c| c.micro_location)
            .unwrap_or(0);

        // bob receives the synced row (gate fans the move to the zone's subscribers).
        let mut seen = false;
        for _ in 0..8 {
            pump_all(&mut sessions, Duration::from_millis(500)).await?;
            let bn = sessions[1].core.clock_ms();
            if let Some(c) = sessions[1].core.world().cards.current(axe, bn) {
                if c.macro_zone == spawn_zone && c.micro_location == placed_micro {
                    seen = true;
                    break;
                }
            }
        }
        if !seen {
            anyhow::bail!("HARNESS FAIL: bob never saw alice's axe {axe} synced into shared zone (3,3)");
        }
        println!("harness: ✓ move-sync — alice's world-placed axe appeared on bob (observer-gated commit)");

        // Move it within the shared zone; the new cell must propagate too.
        sessions[0]
            .core
            .place(axe, Placement::Loose { surface: WORLD_LAYER, macro_zone: spawn_zone, q: 7, r: 3, x: 0, y: 0 })
            .map_err(|e| anyhow::anyhow!("alice move axe failed: {e}"))?;
        let moved_micro = sessions[0]
            .core
            .world()
            .cards
            .current(axe, sessions[0].core.clock_ms())
            .map(|c| c.micro_location)
            .unwrap_or(0);
        if moved_micro == placed_micro {
            anyhow::bail!("HARNESS FAIL: axe micro_location unchanged after in-zone move");
        }
        let mut moved = false;
        for _ in 0..8 {
            pump_all(&mut sessions, Duration::from_millis(500)).await?;
            let bn = sessions[1].core.clock_ms();
            if let Some(c) = sessions[1].core.world().cards.current(axe, bn) {
                if c.micro_location == moved_micro {
                    moved = true;
                    break;
                }
            }
        }
        if !moved {
            anyhow::bail!("HARNESS FAIL: bob never saw alice's axe move to its new cell");
        }
        println!("harness: ✓ move-sync — alice's in-zone move propagated to bob");
        }
    }

    // --- corpus_b_top → corpus_dim chain (incl. the root-only corpus_dim recipe).
    // Run FIRST, on one client, with its own fresh corpus — before triple_corpus
    // leaves a 2-corpus stack that would also spawn corpus_dim.
    if let Some(s) = sessions.first_mut() {
        s.run_corpus_dim_chain().await?;
    }

    // --- Each client runs the recipe on its OWN cards -------------------------
    // Proves multiple clients act independently (and one's destroy never touches
    // another's corpus).
    for s in sessions.iter_mut() {
        s.run_triple_corpus().await?;
    }

    // --- Per-card stock read/write (root-only recipe) -------------------------
    // One client drives the `prime` root-only recipe over a loose tally,
    // climbing its per-card `progress` stock 0→3 via SetCardStock.
    if let Some(s) = sessions.first_mut() {
        s.run_stock_progress().await?;
    }

    println!("harness: HARNESS PASS (individual, {} client(s))", sessions.len());
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Compressed two-client integration test
// ────────────────────────────────────────────────────────────────────────────

/// Card ids of definition `def` owned (transitively) by player `me`, current
/// now, sorted ascending. `loose_only` keeps only loose (un-stacked) cards.
fn owned_cards(s: &Session, me: u32, def: u16, loose_only: bool) -> Vec<u32> {
    let now = s.core.clock_ms();
    let mut v: Vec<u32> = s
        .core
        .world()
        .cards
        .current_all(now)
        .filter(|c| {
            c.packed_definition == def
                && !c.is_dead()
                && (!loose_only || matches!(c.micro(), Micro::Loose { .. }))
                && owning_player(s.core.world(), c.card_id, now) == Some(me)
        })
        .map(|c| c.card_id)
        .collect();
    v.sort_unstable();
    v
}

/// Whether `recipe` currently has a queue entry on this client (queued or
/// in-flight) — the action-queue "start" signal.
fn has_queued(s: &Session, recipe: &str) -> bool {
    s.core.queued().iter().any(|(_, r, _)| r == recipe)
}

/// Per-client plan for the combined run, captured once after seeding.
struct Plan {
    me: u32,
    soul: u32,
    axe: u32,
    /// corpus the stack roots on (destroyed by corpus_b_top).
    parent: u32,
    /// corpus stacked onto `parent`.
    moved: u32,
    /// corpus moved onto the tree tile (consumed by cut_tree).
    tree: u32,
    /// tree tile, zone (3,3) local cell (lq, lr).
    tile: (u8, u8),
}

/// The compressed scenario over two clients (alice=0, bob=1). See the spec in
/// the harness docs; each `harness: ✓` line is one verified guarantee.
async fn run_combined(sessions: &mut [Session], _humans: &[u32]) -> anyhow::Result<()> {
    use anyhow::{anyhow, bail, Context};

    let zone = pack_macro_zone_full(0, WORLD_LAYER, 3, 3);
    // Hard-coded tree tiles in zone (3,3), verified wood≥1, distinct from the
    // souls (alice@local(2,2), bob@local(4,4)) and from each other:
    //   alice → local (5,5) = world (29,29), wood 3
    //   bob   → local (3,6) = world (27,30), wood 2
    let tiles = [(5u8, 5u8), (3u8, 6u8)];

    let cdef = sessions[0].core.packed_def("corpus").context("corpus def")?;
    let cdim = sessions[0].core.packed_def("corpus_dim").context("corpus_dim def")?;
    let logd = sessions[0].core.packed_def("log").context("log def")?;
    let axed = sessions[0].core.packed_def("axe").context("axe def")?;

    // ── Phase 0: capture each client's plan ──────────────────────────────────
    // alice stacks corpus[1]→corpus[0] (stack onto FIRST); bob stacks
    // corpus[1]→corpus[2] (stack onto LAST) — different acquisition order, same
    // recipe, to prove order-independence. The free corpus goes to the tree.
    let mut plans: Vec<Plan> = Vec::new();
    for (i, s) in sessions.iter().enumerate() {
        let me = s.player_id().unwrap_or(0);
        let soul = s.core.souls().next().context("soul")?;
        let axe = *owned_cards(s, me, axed, false).first().ok_or_else(|| anyhow!("client {i}: no axe"))?;
        let corpus = owned_cards(s, me, cdef, true);
        if corpus.len() < 3 {
            bail!("client {i}: expected 3 loose corpus, got {}", corpus.len());
        }
        let (parent, tree) = if i == 0 { (corpus[0], corpus[2]) } else { (corpus[2], corpus[0]) };
        plans.push(Plan { me, soul, axe, parent, moved: corpus[1], tree, tile: tiles[i] });
    }

    // ── Phase 1: concurrent corpus_b_top stack (different acquisition order) ──
    for (i, s) in sessions.iter_mut().enumerate() {
        let p = &plans[i];
        s.core
            .place(p.moved, Placement::Stack { parent_id: p.parent, direction: STACK_DIR_UP })
            .map_err(|e| anyhow!("client {i}: stack corpus {}→{} rejected: {e}", p.moved, p.parent))?;
    }
    pump_all(sessions, Duration::from_secs(3)).await?;
    for (i, s) in sessions.iter().enumerate() {
        if !has_queued(s, "corpus_b_top") {
            bail!("client {i}: corpus_b_top not queued after stack; queue={:?}", s.core.queued());
        }
    }
    println!("harness: ✓ both clients started the corpus_b_top queue (alice stacks-onto-first, bob stacks-onto-last)");

    // ── Phase 2: equip axe on soul (→ soul.slot.1.0) + move corpus onto tree ──
    for (i, s) in sessions.iter_mut().enumerate() {
        let p = &plans[i];
        s.core
            .place(p.axe, Placement::Stack { parent_id: p.soul, direction: STACK_DIR_UP })
            .map_err(|e| anyhow!("client {i}: equip axe {}→soul {} rejected: {e}", p.axe, p.soul))?;
        s.core
            .place(p.tree, Placement::Loose { surface: WORLD_LAYER, macro_zone: zone, q: p.tile.0, r: p.tile.1, x: 0, y: 0 })
            .map_err(|e| anyhow!("client {i}: move corpus {}→tree rejected: {e}", p.tree))?;
    }
    pump_all(sessions, Duration::from_secs(3)).await?;
    // corpus_b_top (queued in Phase 1) may already have fired by now — its 5s
    // debounce overlaps these moves — so we only require cut_tree to have started
    // here; Phase 3 confirms BOTH fired + were accepted.
    for (i, s) in sessions.iter().enumerate() {
        if !has_queued(s, "cut_tree") {
            bail!("client {i}: cut_tree not queued after corpus→tree; queue={:?}", s.core.queued());
        }
    }
    println!("harness: ✓ both clients started the cut_tree queue");

    // ── Phase 3: both recipes fire (5s debounce) + gate-accept ───────────────
    let mut fired = false;
    for _ in 0..40 {
        pump_all(sessions, Duration::from_millis(500)).await?;
        if sessions.iter().all(|s| !has_queued(s, "corpus_b_top") && !has_queued(s, "cut_tree")) {
            fired = true;
            break;
        }
    }
    if !fired {
        bail!("corpus_b_top/cut_tree never fired on both clients");
    }
    for (i, s) in sessions.iter_mut().enumerate() {
        let outs = s.core.drain_action_outcomes();
        for r in ["corpus_b_top", "cut_tree"] {
            match outs.iter().find(|(rr, _)| rr == r) {
                Some((_, Some(err))) => bail!("client {i}: {r} REJECTED by the gate: {err}"),
                None => bail!("client {i}: queue cleared but no {r} outcome ({outs:?})"),
                Some((_, None)) => {}
            }
        }
    }
    println!("harness: ✓ both clients fired + gate-accepted corpus_b_top AND cut_tree");

    // ── Phase 4: completions (durations 10 → +~14s). Both recipes finish:
    //    corpus_b_top → corpus_dim (source corpus destroyed);
    //    cut_tree     → corpus_dim + log (tree corpus destroyed). ───────────────
    pump_all(sessions, Duration::from_secs(16)).await?;
    for (i, s) in sessions.iter().enumerate() {
        let p = &plans[i];
        let now = s.core.clock_ms();
        // source corpora destroyed.
        if s.core.world().cards.current(p.parent, now).map(|c| c.is_dead()) != Some(true) {
            bail!("client {i}: corpus_b_top did not destroy its root corpus {}", p.parent);
        }
        if s.core.world().cards.current(p.tree, now).map(|c| c.is_dead()) != Some(true) {
            bail!("client {i}: cut_tree did not destroy the tree corpus {}", p.tree);
        }
        // two corpus_dim + one log in OUR inventory, owned by us.
        let dims = owned_cards(s, p.me, cdim, false);
        if dims.len() < 2 {
            bail!("client {i}: expected ≥2 corpus_dim (corpus_b_top + cut_tree), got {} ({dims:?})", dims.len());
        }
        if owned_cards(s, p.me, logd, false).is_empty() {
            bail!("client {i}: cut_tree did not yield a log");
        }
    }
    println!("harness: ✓ both completions landed — 2×corpus_dim + log per client, source corpora destroyed");

    // ── Phase 5: corpus_dim recipe queues + proposes on the fresh corpus_dim ──
    let mut queued = false;
    for _ in 0..16 {
        pump_all(sessions, Duration::from_millis(500)).await?;
        if sessions.iter().all(|s| has_queued(s, "corpus_dim")) {
            queued = true;
            break;
        }
    }
    if !queued {
        bail!("corpus_dim recipe never queued on both clients after the corpus_dim cards arrived");
    }
    println!("harness: ✓ both clients started the corpus_dim queue");

    // ── Phase 6: cross-client visibility (moves synced — shared zone (3,3)) ──
    // Each client sees BOTH players' axes (equipped on souls) and the OTHER
    // player's tree corpus at the OTHER player's tile (dead-or-alive: cut_tree
    // may have consumed it, but the synced row persists).
    for (i, s) in sessions.iter().enumerate() {
        let now = s.core.clock_ms();
        for (j, p) in plans.iter().enumerate() {
            if s.core.world().cards.current(p.axe, now).is_none() {
                bail!("client {i}: can't see client {j}'s axe {} (cross-visibility broken)", p.axe);
            }
            if s.core.world().cards.current(p.tree, now).map(|c| c.macro_zone) != Some(zone) {
                bail!("client {i}: can't see client {j}'s corpus {} on its tree tile", p.tree);
            }
        }
    }
    println!("harness: ✓ cross-visibility — each client sees both axes + both tree corpora in zone (3,3)");

    // ── Phase 7: private-inventory ISOLATION (no anchor in the peer's bucket) ─
    // The peer's corpus_dim land in the peer's INVENTORY zone, which we don't
    // observe — so we must NOT see the peer's corpus_dim nor their inventory zone.
    for (i, s) in sessions.iter().enumerate() {
        let now = s.core.clock_ms();
        let peer = &plans[1 - i];
        let peer_inv = pack_macro_zone_full(peer.soul, INVENTORY_LAYER, 0, 0);
        let leaked = s.core.world().cards.current_all(now).filter(|c| c.macro_zone == peer_inv).count();
        if leaked != 0 {
            bail!("client {i}: sees {leaked} card(s) in client {}'s inventory zone (isolation broken)", 1 - i);
        }
        if owned_cards(s, peer.me, cdim, false).iter().any(|&id| s.core.world().cards.current(id, now).is_some()) {
            bail!("client {i}: sees a corpus_dim owned by the peer (isolation broken)");
        }
    }
    println!("harness: ✓ isolation — neither client sees the peer's corpus_dim or inventory zone");

    // ── Phase 8: both corpus_dim recipes complete (duration 30 → +~34s) →
    //    corpus_dim destroyed, corpus returned to our inventory. Two per client
    //    (corpus_b_top's + cut_tree's); pump long enough for both. ──────────────
    let dim_snapshot: Vec<Vec<u32>> =
        sessions.iter().enumerate().map(|(i, s)| owned_cards(s, plans[i].me, cdim, false)).collect();
    let corpus_before: Vec<usize> =
        sessions.iter().enumerate().map(|(i, s)| owned_cards(s, plans[i].me, cdef, false).len()).collect();
    pump_all(sessions, Duration::from_secs(38)).await?;
    for (i, s) in sessions.iter().enumerate() {
        let now = s.core.clock_ms();
        // every corpus_dim that existed pre-pump must now be destroyed.
        for &dim in &dim_snapshot[i] {
            if s.core.world().cards.current(dim, now).map(|c| c.is_dead()) != Some(true) {
                bail!("client {i}: corpus_dim {dim} did not complete back to corpus");
            }
        }
        // and corpus came back (net new loose corpus owned by us).
        let corpus_after = owned_cards(s, plans[i].me, cdef, false).len();
        if corpus_after <= corpus_before[i] {
            bail!("client {i}: corpus_dim recipes returned no corpus ({} → {corpus_after})", corpus_before[i]);
        }
    }
    println!("harness: ✓ both corpus_dim recipes completed — corpus_dim destroyed, corpus returned to each owner");

    for s in sessions.iter_mut() {
        let errors = s.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{}] {} protocol/decode error(s): {errors:?}", s.name, errors.len());
        }
    }
    Ok(())
}
