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
mod outbox;
mod rows;
mod session;
mod transport;
mod world;
mod zones;

use std::time::Duration;

use resonantdust_codec::card_model::Micro;
use resonantdust_codec::packed::{
    pack_macro_zone_full, INVENTORY_LAYER, STACK_DIR_DOWN, STACK_DIR_UP, WORLD_LAYER,
};
use resonantdust_state::recipe_state::owning_player;
use resonantdust_state::stack::Placement;
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
    let loadout = ["test_dust", "corpus", "corpus", "corpus", "axe"];
    // jim (the lock-test player) spawns far out of alice/bob's anchor range and
    // gets a richer kit: dust + 3 corpus + 2 axes + test_dust.
    let jim_loadout = ["dust", "corpus", "corpus", "corpus", "axe", "test_dust", "axe"];
    let mut sessions: Vec<Session> = Vec::new();
    let mut humans: Vec<u32> = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let mut s = Session::start(&url, &http, name).await?;
        // alice/bob share macro_zone (3,3) (7×7 fold of world 20/22 → zone 3,
        // local 2/4), one tile apart (20+2i). jim spawns at world (56,56) — well
        // beyond alice/bob's cold radius (20 tiles), so the three never observe
        // each other.
        let (lo, coord): (&[&str], i32) =
            if name == "jim" { (&jim_loadout, 56) } else { (&loadout, 20 + (i as i32) * 2) };
        let human = s.seed_soul("human", lo, Some((coord, coord))).await?;
        println!("harness: {name} player={:?} human={human}", s.player_id());
        sessions.push(s);
        humans.push(human);
    }

    // --- SPIKE hook: de-risk the tile-context cut_tree primitive in isolation.
    // `CUT_TREE_SPIKE=1` runs ONLY the single-client cut_tree spike on alice (it
    // consumes the axe + a corpus, so it can't share a run with the recipe tests)
    // and returns. alice's tree tile = zone (3,3) local (3,2) [world (21,20)], wood 2.
    if std::env::var("CUT_TREE_SPIKE").is_ok() {
        if let Some(s) = sessions.first_mut() {
            s.run_cut_tree((3, 2)).await?;
        }
        println!("harness: CUT_TREE SPIKE PASS");
        return Ok(());
    }

    // --- MAGNETIC test hook (single-client, alice): place a despair magnet + a
    // dread candidate (both owned by alice's player_soul, so alice's client drives
    // them as the magnetic player) and assert despair_success auto-fires.
    if std::env::var("MAGNETIC_SPIKE").is_ok() {
        if let Some(s) = sessions.first_mut() {
            s.run_magnetic().await?;
        }
        println!("harness: MAGNETIC SPIKE PASS");
        return Ok(());
    }

    // --- `as` handle-binding hook (single-client, alice): spawn an anvil, let
    // the root-only `forge` create a widget bound `&h as` with its progress stock
    // folded to 4 and a pip nested inside it. Asserts the fold on the client;
    // prints the widget id so the tag → id resolution can be confirmed by
    // querying the shard DB for the pip's owner_id.
    if std::env::var("AS_SPIKE").is_ok() {
        if let Some(s) = sessions.first_mut() {
            let widget = s.run_forge_as().await?;
            println!("harness: AS_SPIKE widget card_id = {widget}");
        }
        println!("harness: AS_SPIKE PASS");
        return Ok(());
    }

    // --- MOVEMENT test hook (single-client, alice): the single-step "stays until
    // arrival" property, then a pipelined multi-tile walk.
    if std::env::var("MOVE_TEST").is_ok() {
        if let Some(s) = sessions.first_mut() {
            s.run_move_step().await?;
            s.run_move_to(3).await?;
            s.run_move_reject().await?;
        }
        println!("harness: MOVE_TEST PASS");
        return Ok(());
    }

    // --- DEFAULT: the compressed two-client integration test ------------------
    // One run exercises most of the game: concurrent corpus_b_top (different
    // acquisition order) + cut_tree on a shared world, cross-client visibility +
    // private-inventory isolation, and both corpus_dim chains back to corpus.
    // `INDIVIDUAL=1` falls back to the separate per-feature tests below (the
    // failure-triage path). `CLIENT_NAMES` must yield ≥2 clients for combined.
    // jim present (CLIENT_NAMES=alice,bob,jim) → the stack lock/splice test, with
    // alice/bob seeded only for the isolation check. Otherwise the combined run.
    if names.iter().any(|n| n == "jim") {
        run_jim_lock_test(&mut sessions, &names, &humans).await?;
        println!("harness: HARNESS PASS (jim locks, {} clients)", sessions.len());
        return Ok(());
    }
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
    // Hard-coded tree tiles in zone (3,3), def=4 (the dense tree carrying `wood`;
    // def=5 sparse pine lacks it), distinct from the souls (alice@local(2,2),
    // bob@local(4,4)) and from each other:
    //   alice → local (3,2) = world (21,20), wood 2
    //   bob   → local (6,5) = world (24,23), wood 2 (kept far from alice's tile)
    let tiles = [(3u8, 2u8), (6u8, 5u8)];

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

    // ── Phase 5: each corpus_dim card triggers the root-only corpus_dim recipe
    //    (fire-at-once, no debounce — too fast to reliably catch "queued"), so we
    //    confirm it PROPOSED + was gate-accepted via its outcome. ≥1 per client. ─
    let mut dim_accepted = vec![0usize; sessions.len()];
    for _ in 0..16 {
        pump_all(sessions, Duration::from_millis(500)).await?;
        for (i, s) in sessions.iter_mut().enumerate() {
            for (r, err) in s.core.drain_action_outcomes() {
                if r == "corpus_dim" {
                    if let Some(e) = err {
                        bail!("client {i}: corpus_dim REJECTED by the gate: {e}");
                    }
                    dim_accepted[i] += 1;
                }
            }
        }
        if dim_accepted.iter().all(|&n| n >= 1) {
            break;
        }
    }
    if dim_accepted.iter().any(|&n| n < 1) {
        bail!("corpus_dim recipe never proposed+accepted on both clients: {dim_accepted:?}");
    }
    println!("harness: ✓ both clients proposed + gate-accepted the corpus_dim recipe");

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

    // ── Phase 9: leaf-aware host/join stacking (test_dust + log) ──────────────
    // Each client now holds a loose log (cut_tree), a loose test_dust (loadout),
    // and loose corpus (corpus_dim returned them in Phase 8). Bit-fields:
    //   log       hosts top+bottom, joins hex+top
    //   test_dust hosts NOTHING, joins top+bottom → a leaf that caps a stack
    //   corpus    DEFAULT (hosts hex+top+bottom, joins top+bottom)
    // These moves stay client-local (private inventory, no foreign observers), so
    // they exercise the client's place()/plan_place resolver directly.
    let tdd = sessions[0].core.packed_def("test_dust").context("test_dust def")?;

    // alice (0) drops test_dust→log (forward: log hosts top, test_dust joins it).
    // bob   (1) drops log→test_dust (forward fails — test_dust hosts nothing — so
    // the drop INVERTS, re-rooting test_dust onto the stationary log). Both must
    // reach the SAME state: test_dust capping log's TOP stack, log the loose root.
    let mut td_on: Vec<(u32, u32)> = Vec::new(); // (log_id, test_dust_id) per client
    for (i, s) in sessions.iter_mut().enumerate() {
        let me = plans[i].me;
        let log = *owned_cards(s, me, logd, true)
            .first()
            .ok_or_else(|| anyhow!("client {i}: no loose log for stacking test"))?;
        let td = *owned_cards(s, me, tdd, true)
            .first()
            .ok_or_else(|| anyhow!("client {i}: no loose test_dust"))?;
        if i == 0 {
            s.core
                .place(td, Placement::Stack { parent_id: log, direction: STACK_DIR_UP })
                .map_err(|e| anyhow!("client {i}: forward test_dust→log rejected: {e}"))?;
        } else {
            s.core
                .place(log, Placement::Stack { parent_id: td, direction: STACK_DIR_UP })
                .map_err(|e| anyhow!("client {i}: invert log→test_dust rejected: {e}"))?;
        }
        let now = s.core.clock_ms();
        match s.core.world().cards.current(td, now).map(|c| c.micro()) {
            Some(Micro::Stacked { root, branch, index })
                if root == log && branch == STACK_DIR_UP && index == 0 => {}
            other => bail!(
                "client {i}: expected test_dust {td} on log {log} top (root={log}, branch={STACK_DIR_UP}, idx=0), got {other:?}"
            ),
        }
        if !matches!(s.core.world().cards.current(log, now).map(|c| c.micro()), Some(Micro::Loose { .. })) {
            bail!("client {i}: log {log} should remain the loose root after stacking");
        }
        td_on.push((log, td));
    }
    println!("harness: ✓ stacking forward≡invert — test_dust caps log's top on both clients (alice forward, bob drop-invert)");

    // Reject: a corpus onto the stacked test_dust must FAIL — test_dust hosts
    // nothing (forward fails) and can't re-root (it's a member now; no invert).
    for (i, s) in sessions.iter_mut().enumerate() {
        let me = plans[i].me;
        let (_, td) = td_on[i];
        let corpus = *owned_cards(s, me, cdef, true)
            .first()
            .ok_or_else(|| anyhow!("client {i}: no loose corpus for reject test"))?;
        if s.core.place(corpus, Placement::Stack { parent_id: td, direction: STACK_DIR_UP }).is_ok() {
            bail!("client {i}: corpus {corpus} onto stacked test_dust {td} should be rejected, but place() succeeded");
        }
    }
    println!("harness: ✓ reject — corpus onto a stacked test_dust is refused (leaf hosts nothing, member can't re-root)");

    // Leaf fallback (bob): a corpus onto the LOG asking for top can't extend the
    // top stack (its leaf test_dust hosts nothing), so it falls through to the
    // bottom stack, which the log itself hosts. (The reject above left the corpus
    // loose — it never mutated.)
    {
        let s = &mut sessions[1];
        let me = plans[1].me;
        let (log, _) = td_on[1];
        let corpus = *owned_cards(s, me, cdef, true)
            .first()
            .ok_or_else(|| anyhow!("bob: no loose corpus for fallback test"))?;
        s.core
            .place(corpus, Placement::Stack { parent_id: log, direction: STACK_DIR_UP })
            .map_err(|e| anyhow!("bob: corpus→log (top→bottom fallback) rejected: {e}"))?;
        let now = s.core.clock_ms();
        match s.core.world().cards.current(corpus, now).map(|c| c.micro()) {
            Some(Micro::Stacked { root, branch, index })
                if root == log && branch == STACK_DIR_DOWN && index == 0 => {}
            other => bail!(
                "bob: corpus {corpus} should fall to log {log} bottom (branch={STACK_DIR_DOWN}, idx=0), got {other:?}"
            ),
        }
    }
    println!("harness: ✓ leaf fallback — corpus asking for log's capped top lands on its bottom stack instead");

    for s in sessions.iter_mut() {
        let errors = s.core.drain_errors();
        if !errors.is_empty() {
            bail!("[{}] {} protocol/decode error(s): {errors:?}", s.name, errors.len());
        }
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// jim — stack lock / splice / queue-cancel test (alice & bob present for isolation)
// ────────────────────────────────────────────────────────────────────────────

/// jim drives the whole scenario on his own private inventory chain while alice &
/// bob sit far out of range. Built incrementally; J1 is the queue/cancel dance.
async fn run_jim_lock_test(
    sessions: &mut [Session],
    names: &[String],
    humans: &[u32],
) -> anyhow::Result<()> {
    use anyhow::{anyhow, bail};

    let jim = names.iter().position(|n| n == "jim").ok_or_else(|| anyhow!("no jim session"))?;
    let me = sessions[jim].player_id().unwrap_or(0);
    let cdef = sessions[jim].core.packed_def("corpus").ok_or_else(|| anyhow!("corpus def"))?;

    pump_all(sessions, Duration::from_secs(3)).await?;

    // ── Isolation: jim is past alice/bob's cold radius, so neither side sees the
    //    other's world soul. ────────────────────────────────────────────────────
    let now = sessions[jim].core.clock_ms();
    for (i, &h) in humans.iter().enumerate() {
        if i == jim {
            continue;
        }
        if sessions[jim].core.world().cards.current(h, now).is_some() {
            bail!("isolation broken: jim sees client {i}'s world soul {h}");
        }
        let jn = sessions[i].core.clock_ms();
        if sessions[i].core.world().cards.current(humans[jim], jn).is_some() {
            bail!("isolation broken: client {i} sees jim's world soul {}", humans[jim]);
        }
    }
    println!("harness: ✓ jim isolated — no shared visibility with alice/bob");

    // jim's three loose corpus A,B,C + the inventory zone they live in.
    let corpus = owned_cards(&sessions[jim], me, cdef, true);
    if corpus.len() < 3 {
        bail!("jim: expected 3 loose corpus, got {}", corpus.len());
    }
    let (a, b, c) = (corpus[0], corpus[1], corpus[2]);
    let inv = sessions[jim]
        .core
        .world()
        .cards
        .current(a, now)
        .map(|x| x.macro_zone)
        .ok_or_else(|| anyhow!("jim: corpus A not in world"))?;

    // ── J1: queue/cancel dance — each recipe is canceled (superseded or dropped)
    //    before its 5s debounce fires, so the whole phase emits NOTHING. Exercises
    //    promote→triple_corpus sliding + drag-carry (moving A carries C). ─────────
    macro_rules! q {
        ($r:expr) => {
            has_queued(&sessions[jim], $r)
        };
    }

    // 1. A onto B → corpus_b_top queues (promote B → [B,A]).
    sessions[jim]
        .core
        .place(a, Placement::Stack { parent_id: b, direction: STACK_DIR_UP })
        .map_err(|e| anyhow!("J1.1 stack A→B: {e}"))?;
    pump_all(sessions, Duration::from_millis(1500)).await?;
    if !q!("corpus_b_top") {
        bail!("J1.1: corpus_b_top not queued after A→B; queue={:?}", sessions[jim].core.queued());
    }

    // 2. C onto A → 3 corpus in B's chain (B,A,C) → triple_corpus supersedes.
    sessions[jim]
        .core
        .place(c, Placement::Stack { parent_id: a, direction: STACK_DIR_UP })
        .map_err(|e| anyhow!("J1.2 stack C→A: {e}"))?;
    pump_all(sessions, Duration::from_millis(1500)).await?;
    if !q!("triple_corpus") || q!("corpus_b_top") {
        bail!("J1.2: expected triple_corpus only; queue={:?}", sessions[jim].core.queued());
    }

    // 3. Move A off → drag-carries C; A becomes a loose root with C on top, B alone.
    //    corpus_b_top re-queues on [A,C]; triple_corpus cancels.
    sessions[jim]
        .core
        .place(a, Placement::Loose { surface: INVENTORY_LAYER, macro_zone: inv, q: 0, r: 0, x: 0, y: 0 })
        .map_err(|e| anyhow!("J1.3 move A→inv: {e}"))?;
    pump_all(sessions, Duration::from_millis(1500)).await?;
    {
        let n = sessions[jim].core.clock_ms();
        match sessions[jim].core.world().cards.current(c, n).map(|x| x.micro()) {
            Some(Micro::Stacked { root, .. }) if root == a => {}
            other => bail!("J1.3: C should be drag-carried onto A, got {other:?}"),
        }
    }
    if !q!("corpus_b_top") || q!("triple_corpus") {
        bail!("J1.3: expected corpus_b_top on [A,C]; queue={:?}", sessions[jim].core.queued());
    }

    // 4. Move C off A → corpus_b_top cancels (A, B, C all loose singletons).
    sessions[jim]
        .core
        .place(c, Placement::Loose { surface: INVENTORY_LAYER, macro_zone: inv, q: 0, r: 0, x: 0, y: 0 })
        .map_err(|e| anyhow!("J1.4 move C→inv: {e}"))?;
    pump_all(sessions, Duration::from_millis(1500)).await?;
    if q!("corpus_b_top") || q!("triple_corpus") {
        bail!("J1.4: queue should be empty; queue={:?}", sessions[jim].core.queued());
    }

    let outs = sessions[jim].core.drain_action_outcomes();
    if !outs.is_empty() {
        bail!("J1: queue dance must emit nothing, got {outs:?}");
    }
    println!("harness: ✓ J1 queue dance — corpus_b_top→triple_corpus→corpus_b_top→∅, drag-carry verified, zero outputs");

    // ── J2: build corpus1's stack, then run corpus_dust + corpus_b_top together ──
    // a,b,c are loose again after J1; dust + the two axes are still loose.
    use resonantdust_codec::card_model::stack_index;
    let ddef = sessions[jim].core.packed_def("dust").ok_or_else(|| anyhow!("dust def"))?;
    let axdef = sessions[jim].core.packed_def("axe").ok_or_else(|| anyhow!("axe def"))?;
    let fdef = sessions[jim].core.packed_def("food").ok_or_else(|| anyhow!("food def"))?;
    let cdimdef = sessions[jim].core.packed_def("corpus_dim").ok_or_else(|| anyhow!("corpus_dim def"))?;
    let dust = *owned_cards(&sessions[jim], me, ddef, true).first().ok_or_else(|| anyhow!("no loose dust"))?;
    let axes = owned_cards(&sessions[jim], me, axdef, true);
    if axes.len() < 2 {
        bail!("jim: expected 2 loose axes, got {}", axes.len());
    }
    let (axe1, axe2) = (axes[0], axes[1]);

    // J2.1-3: axe1→a, axe2→axe1, dust→axe2 → root a, top stack [axe1,axe2,dust].
    for (card, parent, label) in [(axe1, a, "axe1→a"), (axe2, axe1, "axe2→axe1"), (dust, axe2, "dust→axe2")] {
        sessions[jim]
            .core
            .place(card, Placement::Stack { parent_id: parent, direction: STACK_DIR_UP })
            .map_err(|e| anyhow!("J2 stack {label}: {e}"))?;
        pump_all(sessions, Duration::from_millis(800)).await?;
    }
    if !q!("corpus_dust") {
        bail!("J2.3: corpus_dust not queued (root corpus + dust slid in top); queue={:?}", sessions[jim].core.queued());
    }
    println!("harness: ✓ J2 corpus_dust queued — root corpus + dust matched anywhere in the top stack");

    // J2.4: corpus2(b)→dust, corpus3(c)→b → top [axe1,axe2,dust,b,c]; corpus_b_top
    // (the adjacent b,c, found by sliding) supersedes corpus_dust.
    sessions[jim]
        .core
        .place(b, Placement::Stack { parent_id: dust, direction: STACK_DIR_UP })
        .map_err(|e| anyhow!("J2.4 stack b→dust: {e}"))?;
    pump_all(sessions, Duration::from_millis(800)).await?;
    sessions[jim]
        .core
        .place(c, Placement::Stack { parent_id: b, direction: STACK_DIR_UP })
        .map_err(|e| anyhow!("J2.4 stack c→b: {e}"))?;
    pump_all(sessions, Duration::from_millis(1000)).await?;
    if !q!("corpus_b_top") {
        bail!("J2.4: corpus_b_top not queued (b,c adjacent deep in top); queue={:?}", sessions[jim].core.queued());
    }
    println!("harness: ✓ J2 corpus_b_top queued — two adjacent corpus matched deep in the top stack");

    // J2.5: let both fire + complete. corpus_b_top (binds 2 > 1) fires first without
    // holding the root, destroys corpus2, splice collapses corpus3; corpus_dust
    // re-queues alongside (root not held) and yields food. Pump until both land.
    let mut done = false;
    for _ in 0..140 {
        pump_all(sessions, Duration::from_millis(500)).await?;
        for (r, err) in sessions[jim].core.drain_action_outcomes() {
            if let Some(e) = err {
                bail!("J2.5: {r} REJECTED by the gate: {e}");
            }
        }
        let n = sessions[jim].core.clock_ms();
        let b_dead = sessions[jim].core.world().cards.current(b, n).map(|x| x.is_dead()).unwrap_or(false);
        let food = !owned_cards(&sessions[jim], me, fdef, false).is_empty();
        if b_dead && food {
            done = true;
            break;
        }
    }
    if !done {
        bail!("J2.5: corpus_b_top + corpus_dust did not both complete (corpus2 dead + food)");
    }
    let n = sessions[jim].core.clock_ms();
    let didx = sessions[jim].core.world().cards.current(dust, n).map(|x| stack_index(x.flags));
    let cidx = sessions[jim].core.world().cards.current(c, n).map(|x| stack_index(x.flags));
    match (didx, cidx) {
        (Some(d), Some(cc)) if cc == d + 1 => {}
        _ => bail!("J2.5: splice wrong — dust idx={didx:?}, corpus3 idx={cidx:?} (want corpus3 = dust+1)"),
    }
    if owned_cards(&sessions[jim], me, cdimdef, false).is_empty() {
        bail!("J2.5: corpus_b_top yielded no corpus_dim");
    }
    println!(
        "harness: ✓ J2 concurrent corpus_b_top + corpus_dust — corpus2 destroyed, corpus3 spliced to dust+1 (idx {}→{}), corpus_dim + food landed",
        didx.unwrap_or(0) + 1,
        cidx.unwrap_or(0)
    );

    // ── J3: position-lock + the axe1-move drag-carry anchored by the locked dust ─
    use resonantdust_codec::card_model::{hold_count, HoldField};

    // J3a: corpus_dust self-advances — dust stays in the top stack, so it re-queues
    //   and produces another food.
    let food0 = owned_cards(&sessions[jim], me, fdef, false).len();
    let mut more = false;
    for _ in 0..90 {
        pump_all(sessions, Duration::from_millis(500)).await?;
        if owned_cards(&sessions[jim], me, fdef, false).len() > food0 {
            more = true;
            break;
        }
    }
    if !more {
        bail!("J3a: corpus_dust did not re-queue for another food");
    }
    println!("harness: ✓ J3 corpus_dust self-advances — re-queued + produced another food");

    // J3b: catch corpus_dust mid-run — dust is `claim`ed (position-locked); corpus1
    //   is `use`d (slot-held, NOT position-locked).
    let mut running = false;
    for _ in 0..80 {
        pump_all(sessions, Duration::from_millis(250)).await?;
        let n = sessions[jim].core.clock_ms();
        let df = sessions[jim].core.world().cards.current(dust, n).map(|x| x.flags).unwrap_or(0);
        if hold_count(df, HoldField::PositionHold) > 0 {
            running = true;
            break;
        }
    }
    if !running {
        bail!("J3b: never caught corpus_dust holding dust");
    }
    let n = sessions[jim].core.clock_ms();
    let af = sessions[jim].core.world().cards.current(a, n).map(|x| x.flags).unwrap_or(0);
    if hold_count(af, HoldField::PositionHold) != 0 {
        bail!("J3b: corpus1 must NOT be position-locked (it is `use`d); flags={af:#x}");
    }
    if hold_count(af, HoldField::SlotClaim) == 0 {
        bail!("J3b: corpus1 must be slot-held (used) by corpus_dust; flags={af:#x}");
    }
    println!("harness: ✓ J3 holds — dust position-locked (claim), corpus1 used not position-locked");

    // J3c: move axe1 off mid-run. Drag-carries axe2; the position-locked dust
    //   terminates the run, so dust + corpus3 stay and collapse (dust 2→0,
    //   corpus3→1). corpus1 is held by the running corpus_dust, so the index shift
    //   queues NO new proposal.
    sessions[jim]
        .core
        .place(axe1, Placement::Loose { surface: INVENTORY_LAYER, macro_zone: inv, q: 0, r: 0, x: 0, y: 0 })
        .map_err(|e| anyhow!("J3c move axe1→inv: {e}"))?;
    {
        let n = sessions[jim].core.clock_ms();
        let w = sessions[jim].core.world();
        let m_axe1 = w.cards.current(axe1, n).map(|x| x.micro());
        let m_axe2 = w.cards.current(axe2, n).map(|x| x.micro());
        let m_dust = w.cards.current(dust, n).map(|x| x.micro());
        let m_c = w.cards.current(c, n).map(|x| x.micro());
        if !matches!(m_axe1, Some(Micro::Loose { .. })) {
            bail!("J3c: axe1 should be a loose root, got {m_axe1:?}");
        }
        match m_axe2 {
            Some(Micro::Stacked { root, index, .. }) if root == axe1 && index == 0 => {}
            other => bail!("J3c: axe2 should be carried onto axe1 top@0, got {other:?}"),
        }
        match (m_dust, m_c) {
            (Some(Micro::Stacked { root: dr, index: di, .. }), Some(Micro::Stacked { root: cr, index: ci, .. }))
                if dr == a && cr == a && di == 0 && ci == 1 => {}
            (d, cc) => bail!("J3c: expected dust@corpus1.top0 + corpus3@corpus1.top1, got dust={d:?} corpus3={cc:?}"),
        }
    }
    // The index shift (corpus1 still held by the running corpus_dust) re-proposes nothing.
    pump_all(sessions, Duration::from_millis(1000)).await?;
    if q!("corpus_dust") {
        bail!("J3c: dust's index shift must not re-propose corpus_dust mid-run; queue={:?}", sessions[jim].core.queued());
    }
    println!("harness: ✓ J3 axe1 carried axe2 off; locked dust anchored, index-shifted 2→0 + corpus3→1; no re-proposal mid-run");

    // ── J3d: when the running corpus_dust completes it emits dust's row at the
    //   GATE's index (2). Our pending local move (dirty_position) + the `pos_want`/
    //   `!pos_need` rule must keep our predicted index (0). Then moving dust off the
    //   stack stops the loop. ────────────────────────────────────────────────────
    let food_b = owned_cards(&sessions[jim], me, fdef, false).len();
    let mut completed = false;
    for _ in 0..120 {
        pump_all(sessions, Duration::from_millis(500)).await?;
        for (r, err) in sessions[jim].core.drain_action_outcomes() {
            if let Some(e) = err {
                bail!("J3d: {r} REJECTED: {e}");
            }
        }
        if owned_cards(&sessions[jim], me, fdef, false).len() > food_b {
            completed = true;
            break;
        }
    }
    if !completed {
        bail!("J3d: corpus_dust did not complete + re-propose for another food");
    }
    let n = sessions[jim].core.clock_ms();
    match sessions[jim].core.world().cards.current(dust, n).map(|x| x.micro()) {
        Some(Micro::Stacked { root, index, .. }) if root == a && index == 0 => {}
        other => bail!("J3d: dust's local index must survive the gate's completion row (want corpus1.top0), got {other:?}"),
    }
    println!("harness: ✓ J3 reconciliation — corpus_dust completed; dust kept its local index 0 over the gate's row (dirty_position + pos_want)");

    // Move dust off when it's free → no dust in corpus1's top → corpus_dust stops.
    let mut moved = false;
    for _ in 0..60 {
        let n = sessions[jim].core.clock_ms();
        let df = sessions[jim].core.world().cards.current(dust, n).map(|x| x.flags).unwrap_or(0);
        if hold_count(df, HoldField::PositionHold) == 0
            && sessions[jim]
                .core
                .place(dust, Placement::Loose { surface: INVENTORY_LAYER, macro_zone: inv, q: 0, r: 0, x: 0, y: 0 })
                .is_ok()
        {
            moved = true;
            break;
        }
        pump_all(sessions, Duration::from_millis(300)).await?;
    }
    if !moved {
        bail!("J3d: never caught dust free to move it off");
    }
    for _ in 0..12 {
        pump_all(sessions, Duration::from_millis(500)).await?;
    }
    if q!("corpus_dust") {
        bail!("J3d: corpus_dust still queued after dust left the stack; queue={:?}", sessions[jim].core.queued());
    }
    println!("harness: ✓ J3 dust moved off — corpus_dust stops; jim lock-test complete");

    Ok(())
}
