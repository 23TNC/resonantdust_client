//! Client world model — the local mirror of server state, fed by gate
//! subscription rows.
//!
//! The Rust analogue of the pixijs client's `DataManager` server tier: a
//! bitemporal store that holds every version-row the gate streams for a card and
//! resolves the one *current at a given wall-clock* on demand. Future-stamped
//! rows (recipe completions / movement queued ahead of now) are kept but excluded
//! from `current` until their time arrives — exactly the server's `prior_at`
//! discipline, so the headless client and the server agree on "what is true now."
//!
//! Transport-agnostic and renderer-free: [`World::ingest`] takes parsed
//! [`GateMsg`]s, so the whole path is exercised in tests with zero network.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use resonantdust_protocol::protocol::RowOp;

use crate::rows::{CardRow, ZoneRow};

/// Anchor-aware GC of one id's version history: keep every **future** row (not
/// yet promoted), the **current-at-`now`** row, and the **current-as-of** each
/// frozen watermark `pin` (a soul's remembered moment); reap the rest. `time_of`
/// reads a row's `time_ms`. Single-version histories are left untouched.
fn gc_history<R>(hist: &mut BTreeMap<u64, R>, now: u64, pins: &[u64], time_of: impl Fn(&R) -> u64) {
    if hist.len() <= 1 {
        return;
    }
    let mut keep: HashSet<u64> = HashSet::new();
    for (valid_at, r) in hist.iter() {
        if time_of(r) > now {
            keep.insert(*valid_at); // future row — promotes later
        }
    }
    // current-as-of(now) plus current-as-of each pin time.
    for &t in std::iter::once(&now).chain(pins.iter()) {
        if let Some(valid_at) = hist
            .iter()
            .filter(|(_, r)| time_of(r) <= t)
            .max_by_key(|(_, r)| time_of(r))
            .map(|(va, _)| *va)
        {
            keep.insert(valid_at);
        }
    }
    hist.retain(|valid_at, _| keep.contains(valid_at));
}

/// Bitemporal card store: per `card_id`, the version rows keyed by `valid_at`
/// (PK ordering), with current-at-now resolution.
#[derive(Default)]
pub struct Cards {
    by_id: BTreeMap<u32, BTreeMap<u64, CardRow>>,
}

impl Cards {
    /// Fold one row event into the store. Insert/Update upsert the row at its
    /// `valid_at`; Delete drops that version (and the card if it was the last).
    pub fn apply(&mut self, op: RowOp, row: CardRow) {
        match op {
            RowOp::Insert | RowOp::Update => {
                self.by_id.entry(row.card_id).or_default().insert(row.valid_at, row);
            }
            RowOp::Delete => {
                let drained = match self.by_id.get_mut(&row.card_id) {
                    Some(hist) => {
                        hist.remove(&row.valid_at);
                        hist.is_empty()
                    }
                    None => false,
                };
                if drained {
                    self.by_id.remove(&row.card_id);
                }
            }
        }
    }

    /// The row for `card_id` current at `now_ms`: the max `time_ms` among rows
    /// stamped at or before `now_ms`. `None` if the card is unknown or all its
    /// rows are future-stamped.
    #[allow(dead_code)] // the per-card accessor NPC decision logic reads from
    pub fn current(&self, card_id: u32, now_ms: u64) -> Option<&CardRow> {
        current_of(self.by_id.get(&card_id)?, now_ms)
    }

    /// Overwrite the POSITION (`macro_zone` + `micro_location` + the placement bits
    /// of `flags`) of EVERY version of `card_id` — present and future — to the
    /// given values, preserving each version's other flags and `valid_at`. Anchors
    /// a pending LOCAL move so a future-stamped server row (e.g. a recipe's
    /// completion finalize, already in the store) can't promote and clobber the
    /// prediction. A version the server REQUIRES (`pos_need`) is left untouched.
    pub fn pin_position(&mut self, card_id: u32, macro_zone: u64, micro_location: u32, flags: u32) {
        use resonantdust_codec::card_model::{placement_mask, pos_need};
        let pmask = placement_mask();
        if let Some(hist) = self.by_id.get_mut(&card_id) {
            for row in hist.values_mut() {
                if pos_need(row.flags) {
                    continue;
                }
                row.macro_zone = macro_zone;
                row.micro_location = micro_location;
                row.flags = (row.flags & !pmask) | (flags & pmask);
            }
        }
    }

    /// Re-derive the `stock` of this card's LOCAL position rows (the `valid_at`s in
    /// `local`) from the authoritative server rows: each local row adopts the
    /// `stock` of the nearest server row at-or-before its own `time_ms`. The dual of
    /// [`pin_position`](Self::pin_position) — a local move overrides POSITION, but
    /// `stock` (holds — the op-log mirror) stays server truth. Without this, a local
    /// move row freezes a stale `stock` and shadows an in-flight hold that lands at
    /// an EARLIER stamp (the hold acquire is stamped at the recipe start), so
    /// `is_held` reads `claim=0` and the matcher re-queues the running recipe. A
    /// local row with no server row before it keeps its own `stock` (nothing to
    /// derive from).
    pub fn resync_local_stock(&mut self, card_id: u32, local: &BTreeSet<u64>) {
        let Some(hist) = self.by_id.get_mut(&card_id) else { return };
        // Server rows' (time_ms, stock) — every row NOT authored locally.
        let server: Vec<(u64, u64)> = hist
            .iter()
            .filter(|(vat, _)| !local.contains(vat))
            .map(|(_, r)| (r.time_ms(), r.stock))
            .collect();
        if server.is_empty() {
            return;
        }
        for (vat, row) in hist.iter_mut() {
            if !local.contains(vat) {
                continue;
            }
            let t = row.time_ms();
            if let Some((_, stock)) =
                server.iter().filter(|(st, _)| *st <= t).max_by_key(|(st, _)| *st)
            {
                row.stock = *stock;
            }
        }
    }

    /// Every card's current-at-`now_ms` row (skipping cards that are entirely
    /// future-stamped). Order follows `card_id`.
    pub fn current_all(&self, now_ms: u64) -> impl Iterator<Item = &CardRow> {
        self.by_id.values().filter_map(move |hist| current_of(hist, now_ms))
    }

    /// The card's earliest FUTURE-stamped row — the one with the smallest
    /// `time_ms` strictly after `now_ms`, or `None`. (No longer drives the build
    /// bar — that's [`progress_window`](Self::progress_window) — but kept as a
    /// generic "next promotion" primitive.)
    pub fn next_future_row(&self, card_id: u32, now_ms: u64) -> Option<&CardRow> {
        let hist = self.by_id.get(&card_id)?;
        hist.values().filter(|r| r.time_ms() > now_ms).min_by_key(|r| r.time_ms())
    }

    /// The `[start, end]` ms window of the first progress run that hasn't finished
    /// by `now_ms`, or `None`. `is_active` tests a row for the progress bit (the
    /// caller decodes `pstatus` against the def schema, which this store can't).
    ///
    /// A "run" is a maximal stretch of consecutive `is_active` rows; the recipe
    /// sets the bit at hold-acquire and clears it at completion, so the run's first
    /// row is the bar's START and the clearing row is its END. We return the first
    /// run whose END is still in the future, which uniformly covers:
    ///   - run straddling now (`start ≤ now < end`) → the live bar;
    ///   - run entirely ahead (`now < start`) → an upcoming bar the caller draws
    ///     empty (clamped) until the clock reaches it;
    /// and skips runs already finished (`end ≤ now`). A run that never closes
    /// (missing `end_bar`, or the completion row not yet streamed) falls back to
    /// the latest known row as END — degenerate (`total == 0` ⇒ caller hides it),
    /// which is the visible symptom of a recipe that set the bit without clearing.
    pub fn progress_window(
        &self,
        card_id: u32,
        now_ms: u64,
        is_active: impl Fn(&CardRow) -> bool,
    ) -> Option<(u64, u64)> {
        let hist = self.by_id.get(&card_id)?;
        // Iterates time-ascending (`valid_at` = `time_ms << 16 | seq`).
        let mut run_start: Option<u64> = None;
        for r in hist.values() {
            let t = r.time_ms();
            match (run_start, is_active(r)) {
                (None, true) => run_start = Some(t), // run opens
                (Some(s), false) => {
                    // Run closes here. First one still open at `now` wins.
                    if t > now_ms {
                        return Some((s, t));
                    }
                    run_start = None; // already finished — keep scanning
                }
                _ => {} // continuing a run / still idle
            }
        }
        // Open-ended tail run (no clearing row): end = latest known row.
        let s = run_start?;
        let last = hist.values().next_back()?.time_ms();
        (last > now_ms).then_some((s, last))
    }

    /// The earliest future-stamped `time_ms` across ALL cards, or `None` if every
    /// row is already current. Mirrors [`Zones::min_future_time`] — a future tile-card
    /// row (a `cut_tree` completion's decrement) fires no event when the clock crosses
    /// it, so the core watches this to kick a re-render/re-match then.
    pub fn min_future_time(&self, now_ms: u64) -> Option<u64> {
        self.by_id
            .values()
            .flat_map(|hist| hist.values())
            .map(|r| r.time_ms())
            .filter(|t| *t > now_ms)
            .min()
    }

    /// Anchor-aware GC: per card, keep its live + remembered rows, reap the rest.
    /// `pins_for_zone(zone)` gives the frozen card watermarks pinning that zone
    /// (from the zone manager). A card's zone is its current (else latest) row's
    /// `macro_zone`.
    pub fn gc(&mut self, now_ms: u64, mut pins_for_zone: impl FnMut(u64) -> Vec<u64>) {
        for hist in self.by_id.values_mut() {
            let zone = match current_of(hist, now_ms).or_else(|| hist.values().next_back()) {
                Some(r) => r.macro_zone,
                None => continue,
            };
            let pins = pins_for_zone(zone);
            gc_history(hist, now_ms, &pins, |r| r.time_ms());
        }
    }

    /// The `macro_zone` of a card's latest known row (current or stale) — the
    /// zone whose per-soul watermark gates the memory-view read of this card.
    pub fn zone_of(&self, card_id: u32) -> Option<u64> {
        self.by_id.get(&card_id)?.values().next_back().map(|r| r.macro_zone)
    }


    /// Number of distinct cards known (any version).
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// Total version-rows across all cards (GC diagnostics / tests).
    #[allow(dead_code)]
    pub fn version_count(&self) -> usize {
        self.by_id.values().map(|h| h.len()).sum()
    }

    #[allow(dead_code)] // used in tests; pairs with `len`
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// Current-at-`now_ms` row within one card's history.
fn current_of(hist: &BTreeMap<u64, CardRow>, now_ms: u64) -> Option<&CardRow> {
    hist.values()
        .filter(|r| r.time_ms() <= now_ms)
        .max_by_key(|r| r.time_ms())
}

/// Bitemporal zone store, keyed by `macro_zone` (the zone's location key) — the
/// soul's active zones live here. Same valid_at-history + current-at-now shape as
/// [`Cards`].
#[derive(Default)]
pub struct Zones {
    by_macro: BTreeMap<u64, BTreeMap<u64, ZoneRow>>,
}

impl Zones {
    pub fn apply(&mut self, op: RowOp, row: ZoneRow) {
        match op {
            RowOp::Insert | RowOp::Update => {
                self.by_macro.entry(row.macro_zone).or_default().insert(row.valid_at, row);
            }
            RowOp::Delete => {
                let drained = match self.by_macro.get_mut(&row.macro_zone) {
                    Some(h) => {
                        h.remove(&row.valid_at);
                        h.is_empty()
                    }
                    None => false,
                };
                if drained {
                    self.by_macro.remove(&row.macro_zone);
                }
            }
        }
    }

    /// The zone current at `now_ms` for `macro_zone`.
    pub fn current(&self, macro_zone: u64, now_ms: u64) -> Option<&ZoneRow> {
        let hist = self.by_macro.get(&macro_zone)?;
        hist.values().filter(|z| z.time_ms() <= now_ms).max_by_key(|z| z.time_ms())
    }

    /// Every zone's current-at-`now_ms` row (skipping zones that are entirely
    /// future-stamped). Order follows `macro_zone`. The render surface walks
    /// these to emit the tile grid.
    pub fn current_all(&self, now_ms: u64) -> impl Iterator<Item = &ZoneRow> {
        self.by_macro.values().filter_map(move |hist| {
            hist.values().filter(|z| z.time_ms() <= now_ms).max_by_key(|z| z.time_ms())
        })
    }

    /// The earliest `valid_at` time still in the FUTURE relative to `now_ms`
    /// across all zones — when the next zone will promote into the current view.
    /// `None` if every stored row is already current. A future-stamped zone row
    /// fires NO event when the clock later crosses its time, so the core watches
    /// this to know when to re-render (the stationary-load render kick).
    pub fn min_future_time(&self, now_ms: u64) -> Option<u64> {
        self.by_macro
            .values()
            .flat_map(|hist| hist.values())
            .map(|z| z.time_ms())
            .filter(|t| *t > now_ms)
            .min()
    }

    /// Anchor-aware GC, keyed by `macro_zone` (the zone IS its own key).
    pub fn gc(&mut self, now_ms: u64, mut pins_for_zone: impl FnMut(u64) -> Vec<u64>) {
        for (zone, hist) in self.by_macro.iter_mut() {
            let pins = pins_for_zone(*zone);
            gc_history(hist, now_ms, &pins, |r| r.time_ms());
        }
    }

    pub fn len(&self) -> usize {
        self.by_macro.len()
    }
}

/// The whole local world. Cards + zones; souls land alongside as the model
/// grows. Rows are routed in by the [`crate::client::Client`] core's `apply`,
/// which owns the gate-frame → world mapping.
#[derive(Default)]
pub struct World {
    pub cards: Cards,
    pub zones: Zones,
    /// Promoted tile-cards (regions-DB `cards`, surfaced by the gate as the
    /// `tile_cards` table), kept apart from `cards` so they never leak into
    /// player-card enumeration / rendering. They carry the live per-cell stock
    /// (a `cut_tree`'s decrement, an in-flight hold) that the bare zone slot only
    /// catches up to on GC demotion — read with card-priority via [`Self::tile_card_at`].
    pub tile_cards: Cards,
}

impl World {
    /// A promoted tile-card's `(packed_definition, stock0, stock1)` current at
    /// `now_ms` for the loose cell `(q, r)` of `macro_zone`, or `None` if none is
    /// promoted there. Feeds the shared [`resonantdust_codec::packed::synthetic_tile`]
    /// card-priority rule (mirrors the gate's `latest_tile_card_at`), but
    /// time-aware: a future-stamped completion row stays hidden until its time.
    pub fn tile_card_at(&self, macro_zone: u64, q: u8, r: u8, now_ms: u64) -> Option<(u16, u8, u8)> {
        use resonantdust_codec::card_model::{micro_is_card, stock, Micro};
        self.tile_cards
            .current_all(now_ms)
            .find(|c| {
                c.macro_zone == macro_zone
                    && !micro_is_card(c.flags)
                    && matches!(
                        Micro::of(c.micro_location, c.flags),
                        Micro::Loose { local_q, local_r, .. } if local_q == q && local_r == r
                    )
            })
            .map(|c| (c.packed_definition, stock(c.stock, 0), stock(c.stock, 1)))
    }
}

// The world is a `StackStore` so the shared `stack::plan_place` runs against it
// client-side — the exact same validation/resolution the shard runs, for
// predicted moves. `card_at` / `members_of` read the current-at-now rows.
impl resonantdust_state::recipe_state::CardStore for World {
    fn card_at(&self, card_id: u32, time_ms: u64) -> Option<resonantdust_state::recipe_state::CardView> {
        self.cards.current(card_id, time_ms).map(card_view)
    }
}

impl resonantdust_state::stack::StackStore for World {
    fn members_of(&self, root_id: u32, now_ms: u64) -> Vec<resonantdust_state::recipe_state::CardView> {
        use resonantdust_codec::card_model::Micro;
        self.cards
            .current_all(now_ms)
            .filter(|r| {
                matches!(Micro::of(r.micro_location, r.flags), Micro::Stacked { root, .. } if root == root_id)
            })
            .map(card_view)
            .collect()
    }

    // The synthetic tile at a cell — the virtual hex member a card seated here
    // would mount. Card-priority via the shared rule: a promoted tile-card's live
    // view wins over the zone's packed grid slot, exactly as the gate matcher does,
    // so the client can't drift. `None` when the cell is empty (def 0) or unloaded.
    fn tile_at(
        &self,
        macro_zone: u64,
        q: u8,
        r: u8,
        now_ms: u64,
    ) -> Option<resonantdust_state::recipe_state::CardView> {
        use resonantdust_codec::card_model::write_stock;
        let zone = self.zones.current(macro_zone, now_ms)?;
        let (packed_definition, s0, s1) = resonantdust_codec::packed::synthetic_tile(
            self.tile_card_at(macro_zone, q, r, now_ms),
            &zone.tile_words(),
            zone.tile_card_type(),
            q,
            r,
        )?;
        Some(resonantdust_state::recipe_state::CardView {
            card_id: 0,
            owner_id: 0,
            micro_location: 0,
            macro_zone,
            packed_definition,
            flags: 0,
            stock: write_stock(write_stock(0, 0, s0), 1, s1),
        })
    }

    // The client tracks terrain (zone tile grids + promoted tile-cards), so a `None`
    // from `tile_at` is authoritative: the cell is genuinely un-tiled. This is what
    // makes a loose drop onto a non-existent tile reject (the player can't place past
    // the map / inventory disk) — the shard/gate stay tile-blind and trust the move.
    fn tracks_tiles(&self) -> bool {
        true
    }
}

/// View a stored row as the shared model's `CardView`.
fn card_view(r: &CardRow) -> resonantdust_state::recipe_state::CardView {
    resonantdust_state::recipe_state::CardView {
        card_id: r.card_id,
        owner_id: r.owner_id,
        micro_location: r.micro_location,
        macro_zone: r.macro_zone,
        packed_definition: r.packed_definition,
        flags: r.flags,
        stock: r.stock,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resonantdust_codec::card_model::Micro;
    use resonantdust_codec::packed::{pack_valid_at, STACK_DIR_UP};

    /// Build a loose card row at `(time_ms)` for `card_id` with a snapped hex
    /// placement (the common world-tile shape), via the codec so the flag bits
    /// are laid out exactly as the server would.
    fn loose_row(card_id: u32, time_ms: u64, owner_id: u32) -> CardRow {
        let (micro_location, flags) = Micro::snap(0, 0).apply(0);
        CardRow {
            valid_at: pack_valid_at(time_ms, 1),
            card_id,
            macro_zone: 0,
            micro_location,
            owner_id,
            packed_definition: 0,
            flags,
            flags_bk: 0,
            stock: 0,
        }
    }

    #[test]
    fn gc_reaps_unpinned_but_keeps_current_future_and_pinned() {
        let mut cards = Cards::default();
        for t in [100u64, 200, 300, 400] {
            cards.apply(RowOp::Insert, loose_row(1024, t, 7));
        }
        assert_eq!(cards.version_count(), 4);
        // now=250, no memory pins: keep current(200) + future(300,400); reap 100.
        cards.gc(250, |_zone| vec![]);
        assert_eq!(cards.version_count(), 3);
        assert!(cards.current(1024, 150).is_none(), "unpinned old version reaped");
        assert_eq!(cards.current(1024, 250).unwrap().time_ms(), 200);

        // Re-add the old version; a frozen watermark at t=150 pins it as memory.
        cards.apply(RowOp::Insert, loose_row(1024, 100, 7));
        assert_eq!(cards.version_count(), 4);
        cards.gc(250, |_zone| vec![150]); // current-as-of-150 = t=100 → retained
        assert_eq!(cards.version_count(), 4, "pinned old version retained as memory");
    }

    #[test]
    fn next_future_row_brackets_the_completion_window() {
        let mut cards = Cards::default();
        // current row at 100, a future completion finalize stamped at 400.
        cards.apply(RowOp::Insert, loose_row(1024, 100, 7));
        cards.apply(RowOp::Insert, loose_row(1024, 400, 7));
        // at now=250: the start is the current row (100), the end is the next
        // future row (400) → a 300ms window with 150ms left.
        assert_eq!(cards.next_future_row(1024, 250).map(|r| r.time_ms()), Some(400));
        assert_eq!(cards.current(1024, 250).unwrap().time_ms(), 100);
        // no future row → no window.
        assert!(cards.next_future_row(1024, 500).is_none());
        // unknown card → None.
        assert!(cards.next_future_row(9999, 250).is_none());
    }

    // Build a row whose `pstatus` channel-0 bit (stock bit 0) is on/off — the test
    // stand-in for `start_bar`/`end_bar` writes, decoded here as `stock & 1`.
    fn bar_row(t: u64, on: bool) -> CardRow {
        let mut r = loose_row(1024, t, 7);
        r.stock = on as u64;
        r
    }
    const BAR_ON: fn(&CardRow) -> bool = |r| r.stock & 1 != 0;

    #[test]
    fn progress_window_scans_the_bit_interval() {
        let mut cards = Cards::default();
        // bit set at hold-acquire (100), cleared at completion (400).
        cards.apply(RowOp::Insert, bar_row(100, true));
        cards.apply(RowOp::Insert, bar_row(400, false));
        // live: now inside [100,400) → the full window.
        assert_eq!(cards.progress_window(1024, 250, BAR_ON), Some((100, 400)));
        // upcoming: now before start → still [100,400] (caller draws it empty).
        assert_eq!(cards.progress_window(1024, 50, BAR_ON), Some((100, 400)));
        // finished: now at/after the clear → no window.
        assert_eq!(cards.progress_window(1024, 400, BAR_ON), None);
        assert_eq!(cards.progress_window(1024, 500, BAR_ON), None);
        // unknown card → None.
        assert_eq!(cards.progress_window(9999, 250, BAR_ON), None);
    }

    #[test]
    fn progress_window_skips_finished_runs() {
        let mut cards = Cards::default();
        // run [100,200] finishes; a second run opens at 300, clears at 600.
        for (t, on) in [(100, true), (200, false), (300, true), (600, false)] {
            cards.apply(RowOp::Insert, bar_row(t, on));
        }
        // now=400: first run already done (end 200) → the live one is [300,600].
        assert_eq!(cards.progress_window(1024, 400, BAR_ON), Some((300, 600)));
        // now=250: first run finished, second still upcoming.
        assert_eq!(cards.progress_window(1024, 250, BAR_ON), Some((300, 600)));
    }

    #[test]
    fn progress_window_open_run_falls_back_to_last_row() {
        // Missing `end_bar`: the bit never clears, so END falls back to the latest
        // known row — a degenerate bar that never completes once that row is past.
        let mut cards = Cards::default();
        cards.apply(RowOp::Insert, bar_row(100, true));
        cards.apply(RowOp::Insert, bar_row(400, true));
        assert_eq!(cards.progress_window(1024, 250, BAR_ON), Some((100, 400)));
        assert_eq!(cards.progress_window(1024, 400, BAR_ON), None);
    }

    #[test]
    fn current_resolves_to_latest_at_or_before_now() {
        let mut cards = Cards::default();
        cards.apply(RowOp::Insert, loose_row(1024, 100, 7));
        cards.apply(RowOp::Insert, loose_row(1024, 200, 7));
        // Before any row exists.
        assert!(cards.current(1024, 50).is_none());
        // Between the two: the t=100 row.
        assert_eq!(cards.current(1024, 150).unwrap().time_ms(), 100);
        // At/after the second: the t=200 row.
        assert_eq!(cards.current(1024, 250).unwrap().time_ms(), 200);
        // One card, two versions.
        assert_eq!(cards.len(), 1);
    }

    #[test]
    fn future_stamped_rows_are_excluded_until_their_time() {
        let mut cards = Cards::default();
        cards.apply(RowOp::Insert, loose_row(1024, 100, 7));
        cards.apply(RowOp::Insert, loose_row(1024, 500, 7)); // future completion
        assert_eq!(cards.current(1024, 300).unwrap().time_ms(), 100);
        assert_eq!(cards.current(1024, 600).unwrap().time_ms(), 500);
    }

    #[test]
    fn delete_drops_version_then_card() {
        let mut cards = Cards::default();
        let r1 = loose_row(1024, 100, 7);
        let r2 = loose_row(1024, 200, 7);
        let (v1, v2) = (r1.valid_at, r2.valid_at);
        cards.apply(RowOp::Insert, r1);
        cards.apply(RowOp::Insert, r2);
        // Reap the older version (GC sweep) — newer stays current.
        cards.apply(RowOp::Delete, with_valid_at(loose_row(1024, 100, 7), v1));
        assert_eq!(cards.current(1024, 250).unwrap().time_ms(), 200);
        // Reap the last version — the card disappears.
        cards.apply(RowOp::Delete, with_valid_at(loose_row(1024, 200, 7), v2));
        assert!(cards.current(1024, 250).is_none());
        assert!(cards.is_empty());
    }

    #[test]
    fn placement_decodes_through_codec() {
        // A stacked member of root 2048 on the UP branch at slot 3.
        let (micro_location, flags) =
            Micro::Stacked { root: 2048, branch: STACK_DIR_UP, index: 3 }.apply(0);
        let row = CardRow {
            valid_at: pack_valid_at(100, 1),
            card_id: 1025,
            macro_zone: 0,
            micro_location,
            owner_id: 0,
            packed_definition: 0,
            flags,
            flags_bk: 0,
            stock: 0,
        };
        match row.micro() {
            Micro::Stacked { root, branch, index } => {
                assert_eq!((root, branch, index), (2048, STACK_DIR_UP, 3));
            }
            other => panic!("expected stacked, got {other:?}"),
        }
    }

}

/// Test helper: clone a row with a specific `valid_at` (so a Delete event can
/// name the exact version row to reap). A free fn, not an inherent method —
/// `CardRow` is defined in the protocol crate now (orphan rule).
#[cfg(test)]
fn with_valid_at(mut row: CardRow, valid_at: u64) -> CardRow {
    row.valid_at = valid_at;
    row
}
