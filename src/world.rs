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

use std::collections::{BTreeMap, HashSet};

use resonantdust_data::protocol::RowOp;

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

    /// Every card's current-at-`now_ms` row (skipping cards that are entirely
    /// future-stamped). Order follows `card_id`.
    pub fn current_all(&self, now_ms: u64) -> impl Iterator<Item = &CardRow> {
        self.by_id.values().filter_map(move |hist| current_of(hist, now_ms))
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
}

// The world is a `StackStore` so the shared `stack::plan_place` runs against it
// client-side — the exact same validation/resolution the shard runs, for
// predicted moves. `card_at` / `members_of` read the current-at-now rows.
impl resonantdust_data::recipe_state::CardStore for World {
    fn card_at(&self, card_id: u32, time_ms: u64) -> Option<resonantdust_data::recipe_state::CardView> {
        self.cards.current(card_id, time_ms).map(card_view)
    }
}

impl resonantdust_data::stack::StackStore for World {
    fn members_of(&self, root_id: u32, now_ms: u64) -> Vec<resonantdust_data::recipe_state::CardView> {
        use resonantdust_data::card_model::Micro;
        self.cards
            .current_all(now_ms)
            .filter(|r| {
                matches!(Micro::of(r.micro_location, r.flags), Micro::Stacked { root, .. } if root == root_id)
            })
            .map(card_view)
            .collect()
    }
}

/// View a stored row as the shared model's `CardView`.
fn card_view(r: &CardRow) -> resonantdust_data::recipe_state::CardView {
    resonantdust_data::recipe_state::CardView {
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
    use resonantdust_data::card_model::Micro;
    use resonantdust_data::packed::{pack_valid_at, STACK_DIR_UP};

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
        cards.apply(RowOp::Delete, loose_row(1024, 100, 7).with_valid_at(v1));
        assert_eq!(cards.current(1024, 250).unwrap().time_ms(), 200);
        // Reap the last version — the card disappears.
        cards.apply(RowOp::Delete, loose_row(1024, 200, 7).with_valid_at(v2));
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

#[cfg(test)]
impl CardRow {
    /// Test helper: clone with a specific `valid_at` (so a Delete event can name
    /// the exact version row to reap).
    fn with_valid_at(mut self, valid_at: u64) -> Self {
        self.valid_at = valid_at;
        self
    }
}
