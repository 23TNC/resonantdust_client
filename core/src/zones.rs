//! Zone manager — aspect-driven anchor tiers + a **sticky subscription manager**.
//!
//! A zone is a `(surface, owner, chunk_q, chunk_r)` cell, keyed by its packed
//! `macro_zone` u64. Each **anchor** (a card carrying the `anchor_*` aspects —
//! typically a soul) projects four nested tile-radii (active ⊃ hot ⊃ warm ⊃
//! cold); every chunk a disk covers gets the tightest tier any anchor assigns it
//! (max-tier-wins). Geometry is **range-disk → covered-chunk-set** so a soul deep
//! in a chunk lights one zone, near an edge only the chunks it overlaps.
//!
//! ## Subscriptions are sticky (Phase B)
//!
//! Two subscription **data types** per zone with graded reach:
//!   - `Card` — held while the zone is within `anchor_hot` of some anchor.
//!   - `Zone` — held while within `anchor_cold`.
//! A zone's tier is the "wanted" signal (the max over anchors *is* the holder
//! refcount): a `Card` sub is wanted at tier Active/Hot, a `Zone` sub whenever
//! the zone is tracked. When an anchor moves away a sub does **not** close — it
//! becomes a **close-candidate** (open but unwanted) and stays subscribed until
//! either **warmth** ([`should_close`], run on each inbound update — a silent
//! zone never triggers it) or **capacity-LRU** ([`enforce_capacity`]) evicts it.
//! Network re-transmit is the binding cost, so we hold generously and drop only
//! the noisy or over-budget. There is no distance release horizon.
//!
//! The `Zone`-sub is region-gated (ensure_region / request_zone materializes the
//! world zone server-side); the `Card`-sub is not (cards stream regardless).
//!
//! Sans-IO: decisions become [`ZoneIntent`]s the driver drains and maps to gate
//! frames. `now` (the server clock, ms) is threaded in for warmth recency.
//!
//! Phase C (memory) layers per-soul watermarks + anchor-aware GC on top; the
//! `Warm` tier's critical-card sub + off-attention tasks are Phase E.

use std::collections::{HashMap, HashSet};

use resonantdust_codec::packed::{
    owner_of, pack_macro_zone_full, region_of_zone, surface_of, unpack_macro_zone, zone_local,
    INVENTORY_LAYER,
};

/// Subscription/memory tier for a tracked zone, in nested distance bands (active
/// tightest). Tier = which band the zone falls in for the closest-binding anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneTier {
    Active,
    Hot,
    Warm,
    Cold,
}

fn tier_rank(t: ZoneTier) -> u8 {
    match t {
        ZoneTier::Active => 3,
        ZoneTier::Hot => 2,
        ZoneTier::Warm => 1,
        ZoneTier::Cold => 0,
    }
}

/// A subscription data type for a zone. `Card` streams the zone's cards; `Zone`
/// streams its tile grid (region-gated).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    Card,
    Zone,
}

/// A decision the manager has made that needs IO. The driver drains these (via
/// [`ZoneManager::take_intents`]) and maps them to gate frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneIntent {
    /// Open (`on = true`) or close (`on = false`) a zone's `data` subscription.
    Subscribe { zone: u64, data: DataType, on: bool },
    /// A region must be subscribed / released (held while a `Zone` sub inside it
    /// is open).
    Region { region: u64, subscribed: bool },
}

/// What a wanted zone needs to materialize, per the region's server-truth bits.
/// The driver reconciles these into idempotent `ensure_region` / `request_zone`
/// calls through the `Outbox` — there is no client-side request dedup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZoneNeed {
    /// No governing region yet — `ensure_region` (call keyed on the region).
    Ensure,
    /// Region known, zone spawnable (`presence`) but not yet `available` —
    /// `request_zone`.
    Request,
    /// Materialized (`available`) or unspawnable (presence clear) — nothing to do.
    Satisfied,
}

/// Per-tier tile radii an anchor projects (active ⊇ hot ⊇ warm ⊇ cold expected).
/// `0` = the anchor does not provide that tier. Read from a card's `anchor_*`
/// aspects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AnchorRadii {
    pub active: i32,
    pub hot: i32,
    pub warm: i32,
    pub cold: i32,
}

impl AnchorRadii {
    fn tiers(&self) -> [(ZoneTier, i32); 4] {
        [
            (ZoneTier::Active, self.active),
            (ZoneTier::Hot, self.hot),
            (ZoneTier::Warm, self.warm),
            (ZoneTier::Cold, self.cold),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Anchor {
    q: i32,
    r: i32,
    surface: u8,
    owner: u32,
    radii: AnchorRadii,
    /// The soul `card_id` this anchor represents (for per-soul memory). `0` for a
    /// non-soul anchor (e.g. a future viewport) — no memory tracked.
    soul: u32,
}

/// One soul's remembered knowledge of one zone. While the soul is `present`
/// (its anchor currently covers the zone at that sub level) the data is live
/// (read at `now`); on leaving, the timestamps **freeze** at the last present
/// moment and the retained rows become the soul's memory of that zone.
#[derive(Debug, Clone, Copy)]
struct MemEntry {
    /// Card-data knowledge time (frozen at the `anchor_hot` exit).
    card_ms: u64,
    /// Zone-tile knowledge time (frozen at the `anchor_cold` exit).
    zone_ms: u64,
    /// The soul currently covers this zone (knowledge is live = `now`).
    present: bool,
}

/// State of one zone subscription (per data type).
#[derive(Debug, Clone, Copy)]
struct SubState {
    /// The subscription is live on the wire.
    open: bool,
    /// An anchor currently wants this data here (vs a close-candidate).
    wanted: bool,
    /// Server clock (ms) the sub was last *present* (wanted) — warmth recency.
    last_present_ms: u64,
    /// Updates received since it became a candidate — warmth volatility.
    updates_since_present: u32,
}

/// Warmth eviction policy: should an open close-candidate be dropped now? The
/// recency-graded update tolerance — a recently-present zone is held through a
/// few updates (so a quick backtrack finds it warm), a long-gone one drops on
/// the next update. A silent candidate (0 updates) is never closed here — only
/// capacity-LRU reclaims it. Pure + tunable; profile to set the constants.
fn should_close(updates_since_present: u32, since_present_ms: u64) -> bool {
    let tolerance = if since_present_ms < 5 * 60_000 {
        5
    } else if since_present_ms < 15 * 60_000 {
        3
    } else {
        0
    };
    updates_since_present > tolerance
}

/// One anchor's covered zones → tightest tier, over its four tier-disks.
fn anchor_coverage(a: &Anchor) -> HashMap<u64, ZoneTier> {
    let mut cov: HashMap<u64, ZoneTier> = HashMap::new();
    for (tier, radius) in a.radii.tiers() {
        if radius <= 0 {
            continue;
        }
        for (cq, cr) in chunks_in_disk(a.q, a.r, radius) {
            let zone = pack_macro_zone_full(a.owner, a.surface, cq, cr);
            let e = cov.entry(zone).or_insert(tier);
            if tier_rank(tier) > tier_rank(*e) {
                *e = tier;
            }
        }
    }
    cov
}

/// Chunk coords whose extent intersects the Chebyshev tile-disk of `radius`
/// tiles around hex `(aq, ar)`.
fn chunks_in_disk(aq: i32, ar: i32, radius: i32) -> Vec<(i16, i16)> {
    // `zone_local` folds a tile coord to its zone, applying TILE_CENTER.
    let lo_q = zone_local(aq - radius).0;
    let hi_q = zone_local(aq + radius).0;
    let lo_r = zone_local(ar - radius).0;
    let hi_r = zone_local(ar + radius).0;
    let mut out = Vec::new();
    for cq in lo_q..=hi_q {
        for cr in lo_r..=hi_r {
            out.push((cq, cr));
        }
    }
    out
}

/// The pure zone registry + sticky subscription manager + region gate.
pub struct ZoneManager {
    /// Anchor-covered zones + their tier (drives `active_zones` / the matcher).
    entries: HashMap<u64, ZoneTier>,
    /// Refcounted container holds (inventory; [`ensure`]) — wanted at Active
    /// regardless of anchors, never released by the anchor recompute.
    refs: HashMap<u64, u32>,
    /// Live subscriptions, including open close-candidates (`wanted == false`).
    subs: HashMap<(u64, DataType), SubState>,

    // ── region gate (Zone subs only) ──
    // Server-truth cache: each subscribed region's presence/available bitmask.
    // The driver's reconcile derives `ensure_region`/`request_zone` straight from
    // these (via [`zone_needs`]) — no client-side request/arrival dedup latch.
    region_bits: HashMap<u64, RegionBits>,
    region_wanted: HashMap<u64, HashSet<u64>>,

    anchors: HashMap<String, Anchor>,

    /// Per-soul memory: `soul → (zone → MemEntry)`. A frozen entry pins the
    /// soul's remembered rows for that zone (anchor-aware GC) and answers "what
    /// does this soul know here" (the matcher's per-soul view, Phase D).
    soul_mem: HashMap<u32, HashMap<u64, MemEntry>>,

    /// Hard ceiling on open subs; over it, capacity-LRU evicts the
    /// least-recently-present candidate. The bound that replaces a distance
    /// release horizon — load-bearing.
    pub max_open_subs: usize,
    /// Per-soul memory bound: a soul remembers at most this many zones; over it,
    /// the least-recently-present frozen zone is forgotten (its rows unpin → GC).
    pub max_memory_zones: usize,

    intents: Vec<ZoneIntent>,
}

#[derive(Debug, Clone, Copy)]
struct RegionBits {
    presence: u64,
    #[allow(dead_code)]
    available: u64,
    /// The container's disk radius (tiles); `u16::MAX` = unbounded (world).
    #[allow(dead_code)]
    distance: u16,
}


impl Default for ZoneManager {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            refs: HashMap::new(),
            subs: HashMap::new(),
            region_bits: HashMap::new(),
            region_wanted: HashMap::new(),
            anchors: HashMap::new(),
            soul_mem: HashMap::new(),
            max_open_subs: 512,
            max_memory_zones: 4096,
            intents: Vec::new(),
        }
    }
}

impl ZoneManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn take_intents(&mut self) -> Vec<ZoneIntent> {
        std::mem::take(&mut self.intents)
    }

    // ── anchors ────────────────────────────────────────────────────────────

    /// Set/update a named anchor with its tier `radii`; recomputes coverage.
    /// `now` (server clock ms) stamps subscription recency.
    #[allow(clippy::too_many_arguments)]
    pub fn set_anchor(
        &mut self,
        name: &str,
        q: i32,
        r: i32,
        surface: u8,
        owner: u32,
        radii: AnchorRadii,
        soul: u32,
        now: u64,
    ) {
        let next = Anchor { q, r, surface, owner, radii, soul };
        if self.anchors.get(name) == Some(&next) {
            return;
        }
        self.anchors.insert(name.to_string(), next);
        self.recompute_anchor_zones(now);
    }

    /// Drop a named anchor; zones only kept alive by it become candidates.
    #[allow(dead_code)]
    pub fn clear_anchor(&mut self, name: &str, now: u64) {
        if self.anchors.remove(name).is_some() {
            self.recompute_anchor_zones(now);
        }
    }

    // ── container holds ──────────────────────────────────────────────────────

    /// Refcounted "track this zone" (container/inventory). First hold opens its
    /// subs; survives anchor recomputes.
    pub fn ensure(&mut self, zone: u64, now: u64) {
        let prev = self.refs.get(&zone).copied().unwrap_or(0);
        self.refs.insert(zone, prev + 1);
        if prev == 0 {
            self.recompute_anchor_zones(now);
        }
    }

    /// Drop one [`ensure`] hold; the last release makes its subs candidates
    /// (unless an anchor also covers it).
    #[allow(dead_code)]
    pub fn release(&mut self, zone: u64, now: u64) {
        let prev = self.refs.get(&zone).copied().unwrap_or(0);
        if prev <= 1 {
            self.refs.remove(&zone);
        } else {
            self.refs.insert(zone, prev - 1);
        }
        self.recompute_anchor_zones(now);
    }

    /// [`ensure`] the per-soul inventory zone for `soul_card_id`.
    pub fn ensure_inventory(&mut self, soul_card_id: u32, now: u64) {
        self.ensure(pack_macro_zone_full(soul_card_id, INVENTORY_LAYER, 0, 0), now);
    }

    // ── accessors ────────────────────────────────────────────────────────────

    #[allow(dead_code)]
    pub fn tier_of(&self, zone: u64) -> Option<ZoneTier> {
        self.entries.get(&zone).copied()
    }

    pub fn zones_in(&self, tier: ZoneTier) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().filter_map(move |(z, t)| (*t == tier).then_some(*z))
    }

    pub fn active_zones(&self) -> impl Iterator<Item = u64> + '_ {
        self.zones_in(ZoneTier::Active)
    }

    /// Open subscriptions, with their data type. (For reconnect catch-up / tests.)
    #[allow(dead_code)]
    pub fn open_subs(&self) -> impl Iterator<Item = (u64, DataType)> + '_ {
        self.subs.iter().filter_map(|(k, s)| s.open.then_some(*k))
    }

    /// Count of open subscriptions (tests / capacity diagnostics).
    #[allow(dead_code)]
    pub fn open_sub_count(&self) -> usize {
        self.subs.values().filter(|s| s.open).count()
    }

    // ── tier recompute ───────────────────────────────────────────────────────

    /// The desired tier per chunk, max-tier-wins across all anchors' disks, plus
    /// ref-held container zones at Active.
    fn desired_tiers(&self) -> HashMap<u64, ZoneTier> {
        let mut desired: HashMap<u64, ZoneTier> = HashMap::new();
        for a in self.anchors.values() {
            for (zone, tier) in anchor_coverage(a) {
                let e = desired.entry(zone).or_insert(tier);
                if tier_rank(tier) > tier_rank(*e) {
                    *e = tier;
                }
            }
        }
        for (zone, cnt) in &self.refs {
            if *cnt > 0 {
                desired.insert(*zone, ZoneTier::Active); // container hold ⇒ Active
            }
        }
        desired
    }

    /// Recompute tiers + the wanted-state of every sub. Subs that lose their
    /// last anchor become close-candidates (not closed); newly-wanted zones open.
    fn recompute_anchor_zones(&mut self, now: u64) {
        let desired = self.desired_tiers();

        // Tier registry = desired (anchor-covered + ref-held).
        let stale: Vec<u64> =
            self.entries.keys().copied().filter(|z| !desired.contains_key(z)).collect();
        for z in stale {
            self.entries.remove(&z);
        }
        for (z, t) in &desired {
            self.entries.insert(*z, *t);
        }

        // Update sub wanted-state over desired zones ∪ currently-open sub zones.
        let mut zones: HashSet<u64> = desired.keys().copied().collect();
        zones.extend(self.subs.keys().map(|(z, _)| *z));
        for zone in zones {
            let tier = desired.get(&zone).copied();
            let card_wanted = matches!(tier, Some(ZoneTier::Active) | Some(ZoneTier::Hot));
            let zone_wanted = tier.is_some();
            self.update_sub(zone, DataType::Card, card_wanted, now);
            self.update_sub(zone, DataType::Zone, zone_wanted, now);
        }

        self.enforce_capacity();
        self.update_memory(now);
    }

    // ── per-soul memory (watermarks) ───────────────────────────────────────────

    /// Refresh each soul's watermarks from its anchor's current coverage: zones
    /// it covers are `present` (knowledge live) with timestamps advanced to
    /// `now`; zones it no longer covers freeze (`present = false`) — those frozen
    /// entries are the soul's memory. Then enforce the per-soul memory bound.
    fn update_memory(&mut self, now: u64) {
        let covs: Vec<(u32, HashMap<u64, ZoneTier>)> = self
            .anchors
            .values()
            .filter(|a| a.soul != 0)
            .map(|a| (a.soul, anchor_coverage(a)))
            .collect();
        for (soul, cov) in covs {
            let mem = self.soul_mem.entry(soul).or_default();
            for (zone, tier) in &cov {
                let e = mem.entry(*zone).or_insert(MemEntry { card_ms: now, zone_ms: now, present: false });
                e.present = true;
                if matches!(tier, ZoneTier::Active | ZoneTier::Hot) {
                    e.card_ms = now; // card knowledge live only within anchor_hot
                }
                e.zone_ms = now; // zone-tile knowledge live within anchor_cold
            }
            for (zone, e) in mem.iter_mut() {
                if !cov.contains_key(zone) {
                    e.present = false; // left → freeze into memory
                }
            }
        }
        self.enforce_memory_lru();
    }

    /// Per-soul forget bound: over `max_memory_zones`, drop the least-recently-
    /// present FROZEN zone (a present zone is never forgotten). Its rows lose
    /// this soul's pin → reclaimable by [`Client::gc`](crate::client).
    fn enforce_memory_lru(&mut self) {
        let cap = self.max_memory_zones;
        for mem in self.soul_mem.values_mut() {
            while mem.len() > cap {
                let victim = mem
                    .iter()
                    .filter(|(_, e)| !e.present)
                    .min_by_key(|(_, e)| e.card_ms.max(e.zone_ms))
                    .map(|(z, _)| *z);
                match victim {
                    Some(z) => {
                        mem.remove(&z);
                    }
                    None => break, // all present — can't forget
                }
            }
        }
    }

    /// The timestamp `soul`'s **card** knowledge of `zone` is current as of:
    /// `now` while present (live), the frozen card watermark once it has left,
    /// `None` if the soul never knew this zone. The matcher reads card rows
    /// `current_at` this time for its per-soul view (Phase D).
    #[allow(dead_code)] // Phase D consumes this
    pub fn card_view_time(&self, soul: u32, zone: u64, now: u64) -> Option<u64> {
        let e = self.soul_mem.get(&soul)?.get(&zone)?;
        Some(if e.present { now } else { e.card_ms })
    }

    /// Frozen **card** watermarks across all souls that remember `zone` but are
    /// no longer present — the times that pin stale card rows in this zone for
    /// the anchor-aware GC. (Present souls pin the global-current row, kept
    /// anyway.)
    pub fn zone_card_pins(&self, zone: u64) -> Vec<u64> {
        self.soul_mem
            .values()
            .filter_map(|m| m.get(&zone))
            .filter(|e| !e.present)
            .map(|e| e.card_ms)
            .collect()
    }

    /// Frozen **zone-tile** watermarks across souls — pins for stale zone rows.
    pub fn zone_tile_pins(&self, zone: u64) -> Vec<u64> {
        self.soul_mem
            .values()
            .filter_map(|m| m.get(&zone))
            .filter(|e| !e.present)
            .map(|e| e.zone_ms)
            .collect()
    }

    /// Set a sub's wanted-state, opening it if newly wanted. Becoming unwanted
    /// leaves it open as a close-candidate (warmth / LRU reclaim it).
    fn update_sub(&mut self, zone: u64, dt: DataType, wanted: bool, now: u64) {
        let entry = self.subs.entry((zone, dt)).or_insert(SubState {
            open: false,
            wanted: false,
            last_present_ms: now,
            updates_since_present: 0,
        });
        let was_wanted = entry.wanted;
        entry.wanted = wanted;
        let mut need_open = false;
        if wanted {
            entry.last_present_ms = now;
            entry.updates_since_present = 0;
            if !entry.open {
                entry.open = true;
                need_open = true;
            }
        } else if was_wanted {
            // Just became a candidate — it was present up to now.
            entry.last_present_ms = now;
            entry.updates_since_present = 0;
        }
        // Drop a never-opened, unwanted placeholder so the map doesn't grow.
        if !entry.open && !entry.wanted {
            self.subs.remove(&(zone, dt));
        }
        if need_open {
            self.open_sub(zone, dt);
        }
    }

    fn open_sub(&mut self, zone: u64, dt: DataType) {
        self.intents.push(ZoneIntent::Subscribe { zone, data: dt, on: true });
        if dt == DataType::Zone {
            self.want_region(zone);
        }
    }

    fn close_sub(&mut self, zone: u64, dt: DataType) {
        self.subs.remove(&(zone, dt));
        self.intents.push(ZoneIntent::Subscribe { zone, data: dt, on: false });
        if dt == DataType::Zone {
            self.unwant_region(zone);
        }
    }

    /// Capacity-LRU: while over `max_open_subs`, evict the least-recently-present
    /// close-candidate. Wanted subs are never evicted (so the cap can be
    /// exceeded if every open sub is currently wanted).
    fn enforce_capacity(&mut self) {
        loop {
            if self.subs.values().filter(|s| s.open).count() <= self.max_open_subs {
                break;
            }
            let victim = self
                .subs
                .iter()
                .filter(|(_, s)| s.open && !s.wanted)
                .min_by_key(|(_, s)| s.last_present_ms)
                .map(|(k, _)| *k);
            match victim {
                Some((zone, dt)) => self.close_sub(zone, dt),
                None => break,
            }
        }
    }

    /// An inbound row for `zone`'s `dt` data arrived — warmth check. Only a
    /// close-candidate (open + unwanted) is affected; a wanted sub ignores it.
    pub fn note_update(&mut self, zone: u64, dt: DataType, now: u64) {
        let close = match self.subs.get_mut(&(zone, dt)) {
            Some(s) if s.open && !s.wanted => {
                s.updates_since_present += 1;
                should_close(s.updates_since_present, now.saturating_sub(s.last_present_ms))
            }
            _ => false,
        };
        if close {
            self.close_sub(zone, dt);
        }
    }

    // ── region gate (Zone subs) ────────────────────────────────────────────────

    fn want_region(&mut self, zone: u64) {
        let (region, _bit) = region_of_zone(zone);
        let set = self.region_wanted.entry(region).or_default();
        let fresh = set.is_empty();
        set.insert(zone);
        if fresh {
            self.intents.push(ZoneIntent::Region { region, subscribed: true });
        }
        // No request fired here — the driver's reconcile derives that from the
        // region's bits once they arrive (or immediately, if already cached).
    }

    fn unwant_region(&mut self, zone: u64) {
        let (region, _bit) = region_of_zone(zone);
        let Some(set) = self.region_wanted.get_mut(&region) else { return };
        set.remove(&zone);
        if set.is_empty() {
            self.region_wanted.remove(&region);
            self.region_bits.remove(&region);
            self.intents.push(ZoneIntent::Region { region, subscribed: false });
        }
    }

    /// What a wanted zone needs from the gate, derived ENTIRELY from server truth
    /// (the region's `presence`/`available` bits) — no client-side "already did
    /// it" latch. The driver's reconcile turns these into `Outbox` calls each
    /// tick; a request that doesn't take leaves `available` clear, so it's simply
    /// re-derived as `Request` again. Nothing can self-latch.
    pub fn zone_needs(&self) -> Vec<(u64, ZoneNeed)> {
        self.entries
            .keys()
            .map(|&zone| {
                let (region, bit) = region_of_zone(zone);
                let need = match self.region_bits.get(&region) {
                    None => ZoneNeed::Ensure, // no governing region yet
                    Some(b) => {
                        let mask = 1u64 << bit;
                        if b.available & mask != 0 {
                            ZoneNeed::Satisfied // materialized server-side
                        } else if b.presence & mask != 0 {
                            ZoneNeed::Request // exists-able, not yet spawned
                        } else {
                            ZoneNeed::Satisfied // presence clear ⇒ can't spawn here
                        }
                    }
                };
                (zone, need)
            })
            .collect()
    }

    /// Feed a subscribed region's current presence/availability bits. Server truth
    /// — the reconcile reads it via [`zone_needs`] to decide what to (re)request;
    /// nothing is fired here. The `regions` table is current-value (one row per
    /// region updated in place), so this is a straight overwrite.
    pub fn note_region(&mut self, region: u64, presence: u64, available: u64, distance: u16) {
        self.region_bits.insert(region, RegionBits { presence, available, distance });
    }

    /// Drop a region from the cache (its row was deleted server-side). Rare — a
    /// region is normally retired by the anchor leaving (`unwant_region`), not by
    /// a server delete.
    pub fn note_region_removed(&mut self, region: u64) {
        self.region_bits.remove(&region);
    }

    #[allow(dead_code)]
    pub fn is_zone_present(&self, zone: u64) -> bool {
        let (region, bit) = region_of_zone(zone);
        match self.region_bits.get(&region) {
            Some(bits) => bits.presence & (1u64 << bit) != 0,
            None => false,
        }
    }
}

/// Split a `macro_zone` into its bands (driver/tests).
#[allow(dead_code)]
pub fn zone_parts(zone: u64) -> (u32, u8, i16, i16) {
    let (q, r) = unpack_macro_zone(zone);
    (owner_of(zone), surface_of(zone), q, r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use resonantdust_codec::packed::WORLD_LAYER;

    fn world_zone(cq: i16, cr: i16) -> u64 {
        pack_macro_zone_full(0, WORLD_LAYER, cq, cr)
    }
    fn soul_radii() -> AnchorRadii {
        AnchorRadii { active: 2, hot: 6, warm: 12, cold: 20 }
    }
    /// Does an intent stream contain a sub open/close for `(zone, dt)`?
    fn has_sub(intents: &[ZoneIntent], zone: u64, dt: DataType, on: bool) -> bool {
        intents.iter().any(|i| matches!(i,
            ZoneIntent::Subscribe { zone: z, data, on: o } if *z == zone && *data == dt && *o == on))
    }

    #[test]
    fn centered_anchor_lights_one_active_zone() {
        let mut zm = ZoneManager::new();
        // Anchor on chunk 1's CENTRE tile (hex 7 = local cell 3 of chunk 1 under the
        // TILE_CENTER=3 shift) so an active radius 2 stays inside the single chunk.
        zm.set_anchor("s", 7, 7, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 0);
        assert_eq!(zm.active_zones().collect::<Vec<_>>(), vec![world_zone(1, 1)]);
    }

    #[test]
    fn tiers_nest_and_split_sub_types() {
        let mut zm = ZoneManager::new();
        // Anchor on chunk 1's centre tile (hex 7) so the nested radii fall on whole
        // chunks: active 2 → chunk 1; hot 6 → chunks 0,2; warm 12 → 3; cold 20 → 4.
        zm.set_anchor("s", 7, 7, WORLD_LAYER, 0, soul_radii(), 0, 0);
        assert_eq!(zm.tier_of(world_zone(1, 1)), Some(ZoneTier::Active));
        assert_eq!(zm.tier_of(world_zone(0, 0)), Some(ZoneTier::Hot));
        assert_eq!(zm.tier_of(world_zone(3, 3)), Some(ZoneTier::Warm));
        assert_eq!(zm.tier_of(world_zone(4, 4)), Some(ZoneTier::Cold));
        let intents = zm.take_intents();
        // Active/Hot get BOTH card + zone subs; Warm/Cold get zone only.
        assert!(has_sub(&intents, world_zone(1, 1), DataType::Card, true));
        assert!(has_sub(&intents, world_zone(1, 1), DataType::Zone, true));
        assert!(has_sub(&intents, world_zone(0, 0), DataType::Card, true), "hot → card sub");
        assert!(has_sub(&intents, world_zone(3, 3), DataType::Zone, true), "warm → zone sub");
        assert!(!has_sub(&intents, world_zone(3, 3), DataType::Card, true), "warm → NO card sub");
        assert!(!has_sub(&intents, world_zone(4, 4), DataType::Card, true), "cold → NO card sub");
    }

    #[test]
    fn moving_away_makes_candidate_not_immediate_unsub() {
        let mut zm = ZoneManager::new();
        zm.set_anchor("s", 12, 12, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 0);
        let z = world_zone(1, 1);
        assert!(zm.open_subs().any(|(zz, _)| zz == z));
        let _ = zm.take_intents();
        // Move far so the disk no longer covers chunk 1.
        zm.set_anchor("s", 200, 200, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 1000);
        // The sub is NOT closed — it's a candidate (still open, no unsub intent).
        let intents = zm.take_intents();
        assert!(!has_sub(&intents, z, DataType::Card, false), "no immediate unsub");
        assert!(zm.open_subs().any(|(zz, _)| zz == z), "abandoned zone still subscribed (candidate)");
        // The zone left the tier registry though (no longer covered).
        assert_eq!(zm.tier_of(z), None);
    }

    #[test]
    fn warmth_closes_a_noisy_candidate() {
        let mut zm = ZoneManager::new();
        zm.set_anchor("s", 12, 12, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 0);
        let z = world_zone(1, 1);
        zm.set_anchor("s", 200, 200, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 1000);
        let _ = zm.take_intents();
        // Candidate, recently present (since≈0 → tolerance 5). 5 updates: held.
        for _ in 0..5 {
            zm.note_update(z, DataType::Zone, 1000);
        }
        assert!(zm.open_subs().any(|(zz, d)| zz == z && d == DataType::Zone), "held through tolerance");
        // The 6th update exceeds tolerance → close.
        zm.note_update(z, DataType::Zone, 1000);
        assert!(!zm.open_subs().any(|(zz, d)| zz == z && d == DataType::Zone), "closed when too warm");
        assert!(has_sub(&zm.take_intents(), z, DataType::Zone, false));
    }

    #[test]
    fn capacity_lru_evicts_oldest_silent_candidate() {
        let mut zm = ZoneManager::new();
        zm.max_open_subs = 2; // tiny cap to force eviction
        // Centred anchor (hex 7 = centre cell 3 of chunk 1) → exactly chunk (1,1) →
        // card+zone = 2 subs = cap.
        zm.set_anchor("s", 7, 7, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 0);
        let z1 = world_zone(1, 1);
        assert!(zm.open_subs().any(|(zz, _)| zz == z1));
        // Move to a far centred chunk (hex 175 = centre cell 3 of chunk 25) → exactly
        // chunk (25,25). z1's 2 subs go silent-candidate; z2 wants 2 → over the
        // cap → LRU evicts z1's (older last_present).
        zm.set_anchor("s", 175, 175, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 10);
        let z2 = world_zone(25, 25);
        assert!(zm.open_subs().any(|(zz, _)| zz == z2), "new zone subscribed");
        assert!(!zm.open_subs().any(|(zz, _)| zz == z1), "LRU evicted the older silent candidate");
    }

    #[test]
    fn zone_needs_track_region_bits() {
        fn need(zm: &ZoneManager, z: u64) -> Option<ZoneNeed> {
            zm.zone_needs().into_iter().find(|(zz, _)| *zz == z).map(|(_, n)| n)
        }
        let mut zm = ZoneManager::new();
        // Anchor on chunk 1's centre tile → covers exactly zone (1,1).
        zm.set_anchor("s", 7, 7, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, 0, 0);
        let z = world_zone(1, 1);
        let (region, bit) = region_of_zone(z);
        let mask = 1u64 << bit;
        // No governing region yet → Ensure.
        assert_eq!(need(&zm, z), Some(ZoneNeed::Ensure));
        // Region known, zone present but not available → Request.
        zm.note_region(region, mask, 0, u16::MAX);
        assert_eq!(need(&zm, z), Some(ZoneNeed::Request));
        // `available` set → Satisfied (materialized; the reconcile won't re-request).
        zm.note_region(region, mask, mask, u16::MAX);
        assert_eq!(need(&zm, z), Some(ZoneNeed::Satisfied));
    }

    #[test]
    fn memory_freezes_on_leave_and_reads_live_when_present() {
        let mut zm = ZoneManager::new();
        let soul = 7000u32;
        let z = world_zone(1, 1);
        // Present at z (card-level): knowledge is live (= now), not frozen.
        zm.set_anchor("soul:7000", 12, 12, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, soul, 100);
        assert_eq!(zm.card_view_time(soul, z, 999), Some(999), "present → live (now)");
        assert!(zm.zone_card_pins(z).is_empty(), "present soul doesn't pin stale rows");
        // Leave: knowledge of z freezes at the last-present time (100).
        zm.set_anchor("soul:7000", 999, 999, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, soul, 5000);
        assert_eq!(zm.card_view_time(soul, z, 9999), Some(100), "left → frozen memory");
        assert_eq!(zm.zone_card_pins(z), vec![100], "frozen watermark pins the zone's stale rows");
    }

    #[test]
    fn memory_lru_forgets_least_recently_present_zone() {
        let mut zm = ZoneManager::new();
        zm.max_memory_zones = 1;
        let soul = 7000u32;
        let z1 = world_zone(1, 1);
        // Visit z1 then leave (frozen). Then visit a far zone, then leave. Anchors on
        // chunk-centre tiles (hex 7 → chunk 1, hex 175 → chunk 25) so each is one zone.
        zm.set_anchor("soul:7000", 7, 7, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, soul, 10);
        zm.set_anchor("soul:7000", 175, 175, WORLD_LAYER, 0, AnchorRadii { active: 2, ..Default::default() }, soul, 20);
        // Now memory has z1 (frozen@10) + z2 (present). cap 1 → z1 (frozen) forgotten.
        assert_eq!(zm.card_view_time(soul, z1, 99), None, "least-recently-present zone forgotten");
        assert_eq!(zm.card_view_time(soul, world_zone(25, 25), 99), Some(99), "current zone kept (present)");
    }

    #[test]
    fn ensure_inventory_survives_anchor_recompute() {
        let mut zm = ZoneManager::new();
        let soul = 1234u32;
        zm.ensure_inventory(soul, 0);
        let inv = pack_macro_zone_full(soul, INVENTORY_LAYER, 0, 0);
        assert_eq!(zm.tier_of(inv), Some(ZoneTier::Active));
        zm.set_anchor("s", 12, 12, WORLD_LAYER, 0, soul_radii(), 0, 0);
        assert_eq!(zm.tier_of(inv), Some(ZoneTier::Active), "inventory hold survives");
        zm.release(inv, 0);
        assert_eq!(zm.tier_of(inv), None);
    }
}
