//! Wire row types — the shard table rows as the gate delivers them.
//!
//! The gate normalizes every row before it hits the socket (see the gateway's
//! `ws::normalize`): object keys are **camelCased** and every number is
//! **stringified** (u64 fields like `validAt` / `macroZone` exceed JS's safe
//! integer range, so they ride as strings). These structs mirror that exactly —
//! `rename_all = "camelCase"` plus a string-parsing deserializer on every numeric
//! field — so a `GateMsg::Row { row, .. }` payload deserializes straight in.
//!
//! Placement / flag *meaning* is not decoded here; that's `resonantdust_codec`'s
//! job, reached through the helpers below so the bit layout has one owner.

use resonantdust_codec::card_model::{self, Micro};
use resonantdust_codec::packed::valid_at_time;
use serde::Deserialize;

/// Deserialize a number that arrived as a JSON string (the gate stringifies all
/// numerics). Generic over the target integer type; serde monomorphizes it per
/// field from the field's type.
fn de_str_num<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    let s = String::deserialize(d)?;
    s.parse::<T>().map_err(serde::de::Error::custom)
}

/// One version-row of the `cards` table, as delivered by the gate. A card has
/// many of these over time (the bitemporal history); `valid_at` is the PK.
//
// Some fields (macro_zone, flags_bk, stock, …) aren't read directly yet — they
// land as the world model grows (zones, tile-stock display, …). They're part of
// the wire row regardless, so carry them now.
#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CardRow {
    #[serde(deserialize_with = "de_str_num")]
    pub valid_at: u64,
    #[serde(deserialize_with = "de_str_num")]
    pub card_id: u32,
    #[serde(deserialize_with = "de_str_num")]
    pub macro_zone: u64,
    #[serde(deserialize_with = "de_str_num")]
    pub micro_location: u32,
    #[serde(deserialize_with = "de_str_num")]
    pub owner_id: u32,
    #[serde(deserialize_with = "de_str_num")]
    pub packed_definition: u16,
    /// Propagating flag word — state bits + placement + refcount holds.
    #[serde(deserialize_with = "de_str_num")]
    pub flags: u32,
    /// Non-propagating bookkeeping byte (dirty/preserve). The client rarely
    /// needs it, but it's on the wire so we carry it.
    #[serde(deserialize_with = "de_str_num")]
    pub flags_bk: u8,
    /// Tile-card per-row stock byte.
    #[serde(deserialize_with = "de_str_num")]
    pub stock: u32,
}

impl CardRow {
    /// Wall-clock ms this row became valid (the high 48 bits of `valid_at`).
    pub fn time_ms(&self) -> u64 {
        valid_at_time(self.valid_at)
    }

    /// Decode this row's placement (loose coords or a stack-member of a root).
    pub fn micro(&self) -> Micro {
        Micro::of(self.micro_location, self.flags)
    }

    /// `dead` state bit set?
    pub fn is_dead(&self) -> bool {
        card_model::is_dead(self.flags)
    }
}

/// One version-row of the `zones` table — a zone's 8×8 tile grid packed into 16
/// u64s, as the gate delivers it (camelCase keys, stringified numbers).
#[allow(dead_code)] // zone_id / owner_id land with region indexing / ownership
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZoneRow {
    #[serde(deserialize_with = "de_str_num")] pub valid_at: u64,
    #[serde(deserialize_with = "de_str_num")] pub zone_id: u32,
    #[serde(deserialize_with = "de_str_num")] pub macro_zone: u64,
    /// `[card_type:u4 | 0:u4]` — the tile card_type for this zone's tiles.
    #[serde(deserialize_with = "de_str_num")] pub packed_definition: u8,
    #[serde(deserialize_with = "de_str_num")] pub owner_id: u32,
    #[serde(deserialize_with = "de_str_num")] t0: u64,
    #[serde(deserialize_with = "de_str_num")] t1: u64,
    #[serde(deserialize_with = "de_str_num")] t2: u64,
    #[serde(deserialize_with = "de_str_num")] t3: u64,
    #[serde(deserialize_with = "de_str_num")] t4: u64,
    #[serde(deserialize_with = "de_str_num")] t5: u64,
    #[serde(deserialize_with = "de_str_num")] t6: u64,
    #[serde(deserialize_with = "de_str_num")] t7: u64,
    #[serde(deserialize_with = "de_str_num")] t8: u64,
    #[serde(deserialize_with = "de_str_num")] t9: u64,
    #[serde(deserialize_with = "de_str_num")] t10: u64,
    #[serde(deserialize_with = "de_str_num")] t11: u64,
    #[serde(deserialize_with = "de_str_num")] t12: u64,
}

/// One version-row of the `regions` table — the per-region spawn
/// presence/availability bitfields, as the gate delivers it (camelCase keys,
/// stringified numbers). Feeds the zone manager's region gate.
#[allow(dead_code)] // valid_at / available read as the gate refines (current-at-now)
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RegionRow {
    #[serde(deserialize_with = "de_str_num")] pub valid_at: u64,
    #[serde(deserialize_with = "de_str_num")] pub macro_region: u64,
    /// Bit `i` set → the zone at region slot `i` MAY be spawned.
    #[serde(deserialize_with = "de_str_num")] pub zone_presence: u64,
    /// Bit `i` set → the zone at region slot `i` HAS been spawned.
    #[serde(deserialize_with = "de_str_num")] pub zone_available: u64,
    /// Disk radius (tiles) the region is bounded by (presence + tile mask). The
    /// client only reads `zone_presence`; carried for completeness / future use.
    #[serde(deserialize_with = "de_str_num")] pub distance: u16,
}

impl RegionRow {
    pub fn time_ms(&self) -> u64 {
        valid_at_time(self.valid_at)
    }
}

impl ZoneRow {
    pub fn time_ms(&self) -> u64 {
        valid_at_time(self.valid_at)
    }

    /// The tile grid as the codec's packed array.
    pub fn tile_words(&self) -> [u64; resonantdust_codec::packed::ZONE_TILE_U64_COUNT] {
        [
            self.t0, self.t1, self.t2, self.t3, self.t4, self.t5, self.t6,
            self.t7, self.t8, self.t9, self.t10, self.t11, self.t12,
        ]
    }

    /// The tile card_type for this zone's tiles (upper nibble of `packed_definition`).
    pub fn tile_card_type(&self) -> u8 {
        resonantdust_codec::packed::unpack_zone_definition(self.packed_definition)
    }

    /// The distinct non-empty tile `def_id`s present in this zone (the "unique
    /// tile attributes" search keys, before aspect resolution).
    pub fn unique_tile_def_ids(&self) -> Vec<u16> {
        use resonantdust_codec::packed::{tile_def_id, ZONE_TILE_COUNT};
        let words = self.tile_words();
        let mut seen: Vec<u16> = Vec::new();
        for idx in 0..ZONE_TILE_COUNT {
            let def_id = tile_def_id(&words, idx);
            if def_id != 0 && !seen.contains(&def_id) {
                seen.push(def_id);
            }
        }
        seen
    }
}
