//! Wire row types — re-exported from the shared protocol crate.
//!
//! These moved to [`resonantdust_protocol::rows`] when the gate→client wire went
//! binary (postcard): one definition serves the gate and the client so the
//! positional encoding can't drift, and numbers ride native (the old JSON path's
//! camelCase + string-coercion is gone). This shim keeps the `crate::rows::…`
//! paths stable for the rest of the core.

pub use resonantdust_protocol::rows::{CardRow, ChatRow, PlayerRow, RegionRow, RowData, ZoneRow};
