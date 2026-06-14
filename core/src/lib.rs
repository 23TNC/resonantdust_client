//! Library root — the crate root for the browser (wasm) build and the reusable
//! module surface beneath the transport seam.
//!
//! The native NPC driver + multi-client harness live in the bin (`main.rs`, which
//! declares its own module tree and owns the tokio `session` driver). This lib
//! exposes the sans-IO core (`client`), the world mirror, the gate framing, and —
//! on wasm — the synchronous `wasm` surface the view's worker hosts. The `session`
//! driver is intentionally NOT here: it's tokio-bound and native-only.

pub mod actions;
pub mod clock;
pub mod client;
pub mod content;
pub mod gate;
pub mod outbox;
pub mod rows;
pub mod transport;
pub mod world;
pub mod zones;
