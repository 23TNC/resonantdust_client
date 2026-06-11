//! Triggered action queue + debounce — port of pixijs `ActionManager`'s queue.
//!
//! A recipe that matches an assembled root is NOT sent immediately — it's
//! QUEUED for a debounce window (the player/NPC cancel affordance, and what lets
//! a stack keep growing before a sub-recipe commits). Re-evaluation on every
//! stack/card change keeps the queue current: a superseding match swaps the
//! entry (resetting the timer), a no-longer-matching config drops it. Only after
//! the config is **stable** for the window does the client `propose`. A
//! single-input recipe (`input_count ≤ 1`, e.g. lifecycle) bypasses the debounce
//! (delay 0 — nothing to cancel). No predicted holds (per the no-prediction
//! decision — dedup + the buffer cover double-submit).
//!
//! The data + timing helpers live here; the orchestration (`evaluate_root` /
//! `fire_ready` / reply handling) is on [`crate::client::Client`], which owns the
//! matcher and `propose`.

/// Default debounce window before a multi-input match commits.
pub const DEFAULT_DELAY_MS: f64 = 5000.0;
/// Cap on time-drift resubmits before dropping (anti clock-skew loop).
pub const MAX_TIME_DRIFT_RETRIES: u32 = 3;
/// Padding over the reported gap when rescheduling a time-drift retry.
pub const TIME_DRIFT_RETRY_PAD_MS: f64 = 250.0;

/// One queued recipe, keyed by its chain root in the client's action queue.
#[derive(Debug, Clone, PartialEq)]
pub struct QueuedAction {
    /// The soul whose memory view matched it (for fire-time re-eval).
    pub soul: u32,
    pub recipe: String,
    /// Root to propose with (`0` when the matcher folded the loose root into a
    /// branch — the promotion passes).
    pub root: u32,
    pub bindings: Vec<Vec<u32>>,
    /// Action location — the original root card's cell.
    pub surface: u8,
    pub macro_zone: u64,
    pub micro_location: u32,
    /// Local monotonic time (ms) the debounce timer last (re)started.
    pub scheduled_at: f64,
    /// Debounce window (0 = fire immediately).
    pub delay_ms: f64,
    /// Time-drift resubmits used so far.
    pub retry_count: u32,
    /// `cid` of the in-flight `propose` — `Some` between fire and its reply, so a
    /// `call_ok`/`call_err` can be matched back to this entry.
    pub submit_cid: Option<u32>,
}

impl QueuedAction {
    /// Ready to fire: not already in flight and the debounce window has elapsed.
    pub fn ready(&self, perf_ms: f64) -> bool {
        self.submit_cid.is_none() && perf_ms - self.scheduled_at >= self.delay_ms
    }

    /// Same recipe + bindings as a fresh match — keep the running timer rather
    /// than resetting it (the config hasn't meaningfully changed).
    pub fn same_as(&self, recipe: &str, bindings: &[Vec<u32>]) -> bool {
        self.recipe == recipe && self.bindings == bindings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(recipe: &str, delay: f64, at: f64) -> QueuedAction {
        QueuedAction {
            soul: 1,
            recipe: recipe.to_string(),
            root: 0,
            bindings: vec![vec![10, 11, 12]],
            surface: 64,
            macro_zone: 0,
            micro_location: 0,
            scheduled_at: at,
            delay_ms: delay,
            retry_count: 0,
            submit_cid: None,
        }
    }

    #[test]
    fn ready_after_window_unless_in_flight() {
        let mut a = entry("triple_corpus", 5000.0, 1000.0);
        assert!(!a.ready(5999.0), "not yet (4999ms elapsed < 5000)");
        assert!(a.ready(6000.0), "elapsed == window → ready");
        a.submit_cid = Some(7);
        assert!(!a.ready(99999.0), "in-flight → never ready");
    }

    #[test]
    fn zero_delay_fires_at_once() {
        let a = entry("fleeting", 0.0, 1000.0);
        assert!(a.ready(1000.0), "single-input bypass → ready immediately");
    }

    #[test]
    fn same_as_keeps_timer_distinguishes_supersede() {
        let a = entry("corpus_b_top", 5000.0, 1000.0);
        assert!(a.same_as("corpus_b_top", &[vec![10, 11, 12]]));
        assert!(!a.same_as("triple_corpus", &[vec![10, 11, 12]]), "different recipe → reset");
        assert!(!a.same_as("corpus_b_top", &[vec![10, 11]]), "different bindings → reset");
    }
}
