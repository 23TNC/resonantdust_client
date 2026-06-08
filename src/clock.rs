//! Client clock discipline — server-time tracking with an adaptive backward
//! buffer (`client_delay`). Port of the pixijs `ReducerManager` clock logic.
//!
//! The client stamps outbound `client_time_ms` on a clock that (a) tracks the
//! gate's wall clock despite RTT jitter and (b) runs **deliberately behind** it
//! by `client_delay`, so the gate's future-stamped completion rows land within
//! the buffer and the client never stamps into the future. Server-time samples
//! arrive as `server_micros` on reducer replies (`call_ok`/`call_err`) and on
//! idle `time` heartbeats; between samples the estimate is extrapolated with a
//! **monotonic local clock** (`perf_ms`) the driver supplies — `std::Instant`
//! natively, `performance.now()` in wasm. The core stays time-pure: perf is
//! injected, never read here, so this is fully deterministic in tests.
//!
//! Algorithm (per [`note_sample`]):
//! - `anchor` = the capture with the largest `server_ms − perf_ms` offset (the
//!   freshest, least-queued sample).
//! - prediction error `delta = (anchor-extrapolation + running_delta) − server`.
//! - **spike** (`|delta| > running_delay/2`): pull `running_delay` toward
//!   `|delta|` (half-step EWMA) and set `client_delay = clamp(2·running_delay)`.
//!   Lag grows fast on a jitter spike.
//! - **normal**: half-step feedback `running_delta -= delta/2`. The offset
//!   creeps toward truth; `client_delay` decays only as old spikes age out.

use std::collections::VecDeque;

const SAMPLE_WINDOW: usize = 16;
/// A sample whose `server_ms` is more than this behind the current extrapolation
/// is treated as STALE (a keepalive that queued during an idle/disconnected gap
/// and is only now being processed) and dropped — see [`ClockSync::note_sample`].
/// Well above real jitter/spike magnitude so legitimate corrections still apply.
const STALE_SAMPLE_MS: f64 = 6000.0;
const RUNNING_DELAY_INIT_MS: f64 = 1500.0;
const CLIENT_DELAY_INIT_MS: f64 = 3000.0;
const CLIENT_DELAY_MIN_MS: f64 = 1500.0;
const CLIENT_DELAY_MAX_MS: f64 = 5000.0;

#[derive(Clone, Copy)]
struct Capture {
    /// Server wall clock at this sample (ms).
    server_ms: f64,
    /// Local monotonic clock when the sample was received (ms).
    perf_ms: f64,
}

/// Adaptive server-time estimator. See module docs.
pub struct ClockSync {
    captures: VecDeque<Capture>,
    /// Feedback-controlled offset correction (creeps toward truth).
    running_delta: f64,
    /// Tracked jitter magnitude — grows fast on spikes, decays slow.
    running_delay: f64,
    /// The backward lag applied to `server_now_ms` (= clamp(2·running_delay)).
    client_delay: f64,
}

impl Default for ClockSync {
    fn default() -> Self {
        Self {
            captures: VecDeque::new(),
            running_delta: 0.0,
            running_delay: RUNNING_DELAY_INIT_MS,
            client_delay: CLIENT_DELAY_INIT_MS,
        }
    }
}

impl ClockSync {
    pub fn new() -> Self {
        Self::default()
    }

    /// The freshest anchor capture (max `server_ms − perf_ms`). `None` until the
    /// first sample.
    fn anchor(&self) -> Option<Capture> {
        self.captures.iter().copied().max_by(|a, b| {
            (a.server_ms - a.perf_ms)
                .partial_cmp(&(b.server_ms - b.perf_ms))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Ingest a server-time sample (`server_ms`, taken at local `perf_ms`),
    /// adapting `running_delta` / `running_delay` / `client_delay`, then record
    /// the capture (trimming to the window).
    pub fn note_sample(&mut self, server_ms: f64, perf_ms: f64) {
        if let Some(prev) = self.anchor() {
            let extrapolation = prev.server_ms + (perf_ms - prev.perf_ms);
            // Reject a STALE sample: its `server_ms` is far behind where we
            // already know the clock to be. This happens when `time` keepalives
            // queue on the socket during an idle/disconnected stretch and are all
            // processed at once on resume — each old `server_ms` paired with the
            // (much later) current `perf_ms`. Feeding them would spike
            // `running_delay` and drag the estimate seconds behind true server
            // (→ the gate rejects our stamps as `client_behind`). Drop them; the
            // freshest queued sample (`server_ms ≈ now`) passes and re-anchors.
            if extrapolation - server_ms > STALE_SAMPLE_MS {
                return;
            }
            let delta = (extrapolation + self.running_delta) - server_ms;
            if delta.abs() > self.running_delay / 2.0 {
                self.running_delay += (delta.abs() - self.running_delay) / 2.0;
                self.client_delay =
                    (2.0 * self.running_delay).clamp(CLIENT_DELAY_MIN_MS, CLIENT_DELAY_MAX_MS);
            } else {
                self.running_delta -= delta / 2.0;
            }
        }
        self.captures.push_back(Capture { server_ms, perf_ms });
        while self.captures.len() > SAMPLE_WINDOW {
            self.captures.pop_front();
        }
    }

    /// The client's current server-time estimate (ms) — running `client_delay`
    /// behind true server time. `perf_ms` is the local monotonic now. `0` before
    /// any sample.
    pub fn server_now_ms(&self, perf_ms: f64) -> f64 {
        match self.anchor() {
            Some(a) => a.server_ms + (perf_ms - a.perf_ms) + self.running_delta - self.client_delay,
            None => 0.0,
        }
    }

    /// Whether at least one sample has landed (so `server_now_ms` is meaningful).
    pub fn is_synced(&self) -> bool {
        !self.captures.is_empty()
    }

    /// Re-seat the clock after a gate `time_drift` rejection. `ahead` = the
    /// client stamped a future time (`client_time_ms` was `gap_ms` ahead of
    /// server); else it was behind. Clears the window and seeds a synthetic
    /// capture so the corrected time takes effect immediately, resetting
    /// `running_delta` (don't double-count against the cleared window) while
    /// keeping the longer-term spike memory (`running_delay`/`client_delay`).
    pub fn correct_from_drift(&mut self, ahead: bool, gap_ms: f64, client_time_ms: f64, perf_ms: f64) {
        let server_ms = if ahead { client_time_ms - gap_ms } else { client_time_ms + gap_ms };
        self.captures.clear();
        self.captures.push_back(Capture { server_ms, perf_ms });
        self.running_delta = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_behind_by_client_delay_on_a_steady_clock() {
        let mut c = ClockSync::new();
        // A steady server clock (server == perf + 1_000_000 offset), sampled
        // every 1000ms perf. After warm-up, server_now should sit ≈ client_delay
        // behind the true server time.
        for i in 0..20 {
            let perf = i as f64 * 1000.0;
            let server = perf + 1_000_000.0; // true server time at receive
            c.note_sample(server, perf);
        }
        let perf = 20_000.0;
        let true_server = perf + 1_000_000.0;
        let est = c.server_now_ms(perf);
        let behind = true_server - est;
        // Steady clock → no spikes → client_delay stays at its 3000 init (≥ min).
        assert!(
            (behind - 3000.0).abs() < 50.0,
            "should run ~client_delay(3000) behind; behind={behind}"
        );
    }

    #[test]
    fn client_delay_grows_on_a_latency_spike() {
        let mut c = ClockSync::new();
        for i in 0..8 {
            let perf = i as f64 * 1000.0;
            c.note_sample(perf + 1_000_000.0, perf);
        }
        let before = c.client_delay;
        // A sample that arrives very late (server jumped +4000 vs extrapolation)
        // is a big prediction error → spike branch grows the delay.
        let perf = 8_000.0;
        c.note_sample(perf + 1_000_000.0 + 4000.0, perf);
        assert!(c.client_delay > before, "spike grew client_delay ({before} → {})", c.client_delay);
        assert!(c.client_delay <= CLIENT_DELAY_MAX_MS, "clamped to max");
    }

    #[test]
    fn correct_from_drift_reseats_immediately() {
        let mut c = ClockSync::new();
        c.note_sample(1_000_000.0, 0.0);
        // Gate says we were 800ms behind at client_time=1_000_000 (perf 100).
        c.correct_from_drift(false, 800.0, 1_000_000.0, 100.0);
        // New anchor server = client + gap = 1_000_800 at perf 100; delay reset.
        // server_now at perf 100 = 1_000_800 + 0 - client_delay.
        let est = c.server_now_ms(100.0);
        assert!((est - (1_000_800.0 - c.client_delay)).abs() < 1.0, "reseated to corrected server time");
    }

    #[test]
    fn rejects_stale_buffered_samples_on_resume() {
        let mut c = ClockSync::new();
        // Warm up on a steady clock (server == perf + 1_000_000).
        for i in 0..8 {
            let perf = i as f64 * 1000.0;
            c.note_sample(perf + 1_000_000.0, perf);
        }
        let before_delay = c.client_delay;
        // Simulate resume after an idle gap: a burst of STALE keepalives (server
        // times from ~20s ago) all processed NOW (perf 8000). These must be
        // dropped, not spike the delay / drag the estimate behind.
        let perf = 8000.0;
        for k in 0..5 {
            c.note_sample(1_000_000.0 - 20_000.0 + k as f64 * 1000.0, perf);
        }
        assert_eq!(c.client_delay, before_delay, "stale samples must not spike client_delay");
        let true_server = perf + 1_000_000.0;
        let behind = true_server - c.server_now_ms(perf);
        assert!(
            behind > 0.0 && behind < before_delay + 100.0,
            "estimate must stay ~client_delay behind, not dragged back by stale samples; behind={behind}"
        );
    }

    #[test]
    fn unsynced_before_first_sample() {
        let c = ClockSync::new();
        assert!(!c.is_synced());
        assert_eq!(c.server_now_ms(123.0), 0.0);
    }
}
