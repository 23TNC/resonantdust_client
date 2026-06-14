//! Fault-tolerant outbound call queue — the ONE path every gateway reducer call
//! walks through.
//!
//! The rule it enforces: **the client may only delay a send, never gate it on
//! its own bookkeeping.** A call is sent at most once while in flight, and the
//! server's *response kind* — not any local "already sent" flag — decides what
//! happens next:
//!
//! - `call_ok`      → done. The call achieved its goal; drop it.
//! - `call_error`   → retry. Re-queue with exponential back-off (by retry count).
//! - `call_promise` → await. The server accepted it as async and will resolve
//!                    later (a subsequent `ok`/`err` on the same cid); hold it in
//!                    the awaiting state until the server's timeout, then retry.
//! - in-flight / awaiting timeout → retry.
//!
//! Because only a server response (or a timeout) ever advances a call, **nothing
//! can self-latch**: a silent failure comes back as `error` (or times out) and
//! re-fires; a real success comes back as `ok` and is done. The server is the
//! authority on completion, never the client's memory of what it did.
//!
//! Duplicate suppression is layered: the client never re-sends a call that's
//! in-flight/awaiting, and the gate drops duplicate inbound requests until a
//! timeout after it last replied.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// An in-flight call with no response is assumed lost after this and retried.
/// Above a normal round-trip (never duplicate a live call) but low enough to
/// self-heal a dropped reply in seconds.
const INFLIGHT_TTL_MS: f64 = 5_000.0;
/// Retry back-off after an `error` (or a timeout): `BASE * 2^retry`, capped.
const BACKOFF_BASE_MS: f64 = 1_000.0;
const BACKOFF_CAP_MS: f64 = 30_000.0;

/// Stable content hash identifying the SAME logical request across re-queues,
/// independent of per-attempt fields (cid, client_time). Producers hash the
/// reducer + an identifying key (e.g. a `macro_zone`).
pub fn call_hash(reducer: &str, key: u64) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    reducer.hash(&mut h);
    key.hash(&mut h);
    h.finish()
}

fn backoff(retry: u32) -> f64 {
    (BACKOFF_BASE_MS * 2f64.powi(retry.min(8) as i32)).min(BACKOFF_CAP_MS)
}

#[derive(Clone, Copy)]
enum State {
    /// Waiting to (re)send no earlier than `send_at`.
    Queued { send_at: f64 },
    /// Sent, awaiting the first response; assumed lost past `expiry`.
    Sent { cid: u32, expiry: f64 },
    /// Server promised an async resolution; awaiting it until `expiry`.
    Awaiting { cid: u32, expiry: f64 },
}

struct Call {
    reducer: String,
    /// Args WITHOUT per-attempt fields (cid / client_time) — stamped fresh by the
    /// driver at send time.
    args: serde_json::Value,
    retry: u32,
    state: State,
}

/// A single call the driver should emit now: `(cid, reducer, args)`.
pub struct Outgoing {
    pub cid: u32,
    pub reducer: String,
    pub args: serde_json::Value,
}

#[derive(Default)]
pub struct Outbox {
    calls: HashMap<u64, Call>,
    by_cid: HashMap<u32, u64>,
}

impl Outbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare intent to send `reducer(args)`, keyed by `hash`. Idempotent:
    /// re-`want`ing a live call just refreshes its args (so a freshened
    /// `client_time` lands) and leaves its state/schedule untouched. A new call
    /// is eligible to send immediately.
    pub fn want(&mut self, hash: u64, reducer: &str, args: serde_json::Value, now: f64) {
        match self.calls.get_mut(&hash) {
            Some(c) => c.args = args,
            None => {
                self.calls.insert(
                    hash,
                    Call { reducer: reducer.to_string(), args, retry: 0, state: State::Queued { send_at: now } },
                );
            }
        }
    }

    /// Retire a call — superseded / no longer wanted. (Distinct from `ok`, which
    /// is the server confirming completion.)
    pub fn done(&mut self, hash: u64) {
        if let Some(c) = self.calls.remove(&hash) {
            if let State::Sent { cid, .. } | State::Awaiting { cid, .. } = c.state {
                self.by_cid.remove(&cid);
            }
        }
    }

    pub fn is_queued(&self, hash: u64) -> bool {
        self.calls.contains_key(&hash)
    }

    /// Emit every due, idle call. Sweeps timed-out in-flight / awaiting calls back
    /// to a (backed-off) retry first, so a lost reply or an unkept promise never
    /// strands a call. `alloc_cid` mints a fresh call id per send.
    pub fn pump(&mut self, now: f64, mut alloc_cid: impl FnMut() -> u32) -> Vec<Outgoing> {
        // Timeouts → retry.
        for (_, c) in self.calls.iter_mut() {
            let expired = match c.state {
                State::Sent { expiry, .. } | State::Awaiting { expiry, .. } => now >= expiry,
                State::Queued { .. } => false,
            };
            if expired {
                if let State::Sent { cid, .. } | State::Awaiting { cid, .. } = c.state {
                    self.by_cid.remove(&cid);
                }
                c.retry += 1;
                c.state = State::Queued { send_at: now + backoff(c.retry) };
            }
        }
        // Send due queued calls.
        let due: Vec<u64> = self
            .calls
            .iter()
            .filter(|(_, c)| matches!(c.state, State::Queued { send_at } if now >= send_at))
            .map(|(h, _)| *h)
            .collect();
        let mut out = Vec::new();
        for hash in due {
            let cid = alloc_cid();
            let c = self.calls.get_mut(&hash).expect("due call present");
            c.state = State::Sent { cid, expiry: now + INFLIGHT_TTL_MS };
            self.by_cid.insert(cid, hash);
            out.push(Outgoing { cid, reducer: c.reducer.clone(), args: c.args.clone() });
        }
        out
    }

    /// `call_ok` — the call reached its goal. Drop it.
    pub fn on_ok(&mut self, cid: u32) {
        if let Some(hash) = self.by_cid.remove(&cid) {
            self.calls.remove(&hash);
        }
    }

    /// `call_error` — re-queue for retry with exponential back-off.
    pub fn on_err(&mut self, cid: u32, now: f64) {
        if let Some(hash) = self.by_cid.remove(&cid) {
            if let Some(c) = self.calls.get_mut(&hash) {
                c.retry += 1;
                c.state = State::Queued { send_at: now + backoff(c.retry) };
            }
        }
    }

    /// `call_promise` — the server accepted it async and will resolve (a later
    /// `ok`/`err` on this cid). Hold it in `Awaiting` until the server's timeout;
    /// if it never resolves, the sweep re-queues it.
    pub fn on_promise(&mut self, cid: u32, timeout_ms: f64, now: f64) {
        if let Some(&hash) = self.by_cid.get(&cid) {
            if let Some(c) = self.calls.get_mut(&hash) {
                c.state = State::Awaiting { cid, expiry: now + timeout_ms };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a() -> serde_json::Value {
        serde_json::json!({ "macro_zone": 1 })
    }
    fn pump(ob: &mut Outbox, now: f64, n: &mut u32) -> usize {
        ob.pump(now, || { *n += 1; *n }).len()
    }

    #[test]
    fn sends_once_and_holds_while_in_flight() {
        let mut ob = Outbox::new();
        let (h, mut n) = (call_hash("request_zone", 1), 0);
        ob.want(h, "request_zone", a(), 0.0);
        assert_eq!(pump(&mut ob, 0.0, &mut n), 1);
        assert_eq!(pump(&mut ob, 4_000.0, &mut n), 0, "live call not duplicated");
    }

    #[test]
    fn ok_retires_it() {
        let mut ob = Outbox::new();
        let (h, mut n) = (call_hash("request_zone", 1), 0);
        ob.want(h, "request_zone", a(), 0.0);
        let cid = ob.pump(0.0, || { n += 1; n })[0].cid;
        ob.on_ok(cid);
        assert!(!ob.is_queued(h));
        assert_eq!(pump(&mut ob, 60_000.0, &mut n), 0);
    }

    #[test]
    fn err_retries_with_backoff() {
        let mut ob = Outbox::new();
        let (h, mut n) = (call_hash("request_zone", 1), 0);
        ob.want(h, "request_zone", a(), 0.0);
        let cid = ob.pump(0.0, || { n += 1; n })[0].cid;
        ob.on_err(cid, 0.0); // retry 1 → backoff 2000ms
        assert_eq!(pump(&mut ob, 1_000.0, &mut n), 0, "within backoff");
        assert_eq!(pump(&mut ob, 2_500.0, &mut n), 1, "re-fires after backoff — never latched");
    }

    #[test]
    fn promise_awaits_then_resolves() {
        let mut ob = Outbox::new();
        let (h, mut n) = (call_hash("request_zone", 1), 0);
        ob.want(h, "request_zone", a(), 0.0);
        let cid = ob.pump(0.0, || { n += 1; n })[0].cid;
        ob.on_promise(cid, 10_000.0, 0.0); // server: "I'll resolve within 10s"
        // While awaiting, no resend.
        assert_eq!(pump(&mut ob, 5_000.0, &mut n), 0);
        // Resolution arrives (same cid) as ok → done.
        ob.on_ok(cid);
        assert!(!ob.is_queued(h));
    }

    #[test]
    fn promise_timeout_retries() {
        let mut ob = Outbox::new();
        let (h, mut n) = (call_hash("request_zone", 1), 0);
        ob.want(h, "request_zone", a(), 0.0);
        let cid = ob.pump(0.0, || { n += 1; n })[0].cid;
        ob.on_promise(cid, 3_000.0, 0.0);
        assert_eq!(pump(&mut ob, 1_000.0, &mut n), 0, "still awaiting");
        // Past its timeout the sweep re-queues it (with back-off)...
        assert_eq!(pump(&mut ob, 4_000.0, &mut n), 0, "swept to a backed-off retry");
        // ...and once the back-off elapses it re-fires. Nothing stuck.
        assert_eq!(pump(&mut ob, 7_000.0, &mut n), 1);
    }

    #[test]
    fn lost_reply_retries_after_ttl() {
        let mut ob = Outbox::new();
        let (h, mut n) = (call_hash("request_zone", 1), 0);
        ob.want(h, "request_zone", a(), 0.0);
        assert_eq!(pump(&mut ob, 0.0, &mut n), 1);
        assert_eq!(pump(&mut ob, 4_000.0, &mut n), 0, "in-flight TTL not yet up");
        // Past the TTL the sweep re-queues it with back-off; then it re-fires.
        assert_eq!(pump(&mut ob, 6_000.0, &mut n), 0, "swept to a backed-off retry");
        assert_eq!(pump(&mut ob, 9_000.0, &mut n), 1, "back-off elapsed → resend");
    }
}
