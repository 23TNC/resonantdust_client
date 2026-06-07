//! Gate connection orchestration over a [`Transport`].
//!
//! Owns the boring-but-load-bearing bookkeeping the raw transport doesn't: it
//! allocates subscription ids (`sid`) and call ids (`cid`), offers a typed
//! `subscribe` / `unsubscribe` / `call` surface that serializes [`ClientMsg`],
//! and parses inbound frames back into [`GateMsg`]. The Rust analogue of the TS
//! `GateSubscriptionManager`, shared by the NPC (native) and browser (wasm)
//! builds — only the [`Transport`] underneath differs.
//!
//! Generic over `T: Transport` (no `dyn`) so the async trait methods stay
//! monomorphized and Send-free for the single-task client loop.

use resonantdust_data::protocol::{ClientMsg, GateMsg};

use crate::transport::Transport;

/// A gateway connection: a transport plus the id counters and framing that turn
/// it into a usable subscribe/call channel.
pub struct GateConnection<T: Transport> {
    transport: T,
    next_sid: u32,
    // Used by `call` (tested; first binary caller lands with reducer-driven flows).
    #[allow(dead_code)]
    next_cid: u32,
}

impl<T: Transport> GateConnection<T> {
    /// Wrap an already-connected transport.
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            next_sid: 1,
            next_cid: 1,
        }
    }

    /// Subscribe to `table` (optionally SQL-filtered). Returns the allocated
    /// `sid`; the gate replies with [`GateMsg::Applied`] once initial rows land,
    /// then streams [`GateMsg::Row`] events tagged with this sid's table.
    pub async fn subscribe(&mut self, table: &str, filter: Option<String>) -> anyhow::Result<u32> {
        let sid = self.next_sid;
        self.next_sid += 1;
        self.send(&ClientMsg::Sub {
            sid,
            table: table.to_string(),
            filter,
        })
        .await?;
        Ok(sid)
    }

    /// Drop a subscription by its `sid`.
    #[allow(dead_code)] // API-complete; first caller lands with the world model
    pub async fn unsubscribe(&mut self, sid: u32) -> anyhow::Result<()> {
        self.send(&ClientMsg::Unsub { sid }).await
    }

    /// Call a gate/shard reducer. Returns the allocated `cid`; the gate replies
    /// with [`GateMsg::CallOk`] / [`GateMsg::CallErr`] carrying this cid.
    #[allow(dead_code)] // API-complete; first caller lands with the world model
    pub async fn call(&mut self, reducer: &str, args: serde_json::Value) -> anyhow::Result<u32> {
        let cid = self.next_cid;
        self.next_cid += 1;
        self.send(&ClientMsg::Call {
            cid,
            reducer: reducer.to_string(),
            args,
        })
        .await?;
        Ok(cid)
    }

    /// The next inbound gate frame, parsed — or `None` at EOF.
    pub async fn next(&mut self) -> anyhow::Result<Option<GateMsg>> {
        match self.transport.recv().await? {
            Some(raw) => Ok(Some(serde_json::from_str(&raw)?)),
            None => Ok(None),
        }
    }

    async fn send(&mut self, msg: &ClientMsg) -> anyhow::Result<()> {
        self.transport.send(serde_json::to_string(msg)?).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// In-memory transport: records what was sent, replays canned inbound frames.
    /// This is the point of the data-first client — the whole subscribe/dispatch
    /// path is exercised with zero network and zero renderer.
    #[derive(Default)]
    struct MockTransport {
        sent: Vec<String>,
        incoming: VecDeque<String>,
    }

    // GateConnection takes the transport by value; impl the trait for `&mut` so a
    // test can hand it a borrow and still inspect `sent` afterward.
    impl Transport for &mut MockTransport {
        async fn send(&mut self, frame: String) -> anyhow::Result<()> {
            self.sent.push(frame);
            Ok(())
        }
        async fn recv(&mut self) -> anyhow::Result<Option<String>> {
            Ok(self.incoming.pop_front())
        }
    }

    #[tokio::test]
    async fn allocates_monotonic_ids_and_serializes_requests() {
        let mut t = MockTransport::default();
        {
            let mut conn = GateConnection::new(&mut t);
            let s1 = conn.subscribe("regions", None).await.unwrap();
            let s2 = conn
                .subscribe("zones", Some("macro_zone = 16382".to_string()))
                .await
                .unwrap();
            let c1 = conn.call("ping", serde_json::json!({})).await.unwrap();
            assert_eq!((s1, s2, c1), (1, 2, 1), "sids and cids count up independently");
        }

        assert_eq!(t.sent.len(), 3);
        assert_eq!(t.sent[0], r#"{"t":"sub","sid":1,"table":"regions"}"#);
        assert_eq!(
            t.sent[1],
            r#"{"t":"sub","sid":2,"table":"zones","filter":"macro_zone = 16382"}"#
        );
        assert_eq!(t.sent[2], r#"{"t":"call","cid":1,"reducer":"ping","args":{}}"#);
    }

    #[tokio::test]
    async fn parses_inbound_frames() {
        let mut t = MockTransport::default();
        t.incoming.push_back(r#"{"t":"time","server_micros":"123"}"#.to_string());
        t.incoming.push_back(r#"{"t":"applied","sid":1}"#.to_string());
        let mut conn = GateConnection::new(&mut t);

        assert!(matches!(conn.next().await.unwrap(), Some(GateMsg::Time { .. })));
        assert!(matches!(
            conn.next().await.unwrap(),
            Some(GateMsg::Applied { sid: 1 })
        ));
        assert!(conn.next().await.unwrap().is_none(), "EOF when drained");
    }
}
