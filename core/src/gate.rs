//! Gate framing over a [`Transport`].
//!
//! The thin seam between the byte pipe ([`Transport`], raw frames) and the
//! sans-IO logic core ([`crate::client::Client`], typed `ClientMsg`/`GateMsg`):
//! it just does the encode/decode. The core owns subscription/call id allocation and all
//! state — this layer is stateless. Shared by the NPC (native) and browser
//! (wasm) builds; only the [`Transport`] underneath differs.
//!
//! Generic over `T: Transport` (no `dyn`) so the async trait methods stay
//! monomorphized and Send-free for the single-task client loop.

use resonantdust_protocol::protocol::{ClientMsg, GateMsg};

use crate::transport::Transport;

/// A gateway connection: a transport plus the JSON framing that turns it into a
/// typed `ClientMsg`-out / `GateMsg`-in channel.
pub struct GateConnection<T: Transport> {
    transport: T,
}

impl<T: Transport> GateConnection<T> {
    /// Wrap an already-connected transport.
    pub fn new(transport: T) -> Self {
        Self { transport }
    }

    /// Encode and send one outbound message. The transport carries bytes; the
    /// `ClientMsg` direction is still JSON (utf-8) here — postcard lands in a
    /// later phase, independently of `GateMsg`.
    pub async fn send(&mut self, msg: &ClientMsg) -> anyhow::Result<()> {
        self.transport.send(serde_json::to_vec(msg)?).await
    }

    /// Send a batch in order — e.g. the frames a single [`crate::client::Command`]
    /// expands to via `Client::dispatch`.
    pub async fn send_all(&mut self, msgs: &[ClientMsg]) -> anyhow::Result<()> {
        for m in msgs {
            self.send(m).await?;
        }
        Ok(())
    }

    /// The next inbound gate frame, decoded — or `None` at EOF. `GateMsg` is
    /// postcard (binary); `ClientMsg` (outbound) is still JSON until its own phase.
    pub async fn next(&mut self) -> anyhow::Result<Option<GateMsg>> {
        match self.transport.recv().await? {
            Some(raw) => Ok(Some(postcard::from_bytes(&raw)?)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// In-memory transport: records what was sent, replays canned inbound frames.
    /// The whole framing path is exercised with zero network.
    #[derive(Default)]
    struct MockTransport {
        sent: Vec<Vec<u8>>,
        incoming: VecDeque<Vec<u8>>,
    }

    impl Transport for &mut MockTransport {
        async fn send(&mut self, frame: Vec<u8>) -> anyhow::Result<()> {
            self.sent.push(frame);
            Ok(())
        }
        async fn recv(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
            Ok(self.incoming.pop_front())
        }
    }

    #[tokio::test]
    async fn serializes_outbound_and_parses_inbound() {
        let mut t = MockTransport::default();
        t.incoming.push_back(GateMsg::Time { server_micros: "123".to_string() }.to_bytes());
        t.incoming.push_back(GateMsg::Applied { sid: 1 }.to_bytes());
        {
            let mut conn = GateConnection::new(&mut t);
            conn.send_all(&[
                ClientMsg::Sub { sid: 1, table: "cards".to_string(), filter: None },
                ClientMsg::Call { cid: 1, reducer: "ping".to_string(), args: serde_json::json!({}) },
            ])
            .await
            .unwrap();
            assert!(matches!(conn.next().await.unwrap(), Some(GateMsg::Time { .. })));
            assert!(matches!(conn.next().await.unwrap(), Some(GateMsg::Applied { sid: 1 })));
            assert!(conn.next().await.unwrap().is_none(), "EOF when drained");
        }
        assert_eq!(t.sent[0], br#"{"t":"sub","sid":1,"table":"cards"}"#);
        assert_eq!(t.sent[1], br#"{"t":"call","cid":1,"reducer":"ping","args":{}}"#);
    }
}
