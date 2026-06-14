//! Transport seam.
//!
//! The client speaks the gate protocol ([`resonantdust_protocol::protocol`]) over
//! *some* duplex channel of binary frames. This trait is that seam: the native
//! build (here) uses a real WebSocket via tokio-tungstenite; the browser build
//! implements the same trait over a `web-sys` WebSocket — the wasm client owns
//! the socket. Everything above this (serde, subscription bookkeeping,
//! prediction) is transport-agnostic and shared.
//!
//! Frames are raw bytes (WebSocket *binary* frames). The protocol layer above
//! does the encode/decode — JSON-over-bytes today, migrating to postcard
//! per-direction — so the transport stays a dumb byte pipe either way.

/// A duplex channel of binary frames to/from a gateway. The protocol layer above
/// does the `ClientMsg`/`GateMsg` encode/decode, so the transport stays a dumb
/// pipe that any backend can satisfy.
#[allow(async_fn_in_trait)] // single-task client; no `dyn Transport` / Send bound needed yet
pub trait Transport {
    /// Send one frame (an encoded `ClientMsg`).
    async fn send(&mut self, frame: Vec<u8>) -> anyhow::Result<()>;
    /// Receive the next frame (an encoded `GateMsg`), or `None` at EOF.
    async fn recv(&mut self) -> anyhow::Result<Option<Vec<u8>>>;
}

// Native WebSocket transport — NPC driver + integration tests. Excluded from
// wasm32, where the browser build supplies a web-sys implementation instead.
#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::Transport;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpStream;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

    /// A live WebSocket to a gateway (`ws://…/ws`).
    pub struct WsTransport {
        ws: WebSocketStream<MaybeTlsStream<TcpStream>>,
    }

    impl WsTransport {
        /// Open a connection. `url` is the gate WS endpoint, e.g.
        /// `ws://gate-claude:8474/ws`.
        pub async fn connect(url: &str) -> anyhow::Result<Self> {
            let (ws, _resp) = connect_async(url).await?;
            Ok(Self { ws })
        }
    }

    impl Transport for WsTransport {
        async fn send(&mut self, frame: Vec<u8>) -> anyhow::Result<()> {
            self.ws.send(Message::Binary(frame.into())).await?;
            Ok(())
        }

        async fn recv(&mut self) -> anyhow::Result<Option<Vec<u8>>> {
            while let Some(msg) = self.ws.next().await {
                match msg? {
                    Message::Binary(b) => return Ok(Some(b.to_vec())),
                    // Tolerate a text frame (shouldn't occur now the wire is
                    // binary) by surfacing its bytes.
                    Message::Text(t) => return Ok(Some(t.as_bytes().to_vec())),
                    // tungstenite auto-replies to pings; nothing to surface.
                    Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => continue,
                    Message::Close(_) => return Ok(None),
                }
            }
            Ok(None)
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::WsTransport;
