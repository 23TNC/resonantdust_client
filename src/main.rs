//! Resonant Dust client core — entry point.
//!
//! This crate grows into the shared game-logic client — run headless to drive
//! NPCs (native target), and compiled to wasm to back the pixijs view (which
//! then handles only drawing + input). Prediction, recipe evaluation, card
//! lifecycle, and the stack model live here, sharing compiled code with the gate
//! so the human client and NPCs can't drift.
//!
//! Milestone: prove the gate connection. Connect over the [`transport`] seam,
//! drive it through a [`gate::GateConnection`], subscribe, and read back the
//! gate's `time` heartbeat + the `applied` for our subscription — the round-trip
//! that everything else builds on.

mod gate;
mod transport;

use std::time::Duration;

use gate::GateConnection;
use transport::WsTransport;

/// Gate WS endpoint. Defaults to the **claude** gate by its service name on the
/// shared `resonantdust` docker network (env isolation — never the dev gate).
/// Override with `GATE_URL`.
fn gate_url() -> String {
    std::env::var("GATE_URL").unwrap_or_else(|_| "ws://gate-claude:8474/ws".to_string())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let url = gate_url();
    println!("client: connecting to {url}");
    let mut conn = GateConnection::new(WsTransport::connect(&url).await?);
    println!("client: connected");

    // Prove the request path: subscribe to the (small) regions table. `applied`
    // comes back once initial rows land — even zero rows — so this round-trips
    // regardless of world state.
    let sid = conn.subscribe("regions", None).await?;
    println!("client: → subscribed regions (sid {sid})");

    // Read frames until the socket goes idle. Expect an immediate `time`
    // heartbeat (the gate sends one on connect) and an `applied { sid }`.
    loop {
        match tokio::time::timeout(Duration::from_secs(3), conn.next()).await {
            Ok(Ok(Some(msg))) => println!("client: ← {msg:?}"),
            Ok(Ok(None)) => {
                println!("client: socket closed");
                break;
            }
            Ok(Err(e)) => {
                println!("client: error: {e}");
                break;
            }
            Err(_) => {
                println!("client: idle 3s — done");
                break;
            }
        }
    }

    Ok(())
}
