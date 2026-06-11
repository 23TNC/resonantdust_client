//! Content bundle — the `.rd` corpus the gate serves at `/content`.
//!
//! The client is a **content peer**: it loads the exact same `.rd` corpus the
//! gate serves (the gate is the authority — see the content-authority topology),
//! so recipe ids, definitions, and matching agree with the server by
//! construction. Parsing is sync ([`parse_bundle`]); only the *fetch* differs by
//! target — the native build pulls it over HTTP ([`fetch_bundle`], reqwest), the
//! wasm build fetches in JS and hands the `rd` pairs straight to [`parse_bundle`].

use resonantdust_dsl::loader::{load, Bundle};

/// The `GET /content` body the gate serves (`{ version, rd, locales }`). We only
/// need `rd` for the recipe/def model; `locales` are display strings.
#[derive(serde::Deserialize)]
pub struct ContentPayload {
    pub rd: Vec<(String, String)>,
}

/// Parse the `rd` source pairs into a [`Bundle`] — the shared DSL loader. Pure /
/// sync; shared by both fetch paths.
pub fn parse_bundle(rd: &[(String, String)]) -> anyhow::Result<Bundle> {
    load(rd).map_err(|errs| {
        anyhow::anyhow!(
            "content load: {} error(s); first: {:?}",
            errs.len(),
            errs.first()
        )
    })
}

/// Fetch + parse the DSL content bundle from `{base_http}/content` (native: the
/// NPC driver + tests). The wasm build fetches in JS instead and calls
/// [`parse_bundle`] directly.
#[cfg(not(target_arch = "wasm32"))]
pub async fn fetch_bundle(base_http: &str) -> anyhow::Result<Bundle> {
    let url = format!("{base_http}/content");
    let payload: ContentPayload = reqwest::get(&url).await?.json().await?;
    parse_bundle(&payload.rd)
}
