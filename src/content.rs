//! Content bundle — fetched from the gate's `/content` endpoint.
//!
//! The client is a **content peer**: it loads the exact same `.rd` corpus the
//! gate serves (the gate is the authority — see the content-authority topology),
//! so recipe ids, definitions, and matching agree with the server by
//! construction. We just fetch the pre-serialized payload and feed its `rd`
//! sources to the shared [`load`]er.

use resonantdust_data::loader::{load, Bundle};

/// The `GET /content` body the gate serves (`{ version, rd, locales }`). We only
/// need `rd` for the recipe/def model; `locales` are display strings.
#[derive(serde::Deserialize)]
struct ContentPayload {
    rd: Vec<(String, String)>,
}

/// Fetch + parse the DSL content bundle from `{base_http}/content`.
pub async fn fetch_bundle(base_http: &str) -> anyhow::Result<Bundle> {
    let url = format!("{base_http}/content");
    let payload: ContentPayload = reqwest::get(&url).await?.json().await?;
    load(&payload.rd).map_err(|errs| {
        anyhow::anyhow!(
            "content load: {} error(s); first: {:?}",
            errs.len(),
            errs.first()
        )
    })
}
