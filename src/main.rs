use anyhow::Result;
use magicmtn_provider_ollama::OllamaProvider;
use magicmtn_sdk::plugin::Capabilities;

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    magicmtn_provider_framework::run_inference(
        "ollama",
        env!("CARGO_PKG_VERSION"),
        Capabilities::inference(vec!["chat".into(), "embedding".into()]),
        OllamaProvider::localhost(),
    )
    .await
}
