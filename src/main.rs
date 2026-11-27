use anyhow::Result;
use magicmtn_provider_ollama::{OllamaProvider, VERSION, capabilities};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    magicmtn_provider_framework::run_inference(
        "ollama",
        VERSION,
        capabilities(),
        OllamaProvider::localhost(),
    )
    .await
}
