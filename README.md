<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/logo-dark.svg">
    <img src=".github/assets/logo-light.svg" alt="MagicMtn logo" width="180" height="180">
  </picture>
</p>

# Ollama provider for MagicMtn

Run chat, tool calls, and text embeddings on your own hardware with MagicMtn and Ollama. This provider translates MagicMtn's shared inference format to Ollama's native API and runs as an independently installed executable.

## Requirements

- A self-hosted MagicMtn installation and the `magic` CLI.
- Ollama running at `http://127.0.0.1:11434` on the machine running the provider, with your chosen models already downloaded.

The provider connects to Ollama at the loopback address in its own network namespace. When running the provider in a container, make Ollama available at that address inside the container.

## Install

On an Apple silicon Mac, install from the MagicMtn Registry.

```sh
magic provider install ollama
magic provider list
```

To build an archive for your host, follow the [source build setup](CONTRIBUTING.md#development-setup), then package and install it.

```sh
./package.sh
magic provider install --file dist/ollama-0.1.0-macos-aarch64.tar.gz
```

Use the archive filename printed by `package.sh`. Install it on the host that runs the provider. MagicMtn manages the executable after installation.

## Use

Start Ollama if it is not already running (`ollama serve`), download the models you intend to use with `ollama pull`, and check them with `ollama list`. Installing this provider does not download models or register them in MagicMtn.

Register models in MagicMtn with provider `ollama` and the full Ollama model name, including its tag. Put the tag in the model name rather than the separate model version field. Choose a model with tool-call support for agents that execute tools, and an embedding model for embedding requests.

Supported behavior:

- Non-streaming chat, including tool calls and tool results.
- Batched text embeddings, without automatic truncation.
- Default model parameters, or exactly `{"temperature": 0}` for chat.

Use models downloaded to your local Ollama server. Requests have a five-minute HTTP timeout.

If inference fails, check that Ollama is reachable from the provider's host, the exact model name appears in `ollama list`, and the model supports the requested task.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development, the Rust library API, and packaging.

## License

[Apache License 2.0](LICENSE).
