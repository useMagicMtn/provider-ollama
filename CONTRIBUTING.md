# Contributing

Preserve the inference contract for users running MagicMtn and Ollama on their own hardware.

## Development setup

Use a Rust toolchain supporting edition 2024 and the public SDK revision recorded in [SDK_REVISION](SDK_REVISION). Packaging also needs Python 3.11+ for its standard-library `tomllib` module. Tests use a loopback mock HTTP server and require neither Ollama nor downloaded models.

The SDK and provider framework currently come from a sibling MagicMtn source checkout:

```text
src/
  magicmtn/
  provider-ollama/
```

```sh
test -f .cargo/config.toml || cp .cargo/config.toml.example .cargo/config.toml
cargo build --locked
```

Adjust the paths in the ignored `.cargo/config.toml` for your checkout layout. `Cargo.lock` records the current SDK and framework dependency graph, including the shared credential and protocol types. The source patches apply to builds; the packaged executable runs independently of the SDK checkout.

## Repository layout

- `src/main.rs`: executable entry point and provider identity.
- `src/lib.rs`: `InferenceBackend`, shared capability declaration, Ollama HTTP translation, validation, and mock-server tests.
- `tests/sdk_host.rs`: persistent RPC discovery, failure handling, and package installation through the current SDK host.
- `package.sh`: native release build and provider archive generation.
- `.cargo/config.toml.example`: portable starting point for local SDK paths.
- `.github/workflows/ci.yml`: formatting, lint, and tests against a separate SDK checkout.

CI checks out the public MagicMtn repository at `SDK_REVISION` and runs on macOS ARM64 and x86-64/ARM64 Linux. It verifies formatting, lint, both feature configurations, the packaged executable, and installation through the standard SDK host. Local builds need no GitHub credentials.

## Inference contract

The executable identifies as `ollama`, advertises `chat` and `embedding`, and serves persistent, newline-delimited JSON RPC on stdin/stdout through the shared provider framework. Keep stdout reserved for protocol responses. Discovery does not contact Ollama. The manifest has no credential or account-setup requirement and declares no provider-distributed model choices or serving lifecycle operations. Users select models already available on their own Ollama instance. Protocol v1 is the current SDK contract for this credential-free adapter; protocol v2 is for providers with explicit broker credentials.

Chat input contains `messages` and optional `tools`. The SDK normalizes shared chat messages before translation. Outgoing Ollama calls use object-valued tool arguments and native tool names; replies restore shared tool-call IDs and JSON-encoded arguments. Replies must be complete assistant messages. Additional response fields are retained in `usage`. The existing `think` setting is enabled for names beginning with `qwen3:` and disabled otherwise.

Embedding input is `{"texts":["first","second"]}`; output is `{"vectors":[[...],[...]]}`. The provider sends `truncate: false` and checks that Ollama returns one vector per input text.

Model configuration accepts provider `ollama` or an omitted provider, a nonempty model name, and no separate version. Parameters must be null, `{}`, or exactly `{"temperature":0}` for chat. Unknown tasks and unsupported parameters fail rather than being silently ignored.

The executable always uses `http://127.0.0.1:11434/`. HTTP proxy settings are bypassed and requests time out after 300 seconds. The Rust constructor also accepts other loopback HTTP base URLs, but rejects remote hosts, HTTPS, credentials, path prefixes, and queries. Model names ending in `:cloud` are rejected. These checks preserve the current local-only contract; do not broaden it as incidental cleanup.

## Rust library

Disable the default `plugin` feature to embed the provider without the RPC executable:

```toml
[dependencies]
magicmtn-provider-ollama = { path = "../provider-ollama", default-features = false }
```

`OllamaProvider::localhost()` uses the default listener. `OllamaProvider::new(url::Url)` accepts a validated loopback base URL. Both implement the SDK's `InferenceBackend`; callers provide their own async runtime. `VERSION` exposes the crate version.

## Verification

```sh
cargo fmt --check
cargo test --locked
cargo test --locked --no-default-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo clippy --locked --all-targets --no-default-features -- -D warnings
```

Mock tests cover request translation, tool round trips, input validation, and failed or malformed responses. Add regression coverage for changes to those contracts. Keep `Cargo.lock` stable unless intentionally changing dependencies.

For HTTP or protocol changes, also validate the compiled provider through a self-hosted MagicMtn installation with a local Ollama service: chat, a tool-call/result round trip, and batched embeddings. Check discovery without Ollama running. Report live validation separately from mock tests; mocks do not verify model compatibility or hardware capacity.

## Packaging and releases

Build natively on the operating system and architecture you intend to distribute, using the default Cargo target directory and no cross-compilation target:

```sh
./package.sh
cargo test --locked --test sdk_host maintainer_archive_installs_and_dispatches_with_current_sdk -- --ignored
magic provider install --file dist/ollama-0.1.1-macos-aarch64.tar.gz
```

Substitute your generated archive's filename. The script writes `dist/ollama-<version>-<os>-<arch>.tar.gz` containing `provider`, `package.json`, and `LICENSE`. The manifest records protocol version 1 from `package.metadata.magicmtn` in `Cargo.toml` and the executable's SHA-256 checksum. The archive test installs it into a temporary SDK Registry, verifies discovery, checks inference rejection and persistent dispatch, and shuts the process down. Rust, the SDK checkout, and Python are build requirements; Ollama and its model files remain external runtime requirements.

Before publishing, inspect the archive, verify its checksum and discovery response, and run the live inference checks on each platform you distribute. Test installation with the open-source MagicMtn CLI. Never overwrite an existing released version with different binary contents: use a new version and digest, and retain older artifacts for existing session bindings. Include the Apache-2.0 license in distributions.

## Submitting changes

Keep changes focused and describe the user-visible effect, compatibility implications, and checks performed. Do not include local SDK configuration, build artifacts, downloaded models, credentials, or machine-specific state. Contributions are provided under this repository's Apache-2.0 license.
