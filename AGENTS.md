# Agent development guide

## Scope and setup

This is the Apache-2.0 Ollama inference provider for open-source, self-hosted MagicMtn. Read [CONTRIBUTING.md](CONTRIBUTING.md) for SDK setup, architecture, protocol details, and release verification. Keep the README focused on installation and use.

Copy `.cargo/config.toml.example` to the ignored `.cargo/config.toml` and adjust the SDK paths if necessary. Keep local paths out of source control. Use the loopback test server for automated checks. Build against the public SDK revision pinned in `SDK_REVISION` and keep the CI checkout aligned with that pin.

## Commands

- Build: `cargo build --locked`
- Tests: `cargo test --locked`
- Library-only tests: `cargo test --locked --no-default-features`
- Format: `cargo fmt --check` (apply with `cargo fmt`)
- Lint: `cargo clippy --locked --all-targets --all-features -- -D warnings`
- Library-only lint: `cargo clippy --locked --all-targets --no-default-features -- -D warnings`
- Package: `./package.sh`
- Archive compatibility: `cargo test --locked --test sdk_host maintainer_archive_installs_and_dispatches_with_current_sdk -- --ignored` after packaging.

Run relevant checks after changes. Tests bind ephemeral loopback ports and use mock responses. Do not start services or download models for routine checks. Report any live Ollama/MagicMtn validation separately.

## Change boundaries

- Preserve provider identity `ollama`, protocol version 1, and advertised `chat`/`embedding` capabilities. Stdout belongs to JSON RPC; discovery must work without Ollama running.
- Preserve the public library API, including `VERSION`, `OllamaProvider::localhost`, `OllamaProvider::new`, and builds without the default `plugin` feature. Public API and trait implementations are not dead code merely because this repository has no callers.
- Keep shared chat normalization, tool-call IDs, JSON argument translation, tool-result names, usage fields, and complete-assistant-response checks intact.
- Preserve embedding batching, disabled truncation, and output count validation.
- Preserve loopback HTTP validation, proxy bypass, the five-minute timeout, cloud-model rejection, supported parameter validation, and the existing `qwen3:` thinking behavior. Changes to these policies require explicit scope and regression coverage.
- Do not add automatic service startup, model downloads, retries, remote endpoints, or hosted-service dependencies as cleanup.
- Keep dependencies and `Cargo.lock` stable unless intentionally changing them. Keep `.cargo/config.toml`, `target/`, `dist/`, secrets, and local state out of source control.
- Preserve native package identity, executable checksum, archive layout, and Apache-2.0 licensing. Changed released binaries require new versions and digests; retain older releases for existing bindings.

Use idiomatic Rust and existing formatting. Remove code only after checking public API, SDK trait requirements, feature combinations, and tests. Keep behavioral changes separate from documentation cleanup where possible.
