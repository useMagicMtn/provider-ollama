#![cfg(feature = "plugin")]

use magicmtn_provider_framework::host::{Client, Registry};
use magicmtn_provider_ollama::{VERSION, capabilities};
use magicmtn_sdk::{
    InferenceRequest,
    inference::ModelConfiguration,
    plugin::{Operation, ProviderError, ProviderManifest, Value},
};
use serde_json::json;
use std::{collections::BTreeMap, path::Path, time::Duration};

fn check_manifest(manifest: &ProviderManifest) {
    assert_eq!(manifest.provider_version, VERSION);
    assert!(manifest.credential.is_none());
    assert_eq!(manifest.capabilities, capabilities());
    assert_eq!(manifest.capabilities.inference_tasks, ["chat", "embedding"]);
    assert!(manifest.capabilities.setup.is_none());
    assert!(manifest.capabilities.inference_models.is_empty());
    assert!(
        manifest
            .capabilities
            .inference_deployment_operations
            .is_empty()
    );
}

fn invalid_inference() -> Operation {
    Operation::Infer {
        request: InferenceRequest {
            model: ModelConfiguration {
                model: "model:cloud".into(),
                provider: Some("ollama".into()),
                version: None,
                parameters: json!({}),
            },
            task: "chat".into(),
            input: json!({"messages":[{"role":"user","content":"private prompt"}]}),
        },
    }
}

#[tokio::test]
async fn current_sdk_discovers_without_ollama_and_keeps_process_after_errors() {
    let client = Client::start(
        Path::new(env!("CARGO_BIN_EXE_magicmtn-provider-ollama")),
        "ollama",
    )
    .await
    .unwrap();
    check_manifest(&client.manifest);
    let error = client
        .call(invalid_inference(), Duration::from_secs(2))
        .await
        .unwrap_err();
    let error = error.downcast_ref::<ProviderError>().unwrap();
    assert_eq!(error.message, "provider operation failed");
    assert!(!error.retryable);
    let error = client
        .call(
            Operation::ProviderConfigure {
                options: BTreeMap::new(),
            },
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "provider does not support this operation"
    );
    let Value::Description(manifest) = client
        .call(Operation::Describe, Duration::from_secs(2))
        .await
        .unwrap()
    else {
        panic!("missing manifest")
    };
    check_manifest(&manifest);
    client.shutdown().await;
}

#[tokio::test]
#[ignore = "requires ./package.sh to build the native release archive"]
async fn maintainer_archive_installs_and_dispatches_with_current_sdk() {
    let root = tempfile::tempdir().unwrap();
    let registry = Registry::new(root.path().to_path_buf()).unwrap();
    let archive = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "dist/ollama-{VERSION}-{}-{}.tar.gz",
        std::env::consts::OS,
        std::env::consts::ARCH,
    ));
    let installed = registry
        .install(&std::fs::read(archive).unwrap())
        .await
        .unwrap();
    assert_eq!(installed.package.name, "ollama");
    assert_eq!(installed.package.version, VERSION);
    assert_eq!(installed.package.protocol_version, 1);
    check_manifest(&installed.manifest);
    let pin = installed.pin();
    let error = registry
        .call(
            &pin,
            "archive-test",
            invalid_inference(),
            Duration::from_secs(2),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<ProviderError>().unwrap().message,
        "provider operation failed"
    );
    let Value::Description(manifest) = registry
        .call(
            &pin,
            "archive-test",
            Operation::Describe,
            Duration::from_secs(2),
        )
        .await
        .unwrap()
    else {
        panic!("missing manifest")
    };
    check_manifest(&manifest);
    registry
        .call(
            &pin,
            "archive-test",
            Operation::Shutdown,
            Duration::from_secs(2),
        )
        .await
        .unwrap();
}
