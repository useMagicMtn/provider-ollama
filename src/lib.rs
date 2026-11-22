//! Inference through a locally running Ollama server.
//!
//! Supported task schemas:
//! - `chat`: `{"messages":[{"role":"user","content":"..."}]}` returns
//!   `{"message":{"role":"assistant","content":"..."}}`.
//! - `embedding`: `{"texts":["..."]}` returns `{"vectors":[[...]]}`.

/// Version of this independently released provider library.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use std::net::IpAddr;
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use url::{Host, Url};

use magicmtn_sdk::{BoxFuture, InferenceBackend, InferenceRequest, InferenceResponse};

#[derive(Clone)]
pub struct OllamaProvider {
    client: Client,
    base_url: Url,
}

impl OllamaProvider {
    /// Connect to the usual local Ollama listener. Ollama must already be running.
    pub fn localhost() -> Self {
        Self::new(Url::parse("http://127.0.0.1:11434/").expect("valid Ollama URL"))
            .expect("valid local Ollama URL")
    }

    /// Use another loopback listener, for example a local test server.
    pub fn new(base_url: Url) -> Result<Self> {
        let loopback = match base_url.host() {
            Some(Host::Domain("localhost")) => true,
            Some(Host::Ipv4(ip)) => IpAddr::V4(ip).is_loopback(),
            Some(Host::Ipv6(ip)) => IpAddr::V6(ip).is_loopback(),
            _ => false,
        };
        ensure!(
            base_url.scheme() == "http" && loopback,
            "Ollama URL must use local HTTP"
        );
        ensure!(
            base_url.username().is_empty() && base_url.password().is_none(),
            "Ollama URL must not contain credentials"
        );
        ensure!(
            base_url.path() == "/" && base_url.query().is_none(),
            "Ollama URL must be a base URL"
        );
        let client = Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self { client, base_url })
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value> {
        let url = self.base_url.join(path)?;
        let response = self
            .client
            .post(url)
            .json(&body)
            .send()
            .await
            .context("could not reach local Ollama; start the Ollama app")?
            .error_for_status()
            .context("Ollama rejected the inference request")?;
        response.json().await.context("invalid Ollama response")
    }

    async fn infer_inner(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        if let Some(provider) = &request.model.provider {
            ensure!(provider == "ollama", "model provider must be ollama");
        }
        ensure!(
            !request.model.model.trim().is_empty(),
            "model name is required"
        );
        ensure!(
            !request.model.model.ends_with(":cloud"),
            "Ollama cloud models are not supported by the local provider"
        );
        ensure!(
            request.model.version.is_none(),
            "put the model version in the Ollama model name"
        );
        ensure!(
            request.model.parameters.is_null()
                || request.model.parameters == json!({})
                || (request.task == "chat"
                    && request.model.parameters == json!({"temperature": 0})),
            "unsupported Ollama inference parameters"
        );

        let model = &request.model.model;
        match request.task.as_str() {
            "chat" => {
                let input: ChatInput =
                    serde_json::from_value(magicmtn_sdk::chat::normalize_input(&request.input)?)
                        .context("chat input must contain messages with role and content")?;
                ensure!(
                    !input.messages.is_empty(),
                    "chat needs at least one message"
                );
                let mut body = json!({
                    "model": model, "messages": ollama_messages(input.messages)?, "tools": input.tools,
                    "stream": false, "think": model.starts_with("qwen3:")
                });
                if request.model.parameters == json!({"temperature": 0}) {
                    body["options"] = request.model.parameters.clone();
                }
                let raw = self.post("api/chat", body).await?;
                let result: ChatResult =
                    serde_json::from_value(raw).context("invalid Ollama chat result")?;
                ensure!(result.done, "Ollama chat result is incomplete");
                ensure!(
                    result.message.role == "assistant",
                    "Ollama did not return an assistant message"
                );
                for call in &result.message.tool_calls {
                    ensure!(
                        call["function"]["name"]
                            .as_str()
                            .is_some_and(|name| !name.is_empty()),
                        "malformed Ollama tool name"
                    );
                    ensure!(
                        call["function"]["arguments"].is_object(),
                        "malformed Ollama tool arguments"
                    );
                }
                Ok(InferenceResponse {
                    model: result.model,
                    output: json!({"message": magicmtn_sdk::chat::normalize_message(
                        &json!({"role":"assistant","content":result.message.content,"tool_calls":result.message.tool_calls}),
                        &format!("call_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_nanos()))?, "usage": result.usage}),
                })
            }
            "embedding" => {
                let input: EmbeddingInput = serde_json::from_value(request.input.clone())
                    .context("embedding input must contain texts")?;
                ensure!(!input.texts.is_empty(), "embedding needs at least one text");
                let count = input.texts.len();
                let raw = self
                    .post(
                        "api/embed",
                        json!({
                            "model": model, "input": input.texts, "truncate": false
                        }),
                    )
                    .await?;
                let result: EmbeddingResult =
                    serde_json::from_value(raw).context("invalid Ollama embedding result")?;
                ensure!(
                    result.embeddings.len() == count,
                    "Ollama returned the wrong number of embeddings"
                );
                Ok(InferenceResponse {
                    model: result.model,
                    output: json!({"vectors": result.embeddings}),
                })
            }
            other => bail!("Ollama does not support inference task {other:?}"),
        }
    }
}

impl InferenceBackend for OllamaProvider {
    fn infer<'a>(&'a self, request: &'a InferenceRequest) -> BoxFuture<'a, InferenceResponse> {
        Box::pin(self.infer_inner(request))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatInput {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    tools: Vec<Value>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: ChatRole,
    content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

// Keep Ollama's native wire format behind the shared chat contract.
fn ollama_messages(messages: Vec<ChatMessage>) -> Result<Vec<Value>> {
    messages
        .into_iter()
        .map(|message| {
            let mut value = serde_json::to_value(message)?;
            if value["role"] == "tool" {
                if let Some(name) = value.get("name").cloned() {
                    value["tool_name"] = name;
                }
                value.as_object_mut().unwrap().remove("name");
                value.as_object_mut().unwrap().remove("tool_call_id");
            }
            if let Some(calls) = value.get_mut("tool_calls").and_then(Value::as_array_mut) {
                for call in calls {
                    if let Some(encoded) = call["function"]["arguments"].as_str() {
                        call["function"]["arguments"] = serde_json::from_str(encoded)?;
                    }
                    call.as_object_mut()
                        .context("tool call required")?
                        .remove("id");
                    call.as_object_mut().unwrap().remove("type");
                }
            }
            Ok(value)
        })
        .collect()
}

#[derive(Deserialize)]
struct ChatResult {
    model: String,
    message: ChatResultMessage,
    done: bool,
    #[serde(flatten)]
    usage: std::collections::BTreeMap<String, Value>,
}

#[derive(Deserialize)]
struct ChatResultMessage {
    role: String,
    content: String,
    #[serde(default)]
    tool_calls: Vec<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingInput {
    texts: Vec<String>,
}

#[derive(Deserialize)]
struct EmbeddingResult {
    model: String,
    embeddings: Vec<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use axum::{Router, extract::Json, routing::post};
    use serde_json::{Value, json};
    use tokio::net::TcpListener;

    use super::*;
    use magicmtn_sdk::inference::ModelConfiguration;

    fn request(model: &str, task: &str, input: Value) -> InferenceRequest {
        InferenceRequest {
            model: ModelConfiguration {
                model: model.into(),
                provider: Some("ollama".into()),
                version: None,
                parameters: Value::Null,
            },
            task: task.into(),
            input,
        }
    }

    #[test]
    fn constructor_preserves_local_endpoint_policy() {
        for endpoint in [
            "http://127.0.0.1:11434/",
            "http://127.0.0.2:1234/",
            "http://localhost:11434/",
            "http://[::1]:11434/",
        ] {
            assert!(OllamaProvider::new(Url::parse(endpoint).unwrap()).is_ok());
        }
        for endpoint in [
            "http://192.0.2.1:11434/",
            "http://example.com/",
            "https://localhost/",
            "http://user:password@localhost/",
            "http://localhost/api/",
            "http://localhost/?model=test",
        ] {
            assert!(
                OllamaProvider::new(Url::parse(endpoint).unwrap()).is_err(),
                "accepted {endpoint}"
            );
        }
    }

    #[tokio::test]
    async fn rejects_unsupported_requests_before_contacting_ollama() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider = OllamaProvider::new(
            Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap(),
        )
        .unwrap();
        let app = Router::new().fallback(|| async { axum::http::StatusCode::IM_A_TEAPOT });
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let valid = request(
            "chat-model",
            "chat",
            json!({"messages":[{"role":"user","content":"Hello"}]}),
        );
        let mut cases = Vec::new();
        for (model, error) in [
            ("  ", "model name is required"),
            ("model:cloud", "cloud models are not supported"),
        ] {
            let mut invalid = valid.clone();
            invalid.model.model = model.into();
            cases.push((invalid, error));
        }
        let mut invalid = valid.clone();
        invalid.model.provider = Some("another-provider".into());
        cases.push((invalid, "model provider must be ollama"));
        let mut invalid = valid.clone();
        invalid.model.version = Some("v1".into());
        cases.push((invalid, "put the model version in the Ollama model name"));
        for parameters in [
            json!({"temperature": 1}),
            json!({"temperature": 0, "seed": 1}),
        ] {
            let mut invalid = valid.clone();
            invalid.model.parameters = parameters;
            cases.push((invalid, "unsupported Ollama inference parameters"));
        }
        cases.push((
            request("chat-model", "chat", json!({"messages":[]})),
            "chat needs at least one message",
        ));
        cases.push((
            request("embed-model", "embedding", json!({"texts":[]})),
            "embedding needs at least one text",
        ));
        let mut invalid = request("embed-model", "embedding", json!({"texts":["Hello"]}));
        invalid.model.parameters = json!({"temperature": 0});
        cases.push((invalid, "unsupported Ollama inference parameters"));
        for (invalid, expected) in cases {
            let error = provider.infer(&invalid).await.unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?}, got {error:?}"
            );
        }
        server.abort();
    }

    #[tokio::test]
    async fn rejects_failed_or_invalid_ollama_responses() {
        let chat = json!({"model":"chat-model","done":true,
            "message":{"role":"assistant","content":"Hello"}});
        let mut incomplete = chat.clone();
        incomplete["done"] = json!(false);
        let mut wrong_role = chat.clone();
        wrong_role["message"]["role"] = json!("user");
        let mut bad_name = chat.clone();
        bad_name["message"]["tool_calls"] = json!([{"function":{"name":"","arguments":{}}}]);
        let mut bad_arguments = chat;
        bad_arguments["message"]["tool_calls"] =
            json!([{"function":{"name":"tool","arguments":"{}"}}]);
        for (task, status, response, expected) in [
            ("chat", 500, json!({"error":"failed"}), "Ollama rejected"),
            ("chat", 200, json!({}), "invalid Ollama chat result"),
            ("chat", 200, incomplete, "result is incomplete"),
            ("chat", 200, wrong_role, "did not return an assistant"),
            ("chat", 200, bad_name, "malformed Ollama tool name"),
            (
                "chat",
                200,
                bad_arguments,
                "malformed Ollama tool arguments",
            ),
            (
                "embedding",
                200,
                json!({"model":"embed-model","embeddings":[]}),
                "wrong number of embeddings",
            ),
        ] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let provider = OllamaProvider::new(
                Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap(),
            )
            .unwrap();
            let app = Router::new().fallback(move || async move {
                (
                    axum::http::StatusCode::from_u16(status).unwrap(),
                    Json(response),
                )
            });
            let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            let input = if task == "chat" {
                json!({"messages":[{"role":"user","content":"Hello"}]})
            } else {
                json!({"texts":["Hello"]})
            };
            let error = provider
                .infer(&request("model", task, input))
                .await
                .unwrap_err()
                .to_string();
            assert!(
                error.contains(expected),
                "expected {expected:?}, got {error:?}"
            );
            server.abort();
        }
    }

    #[tokio::test]
    async fn maps_chat_and_embeddings_without_changing_task_shapes() {
        let app = Router::new()
            .route(
                "/api/chat",
                post(|Json(body): Json<Value>| async move {
                    assert_eq!(body["model"], "chat-model");
                    assert_eq!(body["messages"][0]["content"], "Hello");
                    assert_eq!(body["stream"], false);
                    Json(json!({"model":"chat-model","done":true,
                    "message":{"role":"assistant","content":"Hi"}}))
                }),
            )
            .route(
                "/api/embed",
                post(|Json(body): Json<Value>| async move {
                    assert_eq!(body["model"], "embed-model");
                    assert_eq!(body["input"], json!(["first", "second"]));
                    assert_eq!(body["truncate"], false);
                    Json(json!({"model":"embed-model","embeddings":[[0.1,0.2],[0.3,0.4]]}))
                }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider = OllamaProvider::new(base_url).unwrap();

        let chat = provider
            .infer(&request(
                "chat-model",
                "chat",
                json!({
                    "messages":[{"role":"user","content":"Hello"}]
                }),
            ))
            .await
            .unwrap();
        assert_eq!(
            chat.output,
            json!({"message":{"role":"assistant","content":"Hi","tool_calls":[]},"usage":{}})
        );

        let embedding = provider
            .infer(&request(
                "embed-model",
                "embedding",
                json!({
                    "texts":["first", "second"]
                }),
            ))
            .await
            .unwrap();
        assert_eq!(embedding.output["vectors"].as_array().unwrap().len(), 2);

        let unsupported = provider
            .infer(&request("chat-model", "decision", json!({})))
            .await;
        assert!(
            unsupported
                .unwrap_err()
                .to_string()
                .contains("does not support")
        );
        server.abort();
    }

    #[tokio::test]
    async fn shared_tool_ids_and_encoded_arguments_map_back_to_ollama() {
        let app = Router::new().route("/api/chat", post(|Json(body): Json<Value>| async move {
            let messages = &body["messages"];
            assert!(messages[0].get("tool_calls").is_none());
            assert_eq!(messages[1]["tool_calls"][0]["function"]["arguments"], json!({"program":"echo","args":["a\\b","雪"]}));
            assert!(messages[1]["tool_calls"][0].get("id").is_none());
            assert_eq!(messages[2]["tool_name"], "run_command");
            assert!(messages[2].get("tool_call_id").is_none());
            Json(json!({"model":"chat-model","done":true,"message":{"role":"assistant","content":"Done"}}))
        }));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let provider = OllamaProvider::new(
            Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap(),
        )
        .unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let args = serde_json::to_string(&json!({"program":"echo","args":["a\\b","雪"]})).unwrap();
        let response = provider.infer(&request("chat-model","chat",json!({"messages":[
            {"role":"user","content":"echo"},
            {"role":"assistant","content":"","tool_calls":[{"id":"call_test","type":"function","function":{"name":"run_command","arguments":args}}]},
            {"role":"tool","tool_call_id":"call_test","name":"run_command","content":"done"}]
        }))).await.unwrap();
        assert_eq!(response.output["message"]["content"], "Done");
        server.abort();
    }

    #[tokio::test]
    async fn maps_tool_call_and_result_message() {
        let app = Router::new().route(
            "/api/chat",
            post(|Json(body): Json<Value>| async move {
                assert_eq!(body["messages"][1]["role"], "tool");
                assert_eq!(body["messages"][1]["tool_name"], "run_command");
                assert_eq!(body["options"], json!({"temperature": 0}));
                assert_eq!(body["think"], true);
                Json(
                    json!({"model":"qwen3:0.6b","done":true,"prompt_eval_count":42,
                    "message":{"role":"assistant","content":"","tool_calls":[{
                        "function":{"name":"run_command","arguments":{"program":"pwd","args":[]}}
                    }]}}),
                )
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let provider = OllamaProvider::new(base_url).unwrap();
        let mut input = request(
            "qwen3:0.6b",
            "chat",
            json!({
                "messages":[{"role":"user","content":"pwd"},{"role":"tool","tool_name":"run_command","content":"/tmp"}],
                "tools":[]
            }),
        );
        input.model.parameters = json!({"temperature": 0});
        let result = provider.infer(&input).await.unwrap();
        assert_eq!(
            result.output["message"]["tool_calls"][0]["function"]["name"],
            "run_command"
        );
        assert_eq!(result.output["usage"]["prompt_eval_count"], 42);
        server.abort();
    }
}
