use extism_pdk::*;
use magi_pdk::DataType;
use serde_json::json;

// =============================================================================
// Plugin exports
// =============================================================================

#[plugin_fn]
pub fn describe() -> FnResult<Json<DataType>> {
    Ok(Json(DataType::from_json(json!({
        "name": "ollama",
        "version": "0.1.0",
        "description": "ACP agent for local LLM inference via Ollama",
        "label": "acp",
        "capabilities": [
            {"name": "chat-completion", "description": "Generate chat responses via local Ollama models"},
            {"name": "code-generation", "description": "Generate code with local models"},
            {"name": "embeddings", "description": "Generate text embeddings"}
        ]
    }))))
}

#[plugin_fn]
pub fn config_schema() -> FnResult<Json<serde_json::Value>> {
    Ok(Json(json!({
        "type": "object",
        "properties": {
            "ollama_url": {
                "type": "string",
                "description": "Ollama API base URL",
                "default": "http://localhost:11434"
            },
            "model": {
                "type": "string",
                "description": "Default model to use",
                "default": "llama3.2"
            }
        }
    })))
}

#[plugin_fn]
pub fn init(Json(_input): Json<DataType>) -> FnResult<Json<DataType>> {
    magi_pdk::log_info("Ollama plugin initialized");
    Ok(Json(DataType::from_json(json!({"success": true}))))
}

#[plugin_fn]
pub fn start() -> FnResult<Json<DataType>> {
    let _ = magi_pdk::agent_register(
        "ollama",
        "Local LLM inference agent via Ollama",
        &[
            ("chat-completion", "Generate chat responses via local Ollama models"),
            ("code-generation", "Generate code with local models"),
            ("embeddings", "Generate text embeddings"),
        ],
    );
    magi_pdk::log_info("Ollama ACP agent registered");
    Ok(Json(DataType::from_json(json!({"status": "running"}))))
}

#[plugin_fn]
pub fn stop() -> FnResult<Json<DataType>> {
    magi_pdk::log_info("Ollama ACP agent stopped");
    Ok(Json(DataType::from_json(json!({"status": "stopped"}))))
}

#[plugin_fn]
pub fn process(Json(input): Json<DataType>) -> FnResult<Json<DataType>> {
    let action = input
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("chat")
        .to_string();

    let config = magi_pdk::get_config().unwrap_or_default();
    let base_url = config
        .get("ollama_url")
        .and_then(|v| v.as_str())
        .unwrap_or("http://localhost:11434");
    let model = config
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("llama3.2");

    match action.as_str() {
        "chat" => chat(base_url, model, &input),
        "generate" => generate(base_url, model, &input),
        "embeddings" => embeddings(base_url, model, &input),
        "list_models" => list_models(base_url),
        "poll" => poll_messages(base_url, model),
        _ => Ok(Json(DataType::from_json(
            json!({"error": format!("unknown action: {action}")}),
        ))),
    }
}

// =============================================================================
// Ollama API
// =============================================================================

fn chat(base_url: &str, model: &str, input: &DataType) -> FnResult<Json<DataType>> {
    let messages = if let Some(msgs) = input.get("messages") {
        msgs.to_json()
    } else if let Some(prompt) = input.get("prompt").and_then(|v| v.as_str()) {
        let system = input
            .get("system")
            .and_then(|v| v.as_str())
            .unwrap_or("You are a helpful assistant.");
        json!([
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ])
    } else {
        return Ok(Json(DataType::from_json(
            json!({"error": "prompt or messages required"}),
        )));
    };

    let use_model = input
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(model);

    let body = json!({
        "model": use_model,
        "messages": messages,
        "stream": false
    });

    let url = format!("{base_url}/api/chat");
    let req = HttpRequest::new(&url)
        .with_method("POST")
        .with_header("Content-Type", "application/json");
    let body_str = serde_json::to_string(&body)?;
    let resp = http::request::<String>(&req, Some(body_str))?;
    let data: serde_json::Value = serde_json::from_slice(&resp.body())?;

    let content = data
        .pointer("/message/content")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    Ok(Json(DataType::from_json(json!({
        "content": content,
        "model": data.get("model").and_then(|v| v.as_str()).unwrap_or(use_model),
        "done": data.get("done").and_then(|v| v.as_bool()).unwrap_or(true),
        "total_duration": data.get("total_duration"),
        "eval_count": data.get("eval_count")
    }))))
}

fn generate(base_url: &str, model: &str, input: &DataType) -> FnResult<Json<DataType>> {
    let prompt = input
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if prompt.is_empty() {
        return Ok(Json(DataType::from_json(
            json!({"error": "prompt is required"}),
        )));
    }

    let use_model = input
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(model);

    let body = json!({
        "model": use_model,
        "prompt": prompt,
        "stream": false
    });

    let url = format!("{base_url}/api/generate");
    let req = HttpRequest::new(&url)
        .with_method("POST")
        .with_header("Content-Type", "application/json");
    let body_str = serde_json::to_string(&body)?;
    let resp = http::request::<String>(&req, Some(body_str))?;
    let data: serde_json::Value = serde_json::from_slice(&resp.body())?;

    Ok(Json(DataType::from_json(json!({
        "response": data.get("response").and_then(|v| v.as_str()).unwrap_or(""),
        "model": data.get("model").and_then(|v| v.as_str()).unwrap_or(use_model),
        "done": data.get("done").and_then(|v| v.as_bool()).unwrap_or(true)
    }))))
}

fn embeddings(base_url: &str, model: &str, input: &DataType) -> FnResult<Json<DataType>> {
    let text = input.get("text").and_then(|v| v.as_str()).unwrap_or("");
    if text.is_empty() {
        return Ok(Json(DataType::from_json(
            json!({"error": "text is required"}),
        )));
    }

    let body = json!({
        "model": model,
        "input": text
    });

    let url = format!("{base_url}/api/embed");
    let req = HttpRequest::new(&url)
        .with_method("POST")
        .with_header("Content-Type", "application/json");
    let body_str = serde_json::to_string(&body)?;
    let resp = http::request::<String>(&req, Some(body_str))?;
    let data: serde_json::Value = serde_json::from_slice(&resp.body())?;

    Ok(Json(DataType::from_json(json!({
        "embeddings": data.get("embeddings"),
        "model": data.get("model").and_then(|v| v.as_str()).unwrap_or(model)
    }))))
}

fn list_models(base_url: &str) -> FnResult<Json<DataType>> {
    let url = format!("{base_url}/api/tags");
    let req = HttpRequest::new(&url)
        .with_header("Accept", "application/json");
    let resp = http::request::<String>(&req, None::<String>)?;
    let data: serde_json::Value = serde_json::from_slice(&resp.body())?;
    Ok(Json(DataType::from_json(data)))
}

fn poll_messages(base_url: &str, model: &str) -> FnResult<Json<DataType>> {
    let messages = magi_pdk::agent_receive(10).unwrap_or_default();
    let mut results = Vec::new();

    for msg in &messages {
        let from = msg.get("from").and_then(|v| v.as_str()).unwrap_or("unknown");
        let payload = msg.get("payload").cloned().unwrap_or(json!(null));
        let prompt = payload
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !prompt.is_empty() {
            let input = DataType::from_json(json!({"prompt": prompt}));
            if let Ok(Json(response)) = chat(base_url, model, &input) {
                let content = response
                    .get("content")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let _ = magi_pdk::agent_send(from, json!({"response": content}));
                results.push(json!({"from": from, "response": content}));
            }
        }
    }

    Ok(Json(DataType::from_json(
        json!({"processed": results.len(), "results": results}),
    )))
}
