//! Transport-independent MCP dispatch.

use serde_json::{json, Value};

use crate::jsonrpc::{
    parse_message, Incoming, Parsed, Response, INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND,
};
use crate::tool::ToolRegistry;

/// The MCP revision this server implements.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// An MCP server: identity plus tools. Both transports wrap one of these.
#[derive(Debug, Clone)]
pub struct McpServer {
    name: String,
    version: String,
    registry: ToolRegistry,
}

impl McpServer {
    /// A server called `name` (reported in `serverInfo`) offering `registry`.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        registry: ToolRegistry,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            registry,
        }
    }

    /// The tools on offer.
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// Handle one raw message body. `None` means "send nothing" (a
    /// notification, or a response from the client).
    pub async fn handle_bytes(&self, body: &[u8]) -> Option<Response> {
        match parse_message(body) {
            Err(error_response) => Some(error_response),
            Ok(Parsed::ClientResponse) => None,
            Ok(Parsed::Message(incoming)) => self.dispatch(incoming).await,
        }
    }

    /// Dispatch a parsed message. Notifications (no id) never get a response,
    /// even when the method is unknown.
    pub async fn dispatch(&self, incoming: Incoming) -> Option<Response> {
        let Incoming { id, method, params } = incoming;
        let result = match method.as_str() {
            "initialize" => self.initialize(&params),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({"tools": self.registry.descriptors()})),
            "tools/call" => self.call_tool(&params).await,
            m if m.starts_with("notifications/") => Ok(Value::Null),
            _ => Err((METHOD_NOT_FOUND, format!("method not found: {method}"))),
        };
        // A notification is answered with silence, whatever happened.
        let id = id?;
        Some(match result {
            Ok(value) => Response::ok(id, value),
            Err((code, message)) => Response::err(id, code, message),
        })
    }

    fn initialize(&self, params: &Value) -> Result<Value, (i64, String)> {
        if !params.is_object() {
            return Err((INVALID_PARAMS, "initialize params must be an object".into()));
        }
        // The spec lets the server answer with the version it supports; the
        // client then decides whether to continue.
        Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": self.name, "version": self.version},
        }))
    }

    async fn call_tool(&self, params: &Value) -> Result<Value, (i64, String)> {
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return Err((INVALID_PARAMS, "tools/call needs a string `name`".into()));
        };
        let arguments = match params.get("arguments") {
            None | Some(Value::Null) => json!({}),
            Some(v @ Value::Object(_)) => v.clone(),
            Some(_) => return Err((INVALID_PARAMS, "`arguments` must be an object".into())),
        };
        let Some(tool) = self.registry.get(name) else {
            return Err((INVALID_PARAMS, format!("unknown tool: {name}")));
        };
        let (text, is_error) = match tool.call(arguments).await {
            Ok(text) => (text, false),
            Err(text) => (text, true),
        };
        Ok(json!({
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }))
    }
}

/// `true` when `response` is the "this was not a request at all" error, which
/// HTTP maps to status 400.
#[must_use]
pub fn is_protocol_level_rejection(response: &Response) -> bool {
    matches!(&response.outcome, Err(e) if e.code == INVALID_REQUEST || e.code == crate::jsonrpc::PARSE_ERROR)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jsonrpc::PARSE_ERROR;
    use crate::tool::{arg_str, Tool};

    fn server() -> McpServer {
        let registry = ToolRegistry::new()
            .with(Tool::new(
                "echo",
                "Echo `text` back.",
                json!({"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}),
                |args| async move { arg_str(&args, "text").map(str::to_owned) },
            ))
            .with(Tool::new(
                "fail",
                "Always fails.",
                json!({"type":"object"}),
                |_| async { Err("boom".to_owned()) },
            ));
        McpServer::new("test", "1.2.3", registry)
    }

    async fn rpc(s: &McpServer, body: &str) -> Value {
        s.handle_bytes(body.as_bytes())
            .await
            .expect("a response")
            .to_value()
    }

    #[tokio::test]
    async fn initialize_reports_version_capabilities_and_identity() {
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"c","version":"0"}}}"#,
        )
        .await;
        assert_eq!(v["id"], 1);
        assert_eq!(v["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(v["result"]["capabilities"], json!({"tools": {}}));
        assert_eq!(
            v["result"]["serverInfo"],
            json!({"name": "test", "version": "1.2.3"})
        );
    }

    #[tokio::test]
    async fn initialize_without_params_is_invalid_params() {
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":1,"method":"initialize"}"#,
        )
        .await;
        assert_eq!(v["error"]["code"], INVALID_PARAMS);
    }

    #[tokio::test]
    async fn initialized_notification_gets_no_response() {
        let r = server()
            .handle_bytes(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
            .await;
        assert!(r.is_none());
        // Even an unknown notification is silent.
        let r = server()
            .handle_bytes(br#"{"jsonrpc":"2.0","method":"no/such"}"#)
            .await;
        assert!(r.is_none());
    }

    #[tokio::test]
    async fn tools_list_names_descriptions_and_schemas() {
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":"a","method":"tools/list"}"#,
        )
        .await;
        assert_eq!(v["id"], "a");
        let tools = v["result"]["tools"].as_array().unwrap();
        let names: Vec<_> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["echo", "fail"]);
        assert_eq!(tools[0]["description"], "Echo `text` back.");
        assert_eq!(tools[0]["inputSchema"]["required"], json!(["text"]));
    }

    #[tokio::test]
    async fn tools_call_returns_text_content() {
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}"#,
        )
        .await;
        assert_eq!(
            v["result"],
            json!({"content":[{"type":"text","text":"hi"}],"isError":false})
        );
    }

    #[tokio::test]
    async fn a_failing_tool_is_is_error_true_not_a_protocol_error() {
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fail"}}"#,
        )
        .await;
        assert_eq!(
            v["result"],
            json!({"content":[{"type":"text","text":"boom"}],"isError":true})
        );
        // Bad arguments surface through the tool's own error text.
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"echo","arguments":{}}}"#,
        )
        .await;
        assert_eq!(v["result"]["isError"], true);
    }

    #[tokio::test]
    async fn tools_call_rejects_bad_params() {
        for params in [
            r#"{}"#,
            r#"{"name":5}"#,
            r#"{"name":"nope"}"#,
            r#"{"name":"echo","arguments":[1]}"#,
        ] {
            let v = rpc(
                &server(),
                &format!(r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{params}}}"#),
            )
            .await;
            assert_eq!(v["error"]["code"], INVALID_PARAMS, "{params}");
        }
    }

    #[tokio::test]
    async fn ping_is_an_empty_result() {
        let v = rpc(&server(), r#"{"jsonrpc":"2.0","id":9,"method":"ping"}"#).await;
        assert_eq!(v["result"], json!({}));
    }

    #[tokio::test]
    async fn unknown_method_is_32601_and_bad_json_is_32700() {
        let v = rpc(
            &server(),
            r#"{"jsonrpc":"2.0","id":1,"method":"resources/list"}"#,
        )
        .await;
        assert_eq!(v["error"]["code"], METHOD_NOT_FOUND);
        assert_eq!(v["id"], 1);
        let v = rpc(&server(), "{oops").await;
        assert_eq!(v["error"]["code"], PARSE_ERROR);
        assert_eq!(v["id"], Value::Null);
    }
}
