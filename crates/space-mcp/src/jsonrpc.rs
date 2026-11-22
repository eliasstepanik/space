//! JSON-RPC 2.0 message types and parsing.
//!
//! Only the subset MCP needs: single requests and notifications from the client
//! (the 2025-06-18 revision dropped batching, so an array is an invalid request),
//! and responses carrying either a `result` or an `error`.

use serde::Serialize;
use serde_json::{json, Value};

/// The request body was not valid JSON.
pub const PARSE_ERROR: i64 = -32700;
/// The JSON was valid but not a JSON-RPC request object.
pub const INVALID_REQUEST: i64 = -32600;
/// The method does not exist.
pub const METHOD_NOT_FOUND: i64 = -32601;
/// The method's parameters are invalid.
pub const INVALID_PARAMS: i64 = -32602;
/// The server failed internally.
pub const INTERNAL_ERROR: i64 = -32603;

/// A JSON-RPC error object.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RpcError {
    /// Error code (see the constants in this module).
    pub code: i64,
    /// Short human-readable description.
    pub message: String,
}

impl RpcError {
    /// An error with `code` and `message`.
    #[must_use]
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

/// A JSON-RPC response: exactly one of `result` / `error` is set.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// The id of the request this answers (`null` when it could not be read).
    pub id: Value,
    /// The outcome.
    pub outcome: Result<Value, RpcError>,
}

impl Response {
    /// A success response.
    #[must_use]
    pub fn ok(id: Value, result: Value) -> Self {
        Self {
            id,
            outcome: Ok(result),
        }
    }

    /// An error response.
    #[must_use]
    pub fn err(id: Value, code: i64, message: impl Into<String>) -> Self {
        Self {
            id,
            outcome: Err(RpcError::new(code, message)),
        }
    }

    /// The wire form.
    #[must_use]
    pub fn to_value(&self) -> Value {
        match &self.outcome {
            Ok(result) => json!({"jsonrpc": "2.0", "id": self.id, "result": result}),
            Err(e) => {
                json!({"jsonrpc": "2.0", "id": self.id, "error": {"code": e.code, "message": e.message}})
            }
        }
    }

    /// The wire form as one compact line (no embedded newline).
    #[must_use]
    pub fn to_json(&self) -> String {
        // Hand-assembled so `jsonrpc` comes first and the bytes are stable.
        let id = &self.id;
        match &self.outcome {
            Ok(result) => format!(r#"{{"jsonrpc":"2.0","id":{id},"result":{result}}}"#),
            Err(e) => format!(
                r#"{{"jsonrpc":"2.0","id":{id},"error":{{"code":{},"message":{}}}}}"#,
                e.code,
                Value::String(e.message.clone())
            ),
        }
    }
}

/// A parsed client request or notification.
#[derive(Debug, Clone, PartialEq)]
pub struct Incoming {
    /// `None` for a notification (no response is sent).
    pub id: Option<Value>,
    /// The method name.
    pub method: String,
    /// The `params` member (`Null` when absent).
    pub params: Value,
}

/// What a message body turned out to be.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    /// A request or notification to dispatch.
    Message(Incoming),
    /// A JSON-RPC *response* sent by the client (no method). Servers that issue
    /// no requests have nothing to do with it.
    ClientResponse,
}

/// Parse one message body.
///
/// # Errors
///
/// A ready-to-send error [`Response`]: [`PARSE_ERROR`] (`id: null`) for bad JSON,
/// [`INVALID_REQUEST`] for JSON that is not a request object.
pub fn parse_message(bytes: &[u8]) -> Result<Parsed, Response> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|e| Response::err(Value::Null, PARSE_ERROR, format!("parse error: {e}")))?;
    let invalid = |id: Value, why: &str| Response::err(id, INVALID_REQUEST, why);
    let Value::Object(mut object) = value else {
        return Err(invalid(
            Value::Null,
            "request must be a single JSON object (batches are not supported)",
        ));
    };
    let id = match object.remove("id") {
        None => None,
        Some(v @ (Value::String(_) | Value::Number(_))) => Some(v),
        Some(_) => return Err(invalid(Value::Null, "id must be a string or a number")),
    };
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(invalid(
            id.unwrap_or(Value::Null),
            "jsonrpc must be \"2.0\"",
        ));
    }
    let Some(method) = object.remove("method") else {
        if object.contains_key("result") || object.contains_key("error") {
            return Ok(Parsed::ClientResponse);
        }
        return Err(invalid(id.unwrap_or(Value::Null), "missing method"));
    };
    let Value::String(method) = method else {
        return Err(invalid(
            id.unwrap_or(Value::Null),
            "method must be a string",
        ));
    };
    Ok(Parsed::Message(Incoming {
        id,
        method,
        params: object.remove("params").unwrap_or(Value::Null),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_parses_with_its_id_and_params() {
        let p = parse_message(br#"{"jsonrpc":"2.0","id":7,"method":"ping","params":{"a":1}}"#);
        assert_eq!(
            p,
            Ok(Parsed::Message(Incoming {
                id: Some(json!(7)),
                method: "ping".into(),
                params: json!({"a": 1}),
            }))
        );
    }

    #[test]
    fn a_notification_has_no_id() {
        let Ok(Parsed::Message(m)) =
            parse_message(br#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#)
        else {
            panic!("should parse");
        };
        assert_eq!((m.id, m.params), (None, Value::Null));
    }

    #[test]
    fn bad_json_is_a_parse_error_with_a_null_id() {
        let e = parse_message(b"{not json").unwrap_err();
        assert_eq!(e.id, Value::Null);
        assert_eq!(e.outcome.unwrap_err().code, PARSE_ERROR);
    }

    #[test]
    fn non_requests_are_invalid_requests() {
        for body in [
            &b"[]"[..],
            b"42",
            br#"{"id":1,"method":"x"}"#,
            br#"{"jsonrpc":"1.0","id":1,"method":"x"}"#,
            br#"{"jsonrpc":"2.0","id":1}"#,
            br#"{"jsonrpc":"2.0","id":1,"method":5}"#,
            br#"{"jsonrpc":"2.0","id":{"a":1},"method":"x"}"#,
        ] {
            let e = parse_message(body).unwrap_err();
            assert_eq!(
                e.outcome.unwrap_err().code,
                INVALID_REQUEST,
                "{}",
                String::from_utf8_lossy(body)
            );
        }
    }

    #[test]
    fn a_client_response_is_recognised() {
        assert_eq!(
            parse_message(br#"{"jsonrpc":"2.0","id":1,"result":{}}"#),
            Ok(Parsed::ClientResponse)
        );
    }

    #[test]
    fn responses_serialise_to_one_line() {
        let ok = Response::ok(json!(1), json!({"a": "x\ny"})).to_json();
        assert!(!ok.contains('\n'));
        assert_eq!(
            serde_json::from_str::<Value>(&ok).unwrap(),
            json!({"jsonrpc":"2.0","id":1,"result":{"a":"x\ny"}})
        );
        let err = Response::err(json!("a"), METHOD_NOT_FOUND, "nope").to_value();
        assert_eq!(err["error"]["code"], -32601);
        assert!(err.get("result").is_none());
    }
}
