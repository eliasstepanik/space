//! MCP **Streamable HTTP** transport, hand-rolled on tokio (no HTTP crate).
//!
//! One endpoint, [`MCP_PATH`]:
//!
//! * `POST /mcp` — one JSON-RPC message per request. A request is answered with
//!   `200` and either `Content-Type: application/json` (one JSON-RPC response)
//!   or `Content-Type: text/event-stream` carrying exactly one
//!   `event: message\ndata: <json>\n\n` and then EOF. A notification or a client
//!   response is answered `202 Accepted` with no body.
//! * `GET /mcp` — **`405 Method Not Allowed`**. The server never initiates
//!   messages, so there is no stream to open (the spec permits this).
//! * `DELETE /mcp` — ends the session named by `Mcp-Session-Id` (`200`).
//!
//! # Content negotiation
//!
//! The first media type in `Accept` that we can produce wins: listing
//! `application/json` first (or `*/*`, or no `Accept` at all) selects JSON;
//! listing `text/event-stream` first selects SSE. `q`-values are ignored. An
//! `Accept` naming neither is `406`.
//!
//! # Sessions
//!
//! A successful `initialize` mints an `Mcp-Session-Id` (returned as a response
//! header). Every other request must carry a live one, otherwise `400` with a
//! JSON-RPC error body.
//!
//! # Hardening
//!
//! Plain TCP, bound where the caller says (the binaries default to loopback).
//! A browser `Origin` that is not loopback is `403` (DNS-rebinding guard, per the
//! spec). Heads are capped at [`MAX_HEAD`] bytes, bodies at [`MAX_BODY`], reads
//! time out, concurrent connections and sessions are capped. Connections are
//! one-shot: every response carries `Connection: close`; `Transfer-Encoding`
//! (chunked bodies) is not supported (`501`).

use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::Value;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

use crate::jsonrpc::{Parsed, Response, INTERNAL_ERROR, INVALID_REQUEST};
use crate::server::{is_protocol_level_rejection, McpServer};

/// The single MCP endpoint.
pub const MCP_PATH: &str = "/mcp";
/// The session header (lower-case; HTTP header names are case-insensitive).
pub const SESSION_HEADER: &str = "mcp-session-id";
/// Largest accepted request body: 1 MiB.
pub const MAX_BODY: usize = 1 << 20;
/// Largest accepted request line + headers.
pub const MAX_HEAD: usize = 16 * 1024;
/// How long a client has to deliver a complete request.
pub const READ_TIMEOUT: Duration = Duration::from_secs(10);
/// Concurrent connections served at once.
pub const MAX_CONNECTIONS: usize = 64;
/// Live sessions kept at once.
pub const MAX_SESSIONS: usize = 256;

// ------------------------------------------------------------------ parsing

/// A failure to read a request, carrying the HTTP status to answer with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpError {
    /// Status code.
    pub status: u16,
    /// What was wrong.
    pub message: String,
}

impl HttpError {
    fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// The response for this error: its status plus a JSON-RPC error body.
    #[must_use]
    pub fn into_response(self) -> HttpResponse {
        HttpResponse::json(
            self.status,
            &Response::err(Value::Null, INVALID_REQUEST, self.message),
        )
    }
}

/// The request line and headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RequestHead {
    /// `GET`, `POST`, ...
    pub method: String,
    /// The request target as sent (path and optional query).
    pub target: String,
    /// Headers with lower-cased names, values trimmed, in order.
    pub headers: Vec<(String, String)>,
}

impl RequestHead {
    /// The first header called `name` (lower-case).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    /// The target without its query string.
    #[must_use]
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or("")
    }

    /// The declared `Content-Length`, if any.
    ///
    /// # Errors
    ///
    /// `400` when it is not a plain decimal number or repeats with a different
    /// value; `501` when `Transfer-Encoding` is present (chunked is unsupported).
    pub fn content_length(&self) -> Result<Option<usize>, HttpError> {
        if self.header("transfer-encoding").is_some() {
            return Err(HttpError::new(501, "Transfer-Encoding is not supported"));
        }
        let mut found: Option<usize> = None;
        for (_, value) in self.headers.iter().filter(|(n, _)| n == "content-length") {
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(HttpError::new(400, "invalid Content-Length"));
            }
            let n: usize = value
                .parse()
                .map_err(|_| HttpError::new(413, "Content-Length too large"))?;
            if found.is_some_and(|f| f != n) {
                return Err(HttpError::new(400, "conflicting Content-Length headers"));
            }
            found = Some(n);
        }
        Ok(found)
    }
}

/// A complete request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// Request line and headers.
    pub head: RequestHead,
    /// The body (exactly `Content-Length` bytes).
    pub body: Vec<u8>,
}

/// Parse the bytes **before** the blank line: the request line and headers.
///
/// # Errors
///
/// `400` for anything malformed, `505` for an HTTP version other than 1.0/1.1.
pub fn parse_head(head: &[u8]) -> Result<RequestHead, HttpError> {
    let text =
        std::str::from_utf8(head).map_err(|_| HttpError::new(400, "request is not UTF-8"))?;
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let (Some(method), Some(target), Some(version), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(HttpError::new(400, "malformed request line"));
    };
    if method.is_empty() || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(HttpError::new(400, "malformed method"));
    }
    if target.is_empty() || target.bytes().any(|b| b <= b' ' || b == 0x7f) {
        return Err(HttpError::new(400, "malformed request target"));
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(HttpError::new(505, "HTTP version not supported"));
    }
    let mut headers = Vec::new();
    for line in lines {
        let Some((name, value)) = line.split_once(':') else {
            return Err(HttpError::new(400, "malformed header line"));
        };
        if name.is_empty() || name.bytes().any(|b| b <= b' ' || b == 0x7f) {
            return Err(HttpError::new(400, "malformed header name"));
        }
        headers.push((name.to_ascii_lowercase(), value.trim().to_owned()));
    }
    Ok(RequestHead {
        method: method.to_owned(),
        target: target.to_owned(),
        headers,
    })
}

fn find_head_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

/// Read one request from `stream`: head, then exactly `Content-Length` body
/// bytes. `Ok(None)` means the peer closed before sending anything.
///
/// The body size is checked against `max_body` **before** it is read, and an
/// `Expect: 100-continue` is answered so clients like curl send the body.
///
/// # Errors
///
/// An [`HttpError`] naming the status to reply with.
pub async fn read_request<S>(stream: &mut S, max_body: usize) -> Result<Option<Request>, HttpError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(end) = find_head_end(&buf) {
            break end;
        }
        if buf.len() > MAX_HEAD {
            return Err(HttpError::new(431, "request head too large"));
        }
        let n = stream
            .read(&mut chunk)
            .await
            .map_err(|e| HttpError::new(400, format!("read failed: {e}")))?;
        if n == 0 {
            return if buf.is_empty() {
                Ok(None)
            } else {
                Err(HttpError::new(400, "connection closed mid-request"))
            };
        }
        buf.extend_from_slice(&chunk[..n]);
    };
    if head_end > MAX_HEAD {
        return Err(HttpError::new(431, "request head too large"));
    }
    let head = parse_head(&buf[..head_end])?;
    let length = match (head.content_length()?, head.method.as_str()) {
        (Some(n), _) => n,
        (None, "POST") => return Err(HttpError::new(411, "Content-Length required")),
        (None, _) => 0,
    };
    if length > max_body {
        return Err(HttpError::new(
            413,
            format!("body of {length} bytes exceeds the {max_body} byte limit"),
        ));
    }
    let mut body = buf.split_off(head_end + 4);
    if length > body.len()
        && head
            .header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        stream
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .await
            .map_err(|e| HttpError::new(400, format!("write failed: {e}")))?;
    }
    body.truncate(length);
    let have = body.len();
    body.resize(length, 0);
    stream
        .read_exact(&mut body[have..])
        .await
        .map_err(|_| HttpError::new(400, "connection closed before the full body arrived"))?;
    Ok(Some(Request { head, body }))
}

// ----------------------------------------------------------------- response

/// An HTTP response to serialise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// Status code.
    pub status: u16,
    /// Extra headers (`Content-Length` and `Connection` are added on write).
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
    /// A streamed (SSE) body is delimited by closing the connection, so it
    /// carries no `Content-Length`.
    pub streamed: bool,
}

/// The SSE framing of one JSON-RPC message: `event: message\ndata: <json>\n\n`.
///
/// `json` must be a single line (compact JSON), which is what
/// [`Response::to_json`] produces.
#[must_use]
pub fn sse_event(json: &str) -> String {
    format!("event: message\ndata: {json}\n\n")
}

const fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Content Too Large",
        415 => "Unsupported Media Type",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Internal Server Error",
    }
}

impl HttpResponse {
    /// A bodiless response.
    #[must_use]
    pub fn empty(status: u16) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: Vec::new(),
            streamed: false,
        }
    }

    /// A `application/json` response carrying `response`.
    #[must_use]
    pub fn json(status: u16, response: &Response) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".into(), "application/json".into())],
            body: response.to_json().into_bytes(),
            streamed: false,
        }
    }

    /// A `text/event-stream` response carrying `response` as one event.
    #[must_use]
    pub fn sse(response: &Response) -> Self {
        Self {
            status: 200,
            headers: vec![
                ("Content-Type".into(), "text/event-stream".into()),
                ("Cache-Control".into(), "no-cache".into()),
            ],
            body: sse_event(&response.to_json()).into_bytes(),
            streamed: true,
        }
    }

    /// Add a header.
    #[must_use]
    pub fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
        self.headers.push((name.to_owned(), value.into()));
        self
    }

    /// The first header called `name` (case-insensitive).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The exact bytes written to the socket.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = format!("HTTP/1.1 {} {}\r\n", self.status, reason(self.status)).into_bytes();
        for (name, value) in &self.headers {
            out.extend_from_slice(format!("{name}: {value}\r\n").as_bytes());
        }
        if !self.streamed {
            out.extend_from_slice(format!("Content-Length: {}\r\n", self.body.len()).as_bytes());
        }
        out.extend_from_slice(b"Connection: close\r\n\r\n");
        out.extend_from_slice(&self.body);
        out
    }
}

// --------------------------------------------------------------- negotiation

/// How to send a JSON-RPC response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Format {
    Json,
    Sse,
}

/// Pick the response format from an `Accept` header (see the module docs).
fn negotiate(accept: Option<&str>) -> Option<Format> {
    let Some(accept) = accept else {
        return Some(Format::Json);
    };
    for item in accept.split(',') {
        let media = item.split(';').next().unwrap_or("").trim();
        match media.to_ascii_lowercase().as_str() {
            "application/json" | "application/*" | "*/*" => return Some(Format::Json),
            "text/event-stream" | "text/*" => return Some(Format::Sse),
            _ => {}
        }
    }
    None
}

/// Whether an `Origin` header names a loopback page (or is harmless).
fn origin_allowed(origin: &str) -> bool {
    let rest = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"));
    let Some(rest) = rest else {
        return false;
    };
    let authority = rest.split('/').next().unwrap_or("");
    let host = if authority.starts_with('[') {
        authority
            .split(']')
            .next()
            .map_or("", |h| &authority[..h.len() + 1])
    } else {
        authority.split(':').next().unwrap_or("")
    };
    matches!(host, "localhost" | "127.0.0.1" | "[::1]")
}

// ---------------------------------------------------------------- transport

/// The HTTP front end for one [`McpServer`].
#[derive(Debug)]
pub struct HttpTransport {
    server: Arc<McpServer>,
    sessions: Mutex<HashSet<String>>,
    counter: AtomicU64,
}

impl HttpTransport {
    /// Wrap `server`.
    #[must_use]
    pub fn new(server: Arc<McpServer>) -> Arc<Self> {
        Arc::new(Self {
            server,
            sessions: Mutex::new(HashSet::new()),
            counter: AtomicU64::new(0),
        })
    }

    /// How many sessions are live.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions().len()
    }

    fn sessions(&self) -> std::sync::MutexGuard<'_, HashSet<String>> {
        self.sessions.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A fresh, unguessable-enough session id: 128 bits from two independently
    /// seeded `RandomState`s (the OS seeds them) mixed with a counter.
    fn new_session_id(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        let part = || {
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u64(n);
            h.finish()
        };
        format!("{:016x}{:016x}", part(), part())
    }

    fn session_ok(&self, request: &Request) -> bool {
        request
            .head
            .header(SESSION_HEADER)
            .is_some_and(|id| self.sessions().contains(id))
    }

    /// Answer one request. Pure of any socket, so tests call it directly.
    pub async fn handle(&self, request: Request) -> HttpResponse {
        let head = &request.head;
        if head.path() != MCP_PATH {
            return HttpError::new(404, "no such endpoint; use /mcp").into_response();
        }
        if head.header("origin").is_some_and(|o| !origin_allowed(o)) {
            return HttpError::new(403, "Origin not allowed").into_response();
        }
        match head.method.as_str() {
            "POST" => self.handle_post(&request).await,
            "DELETE" => {
                let id = head.header(SESSION_HEADER).unwrap_or("");
                if self.sessions().remove(id) {
                    HttpResponse::empty(200)
                } else {
                    HttpError::new(400, "missing or unknown Mcp-Session-Id").into_response()
                }
            }
            // No server-initiated stream is offered.
            "GET" => HttpError::new(405, "this server does not open a GET event stream")
                .into_response()
                .with_header("Allow", "POST, DELETE"),
            _ => HttpError::new(405, "method not allowed")
                .into_response()
                .with_header("Allow", "POST, DELETE"),
        }
    }

    async fn handle_post(&self, request: &Request) -> HttpResponse {
        let head = &request.head;
        if head
            .header("content-type")
            .is_some_and(|ct| !ct.to_ascii_lowercase().starts_with("application/json"))
        {
            return HttpError::new(415, "Content-Type must be application/json").into_response();
        }
        let Some(format) = negotiate(head.header("accept")) else {
            return HttpError::new(
                406,
                "Accept must allow application/json or text/event-stream",
            )
            .into_response();
        };
        let message = match crate::jsonrpc::parse_message(&request.body) {
            Err(error) => return HttpResponse::json(400, &error),
            Ok(Parsed::ClientResponse) => {
                return if self.session_ok(request) {
                    HttpResponse::empty(202)
                } else {
                    missing_session(Value::Null)
                };
            }
            Ok(Parsed::Message(message)) => message,
        };
        let is_initialize = message.method == "initialize";
        if !is_initialize && !self.session_ok(request) {
            return missing_session(message.id.unwrap_or(Value::Null));
        }
        let Some(response) = self.server.dispatch(message).await else {
            return HttpResponse::empty(202);
        };
        let mut http = match format {
            Format::Json => HttpResponse::json(200, &response),
            Format::Sse => HttpResponse::sse(&response),
        };
        if is_initialize && response.outcome.is_ok() {
            let mut sessions = self.sessions();
            if sessions.len() >= MAX_SESSIONS {
                return HttpResponse::json(
                    503,
                    &Response::err(response.id, INTERNAL_ERROR, "too many sessions"),
                );
            }
            let id = self.new_session_id();
            sessions.insert(id.clone());
            http = http.with_header("Mcp-Session-Id", id);
        } else if is_protocol_level_rejection(&response) {
            http.status = 400;
        }
        http
    }

    /// Accept connections on `listener` until it errors. Each connection serves
    /// one request.
    ///
    /// # Errors
    ///
    /// The `accept` error that ended the loop.
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> io::Result<()> {
        let permits = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        loop {
            let (stream, _) = listener.accept().await?;
            let Ok(permit) = Arc::clone(&permits).acquire_owned().await else {
                return Ok(());
            };
            let this = Arc::clone(&self);
            tokio::spawn(async move {
                this.serve_connection(stream).await;
                drop(permit);
            });
        }
    }

    async fn serve_connection(&self, mut stream: TcpStream) {
        let read = tokio::time::timeout(READ_TIMEOUT, read_request(&mut stream, MAX_BODY)).await;
        let response = match read {
            Ok(Ok(Some(request))) => self.handle(request).await,
            Ok(Ok(None)) => return,
            Ok(Err(e)) => e.into_response(),
            Err(_) => HttpError::new(408, "request timed out").into_response(),
        };
        let _ = stream.write_all(&response.to_bytes()).await;
        let _ = stream.shutdown().await;
    }
}

fn missing_session(id: Value) -> HttpResponse {
    HttpResponse::json(
        400,
        &Response::err(id, INVALID_REQUEST, "missing or unknown Mcp-Session-Id"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{Tool, ToolRegistry};
    use serde_json::json;

    fn head(raw: &str) -> Result<RequestHead, HttpError> {
        parse_head(raw.as_bytes())
    }

    #[test]
    fn request_line_and_headers_are_parsed() {
        let h = head("POST /mcp?x=1 HTTP/1.1\r\nHost: localhost\r\nContent-Length:  12 \r\nACCEPT: application/json, text/event-stream").unwrap();
        assert_eq!((h.method.as_str(), h.target.as_str()), ("POST", "/mcp?x=1"));
        assert_eq!(h.path(), "/mcp");
        assert_eq!(h.header("host"), Some("localhost"));
        assert_eq!(
            h.header("accept"),
            Some("application/json, text/event-stream")
        );
        assert_eq!(h.content_length(), Ok(Some(12)));
    }

    #[test]
    fn malformed_heads_are_rejected() {
        for bad in [
            "POST /mcp",
            "POST /mcp HTTP/1.1 extra",
            "post /mcp HTTP/1.1",
            "POST  HTTP/1.1",
            "POST /mcp HTTP/1.1\r\nno-colon-here",
            "POST /mcp HTTP/1.1\r\n: empty-name",
            "POST /mcp HTTP/1.1\r\nBad Name: x",
        ] {
            assert_eq!(head(bad).unwrap_err().status, 400, "{bad}");
        }
        assert_eq!(head("GET /mcp HTTP/2").unwrap_err().status, 505);
        assert_eq!(parse_head(&[0xff, 0xfe]).unwrap_err().status, 400);
    }

    #[test]
    fn content_length_is_validated() {
        let cl = |v: &str| {
            head(&format!("POST /mcp HTTP/1.1\r\nContent-Length: {v}"))
                .unwrap()
                .content_length()
        };
        assert_eq!(cl("0"), Ok(Some(0)));
        assert_eq!(cl("-1").unwrap_err().status, 400);
        assert_eq!(cl("1x").unwrap_err().status, 400);
        assert_eq!(cl("").unwrap_err().status, 400);
        assert_eq!(cl("99999999999999999999999").unwrap_err().status, 413);
        let dup = head("POST /mcp HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2").unwrap();
        assert_eq!(dup.content_length().unwrap_err().status, 400);
        let te = head("POST /mcp HTTP/1.1\r\nTransfer-Encoding: chunked").unwrap();
        assert_eq!(te.content_length().unwrap_err().status, 501);
    }

    async fn read(raw: &[u8], max: usize) -> Result<Option<Request>, HttpError> {
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        client.write_all(raw).await.unwrap();
        drop(client);
        read_request(&mut server, max).await
    }

    #[tokio::test]
    async fn a_content_length_body_is_read_exactly() {
        let req = read(
            b"POST /mcp HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloEXTRA",
            1024,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(req.body, b"hello");
        assert_eq!(req.head.method, "POST");
    }

    #[tokio::test]
    async fn a_body_split_across_reads_is_reassembled() {
        let (mut client, mut server) = tokio::io::duplex(8);
        let writer = tokio::spawn(async move {
            client
                .write_all(b"POST /mcp HTTP/1.1\r\nContent-Length: 20\r\n\r\n01234567890123456789")
                .await
                .unwrap();
        });
        let req = read_request(&mut server, 1024).await.unwrap().unwrap();
        writer.await.unwrap();
        assert_eq!(req.body, b"01234567890123456789");
    }

    #[tokio::test]
    async fn oversized_bodies_are_rejected_before_being_read() {
        // Only the head is sent: a 413 means the body was never awaited.
        let e = read(
            b"POST /mcp HTTP/1.1\r\nContent-Length: 1048577\r\n\r\n",
            MAX_BODY,
        )
        .await
        .unwrap_err();
        assert_eq!(e.status, 413);
        let ok = read(b"POST /mcp HTTP/1.1\r\nContent-Length: 0\r\n\r\n", MAX_BODY).await;
        assert!(ok.unwrap().unwrap().body.is_empty());
    }

    #[tokio::test]
    async fn other_malformed_requests_get_proper_statuses() {
        assert_eq!(read(b"", 10).await.unwrap(), None);
        assert_eq!(read(b"POST /mcp HTT", 10).await.unwrap_err().status, 400);
        assert_eq!(
            read(b"POST /mcp HTTP/1.1\r\n\r\n", 10)
                .await
                .unwrap_err()
                .status,
            411
        );
        assert_eq!(
            read(b"POST /mcp HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort", 10)
                .await
                .unwrap_err()
                .status,
            400
        );
        let huge = format!(
            "GET /mcp HTTP/1.1\r\nX: {}\r\n\r\n",
            "a".repeat(MAX_HEAD + 100)
        );
        assert_eq!(read(huge.as_bytes(), 10).await.unwrap_err().status, 431);
        let no_terminator = format!("GET /mcp HTTP/1.1\r\nX: {}", "a".repeat(MAX_HEAD + 100));
        assert_eq!(
            read(no_terminator.as_bytes(), 10).await.unwrap_err().status,
            431
        );
    }

    #[tokio::test]
    async fn expect_continue_is_answered() {
        let (mut client, mut server) = tokio::io::duplex(1024);
        let reader = tokio::spawn(async move { read_request(&mut server, 1024).await });
        client
            .write_all(b"POST /mcp HTTP/1.1\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\n")
            .await
            .unwrap();
        let mut got = [0u8; 25];
        client.read_exact(&mut got).await.unwrap();
        assert_eq!(&got, b"HTTP/1.1 100 Continue\r\n\r\n");
        client.write_all(b"{}").await.unwrap();
        assert_eq!(reader.await.unwrap().unwrap().unwrap().body, b"{}");
    }

    #[test]
    fn sse_framing_is_exact() {
        assert_eq!(
            sse_event(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#),
            "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"
        );
        let r = Response::ok(json!(1), json!({}));
        let http = HttpResponse::sse(&r);
        let bytes = String::from_utf8(http.to_bytes()).unwrap();
        assert_eq!(
            bytes,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
             Connection: close\r\n\r\n\
             event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"
        );
    }

    #[test]
    fn json_responses_carry_content_length() {
        let r = Response::ok(json!(1), json!({}));
        let bytes = String::from_utf8(HttpResponse::json(200, &r).to_bytes()).unwrap();
        let body = r#"{"jsonrpc":"2.0","id":1,"result":{}}"#;
        assert!(bytes.starts_with("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"));
        assert!(
            bytes.contains(&format!("Content-Length: {}\r\n", body.len())),
            "{bytes}"
        );
        assert!(bytes.ends_with(&format!("\r\n\r\n{body}")), "{bytes}");
    }

    #[test]
    fn accept_negotiation_and_origin_rules() {
        assert_eq!(negotiate(None), Some(Format::Json));
        assert_eq!(negotiate(Some("*/*")), Some(Format::Json));
        assert_eq!(
            negotiate(Some("application/json, text/event-stream")),
            Some(Format::Json)
        );
        assert_eq!(negotiate(Some("text/event-stream")), Some(Format::Sse));
        assert_eq!(
            negotiate(Some("text/event-stream;q=1, application/json")),
            Some(Format::Sse)
        );
        assert_eq!(negotiate(Some("text/html")), None);
        assert!(origin_allowed("http://localhost:3000"));
        assert!(origin_allowed("http://127.0.0.1"));
        assert!(origin_allowed("http://[::1]:8080"));
        assert!(!origin_allowed("http://evil.example"));
        assert!(!origin_allowed("http://localhost.evil.example"));
        assert!(!origin_allowed("null"));
    }

    // ---- session enforcement, driven through `handle` (no socket)

    fn transport() -> Arc<HttpTransport> {
        let registry = ToolRegistry::new().with(Tool::new(
            "hello",
            "Say hello.",
            json!({"type": "object"}),
            |_| async { Ok("hello".to_owned()) },
        ));
        HttpTransport::new(Arc::new(McpServer::new("t", "0", registry)))
    }

    fn post(body: &str, headers: &[(&str, &str)]) -> Request {
        let mut h = vec![
            ("content-type".to_owned(), "application/json".to_owned()),
            (
                "accept".to_owned(),
                "application/json, text/event-stream".to_owned(),
            ),
        ];
        h.extend(
            headers
                .iter()
                .map(|(n, v)| ((*n).to_owned(), (*v).to_owned())),
        );
        Request {
            head: RequestHead {
                method: "POST".into(),
                target: "/mcp".into(),
                headers: h,
            },
            body: body.as_bytes().to_vec(),
        }
    }

    const INIT: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}"#;
    const LIST: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#;

    #[tokio::test]
    async fn initialize_assigns_a_session_that_later_requests_need() {
        let t = transport();
        let init = t.handle(post(INIT, &[])).await;
        assert_eq!(init.status, 200);
        assert_eq!(init.header("content-type"), Some("application/json"));
        let session = init
            .header("mcp-session-id")
            .expect("session id")
            .to_owned();
        assert_eq!(session.len(), 32);
        assert_eq!(t.session_count(), 1);

        // No header, wrong header: 400 with a JSON-RPC error carrying the id.
        for headers in [vec![], vec![("mcp-session-id", "bogus")]] {
            let r = t.handle(post(LIST, &headers)).await;
            assert_eq!(r.status, 400);
            let v: Value = serde_json::from_slice(&r.body).unwrap();
            assert_eq!(v["error"]["code"], INVALID_REQUEST);
            assert_eq!(v["id"], 2);
        }
        let ok = t.handle(post(LIST, &[("mcp-session-id", &session)])).await;
        assert_eq!(ok.status, 200);
        let v: Value = serde_json::from_slice(&ok.body).unwrap();
        assert_eq!(v["result"]["tools"][0]["name"], "hello");

        // Sessions are distinct.
        let second = t.handle(post(INIT, &[])).await;
        assert_ne!(second.header("mcp-session-id"), Some(session.as_str()));
    }

    #[tokio::test]
    async fn notifications_are_202_and_need_a_session_too() {
        let t = transport();
        let note = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        assert_eq!(t.handle(post(note, &[])).await.status, 400);
        let session = t
            .handle(post(INIT, &[]))
            .await
            .header("mcp-session-id")
            .unwrap()
            .to_owned();
        let r = t.handle(post(note, &[("mcp-session-id", &session)])).await;
        assert_eq!((r.status, r.body.len()), (202, 0));
    }

    #[tokio::test]
    async fn sse_is_chosen_when_asked_for_and_the_session_header_still_rides() {
        let t = transport();
        let mut req = post(INIT, &[]);
        req.head.headers.retain(|(n, _)| n != "accept");
        req.head
            .headers
            .push(("accept".into(), "text/event-stream".into()));
        let r = t.handle(req).await;
        assert_eq!(r.header("content-type"), Some("text/event-stream"));
        assert!(r.header("mcp-session-id").is_some());
        let body = String::from_utf8(r.body).unwrap();
        assert!(body.starts_with("event: message\ndata: {"), "{body}");
        assert!(body.ends_with("}\n\n"), "{body}");
    }

    #[tokio::test]
    async fn http_level_rejections() {
        let t = transport();
        // bad JSON → 400 + -32700
        let r = t.handle(post("{nope", &[])).await;
        assert_eq!(r.status, 400);
        assert_eq!(
            serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["code"],
            -32700
        );
        // unknown method after init is a JSON-RPC error inside a 200
        let session = t
            .handle(post(INIT, &[]))
            .await
            .header("mcp-session-id")
            .unwrap()
            .to_owned();
        let r = t
            .handle(post(
                r#"{"jsonrpc":"2.0","id":3,"method":"x/y"}"#,
                &[("mcp-session-id", &session)],
            ))
            .await;
        assert_eq!(r.status, 200);
        assert_eq!(
            serde_json::from_slice::<Value>(&r.body).unwrap()["error"]["code"],
            -32601
        );
        // wrong path / method / content type / accept / origin
        let mut wrong_path = post(INIT, &[]);
        wrong_path.head.target = "/other".into();
        assert_eq!(t.handle(wrong_path).await.status, 404);
        let mut get = post("", &[]);
        get.head.method = "GET".into();
        let r = t.handle(get).await;
        assert_eq!((r.status, r.header("allow")), (405, Some("POST, DELETE")));
        let mut put = post("", &[]);
        put.head.method = "PUT".into();
        assert_eq!(t.handle(put).await.status, 405);
        assert_eq!(
            t.handle(post(INIT, &[("content-type", "text/plain")]))
                .await
                .status,
            415
        );
        let mut html = post(INIT, &[]);
        html.head.headers.retain(|(n, _)| n != "accept");
        html.head
            .headers
            .push(("accept".into(), "text/html".into()));
        assert_eq!(t.handle(html).await.status, 406);
        assert_eq!(
            t.handle(post(INIT, &[("origin", "http://evil.example")]))
                .await
                .status,
            403
        );
        assert_eq!(
            t.handle(post(INIT, &[("origin", "http://localhost:5173")]))
                .await
                .status,
            200
        );
    }

    #[tokio::test]
    async fn delete_ends_a_session() {
        let t = transport();
        let session = t
            .handle(post(INIT, &[]))
            .await
            .header("mcp-session-id")
            .unwrap()
            .to_owned();
        let mut del = post("", &[("mcp-session-id", &session)]);
        del.head.method = "DELETE".into();
        assert_eq!(t.handle(del.clone()).await.status, 200);
        assert_eq!(t.handle(del).await.status, 400);
        assert_eq!(
            t.handle(post(LIST, &[("mcp-session-id", &session)]))
                .await
                .status,
            400
        );
    }

    // ---- the real socket path

    async fn raw(addr: std::net::SocketAddr, request: &[u8]) -> String {
        let mut s = TcpStream::connect(addr).await.unwrap();
        s.write_all(request).await.unwrap();
        let mut out = Vec::new();
        s.read_to_end(&mut out).await.unwrap();
        String::from_utf8(out).unwrap()
    }

    #[tokio::test]
    async fn the_socket_path_serves_and_rejects() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let task = tokio::spawn(transport().serve(listener));

        let body = INIT;
        let req = format!(
            "POST /mcp HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nAccept: application/json, text/event-stream\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let resp = raw(addr, req.as_bytes()).await;
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"), "{resp}");
        assert!(resp.contains("Mcp-Session-Id: "), "{resp}");
        assert!(
            resp.contains("\"protocolVersion\":\"2025-06-18\""),
            "{resp}"
        );

        let resp = raw(
            addr,
            b"POST /mcp HTTP/1.1\r\nContent-Length: 2000000\r\n\r\n",
        )
        .await;
        assert!(resp.starts_with("HTTP/1.1 413 "), "{resp}");
        let resp = raw(addr, b"GARBAGE\r\n\r\n").await;
        assert!(resp.starts_with("HTTP/1.1 400 "), "{resp}");
        assert!(resp.contains("-32600"), "{resp}");
        let resp = raw(addr, b"GET /mcp HTTP/1.1\r\n\r\n").await;
        assert!(resp.starts_with("HTTP/1.1 405 "), "{resp}");
        task.abort();
    }
}
