//! The HTTP API: a client's [`Control`] over plain HTTP/1.1, for an agent with
//! nothing but `curl`.
//!
//! Hand-rolled on tokio like `space-mcp`'s transport (no HTTP crate), with the
//! same care: the head is capped at [`MAX_HEAD`] bytes and the body at
//! [`HttpConfig::max_body`], reads and writes time out, at most
//! [`MAX_CONNECTIONS`] connections are served at once (further ones get `503`),
//! and every connection serves one request (`Connection: close`). Chunked
//! bodies are not supported (`501`).
//!
//! # Routes
//!
//! `GET /` returns this list as plain text ([`HELP`]).
//!
//! | route | request | response |
//! |---|---|---|
//! | `GET /state` | | the [`ClientState`](crate::ClientState) as JSON |
//! | `GET /frame.png`, `GET /frame.jpg` | `?wait=1` the next frame; `?after=N` the first frame after number N; `?quality=Q` (JPEG) | the image; headers `X-Frame-Seq`, `X-Frame-Width`, `X-Frame-Height` |
//! | `POST /input` | one event object or an array of them (below) | `{"ok":true,"events":N}` |
//! | `POST /connect` | `{"address":"host:port"}`, or the address as plain text | `{"ok":true}` (the outcome shows in `/state`) |
//! | `POST /disconnect` | | `{"ok":true}` |
//! | `POST /quit` | | `{"ok":true}` |
//! | `POST /script` | Lua source; `?timeout=SECONDS`, `?name=CHUNK` | the [`ScriptReport`](crate::ScriptReport) as JSON; `409` while another script runs |
//! | `POST /script/stop` | | `{"ok":true,"stopped":BOOL}` |
//!
//! Input events: `{"key":"w","down":true}`, `{"press":"enter"}` (optionally
//! `"ms":N` to hold it), `{"text":"127.0.0.1:7777"}`, `{"turn":[dyaw,dpitch]}`
//! in degrees, `{"pointer":[x,y],"buttons":0}`. A request is checked whole
//! before any of it is applied.
//!
//! A script that fails is still `200`: the request worked, and the report says
//! `"ok":false` with the error. Errors of the API itself are JSON too:
//! `{"error":"..."}` with a 4xx/5xx status.
//!
//! # Security
//!
//! This is remote control plus script execution, so:
//!
//! * **Loopback needs no credentials; anything else needs a token.**
//!   [`serve`] refuses a listener that is not bound to a loopback address
//!   unless [`HttpConfig::token`] is set (at least [`MIN_TOKEN_LEN`] bytes).
//!   With a token, every request must carry `Authorization: Bearer <token>`
//!   (compared in constant time), on loopback too. There is no TLS: the token
//!   crosses the network in the clear, so tunnel it (SSH) or put a TLS proxy in
//!   front when the network is not trusted.
//! * **`Host` must name the machine by address.** An IP literal, `localhost`,
//!   or a name listed in [`HttpConfig::allowed_hosts`]; anything else is `403`.
//!   A web page that points a DNS name it controls at `127.0.0.1` (DNS
//!   rebinding) still sends that name as `Host`.
//! * **No cross-origin requests.** A request with an `Origin` header that is
//!   not this server's own origin is `403`, as is one a browser marks
//!   `Sec-Fetch-Site: cross-site` / `same-site`. No CORS header is ever sent.
//!   A web page must not be able to drive the client.
//! * Scripts run in the sandbox described in [`crate::script`].

use std::fmt;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;

use crate::control::{key_from_name, key_names, Control, ControlError, Key, ViewerInput};
use crate::image::{encode_frame, ImageFormat};
use crate::script::{run_script, ScriptConfig, StopHandle};

/// Largest accepted request line + headers, in bytes.
pub const MAX_HEAD: usize = 16 * 1024;
/// Connections served at once.
pub const MAX_CONNECTIONS: usize = 32;
/// How long a client has to deliver a complete request, by default
/// ([`HttpConfig::read_timeout`]).
pub const READ_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a client has to take the response off the socket.
pub const WRITE_TIMEOUT: Duration = Duration::from_secs(30);
/// Shortest token accepted for a listener that is not on loopback, in bytes.
pub const MIN_TOKEN_LEN: usize = 16;
/// Most input events in one `POST /input`.
pub const MAX_EVENTS: usize = 256;
/// Most characters in one `text` event.
pub const MAX_TEXT_CHARS: usize = 1024;
/// Longest total time the `press` events of one request may hold keys, ms.
pub const MAX_PRESS_MS: u64 = 10_000;
/// How long a movement key is held by a `press` without `ms`, ms.
pub const DEFAULT_PRESS_MS: u64 = 100;
/// How long the accept loop pauses after a failed `accept`.
const ACCEPT_RETRY_DELAY: Duration = Duration::from_millis(50);

/// What `GET /` answers: the routes, for whoever curls the port.
pub const HELP: &str = "\
space-harness: remote control of a running space client (plain HTTP, one request per connection).

GET  /state            the client's state as JSON: screen (connect|connecting|online|error),
                       error, address, shard, session, tick, position, scene_position,
                       velocity, speed,
                       yaw_deg, pitch_deg, handoffs, scene, scene_revision, retry_in_s,
                       frame, width, height, fps,
                       gravity, landed, bodies (nearest first: guid, visual, visual_id,
                       position, distance, speed, parts, in_view), bodies_in_view,
                       correction_m, correction_max_m
GET  /frame.png        the current picture, lossless PNG
GET  /frame.jpg        the same as JPEG (?quality=1..100)
                       ?wait=1   block until the next frame
                       ?after=N  block until a frame with a number above N
                       response headers: X-Frame-Seq, X-Frame-Width, X-Frame-Height
POST /input            JSON: one event or an array of events
                         {\"key\":\"w\",\"down\":true}      press or release a key
                         {\"press\":\"enter\"}             one key press (\"ms\":N holds it)
                         {\"text\":\"127.0.0.1:7777\"}     type characters
                         {\"turn\":[30,-10]}             turn the view, degrees (yaw left+, pitch up+)
                         {\"pointer\":[0.5,0.5],\"buttons\":0}
POST /connect          {\"address\":\"host:port\"} (or the address as plain text); then watch /state
POST /disconnect
POST /quit             ask the client to exit
POST /script           body: Lua source. Runs it (one at a time; 409 if one is running) and returns
                       {\"ok\":bool,\"error\":{kind,message,line}|null,\"output\":[lines],
                        \"screenshots\":[paths],\"last_call_line\":N|null,\"elapsed_ms\":N}
                       ?timeout=SECONDS  ?name=CHUNKNAME
POST /script/stop      stop the running script

keys: w a s d space shift (forward back left right up down), enter backspace escape tab
      delete home end caret_left caret_right

Lua (POST /script): client.type(text) client.press(key) client.key_down(key) client.key_up(key)
  client.hold(key, seconds) client.turn(dyaw, dpitch) client.pointer(x, y [, buttons])
  client.connect(address [, timeout]) client.disconnect() client.state()
  client.wait(seconds) client.wait_frames(n) client.wait_until(fn [, timeout [, interval]])
  client.screenshot(\"name.png\") client.pixel(x, y) client.log(...) print(...)
  client.time() client.quit()
  A failing assert/error ends the script: ok=false, with the message and line.

With a token configured every request needs:  Authorization: Bearer <token>
";

/// How the API is served.
#[derive(Clone)]
pub struct HttpConfig {
    /// The bearer token. `None` is only accepted on a loopback listener.
    pub token: Option<String>,
    /// Host names (besides IP literals and `localhost`) that requests may use
    /// in their `Host` header, e.g. the name a reverse proxy forwards.
    pub allowed_hosts: Vec<String>,
    /// How scripts are run: output directory, default timeout, limits. The
    /// `name` and `timeout` can be overridden per request.
    pub script: ScriptConfig,
    /// The longest `?timeout=` a `POST /script` may ask for.
    pub max_script_timeout: Duration,
    /// Largest accepted request body, in bytes.
    pub max_body: usize,
    /// How long a frame request waits for the client to render one.
    pub frame_timeout: Duration,
    /// Default JPEG quality of `/frame.jpg`.
    pub jpeg_quality: u8,
    /// How long a client has to deliver a complete request
    /// ([`READ_TIMEOUT`] by default).
    pub read_timeout: Duration,
}

impl Default for HttpConfig {
    /// No token (loopback only), 1 MiB bodies, scripts as
    /// [`ScriptConfig::default`] but at most ten minutes.
    fn default() -> Self {
        Self {
            token: None,
            allowed_hosts: Vec::new(),
            script: ScriptConfig::default(),
            max_script_timeout: Duration::from_secs(600),
            max_body: 1 << 20,
            frame_timeout: Duration::from_secs(5),
            jpeg_quality: 85,
            read_timeout: READ_TIMEOUT,
        }
    }
}

impl fmt::Debug for HttpConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpConfig")
            // Never the token itself.
            .field("token", &self.token.as_ref().map(|_| "(set)"))
            .field("allowed_hosts", &self.allowed_hosts)
            .field("script", &self.script)
            .field("max_script_timeout", &self.max_script_timeout)
            .field("max_body", &self.max_body)
            .field("frame_timeout", &self.frame_timeout)
            .field("jpeg_quality", &self.jpeg_quality)
            .field("read_timeout", &self.read_timeout)
            .finish()
    }
}

/// Why the API cannot be served as configured.
#[derive(Debug)]
pub enum ServeError {
    /// The listener is not on a loopback address and no token is configured.
    TokenRequired(SocketAddr),
    /// The token is empty, shorter than [`MIN_TOKEN_LEN`] on a non-loopback
    /// listener, or has characters that cannot go in a header.
    WeakToken(&'static str),
    /// The listener's address could not be read.
    Io(std::io::Error),
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TokenRequired(addr) => write!(
                f,
                "refusing to serve the control API on {addr} without a token: it is not a loopback \
                 address, and the API can drive the client and run scripts"
            ),
            Self::WeakToken(why) => write!(f, "unusable control API token: {why}"),
            Self::Io(e) => write!(f, "control API listener: {e}"),
        }
    }
}

impl std::error::Error for ServeError {}

// ------------------------------------------------------------------ parsing

/// A request that cannot be served, with the status to answer.
#[derive(Clone, Debug, PartialEq, Eq)]
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
}

impl From<HttpError> for Response {
    fn from(error: HttpError) -> Self {
        Response::error(error.status, &error.message)
    }
}

/// A parsed request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Request {
    /// `GET`, `POST`, ...
    pub method: String,
    /// The path, without the query.
    pub path: String,
    /// The decoded query parameters, in order.
    pub query: Vec<(String, String)>,
    /// Headers with lower-cased names and trimmed values, in order.
    pub headers: Vec<(String, String)>,
    /// The body (exactly `Content-Length` bytes).
    pub body: Vec<u8>,
}

impl Request {
    /// The first header called `name` (lower-case).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }

    /// The first query parameter called `name`.
    #[must_use]
    pub fn param(&self, name: &str) -> Option<&str> {
        self.query
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

/// `%XX` and `+` decoding of one query component. `None` for a malformed
/// escape or a result that is not UTF-8.
fn percent_decode(text: &str) -> Option<String> {
    let mut out = Vec::with_capacity(text.len());
    let mut bytes = text.bytes();
    while let Some(b) = bytes.next() {
        match b {
            b'%' => {
                let hex = [bytes.next()?, bytes.next()?];
                let hex = std::str::from_utf8(&hex).ok()?;
                out.push(u8::from_str_radix(hex, 16).ok()?);
            }
            b'+' => out.push(b' '),
            other => out.push(other),
        }
    }
    String::from_utf8(out).ok()
}

/// Split a request target into its path and decoded query parameters.
fn parse_target(target: &str) -> Result<(String, Vec<(String, String)>), HttpError> {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    if !path.starts_with('/') {
        return Err(HttpError::new(400, "the request target must be a path"));
    }
    let mut params = Vec::new();
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        match (percent_decode(name), percent_decode(value)) {
            (Some(name), Some(value)) => params.push((name, value)),
            _ => return Err(HttpError::new(400, "malformed query string")),
        }
    }
    Ok((path.to_owned(), params))
}

/// The request line and headers, before the body is read.
struct Head {
    method: String,
    path: String,
    query: Vec<(String, String)>,
    headers: Vec<(String, String)>,
}

/// Parse the bytes before the blank line.
fn parse_head(head: &[u8]) -> Result<Head, HttpError> {
    let text =
        std::str::from_utf8(head).map_err(|_| HttpError::new(400, "the request is not UTF-8"))?;
    let mut lines = text.split("\r\n");
    let mut parts = lines.next().unwrap_or("").split(' ');
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
    let (path, query) = parse_target(target)?;
    Ok(Head {
        method: method.to_owned(),
        path,
        query,
        headers,
    })
}

/// The declared body length: 0 without a `Content-Length`.
fn content_length(headers: &[(String, String)]) -> Result<usize, HttpError> {
    if headers.iter().any(|(n, _)| n == "transfer-encoding") {
        return Err(HttpError::new(501, "Transfer-Encoding is not supported"));
    }
    let mut found: Option<usize> = None;
    for (_, value) in headers.iter().filter(|(n, _)| n == "content-length") {
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
    Ok(found.unwrap_or(0))
}

/// Read one request: the head, then exactly `Content-Length` body bytes.
/// `Ok(None)` when the peer closed without sending anything.
///
/// The body size is checked against `max_body` before any of it is read, and
/// an `Expect: 100-continue` is answered so that curl sends the body.
async fn read_request<S>(stream: &mut S, max_body: usize) -> Result<Option<Request>, HttpError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
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
    let length = content_length(&head.headers)?;
    if length > max_body {
        return Err(HttpError::new(
            413,
            format!("a body of {length} bytes exceeds the {max_body} byte limit"),
        ));
    }
    let mut body = buf.split_off(head_end + 4);
    let expects_continue = head
        .headers
        .iter()
        .any(|(n, v)| n == "expect" && v.eq_ignore_ascii_case("100-continue"));
    if length > body.len() && expects_continue {
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
    Ok(Some(Request {
        method: head.method,
        path: head.path,
        query: head.query,
        headers: head.headers,
        body,
    }))
}

// ----------------------------------------------------------------- response

/// A response to serialise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Response {
    /// Status code.
    pub status: u16,
    /// Extra headers (`Content-Length`, `Connection` and the fixed hardening
    /// headers are added on write).
    pub headers: Vec<(String, String)>,
    /// Body bytes.
    pub body: Vec<u8>,
}

const fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Content Too Large",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        505 => "HTTP Version Not Supported",
        _ => "Internal Server Error",
    }
}

impl Response {
    fn with_type(status: u16, content_type: &str, body: Vec<u8>) -> Self {
        Self {
            status,
            headers: vec![("Content-Type".into(), content_type.into())],
            body,
        }
    }

    /// A JSON response.
    #[must_use]
    pub fn json(status: u16, value: &Value) -> Self {
        let mut body = value.to_string().into_bytes();
        body.push(b'\n');
        Self::with_type(status, "application/json", body)
    }

    /// `{"error": message}` with `status`.
    #[must_use]
    pub fn error(status: u16, message: &str) -> Self {
        Self::json(status, &json!({ "error": message }))
    }

    fn ok() -> Self {
        Self::json(200, &json!({ "ok": true }))
    }

    fn with_header(mut self, name: &str, value: impl Into<String>) -> Self {
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
        out.extend_from_slice(
            format!(
                "Content-Length: {}\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n",
                self.body.len()
            )
            .as_bytes(),
        );
        out.extend_from_slice(&self.body);
        out
    }
}

// ----------------------------------------------------------------- security

/// The host part of a `Host` / authority value: the name or address without
/// the port, an IPv6 literal without its brackets. `None` if it is not one.
fn authority_host(authority: &str) -> Option<&str> {
    if let Some(rest) = authority.strip_prefix('[') {
        let (host, after) = rest.split_once(']')?;
        let port_ok = after.is_empty()
            || after
                .strip_prefix(':')
                .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
        return (port_ok && !host.is_empty()).then_some(host);
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port)) => (host, Some(port)),
        None => (authority, None),
    };
    let port_ok = port.is_none_or(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    (port_ok && !host.is_empty() && !host.contains([':', '[', ']', '/', '@'])).then_some(host)
}

/// Whether a `Host` header names this machine in a way a DNS-rebinding page
/// cannot: an IP literal, `localhost`, or a configured name.
fn host_allowed(host_header: &str, allowed: &[String]) -> bool {
    let Some(host) = authority_host(host_header) else {
        return false;
    };
    // Brackets are for IPv6 literals and nothing else ("[127.0.0.1]" and
    // "[localhost]" are not hosts); an IPv6 literal without them is already
    // refused by `authority_host`.
    let bracketed = host_header.starts_with('[');
    match host.parse::<IpAddr>() {
        Ok(IpAddr::V6(_)) => bracketed,
        Ok(IpAddr::V4(_)) => !bracketed,
        Err(_) if bracketed => false,
        Err(_) => {
            let name = host.strip_suffix('.').unwrap_or(host);
            name.eq_ignore_ascii_case("localhost")
                || allowed.iter().any(|a| a.eq_ignore_ascii_case(name))
        }
    }
}

/// Whether an `Origin` header is this server's own origin (the scheme is
/// always `http`, the authority is what the request called us in `Host`).
fn same_origin(origin: &str, host_header: &str) -> bool {
    origin
        .strip_prefix("http://")
        .is_some_and(|authority| authority.eq_ignore_ascii_case(host_header))
}

/// Compare two byte strings without stopping at the first difference, so the
/// time taken does not tell how much of a guess was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let mut diff = u8::from(a.len() != b.len());
    // Walk the longer of the two, so the length of the secret is not the loop
    // bound either.
    for i in 0..a.len().max(b.len()) {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        diff |= x ^ y;
    }
    diff == 0
}

/// The token in an `Authorization: Bearer <token>` header.
fn bearer(header: &str) -> Option<&str> {
    let (scheme, token) = header.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| token.trim())
}

fn check_token(token: Option<&str>, loopback: bool) -> Result<(), ServeError> {
    let Some(token) = token else {
        return Ok(());
    };
    if token.is_empty() {
        return Err(ServeError::WeakToken("it is empty"));
    }
    if !token.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(ServeError::WeakToken(
            "it must be printable ASCII without spaces",
        ));
    }
    if !loopback && token.len() < MIN_TOKEN_LEN {
        return Err(ServeError::WeakToken(
            "it is too short for a listener that is reachable from the network",
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------ routing

/// One checked input event of a `POST /input`.
#[derive(Clone, Debug, PartialEq)]
enum Event {
    Input(ViewerInput),
    Press { key: Key, hold_ms: u64 },
    Text(String),
    Turn(f64, f64),
}

fn event_key(value: &Value) -> Result<Key, String> {
    let name = value.as_str().ok_or("a key must be a string")?;
    key_from_name(name).ok_or_else(|| format!("unknown key {name:?} (keys: {})", key_names()))
}

fn pair(value: &Value, what: &str) -> Result<(f64, f64), String> {
    match value.as_array().map(Vec::as_slice) {
        Some([a, b]) => match (a.as_f64(), b.as_f64()) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => Ok((a, b)),
            _ => Err(format!("{what} must be two finite numbers")),
        },
        _ => Err(format!("{what} must be an array of two numbers")),
    }
}

fn parse_event(value: &Value) -> Result<Event, String> {
    let object = value.as_object().ok_or("an event must be a JSON object")?;
    let only = |allowed: &[&str]| match object.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(extra) => Err(format!("unexpected field {extra:?}")),
        None => Ok(()),
    };
    if let Some(key) = object.get("key") {
        only(&["key", "down"])?;
        let down = object
            .get("down")
            .and_then(Value::as_bool)
            .ok_or("a key event needs \"down\": true or false")?;
        let key = event_key(key)?;
        Ok(Event::Input(ViewerInput::Key { key, down }))
    } else if let Some(key) = object.get("press") {
        only(&["press", "ms"])?;
        let key = event_key(key)?;
        let hold_ms = match object.get("ms") {
            None => {
                if key.is_movement() {
                    DEFAULT_PRESS_MS
                } else {
                    0
                }
            }
            Some(ms) => ms
                .as_u64()
                .filter(|ms| *ms <= MAX_PRESS_MS)
                .ok_or_else(|| format!("\"ms\" must be a whole number up to {MAX_PRESS_MS}"))?,
        };
        Ok(Event::Press { key, hold_ms })
    } else if let Some(text) = object.get("text") {
        only(&["text"])?;
        let text = text.as_str().ok_or("\"text\" must be a string")?;
        if text.chars().count() > MAX_TEXT_CHARS {
            return Err(format!(
                "\"text\" is longer than {MAX_TEXT_CHARS} characters"
            ));
        }
        if let Some(bad) = text
            .chars()
            .find(|c| c.is_control() && !matches!(c, '\n' | '\t'))
        {
            return Err(format!(
                "\"text\" contains the control character U+{:04X}",
                u32::from(bad)
            ));
        }
        Ok(Event::Text(text.to_owned()))
    } else if let Some(turn) = object.get("turn") {
        only(&["turn"])?;
        let (dyaw, dpitch) = pair(turn, "\"turn\"")?;
        Ok(Event::Turn(dyaw, dpitch))
    } else if let Some(pointer) = object.get("pointer") {
        only(&["pointer", "buttons"])?;
        let (x, y) = pair(pointer, "\"pointer\"")?;
        if !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
            return Err("\"pointer\" coordinates must be in 0..1".into());
        }
        let buttons = match object.get("buttons") {
            None => 0,
            Some(b) => b
                .as_u64()
                .and_then(|b| u8::try_from(b).ok())
                .ok_or("\"buttons\" must be a whole number in 0..=255")?,
        };
        let (x, y) = (x as f32, y as f32);
        Ok(Event::Input(ViewerInput::Pointer { x, y, buttons }))
    } else {
        Err("an event needs one of \"key\", \"press\", \"text\", \"turn\", \"pointer\"".into())
    }
}

/// Parse the body of a `POST /input`: one event or an array of events.
fn parse_events(body: &[u8]) -> Result<Vec<Event>, String> {
    let json: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
    let items = match &json {
        Value::Array(items) => items.as_slice(),
        other => std::slice::from_ref(other),
    };
    if items.len() > MAX_EVENTS {
        return Err(format!("more than {MAX_EVENTS} events in one request"));
    }
    let mut events = Vec::with_capacity(items.len());
    let mut held_ms = 0u64;
    for (i, item) in items.iter().enumerate() {
        let event = parse_event(item).map_err(|e| format!("event {}: {e}", i + 1))?;
        if let Event::Press { hold_ms, .. } = event {
            held_ms += hold_ms;
        }
        events.push(event);
    }
    if held_ms > MAX_PRESS_MS {
        return Err(format!(
            "the presses of one request hold keys for more than {MAX_PRESS_MS} ms"
        ));
    }
    Ok(events)
}

fn control_error(error: &ControlError) -> Response {
    let status = match error {
        ControlError::Timeout => 504,
        ControlError::Busy | ControlError::Closed => 503,
        ControlError::Invalid(_) => 400,
    };
    Response::error(status, &error.to_string())
}

/// The script slot: at most one script runs at a time.
#[derive(Debug, Default)]
struct ScriptSlot {
    running: AtomicBool,
    stop: Mutex<Option<StopHandle>>,
}

/// Frees the script slot when the request that took it ends, however it ends.
struct SlotGuard<'a>(&'a ScriptSlot);

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        *self
            .0
            .stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        self.0.running.store(false, Ordering::Release);
    }
}

/// The API for one client. [`serve`] wraps it around a listener; tests can call
/// [`Api::handle`] directly.
pub struct Api {
    control: Arc<dyn Control>,
    config: HttpConfig,
    script: ScriptSlot,
}

impl fmt::Debug for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Api")
            .field("control", &self.control)
            .field("config", &self.config)
            .field("script", &self.script)
            .finish()
    }
}

impl Api {
    /// The API for `control`, to be served on `bound`.
    ///
    /// # Errors
    ///
    /// [`ServeError::TokenRequired`] when `bound` is not a loopback address
    /// and no token is configured; [`ServeError::WeakToken`] for a token that
    /// is empty, has spaces or non-ASCII characters, or is shorter than
    /// [`MIN_TOKEN_LEN`] on a non-loopback address.
    pub fn new(
        control: Arc<dyn Control>,
        config: HttpConfig,
        bound: SocketAddr,
    ) -> Result<Arc<Self>, ServeError> {
        let loopback = bound.ip().is_loopback();
        if !loopback && config.token.is_none() {
            return Err(ServeError::TokenRequired(bound));
        }
        check_token(config.token.as_deref(), loopback)?;
        Ok(Arc::new(Self {
            control,
            config,
            script: ScriptSlot::default(),
        }))
    }

    /// The reason to refuse `request` before looking at what it asks for.
    fn refusal(&self, request: &Request) -> Option<Response> {
        let Some(host) = request.header("host") else {
            return Some(Response::error(400, "a Host header is required"));
        };
        if !host_allowed(host, &self.config.allowed_hosts) {
            return Some(Response::error(
                403,
                "this Host is not allowed: use the machine's IP address or localhost",
            ));
        }
        if request
            .header("origin")
            .is_some_and(|origin| !same_origin(origin, host))
        {
            return Some(Response::error(
                403,
                "cross-origin requests are not allowed",
            ));
        }
        if request.header("sec-fetch-site").is_some_and(|site| {
            !site.eq_ignore_ascii_case("same-origin") && !site.eq_ignore_ascii_case("none")
        }) {
            return Some(Response::error(403, "cross-site requests are not allowed"));
        }
        if let Some(token) = &self.config.token {
            let offered = request.header("authorization").and_then(bearer);
            // A missing header is compared too (against nothing), so "no
            // token" and "wrong token" take the same path.
            if !constant_time_eq(offered.unwrap_or("").as_bytes(), token.as_bytes())
                || offered.is_none()
            {
                return Some(
                    Response::error(401, "missing or wrong bearer token")
                        .with_header("WWW-Authenticate", "Bearer"),
                );
            }
        }
        None
    }

    /// Answer one request.
    pub async fn handle(&self, request: Request) -> Response {
        if let Some(refusal) = self.refusal(&request) {
            return refusal;
        }
        let allow = match request.path.as_str() {
            "/" | "/state" | "/frame.png" | "/frame.jpg" => "GET",
            "/input" | "/connect" | "/disconnect" | "/quit" | "/script" | "/script/stop" => "POST",
            _ => return Response::error(404, "no such route; GET / lists the routes"),
        };
        if request.method != allow {
            return Response::error(405, &format!("use {allow} for {}", request.path))
                .with_header("Allow", allow);
        }
        match request.path.as_str() {
            "/" => Response::with_type(200, "text/plain; charset=utf-8", HELP.as_bytes().to_vec()),
            "/state" => match serde_json::to_value(self.control.state()) {
                Ok(state) => Response::json(200, &state),
                Err(e) => Response::error(500, &e.to_string()),
            },
            "/frame.png" => self.frame(&request, ImageFormat::Png).await,
            "/frame.jpg" => self.frame(&request, ImageFormat::Jpeg).await,
            "/input" => self.input(&request).await,
            "/connect" => self.connect(&request),
            "/disconnect" => match self.control.disconnect() {
                Ok(()) => Response::ok(),
                Err(e) => control_error(&e),
            },
            "/quit" => {
                self.control.quit();
                Response::ok()
            }
            "/script" => self.script(&request).await,
            _ => {
                let stop = self
                    .script
                    .stop
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone();
                if let Some(stop) = &stop {
                    stop.stop();
                }
                Response::json(200, &json!({ "ok": true, "stopped": stop.is_some() }))
            }
        }
    }

    async fn frame(&self, request: &Request, format: ImageFormat) -> Response {
        let after = match request.param("after") {
            Some(text) => match text.parse::<u64>() {
                Ok(seq) => Some(seq),
                Err(_) => return Response::error(400, "?after= must be a frame number"),
            },
            None => match request.param("wait") {
                None | Some("0" | "false") => None,
                Some("" | "1" | "true") => Some(self.control.state().frame),
                Some(_) => return Response::error(400, "?wait= must be 1 or 0"),
            },
        };
        let quality = match request.param("quality") {
            None => self.config.jpeg_quality,
            Some(text) => match text.parse::<u8>() {
                Ok(quality @ 1..=100) => quality,
                _ => return Response::error(400, "?quality= must be in 1..=100"),
            },
        };
        let (control, timeout) = (Arc::clone(&self.control), self.config.frame_timeout);
        // Waiting for the frame and encoding it both block: off the runtime.
        let work = tokio::task::spawn_blocking(move || {
            let frame = control
                .frame(after, timeout)
                .map_err(|e| control_error(&e))?;
            let bytes = encode_frame(&frame, format, quality)
                .map_err(|e| Response::error(500, &e.to_string()))?;
            Ok::<_, Response>((frame, bytes))
        });
        match work.await {
            Ok(Ok((frame, bytes))) => Response::with_type(200, format.content_type(), bytes)
                .with_header("X-Frame-Seq", frame.seq.to_string())
                .with_header("X-Frame-Width", frame.width.to_string())
                .with_header("X-Frame-Height", frame.height.to_string()),
            Ok(Err(response)) => response,
            Err(e) => Response::error(500, &format!("frame task failed: {e}")),
        }
    }

    async fn input(&self, request: &Request) -> Response {
        let events = match parse_events(&request.body) {
            Ok(events) => events,
            Err(message) => return Response::error(400, &message),
        };
        for event in &events {
            let applied = match event {
                Event::Input(input) => self.control.input(*input),
                Event::Text(text) => self.control.type_text(text),
                Event::Turn(dyaw, dpitch) => self.control.turn_degrees(*dyaw, *dpitch),
                Event::Press { key, hold_ms } => {
                    let down = self.control.key(*key, true);
                    if down.is_ok() && *hold_ms > 0 {
                        tokio::time::sleep(Duration::from_millis(*hold_ms)).await;
                    }
                    // The key is released even if the request is being cut
                    // short: a press must not leave a key down.
                    down.and(self.control.key(*key, false))
                }
            };
            if let Err(e) = applied {
                return control_error(&e);
            }
        }
        Response::json(200, &json!({ "ok": true, "events": events.len() }))
    }

    fn connect(&self, request: &Request) -> Response {
        let Ok(body) = std::str::from_utf8(&request.body) else {
            return Response::error(400, "the body is not UTF-8");
        };
        let body = body.trim();
        let address = if body.starts_with('{') {
            let parsed: Result<Value, _> = serde_json::from_str(body);
            match parsed
                .as_ref()
                .map(|json| json.get("address").and_then(Value::as_str))
            {
                Ok(Some(address)) => address.to_owned(),
                Ok(None) => return Response::error(400, "expected {\"address\":\"host:port\"}"),
                Err(e) => return Response::error(400, &format!("invalid JSON: {e}")),
            }
        } else {
            body.to_owned()
        };
        match self.control.connect(&address) {
            Ok(()) => Response::ok(),
            Err(e) => control_error(&e),
        }
    }

    async fn script(&self, request: &Request) -> Response {
        let Ok(source) = String::from_utf8(request.body.clone()) else {
            return Response::error(400, "the script is not UTF-8");
        };
        let mut config = self.config.script.clone();
        if let Some(name) = request.param("name") {
            config.name = name.to_owned();
        }
        if let Some(text) = request.param("timeout") {
            let seconds = text
                .parse::<f64>()
                .ok()
                .filter(|s| s.is_finite() && *s > 0.0);
            let Some(timeout) = seconds.and_then(|s| Duration::try_from_secs_f64(s).ok()) else {
                return Response::error(400, "?timeout= must be a positive number of seconds");
            };
            config.timeout = timeout;
        }
        config.timeout = config.timeout.min(self.config.max_script_timeout);

        if self
            .script
            .running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Response::error(
                409,
                "a script is already running (POST /script/stop stops it)",
            );
        }
        let _slot = SlotGuard(&self.script);
        let stop = StopHandle::new();
        *self
            .script
            .stop
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(stop.clone());

        let control = Arc::clone(&self.control);
        let run = tokio::task::spawn_blocking(move || run_script(control, &source, &config, &stop));
        match run.await {
            Ok(report) => match serde_json::to_value(&report) {
                Ok(json) => Response::json(200, &json),
                Err(e) => Response::error(500, &e.to_string()),
            },
            Err(e) => Response::error(500, &format!("script task failed: {e}")),
        }
    }
}

// ---------------------------------------------------------------- transport

/// Serve the control API for `control` on `listener` until the returned future
/// is dropped (it does not end on its own).
///
/// A failed `accept` is logged to stderr and the server keeps accepting.
///
/// # Errors
///
/// Before accepting anything: [`ServeError`] when the listener is not on a
/// loopback address and the configuration has no (adequate) token.
pub async fn serve(
    listener: TcpListener,
    control: Arc<dyn Control>,
    config: HttpConfig,
) -> Result<(), ServeError> {
    let bound = listener.local_addr().map_err(ServeError::Io)?;
    let api = Api::new(control, config, bound)?;
    let permits = Arc::new(Semaphore::new(MAX_CONNECTIONS));
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(err) => {
                eprintln!("space-harness: accept failed ({err}); still listening");
                tokio::time::sleep(ACCEPT_RETRY_DELAY).await;
                continue;
            }
        };
        let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
            // Full: say so (if the socket takes it without waiting) and hang
            // up, rather than queueing connections or tasks without bound.
            let busy = Response::error(503, "too many connections").to_bytes();
            let _ = stream.try_write(&busy);
            continue;
        };
        let api = Arc::clone(&api);
        tokio::spawn(async move {
            serve_connection(&api, stream).await;
            drop(permit);
        });
    }
}

async fn serve_connection(api: &Api, mut stream: TcpStream) {
    let read = tokio::time::timeout(
        api.config.read_timeout,
        read_request(&mut stream, api.config.max_body),
    );
    let (response, unread) = match read.await {
        Ok(Ok(Some(request))) => (api.handle(request).await, false),
        Ok(Ok(None)) => return,
        Ok(Err(error)) => (error.into(), true),
        Err(_) => (
            Response::error(408, "the request did not arrive in time"),
            true,
        ),
    };
    let sent = write_response(&mut stream, &response.to_bytes(), WRITE_TIMEOUT).await;
    if sent && unread {
        discard_unread(&mut stream).await;
    }
    // Dropping the stream closes it.
}

/// After refusing a request that was not read to its end: swallow what the
/// client is still sending, briefly. Closing a socket with unread input resets
/// the connection, and a reset can reach the client before it has read the
/// refusal (it would see "connection reset" instead of the `413`).
async fn discard_unread<S>(stream: &mut S)
where
    S: AsyncRead + Unpin,
{
    const LINGER: Duration = Duration::from_millis(500);
    const MOST: usize = 1 << 20;
    let drain = async {
        let mut sink = [0u8; 4096];
        let mut total = 0;
        while total < MOST {
            match stream.read(&mut sink).await {
                Ok(0) | Err(_) => break,
                Ok(n) => total += n,
            }
        }
    };
    let _ = tokio::time::timeout(LINGER, drain).await;
}

/// Write `bytes` and shut the stream down, giving up after `limit`. Returns
/// whether everything went out: a peer that does not read must not hold its
/// connection (and its permit) for ever.
async fn write_response<S>(stream: &mut S, bytes: &[u8], limit: Duration) -> bool
where
    S: AsyncWrite + Unpin,
{
    let write = async {
        stream.write_all(bytes).await?;
        stream.shutdown().await
    };
    matches!(tokio::time::timeout(limit, write).await, Ok(Ok(())))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn head(raw: &str) -> Result<Head, HttpError> {
        parse_head(raw.as_bytes())
    }

    #[test]
    fn request_lines_headers_and_queries_are_parsed() {
        let h = head("GET /frame.png?wait=1&after=12&name=a%20b+c&flag HTTP/1.1\r\nHost: localhost:8098\r\nX-Thing:  padded \r\nACCEPT: */*").unwrap();
        assert_eq!((h.method.as_str(), h.path.as_str()), ("GET", "/frame.png"));
        let query: Vec<(&str, &str)> = h
            .query
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            query,
            [
                ("wait", "1"),
                ("after", "12"),
                ("name", "a b c"),
                ("flag", "")
            ]
        );
        assert_eq!(
            h.headers[0],
            ("host".to_owned(), "localhost:8098".to_owned())
        );
        assert_eq!(h.headers[1].1, "padded");
        assert_eq!(h.headers[2].0, "accept");
    }

    #[test]
    fn malformed_heads_are_rejected() {
        for bad in [
            "GET /",
            "GET / HTTP/1.1 extra",
            "get / HTTP/1.1",
            "GET  HTTP/1.1",
            "GET / HTTP/1.1\r\nno-colon-here",
            "GET / HTTP/1.1\r\n: empty-name",
            "GET / HTTP/1.1\r\nBad Name: x",
            "GET nopath HTTP/1.1",
            "GET http://127.0.0.1/state HTTP/1.1",
            "GET /?a=%zz HTTP/1.1",
            "GET /?a=%4 HTTP/1.1",
            "GET /?a=%ff HTTP/1.1",
        ] {
            assert_eq!(head(bad).err().map(|e| e.status), Some(400), "{bad}");
        }
        assert_eq!(head("GET / HTTP/2").err().map(|e| e.status), Some(505));
        assert_eq!(parse_head(&[0xff, 0xfe]).err().map(|e| e.status), Some(400));
    }

    #[test]
    fn content_length_is_validated() {
        let length = |v: &str| content_length(&[("content-length".to_owned(), v.to_owned())]);
        assert_eq!(length("0"), Ok(0));
        assert_eq!(length("12"), Ok(12));
        assert_eq!(content_length(&[]), Ok(0), "no body without a length");
        for bad in ["-1", "1x", "", " 1"] {
            assert_eq!(length(bad).unwrap_err().status, 400, "{bad:?}");
        }
        assert_eq!(length("99999999999999999999999").unwrap_err().status, 413);
        let pair = |a: &str, b: &str| {
            content_length(&[
                ("content-length".to_owned(), a.to_owned()),
                ("content-length".to_owned(), b.to_owned()),
            ])
        };
        assert_eq!(pair("1", "2").unwrap_err().status, 400);
        assert_eq!(pair("2", "2"), Ok(2));
        let chunked = [("transfer-encoding".to_owned(), "chunked".to_owned())];
        assert_eq!(content_length(&chunked).unwrap_err().status, 501);
    }

    async fn read(raw: &[u8], max: usize) -> Result<Option<Request>, HttpError> {
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        client.write_all(raw).await.unwrap();
        drop(client);
        read_request(&mut server, max).await
    }

    #[tokio::test]
    async fn bodies_are_read_exactly_and_capped_before_being_read() {
        let request = read(
            b"POST /script HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloEXTRA",
            64,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(request.body, b"hello");
        // A POST without a length has no body (curl -X POST sends none).
        let bare = read(b"POST /quit HTTP/1.1\r\nHost: x\r\n\r\n", 64).await;
        assert!(bare.unwrap().unwrap().body.is_empty());
        // Only the head is sent: a 413 means the body was never awaited.
        let big = read(b"POST /script HTTP/1.1\r\nContent-Length: 65\r\n\r\n", 64).await;
        assert_eq!(big.unwrap_err().status, 413);
        let short = read(
            b"POST /script HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort",
            64,
        )
        .await;
        assert_eq!(short.unwrap_err().status, 400);
        assert_eq!(read(b"", 64).await.unwrap(), None);
        assert_eq!(read(b"GET / HTT", 64).await.unwrap_err().status, 400);
        let huge = format!(
            "GET / HTTP/1.1\r\nX: {}\r\n\r\n",
            "a".repeat(MAX_HEAD + 100)
        );
        assert_eq!(read(huge.as_bytes(), 64).await.unwrap_err().status, 431);
        let endless = format!("GET / HTTP/1.1\r\nX: {}", "a".repeat(MAX_HEAD + 100));
        assert_eq!(read(endless.as_bytes(), 64).await.unwrap_err().status, 431);
    }

    #[tokio::test]
    async fn a_body_split_across_reads_is_reassembled_and_expect_is_answered() {
        let (mut client, mut server) = tokio::io::duplex(8);
        let writer = tokio::spawn(async move {
            client
                .write_all(
                    b"POST /script HTTP/1.1\r\nContent-Length: 20\r\n\r\n01234567890123456789",
                )
                .await
                .unwrap();
        });
        let request = read_request(&mut server, 1024).await.unwrap().unwrap();
        writer.await.unwrap();
        assert_eq!(request.body, b"01234567890123456789");

        let (mut client, mut server) = tokio::io::duplex(1024);
        let reader = tokio::spawn(async move { read_request(&mut server, 1024).await });
        client
            .write_all(
                b"POST /script HTTP/1.1\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\n",
            )
            .await
            .unwrap();
        let mut got = [0u8; 25];
        client.read_exact(&mut got).await.unwrap();
        assert_eq!(&got, b"HTTP/1.1 100 Continue\r\n\r\n");
        client.write_all(b"{}").await.unwrap();
        assert_eq!(reader.await.unwrap().unwrap().unwrap().body, b"{}");
    }

    #[test]
    fn responses_are_framed_with_length_and_close() {
        let bytes = String::from_utf8(Response::error(409, "busy").to_bytes()).unwrap();
        let body = "{\"error\":\"busy\"}\n";
        assert_eq!(
            bytes,
            format!(
                "HTTP/1.1 409 Conflict\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
                 Cache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        );
        for (status, text) in [(200, "OK"), (401, "Unauthorized"), (504, "Gateway Timeout")] {
            assert_eq!(reason(status), text);
        }
        let never_cors = Response::ok().to_bytes();
        assert!(!String::from_utf8(never_cors)
            .unwrap()
            .contains("Access-Control"));
    }

    /// Mutation this catches: accepting any `Host`. A page served from
    /// `evil.example` that re-points its name at 127.0.0.1 is same-origin with
    /// this server as far as the browser is concerned; the `Host` header is
    /// what gives it away.
    #[test]
    fn only_address_hosts_and_localhost_are_allowed() {
        let allowed = ["client.lan".to_owned()];
        for host in [
            "127.0.0.1",
            "127.0.0.1:8098",
            "192.168.1.20:8098",
            "[::1]",
            "[::1]:8098",
            "[fe80::1]:1",
            "localhost",
            "LOCALHOST:8098",
            "localhost.",
            "client.lan",
            "Client.LAN:80",
        ] {
            assert!(host_allowed(host, &allowed), "{host}");
        }
        for host in [
            "",
            "evil.example",
            "evil.example:8098",
            "127.0.0.1.evil.example",
            "localhost.evil.example",
            "evil.example.localhost",
            "notlocalhost",
            "2130706433",
            "0x7f.1",
            "127.1",
            "::1",
            "[::1",
            "[::1]x",
            "[::1]:",
            "[::1]:port",
            "[127.0.0.1]",
            "[localhost]",
            "127.0.0.1:",
            "127.0.0.1:80:80",
            "127.0.0.1:x",
            "user@127.0.0.1",
            "127.0.0.1/path",
            "other.lan",
            " 127.0.0.1",
        ] {
            assert!(!host_allowed(host, &allowed), "{host:?}");
        }
        assert!(!host_allowed("client.lan", &[]));
    }

    #[test]
    fn only_the_servers_own_origin_is_same_origin() {
        assert!(same_origin("http://127.0.0.1:8098", "127.0.0.1:8098"));
        assert!(same_origin("http://LOCALHOST:8098", "localhost:8098"));
        for (origin, host) in [
            ("http://evil.example", "127.0.0.1:8098"),
            ("http://127.0.0.1:9999", "127.0.0.1:8098"),
            ("http://127.0.0.1", "127.0.0.1:8098"),
            ("https://127.0.0.1:8098", "127.0.0.1:8098"),
            ("http://127.0.0.1:8098/", "127.0.0.1:8098"),
            ("null", "127.0.0.1:8098"),
            ("", "127.0.0.1:8098"),
            ("127.0.0.1:8098", "127.0.0.1:8098"),
        ] {
            assert!(!same_origin(origin, host), "{origin:?}");
        }
    }

    #[test]
    fn tokens_are_compared_whole_and_parsed_from_bearer_headers() {
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"secret-token", b"secret-token"));
        for (a, b) in [
            ("secret-token", "secret-tokem"),
            ("secret-token", "Secret-token"),
            ("secret-token", "secret-toke"),
            ("secret-token", "secret-token "),
            ("secret-token", ""),
            ("", "x"),
            ("a\0", "a"),
        ] {
            assert!(!constant_time_eq(a.as_bytes(), b.as_bytes()), "{a:?} {b:?}");
            assert!(!constant_time_eq(b.as_bytes(), a.as_bytes()), "{b:?} {a:?}");
        }
        assert_eq!(bearer("Bearer abc"), Some("abc"));
        assert_eq!(bearer("bearer   abc "), Some("abc"));
        assert_eq!(bearer("Basic abc"), None);
        assert_eq!(bearer("Bearer"), None);
        assert_eq!(bearer("abc"), None);
    }

    #[test]
    fn a_token_must_be_usable_and_long_enough_off_loopback() {
        assert!(check_token(None, true).is_ok());
        assert!(
            check_token(Some("short"), true).is_ok(),
            "loopback: any length"
        );
        assert!(check_token(Some("0123456789abcdef"), false).is_ok());
        for (token, loopback) in [
            ("", true),
            ("", false),
            ("has space 0123456789", false),
            ("tab\t0123456789abcdef", true),
            ("ünïcödé-0123456789", false),
            ("short", false),
            ("0123456789abcde", false),
        ] {
            let refused = check_token(Some(token), loopback);
            assert!(
                matches!(refused, Err(ServeError::WeakToken(_))),
                "{token:?}"
            );
        }
    }

    fn events(json: &str) -> Result<Vec<Event>, String> {
        parse_events(json.as_bytes())
    }

    #[test]
    fn input_events_are_parsed() {
        let key = |key, down| Event::Input(ViewerInput::Key { key, down });
        assert_eq!(
            events(r#"{"key":"w","down":true}"#),
            Ok(vec![key(Key::Forward, true)])
        );
        assert_eq!(
            events(
                r#"[{"key":"shift","down":false},{"press":"enter"},{"press":"d"},
                    {"press":"space","ms":250},{"text":"[::1]:7777\n"},{"turn":[30,-10.5]},
                    {"pointer":[0.25,1],"buttons":1},{"pointer":[0,0]}]"#
            ),
            Ok(vec![
                key(Key::Down, false),
                Event::Press {
                    key: Key::Enter,
                    hold_ms: 0
                },
                Event::Press {
                    key: Key::Right,
                    hold_ms: DEFAULT_PRESS_MS
                },
                Event::Press {
                    key: Key::Up,
                    hold_ms: 250
                },
                Event::Text("[::1]:7777\n".into()),
                Event::Turn(30.0, -10.5),
                Event::Input(ViewerInput::Pointer {
                    x: 0.25,
                    y: 1.0,
                    buttons: 1
                }),
                Event::Input(ViewerInput::Pointer {
                    x: 0.0,
                    y: 0.0,
                    buttons: 0
                }),
            ])
        );
        assert_eq!(events("[]"), Ok(vec![]));
    }

    /// A request is refused whole, with a message that says which event and
    /// why, so nothing is half-applied.
    #[test]
    fn malformed_input_events_are_refused_with_a_reason() {
        for (json, why) in [
            ("{nope", "invalid JSON"),
            ("42", "must be a JSON object"),
            ("[[]]", "event 1: an event must be a JSON object"),
            ("{}", "needs one of"),
            (r#"{"key":"w"}"#, "\"down\""),
            (r#"{"key":"w","down":1}"#, "\"down\""),
            (r#"{"key":"hyper","down":true}"#, "unknown key \"hyper\""),
            (r#"{"key":7,"down":true}"#, "a key must be a string"),
            (
                r#"{"key":"w","down":true,"extra":1}"#,
                "unexpected field \"extra\"",
            ),
            (r#"{"press":"enter","ms":-1}"#, "\"ms\""),
            (r#"{"press":"enter","ms":10001}"#, "\"ms\""),
            (r#"{"press":"enter","ms":1.5}"#, "\"ms\""),
            (r#"{"text":5}"#, "must be a string"),
            (r#"{"text":"a\u001bb"}"#, "U+001B"),
            (r#"{"text":"a\u0000"}"#, "U+0000"),
            (r#"{"turn":[1]}"#, "two numbers"),
            (r#"{"turn":[1,"2"]}"#, "finite numbers"),
            (r#"{"turn":"left"}"#, "two numbers"),
            (r#"{"pointer":[2,0]}"#, "0..1"),
            (r#"{"pointer":[0,-0.1]}"#, "0..1"),
            (r#"{"pointer":[0,0],"buttons":256}"#, "\"buttons\""),
            (r#"[{"text":"ok"},{"turn":[1,2]},{"key":"w"}]"#, "event 3"),
        ] {
            let error = events(json).expect_err(json);
            assert!(error.contains(why), "{json}: {error}");
        }
        let many = format!("[{}]", vec![r#"{"turn":[1,1]}"#; MAX_EVENTS + 1].join(","));
        assert!(events(&many).unwrap_err().contains("more than"));
        let exactly = format!("[{}]", vec![r#"{"turn":[1,1]}"#; MAX_EVENTS].join(","));
        assert_eq!(events(&exactly).unwrap().len(), MAX_EVENTS);
        let long = format!(r#"{{"text":"{}"}}"#, "x".repeat(MAX_TEXT_CHARS + 1));
        assert!(events(&long).unwrap_err().contains("longer than"));
        // Presses add up: two of 6 s are over the 10 s a request may hold keys.
        let held = r#"[{"press":"w","ms":6000},{"press":"w","ms":6000}]"#;
        assert!(events(held).unwrap_err().contains("hold keys"));
    }

    #[tokio::test]
    async fn a_response_nobody_reads_is_given_up_on() {
        // A 64-byte pipe whose far end is kept open and never read.
        let (mut server_side, _client_side) = tokio::io::duplex(64);
        let big = vec![b'x'; 64 * 1024];
        let started = std::time::Instant::now();
        let sent = tokio::time::timeout(
            Duration::from_secs(10),
            write_response(&mut server_side, &big, Duration::from_millis(100)),
        )
        .await
        .expect("the write must give up on its own");
        assert!(!sent, "the response did not go out");
        assert!(started.elapsed() >= Duration::from_millis(100));
    }
}
