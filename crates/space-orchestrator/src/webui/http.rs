//! HTTP/1.1 for the web UI: reading one request under hard limits, and
//! writing one response. Hand-rolled on tokio — there is no HTTP crate in this
//! workspace (`space-mcp` does the same for its transport; this module follows
//! the same rules and is deliberately a little stricter, because it faces
//! browsers and may face the internet).
//!
//! # What is accepted
//!
//! One request per connection (every response says `Connection: close`):
//! a request line with an origin-form target (`/path?query`), headers, and a
//! body of exactly `Content-Length` bytes. Everything is bounded **before** it
//! is buffered:
//!
//! | limit | value | answer when exceeded |
//! |---|---|---|
//! | request line + headers | [`MAX_HEAD`] | `431` |
//! | request target | [`MAX_TARGET`] | `414` |
//! | header lines | [`MAX_HEADERS`] | `431` |
//! | body | the caller's `max_body` ([`MAX_BODY`]) | `413`, before reading it |
//! | form fields | [`MAX_FORM_FIELDS`] | `400` |
//!
//! `Transfer-Encoding` is refused (`501`): with no chunked bodies and no
//! connection reuse there is nothing for a request-smuggling trick to hide in.
//! Time limits are the caller's ([`super::Limits`]): the request head has to
//! arrive within one (short) deadline and the whole request within another, so
//! a client trickling bytes cannot hold a connection open (slowloris).

use std::time::Duration;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Largest accepted request line + headers, in bytes.
pub const MAX_HEAD: usize = 16 * 1024;
/// Longest accepted request target (path and query), in bytes.
pub const MAX_TARGET: usize = 2 * 1024;
/// Most header lines accepted in one request.
pub const MAX_HEADERS: usize = 64;
/// Largest accepted request body, in bytes. The forms post a few short fields.
pub const MAX_BODY: usize = 8 * 1024;
/// Most `name=value` pairs accepted in one query string or form body.
pub const MAX_FORM_FIELDS: usize = 32;

/// A request that cannot be served, with the status to answer it with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpError {
    /// The HTTP status code.
    pub status: u16,
    /// What was wrong, for the response body.
    pub message: String,
}

impl HttpError {
    /// An error answering `status` with `message`.
    #[must_use]
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }
}

/// One complete request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// `GET`, `POST`, …
    pub method: String,
    /// The request target as sent: a path and an optional query.
    pub target: String,
    /// Headers in order, names lower-cased, values trimmed.
    pub headers: Vec<(String, String)>,
    /// The body: exactly `Content-Length` bytes.
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

    /// The header called `name` (lower-case) when it must not be ambiguous.
    ///
    /// # Errors
    ///
    /// `400` when the header appears more than once: for `Host`, `Origin` or
    /// `Authorization`, two values mean two readers can disagree about which
    /// one counts.
    pub fn sole_header(&self, name: &str) -> Result<Option<&str>, HttpError> {
        let mut values = self.headers.iter().filter(|(n, _)| n == name);
        let first = values.next().map(|(_, v)| v.as_str());
        if values.next().is_some() {
            return Err(HttpError::new(400, format!("repeated {name} header")));
        }
        Ok(first)
    }

    /// The target without its query string.
    #[must_use]
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or("")
    }

    /// The query string (without the `?`), empty when there is none.
    #[must_use]
    pub fn query(&self) -> &str {
        self.target.split_once('?').map_or("", |(_, query)| query)
    }

    /// The declared `Content-Length`, if any.
    ///
    /// # Errors
    ///
    /// `400` when it is not a plain decimal number or repeats with a different
    /// value, `413` when it does not fit a `usize`, `501` when
    /// `Transfer-Encoding` is present.
    pub fn content_length(&self) -> Result<Option<usize>, HttpError> {
        if self.header("transfer-encoding").is_some() {
            return Err(HttpError::new(501, "Transfer-Encoding is not supported"));
        }
        let mut found: Option<usize> = None;
        for (_, value) in self.headers.iter().filter(|(n, _)| n == "content-length") {
            if value.is_empty() || !value.bytes().all(|b| b.is_ascii_digit()) {
                return Err(HttpError::new(400, "invalid Content-Length"));
            }
            let length: usize = value
                .parse()
                .map_err(|_| HttpError::new(413, "Content-Length too large"))?;
            if found.is_some_and(|earlier| earlier != length) {
                return Err(HttpError::new(400, "conflicting Content-Length headers"));
            }
            found = Some(length);
        }
        Ok(found)
    }
}

/// Whether `byte` may appear in a header name (an HTTP `token`).
const fn is_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'!' | b'#'
                | b'$'
                | b'%'
                | b'&'
                | b'\''
                | b'*'
                | b'+'
                | b'-'
                | b'.'
                | b'^'
                | b'_'
                | b'`'
                | b'|'
                | b'~'
        )
}

/// Parse the bytes **before** the blank line: the request line and headers.
/// The returned request has an empty body.
///
/// # Errors
///
/// `400` for anything malformed, `414` for a target longer than
/// [`MAX_TARGET`], `431` for more than [`MAX_HEADERS`] header lines, `505` for
/// an HTTP version other than 1.0/1.1.
pub fn parse_head(head: &[u8]) -> Result<Request, HttpError> {
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
    if method.is_empty() || method.len() > 16 || !method.bytes().all(|b| b.is_ascii_uppercase()) {
        return Err(HttpError::new(400, "malformed method"));
    }
    if target.len() > MAX_TARGET {
        return Err(HttpError::new(414, "request target too long"));
    }
    // Origin-form only: a path, visible ASCII, nothing a proxy could read
    // differently (no absolute URI, no `*`, no control or non-ASCII bytes).
    if !target.starts_with('/') || !target.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(HttpError::new(400, "malformed request target"));
    }
    if version != "HTTP/1.1" && version != "HTTP/1.0" {
        return Err(HttpError::new(505, "HTTP version not supported"));
    }
    let mut headers = Vec::new();
    for line in lines {
        if headers.len() >= MAX_HEADERS {
            return Err(HttpError::new(431, "too many header lines"));
        }
        let Some((name, value)) = line.split_once(':') else {
            return Err(HttpError::new(400, "malformed header line"));
        };
        if name.is_empty() || !name.bytes().all(is_token_byte) {
            return Err(HttpError::new(400, "malformed header name"));
        }
        let value = value.trim_matches([' ', '\t']);
        if value.bytes().any(|b| (b < 0x20 && b != b'\t') || b == 0x7f) {
            return Err(HttpError::new(400, "malformed header value"));
        }
        headers.push((name.to_ascii_lowercase(), value.to_owned()));
    }
    Ok(Request {
        method: method.to_owned(),
        target: target.to_owned(),
        headers,
        body: Vec::new(),
    })
}

/// Read one request from `stream`: the head, then exactly `Content-Length`
/// body bytes. `Ok(None)` means the peer closed before sending anything.
///
/// The body size is checked against `max_body` **before** it is read. An
/// `Expect: 100-continue` is answered so a client waiting for it sends the body.
///
/// The caller bounds the time ([`super::Limits::read_timeout`]); this bounds
/// the memory.
///
/// # Errors
///
/// An [`HttpError`] naming the status to reply with.
pub async fn read_request<S>(stream: &mut S, max_body: usize) -> Result<Option<Request>, HttpError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    read_request_within(stream, max_body, None).await
}

/// [`read_request`] with a deadline of its own for the **head**: the request
/// line and headers have to be complete within `head_limit` (when given), or
/// the answer is `408`.
///
/// A connection that has not even said what it wants is the cheapest way to
/// hold a slot, so it gets less time than a request whose (bounded) body is
/// still arriving; the caller's deadline covers the whole request.
///
/// # Errors
///
/// As [`read_request`], plus `408` when the head did not arrive in time.
pub async fn read_request_within<S>(
    stream: &mut S,
    max_body: usize,
    head_limit: Option<Duration>,
) -> Result<Option<Request>, HttpError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let head = read_head(stream);
    let head = match head_limit {
        Some(limit) => tokio::time::timeout(limit, head)
            .await
            .map_err(|_| HttpError::new(408, "the request head did not arrive in time"))?,
        None => head.await,
    };
    let Some((buf, head_end)) = head? else {
        return Ok(None);
    };
    read_body(stream, buf, head_end, max_body).await.map(Some)
}

/// Read up to and including the blank line that ends the head. Returns what
/// was read (which may run into the body) and where the head ends in it;
/// `Ok(None)` means the peer closed before sending anything.
async fn read_head<S>(stream: &mut S) -> Result<Option<(Vec<u8>, usize)>, HttpError>
where
    S: AsyncRead + Unpin,
{
    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    let mut chunk = [0u8; 2048];
    // Where the search for the blank line resumes: three bytes back from what
    // was already searched, so a terminator split across two reads is found
    // and a client sending one byte at a time costs linear, not quadratic, work.
    let mut searched = 0usize;
    let head_end = loop {
        let from = searched.saturating_sub(3);
        if let Some(at) = buf
            .get(from..)
            .and_then(|tail| tail.windows(4).position(|w| w == b"\r\n\r\n"))
        {
            break from + at;
        }
        searched = buf.len();
        if buf.len() > MAX_HEAD {
            return Err(HttpError::new(431, "request head too large"));
        }
        let n = stream
            .read(&mut chunk)
            .await
            .map_err(|_| HttpError::new(400, "read failed"))?;
        if n == 0 {
            return if buf.is_empty() {
                Ok(None)
            } else {
                Err(HttpError::new(400, "connection closed mid-request"))
            };
        }
        buf.extend_from_slice(chunk.get(..n).unwrap_or(&[]));
    };
    if head_end > MAX_HEAD {
        return Err(HttpError::new(431, "request head too large"));
    }
    Ok(Some((buf, head_end)))
}

/// Parse the head in `buf[..head_end]` and read the body it declares.
async fn read_body<S>(
    stream: &mut S,
    mut buf: Vec<u8>,
    head_end: usize,
    max_body: usize,
) -> Result<Request, HttpError>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let head = buf.get(..head_end).unwrap_or(&[]);
    // `100 Continue` exists from HTTP/1.1 on; a 1.0 client that sent `Expect`
    // anyway would read the interim response as the final one.
    let speaks_http11 = head
        .split(|byte| *byte == b'\r')
        .next()
        .is_some_and(|request_line| request_line.ends_with(b" HTTP/1.1"));
    let mut request = parse_head(head)?;
    let length = match (request.content_length()?, request.method.as_str()) {
        (Some(length), _) => length,
        (None, "POST") => return Err(HttpError::new(411, "Content-Length required")),
        (None, _) => 0,
    };
    if length > max_body {
        return Err(HttpError::new(
            413,
            format!("a body of {length} bytes exceeds the {max_body} byte limit"),
        ));
    }
    let mut body = buf.split_off(head_end + 4);
    if length > body.len()
        && speaks_http11
        && request
            .header("expect")
            .is_some_and(|v| v.eq_ignore_ascii_case("100-continue"))
    {
        stream
            .write_all(b"HTTP/1.1 100 Continue\r\n\r\n")
            .await
            .map_err(|_| HttpError::new(400, "write failed"))?;
    }
    // Anything past the declared length is not part of this request, and the
    // connection is not reused: drop it.
    body.truncate(length);
    let have = body.len();
    body.resize(length, 0);
    if let Some(rest) = body.get_mut(have..) {
        stream
            .read_exact(rest)
            .await
            .map_err(|_| HttpError::new(400, "connection closed before the full body arrived"))?;
    }
    request.body = body;
    Ok(request)
}

/// Decode one `application/x-www-form-urlencoded` component: `+` is a space,
/// `%XX` a byte. `None` for a malformed escape or a result that is not UTF-8.
#[must_use]
pub fn percent_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut at = 0;
    while let Some(&byte) = bytes.get(at) {
        match byte {
            b'+' => out.push(b' '),
            b'%' => {
                let high = char::from(*bytes.get(at + 1)?).to_digit(16)?;
                let low = char::from(*bytes.get(at + 2)?).to_digit(16)?;
                out.push(u8::try_from(high * 16 + low).ok()?);
                at += 2;
            }
            other => out.push(other),
        }
        at += 1;
    }
    String::from_utf8(out).ok()
}

/// Parse a query string or a form body into `(name, value)` pairs, in order.
///
/// # Errors
///
/// `400` for a malformed escape, text that is not UTF-8, or more than
/// [`MAX_FORM_FIELDS`] pairs.
pub fn parse_urlencoded(input: &str) -> Result<Vec<(String, String)>, HttpError> {
    let mut fields = Vec::new();
    for pair in input.split('&').filter(|pair| !pair.is_empty()) {
        if fields.len() >= MAX_FORM_FIELDS {
            return Err(HttpError::new(400, "too many form fields"));
        }
        let (name, value) = pair.split_once('=').unwrap_or((pair, ""));
        let (Some(name), Some(value)) = (percent_decode(name), percent_decode(value)) else {
            return Err(HttpError::new(400, "malformed form encoding"));
        };
        fields.push((name, value));
    }
    Ok(fields)
}

/// The first value of `name` in `fields`.
#[must_use]
pub fn field<'a>(fields: &'a [(String, String)], name: &str) -> Option<&'a str> {
    fields
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, v)| v.as_str())
}

/// The reason phrase for `status`.
#[must_use]
pub const fn reason(status: u16) -> &'static str {
    match status {
        100 => "Continue",
        200 => "OK",
        303 => "See Other",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Content Too Large",
        414 => "URI Too Long",
        415 => "Unsupported Media Type",
        429 => "Too Many Requests",
        431 => "Request Header Fields Too Large",
        501 => "Not Implemented",
        503 => "Service Unavailable",
        505 => "HTTP Version Not Supported",
        _ => "Internal Server Error",
    }
}

/// The headers every response carries, whatever it is.
///
/// * `Cache-Control: no-store` — the page shows live state and, behind auth,
///   must not sit in a shared cache.
/// * `X-Content-Type-Options: nosniff` — nothing is ever reinterpreted as
///   script or style.
/// * `X-Frame-Options: DENY` (and `frame-ancestors 'none'` in the CSP) — the
///   page cannot be framed, so its buttons cannot be click-jacked.
/// * `Referrer-Policy: same-origin` — **not** `no-referrer`: that makes
///   browsers send `Origin: null` on the page's own form posts, which the
///   cross-site check would then have to refuse.
/// * `Cross-Origin-Resource-Policy` / `Cross-Origin-Opener-Policy:
///   same-origin` — no other site can embed or script against the responses.
pub const SECURITY_HEADERS: [(&str, &str); 6] = [
    ("Cache-Control", "no-store"),
    ("X-Content-Type-Options", "nosniff"),
    ("X-Frame-Options", "DENY"),
    ("Referrer-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Cross-Origin-Opener-Policy", "same-origin"),
];

/// One response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Response {
    /// The status code.
    pub status: u16,
    /// The `Content-Type`.
    pub content_type: &'static str,
    /// Extra headers (`Location`, `WWW-Authenticate`, `Allow`, …).
    pub headers: Vec<(&'static str, String)>,
    /// The body.
    pub body: Vec<u8>,
    /// The nonce of the page's one inline `<style>` element, when it has one.
    /// Without it the policy allows no style at all.
    pub style_nonce: Option<String>,
}

impl Response {
    /// A response with `body` of type `content_type`.
    #[must_use]
    pub fn new(status: u16, content_type: &'static str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status,
            content_type,
            headers: Vec::new(),
            body: body.into(),
            style_nonce: None,
        }
    }

    /// A plain-text response.
    #[must_use]
    pub fn text(status: u16, message: impl AsRef<str>) -> Self {
        let mut body = message.as_ref().as_bytes().to_vec();
        body.push(b'\n');
        Self::new(status, "text/plain; charset=utf-8", body)
    }

    /// Add a header.
    #[must_use]
    pub fn with_header(mut self, name: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((name, value.into()));
        self
    }

    /// The first extra header called `name` (case-insensitive).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The `Content-Security-Policy` of this response.
    ///
    /// `default-src 'none'` and no `script-src`: **no script runs, from
    /// anywhere, ever** — the UI has none. The only things allowed are the
    /// page's one inline stylesheet (by nonce, so injected markup could not
    /// bring its own), the empty `data:` favicon, and form posts back to the
    /// same origin.
    #[must_use]
    pub fn content_security_policy(&self) -> String {
        let style = self.style_nonce.as_ref().map_or_else(String::new, |nonce| {
            format!("style-src 'nonce-{nonce}'; img-src data:; ")
        });
        format!(
            "default-src 'none'; {style}form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
        )
    }

    /// The exact bytes written to the socket. `head_only` leaves the body out
    /// (the answer to a `HEAD`) while keeping its `Content-Length`.
    #[must_use]
    pub fn to_bytes(&self, head_only: bool) -> Vec<u8> {
        let mut head = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\n",
            self.status,
            reason(self.status),
            self.content_type,
            self.body.len()
        );
        for (name, value) in SECURITY_HEADERS {
            push_header(&mut head, name, value);
        }
        push_header(
            &mut head,
            "Content-Security-Policy",
            &self.content_security_policy(),
        );
        for (name, value) in &self.headers {
            push_header(&mut head, name, value);
        }
        head.push_str("Connection: close\r\n\r\n");
        let mut out = head.into_bytes();
        if !head_only {
            out.extend_from_slice(&self.body);
        }
        out
    }
}

/// Append `name: value`, with anything that could end the line removed from
/// the value (no response header is built from request text today; this keeps
/// it impossible to split a response by accident later).
fn push_header(head: &mut String, name: &str, value: &str) {
    head.push_str(name);
    head.push_str(": ");
    head.extend(value.chars().filter(|c| !c.is_control()));
    head.push_str("\r\n");
}

/// Write `bytes` to `stream` and shut it down, giving up after `limit`.
///
/// Returns whether the whole response went out in time. A client that does not
/// read must not be able to hold its connection (and its slot) open.
pub async fn write_response<S>(stream: &mut S, bytes: &[u8], limit: Duration) -> bool
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

    fn head(raw: &str) -> Result<Request, HttpError> {
        parse_head(raw.as_bytes())
    }

    #[test]
    fn a_request_line_and_its_headers_are_parsed() {
        let request = head(
            "POST /action/split?x=1 HTTP/1.1\r\nHost: 127.0.0.1:7980\r\nContent-Length:  12 \r\nX-Thing:\tv",
        )
        .expect("parses");
        assert_eq!(request.method, "POST");
        assert_eq!(request.target, "/action/split?x=1");
        assert_eq!(request.path(), "/action/split");
        assert_eq!(request.query(), "x=1");
        assert_eq!(request.header("host"), Some("127.0.0.1:7980"));
        assert_eq!(request.header("x-thing"), Some("v"));
        assert_eq!(request.content_length(), Ok(Some(12)));
        assert_eq!(head("GET / HTTP/1.0").expect("1.0 is fine").query(), "");
    }

    #[test]
    fn malformed_heads_are_refused() {
        for bad in [
            "GET /",
            "GET / HTTP/1.1 extra",
            "get / HTTP/1.1",
            "GET  HTTP/1.1",
            "GET http://evil.example/ HTTP/1.1",
            "GET * HTTP/1.1",
            "GET /a\tb HTTP/1.1",
            "GET /caf\u{e9} HTTP/1.1",
            "GET / HTTP/1.1\r\nno-colon-here",
            "GET / HTTP/1.1\r\n: empty-name",
            "GET / HTTP/1.1\r\nBad Name: x",
            "GET / HTTP/1.1\r\n folded: x",
            "GET / HTTP/1.1\r\nX: a\u{0}b",
            "GET / HTTP/1.1\r\nX: a\nInjected: b",
            "AVERYLONGMETHODNAMEINDEED / HTTP/1.1",
        ] {
            assert_eq!(head(bad).expect_err(bad).status, 400, "{bad:?}");
        }
        assert_eq!(head("GET / HTTP/2").unwrap_err().status, 505);
        assert_eq!(parse_head(&[0xff, 0xfe]).unwrap_err().status, 400);
    }

    #[test]
    fn the_target_and_the_header_count_are_bounded() {
        let long = format!("GET /{} HTTP/1.1", "a".repeat(MAX_TARGET));
        assert_eq!(head(&long).unwrap_err().status, 414);
        let fits = format!("GET /{} HTTP/1.1", "a".repeat(MAX_TARGET - 1));
        assert!(head(&fits).is_ok());

        let mut many = String::from("GET / HTTP/1.1");
        for n in 0..MAX_HEADERS {
            many.push_str(&format!("\r\nX-{n}: v"));
        }
        assert!(head(&many).is_ok(), "exactly the limit is accepted");
        many.push_str("\r\nOne-More: v");
        assert_eq!(head(&many).unwrap_err().status, 431);
    }

    #[test]
    fn content_length_is_validated() {
        let length = |v: &str| {
            head(&format!("POST / HTTP/1.1\r\nContent-Length: {v}"))
                .expect("head parses")
                .content_length()
        };
        assert_eq!(length("0"), Ok(Some(0)));
        assert_eq!(length("-1").unwrap_err().status, 400);
        assert_eq!(length("1x").unwrap_err().status, 400);
        assert_eq!(length("").unwrap_err().status, 400);
        assert_eq!(length("99999999999999999999999").unwrap_err().status, 413);
        let twice = head("POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2").unwrap();
        assert_eq!(twice.content_length().unwrap_err().status, 400);
        let chunked = head("POST / HTTP/1.1\r\nTransfer-Encoding: chunked").unwrap();
        assert_eq!(chunked.content_length().unwrap_err().status, 501);
    }

    #[test]
    fn a_repeated_header_that_must_be_unique_is_refused() {
        let request = head("GET / HTTP/1.1\r\nHost: a\r\nHost: b\r\nAccept: x").unwrap();
        assert_eq!(request.sole_header("host").unwrap_err().status, 400);
        assert_eq!(request.sole_header("accept"), Ok(Some("x")));
        assert_eq!(request.sole_header("origin"), Ok(None));
    }

    async fn read(raw: &[u8], max: usize) -> Result<Option<Request>, HttpError> {
        let (mut client, mut server) = tokio::io::duplex(64 * 1024);
        client.write_all(raw).await.unwrap();
        drop(client);
        read_request(&mut server, max).await
    }

    #[tokio::test]
    async fn a_body_is_read_to_exactly_its_declared_length() {
        let request = read(
            b"POST / HTTP/1.1\r\nContent-Length: 5\r\n\r\nhelloEXTRA",
            64,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(request.body, b"hello");
        let none = read(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n", 64)
            .await
            .unwrap()
            .unwrap();
        assert!(none.body.is_empty());
    }

    /// The head arrives one byte at a time: the terminator is found although it
    /// is never inside a single read.
    #[tokio::test]
    async fn a_request_trickled_byte_by_byte_is_reassembled() {
        let (mut client, mut server) = tokio::io::duplex(1);
        let writer = tokio::spawn(async move {
            for byte in b"POST /a HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody" {
                client.write_all(&[*byte]).await.unwrap();
            }
        });
        let request = read_request(&mut server, 64).await.unwrap().unwrap();
        writer.await.unwrap();
        assert_eq!((request.path(), &request.body[..]), ("/a", &b"body"[..]));
    }

    /// Mutation this catches: checking the size after buffering. Only the head
    /// is sent here, so a `413` proves the body was never waited for.
    #[tokio::test]
    async fn an_oversized_body_is_refused_before_it_is_read() {
        let raw = format!(
            "POST / HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        );
        assert_eq!(
            read(raw.as_bytes(), MAX_BODY).await.unwrap_err().status,
            413
        );
        let fits = format!(
            "POST / HTTP/1.1\r\nContent-Length: {MAX_BODY}\r\n\r\n{}",
            "a".repeat(MAX_BODY)
        );
        assert_eq!(
            read(fits.as_bytes(), MAX_BODY)
                .await
                .unwrap()
                .unwrap()
                .body
                .len(),
            MAX_BODY
        );
    }

    #[tokio::test]
    async fn an_oversized_or_unterminated_head_is_refused() {
        let huge = format!(
            "GET / HTTP/1.1\r\nX: {}\r\n\r\n",
            "a".repeat(MAX_HEAD + 100)
        );
        assert_eq!(read(huge.as_bytes(), 16).await.unwrap_err().status, 431);
        let endless = format!("GET / HTTP/1.1\r\nX: {}", "a".repeat(MAX_HEAD + 100));
        assert_eq!(read(endless.as_bytes(), 16).await.unwrap_err().status, 431);
    }

    #[tokio::test]
    async fn other_broken_requests_get_their_status() {
        assert_eq!(read(b"", 16).await.unwrap(), None);
        assert_eq!(read(b"GET / HTT", 16).await.unwrap_err().status, 400);
        assert_eq!(
            read(b"POST / HTTP/1.1\r\n\r\n", 16)
                .await
                .unwrap_err()
                .status,
            411
        );
        assert_eq!(
            read(b"POST / HTTP/1.1\r\nContent-Length: 9\r\n\r\nshort", 16)
                .await
                .unwrap_err()
                .status,
            400
        );
        assert_eq!(
            read(
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n",
                16
            )
            .await
            .unwrap_err()
            .status,
            501
        );
    }

    #[tokio::test]
    async fn expect_continue_is_answered() {
        let (mut client, mut server) = tokio::io::duplex(1024);
        let reader = tokio::spawn(async move { read_request(&mut server, 64).await });
        client
            .write_all(b"POST / HTTP/1.1\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\n")
            .await
            .unwrap();
        let mut got = [0u8; 25];
        client.read_exact(&mut got).await.unwrap();
        assert_eq!(&got, b"HTTP/1.1 100 Continue\r\n\r\n");
        client.write_all(b"ok").await.unwrap();
        assert_eq!(reader.await.unwrap().unwrap().unwrap().body, b"ok");
    }

    /// Mutation this catches: answering `Expect: 100-continue` whatever the
    /// version. `100 Continue` does not exist in HTTP/1.0: a 1.0 client would
    /// take the interim response for the answer.
    #[tokio::test]
    async fn expect_continue_is_not_answered_to_an_http_1_0_client() {
        let (mut client, mut server) = tokio::io::duplex(1024);
        let reader = tokio::spawn(async move {
            let request = read_request(&mut server, 64).await;
            // Whatever the server wrote while reading stays readable.
            drop(server);
            request
        });
        client
            .write_all(b"POST / HTTP/1.0\r\nContent-Length: 2\r\nExpect: 100-continue\r\n\r\n")
            .await
            .unwrap();
        // Give a wrong implementation the time to write its interim response.
        tokio::time::sleep(Duration::from_millis(50)).await;
        client.write_all(b"ok").await.unwrap();
        assert_eq!(reader.await.unwrap().unwrap().unwrap().body, b"ok");
        let mut written = Vec::new();
        client.read_to_end(&mut written).await.unwrap();
        assert!(
            written.is_empty(),
            "nothing is written before the response: {:?}",
            String::from_utf8_lossy(&written)
        );
    }

    /// The head has a deadline of its own, shorter than the request's: a
    /// connection that has not finished saying what it wants is answered `408`
    /// when it runs out, while a head that is complete leaves the body the
    /// rest of the caller's time.
    ///
    /// Mutation this catches: one deadline for everything — an idle connection
    /// then holds its slot for as long as a slow upload may take.
    #[tokio::test]
    async fn an_unfinished_head_runs_out_of_time_before_a_body_does() {
        let short = Some(Duration::from_millis(100));
        // Half a head, then silence.
        let (mut client, mut server) = tokio::io::duplex(1024);
        client
            .write_all(b"GET / HTTP/1.1\r\nHost: x")
            .await
            .unwrap();
        let started = std::time::Instant::now();
        let err = read_request_within(&mut server, 64, short)
            .await
            .expect_err("the head never completes");
        assert_eq!(err.status, 408);
        assert!(started.elapsed() >= Duration::from_millis(90));
        assert!(started.elapsed() < Duration::from_secs(5));
        drop(client);

        // A complete head, and a body that arrives after the head's deadline.
        let (mut client, mut server) = tokio::io::duplex(1024);
        let reader = tokio::spawn(async move { read_request_within(&mut server, 64, short).await });
        client
            .write_all(b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\n")
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(300)).await;
        client.write_all(b"late").await.unwrap();
        assert_eq!(reader.await.unwrap().unwrap().unwrap().body, b"late");

        // No head deadline at all behaves as before.
        let (mut client, mut server) = tokio::io::duplex(1024);
        client.write_all(b"GET / HTTP/1.1\r\n\r\n").await.unwrap();
        assert!(read_request_within(&mut server, 64, None)
            .await
            .unwrap()
            .is_some());
        assert_eq!(reason(429), "Too Many Requests");
    }

    #[test]
    fn form_encoding_is_decoded() {
        assert_eq!(percent_decode("a+b%20c%2B"), Some("a b c+".into()));
        assert_eq!(percent_decode("%C3%A9"), Some("é".into()));
        assert_eq!(percent_decode(""), Some(String::new()));
        for bad in ["%", "%2", "%zz", "%+1", "%ff"] {
            assert_eq!(percent_decode(bad), None, "{bad}");
        }
        let fields = parse_urlencoded("shard=1&octant=7&&csrf=ab%3Dcd&flag").expect("parses");
        assert_eq!(field(&fields, "shard"), Some("1"));
        assert_eq!(field(&fields, "csrf"), Some("ab=cd"));
        assert_eq!(field(&fields, "flag"), Some(""));
        assert_eq!(field(&fields, "missing"), None);
        assert_eq!(parse_urlencoded("").expect("empty").len(), 0);
        assert_eq!(parse_urlencoded("a=%zz").unwrap_err().status, 400);
        // The first value wins when a name repeats.
        let twice = parse_urlencoded("shard=1&shard=2").expect("parses");
        assert_eq!(field(&twice, "shard"), Some("1"));
    }

    #[test]
    fn the_number_of_form_fields_is_bounded() {
        let many = vec!["a=1"; MAX_FORM_FIELDS].join("&");
        assert_eq!(
            parse_urlencoded(&many).expect("at the limit").len(),
            MAX_FORM_FIELDS
        );
        let over = vec!["a=1"; MAX_FORM_FIELDS + 1].join("&");
        assert_eq!(parse_urlencoded(&over).unwrap_err().status, 400);
    }

    #[test]
    fn every_response_carries_the_security_headers_and_closes() {
        let bytes = String::from_utf8(Response::text(404, "nope").to_bytes(false)).unwrap();
        assert!(bytes.starts_with("HTTP/1.1 404 Not Found\r\n"), "{bytes}");
        for needle in [
            "Content-Type: text/plain; charset=utf-8\r\n",
            "Content-Length: 5\r\n",
            "Cache-Control: no-store\r\n",
            "X-Content-Type-Options: nosniff\r\n",
            "X-Frame-Options: DENY\r\n",
            "Referrer-Policy: same-origin\r\n",
            "Content-Security-Policy: default-src 'none'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'\r\n",
            "Connection: close\r\n\r\nnope\n",
        ] {
            assert!(bytes.contains(needle), "missing {needle:?} in {bytes}");
        }
        assert!(!bytes.contains("script-src"), "no script is ever allowed");
    }

    #[test]
    fn a_page_allows_exactly_its_own_stylesheet() {
        let mut page = Response::new(200, "text/html; charset=utf-8", "<p>x</p>");
        page.style_nonce = Some("abc123".into());
        let policy = page.content_security_policy();
        assert_eq!(
            policy,
            "default-src 'none'; style-src 'nonce-abc123'; img-src data:; form-action 'self'; \
             base-uri 'none'; frame-ancestors 'none'"
        );
        assert!(!policy.contains("unsafe-inline"));
    }

    #[test]
    fn a_head_response_keeps_the_length_and_drops_the_body() {
        let response = Response::text(200, "hello");
        let bytes = String::from_utf8(response.to_bytes(true)).unwrap();
        assert!(bytes.contains("Content-Length: 6\r\n"));
        assert!(bytes.ends_with("Connection: close\r\n\r\n"));
    }

    /// Mutation this catches: writing a header value as given. A value with a
    /// line break would start a header (or a body) of the attacker's choosing.
    #[test]
    fn a_header_value_cannot_split_the_response() {
        let response =
            Response::text(303, "x").with_header("Location", "/?a=1\r\nSet-Cookie: evil=1");
        let bytes = String::from_utf8(response.to_bytes(false)).unwrap();
        assert!(bytes.contains("Location: /?a=1Set-Cookie: evil=1\r\n"));
        assert!(!bytes.contains("\r\nSet-Cookie"));
        assert_eq!(
            response.header("location"),
            Some("/?a=1\r\nSet-Cookie: evil=1")
        );
    }

    /// Mutation this catches: writing with no bound. A client that never reads
    /// must not keep its connection slot.
    #[tokio::test]
    async fn a_response_nobody_reads_is_given_up_on() {
        let (mut server_side, _client_side) = tokio::io::duplex(64);
        let big = vec![b'x'; 64 * 1024];
        let sent = tokio::time::timeout(
            Duration::from_secs(10),
            write_response(&mut server_side, &big, Duration::from_millis(100)),
        )
        .await
        .expect("the write gives up on its own");
        assert!(!sent, "the response did not go out");

        let (mut server_side, mut client_side) = tokio::io::duplex(64);
        let reader = tokio::spawn(async move {
            let mut got = Vec::new();
            client_side.read_to_end(&mut got).await.unwrap();
            got
        });
        assert!(write_response(&mut server_side, &big, Duration::from_secs(10)).await);
        drop(server_side);
        assert_eq!(reader.await.unwrap().len(), big.len());
    }
}
