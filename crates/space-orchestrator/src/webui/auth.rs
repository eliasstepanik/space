//! Who may talk to the web UI: the token check, the `Host` check, the
//! cross-site check, and the per-process random values they rely on.
//!
//! Three independent questions are answered here, each a pure function so it
//! can be tested without a socket:
//!
//! * **Is the caller an operator?** ([`check_basic`]) — HTTP Basic, any user
//!   name, the password is the configured token, compared in constant time
//!   ([`constant_time_eq`]). Only asked when a token is configured; a
//!   non-loopback bind cannot be configured without one
//!   ([`super::WebUiConfig::new`]).
//! * **Was the request meant for this server?** ([`host_allowed`]) — the `Host`
//!   header must be an IP literal, `localhost`, or a name the operator listed
//!   ([`AllowedHost`]). A page on `evil.example` whose DNS is re-pointed at this
//!   server (DNS rebinding) still sends `Host: evil.example`, and is refused. An
//!   IP literal involves no DNS, so it cannot be rebound.
//! * **Is a server without a token really only reachable locally?**
//!   ([`loopback_host`], [`proxy_header`]) — a UI bound to loopback needs no
//!   token, on the understanding that only this machine can reach it. That
//!   stops being true the moment a proxy on the same machine forwards to it, so
//!   without a token the `Host` has to be a loopback address itself and a
//!   request that shows a proxy's fingerprints is refused.
//! * **Did the UI's own page send this form?** ([`origin_allowed`], plus the
//!   form token from [`random_token`]) — a browser attaches the operator's
//!   Basic credentials to *any* request to this origin, including one a
//!   hostile page submits. So a state-changing request must carry an `Origin`
//!   naming this server (when the browser sent one) **and** the per-process
//!   form token that only a page served by this process contains.

use std::hash::{BuildHasher, Hasher};
use std::net::{Ipv4Addr, Ipv6Addr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// The realm named in the `WWW-Authenticate` challenge.
pub const REALM: &str = "space-orchestrator";

/// Shortest token accepted for a non-loopback bind, in bytes.
///
/// The token is the only thing between the internet and the split/merge
/// buttons; anything short is guessable at the rate the connection limits
/// allow.
pub const MIN_TOKEN_LEN: usize = 16;

/// Whether `given` equals `secret`, in time that depends only on `given`'s
/// length — not on where the two differ, and not on the secret.
///
/// An empty secret matches nothing.
#[must_use]
pub fn constant_time_eq(given: &[u8], secret: &[u8]) -> bool {
    // One implementation for every shared secret in the cluster: the web UI's
    // token, the cluster secret of a shard's registration and the peer token.
    space_net::constant_time_eq(given, secret)
}

/// Decode standard base64 (`A-Z a-z 0-9 + /`, optional `=` padding). `None`
/// for anything else — no whitespace, no URL-safe alphabet, no stray bits.
#[must_use]
pub fn base64_decode(input: &str) -> Option<Vec<u8>> {
    let trimmed = input.trim_end_matches('=');
    if input.len() - trimmed.len() > 2 {
        return None;
    }
    let mut out = Vec::with_capacity(trimmed.len() * 3 / 4);
    let mut bits = 0u32;
    let mut held = 0u32;
    for byte in trimmed.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        bits = (bits << 6) | u32::from(value);
        held += 6;
        if held >= 8 {
            held -= 8;
            out.push(u8::try_from((bits >> held) & 0xff).ok()?);
        }
    }
    // A single leftover character carries six bits of nothing; leftover bits
    // that are not zero mean the text was not produced by an encoder.
    if held >= 6 || bits & ((1 << held) - 1) != 0 {
        return None;
    }
    Some(out)
}

/// The password of an `Authorization: Basic …` header value, as bytes.
/// `None` when the header is not well-formed Basic credentials.
#[must_use]
pub fn basic_password(header: &str) -> Option<Vec<u8>> {
    let (scheme, encoded) = header.split_once(' ')?;
    if !scheme.eq_ignore_ascii_case("basic") {
        return None;
    }
    let decoded = base64_decode(encoded.trim())?;
    // `user:password`; the user name is not looked at, and a password may
    // itself contain colons.
    let colon = decoded.iter().position(|byte| *byte == b':')?;
    decoded.get(colon + 1..).map(<[u8]>::to_vec)
}

/// The result of the token check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Auth {
    /// The right token was presented.
    Granted,
    /// No credentials were sent: answer `401` so a browser asks for them.
    Missing,
    /// Credentials were sent and are wrong (or not Basic at all).
    Wrong,
}

/// Check an `Authorization` header against the configured `token`.
#[must_use]
pub fn check_basic(header: Option<&str>, token: &str) -> Auth {
    let Some(header) = header else {
        return Auth::Missing;
    };
    // A malformed header is compared too (against an empty guess), so "not
    // Basic" and "wrong password" take the same path.
    let given = basic_password(header).unwrap_or_default();
    if constant_time_eq(&given, token.as_bytes()) {
        Auth::Granted
    } else {
        Auth::Wrong
    }
}

/// Split an authority (`host`, `host:port`, `[v6]`, `[v6]:port`) into its host
/// and optional port. `None` when it is not one of those shapes.
///
/// A bracketed host is returned with its brackets.
fn split_authority(authority: &str) -> Option<(&str, Option<&str>)> {
    let (host, port) = if authority.starts_with('[') {
        let end = authority.find(']')?;
        let host = authority.get(..=end)?;
        match authority.get(end + 1..)? {
            "" => (host, None),
            rest => (host, Some(rest.strip_prefix(':')?)),
        }
    } else {
        match authority.rsplit_once(':') {
            Some((host, port)) => (host, Some(port)),
            None => (authority, None),
        }
    };
    if let Some(port) = port {
        if port.is_empty() || port.len() > 5 || !port.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
    }
    if host.is_empty() || (!host.starts_with('[') && host.contains(':')) {
        return None;
    }
    Some((host, port))
}

/// A public name the server may be addressed by (`--webui-host`): `name`, or
/// `name:port` to pin the port as well.
///
/// With a port, a `Host` or `Origin` naming the same host on **another** port
/// is not this server — on a machine that runs several sites under one name,
/// the one on `:8443` has no business posting forms to the one on `:443`. A
/// port that is not written in a `Host` or `Origin` is the default one: 80 or
/// 443 for a `Host` (which says no scheme), the scheme's own for an `Origin`.
///
/// The **scheme** is not, and cannot be, checked: behind a TLS-terminating
/// proxy the browser says `https` while this server only ever sees plain HTTP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedHost {
    /// The name, lower-case, without a trailing dot.
    name: String,
    /// The port it must be reached on (`None`: any).
    port: Option<u16>,
}

impl AllowedHost {
    /// Parse `name` or `name:port`.
    ///
    /// # Errors
    ///
    /// `text` is not a bare host name with an optional port (it has a scheme,
    /// a path, a space, a port that is not 1..=65535, …).
    pub fn parse(text: &str) -> Result<Self, String> {
        let refuse = || {
            format!("{text:?} is not a host name (give the name, or name:port: no scheme, no path)")
        };
        let trimmed = text.trim();
        let (name, port) = match trimmed.rsplit_once(':') {
            Some((name, port)) => {
                let port = port.parse::<u16>().ok().filter(|port| *port != 0);
                (name, Some(port.ok_or_else(refuse)?))
            }
            None => (trimmed, None),
        };
        let name = name.trim_end_matches('.').to_ascii_lowercase();
        let plain = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.');
        if !plain {
            return Err(refuse());
        }
        Ok(Self { name, port })
    }

    /// Whether this entry covers `name` reached on `port`; an unwritten port
    /// counts as any of `defaults`.
    fn covers(&self, name: &str, port: Option<u16>, defaults: &[u16]) -> bool {
        let name = name.strip_suffix('.').unwrap_or(name);
        self.name.eq_ignore_ascii_case(name)
            && match (self.port, port) {
                (None, _) => true,
                (Some(pinned), Some(seen)) => pinned == seen,
                (Some(pinned), None) => defaults.contains(&pinned),
            }
    }
}

impl core::fmt::Display for AllowedHost {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self.port {
            Some(port) => write!(f, "{}:{port}", self.name),
            None => f.write_str(&self.name),
        }
    }
}

/// The port of an authority's port text ([`split_authority`] has checked it
/// is 1 to 5 digits).
fn port_of(port: Option<&str>) -> Option<Option<u16>> {
    match port {
        None => Some(None),
        Some(text) => text.parse::<u16>().ok().map(Some),
    }
}

/// Whether a `Host` header names this server (see the module docs).
///
/// Accepted: any IPv4 or bracketed IPv6 literal, `localhost`, and the names in
/// `allowed` (compared without case; a trailing dot is ignored). The port of
/// an IP literal or of `localhost` is not checked — behind a tunnel, a
/// container port mapping or a proxy it is not the port this process bound;
/// the port of a listed name is checked when the entry pins one
/// ([`AllowedHost`]).
///
/// This is the rule for a server that has a token. Without one, the stricter
/// [`loopback_host`] applies.
#[must_use]
pub fn host_allowed(host: &str, allowed: &[AllowedHost]) -> bool {
    let Some((host, port)) = split_authority(host) else {
        return false;
    };
    if let Some(inner) = host.strip_prefix('[') {
        return inner
            .strip_suffix(']')
            .is_some_and(|v6| v6.parse::<Ipv6Addr>().is_ok());
    }
    if host.parse::<Ipv4Addr>().is_ok() {
        return true;
    }
    let name = host.strip_suffix('.').unwrap_or(host);
    if name.eq_ignore_ascii_case("localhost") {
        return true;
    }
    let Some(port) = port_of(port) else {
        return false;
    };
    // A `Host` says no scheme, so an unwritten port may be either default.
    allowed.iter().any(|a| a.covers(name, port, &[80, 443]))
}

/// Whether a `Host` header is a loopback address: `localhost`, anything in
/// `127.0.0.0/8`, or `[::1]` (any port).
///
/// This is the `Host` rule for a server **without a token**. Such a server is
/// only allowed to exist on a loopback bind, and a browser on the same machine
/// reaches it by one of these names. Any other `Host` — another IP literal, a
/// public name — means the request was addressed to something else and found
/// its way here: through a proxy, a port forward or a DNS trick. Those are
/// exactly the cases in which "loopback, so no token needed" is no longer true.
#[must_use]
pub fn loopback_host(host: &str) -> bool {
    let Some((host, _port)) = split_authority(host) else {
        return false;
    };
    if let Some(inner) = host.strip_prefix('[') {
        return inner
            .strip_suffix(']')
            .and_then(|v6| v6.parse::<Ipv6Addr>().ok())
            .is_some_and(|v6| v6.is_loopback());
    }
    if let Ok(v4) = host.parse::<Ipv4Addr>() {
        return v4.is_loopback();
    }
    host.strip_suffix('.')
        .unwrap_or(host)
        .eq_ignore_ascii_case("localhost")
}

/// The request headers a reverse proxy adds (lower-case).
pub const PROXY_HEADERS: [&str; 6] = [
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "via",
];

/// The first of [`PROXY_HEADERS`] among `names` (lower-case header names), if
/// any: proof that the request came through a proxy.
///
/// A server without a token refuses such a request: the proxy made it
/// reachable from wherever the proxy is reachable, which is not "this machine
/// only". **This is a tripwire, not a guarantee** — a proxy that adds none of
/// these headers and rewrites `Host` to a loopback address is indistinguishable
/// from a local browser. The guarantee is a token.
#[must_use]
pub fn proxy_header<'a>(names: impl IntoIterator<Item = &'a str>) -> Option<&'static str> {
    names
        .into_iter()
        .find_map(|name| PROXY_HEADERS.into_iter().find(|proxy| *proxy == name))
}

/// Whether the `Origin` of a state-changing request is this server.
///
/// `origin` must be `http://authority` or `https://authority` (an `Origin` has
/// no path) and its authority must equal the request's own `Host` header, or
/// be covered by one of the names in `allowed` (for a proxy that rewrites
/// `Host`) — on the port that entry pins, if it pins one ([`AllowedHost`]).
/// The scheme is not compared: behind a TLS proxy the browser says `https`
/// while this server speaks plain HTTP. It only supplies the default port.
///
/// `null` — what a browser sends from a sandboxed frame, a `data:` page, or
/// when the page asked for no referrer — is never this server.
#[must_use]
pub fn origin_allowed(origin: &str, host: &str, allowed: &[AllowedHost]) -> bool {
    let (authority, default_port) = if let Some(rest) = origin.strip_prefix("http://") {
        (rest, 80)
    } else if let Some(rest) = origin.strip_prefix("https://") {
        (rest, 443)
    } else {
        return false;
    };
    if authority.is_empty() || authority.contains(['/', '?', '#', '@', ' ']) {
        return false;
    }
    if authority.eq_ignore_ascii_case(host) {
        return true;
    }
    let Some((name, port)) = split_authority(authority) else {
        return false;
    };
    let Some(port) = port_of(port) else {
        return false;
    };
    allowed
        .iter()
        .any(|a| a.covers(name, port, &[default_port]))
}

/// Whether a `Sec-Fetch-Site` header value is compatible with a form on this
/// server's own page (`same-origin`), or with no page at all (`none`).
/// `cross-site` and `same-site` (a sibling sub-domain is not this origin) are
/// not.
#[must_use]
pub fn fetch_site_allowed(value: &str) -> bool {
    value.eq_ignore_ascii_case("same-origin") || value.eq_ignore_ascii_case("none")
}

/// A value that differs between processes, runs and call sites without being
/// secret: the process id, both clocks, and where the stack and this code
/// happen to live (address-space layout randomisation).
///
/// None of it is unpredictable to somebody on the same machine. It is mixed
/// into [`random_token`] as a second line: if the keys the standard library
/// hands out were ever weaker than assumed, an attacker would still have to
/// guess all of this for the same instant.
fn ambient_entropy() -> [u64; 5] {
    static STARTED: OnceLock<Instant> = OnceLock::new();
    let on_the_stack = 0u8;
    let stack = std::ptr::from_ref(&on_the_stack) as usize as u64;
    let code = ambient_entropy as fn() -> [u64; 5] as usize as u64;
    let monotonic = STARTED.get_or_init(Instant::now).elapsed().as_nanos();
    let wall = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    // The nanosecond counts are folded to 64 bits; the low bits are the ones
    // that vary.
    [
        u64::from(std::process::id()),
        (monotonic as u64) ^ ((monotonic >> 64) as u64),
        (wall as u64) ^ ((wall >> 64) as u64),
        stack,
        code,
    ]
}

/// A keyed hash of `words` under a fresh [`std::collections::hash_map::RandomState`].
fn keyed_hash(words: &[u64]) -> u64 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    for word in words {
        hasher.write_u64(*word);
    }
    hasher.finish()
}

/// 128 bits, chosen once per process, that every token depends on.
///
/// Built from keyed hashes taken on **two threads** — the calling one and a
/// helper started for the purpose — because the standard library draws a
/// thread's hasher keys from the operating system once per thread: two threads
/// are two independent draws, which several `RandomState`s on one thread are
/// not (those only differ by a counter).
fn process_secret() -> [u64; 2] {
    static SECRET: OnceLock<[u64; 2]> = OnceLock::new();
    *SECRET.get_or_init(|| {
        let here = ambient_entropy();
        let draw = move |salt: u64| {
            let mut words = ambient_entropy().to_vec();
            words.extend_from_slice(&here);
            words.push(salt);
            keyed_hash(&words)
        };
        // If no thread can be started the helper's share is drawn here too:
        // weaker (one OS draw instead of two), never absent.
        let helper = std::thread::Builder::new()
            .name("space-orchestrator-seed".into())
            .spawn(move || [draw(0x5eed), draw(0xfeed)])
            .ok()
            .and_then(|thread| thread.join().ok())
            .unwrap_or_else(|| [draw(0x0bad), draw(0xcafe)]);
        [draw(1) ^ helper[0], draw(2) ^ helper[1]]
    })
}

/// 128 unpredictable bits as 32 hex characters.
///
/// There is no random-number crate in this workspace, so this is built on the
/// one source of operating-system randomness the standard library exposes:
/// the keys of a [`std::collections::hash_map::RandomState`]. Each half of a
/// token is a SipHash, under a fresh such state, of a process-wide secret
/// (`process_secret`: keyed hashes drawn on two threads), a process-wide
/// counter (no two tokens hash the same input), and the process id, both
/// clocks and a stack and a code address (`ambient_entropy`).
///
/// # What this relies on
///
/// That `RandomState` keys are unpredictable to anyone outside the process and
/// that SipHash under an unknown key is unpredictable. The standard library
/// documents the keys as "randomly generated" and, today, draws them from the
/// operating system's random source once per thread (later states on that
/// thread differ by a counter) — but it does **not** promise a
/// cryptographically secure source, on every platform, for ever. The other
/// inputs are there for that case; they are guessable by somebody on the same
/// machine and are not what the token's strength is claimed to rest on. A
/// deployment that cannot accept that assumption should not rely on the form
/// token alone: the operator token is the access control, and it is the
/// operator's own secret.
///
/// Used for the form token (one per process) and for the style nonce (one per
/// page).
#[must_use]
pub fn random_token() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let secret = process_secret();
    let ambient = ambient_entropy();
    let half = |salt: u64| {
        let mut words = vec![secret[0], secret[1], n, salt];
        words.extend_from_slice(&ambient);
        keyed_hash(&words)
    };
    format!("{:016x}{:016x}", half(0x5eed), half(0xfeed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_secrets_compare_equal_and_nothing_else_does() {
        assert!(constant_time_eq(b"correct horse", b"correct horse"));
        for wrong in [
            &b""[..],
            b"c",
            b"correct hors",
            b"correct horsf",
            b"correct horse ",
            b"correct horsecorrect horse",
            b"Correct horse",
        ] {
            assert!(!constant_time_eq(wrong, b"correct horse"), "{wrong:?}");
        }
        // A guess that is the secret repeated must not pass, and an empty
        // secret matches nothing — not even an empty guess.
        assert!(!constant_time_eq(b"abab", b"ab"));
        assert!(!constant_time_eq(b"", b""));
        assert!(!constant_time_eq(b"x", b""));
    }

    #[test]
    fn base64_round_trips_the_standard_alphabet() {
        assert_eq!(base64_decode(""), Some(vec![]));
        assert_eq!(base64_decode("Zg=="), Some(b"f".to_vec()));
        assert_eq!(base64_decode("Zm8="), Some(b"fo".to_vec()));
        assert_eq!(base64_decode("Zm9v"), Some(b"foo".to_vec()));
        assert_eq!(base64_decode("Zm9vYg=="), Some(b"foob".to_vec()));
        assert_eq!(base64_decode("Zm9vYmFy"), Some(b"foobar".to_vec()));
        assert_eq!(base64_decode("Zg"), Some(b"f".to_vec()), "padding optional");
        assert_eq!(base64_decode("+/+/"), Some(vec![0xfb, 0xff, 0xbf]));
        for bad in [
            "Z", "Zg===", "Zm9v Zg", "Zm9v\n", "Zm-_", "Zh==", "Zm9=", "====", "é",
        ] {
            assert_eq!(base64_decode(bad), None, "{bad:?}");
        }
    }

    fn basic(user_pass: &str) -> String {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::from("Basic ");
        for chunk in user_pass.as_bytes().chunks(3) {
            let b = [
                chunk[0],
                chunk.get(1).copied().unwrap_or(0),
                chunk.get(2).copied().unwrap_or(0),
            ];
            let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
            for (i, shift) in [18u32, 12, 6, 0].into_iter().enumerate() {
                if i <= chunk.len() {
                    out.push(char::from(ALPHABET[((n >> shift) & 63) as usize]));
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    #[test]
    fn the_password_of_basic_credentials_is_the_token() {
        assert_eq!(
            basic("Aladdin:open sesame"),
            "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="
        );
        let token = "s3cret-token-0123456789";
        assert_eq!(
            check_basic(Some(&basic(&format!("ops:{token}"))), token),
            Auth::Granted
        );
        // Any user name, including none; the scheme is case-insensitive.
        assert_eq!(
            check_basic(Some(&basic(&format!(":{token}"))), token),
            Auth::Granted
        );
        let lower = basic(&format!("x:{token}")).replacen("Basic", "basic", 1);
        assert_eq!(check_basic(Some(&lower), token), Auth::Granted);
        // A password may contain colons: everything after the first one counts.
        assert_eq!(
            check_basic(Some(&basic("user:a:b:c")), "a:b:c"),
            Auth::Granted
        );
    }

    #[test]
    fn wrong_missing_and_malformed_credentials_are_told_apart() {
        let token = "s3cret-token-0123456789";
        assert_eq!(check_basic(None, token), Auth::Missing);
        for wrong in [
            basic("ops:wrong"),
            basic(&format!("ops:{token}x")),
            basic(&format!("{token}:")),
            basic("no-colon-at-all"),
            format!("Bearer {token}"),
            "Basic".to_owned(),
            "Basic !!!".to_owned(),
            String::new(),
        ] {
            assert_eq!(check_basic(Some(&wrong), token), Auth::Wrong, "{wrong:?}");
        }
        // The token as the *user name* is not the token.
        assert_eq!(
            check_basic(Some(&basic(&format!("{token}:other"))), token),
            Auth::Wrong
        );
    }

    fn listed(names: &[&str]) -> Vec<AllowedHost> {
        names
            .iter()
            .map(|name| AllowedHost::parse(name).expect("a host name"))
            .collect()
    }

    #[test]
    fn a_host_is_an_ip_literal_localhost_or_a_listed_name() {
        let none: [AllowedHost; 0] = [];
        for ok in [
            "127.0.0.1",
            "127.0.0.1:7980",
            "localhost",
            "LOCALHOST:7980",
            "localhost.",
            "[::1]",
            "[::1]:7980",
            "10.0.0.5:80",
            "203.0.113.9:7980",
            "[2001:db8::1]:443",
        ] {
            assert!(host_allowed(ok, &none), "{ok}");
        }
        for bad in [
            "",
            "evil.example",
            "evil.example:7980",
            "localhost.evil.example",
            "127.0.0.1.evil.example",
            "127.0.0.1:",
            "127.0.0.1:port",
            "127.0.0.1:123456",
            "::1",
            "[::1",
            "[::1]x",
            "[::1]:",
            "[not-v6]",
            "user@127.0.0.1",
            "127.0.0.1/path",
            "0x7f.0.0.1",
        ] {
            assert!(!host_allowed(bad, &none), "{bad:?}");
        }
        let listed = listed(&["orchestrator.example.org"]);
        assert!(host_allowed("orchestrator.example.org", &listed));
        assert!(host_allowed("Orchestrator.Example.Org:8443", &listed));
        assert!(!host_allowed("other.example.org", &listed));
        assert!(!host_allowed(
            "orchestrator.example.org.evil.example",
            &listed
        ));
    }

    /// Mutation this catches: accepting any IP literal as `Host` when there is
    /// no token. A request for `10.1.2.3` (or a public name) that arrives at a
    /// loopback-only server was sent to something in front of it.
    #[test]
    fn without_a_token_only_a_loopback_host_is_this_server() {
        for ok in [
            "localhost",
            "LOCALHOST:7980",
            "localhost.",
            "127.0.0.1",
            "127.0.0.1:7980",
            "127.9.9.9:1",
            "[::1]",
            "[::1]:7980",
        ] {
            assert!(loopback_host(ok), "{ok}");
        }
        for bad in [
            "",
            "10.1.2.3",
            "192.168.1.5:7980",
            "0.0.0.0",
            "203.0.113.9",
            "[::]",
            "[2001:db8::1]:443",
            "[::ffff:127.0.0.1]",
            "ops.example.org",
            "localhost.evil.example",
            "127.0.0.1.evil.example",
            "128.0.0.1",
            "::1",
            "127.0.0.1:port",
        ] {
            assert!(!loopback_host(bad), "{bad:?}");
        }
    }

    #[test]
    fn a_proxy_gives_itself_away_by_its_headers() {
        assert_eq!(proxy_header(["host", "accept", "user-agent"]), None);
        assert_eq!(proxy_header([]), None);
        for (headers, found) in [
            (vec!["host", "x-forwarded-for"], "x-forwarded-for"),
            (vec!["forwarded", "host"], "forwarded"),
            (vec!["host", "x-real-ip"], "x-real-ip"),
            (vec!["via"], "via"),
            (
                vec!["x-forwarded-host", "x-forwarded-proto"],
                "x-forwarded-host",
            ),
            (vec!["accept", "x-forwarded-proto"], "x-forwarded-proto"),
        ] {
            assert_eq!(
                proxy_header(headers.iter().copied()),
                Some(found),
                "{headers:?}"
            );
        }
        // Only the exact names: a header that merely looks similar is not one.
        assert_eq!(
            proxy_header(["x-forwarded", "viaduct", "x-real-ip-2"]),
            None
        );
    }

    /// `--webui-host name:port` pins the port; without one any port passes.
    ///
    /// Mutation this catches: comparing the name only. Another site under the
    /// same name on another port would pass as "this server" in `Origin`.
    #[test]
    fn a_listed_name_can_pin_its_port() {
        assert_eq!(
            AllowedHost::parse(" Ops.Example.Org. ").map(|h| h.to_string()),
            Ok("ops.example.org".to_owned())
        );
        assert_eq!(
            AllowedHost::parse("ops.example.org:8443").map(|h| h.to_string()),
            Ok("ops.example.org:8443".to_owned())
        );
        for bad in [
            "",
            ":443",
            "http://ops.example.org",
            "ops.example.org/path",
            "a b",
            "ops.example.org:0",
            "ops.example.org:65536",
            "ops.example.org:https",
            "ops.example.org:",
            "[::1]:443",
        ] {
            assert!(AllowedHost::parse(bad).is_err(), "{bad:?}");
        }

        let pinned = listed(&["ops.example.org:8443"]);
        assert!(host_allowed("ops.example.org:8443", &pinned));
        assert!(host_allowed("OPS.example.org.:8443", &pinned));
        assert!(!host_allowed("ops.example.org:9000", &pinned));
        assert!(
            !host_allowed("ops.example.org", &pinned),
            "the default port is not 8443"
        );
        assert!(origin_allowed(
            "https://ops.example.org:8443",
            "127.0.0.1:7980",
            &pinned
        ));
        assert!(!origin_allowed(
            "https://ops.example.org:9000",
            "127.0.0.1:7980",
            &pinned
        ));
        assert!(!origin_allowed(
            "https://ops.example.org",
            "127.0.0.1:7980",
            &pinned
        ));

        // Pinned to a default port: an unwritten port is that port — for the
        // scheme the Origin names, and for either scheme in a Host.
        let default = listed(&["ops.example.org:443"]);
        assert!(host_allowed("ops.example.org", &default));
        assert!(host_allowed("ops.example.org:443", &default));
        assert!(!host_allowed("ops.example.org:8443", &default));
        assert!(origin_allowed(
            "https://ops.example.org",
            "127.0.0.1:7980",
            &default
        ));
        assert!(origin_allowed(
            "https://ops.example.org:443",
            "127.0.0.1:7980",
            &default
        ));
        assert!(
            !origin_allowed("http://ops.example.org", "127.0.0.1:7980", &default),
            "http's default port is 80, not the pinned 443"
        );
        assert!(!origin_allowed(
            "https://ops.example.org:8443",
            "127.0.0.1:7980",
            &default
        ));
    }

    #[test]
    fn an_origin_must_be_this_server() {
        let none: [AllowedHost; 0] = [];
        assert!(origin_allowed(
            "http://127.0.0.1:7980",
            "127.0.0.1:7980",
            &none
        ));
        assert!(origin_allowed(
            "http://localhost:7980",
            "LocalHost:7980",
            &none
        ));
        assert!(origin_allowed("http://[::1]:7980", "[::1]:7980", &none));
        // Behind a TLS proxy the scheme differs; the authority is what counts.
        assert!(origin_allowed(
            "https://ops.example.org",
            "ops.example.org",
            &none
        ));
        for bad in [
            "null",
            "",
            "http://",
            "http://evil.example",
            "http://127.0.0.1:7981",
            "http://127.0.0.1",
            "http://localhost:7980",
            "http://127.0.0.1:7980/",
            "http://127.0.0.1:7980@evil.example",
            "http://evil.example/127.0.0.1:7980",
            "ftp://127.0.0.1:7980",
            "127.0.0.1:7980",
            "http://[::1",
        ] {
            assert!(!origin_allowed(bad, "127.0.0.1:7980", &none), "{bad:?}");
        }
        // A proxy that rewrites Host: the operator lists the public name.
        let listed = listed(&["ops.example.org"]);
        assert!(origin_allowed(
            "https://ops.example.org",
            "127.0.0.1:7980",
            &listed
        ));
        assert!(origin_allowed(
            "https://OPS.example.org:8443",
            "orchestrator:7980",
            &listed
        ));
        assert!(!origin_allowed(
            "https://evil.example",
            "127.0.0.1:7980",
            &listed
        ));
        assert!(!origin_allowed(
            "https://ops.example.org.evil.example",
            "127.0.0.1:7980",
            &listed
        ));
    }

    #[test]
    fn only_a_same_origin_fetch_site_passes() {
        assert!(fetch_site_allowed("same-origin"));
        assert!(fetch_site_allowed("none"));
        assert!(!fetch_site_allowed("cross-site"));
        assert!(!fetch_site_allowed("same-site"));
        assert!(!fetch_site_allowed(""));
    }

    #[test]
    fn random_tokens_are_long_hex_and_never_repeat() {
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..1_000 {
            let token = random_token();
            assert_eq!(token.len(), 32);
            assert!(token.bytes().all(|b| b.is_ascii_hexdigit()));
            assert!(seen.insert(token), "a token repeated");
        }
        // From other threads too (each has hasher keys of its own).
        let elsewhere: Vec<String> = (0..4)
            .map(|_| std::thread::spawn(random_token))
            .map(|thread| thread.join().expect("a token"))
            .collect();
        for token in elsewhere {
            assert!(seen.insert(token), "a token repeated across threads");
        }
    }

    /// Uniqueness alone would be satisfied by a counter. This looks at the
    /// bits: over many tokens every one of the 128 positions is set about half
    /// the time, and the two halves of a token are not copies of each other.
    ///
    /// Mutation this catches: a half that does not depend on the key (a hash
    /// of the counter alone, a constant, the same half twice). It cannot prove
    /// unpredictability — nothing run in-process can; see `random_token`'s docs
    /// for what that rests on.
    #[test]
    fn random_tokens_have_no_stuck_or_copied_bits() {
        const TOKENS: u32 = 2_000;
        let mut ones = [0u32; 128];
        let mut equal_halves = 0;
        for _ in 0..TOKENS {
            let token = random_token();
            let value = u128::from_str_radix(&token, 16).expect("hex");
            for (bit, count) in ones.iter_mut().enumerate() {
                *count += u32::from(value >> bit & 1 == 1);
            }
            equal_halves += u32::from(token[..16] == token[16..]);
        }
        assert_eq!(equal_halves, 0, "the halves are separate draws");
        for (bit, count) in ones.iter().enumerate() {
            // Binomial(2000, 1/2): mean 1000, sigma ~22; this is over 6 sigma.
            assert!(
                (850..=1150).contains(count),
                "bit {bit} was set in {count} of {TOKENS} tokens"
            );
        }
        // The process secret is drawn once and the ambient values are not all
        // zero (a platform where every source came back empty would show here).
        assert_eq!(process_secret(), process_secret());
        assert!(ambient_entropy().iter().filter(|word| **word != 0).count() >= 4);
    }
}
