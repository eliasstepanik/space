//! Server addresses as a person types them: `host`, `host:port`, `[v6]:port`.
//!
//! [`ServerAddress::parse`] only checks the *shape* of the text — it never
//! touches the network, so the connect screen can refuse nonsense at once.
//! [`ServerAddress::resolve`] turns it into socket addresses (a DNS lookup for
//! a name), and [`dial`] connects to the first one that answers.

use std::fmt;
use std::net::{IpAddr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use space_net::NetError;
use space_protocol::RejectReason;

use crate::link::ServerLink;

/// The port a shard listens on unless told otherwise (`space-server --bind`
/// defaults to `127.0.0.1:7777`), and so the port assumed when none is typed.
pub const DEFAULT_PORT: u16 = 7777;

/// What the connect screen's address field starts with.
pub const DEFAULT_ADDRESS: &str = "127.0.0.1:7777";

/// Longest host name accepted, in bytes (the DNS limit).
pub const MAX_HOST_LEN: usize = 253;

/// How long a name lookup may take.
pub const RESOLVE_TIMEOUT: Duration = Duration::from_secs(5);

/// The most addresses of one name that are tried.
const MAX_CANDIDATES: usize = 4;

/// The host part of a [`ServerAddress`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Host {
    /// An IPv4 or IPv6 literal.
    Ip(IpAddr),
    /// A host name, to be resolved.
    Name(String),
}

/// A server address: a host and a port.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServerAddress {
    /// Where.
    pub host: Host,
    /// Which port ([`DEFAULT_PORT`] when none was typed).
    pub port: u16,
}

/// Why a text is not a server address. The message is written for the connect
/// screen's error line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AddressError(
    /// The message.
    pub String,
);

impl fmt::Display for AddressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for AddressError {}

fn bad<T>(message: impl Into<String>) -> Result<T, AddressError> {
    Err(AddressError(message.into()))
}

fn parse_port(text: &str) -> Result<u16, AddressError> {
    match text.parse::<u16>() {
        Ok(port) if port != 0 => Ok(port),
        _ => bad(format!(
            "\"{text}\" is not a port (a number from 1 to 65535)"
        )),
    }
}

fn parse_name(host: &str) -> Result<Host, AddressError> {
    if host.is_empty() {
        return bad("the address has no host name");
    }
    if host.len() > MAX_HOST_LEN {
        return bad("the host name is too long");
    }
    let label_ok = |label: &str| {
        !label.is_empty()
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    };
    // One trailing dot (a fully qualified name) is fine.
    let labels = host.strip_suffix('.').unwrap_or(host);
    if !labels.split('.').all(label_ok) {
        return bad(format!("\"{host}\" is not a host name or an IP address"));
    }
    // All digits and dots, but not an IPv4 address (that parsed earlier):
    // "1.2.3", "999.1.1.1" — a typo, not a name worth asking DNS about.
    if labels.chars().all(|c| c.is_ascii_digit() || c == '.') {
        return bad(format!("\"{host}\" is not a complete IPv4 address"));
    }
    Ok(Host::Name(host.to_ascii_lowercase()))
}

impl ServerAddress {
    /// Read an address as typed: `host`, `host:port`, `1.2.3.4`,
    /// `1.2.3.4:port`, `[v6]`, `[v6]:port` or a bare IPv6 address. Surrounding
    /// white space is ignored; a missing port is [`DEFAULT_PORT`].
    ///
    /// # Errors
    ///
    /// [`AddressError`] with a message fit for the connect screen.
    pub fn parse(text: &str) -> Result<Self, AddressError> {
        let text = text.trim();
        if text.is_empty() {
            return bad("enter the address of a shard, for example 127.0.0.1:7777");
        }
        if text.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return bad("an address has no spaces in it");
        }
        // [v6] or [v6]:port
        if let Some(rest) = text.strip_prefix('[') {
            let Some((inside, after)) = rest.split_once(']') else {
                return bad("the address has a \"[\" without a closing \"]\"");
            };
            let Ok(ip) = inside.parse::<Ipv6Addr>() else {
                return bad(format!("\"{inside}\" is not an IPv6 address"));
            };
            let port = match after.strip_prefix(':') {
                Some(port) => parse_port(port)?,
                None if after.is_empty() => DEFAULT_PORT,
                None => return bad("after \"]\" only \":port\" may follow"),
            };
            return Ok(Self {
                host: Host::Ip(IpAddr::V6(ip)),
                port,
            });
        }
        // A bare IP address (v4, or v6 without brackets) has no port.
        if let Ok(ip) = text.parse::<IpAddr>() {
            return Ok(Self {
                host: Host::Ip(ip),
                port: DEFAULT_PORT,
            });
        }
        let (host, port) = match text.rsplit_once(':') {
            Some((host, _)) if host.contains(':') => {
                return bad(
                    "an address has one \":port\" at most (an IPv6 address with a port is \
                     written [address]:port)",
                );
            }
            Some((host, port)) => (host, parse_port(port)?),
            None => (text, DEFAULT_PORT),
        };
        let host = match host.parse::<IpAddr>() {
            Ok(ip) => Host::Ip(ip),
            Err(_) => parse_name(host)?,
        };
        Ok(Self { host, port })
    }

    /// The socket addresses to try, in order: the literal itself, or what the
    /// name resolves to (IPv4 first — a shard usually listens on IPv4 only —
    /// and at most a handful).
    ///
    /// Must be called inside a tokio runtime.
    ///
    /// # Errors
    ///
    /// A message for the connect screen when the name does not resolve.
    pub async fn resolve(&self) -> Result<Vec<SocketAddr>, String> {
        let name = match &self.host {
            Host::Ip(ip) => return Ok(vec![SocketAddr::new(*ip, self.port)]),
            Host::Name(name) => name,
        };
        let lookup = tokio::net::lookup_host((name.as_str(), self.port));
        let found = match tokio::time::timeout(RESOLVE_TIMEOUT, lookup).await {
            Ok(Ok(found)) => found,
            Ok(Err(e)) => {
                // (The system's own wording goes to the log: it is long, and
                // in the system's language.)
                log::debug!("space-client: looking up {name}: {e}");
                return Err(format!("cannot find the host \"{name}\""));
            }
            Err(_) => return Err(format!("looking up \"{name}\" took too long")),
        };
        let mut addrs: Vec<SocketAddr> = Vec::new();
        for addr in found {
            if !addrs.contains(&addr) {
                addrs.push(addr);
            }
        }
        // Stable: the resolver's order is kept within each family.
        addrs.sort_by_key(|addr| !addr.is_ipv4());
        addrs.truncate(MAX_CANDIDATES);
        if addrs.is_empty() {
            return Err(format!("the host \"{name}\" has no address"));
        }
        Ok(addrs)
    }
}

impl fmt::Display for ServerAddress {
    /// The canonical spelling, always with the port: `host:port`,
    /// `1.2.3.4:port`, `[v6]:port`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.host {
            Host::Ip(IpAddr::V6(ip)) => write!(f, "[{ip}]:{}", self.port),
            Host::Ip(IpAddr::V4(ip)) => write!(f, "{ip}:{}", self.port),
            Host::Name(name) => write!(f, "{name}:{}", self.port),
        }
    }
}

/// Why a connection attempt failed, in words for the connect screen.
#[must_use]
pub fn explain(addr: SocketAddr, error: &NetError) -> String {
    match error {
        NetError::VersionMismatch { seen, expected } => format!(
            "version mismatch: this client speaks protocol {seen}, the server at {addr} \
             requires {expected}"
        ),
        NetError::Rejected(RejectReason::Full) => format!("the server at {addr} is full"),
        NetError::Rejected(RejectReason::Draining) => {
            format!("the server at {addr} is not taking players right now; try again shortly")
        }
        NetError::Rejected(RejectReason::Unauthorized) => {
            format!("the server at {addr} did not accept this client")
        }
        NetError::Rejected(reason) => format!("the server at {addr} refused: {reason:?}"),
        // QUIC runs over UDP: a port nobody listens on does not refuse, it
        // just never answers.
        NetError::Handshake(why) if why.contains("within") || why.contains("timed out") => {
            format!("no answer from {addr} (is a shard running there?)")
        }
        NetError::Handshake(why) => format!("cannot connect to {addr}: {why}"),
        NetError::Protocol(why) => format!("{addr} is not a space shard: {why}"),
        NetError::Io(why) => format!("cannot reach {addr}: {why}"),
    }
}

/// Resolve `address` and join the first shard that answers.
///
/// Every candidate address gets one bounded handshake
/// ([`space_net::HANDSHAKE_TIMEOUT`]); a server that answers with a refusal
/// (wrong protocol version, full) ends the attempt at once, since the other
/// addresses of the same name are the same server. Must be called inside a
/// tokio runtime.
///
/// # Errors
///
/// The reason, in words for the connect screen.
pub async fn dial(address: &ServerAddress) -> Result<ServerLink, String> {
    let candidates = address.resolve().await?;
    let mut last = String::new();
    for addr in candidates {
        match ServerLink::connect(addr).await {
            Ok(link) => return Ok(link),
            Err(e) => {
                let refused = matches!(e, NetError::VersionMismatch { .. } | NetError::Rejected(_));
                last = explain(addr, &e);
                if refused {
                    break;
                }
            }
        }
    }
    Err(last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn parse(text: &str) -> ServerAddress {
        ServerAddress::parse(text).unwrap_or_else(|e| panic!("{text:?}: {e}"))
    }

    #[test]
    fn ipv4_with_and_without_a_port() {
        let ip = Host::Ip(IpAddr::V4(Ipv4Addr::new(192, 168, 0, 12)));
        assert_eq!(
            parse("192.168.0.12"),
            ServerAddress {
                host: ip.clone(),
                port: DEFAULT_PORT
            }
        );
        assert_eq!(
            parse("  192.168.0.12:7000\n"),
            ServerAddress {
                host: ip,
                port: 7000
            }
        );
        assert_eq!(parse("127.0.0.1").to_string(), "127.0.0.1:7777");
        assert_eq!(parse(DEFAULT_ADDRESS).to_string(), DEFAULT_ADDRESS);
    }

    #[test]
    fn ipv6_bare_bracketed_and_with_a_port() {
        let ip = Host::Ip(IpAddr::V6(Ipv6Addr::LOCALHOST));
        for (text, port) in [
            ("::1", DEFAULT_PORT),
            ("[::1]", DEFAULT_PORT),
            ("[::1]:9", 9),
        ] {
            assert_eq!(
                parse(text),
                ServerAddress {
                    host: ip.clone(),
                    port
                },
                "{text}"
            );
        }
        assert_eq!(parse("[fe80::1]:7778").to_string(), "[fe80::1]:7778");
        assert_eq!(parse("2001:db8::7").to_string(), "[2001:db8::7]:7777");
    }

    #[test]
    fn names_with_and_without_a_port() {
        assert_eq!(
            parse("Shard.Example.org"),
            ServerAddress {
                host: Host::Name("shard.example.org".into()),
                port: DEFAULT_PORT
            }
        );
        assert_eq!(parse("localhost:7778").to_string(), "localhost:7778");
        assert_eq!(parse("space-shard_1").port, DEFAULT_PORT);
        assert_eq!(parse("host.example.").to_string(), "host.example.:7777");
    }

    #[test]
    fn nonsense_is_refused_with_a_reason() {
        for bad in [
            "",
            "   ",
            "host name",
            "host:",
            "host:0",
            "host:65536",
            "host:port",
            ":7777",
            "-host",
            "host-.org",
            "a..b",
            "ho$t",
            "1.2.3",
            "999.1.1.1",
            "1.2.3.4.5:7",
            "[::1",
            "[::1]7777",
            "[nope]:1",
            "::1:7777:x",
            "fe80::1:99999",
            "a:b:c",
        ] {
            let err = ServerAddress::parse(bad).expect_err(bad);
            assert!(!err.to_string().is_empty(), "{bad:?}");
        }
        let long = "a".repeat(MAX_HOST_LEN + 1);
        assert!(ServerAddress::parse(&long).is_err());
    }

    #[tokio::test]
    async fn literals_resolve_to_themselves_and_localhost_resolves() {
        let v4 = parse("10.1.2.3:99").resolve().await.unwrap();
        assert_eq!(v4, vec!["10.1.2.3:99".parse().unwrap()]);
        let v6 = parse("[::1]:5").resolve().await.unwrap();
        assert_eq!(v6, vec!["[::1]:5".parse().unwrap()]);
        // `localhost` is in every hosts file; IPv4 is tried first when present.
        let local = parse("localhost:7777").resolve().await.unwrap();
        assert!(!local.is_empty() && local.len() <= MAX_CANDIDATES);
        assert!(local
            .iter()
            .all(|a| a.ip().is_loopback() && a.port() == 7777));
        if local.iter().any(SocketAddr::is_ipv4) {
            assert!(local[0].is_ipv4(), "{local:?}");
        }
    }

    #[tokio::test]
    async fn a_name_that_does_not_exist_fails_with_its_name() {
        // `.invalid` is reserved (RFC 2606): it never resolves.
        let err = parse("no-such-shard.invalid")
            .resolve()
            .await
            .expect_err("resolved a reserved name");
        assert!(err.contains("no-such-shard.invalid"), "{err}");
    }

    #[test]
    fn failures_are_explained_in_plain_words() {
        let addr: SocketAddr = "127.0.0.1:7".parse().unwrap();
        let version = explain(
            addr,
            &NetError::VersionMismatch {
                seen: 4,
                expected: 5,
            },
        );
        assert!(
            version.contains("version mismatch") && version.contains('4') && version.contains('5'),
            "{version}"
        );
        let silent = explain(
            addr,
            &NetError::Handshake("no welcome from 127.0.0.1:7 within 5s".into()),
        );
        assert!(silent.contains("no answer from 127.0.0.1:7"), "{silent}");
        let timed_out = explain(addr, &NetError::Handshake("timed out".into()));
        assert!(timed_out.contains("no answer"), "{timed_out}");
        let other = explain(addr, &NetError::Handshake("bad certificate".into()));
        assert!(other.contains("bad certificate"), "{other}");
    }
}
