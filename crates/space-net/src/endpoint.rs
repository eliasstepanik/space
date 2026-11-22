//! Endpoints, the hello handshake and per-connection state.
//!
//! [`server_endpoint`] / [`client_endpoint`] build a quinn endpoint (datagrams
//! enabled, which is the quinn default); [`connect`] and [`accept_connection`]
//! add the `space` handshake on top:
//!
//! ```text
//!   client                          server
//!     │  QUIC connect + TLS 1.3      │
//!     ├────────────────────────────►│  accept
//!     │  open_bi → control stream    │
//!     ├── frame(Hello{protocol}) ──►│  check_hello
//!     │                              │   ├ mismatch ─► frame(Reject) + close
//!     │◄── frame(Welcome{session}) ─┤   └ ok ─► Welcome
//! ```
//!
//! The control stream stays open after the handshake: either side writes
//! point-to-point control messages ([`space_protocol::Redirect`],
//! [`space_protocol::StaleEpoch`]) on it, while bulk state (snapshots) and
//! inputs use the paths in [`crate::framed`].

use std::fmt;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use quinn::{Connection, Endpoint, RecvStream, SendStream, ServerConfig, TransportConfig, VarInt};
use space_protocol::{
    check_hello, Hello, Message, Reject, RejectReason, Welcome, PROTOCOL_VERSION,
};
use tokio::sync::Mutex;

use crate::crypto::make_client_config;
use crate::framed::{read_frame, recv_datagram, recv_stream_message, send_message, send_on_stream};

/// Application close code used when a [`Hello`] is refused.
pub const REJECT_CLOSE_CODE: u32 = 0x20;

/// How long a refusing server waits for the client to acknowledge a [`Reject`]
/// before it closes the connection anyway.
const REJECT_DRAIN: Duration = Duration::from_secs(1);

/// Silence after which a connection is declared dead.
///
/// The peer must not have been heard from for this long: with
/// [`KEEP_ALIVE_INTERVAL`] in force that only happens when the peer (or the
/// path to it) is really gone, never because the application had nothing to say.
pub const IDLE_TIMEOUT: Duration = Duration::from_secs(10);

/// How often an otherwise silent connection sends a QUIC `PING`.
///
/// A third of [`IDLE_TIMEOUT`], so three consecutive pings have to be lost
/// before a live peer is mistaken for a dead one. Without it every link that
/// carries only occasional traffic — a shard's control connection to the
/// orchestrator (epoch messages only), a shard-to-shard connection (handoff
/// messages only) — timed out after ten quiet seconds and was never re-opened.
pub const KEEP_ALIVE_INTERVAL: Duration = Duration::from_secs(3);

/// How long [`connect`] and [`accept_incoming`] wait for the whole handshake
/// (QUIC/TLS plus the hello exchange) before giving up.
///
/// The idle timeout used to bound this implicitly. With keep-alive on it no
/// longer does: a peer whose transport answers but whose application never
/// sends its half of the hello would hold the caller forever.
pub const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

/// Why a connection could not be established or was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum NetError {
    /// The peer's [`Hello`] announced a different
    /// [`PROTOCOL_VERSION`]; the exact reason the server sent back.
    VersionMismatch {
        /// Version the peer announced.
        seen: u32,
        /// Version we speak.
        expected: u32,
    },
    /// The server refused the connection for a non-version reason.
    Rejected(RejectReason),
    /// The QUIC/TLS handshake failed.
    Handshake(String),
    /// The first frame was missing, malformed or of the wrong type.
    Protocol(String),
    /// A transport error.
    Io(String),
}

impl fmt::Display for NetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetError::VersionMismatch { seen, expected } => write!(
                f,
                "peer speaks protocol version {seen}, this build requires {expected}"
            ),
            NetError::Rejected(reason) => write!(f, "connection rejected: {reason:?}"),
            NetError::Handshake(why) => write!(f, "handshake failed: {why}"),
            NetError::Protocol(why) => write!(f, "protocol violation: {why}"),
            NetError::Io(why) => write!(f, "transport error: {why}"),
        }
    }
}

impl std::error::Error for NetError {}

impl From<io::Error> for NetError {
    fn from(err: io::Error) -> Self {
        NetError::Io(err.to_string())
    }
}

/// The transport settings every `space` endpoint uses.
///
/// * **datagrams enabled** (the quinn default, made explicit here because the
///   input path depends on it), with a receive buffer that comfortably holds a
///   second of 30 Hz input traffic.
/// * **keep-alive every [`KEEP_ALIVE_INTERVAL`], idle timeout
///   [`IDLE_TIMEOUT`]**: a quiet link stays up, and a dead peer is still
///   detected by the idle timeout, which [`Conn::is_closed`] reports (a ping
///   that is never acknowledged does not reset the idle timer).
#[must_use]
pub fn transport_config() -> Arc<TransportConfig> {
    let mut t = TransportConfig::default();
    // 1 MiB datagram receive buffer: at 30 Hz a client sends ~30 × ~60 B of
    // input frames per second, so this is minutes of buffering — the point is
    // that a slow poll loop never silently drops an input tick.
    t.datagram_receive_buffer_size(Some(1 << 20));
    t.datagram_send_buffer_size(1 << 20);
    // Generous stream limits: one uni stream per bulk message (see
    // [`crate::framed`]), and a snapshot burst must not run out of stream ids.
    t.max_concurrent_bidi_streams(VarInt::from_u32(64));
    t.max_concurrent_uni_streams(VarInt::from_u32(1024));
    // 10 s of silence = dead peer. QUIC then reports the connection closed,
    // which is what the lifecycle tests observe. The timeout must be set
    // explicitly: quinn's default is 30 s, so the comment above was a promise
    // the code did not keep.
    //
    // Keep-alive is what stops *application* silence from looking like a dead
    // peer (see [`KEEP_ALIVE_INTERVAL`]). Both sides use this config, so either
    // end's pings keep the link up.
    t.keep_alive_interval(Some(KEEP_ALIVE_INTERVAL));
    t.max_idle_timeout(Some(
        quinn::IdleTimeout::try_from(IDLE_TIMEOUT).expect("10 s fits in a QUIC idle timeout"),
    ));
    Arc::new(t)
}

/// A server endpoint bound to `bind`, presenting `server_config`.
///
/// # Errors
///
/// [`io::Error`] when the socket cannot be bound.
pub fn server_endpoint(bind: SocketAddr, server_config: ServerConfig) -> io::Result<Endpoint> {
    let mut config = server_config;
    config.transport_config(transport_config());
    Endpoint::server(config, bind).map_err(io::Error::other)
}

/// A client endpoint that can dial `addr`: bound to an ephemeral port on the
/// wildcard address of **`addr`'s own address family**.
///
/// [`client_endpoint`] picks the family by operating system (IPv4 on Windows,
/// IPv6 elsewhere), which makes an IPv6 target un-diallable on Windows (quinn
/// refuses it with `InvalidRemoteAddress`) and requires a working IPv6 stack
/// everywhere else, even to reach `127.0.0.1`. Binding the family of the
/// address actually being dialled has neither problem, so this is the one to
/// use whenever the destination is known.
///
/// # Errors
///
/// [`io::Error`] when the socket cannot be bound.
pub fn client_endpoint_for(addr: SocketAddr) -> io::Result<Endpoint> {
    let wildcard = match addr {
        SocketAddr::V4(_) => IpAddr::V4(Ipv4Addr::UNSPECIFIED),
        SocketAddr::V6(_) => IpAddr::V6(Ipv6Addr::UNSPECIFIED),
    };
    client_endpoint_on(SocketAddr::new(wildcard, 0))
}

/// A client endpoint bound to an ephemeral port on the wildcard address.
///
/// Prefer [`client_endpoint_for`] when the destination is known: this one
/// cannot know which address family will be dialled and guesses by platform.
///
/// # Errors
///
/// [`io::Error`] when the socket cannot be bound.
pub fn client_endpoint() -> io::Result<Endpoint> {
    let bind: SocketAddr = if cfg!(windows) {
        "0.0.0.0:0".parse().expect("valid wildcard address")
    } else {
        "[::]:0".parse().expect("valid wildcard address")
    };
    client_endpoint_on(bind)
}

/// A client endpoint bound to a specific local address.
///
/// # Errors
///
/// [`io::Error`] when the socket cannot be bound.
pub fn client_endpoint_on(bind: SocketAddr) -> io::Result<Endpoint> {
    let mut config = make_client_config();
    config.transport_config(transport_config());
    let mut endpoint = Endpoint::client(bind).map_err(io::Error::other)?;
    endpoint.set_default_client_config(config);
    Ok(endpoint)
}

/// A connected `space` peer: the quinn connection plus its control stream.
///
/// Cloning a `Conn` is cheap (the quinn connection is `Arc`-backed and the
/// control stream halves sit behind an [`Arc`]<[`Mutex`>), which is what lets a
/// server hand one `Conn` to a replication task and another to the input
/// reader.
#[derive(Clone)]
pub struct Conn {
    conn: Connection,
    control_send: Arc<Mutex<SendStream>>,
    control_recv: Arc<Mutex<RecvStream>>,
}

impl fmt::Debug for Conn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Conn")
            .field("remote", &self.conn.remote_address())
            .field("closed", &self.conn.close_reason().is_some())
            .finish()
    }
}

impl Conn {
    /// The underlying quinn connection.
    #[must_use]
    pub const fn connection(&self) -> &Connection {
        &self.conn
    }

    /// The peer's address.
    #[must_use]
    pub fn remote_address(&self) -> SocketAddr {
        self.conn.remote_address()
    }

    /// Send `message` on the path its class dictates (see
    /// [`crate::framed::send_message`]).
    pub async fn send(&self, message: &Message) -> io::Result<()> {
        send_message(&self.conn, message).await
    }

    /// Send `message` on a fresh reliable, ordered stream regardless of class.
    pub async fn send_reliable(&self, message: &Message) -> io::Result<()> {
        let frame = encode(message)?;
        send_on_stream(&self.conn, &frame).await
    }

    /// Send `message` on the **control** stream, which is reliable and ordered
    /// and shared with the peer's side of the handshake. Used for point-to-point
    /// control traffic such as [`space_protocol::Redirect`].
    pub async fn send_control(&self, message: &Message) -> io::Result<()> {
        let frame = encode(message)?;
        let mut stream = self.control_send.lock().await;
        stream
            .write_all(&frame)
            .await
            .map_err(|e| io::Error::other(e.to_string()))
    }

    /// Read the next frame from the control stream.
    ///
    /// `Ok(None)` when the stream ended cleanly or the connection closed.
    pub async fn recv_control(&self) -> Result<Option<Message>, io::Error> {
        let mut stream = self.control_recv.lock().await;
        read_frame(&mut stream).await
    }

    /// Read the next datagram frame (`Ok(None)` = peer gone).
    pub async fn recv_datagram(&self) -> Result<Option<Message>, io::Error> {
        recv_datagram(&self.conn).await
    }

    /// Accept the next incoming reliable stream and read its one frame.
    ///
    /// `Ok(None)` when the connection has closed.
    pub async fn recv_stream(&self) -> Result<Option<Message>, io::Error> {
        recv_stream_message(&self.conn).await
    }

    /// Refuse the connection: tell the peer why on the control stream, wait for
    /// it to read that, then close.
    ///
    /// This is what a server calls when its own policy turns a hello away after
    /// the version gate passed (a full or draining shard). Simply writing a
    /// [`Reject`] and dropping the connection does **not** work: dropping the
    /// last handle closes the connection at once, the `CONNECTION_CLOSE` beats
    /// the stream data, and the peer sees "connection lost" instead of the
    /// reason. So the reject is written and finished, and the close waits (up to
    /// a second) for the peer to close first — [`connect`] does exactly that as
    /// soon as it has read a `Reject`.
    pub async fn reject(&self, reason: RejectReason) {
        {
            let mut send = self.control_send.lock().await;
            send_reject(&mut send, &reason).await;
        }
        drain_then_close(&self.conn, b"rejected").await;
    }

    /// Whether the connection is already gone (locally closed, timed out, or the
    /// peer closed it). This is the disconnect signal a poll loop checks.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.conn.close_reason().is_some()
    }

    /// The close reason, if the connection has closed.
    #[must_use]
    pub fn close_reason(&self) -> Option<quinn::ConnectionError> {
        self.conn.close_reason()
    }

    /// Close the connection with an application error code and reason.
    pub fn close(&self, error_code: u32, reason: &str) {
        self.conn
            .close(VarInt::from_u32(error_code), reason.as_bytes());
    }
}

/// Encode `message` into a frame, mapping an oversized payload to
/// [`io::ErrorKind::InvalidInput`] (rejected before anything is sent).
fn encode(message: &Message) -> io::Result<Vec<u8>> {
    space_protocol::try_encode_frame(message)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))
}

/// Write a [`Reject`] carrying `reason` on `send` and finish the stream.
///
/// Best-effort: the peer may already be gone.
async fn send_reject(send: &mut SendStream, reason: &RejectReason) {
    let frame = space_protocol::encode_frame(&Message::Reject(Reject {
        reason: reason.clone(),
    }));
    let _ = send.write_all(&frame).await;
    let _ = send.finish();
}

/// Close `conn` after giving the peer a moment to read what was just written.
///
/// Closing right after a write would race it: a `CONNECTION_CLOSE` can reach
/// the peer before the stream data does, and the peer then sees "connection
/// lost" instead of the message. A client that reads a reject closes the
/// connection itself (see [`connect`]), so this normally returns immediately;
/// [`REJECT_DRAIN`] bounds it for a client that never reads.
async fn drain_then_close(conn: &Connection, why: &[u8]) {
    let _ = tokio::time::timeout(REJECT_DRAIN, conn.closed()).await;
    conn.close(VarInt::from_u32(REJECT_CLOSE_CODE), why);
}

/// Connect to `addr`, complete the TLS handshake and run the hello exchange.
///
/// On success the returned [`Welcome`] is the server's answer and the connection
/// is ready for [`Message::InputTick`] traffic. The whole exchange is bounded by
/// [`HANDSHAKE_TIMEOUT`]; see [`connect_within`] for a caller-chosen bound.
///
/// # Errors
///
/// * [`NetError::Handshake`] when QUIC/TLS cannot be established, or the server
///   has not answered the hello within [`HANDSHAKE_TIMEOUT`].
/// * [`NetError::VersionMismatch`] when the server answers with a
///   [`Reject`] carrying [`RejectReason::VersionMismatch`] (the server-side
///   check is [`check_hello`]; this is the client's view of it).
/// * [`NetError::Rejected`] for any other refusal.
/// * [`NetError::Protocol`] when the first frame is not a `Welcome`/`Reject`.
pub async fn connect(
    endpoint: &Endpoint,
    addr: SocketAddr,
    server_name: &str,
    hello: &Hello,
) -> Result<(Conn, Welcome), NetError> {
    connect_within(endpoint, addr, server_name, hello, HANDSHAKE_TIMEOUT).await
}

/// [`connect`] with a caller-chosen bound on the whole handshake.
///
/// Giving up drops the half-open connection, which closes it.
///
/// # Errors
///
/// As [`connect`]; running out of `limit` is a [`NetError::Handshake`].
pub async fn connect_within(
    endpoint: &Endpoint,
    addr: SocketAddr,
    server_name: &str,
    hello: &Hello,
    limit: Duration,
) -> Result<(Conn, Welcome), NetError> {
    let handshake = connect_unbounded(endpoint, addr, server_name, hello);
    match tokio::time::timeout(limit, handshake).await {
        Ok(result) => result,
        Err(_) => Err(NetError::Handshake(format!(
            "no welcome from {addr} within {limit:?}"
        ))),
    }
}

/// The handshake itself, with no bound of its own (see [`connect_within`]).
async fn connect_unbounded(
    endpoint: &Endpoint,
    addr: SocketAddr,
    server_name: &str,
    hello: &Hello,
) -> Result<(Conn, Welcome), NetError> {
    let connecting = endpoint
        .connect(addr, server_name)
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    let conn = connecting
        .await
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    // The control stream is the first bi stream the client opens; the server
    // reads the hello from it and answers on its own control stream.
    let (mut send, mut recv) = conn
        .open_bi()
        .await
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    let frame = space_protocol::encode_frame(&Message::Hello(hello.clone()));
    send.write_all(&frame)
        .await
        .map_err(|e| NetError::Io(e.to_string()))?;
    let reply = read_frame(&mut recv)
        .await?
        .ok_or_else(|| NetError::Protocol("the server closed the control stream".into()))?;
    match reply {
        Message::Welcome(welcome) => Ok((
            Conn {
                conn,
                control_send: Arc::new(Mutex::new(send)),
                control_recv: Arc::new(Mutex::new(recv)),
            },
            welcome,
        )),
        Message::Reject(reject) => {
            // Tell the server it has been read, so it can tear the connection
            // down immediately instead of waiting for its drain timeout.
            conn.close(VarInt::from_u32(REJECT_CLOSE_CODE), b"rejected by client");
            Err(reject_to_err(reject.reason))
        }
        other => Err(NetError::Protocol(format!(
            "expected Welcome or Reject, got {other:?}"
        ))),
    }
}

/// Accept one connection and run the hello exchange as the **server**.
///
/// The first frame on the first incoming bi stream must be a [`Hello`]; it is
/// checked with [`check_hello`]. A mismatch is answered with a [`Reject`] (so
/// the client learns *why*) and the connection is closed, and reported as
/// [`NetError::VersionMismatch`] / [`NetError::Rejected`]. On success the caller
/// answers with [`Conn::send_control`] and a [`Welcome`].
///
/// # Errors
///
/// [`NetError::Handshake`] on a failed QUIC/TLS handshake,
/// [`NetError::VersionMismatch`] / [`NetError::Rejected`] on a refused hello,
/// [`NetError::Protocol`] when the first frame is not a hello at all.
pub async fn accept_connection(accept: quinn::Accept<'_>) -> Result<(Conn, Hello), NetError> {
    let incoming = accept
        .await
        .ok_or_else(|| NetError::Handshake("the endpoint was closed".into()))?;
    accept_incoming(incoming).await
}

/// [`accept_connection`] for an already-received [`quinn::Incoming`].
///
/// The handshake and the wait for the client's hello are bounded by
/// [`HANDSHAKE_TIMEOUT`]; see [`accept_incoming_within`] for a caller-chosen
/// bound.
///
/// # Errors
///
/// As [`accept_connection`].
pub async fn accept_incoming(incoming: quinn::Incoming) -> Result<(Conn, Hello), NetError> {
    accept_incoming_within(incoming, HANDSHAKE_TIMEOUT).await
}

/// The transport half of an accept: the QUIC/TLS handshake, the control stream
/// and the first frame on it.
async fn accept_first_frame(
    incoming: quinn::Incoming,
) -> Result<(Connection, SendStream, RecvStream, Message), NetError> {
    let conn = incoming
        .await
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    // The control stream is the bi stream the client opened: the server reads the
    // hello from its receive half and answers on its send half, so both sides
    // share one reliable, ordered control channel.
    let (send, mut recv) = conn
        .accept_bi()
        .await
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    let first = read_frame(&mut recv)
        .await?
        .ok_or_else(|| NetError::Protocol("the client sent no hello".into()))?;
    Ok((conn, send, recv, first))
}

/// [`accept_incoming`] with a caller-chosen bound on the handshake and on the
/// wait for the client's hello.
///
/// A client that completes the QUIC handshake and then never opens the control
/// stream (or never finishes its hello) would otherwise hold this task for as
/// long as it keeps the connection alive. Giving up drops the connection, which
/// closes it.
///
/// # Errors
///
/// As [`accept_connection`]; running out of `limit` is a [`NetError::Handshake`].
pub async fn accept_incoming_within(
    incoming: quinn::Incoming,
    limit: Duration,
) -> Result<(Conn, Hello), NetError> {
    let (conn, mut send, recv, first) =
        match tokio::time::timeout(limit, accept_first_frame(incoming)).await {
            Ok(result) => result?,
            Err(_) => {
                return Err(NetError::Handshake(format!(
                    "the peer sent no hello within {limit:?}"
                )))
            }
        };
    let hello = match first {
        Message::Hello(h) => h,
        other => {
            return Err(NetError::Protocol(format!(
                "the first message must be a Hello, got {other:?}"
            )))
        }
    };
    // The version gate. Answer the mismatch *before* returning so the client
    // gets a reason instead of a timeout.
    if let Err(reason) = check_hello(&hello) {
        send_reject(&mut send, &reason).await;
        // Closing the connection right here would race the reject frame (see
        // [`drain_then_close`]).
        drain_then_close(&conn, b"protocol version mismatch").await;
        return Err(reject_to_err(reason));
    }
    Ok((
        Conn {
            conn,
            control_send: Arc::new(Mutex::new(send)),
            control_recv: Arc::new(Mutex::new(recv)),
        },
        hello,
    ))
}

fn reject_to_err(reason: RejectReason) -> NetError {
    match reason {
        RejectReason::VersionMismatch { seen, expected } => {
            NetError::VersionMismatch { seen, expected }
        }
        other => NetError::Rejected(other),
    }
}

/// Build a [`Hello`] for this build's protocol version.
#[must_use]
pub fn hello(player_token: impl Into<String>) -> Hello {
    Hello {
        protocol: PROTOCOL_VERSION,
        player_token: player_token.into(),
    }
}
