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
use std::net::SocketAddr;
use std::sync::Arc;

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
const REJECT_DRAIN: std::time::Duration = std::time::Duration::from_secs(1);

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
/// * **no keep-alive**: disconnects are detected by the idle timeout, which
///   [`Conn::is_closed`] reports.
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
    t.keep_alive_interval(None);
    t.max_idle_timeout(Some(
        quinn::IdleTimeout::try_from(std::time::Duration::from_secs(10))
            .expect("10 s fits in a QUIC idle timeout"),
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

/// A client endpoint bound to an ephemeral port on the wildcard address.
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

/// Connect to `addr`, complete the TLS handshake and run the hello exchange.
///
/// On success the returned [`Welcome`] is the server's answer and the connection
/// is ready for [`Message::InputTick`] traffic.
///
/// # Errors
///
/// * [`NetError::Handshake`] when QUIC/TLS cannot be established.
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
/// # Errors
///
/// As [`accept_connection`].
pub async fn accept_incoming(incoming: quinn::Incoming) -> Result<(Conn, Hello), NetError> {
    let conn = incoming
        .await
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    // The control stream is the bi stream the client opened: the server reads the
    // hello from its receive half and answers on its send half, so both sides
    // share one reliable, ordered control channel.
    let (mut send, mut recv) = conn
        .accept_bi()
        .await
        .map_err(|e| NetError::Handshake(e.to_string()))?;
    let first = read_frame(&mut recv)
        .await?
        .ok_or_else(|| NetError::Protocol("the client sent no hello".into()))?;
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
        let frame = space_protocol::encode_frame(&Message::Reject(Reject {
            reason: reason.clone(),
        }));
        // Best-effort: the client may already be gone.
        let _ = send.write_all(&frame).await;
        let _ = send.finish();
        // Closing the connection right here would race the reject frame: a
        // CONNECTION_CLOSE can reach the peer before the stream data does, and
        // the peer then sees "connection lost" instead of the reason. A client
        // that reads the reject closes the connection itself (see [`connect`]),
        // so this normally returns immediately; the timeout bounds it for a
        // client that never reads.
        let _ = tokio::time::timeout(REJECT_DRAIN, conn.closed()).await;
        conn.close(
            VarInt::from_u32(REJECT_CLOSE_CODE),
            b"protocol version mismatch",
        );
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
