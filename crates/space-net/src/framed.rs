//! Message classes and the two framing paths.
//!
//! `docs/plan.md` P9 splits the traffic by class:
//!
//! | class | messages | why |
//! |-------|----------|-----|
//! | **datagram** (unreliable, unordered) | [`Message::InputTick`], [`Message::NeighbourStatus`], small [`Message::Delta`]s | small, idempotent, gap-detectable — a lost input is a coasting tick the client reconciles, not a hole in the state |
//! | **stream** (reliable, ordered) | [`Message::Snapshot`], large [`Message::Delta`]s, the handoff/redirect/handshake messages | a full state dump needs ordering and flow control: 1 000 entities × ~158 B × 30 Hz = 4.7 MB/s, far past what datagrams carry |
//!
//! The datagram budget is **derived**, not hard-coded:
//! [`Connection::max_datagram_size`](quinn::Connection::max_datagram_size)
//! reports the largest payload that fits one packet after QUIC/TLS overhead,
//! and a frame larger than that is promoted to the stream path automatically. A
//! `Delta` that grew past one packet therefore becomes a stream frame without
//! the caller deciding.
//!
//! Bulk messages ride a **fresh one-way stream each** ([`quinn::Connection::open_uni`]):
//! ordering within one message is what a state dump needs, ordering *between*
//! messages is not (each carries its own tick), and one stream per message means
//! a slow snapshot can never head-of-line block a later one.

use std::io;

use quinn::{Connection, RecvStream};
use space_protocol::codec::{FRAME_HEADER_LEN, MAX_FRAME_SIZE};
use space_protocol::Message;

/// The largest payload the **first** frame of a connection — a
/// [`space_protocol::Hello`] — may declare, in bytes.
///
/// A hello is a tag, a version and one token: about 60 bytes from a player
/// (`space-client-<unique>`), and `shard:<id>:<peer token>` between shards —
/// under 100 bytes with a generated peer token, and 1 051 bytes at most with
/// a configured one (the orchestrator refuses a peer token over
/// [`MAX_SECRET_LEN`] characters at startup). Four times that is headroom,
/// not an invitation: the general frame limit is 1 MiB, and before this cap a
/// stranger's first eight bytes made the receiver allocate all of it.
pub const MAX_HELLO_PAYLOAD: usize = 4 * 1024;

/// The longest shared secret (a cluster's peer token) that is certain to fit
/// a hello ([`MAX_HELLO_PAYLOAD`]) with room to spare, in bytes.
pub const MAX_SECRET_LEN: usize = 1024;

/// Which transport path a message uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageClass {
    /// Unreliable, unordered datagram.
    Datagram,
    /// Reliable, ordered stream.
    Stream,
}

impl MessageClass {
    /// The class of `message` by type, ignoring size.
    ///
    /// A [`Message::Delta`] is classified `Datagram` because it is *usually*
    /// small; [`crate::send_message`] promotes the ones that do not fit one
    /// packet to [`MessageClass::Stream`].
    #[must_use]
    pub const fn of(message: &Message) -> Self {
        match message {
            // Player inputs and the tiny per-connection status ping.
            Message::InputTick(_) | Message::NeighbourStatus(_) | Message::Delta(_) => {
                MessageClass::Datagram
            }
            // Everything else must arrive: state dumps, the handoff protocol,
            // the handshake, redirects and ghost replicas.
            _ => MessageClass::Stream,
        }
    }
}

/// Largest frame **payload** that can ride one datagram on `conn`.
///
/// `max_datagram_size` is the limit for one datagram's whole payload, and the
/// frame header is part of it, so [`FRAME_HEADER_LEN`] is subtracted here.
/// `None` when the peer does not support datagrams at all (quinn reports `None`
/// or 0), which forces every message onto the stream path.
#[must_use]
pub fn datagram_frame_budget(conn: &Connection) -> Option<usize> {
    conn.max_datagram_size()
        .and_then(|n| n.checked_sub(FRAME_HEADER_LEN))
        .filter(|n| *n > 0)
}

/// Send `message` on the path its class dictates.
///
/// * A datagram-class message whose encoded frame fits [`datagram_frame_budget`]
///   goes out as one datagram.
/// * Anything else — a snapshot, a delta that outgrew one packet, a handoff
///   message — is written to a fresh reliable, ordered stream.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] for a payload over
/// [`MAX_FRAME_SIZE`](space_protocol::MAX_FRAME_SIZE) (rejected before anything
/// is sent) or a datagram too large for the path; other I/O errors when the
/// connection is gone.
pub async fn send_message(conn: &Connection, message: &Message) -> io::Result<()> {
    let frame = space_protocol::try_encode_frame(message)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
    let payload_len = frame.len() - FRAME_HEADER_LEN;
    let fits_datagram = datagram_frame_budget(conn).is_some_and(|budget| payload_len <= budget);
    match MessageClass::of(message) {
        // The frame buffer is handed to quinn as it is: `Bytes::from(Vec)` takes
        // the allocation over, where `send_datagram(&frame)` would copy it.
        MessageClass::Datagram if fits_datagram => send_datagram_owned(conn, frame.into()),
        // A datagram-class message that outgrew one packet is promoted here.
        _ => send_on_stream(conn, &frame).await,
    }
}

/// Send one already-encoded frame as one datagram.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidInput`] when the frame is larger than one datagram
/// (the caller should have used [`send_on_stream`]); [`io::ErrorKind::BrokenPipe`]
/// when the connection or datagram support is gone.
pub fn send_datagram(conn: &Connection, frame: &[u8]) -> io::Result<()> {
    send_datagram_owned(conn, bytes::Bytes::copy_from_slice(frame))
}

/// [`send_datagram`] for a frame the caller already owns (no copy).
fn send_datagram_owned(conn: &Connection, frame: bytes::Bytes) -> io::Result<()> {
    let len = frame.len();
    conn.send_datagram(frame).map_err(|e| match e {
        quinn::SendDatagramError::TooLarge => io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("datagram frame of {len} bytes exceeds the datagram size"),
        ),
        quinn::SendDatagramError::ConnectionLost(_)
        | quinn::SendDatagramError::UnsupportedByPeer
        | quinn::SendDatagramError::Disabled => {
            io::Error::new(io::ErrorKind::BrokenPipe, e.to_string())
        }
    })
}

/// Write one already-encoded frame to a fresh reliable, ordered stream and
/// finish it, so the reader sees a clean end-of-stream.
///
/// # Errors
///
/// Any I/O error from the stream (connection gone, stream reset).
pub async fn send_on_stream(conn: &Connection, frame: &[u8]) -> io::Result<()> {
    let mut stream = conn.open_uni().await.map_err(io::Error::other)?;
    stream
        .write_all(frame)
        .await
        .map_err(|e| io::Error::other(e.to_string()))?;
    stream
        .finish()
        .map_err(|e| io::Error::other(e.to_string()))?;
    Ok(())
}

/// Receive one datagram frame and decode it.
///
/// Returns `Ok(None)` when the connection has closed — the disconnect signal a
/// caller must treat as "peer gone". A malformed frame is an error: an oversized
/// length is rejected by [`space_protocol::decode_frame`] before the payload is
/// trusted, and QUIC datagrams are message-delimited, so a datagram is never
/// silently truncated by the transport.
///
/// # Errors
///
/// [`space_protocol::CodecError`] (wrapped in an [`io::Error`]) on a malformed
/// frame; I/O errors on transport failure.
pub async fn recv_datagram(conn: &Connection) -> Result<Option<Message>, io::Error> {
    match conn.read_datagram().await {
        Ok(bytes) => space_protocol::decode_frame(&bytes)
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string())),
        Err(
            quinn::ConnectionError::ApplicationClosed(_)
            | quinn::ConnectionError::ConnectionClosed(_)
            | quinn::ConnectionError::TimedOut
            | quinn::ConnectionError::LocallyClosed,
        ) => Ok(None),
        Err(e) => Err(io::Error::other(e.to_string())),
    }
}

/// Accept the next incoming one-way stream and read its one frame.
///
/// `Ok(None)` when the connection has closed.
///
/// # Errors
///
/// As [`read_frame`].
pub async fn recv_stream_message(conn: &Connection) -> Result<Option<Message>, io::Error> {
    let mut stream = match conn.accept_uni().await {
        Ok(s) => s,
        Err(
            quinn::ConnectionError::ApplicationClosed(_)
            | quinn::ConnectionError::ConnectionClosed(_)
            | quinn::ConnectionError::TimedOut
            | quinn::ConnectionError::LocallyClosed,
        ) => return Ok(None),
        Err(e) => return Err(io::Error::other(e.to_string())),
    };
    read_frame(&mut stream).await
}

/// Read exactly one frame from a reliable stream.
///
/// The 8-byte header is read first, so a declared payload length above
/// [`MAX_FRAME_SIZE`] is rejected **before** the buffer is allocated — a hostile
/// or corrupt stream cannot make this function allocate gigabytes. A length that
/// is plausible but never arrives is a truncated frame
/// ([`space_protocol::CodecError::Truncated`]), and the header's version is
/// re-checked by [`space_protocol::decode_frame`].
///
/// `Ok(None)` only when the stream ended cleanly *before* a header (an empty
/// stream): not an error, but no message either.
///
/// # Errors
///
/// [`space_protocol::CodecError`] (wrapped in an [`io::Error`]) on an oversized,
/// truncated or otherwise invalid frame; I/O errors on transport failure.
pub async fn read_frame(stream: &mut RecvStream) -> Result<Option<Message>, io::Error> {
    read_frame_capped(stream, MAX_FRAME_SIZE).await
}

/// [`read_frame`] for a frame whose payload may be at most `max_payload`
/// bytes (never more than [`MAX_FRAME_SIZE`]): a larger declared length is
/// refused from the header alone, before anything is allocated for it.
///
/// This is what reads the **first** frame of a connection
/// ([`MAX_HELLO_PAYLOAD`]): whoever sends it has proved nothing yet, and a
/// megabyte per five header bytes is not something to hand a stranger.
///
/// # Errors
///
/// As [`read_frame`].
pub async fn read_frame_capped(
    stream: &mut RecvStream,
    max_payload: usize,
) -> Result<Option<Message>, io::Error> {
    let mut header = [0u8; FRAME_HEADER_LEN];
    if !read_exact_opt(stream, &mut header).await? {
        return Ok(None);
    }
    let declared = u32::from_be_bytes([header[0], header[1], header[2], header[3]]) as usize;
    if declared > max_payload.min(MAX_FRAME_SIZE) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            space_protocol::CodecError::Oversized { declared }.to_string(),
        ));
    }
    // The payload is read straight into the frame buffer, which is exactly as
    // large as the (already bounded) declared length. An earlier revision read
    // through a 64 KiB scratch buffer that was allocated and zeroed for every
    // frame, however small, and then copied out of.
    let mut frame = vec![0u8; FRAME_HEADER_LEN + declared];
    frame[..FRAME_HEADER_LEN].copy_from_slice(&header);
    let mut filled = FRAME_HEADER_LEN;
    while filled < frame.len() {
        let n = stream
            .read(&mut frame[filled..])
            .await
            .map_err(|e| io::Error::other(e.to_string()))?
            .unwrap_or(0);
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                space_protocol::CodecError::Truncated.to_string(),
            ));
        }
        filled += n;
    }
    // The codec error is kept as the error's source (its text is unchanged),
    // so the handshake can tell "this peer speaks another protocol version"
    // from every other undecodable frame ([`frame_version_mismatch`]).
    space_protocol::decode_frame(&frame)
        .map(Some)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// If `err` is [`read_frame`] refusing a frame because its **header** names
/// another protocol version, the two versions: `(the frame's, ours)`.
///
/// Every frame carries the version in its header, so a peer built for another
/// version cannot get a single frame through — not even the hello, and not
/// even the `Reject` that would tell it why. The handshake uses this to turn
/// that into what it is: a version mismatch, on both ends.
pub(crate) fn frame_version_mismatch(err: &io::Error) -> Option<(u32, u32)> {
    match err
        .get_ref()?
        .downcast_ref::<space_protocol::CodecError>()?
    {
        space_protocol::CodecError::VersionMismatch { seen, expected } => Some((*seen, *expected)),
        _ => None,
    }
}

/// `read_exact` that reports a clean end-of-stream *at a boundary* as `false`
/// instead of an error, so an empty stream is not an error.
async fn read_exact_opt(stream: &mut RecvStream, buf: &mut [u8]) -> Result<bool, io::Error> {
    let mut filled = 0;
    while filled < buf.len() {
        let n = stream
            .read(&mut buf[filled..])
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;
        match n {
            Some(n) if n > 0 => filled += n,
            // A clean end before any byte of the header: an empty stream.
            _ if filled == 0 => return Ok(false),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    space_protocol::CodecError::Truncated.to_string(),
                ))
            }
        }
    }
    Ok(true)
}
