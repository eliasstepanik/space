//! Frame codec: `len:u32 | protocol:u32 | payload`.
//!
//! The payload is the [`Message`] encoded with [`bincode`] 2 through its serde
//! bridge, under a pinned `Config`:
//!
//! * **big endian** — network byte order, so a hex dump of a frame reads the way
//!   a protocol spec says it should;
//! * **fixed-width integers** — every field has a constant size, which makes the
//!   byte budget in [`crate::SnapshotBuilder`] predictable and decoding branch
//!   free (varints buy little here: the ids that dominate a snapshot — guids,
//!   ticks, component payloads — are large or arbitrary anyway);
//! * **a [`MAX_FRAME_SIZE`] limit** — bincode checks every container length
//!   against it, so a forged `Vec` length cannot make the decoder allocate
//!   gigabytes before it runs out of input.
//!
//! The header is written by hand (four big-endian `u32`s) rather than through
//! bincode, so a version mismatch is detected *before* any payload is decoded
//! and the header layout can never drift with a dependency default.

use std::fmt;

use bincode::config::Configuration;
use serde::Serialize;

use crate::messages::Message;

/// Protocol version carried in every frame header and in [`crate::Hello`].
///
/// Bump it whenever the wire format changes incompatibly; a peer speaking a
/// different version is rejected with [`CodecError::VersionMismatch`] (frame
/// header) or [`crate::RejectReason::VersionMismatch`] (handshake).
///
/// **Version 2** (Phase-1 remediation): `InputTick` gained `ack_tick`,
/// `Snapshot`/`Delta` gained `input_ack`, `Delta` gained `removed`,
/// `HandoffCommit` gained the entity's final `state`, `Redirect` gained
/// `activation_tick`, `HandoffReason` gained `ClientNotConnected`, the
/// `NeighbourStatus` message was appended, and shard-map octree nodes are
/// validated on decode. None of these are backward-compatible with a v1 peer,
/// hence the major bump rather than a tolerant field.
///
/// **Version 3** (Phase 3, P3.1): `ShardHello`/`ShardWelcome` were appended
/// (wire tags 23/24) — the shard↔orchestrator "give me an id and my map"
/// exchange. It is a *new required* exchange for an orchestrated shard, so the
/// version bumps rather than smuggling the claim through `Hello.player_token`.
///
/// **Version 4** (server-owned scene): `SceneSet`/`SceneAck`/`SceneUpdate` were
/// appended (wire tags 25/26/27, see [`crate::scene`]). No existing layout
/// changed — by the append-only rule alone the version could have stayed — but
/// the new messages are sent *unasked* to every peer: the orchestrator sends a
/// `SceneSet` to every shard right behind its `ShardWelcome`, and every shard
/// sends a `SceneUpdate` to every client right behind its `Welcome`. A
/// version-3 peer can decode neither tag and would find that out in the middle
/// of an established session: a shard's control-link reader ends on the first
/// frame it cannot decode, so the shard would silently stop following epoch
/// changes; a client would report an unknown tag for every update and never
/// learn the scene. So the version bumps, exactly as it did for
/// `ShardHello`/`ShardWelcome`, and such a peer is turned away at the hello
/// instead, with the reason, before any session exists (`space-net` answers a
/// hello framed at another version with a `Reject` naming both).
///
/// **Version 5** (physics across shards): snapshots and deltas carry every
/// physics body in a client's interest, not only ships, with the physics
/// components (ids 4–10: `Body`, `Visual`, `Springs`, `Articulation`,
/// `PartStates`, `PosResidual`, see `space-sim`); a [`crate::Delta`] entry for
/// an entity the recipient already holds lists only the components that
/// changed (an empty payload removes one), which a v4 client would take for
/// the whole set; [`crate::GhostUpdate`] (wire tag 10, never sent before) is
/// now a per-tick batch of border ghosts between shards; and a handoff's state
/// may describe an unowned body. None of it is readable by a v4 peer, so the
/// version bumps and such a peer is refused at the hello.
pub const PROTOCOL_VERSION: u32 = 5;

/// Largest accepted frame payload, in bytes (1 MiB).
///
/// The length prefix is checked against this before the payload is decoded, and
/// the bincode decoder runs with the same limit.
pub const MAX_FRAME_SIZE: usize = 1024 * 1024;

/// Size of the frame header: `len:u32 + protocol:u32`.
pub const FRAME_HEADER_LEN: usize = 8;

/// The pinned bincode configuration (big endian, fixed-width integers, limited).
type Bincode = Configuration<
    bincode::config::BigEndian,
    bincode::config::Fixint,
    bincode::config::Limit<MAX_FRAME_SIZE>,
>;

/// The pinned config value (see the module docs for why these options).
const BINCODE: Bincode = bincode::config::standard()
    .with_big_endian()
    .with_fixed_int_encoding()
    .with_limit::<MAX_FRAME_SIZE>();

/// Why a frame could not be decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// The buffer is shorter than the frame it describes: either shorter than
    /// the 8-byte header, or shorter than `FRAME_HEADER_LEN + len`.
    Truncated,
    /// The declared payload length exceeds [`MAX_FRAME_SIZE`].
    Oversized {
        /// The length the frame declared.
        declared: usize,
    },
    /// The header's protocol version is not [`PROTOCOL_VERSION`].
    VersionMismatch {
        /// Version the peer announced.
        seen: u32,
        /// Version we speak.
        expected: u32,
    },
    /// The payload's message tag is not a known [`Message`] variant.
    UnknownTag {
        /// The variant index found in the payload.
        tag: u32,
    },
    /// The payload is structurally invalid (bad length, bad utf-8, trailing
    /// bytes, …) but its tag was recognised.
    InvalidPayload(String),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::Truncated => write!(f, "frame is truncated"),
            CodecError::Oversized { declared } => {
                write!(
                    f,
                    "frame declares {declared} bytes, limit is {MAX_FRAME_SIZE}"
                )
            }
            CodecError::VersionMismatch { seen, expected } => {
                write!(
                    f,
                    "protocol version {seen} is not supported (expected {expected})"
                )
            }
            CodecError::UnknownTag { tag } => write!(f, "unknown message tag {tag}"),
            CodecError::InvalidPayload(reason) => write!(f, "invalid payload: {reason}"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Encode a message into a complete frame (header + payload).
///
/// # Panics
///
/// Panics if the payload exceeds [`MAX_FRAME_SIZE`] — that is a programming
/// error, not network input. Use [`try_encode_frame`] when the payload size is
/// not under your control (it is not for a `Snapshot`, which is why
/// [`crate::SnapshotBuilder`] exists).
#[must_use]
pub fn encode_frame(message: &Message) -> Vec<u8> {
    try_encode_frame(message).expect("encode_frame: payload exceeds MAX_FRAME_SIZE")
}

/// Encode a message into a frame, reporting an oversized payload as
/// [`CodecError::Oversized`] instead of panicking.
pub fn try_encode_frame(message: &Message) -> Result<Vec<u8>, CodecError> {
    // The payload is encoded straight into the frame buffer, behind a header
    // whose length field is patched in afterwards: one buffer and one pass
    // instead of encoding into a scratch `Vec` and copying it behind the header.
    // A replication frame is sized up front so the buffer is allocated once.
    let mut frame = Vec::with_capacity(FRAME_HEADER_LEN + payload_size_hint(message));
    frame.extend_from_slice(&[0u8; 4]);
    frame.extend_from_slice(&PROTOCOL_VERSION.to_be_bytes());
    bincode::serde::encode_into_std_write(message, &mut frame, BINCODE)
        .map_err(|e| CodecError::InvalidPayload(e.to_string()))?;
    let payload_len = frame.len() - FRAME_HEADER_LEN;
    if payload_len > MAX_FRAME_SIZE {
        return Err(CodecError::Oversized {
            declared: payload_len,
        });
    }
    frame[..4].copy_from_slice(&(payload_len as u32).to_be_bytes());
    Ok(frame)
}

/// A capacity hint for `message`'s payload: exact for the two replication
/// frames (the only large messages on the hot path), a small constant otherwise.
///
/// Only a hint — the buffer grows if it is short — so a wrong value costs a
/// reallocation, never a wrong frame.
fn payload_size_hint(message: &Message) -> usize {
    use crate::messages::{EntitySnapshot, ENTITY_GUID_WIRE_LEN};
    let entities = |list: &[EntitySnapshot]| -> usize {
        list.iter().map(EntitySnapshot::encoded_len).sum::<usize>()
    };
    match message {
        Message::Snapshot(s) => crate::snapshot::MESSAGE_OVERHEAD + entities(&s.entities),
        Message::Delta(d) => {
            crate::snapshot::MESSAGE_OVERHEAD
                + entities(&d.entities)
                + d.removed.len() * ENTITY_GUID_WIRE_LEN
        }
        _ => 256,
    }
}

/// Decode one complete frame.
///
/// `frame` must be exactly one frame (header + payload); trailing bytes are
/// rejected so a stream framer cannot silently lose a message.
pub fn decode_frame(frame: &[u8]) -> Result<Message, CodecError> {
    if frame.len() < FRAME_HEADER_LEN {
        return Err(CodecError::Truncated);
    }
    let declared = u32::from_be_bytes([frame[0], frame[1], frame[2], frame[3]]) as usize;
    let seen = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]);
    if seen != PROTOCOL_VERSION {
        return Err(CodecError::VersionMismatch {
            seen,
            expected: PROTOCOL_VERSION,
        });
    }
    if declared > MAX_FRAME_SIZE {
        return Err(CodecError::Oversized { declared });
    }
    if frame.len() < FRAME_HEADER_LEN + declared {
        return Err(CodecError::Truncated);
    }
    if frame.len() > FRAME_HEADER_LEN + declared {
        return Err(CodecError::InvalidPayload(format!(
            "{} trailing bytes after the frame",
            frame.len() - FRAME_HEADER_LEN - declared
        )));
    }
    let payload = &frame[FRAME_HEADER_LEN..];
    // The tag is checked here rather than left to the decoder: bincode's serde
    // bridge reports an out-of-range variant index as a generic "invalid value"
    // error, and an unknown tag deserves its own error (and its own metric).
    if payload.len() < 4 {
        return Err(CodecError::Truncated);
    }
    let tag = u32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
    if tag >= Message::VARIANT_COUNT {
        return Err(CodecError::UnknownTag { tag });
    }
    let (message, read) =
        bincode::serde::decode_from_slice(payload, BINCODE).map_err(|e| classify(&e))?;
    if read != payload.len() {
        return Err(CodecError::InvalidPayload(format!(
            "{read} of {} payload bytes were consumed",
            payload.len()
        )));
    }
    Ok(message)
}

/// Map a bincode decode error onto [`CodecError`], pulling the offending tag
/// out of the "unexpected variant" error so callers can log it.
fn classify(err: &bincode::error::DecodeError) -> CodecError {
    use bincode::error::DecodeError;
    match err {
        DecodeError::UnexpectedEnd { .. } => CodecError::Truncated,
        DecodeError::LimitExceeded => CodecError::Oversized {
            declared: MAX_FRAME_SIZE,
        },
        DecodeError::UnexpectedVariant { found, .. } => CodecError::UnknownTag { tag: *found },
        other => CodecError::InvalidPayload(other.to_string()),
    }
}

/// Check a [`Hello`] handshake against [`PROTOCOL_VERSION`].
///
/// The frame header already carries the version and is checked by
/// [`decode_frame`] (a mismatch there is [`CodecError::VersionMismatch`]), but
/// the version *also* travels in [`Hello::protocol`] (see the crate docs: it
/// "travels twice"). This is the handshake-level check: a server that has decoded
/// a [`Hello`] calls it before accepting the connection and answers a mismatch
/// with [`Reject`] / [`RejectReason::VersionMismatch`], so the client learns
/// *why* it was turned away rather than just having its frame dropped.
///
/// [`Hello`]: crate::Hello
/// [`Reject`]: crate::Reject
/// [`RejectReason::VersionMismatch`]: crate::RejectReason::VersionMismatch
pub fn check_hello(h: &crate::Hello) -> Result<(), crate::RejectReason> {
    if h.protocol != PROTOCOL_VERSION {
        return Err(crate::RejectReason::VersionMismatch {
            seen: h.protocol,
            expected: PROTOCOL_VERSION,
        });
    }
    Ok(())
}

/// Encoded size of a value under the pinned config, measured by encoding it.
///
/// Test-only: the hot paths compute their sizes instead
/// ([`crate::EntitySnapshot::encoded_len`], `snapshot::MESSAGE_OVERHEAD`), and
/// the tests use this to pin those computed sizes to what the codec writes.
#[cfg(test)]
pub(crate) fn encoded_len<T: Serialize>(value: &T) -> usize {
    bincode::serde::encode_to_vec(value, BINCODE)
        .map(|v| v.len())
        .unwrap_or(usize::MAX)
}

/// Encode a value with the pinned config (used by tests that build raw
/// payloads, e.g. an unknown message tag, and by component codecs such as
/// `space_sim::wire::encode_ship_state`).
pub fn encode_raw<T: Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(value, BINCODE).expect("encode_raw: value too large")
}

/// Decode a value with the pinned config, returning it and the number of bytes
/// consumed.
///
/// The counterpart of [`encode_raw`]: a caller that stores a raw component
/// payload (e.g. a [`crate::EntitySnapshot`] component) decodes it back through
/// here, so the component and the frame agree on the byte layout.
///
/// # Errors
///
/// [`CodecError::Truncated`] when the buffer is shorter than the value, or
/// [`CodecError::InvalidPayload`] for any other structural problem.
pub fn decode_raw<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<(T, usize), CodecError> {
    bincode::serde::decode_from_slice(bytes, BINCODE).map_err(|e| classify(&e))
}
