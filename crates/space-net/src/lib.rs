//! QUIC transport for `space` (Phase 2, task 2.1).
//!
//! One thin library over [`quinn`] that owns everything network-shaped, so the
//! crates above it ([`space_server`](https://example.invalid) and the client)
//! only ever see [`space_protocol::Message`]s:
//!
//! * **Endpoints** — [`server_endpoint`] and [`client_endpoint`] build a QUIC
//!   endpoint with rustls TLS ([`rustls`] + [`rcgen`] self-signed certificate; no
//!   C bindings this repository maintains — see [`crypto`]'s module docs for the
//!   exact statement, per `docs/plan.md`).
//! * **Handshake** — [`accept_connection`] / [`connect`] perform the
//!   [`space_protocol::Hello`] exchange and enforce
//!   [`PROTOCOL_VERSION`](space_protocol::PROTOCOL_VERSION) through
//!   [`space_protocol::check_hello`]. A version mismatch is answered with a
//!   [`space_protocol::Reject`] carrying
//!   [`RejectReason::VersionMismatch`](space_protocol::RejectReason::VersionMismatch)
//!   and surfaced as [`NetError::VersionMismatch`], so the client learns *why*
//!   it was turned away.
//! * **Message classes** — [`send_message`] routes by message class, exactly as
//!   `docs/plan.md` P9 requires: player inputs and small deltas ride unreliable
//!   **datagrams** (small, idempotent, gap-detectable), snapshots and large
//!   deltas ride a **reliable, ordered stream** (a 1k-entity interest set is
//!   ~158 KB at ~158 B/entity — 4.7 MB/s at 30 Hz, far past datagram capacity).
//!   The datagram threshold is derived from
//!   [`Connection::max_datagram_size`](quinn::Connection::max_datagram_size),
//!   never hard-coded.
//! * **Lifecycle** — [`Conn`] wraps a quinn connection with per-connection state
//!   and detects disconnects: every receive path returns `Ok(None)` when the
//!   peer is gone, and [`Conn::is_closed`] polls the connection error.
//!
//! # Framing
//!
//! Every payload — datagram or stream — is exactly one space-protocol frame
//! (`len:u32 | protocol:u32 | payload`, see [`space_protocol::codec`]). On a
//! datagram the frame *is* the packet, so a truncated or oversized frame is
//! rejected by [`space_protocol::decode_frame`] before anything is trusted. On a
//! stream, [`read_frame`] reads the 8-byte header first, rejects a declared
//! length above [`space_protocol::MAX_FRAME_SIZE`] **before** allocating, then
//! reads exactly that many payload bytes and re-checks the version.
//!
//! ```text
//!   datagram:  [ frame(Message::InputTick) ][ frame(Message::Delta) ] …
//!   stream:    [ frame(Message::Snapshot) ][ frame(Message::Snapshot) ] …
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod crypto;
pub mod endpoint;
pub mod framed;

pub use crypto::{make_server_config, make_tls_client_config, make_tls_server_config, CertPair};
pub use endpoint::{
    accept_connection, accept_incoming, client_endpoint, client_endpoint_on, connect, hello,
    server_endpoint, transport_config, Conn, NetError,
};
pub use framed::{
    read_frame, recv_datagram, send_datagram, send_message, send_on_stream, MessageClass,
};

/// The ALPN protocol name both sides advertise.
///
/// QUIC negotiates it during the TLS handshake, so a peer that does not speak
/// `space` is rejected before any application frame is exchanged.
pub const ALPN: &[u8] = b"space0";
