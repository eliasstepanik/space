//! `space-stream` — Phase 6 pixel streaming.
//!
//! A native client renders a frame offscreen, JPEG-encodes it ([`jpeg`]) and
//! pushes it into a [`StreamServer`] ([`server`]); the server broadcasts it over
//! `ws://` to thin browser viewers (the `space-viewer` wasm crate) and hands the
//! viewers' input back to the native client ([`StreamClient::poll_input`]).
//!
//! # Wire format ([`protocol`])
//!
//! * server → viewer: one binary WebSocket message per frame,
//!   `[seq: u64 BE][flags: u8][JPEG bytes…]`.
//! * viewer → server: one small binary message per input event
//!   ([`protocol::ViewerInput`]): pointer position, key transitions,
//!   *relative* view rotation (`Turn`) and typed characters (`Text`, plus the
//!   text-entry keys such as Enter and Backspace). The proxy owns the absolute
//!   view direction; a viewer never sends one. It also owns the meaning of a
//!   key press: the viewer cannot see whether the proxy shows a text field, so
//!   it sends a printable key both as a movement key and as text.
//!
//! # Multiplayer seam
//!
//! A browser viewer is **not** a new kind of peer for the authoritative game
//! server and this crate adds **no server protocol**. The viewer only talks to
//! a *headless native proxy client* — `space-client --stream ADDR`. That proxy
//! is the real player: it connects to the authoritative server through the
//! ordinary `space-net` client path, exactly like a windowed native client, and
//! is the only thing the server ever sees. Viewer input arrives at the proxy via
//! [`StreamClient::poll_input`] and is injected into the proxy's own input state
//! (the same state its local input would drive), so from there it travels to the
//! server over the existing netcode as that player's ordinary input.
//!
//! # Transport security
//!
//! Plain `ws://` only. `wss://` would need a TLS stack, i.e. `rustls` + `ring`
//! (the sanctioned C/asm exception, see the workspace manifest); it is deferred.
//! No TLS feature of `tokio-tungstenite` is enabled. Terminate TLS in front
//! (reverse proxy) or bind to loopback until it lands.
//!
//! The [`protocol`] module has no dependencies; everything else is behind the
//! default `net` feature so the wasm viewer can share the wire format without
//! pulling in tokio. The JPEG encoder alone (no tokio, no WebSocket) is the
//! `jpeg` feature, which `net` includes.
#![forbid(unsafe_code)]

pub mod protocol;

#[cfg(feature = "jpeg")]
pub mod jpeg;
#[cfg(feature = "net")]
pub mod server;

#[cfg(feature = "jpeg")]
pub use jpeg::{jpeg_dimensions, EncodeScratch, EncodedJpeg, JpegEncoder, JpegError};
pub use protocol::{FrameHeader, Key, ProtocolError, ViewerInput};
#[cfg(feature = "net")]
pub use server::{StreamClient, StreamServer, MAX_VIEWERS};
