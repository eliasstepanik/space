//! `space-client` library: the client application without its window.
//!
//! The client is always online — it shows a connect screen until it is playing
//! on a shard — and it has three front ends over one core: a window, a
//! headless offscreen loop and the pixel stream for browser viewers. The core
//! lives here so the front ends, the integration tests and other programs
//! (the MCP client server) share it:
//!
//! * [`app`] — [`app::App`]: the state machine (connect screen, connecting,
//!   playing), the one input path, the scene of each frame and the harness
//!   plumbing ([`space_harness::Hub`]);
//! * [`headless`] — the offscreen render loop (`--headless`, `--stream`);
//! * [`connect`], [`address`] — the address field and what can be typed into it;
//! * [`input`] — what an input event means on which screen;
//! * [`scenes`], [`demo_area`] — what is drawn for each scene a shard can show;
//! * [`kestrel`] — the ship model of `assets/ships/kestrel/`, embedded, in two
//!   levels of detail, and how a scene parks it;
//! * [`ui`] — the connect screen and the heads-up display (2D overlay);
//! * [`link`] — the player connection to the authoritative shard;
//! * [`bodies`], [`meshes`] — the physics bodies a shard replicates, drawn
//!   smoothly from their collision shapes, with their springs and joints;
//! * [`predict`] — the own ship run ahead of the shard through the same
//!   physics, and the clock everything is drawn by.
//!
//! The window front end and the command line stay in the binary.

#![warn(missing_docs)]

pub mod address;
pub mod app;
pub mod bodies;
pub mod connect;
pub mod demo_area;
pub mod headless;
pub mod input;
pub mod kestrel;
pub mod link;
pub mod meshes;
pub mod predict;
pub mod scenes;
pub mod ui;
