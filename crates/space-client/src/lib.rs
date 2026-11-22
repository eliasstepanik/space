//! `space-client` library: the pieces other front ends (the MCP client server)
//! share with the native client binary.
//!
//! Only the multiplayer seam lives here; windowing, rendering setup and the
//! pixel-stream proxy stay in the binary.

#![warn(missing_docs)]

pub mod link;
