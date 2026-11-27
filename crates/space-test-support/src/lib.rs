//! Helpers shared by the workspace's tests: a seeded PRNG, temp paths, a loopback address.
//!
//! Four rules keep this crate harmless:
//!
//! 1. It depends on no workspace crate, so it can never be part of a
//!    dependency cycle.
//! 2. Other crates list it under `[dev-dependencies]` only.
//! 3. It has no cargo features.
//! 4. A helper that needs a workspace type does not belong here but in that
//!    crate's `tests/common/mod.rs`.
//!
//! `tests/boundaries.rs` checks rules 1 and 2 against the manifests on disk.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod fs;
pub mod net;
pub mod rng;

pub use fs::temp_path;
pub use net::loopback0;
pub use rng::Rng;
