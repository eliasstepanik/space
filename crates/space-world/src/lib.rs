//! Deterministic world model for `space` (Phase 1, task 1.1).
//!
//! This crate is the foundation every later crate builds on. Its two defining
//! properties are **precision** and **determinism**:
//!
//! * Precision — world positions are [`WorldPos`] = `i64` cell index + `f32`
//!   offset inside the cell ([`CELL_SIZE`] = 1024 m cells), exactly the model
//!   fixed by `docs/plan.md` ("Position model: i64 cell ×3 + f32 local ×3,
//!   1024 m cells (P1)"). Differences between two positions are computed from
//!   the *cell delta* in `i128` metres plus a small `f32` local correction, so
//!   two points 1 m apart 50 000 light-years from the origin still differ by
//!   exactly 1 m (see the `position_precision` acceptance test).
//! * Determinism — every generator ([`Galaxy`], [`StarSystem`]) is driven only
//!   by integer hashing ([`splitmix64`]); there is no floating-point RNG state
//!   and no dependency on evaluation order, so the same seed yields bit-identical
//!   output on every platform.
//!
//! There is **no global state** anywhere in the crate: every conversion takes
//! its origin as an explicit argument, and the crate compiles with
//! `#![forbid(unsafe_code)]` and no third-party dependencies.
//!
//! # Modules
//!
//! * [`pos`] — [`WorldPos`], [`CELL_SIZE`], [`CELL_HALF`]
//! * [`cell`] — [`CellCoord`]
//! * [`octree`] — [`Octree`], [`NodeRef`] (implicit octree over the cell space)
//! * [`lod`] — [`LodSelector`], [`MAX_SELECTED`] (3D-Tiles-style screen-space error)
//! * [`galaxy`] — [`Galaxy`], [`StarInfo`]
//! * [`system`] — [`StarSystem`]
//! * [`interest`] — [`InterestGrid`]
//!
//! # Benchmarks
//!
//! The benchmark tests in `tests/acceptance.rs` assert *debug* timings
//! (1 000 000 [`InterestGrid::insert`] + 1 000 [`InterestGrid::query_sphere`]
//! calls with `r = 10 km`, `cap = 1000` in under 2 s). Measured on the
//! development machine (AMD gfx1151 laptop iGPU host, `x86_64`, `--release`,
//! `lto = "thin"`):
//!
//! | workload (release)                                | time   |
//! |---------------------------------------------------|--------|
//! | 1 000 000 × [`InterestGrid::insert`]              | ~0.35 s |
//! | 1 000 × [`InterestGrid::query_sphere`] (r=10 km, cap=1000) | ~0.05 s |
//!
//! Run them with `cargo test --release -p space-world bench_`.

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod cell;
pub mod galaxy;
pub mod hash;
pub mod interest;
pub mod lod;
pub mod octree;
pub mod pos;
pub mod system;

pub use cell::CellCoord;
pub use galaxy::{Galaxy, StarInfo};
pub use hash::splitmix64;
pub use interest::InterestGrid;
pub use lod::{LodSelector, MAX_SELECTED};
pub use octree::{NodeRef, Octree, MAX_DEPTH};
pub use pos::{WorldPos, CELL_HALF, CELL_SIZE};
pub use system::{PlanetInfo, StarSystem};
