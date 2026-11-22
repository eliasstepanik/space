//! Lossless wire codec for [`space_world::WorldPos`].
//!
//! `space-world` deliberately carries no `serde` dependency (see its crate
//! docs), so — exactly as [`crate::shard`] converts [`space_world::NodeRef`] —
//! the protocol converts [`WorldPos`] here rather than pushing `serde` into the
//! world model.
//!
//! # Layout (36 bytes, big endian)
//!
//! ```text
//! cell  : i64 × 3   (24 B)
//! local : f32 × 3   (12 B, each written as its IEEE-754 bits)
//! ```
//!
//! The `cell` part is a plain big-endian `i64`, so it is exact for every value
//! the world addresses. The `local` part is written as the raw `f32` bits
//! ([`f32::to_bits`]) rather than through a decimal or a varint: that is what
//! makes the codec **lossless** — a decoded `local` is bit-identical to the
//! encoded one, including `-0.0` and every finite subnormal — which the
//! change-tick replication model depends on (a ship that did not move must
//! re-encode to the same bytes it started with, or a delta would look like a
//! change).
//!
//! # Validation on decode
//!
//! A [`WorldPos`] read off the wire is hostile input, and `WorldPos::new`
//! *normalises* any local — it would silently fold a bogus `local = 1e30` into a
//! huge cell carry instead of rejecting it. Decoding therefore rejects, before
//! constructing anything:
//!
//! * a non-finite `local` component (NaN or ±∞): these break every downstream
//!   `f64` distance/octree computation and can never be produced by the physics;
//! * a `local` component outside the canonical half-open range
//!   [`-CELL_HALF`, [`CELL_HALF`]): a normalised position always satisfies this,
//!   so an out-of-range value means a peer that is not running the same
//!   normalisation (or is attacking us), and accepting it would let two encodings
//!   name the same point and break the lossless round-trip property.
//!
//! The `cell` part needs no validation: every `i64` is a legal cell index.

use serde::{Deserialize, Serialize};
use space_world::{WorldPos, CELL_HALF};

/// Encoded size of a [`WorldPos`]: `3 × i64 + 3 × f32` = 36 bytes.
pub const WORLD_POS_LEN: usize = 3 * 8 + 3 * 4;

/// The wire form of a [`WorldPos`].
///
/// `local` is carried as the raw `f32` bit patterns (`u32`s) so the encoding is
/// lossless and the validation in [`WorldPosWire::into_position`] can reject a
/// hostile value before a `WorldPos` is ever built.
#[derive(Serialize, Deserialize)]
struct WorldPosWire {
    cell: [i64; 3],
    /// `local`, each component as its IEEE-754 bits.
    local_bits: [u32; 3],
}

impl WorldPosWire {
    fn from_position(pos: &WorldPos) -> Self {
        Self {
            cell: pos.cell,
            local_bits: [
                pos.local[0].to_bits(),
                pos.local[1].to_bits(),
                pos.local[2].to_bits(),
            ],
        }
    }

    fn into_position(self) -> Result<WorldPos, String> {
        let mut local = [0f32; 3];
        for (axis, &bits) in self.local_bits.iter().enumerate() {
            let v = f32::from_bits(bits);
            validate_local(axis, v)?;
            local[axis] = v;
        }
        Ok(WorldPos {
            cell: self.cell,
            local,
        })
    }
}

/// Reject a `local` component that is non-finite or outside the canonical
/// half-open range `[-CELL_HALF, CELL_HALF)`.
fn validate_local(axis: usize, v: f32) -> Result<(), String> {
    if !v.is_finite() {
        return Err(format!("WorldPos local axis {axis} is not finite: {v}"));
    }
    if v < -CELL_HALF || v >= CELL_HALF {
        return Err(format!(
            "WorldPos local axis {axis} value {v} is outside [{}, {CELL_HALF})",
            -CELL_HALF
        ));
    }
    Ok(())
}

/// Encode a [`WorldPos`] into its canonical 36-byte big-endian form.
///
/// The bytes are exactly what the [`serialize`]/[`deserialize`] serde adapters
/// produce under the pinned big-endian fixed-width config, so a caller that
/// wants the raw payload (e.g. to place inside an [`crate::EntitySnapshot`]
/// component) and a caller that serialises a struct containing a `WorldPos`
/// agree byte-for-byte.
#[must_use]
pub fn encode_world_pos(pos: &WorldPos) -> [u8; WORLD_POS_LEN] {
    let mut out = [0u8; WORLD_POS_LEN];
    let mut i = 0;
    for &c in &pos.cell {
        out[i..i + 8].copy_from_slice(&c.to_be_bytes());
        i += 8;
    }
    for &l in &pos.local {
        out[i..i + 4].copy_from_slice(&l.to_bits().to_be_bytes());
        i += 4;
    }
    out
}

/// Decode a [`WorldPos`] from its canonical form, validating every `local`
/// component (see the module docs).
///
/// `src` must be exactly [`WORLD_POS_LEN`] bytes.
pub fn decode_world_pos(src: &[u8]) -> Result<WorldPos, String> {
    if src.len() != WORLD_POS_LEN {
        return Err(format!(
            "WorldPos needs {WORLD_POS_LEN} bytes, got {}",
            src.len()
        ));
    }
    let mut cell = [0i64; 3];
    let mut i = 0;
    for slot in &mut cell {
        let mut b = [0u8; 8];
        b.copy_from_slice(&src[i..i + 8]);
        *slot = i64::from_be_bytes(b);
        i += 8;
    }
    let mut local = [0f32; 3];
    for (axis, slot) in local.iter_mut().enumerate() {
        let mut b = [0u8; 4];
        b.copy_from_slice(&src[i..i + 4]);
        let v = f32::from_bits(u32::from_be_bytes(b));
        validate_local(axis, v)?;
        *slot = v;
        i += 4;
    }
    Ok(WorldPos { cell, local })
}

/// Serialize a [`WorldPos`] field (usable via `#[serde(with = "...")]`).
pub fn serialize<S: serde::Serializer>(pos: &WorldPos, s: S) -> Result<S::Ok, S::Error> {
    WorldPosWire::from_position(pos).serialize(s)
}

/// Deserialize a [`WorldPos`] field, validating on the way in.
pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<WorldPos, D::Error> {
    let wire = WorldPosWire::deserialize(d)?;
    wire.into_position().map_err(serde::de::Error::custom)
}
