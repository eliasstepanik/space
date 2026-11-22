//! Lossless wire codec for [`ShipState`].
//!
//! A [`ShipState`] is the one replicated component the physics writes, so its
//! encoding is what a `Delta`/`Snapshot` actually carries per ship. The codec is
//! **lossless**: a decoded state is bit-identical to the encoded one, which is
//! what lets the change-tick replication model treat "re-encoded to the same
//! bytes" as "did not change".
//!
//! # Layout (big endian, fixed width)
//!
//! ```text
//! pos     : WorldPos  (36 B — see [`space_protocol::pos`])
//! vel     : f32 × 3   (12 B, raw bits)
//! quat    : f32 × 4   (16 B, raw bits)
//! ang_vel : f32 × 3   (12 B, raw bits)
//! ```
//!
//! Total: 76 bytes. Every `f32` is written as its IEEE-754 bits (not a varint,
//! not a decimal), so `-0.0`, every finite subnormal and every finite value
//! survive the round trip exactly. The `pos` field reuses
//! [`space_protocol::pos`], which validates its `local` on decode. The velocity,
//! quaternion and angular-velocity fields are plain `f32`s the physics can
//! legitimately produce any *finite* value of, so they are not range-checked —
//! but they **are** checked for finiteness on decode: a decoded `ShipState` is
//! untrusted input (a [`space_protocol::HandoffCommit::state`] arrives from
//! another shard, a component payload from a client), and a non-finite value
//! there would poison the integrator permanently, so it is rejected with
//! [`space_protocol::CodecError::InvalidPayload`] rather than trusted.

use serde::{Deserialize, Serialize};

use crate::ShipState;

/// Encoded size of a [`ShipState`]: `WorldPos (36) + vel (12) + quat (16) +
/// ang_vel (12)` = 76 bytes.
pub const SHIP_STATE_LEN: usize = 36 + 12 + 16 + 12;

/// The wire form of a [`ShipState`].
///
/// Floats are carried as raw bit patterns so the encoding is lossless and
/// independent of any float formatting.
#[derive(Serialize, Deserialize)]
struct ShipStateWire {
    #[serde(with = "space_protocol::pos")]
    pos: space_world::WorldPos,
    vel_bits: [u32; 3],
    quat_bits: [u32; 4],
    ang_vel_bits: [u32; 3],
}

impl ShipStateWire {
    fn from_state(state: &ShipState) -> Self {
        Self {
            pos: state.pos,
            vel_bits: to_bits(&state.vel),
            quat_bits: to_bits(&state.quat),
            ang_vel_bits: to_bits(&state.ang_vel),
        }
    }

    fn into_state(self) -> Result<ShipState, space_protocol::CodecError> {
        let vel = from_bits(&self.vel_bits);
        let quat = from_bits(&self.quat_bits);
        let ang_vel = from_bits(&self.ang_vel_bits);
        // A `ShipState` decoded off the wire is *untrusted* input: a
        // [`space_protocol::HandoffCommit::state`] is produced by another shard,
        // and a ghost/delta payload by a client. A non-finite velocity,
        // quaternion or angular velocity would poison the integrator permanently
        // (a NaN never leaves a float state), so it is rejected here rather than
        // trusted. `pos.local` is validated by the `pos` codec, but re-check it
        // here too so this function is self-contained about finiteness.
        for (name, v) in [
            ("vel", vel.as_slice()),
            ("quat", quat.as_slice()),
            ("ang_vel", ang_vel.as_slice()),
            ("pos.local", self.pos.local.as_slice()),
        ] {
            for (axis, &f) in v.iter().enumerate() {
                if !f.is_finite() {
                    return Err(space_protocol::CodecError::InvalidPayload(format!(
                        "ShipState {name}[{axis}] is not finite: {f}"
                    )));
                }
            }
        }
        Ok(ShipState {
            pos: self.pos,
            vel,
            quat,
            ang_vel,
        })
    }
}

fn to_bits<const N: usize>(arr: &[f32; N]) -> [u32; N] {
    core::array::from_fn(|i| arr[i].to_bits())
}

fn from_bits<const N: usize>(arr: &[u32; N]) -> [f32; N] {
    core::array::from_fn(|i| f32::from_bits(arr[i]))
}

/// Encode a [`ShipState`] with the pinned protocol config.
///
/// This is the payload a replication pass places in an
/// [`space_protocol::EntitySnapshot`] for the `ShipState` component id.
#[must_use]
pub fn encode_ship_state(state: &ShipState) -> Vec<u8> {
    space_protocol::codec::encode_raw(&ShipStateWire::from_state(state))
}

/// Decode a [`ShipState`] previously produced by [`encode_ship_state`].
///
/// # Errors
///
/// [`space_protocol::CodecError::InvalidPayload`] when the bytes are not a valid
/// encoding (wrong length, or a `pos.local` that fails validation — see
/// [`space_protocol::pos`]).
pub fn decode_ship_state(bytes: &[u8]) -> Result<ShipState, space_protocol::CodecError> {
    let (wire, read) = space_protocol::codec::decode_raw::<ShipStateWire>(bytes)?;
    if read != bytes.len() {
        return Err(space_protocol::CodecError::InvalidPayload(format!(
            "{} of {} bytes were consumed",
            read,
            bytes.len()
        )));
    }
    wire.into_state()
}

/// Serialize a [`ShipState`] field (usable via `#[serde(with = "...")]`).
pub fn serialize<S: serde::Serializer>(state: &ShipState, s: S) -> Result<S::Ok, S::Error> {
    ShipStateWire::from_state(state).serialize(s)
}

/// Deserialize a [`ShipState`] field, rejecting a non-finite field (see
/// [`decode_ship_state`]).
pub fn deserialize<'de, D: serde::Deserializer<'de>>(d: D) -> Result<ShipState, D::Error> {
    ShipStateWire::deserialize(d)?
        .into_state()
        .map_err(serde::de::Error::custom)
}
