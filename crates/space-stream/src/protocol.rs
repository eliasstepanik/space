//! The stream wire format: frame framing (server → viewer) and viewer input
//! messages (viewer → server). Dependency-free so the wasm viewer shares it.

use std::fmt;

/// Length of the binary frame header: `seq: u64` big-endian + `flags: u8`.
pub const FRAME_HEADER_LEN: usize = 9;

/// Frame flag: the payload is a self-contained (intra) JPEG image.
pub const FLAG_KEYFRAME: u8 = 1;
/// Frame flag: the image was downscaled before encoding; the viewer should
/// stretch it to its canvas.
pub const FLAG_DOWNSCALED: u8 = 2;

/// Largest viewer → server message accepted, in bytes (every message is ≤ 10).
pub const MAX_INPUT_MESSAGE_LEN: usize = 16;

/// Why a wire message could not be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProtocolError {
    /// Shorter than the header / message layout requires.
    Truncated,
    /// Unknown message tag or key id.
    BadTag(u8),
    /// A `f32` field was NaN or infinite.
    NotFinite,
    /// Trailing bytes after a complete message.
    TrailingBytes,
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("message truncated"),
            Self::BadTag(t) => write!(f, "unknown tag {t}"),
            Self::NotFinite => f.write_str("non-finite float"),
            Self::TrailingBytes => f.write_str("trailing bytes"),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// The header in front of every JPEG frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    /// Monotonic frame sequence number (starts at 1).
    pub seq: u64,
    /// `FLAG_*` bits.
    pub flags: u8,
}

/// Frame a JPEG for the wire: header followed by the JPEG bytes.
pub fn encode_frame(header: FrameHeader, jpeg: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FRAME_HEADER_LEN + jpeg.len());
    out.extend_from_slice(&header.seq.to_be_bytes());
    out.push(header.flags);
    out.extend_from_slice(jpeg);
    out
}

/// Split a wire frame into its header and JPEG payload.
pub fn decode_frame(msg: &[u8]) -> Result<(FrameHeader, &[u8]), ProtocolError> {
    if msg.len() < FRAME_HEADER_LEN {
        return Err(ProtocolError::Truncated);
    }
    let (head, jpeg) = msg.split_at(FRAME_HEADER_LEN);
    let mut seq = [0u8; 8];
    seq.copy_from_slice(&head[..8]);
    Ok((
        FrameHeader {
            seq: u64::from_be_bytes(seq),
            flags: head[8],
        },
        jpeg,
    ))
}

/// A held-key input, device independent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Key {
    Forward = 1,
    Back = 2,
    Left = 3,
    Right = 4,
    Up = 5,
    Down = 6,
}

impl Key {
    /// Number of distinct keys.
    pub const COUNT: usize = 6;

    /// Every key, in wire-id order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Forward,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Up,
        Self::Down,
    ];

    /// Dense index `0..Key::COUNT` (the wire id minus one), for per-key tables.
    pub const fn index(self) -> usize {
        self as usize - 1
    }

    fn from_id(id: u8) -> Option<Self> {
        Some(match id {
            1 => Self::Forward,
            2 => Self::Back,
            3 => Self::Left,
            4 => Self::Right,
            5 => Self::Up,
            6 => Self::Down,
            _ => return None,
        })
    }

    /// Map a DOM `KeyboardEvent.code` to a key (`None` for unmapped keys).
    pub fn from_dom_code(code: &str) -> Option<Self> {
        Some(match code {
            "KeyW" | "ArrowUp" => Self::Forward,
            "KeyS" | "ArrowDown" => Self::Back,
            "KeyA" | "ArrowLeft" => Self::Left,
            "KeyD" | "ArrowRight" => Self::Right,
            "Space" => Self::Up,
            "ShiftLeft" | "ShiftRight" => Self::Down,
            _ => return None,
        })
    }
}

/// One viewer input event (viewer → server).
///
/// Wire layout (big-endian), one WebSocket binary message each:
///
/// | message | bytes |
/// |---|---|
/// | `Pointer` | `1, x: f32, y: f32, buttons: u8` (10) |
/// | `Key`     | `2, key: u8, down: u8` (3) |
/// | `Turn`    | `3, dyaw: f32, dpitch: f32` (9) |
///
/// Tag 3 carried an *absolute* `View { yaw, pitch }` before; the layout is
/// unchanged but the meaning is now relative. A viewer cannot know the proxy's
/// current view direction (it only receives pixels), so an absolute angle made
/// the camera snap to the viewer's private idea of "yaw 0" on the first drag,
/// on every page reload and whenever a second viewer touched the mouse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewerInput {
    /// Pointer position, normalised to `0..=1` over the canvas, and the DOM
    /// `MouseEvent.buttons` bitmask.
    Pointer { x: f32, y: f32, buttons: u8 },
    /// A key went down (`true`) or up (`false`).
    Key { key: Key, down: bool },
    /// Rotate the view by this many radians since the previous `Turn`
    /// (`dyaw` about +Y, positive turns left; positive `dpitch` looks up).
    /// The receiver owns the absolute yaw/pitch and their limits.
    Turn { dyaw: f32, dpitch: f32 },
}

impl ViewerInput {
    /// Serialise to the compact binary wire form.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(10);
        match *self {
            Self::Pointer { x, y, buttons } => {
                out.push(1);
                out.extend_from_slice(&x.to_be_bytes());
                out.extend_from_slice(&y.to_be_bytes());
                out.push(buttons);
            }
            Self::Key { key, down } => {
                out.push(2);
                out.push(key as u8);
                out.push(u8::from(down));
            }
            Self::Turn { dyaw, dpitch } => {
                out.push(3);
                out.extend_from_slice(&dyaw.to_be_bytes());
                out.extend_from_slice(&dpitch.to_be_bytes());
            }
        }
        out
    }

    /// Parse one wire message. Strict: exact length, known tags, finite floats.
    pub fn decode(msg: &[u8]) -> Result<Self, ProtocolError> {
        let (&tag, body) = msg.split_first().ok_or(ProtocolError::Truncated)?;
        let f32_at = |i: usize| -> Result<f32, ProtocolError> {
            let b: [u8; 4] = body
                .get(i..i + 4)
                .and_then(|s| s.try_into().ok())
                .ok_or(ProtocolError::Truncated)?;
            let v = f32::from_be_bytes(b);
            if v.is_finite() {
                Ok(v)
            } else {
                Err(ProtocolError::NotFinite)
            }
        };
        let (input, used) = match tag {
            1 => {
                let buttons = *body.get(8).ok_or(ProtocolError::Truncated)?;
                (
                    Self::Pointer {
                        x: f32_at(0)?,
                        y: f32_at(4)?,
                        buttons,
                    },
                    9,
                )
            }
            2 => {
                let id = *body.first().ok_or(ProtocolError::Truncated)?;
                let down = *body.get(1).ok_or(ProtocolError::Truncated)?;
                let key = Key::from_id(id).ok_or(ProtocolError::BadTag(id))?;
                (
                    Self::Key {
                        key,
                        down: down != 0,
                    },
                    2,
                )
            }
            3 => (
                Self::Turn {
                    dyaw: f32_at(0)?,
                    dpitch: f32_at(4)?,
                },
                8,
            ),
            other => return Err(ProtocolError::BadTag(other)),
        };
        if body.len() > used {
            return Err(ProtocolError::TrailingBytes);
        }
        Ok(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_header_round_trips_seq_and_flag() {
        for (seq, flags) in [
            (0u64, 0u8),
            (1, FLAG_KEYFRAME),
            (0x0102_0304_0506_0708, FLAG_KEYFRAME | FLAG_DOWNSCALED),
            (u64::MAX, 0xff),
        ] {
            let wire = encode_frame(FrameHeader { seq, flags }, b"\xff\xd8jpeg");
            assert_eq!(wire.len(), FRAME_HEADER_LEN + 6);
            let (h, payload) = decode_frame(&wire).unwrap();
            assert_eq!(h, FrameHeader { seq, flags });
            assert_eq!(payload, b"\xff\xd8jpeg");
        }
    }

    #[test]
    fn frame_wire_bytes_are_big_endian_seq_then_flag() {
        let wire = encode_frame(
            FrameHeader {
                seq: 0x0102,
                flags: 3,
            },
            &[9],
        );
        assert_eq!(wire, [0, 0, 0, 0, 0, 0, 1, 2, 3, 9]);
    }

    #[test]
    fn short_frame_is_rejected() {
        assert_eq!(decode_frame(&[0; 8]), Err(ProtocolError::Truncated));
        // A header with an empty payload is valid.
        assert_eq!(decode_frame(&[0; 9]).unwrap().1, &[] as &[u8]);
    }

    #[test]
    fn viewer_input_round_trips() {
        let cases = [
            ViewerInput::Pointer {
                x: 0.25,
                y: 0.75,
                buttons: 5,
            },
            ViewerInput::Key {
                key: Key::Forward,
                down: true,
            },
            ViewerInput::Key {
                key: Key::Down,
                down: false,
            },
            ViewerInput::Turn {
                dyaw: -1.5,
                dpitch: 0.5,
            },
        ];
        for c in cases {
            let wire = c.encode();
            assert!(wire.len() <= MAX_INPUT_MESSAGE_LEN);
            assert_eq!(ViewerInput::decode(&wire).unwrap(), c);
        }
    }

    #[test]
    fn viewer_input_wire_bytes_are_stable() {
        // The wasm viewer test asserts the same vectors.
        assert_eq!(
            ViewerInput::Key {
                key: Key::Left,
                down: true
            }
            .encode(),
            [2, 3, 1]
        );
        // Tag 3 is the relative `Turn`: tag, then `dyaw` and `dpitch` as
        // big-endian f32 (the layout the absolute `View` used to have).
        assert_eq!(
            ViewerInput::Turn {
                dyaw: 1.0,
                dpitch: 0.0
            }
            .encode(),
            [3, 0x3f, 0x80, 0, 0, 0, 0, 0, 0]
        );
        assert_eq!(
            ViewerInput::Turn {
                dyaw: 0.0,
                dpitch: -2.0
            }
            .encode(),
            [3, 0, 0, 0, 0, 0xc0, 0, 0, 0]
        );
        assert_eq!(
            ViewerInput::decode(&[3, 0xbf, 0x80, 0, 0, 0x3f, 0, 0, 0]),
            Ok(ViewerInput::Turn {
                dyaw: -1.0,
                dpitch: 0.5
            })
        );
    }

    #[test]
    fn key_indices_are_dense_and_match_the_wire_ids() {
        assert_eq!(Key::ALL.len(), Key::COUNT);
        for (i, key) in Key::ALL.into_iter().enumerate() {
            assert_eq!(key.index(), i);
            assert_eq!(key as usize, i + 1);
            assert_eq!(Key::from_id(key as u8), Some(key));
        }
        assert_eq!(Key::from_id(0), None);
        assert_eq!(Key::from_id(Key::COUNT as u8 + 1), None);
    }

    #[test]
    fn malformed_input_is_rejected() {
        assert_eq!(ViewerInput::decode(&[]), Err(ProtocolError::Truncated));
        assert_eq!(ViewerInput::decode(&[9]), Err(ProtocolError::BadTag(9)));
        assert_eq!(ViewerInput::decode(&[2, 1]), Err(ProtocolError::Truncated));
        assert_eq!(
            ViewerInput::decode(&[2, 99, 1]),
            Err(ProtocolError::BadTag(99))
        );
        assert_eq!(
            ViewerInput::decode(&[2, 1, 1, 0]),
            Err(ProtocolError::TrailingBytes)
        );
        let mut nan = vec![3];
        nan.extend_from_slice(&f32::NAN.to_be_bytes());
        nan.extend_from_slice(&0f32.to_be_bytes());
        assert_eq!(ViewerInput::decode(&nan), Err(ProtocolError::NotFinite));
        let mut inf = vec![1];
        inf.extend_from_slice(&f32::INFINITY.to_be_bytes());
        inf.extend_from_slice(&[0; 5]);
        assert_eq!(ViewerInput::decode(&inf), Err(ProtocolError::NotFinite));
    }

    #[test]
    fn dom_codes_map_to_keys() {
        assert_eq!(Key::from_dom_code("KeyW"), Some(Key::Forward));
        assert_eq!(Key::from_dom_code("ArrowLeft"), Some(Key::Left));
        assert_eq!(Key::from_dom_code("Space"), Some(Key::Up));
        assert_eq!(Key::from_dom_code("F13"), None);
    }
}
