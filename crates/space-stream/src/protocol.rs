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

/// Largest viewer → server message accepted, in bytes (every message is ≤ 10;
/// typed text travels one character per message, so it does not raise this).
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
    /// A `Text` code point that is not a Unicode scalar value (a surrogate, or
    /// past `U+10FFFF`) or is a control character.
    BadChar(u32),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => f.write_str("message truncated"),
            Self::BadTag(t) => write!(f, "unknown tag {t}"),
            Self::NotFinite => f.write_str("non-finite float"),
            Self::TrailingBytes => f.write_str("trailing bytes"),
            Self::BadChar(c) => write!(f, "U+{c:04X} is not a typeable character"),
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

/// A key, device independent. Two families share the id space:
///
/// * **Movement keys** (ids 1–6): *held* keys. Their state matters for as long
///   as they are down, and every down is eventually followed by an up.
/// * **Text-entry keys** (ids 7–15): the keys a text field needs besides the
///   characters themselves (those travel as [`ViewerInput::Text`]). They are
///   *strokes*: the receiver acts on `down: true` and ignores the up, and a
///   sender reports one stroke as a down directly followed by an up (and an
///   auto-repeat as one more such pair), so nothing can be left "held".
///
/// The caret keys are separate from [`Key::Left`] / [`Key::Right`] on purpose:
/// those two also come from the `A` / `D` keys, which in a text field type a
/// letter and must not move the caret.
///
/// Ids are append-only: an id is never renumbered or reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Key {
    Forward = 1,
    Back = 2,
    Left = 3,
    Right = 4,
    Up = 5,
    Down = 6,
    /// Text entry: confirm (Enter / Return).
    Enter = 7,
    /// Text entry: delete the character before the caret.
    Backspace = 8,
    /// Text entry: cancel.
    Escape = 9,
    /// Text entry: next field.
    Tab = 10,
    /// Text entry: delete the character after the caret.
    Delete = 11,
    /// Text entry: caret to the start.
    Home = 12,
    /// Text entry: caret to the end.
    End = 13,
    /// Text entry: caret one character left.
    CaretLeft = 14,
    /// Text entry: caret one character right.
    CaretRight = 15,
}

impl Key {
    /// Number of distinct keys.
    pub const COUNT: usize = 15;

    /// Number of movement keys; they are the first ids, so their
    /// [`Key::index`] is `0..Key::MOVEMENT_COUNT`.
    pub const MOVEMENT_COUNT: usize = 6;

    /// Every key, in wire-id order.
    pub const ALL: [Self; Self::COUNT] = [
        Self::Forward,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Up,
        Self::Down,
        Self::Enter,
        Self::Backspace,
        Self::Escape,
        Self::Tab,
        Self::Delete,
        Self::Home,
        Self::End,
        Self::CaretLeft,
        Self::CaretRight,
    ];

    /// Dense index `0..Key::COUNT` (the wire id minus one), for per-key tables.
    pub const fn index(self) -> usize {
        self as usize - 1
    }

    /// Whether this is a held movement key (as opposed to a text-entry stroke).
    pub const fn is_movement(self) -> bool {
        self.index() < Self::MOVEMENT_COUNT
    }

    /// Whether this is a text-entry key: a stroke that acts on `down: true`.
    pub const fn is_text_entry(self) -> bool {
        !self.is_movement()
    }

    fn from_id(id: u8) -> Option<Self> {
        Self::ALL.get(usize::from(id).checked_sub(1)?).copied()
    }

    /// Map a DOM `KeyboardEvent.code` (the physical key, whatever the keyboard
    /// layout) to a **movement** key; `None` for every other key.
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

    /// Map a DOM `KeyboardEvent.key` (the key's meaning under the current
    /// layout and modifiers) to a **text-entry** key; `None` for every other
    /// key. Printable keys are not keys here: their character is sent as
    /// [`ViewerInput::Text`].
    pub fn from_dom_key(key: &str) -> Option<Self> {
        Some(match key {
            "Enter" => Self::Enter,
            "Backspace" => Self::Backspace,
            "Escape" | "Esc" => Self::Escape,
            "Tab" => Self::Tab,
            "Delete" | "Del" => Self::Delete,
            "Home" => Self::Home,
            "End" => Self::End,
            "ArrowLeft" | "Left" => Self::CaretLeft,
            "ArrowRight" | "Right" => Self::CaretRight,
            _ => return None,
        })
    }
}

/// Whether `c` may travel as [`ViewerInput::Text`]: any Unicode scalar value
/// that is not a control character (so no NUL, newline, tab, escape or DEL —
/// those are keys, see [`Key`]).
pub fn is_typeable(c: char) -> bool {
    !c.is_control()
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
/// | `Text`    | `4, scalar: u32` (5) |
///
/// Tag 3 carried an *absolute* `View { yaw, pitch }` before; the layout is
/// unchanged but the meaning is now relative. A viewer cannot know the proxy's
/// current view direction (it only receives pixels), so an absolute angle made
/// the camera snap to the viewer's private idea of "yaw 0" on the first drag,
/// on every page reload and whenever a second viewer touched the mouse.
///
/// Tags and key ids are append-only (tag 4 and key ids 7–15 were added for
/// text entry). A receiver that predates a tag rejects the message
/// ([`ProtocolError::BadTag`]) and carries on.
///
/// A viewer cannot see which screen the proxy shows, so it does not decide
/// what a key press *means*: a printable key is sent both as its movement key
/// (if it has one) and as `Text`, and the receiver uses whichever its current
/// screen wants.
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
    /// One typed character, as the keyboard layout produced it (a pasted or
    /// longer text is one `Text` per character, in order). Never a control
    /// character ([`is_typeable`]): Enter, Backspace and the like are [`Key`]s.
    Text(char),
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
            Self::Text(c) => {
                out.push(4);
                out.extend_from_slice(&u32::from(c).to_be_bytes());
            }
        }
        out
    }

    /// `Text(c)` if `c` may be sent as text ([`is_typeable`]), else `None`.
    pub fn text(c: char) -> Option<Self> {
        is_typeable(c).then_some(Self::Text(c))
    }

    /// Parse one wire message. Strict: exact length, known tags, finite floats,
    /// typeable characters. Total: any byte string gives `Ok` or `Err`, never a
    /// panic, and nothing is allocated.
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
            4 => {
                let b: [u8; 4] = body
                    .get(..4)
                    .and_then(|s| s.try_into().ok())
                    .ok_or(ProtocolError::Truncated)?;
                let scalar = u32::from_be_bytes(b);
                let c = char::from_u32(scalar)
                    .filter(|c| is_typeable(*c))
                    .ok_or(ProtocolError::BadChar(scalar))?;
                (Self::Text(c), 4)
            }
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
            ViewerInput::Text('a'),
            ViewerInput::Text(' '),
            ViewerInput::Text(':'),
            ViewerInput::Text('ö'),
            ViewerInput::Text('€'),
            ViewerInput::Text('\u{1F680}'),
            ViewerInput::Text(char::MAX),
        ];
        for c in cases {
            let wire = c.encode();
            assert!(wire.len() <= MAX_INPUT_MESSAGE_LEN);
            assert_eq!(ViewerInput::decode(&wire).unwrap(), c);
        }
        // Every key, both ways.
        for key in Key::ALL {
            for down in [true, false] {
                let c = ViewerInput::Key { key, down };
                assert_eq!(ViewerInput::decode(&c.encode()), Ok(c));
            }
        }
    }

    /// The ids that were on the wire before text entry existed keep their
    /// meaning (append-only): a viewer and a proxy of different ages agree on
    /// every message both know.
    #[test]
    fn the_wire_ids_are_append_only() {
        let ids = Key::ALL.map(|k| (k, k as u8));
        assert_eq!(
            ids,
            [
                (Key::Forward, 1),
                (Key::Back, 2),
                (Key::Left, 3),
                (Key::Right, 4),
                (Key::Up, 5),
                (Key::Down, 6),
                (Key::Enter, 7),
                (Key::Backspace, 8),
                (Key::Escape, 9),
                (Key::Tab, 10),
                (Key::Delete, 11),
                (Key::Home, 12),
                (Key::End, 13),
                (Key::CaretLeft, 14),
                (Key::CaretRight, 15),
            ]
        );
        let tag = |input: ViewerInput| input.encode()[0];
        let (x, y, buttons) = (0.0, 0.0, 0);
        assert_eq!(tag(ViewerInput::Pointer { x, y, buttons }), 1);
        let (key, down) = (Key::Up, true);
        assert_eq!(tag(ViewerInput::Key { key, down }), 2);
        let (dyaw, dpitch) = (0.0, 0.0);
        assert_eq!(tag(ViewerInput::Turn { dyaw, dpitch }), 3);
        assert_eq!(tag(ViewerInput::Text('x')), 4);
    }

    #[test]
    fn movement_and_text_entry_keys_are_told_apart() {
        for (i, key) in Key::ALL.into_iter().enumerate() {
            assert_eq!(key.is_movement(), i < Key::MOVEMENT_COUNT, "{key:?}");
            assert_eq!(key.is_text_entry(), !key.is_movement(), "{key:?}");
        }
        assert!(Key::Down.is_movement() && Key::Enter.is_text_entry());
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
        // Text: tag, then the scalar value as a big-endian u32.
        assert_eq!(ViewerInput::Text('A').encode(), [4, 0, 0, 0, 0x41]);
        assert_eq!(ViewerInput::Text('€').encode(), [4, 0, 0, 0x20, 0xac]);
        assert_eq!(
            ViewerInput::decode(&[4, 0, 0x01, 0xf6, 0x80]),
            Ok(ViewerInput::Text('\u{1F680}'))
        );
        // A text-entry key is an ordinary key message with a newer id.
        assert_eq!(
            ViewerInput::Key {
                key: Key::Enter,
                down: true
            }
            .encode(),
            [2, 7, 1]
        );
        assert_eq!(
            ViewerInput::decode(&[2, 15, 0]),
            Ok(ViewerInput::Key {
                key: Key::CaretRight,
                down: false
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
        // Key id 0 and the first id past the table.
        assert_eq!(
            ViewerInput::decode(&[2, 0, 1]),
            Err(ProtocolError::BadTag(0))
        );
        assert_eq!(
            ViewerInput::decode(&[2, 16, 1]),
            Err(ProtocolError::BadTag(16))
        );
        // The first tag past the table.
        assert_eq!(
            ViewerInput::decode(&[5, 0, 0, 0, 0]),
            Err(ProtocolError::BadTag(5))
        );
    }

    /// Text is exactly one typeable Unicode scalar value: nothing shorter,
    /// nothing longer, no surrogate, nothing past the code space, and no
    /// control character (a newline is the Enter *key*; an escape character
    /// must never reach whatever prints the text).
    #[test]
    fn malformed_text_is_rejected() {
        use ProtocolError::{BadChar, TrailingBytes, Truncated};
        assert_eq!(ViewerInput::decode(&[4]), Err(Truncated));
        assert_eq!(ViewerInput::decode(&[4, 0, 0, 0]), Err(Truncated));
        assert_eq!(
            ViewerInput::decode(&[4, 0, 0, 0, 0x41, 0]),
            Err(TrailingBytes)
        );
        for scalar in [
            0xd800u32, // first surrogate
            0xdfff,    // last surrogate
            0x11_0000, // first value past the code space
            u32::MAX,
            0,    // NUL
            0x0a, // newline
            0x09, // tab
            0x08, // backspace
            0x1b, // escape
            0x7f, // DEL
            0x85, // a C1 control
        ] {
            let mut wire = vec![4];
            wire.extend_from_slice(&scalar.to_be_bytes());
            assert_eq!(
                ViewerInput::decode(&wire),
                Err(BadChar(scalar)),
                "{scalar:#x}"
            );
        }
        // The neighbours of the rejected ranges are fine.
        for scalar in [0x20u32, 0x7e, 0xa0, 0xd7ff, 0xe000, 0x10_ffff] {
            let mut wire = vec![4];
            wire.extend_from_slice(&scalar.to_be_bytes());
            let c = char::from_u32(scalar).unwrap();
            assert_eq!(ViewerInput::decode(&wire), Ok(ViewerInput::Text(c)));
        }
        // The constructor refuses what the decoder would refuse.
        assert_eq!(ViewerInput::text('x'), Some(ViewerInput::Text('x')));
        assert_eq!(ViewerInput::text('\n'), None);
        assert_eq!(ViewerInput::text('\u{1b}'), None);
        assert!(is_typeable(' ') && !is_typeable('\0'));
        assert!(BadChar(0x0a).to_string().contains("U+000A"));
    }

    /// Decoding is total: no byte string makes it panic, and whatever it
    /// accepts survives a round trip. Every message of up to three bytes is
    /// tried, then every tag with pseudo-random bodies of every length up to
    /// (and past) the largest message.
    #[test]
    fn decoding_never_panics_and_accepted_messages_round_trip() {
        fn check(msg: &[u8]) -> bool {
            match ViewerInput::decode(msg) {
                Ok(input) => {
                    assert!(msg.len() <= MAX_INPUT_MESSAGE_LEN, "{msg:?}");
                    assert_eq!(ViewerInput::decode(&input.encode()), Ok(input), "{msg:?}");
                    true
                }
                Err(_) => false,
            }
        }
        assert!(!check(&[]));
        let mut accepted = 0u32;
        for a in 0..=255u8 {
            accepted += u32::from(check(&[a]));
            for b in 0..=255u8 {
                accepted += u32::from(check(&[a, b]));
                for c in [0u8, 1, 2, 0x7f, 0x80, 0xff] {
                    accepted += u32::from(check(&[a, b, c]));
                }
            }
        }
        // Only key messages are that short: 15 keys, six third bytes each.
        assert_eq!(accepted, 15 * 6);

        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        };
        let mut accepted = [0u32; 8];
        for round in 0..20_000 {
            for tag in 0..8u8 {
                let len = round % (MAX_INPUT_MESSAGE_LEN + 4);
                let mut msg = vec![tag];
                msg.extend((0..len).map(|_| next()));
                // Bias towards plausible values so the accepting paths run too.
                if round % 3 == 0 && msg.len() > 2 {
                    msg[1] = 0;
                    msg[2] &= 0x0f;
                }
                accepted[usize::from(tag)] += u32::from(check(&msg));
            }
        }
        assert_eq!(accepted[0], 0, "tag 0 does not exist");
        for (tag, count) in accepted.iter().enumerate().take(5).skip(1) {
            assert!(*count > 0, "tag {tag} was never accepted");
        }
        assert_eq!(accepted[5..], [0; 3], "tags past the table are rejected");
    }

    #[test]
    fn dom_codes_map_to_keys() {
        assert_eq!(Key::from_dom_code("KeyW"), Some(Key::Forward));
        assert_eq!(Key::from_dom_code("ArrowLeft"), Some(Key::Left));
        assert_eq!(Key::from_dom_code("Space"), Some(Key::Up));
        assert_eq!(Key::from_dom_code("F13"), None);
        // Only movement keys come from the physical key.
        for code in ["Enter", "Backspace", "Tab", "Escape", "Home", "Delete"] {
            assert_eq!(Key::from_dom_code(code), None, "{code}");
        }
        for code in [
            "KeyW",
            "KeyA",
            "KeyS",
            "KeyD",
            "Space",
            "ShiftLeft",
            "ArrowUp",
        ] {
            assert!(Key::from_dom_code(code).is_some_and(Key::is_movement));
        }
    }

    #[test]
    fn dom_key_values_map_to_text_entry_keys() {
        for (dom, key) in [
            ("Enter", Key::Enter),
            ("Backspace", Key::Backspace),
            ("Escape", Key::Escape),
            ("Esc", Key::Escape),
            ("Tab", Key::Tab),
            ("Delete", Key::Delete),
            ("Del", Key::Delete),
            ("Home", Key::Home),
            ("End", Key::End),
            ("ArrowLeft", Key::CaretLeft),
            ("Left", Key::CaretLeft),
            ("ArrowRight", Key::CaretRight),
            ("Right", Key::CaretRight),
        ] {
            assert_eq!(Key::from_dom_key(dom), Some(key), "{dom}");
            assert!(key.is_text_entry());
        }
        // Printable keys, modifiers, function keys and the vertical arrows
        // (a one-line field has no use for them) are not text-entry keys.
        for dom in [
            "a",
            "A",
            " ",
            "1",
            "Shift",
            "Control",
            "F5",
            "F12",
            "ArrowUp",
            "ArrowDown",
            "Dead",
            "Unidentified",
            "Process",
            "enter",
            "",
        ] {
            assert_eq!(Key::from_dom_key(dom), None, "{dom:?}");
        }
    }
}
