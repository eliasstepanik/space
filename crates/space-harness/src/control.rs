//! The control surface: everything a user can do to a running client and
//! everything a user can see of it, and nothing more.
//!
//! [`Control`] is implemented by the client (most easily through
//! [`crate::Hub`]) and used by the two front ends, [`crate::script`] (Lua) and
//! [`crate::http`]. Input goes in as [`ViewerInput`] — the vocabulary the
//! browser viewer already speaks — so the window, the viewer, the HTTP API and
//! a Lua script all feed one input path.

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use space_stream::protocol::is_typeable;
pub use space_stream::protocol::{Key, ViewerInput};

/// Which screen the client is showing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Screen {
    /// The connect prompt: the user types a server address.
    #[default]
    Connect,
    /// A connection attempt is in flight.
    Connecting,
    /// Playing on a server.
    Online,
    /// The connection failed or was lost; [`ClientState::error`] says why.
    Error,
}

impl Screen {
    /// The name used in JSON and in Lua (`"connect"`, `"connecting"`,
    /// `"online"`, `"error"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Connect => "connect",
            Self::Connecting => "connecting",
            Self::Online => "online",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for Screen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A snapshot of what the client shows and knows, as plain owned data.
///
/// It serialises to a flat JSON object (`GET /state`) and becomes a Lua table
/// with the same field names (`client.state()`). Fields that have no value yet
/// are `null` in JSON and `nil` in Lua.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClientState {
    /// The screen being shown.
    pub screen: Screen,
    /// The error text on the [`Screen::Error`] screen.
    pub error: Option<String>,
    /// The server address: what is typed in the connect prompt, or the server
    /// being connected to / played on.
    pub address: String,
    /// The shard currently played on.
    pub shard: Option<u32>,
    /// The session id the current shard assigned.
    pub session: Option<u64>,
    /// The newest server tick seen (0 before the first).
    pub tick: u64,
    /// The ship's position in metres, world frame.
    pub position: Option<[f64; 3]>,
    /// The ship's position in metres from the anchor of the scene being
    /// drawn: (0, 0, 0) is the scene's spawn point, wherever in the world the
    /// cluster's map lies. It is what the HUD shows, and what a script steers
    /// by when it wants to reach something in the scene.
    pub scene_position: Option<[f64; 3]>,
    /// The ship's velocity in metres/second, world frame.
    pub velocity: Option<[f64; 3]>,
    /// The length of [`velocity`](Self::velocity) (0 without a ship). Set
    /// together with it by [`ClientState::set_ship`].
    pub speed: f64,
    /// View yaw in degrees (0 looks down -Z, positive turns left).
    pub yaw_deg: f32,
    /// View pitch in degrees (positive looks up).
    pub pitch_deg: f32,
    /// Completed shard transitions.
    pub handoffs: u64,
    /// The scene being drawn while playing, by its catalogue name
    /// (`"demo-area"`, `"materials"`, `"solar"`, `"galaxy"`). The server
    /// chooses it; `None` while not on a server.
    pub scene: Option<String>,
    /// The revision the server chose the scene at; it goes up with every
    /// change. 0 while not on a server, and while the server has not said
    /// (the default scene is drawn then).
    pub scene_revision: u64,
    /// Seconds until the client tries to connect again on its own (it was
    /// started with a server address and the connection failed or was lost).
    /// `None` when no automatic attempt is pending.
    pub retry_in_s: Option<f32>,
    /// The number of the last frame rendered (the [`Frame::seq`] of its
    /// picture); it goes up by one per frame.
    pub frame: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frames per second, as the client measures it.
    pub fps: f32,
    /// The strength of the gravity at the ship, m/s² (0 in weightless
    /// space): what the client's prediction of the scene's physics finds
    /// there (the demo area's gravity room pulls with 9.81).
    #[serde(default)]
    pub gravity: f64,
    /// Whether the ship stands on something in gravity (the floor of the
    /// gravity room, a crate).
    #[serde(default)]
    pub landed: bool,
    /// The physics bodies the client draws (crates, balls, the door, ...),
    /// nearest to the ship first, at most 64: `bodies[1]` in Lua is the
    /// nearest one.
    #[serde(default)]
    pub bodies: Vec<BodyState>,
    /// How many of [`bodies`](Self::bodies) are in the picture.
    #[serde(default)]
    pub bodies_in_view: u32,
    /// The last correction the client's prediction of its own ship needed
    /// when the server's word arrived, metres.
    #[serde(default)]
    pub correction_m: f64,
    /// The largest such correction of the last five seconds, metres.
    #[serde(default)]
    pub correction_max_m: f64,
}

/// One physics body as the client draws it ([`ClientState::bodies`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct BodyState {
    /// The body's guid, `"shard:serial"` (stable across shard transitions).
    pub guid: String,
    /// What it is: the model's name (`"crate"`, `"ball"`, `"plank"`,
    /// `"door"`, `"chain"`, `"pendulum"`, ...).
    pub visual: String,
    /// The model's id (`space_scene::visual`).
    pub visual_id: u32,
    /// Where its root part is drawn, metres from the scene's spawn point
    /// (the frame of [`ClientState::scene_position`]).
    pub position: [f64; 3],
    /// How far that is from the ship (as drawn), metres.
    pub distance: f64,
    /// Its speed, m/s.
    pub speed: f64,
    /// How many rigid parts it has (a chain has six).
    pub parts: u32,
    /// Whether it is in the picture.
    pub in_view: bool,
}

impl ClientState {
    /// Set the ship's position and velocity, and [`speed`](Self::speed) from
    /// the velocity.
    pub fn set_ship(&mut self, position: [f64; 3], velocity: [f64; 3]) {
        self.position = Some(position);
        self.velocity = Some(velocity);
        self.speed = velocity.iter().map(|v| v * v).sum::<f64>().sqrt();
    }

    /// Forget the ship (not playing).
    pub fn clear_ship(&mut self) {
        self.position = None;
        self.scene_position = None;
        self.velocity = None;
        self.speed = 0.0;
    }
}

/// One rendered frame: tightly packed RGBA8, top row first.
///
/// Cloning shares the pixels.
#[derive(Clone, PartialEq, Eq)]
pub struct Frame {
    /// The frame's number (see [`ClientState::frame`]).
    pub seq: u64,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 4` bytes.
    pub rgba: Arc<Vec<u8>>,
}

impl Frame {
    /// Wrap a rendered picture. `None` if either dimension is zero or `rgba`
    /// is not `width * height * 4` bytes.
    #[must_use]
    pub fn new(seq: u64, width: u32, height: u32, rgba: Vec<u8>) -> Option<Self> {
        let expected = u64::from(width) * u64::from(height) * 4;
        if width == 0 || height == 0 || rgba.len() as u64 != expected {
            return None;
        }
        Some(Self {
            seq,
            width,
            height,
            rgba: Arc::new(rgba),
        })
    }

    /// The `[r, g, b, a]` of the pixel at column `x`, row `y` (row 0 is the
    /// top), or `None` outside the picture.
    #[must_use]
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let at = (u64::from(y) * u64::from(self.width) + u64::from(x)) * 4;
        let at = usize::try_from(at).ok()?;
        self.rgba.get(at..at + 4)?.try_into().ok()
    }
}

impl fmt::Debug for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Not the pixels: a frame is megabytes.
        f.debug_struct("Frame")
            .field("seq", &self.seq)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("bytes", &self.rgba.len())
            .finish()
    }
}

/// Why a [`Control`] call did not do what was asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ControlError {
    /// Nothing arrived within the timeout (no frame was rendered in time).
    Timeout,
    /// The client is not taking input: its queue is full, so it has not run a
    /// frame for a long time.
    Busy,
    /// The client has shut down.
    Closed,
    /// The request itself is wrong; the text says how.
    Invalid(String),
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout => f.write_str("timed out waiting for the client"),
            Self::Busy => f.write_str("the client is not taking input (its queue is full)"),
            Self::Closed => f.write_str("the client has shut down"),
            Self::Invalid(why) => f.write_str(why),
        }
    }
}

impl std::error::Error for ControlError {}

/// A running client, as a user has it: input goes in, a state snapshot and
/// pictures come out.
///
/// The implementor is a render loop that must never be stalled by a script or
/// an HTTP client, so the contract is: **every method returns promptly**.
/// [`input`](Self::input), [`connect`](Self::connect),
/// [`disconnect`](Self::disconnect), [`state`](Self::state) and
/// [`quit`](Self::quit) only queue a request or copy a snapshot;
/// [`frame`](Self::frame) is the one call that may wait, and it is bounded by
/// its `timeout`. Requests are asynchronous: their effect shows up in a later
/// [`state`](Self::state).
///
/// The trait is object safe; front ends hold an `Arc<dyn Control>`.
pub trait Control: Send + Sync + fmt::Debug {
    /// Feed one input event, exactly as if it had come from the keyboard or
    /// mouse: a key going down or up, a typed character, a relative view turn,
    /// a pointer position.
    ///
    /// # Errors
    ///
    /// [`ControlError::Busy`] when the client is not consuming input,
    /// [`ControlError::Closed`] after it shut down.
    fn input(&self, input: ViewerInput) -> Result<(), ControlError>;

    /// Ask the client to connect to the server at `address` (`host:port`), as
    /// if the user had typed it into the connect prompt and pressed Enter.
    /// Returns as soon as the request is queued; follow
    /// [`ClientState::screen`] for the outcome.
    ///
    /// # Errors
    ///
    /// [`ControlError::Invalid`] for an address that cannot be one (empty, too
    /// long, control characters), otherwise as [`input`](Self::input).
    fn connect(&self, address: &str) -> Result<(), ControlError>;

    /// Ask the client to leave the server and return to the connect prompt.
    ///
    /// # Errors
    ///
    /// As [`input`](Self::input).
    fn disconnect(&self) -> Result<(), ControlError>;

    /// The current snapshot. Never waits.
    fn state(&self) -> ClientState;

    /// A rendered frame.
    ///
    /// With `newer_than: None`: the picture the client is showing now — the
    /// frame of [`ClientState::frame`] or a later one. A client that only reads
    /// pixels back on demand renders one for this call.
    ///
    /// With `newer_than: Some(seq)`: the first frame with a number above `seq`,
    /// so `Some(state().frame)` is "the next frame".
    ///
    /// # Errors
    ///
    /// [`ControlError::Timeout`] when no such frame was rendered within
    /// `timeout`, [`ControlError::Closed`] after the client shut down.
    fn frame(&self, newer_than: Option<u64>, timeout: Duration) -> Result<Frame, ControlError>;

    /// Ask the client to exit. Idempotent.
    fn quit(&self);

    /// A key going down or up.
    ///
    /// # Errors
    ///
    /// As [`input`](Self::input).
    fn key(&self, key: Key, down: bool) -> Result<(), ControlError> {
        self.input(ViewerInput::Key { key, down })
    }

    /// One key stroke: down directly followed by up. This is how a text-entry
    /// key (Enter, Backspace, ...) is pressed; a movement key pressed this way
    /// is released before a frame has seen it held.
    ///
    /// # Errors
    ///
    /// As [`input`](Self::input).
    fn stroke(&self, key: Key) -> Result<(), ControlError> {
        self.key(key, true)?;
        self.key(key, false)
    }

    /// Type `text`, one character at a time. A newline is the Enter key and a
    /// tab the Tab key, as on a keyboard.
    ///
    /// # Errors
    ///
    /// [`ControlError::Invalid`] — before anything is typed — if `text` holds
    /// any other control character; otherwise as [`input`](Self::input).
    fn type_text(&self, text: &str) -> Result<(), ControlError> {
        if let Some(bad) = text.chars().find(|c| how_typed(*c).is_none()) {
            return Err(ControlError::Invalid(format!(
                "cannot type the control character U+{:04X}",
                u32::from(bad)
            )));
        }
        for typed in text.chars().filter_map(how_typed) {
            match typed {
                Typed::Char(c) => self.input(ViewerInput::Text(c))?,
                Typed::Stroke(key) => self.stroke(key)?,
            }
        }
        Ok(())
    }

    /// Turn the view by `dyaw` and `dpitch` **degrees** (positive yaw turns
    /// left, positive pitch looks up).
    ///
    /// # Errors
    ///
    /// [`ControlError::Invalid`] for a non-finite angle, otherwise as
    /// [`input`](Self::input).
    fn turn_degrees(&self, dyaw: f64, dpitch: f64) -> Result<(), ControlError> {
        let radians = |degrees: f64| {
            let radians = degrees.to_radians() as f32;
            (degrees.is_finite() && radians.is_finite()).then_some(radians)
        };
        match (radians(dyaw), radians(dpitch)) {
            (Some(dyaw), Some(dpitch)) => self.input(ViewerInput::Turn { dyaw, dpitch }),
            _ => Err(ControlError::Invalid(
                "the turn angles must be finite".into(),
            )),
        }
    }
}

/// What typing one character amounts to.
enum Typed {
    Char(char),
    Stroke(Key),
}

/// How `c` is typed, or `None` if it cannot be.
fn how_typed(c: char) -> Option<Typed> {
    match c {
        '\n' => Some(Typed::Stroke(Key::Enter)),
        '\t' => Some(Typed::Stroke(Key::Tab)),
        c if is_typeable(c) => Some(Typed::Char(c)),
        _ => None,
    }
}

/// Every key with the names scripts and the HTTP API know it by; the first
/// name is the canonical one ([`key_name`]).
const KEY_NAMES: [(Key, &[&str]); Key::COUNT] = [
    (Key::Forward, &["w", "forward"]),
    (Key::Back, &["s", "back"]),
    (Key::Left, &["a", "left"]),
    (Key::Right, &["d", "right"]),
    (Key::Up, &["space", "up"]),
    (Key::Down, &["shift", "down"]),
    (Key::Enter, &["enter", "return"]),
    (Key::Backspace, &["backspace"]),
    (Key::Escape, &["escape", "esc"]),
    (Key::Tab, &["tab"]),
    (Key::Delete, &["delete", "del"]),
    (Key::Home, &["home"]),
    (Key::End, &["end"]),
    (Key::CaretLeft, &["caret_left"]),
    (Key::CaretRight, &["caret_right"]),
];

/// The key called `name` (case-insensitive), or `None`.
///
/// Movement keys go by their default binding or their meaning: `w`/`forward`,
/// `s`/`back`, `a`/`left`, `d`/`right`, `space`/`up`, `shift`/`down`.
/// Text-entry keys: `enter`/`return`, `backspace`, `escape`/`esc`, `tab`,
/// `delete`/`del`, `home`, `end`, `caret_left`, `caret_right` (the arrow keys
/// in a text field; `left` and `right` are the strafe keys).
#[must_use]
pub fn key_from_name(name: &str) -> Option<Key> {
    KEY_NAMES
        .iter()
        .find(|(_, names)| names.iter().any(|n| n.eq_ignore_ascii_case(name)))
        .map(|(key, _)| *key)
}

/// The canonical name of `key` (the first of its names in [`key_from_name`]).
#[must_use]
pub fn key_name(key: Key) -> &'static str {
    KEY_NAMES[key.index()].1[0]
}

/// Every name [`key_from_name`] accepts, space separated (for help texts and
/// error messages).
#[must_use]
pub fn key_names() -> String {
    let names: Vec<&str> = KEY_NAMES
        .iter()
        .flat_map(|(_, names)| names.iter().copied())
        .collect();
    names.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Records what it is asked to do.
    #[derive(Debug, Default)]
    struct Recorder {
        inputs: Mutex<Vec<ViewerInput>>,
    }

    impl Control for Recorder {
        fn input(&self, input: ViewerInput) -> Result<(), ControlError> {
            self.inputs.lock().unwrap().push(input);
            Ok(())
        }
        fn connect(&self, _: &str) -> Result<(), ControlError> {
            Ok(())
        }
        fn disconnect(&self) -> Result<(), ControlError> {
            Ok(())
        }
        fn state(&self) -> ClientState {
            ClientState::default()
        }
        fn frame(&self, _: Option<u64>, _: Duration) -> Result<Frame, ControlError> {
            Err(ControlError::Timeout)
        }
        fn quit(&self) {}
    }

    fn key(key: Key, down: bool) -> ViewerInput {
        ViewerInput::Key { key, down }
    }

    #[test]
    fn the_trait_is_object_safe_and_shareable() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn Control>();
        let control: Arc<dyn Control> = Arc::new(Recorder::default());
        control.stroke(Key::Enter).unwrap();
        assert!(format!("{control:?}").contains("Recorder"));
    }

    #[test]
    fn typing_sends_characters_and_turns_newline_and_tab_into_keys() {
        let r = Recorder::default();
        r.type_text("a:1\nö\t").unwrap();
        assert_eq!(
            *r.inputs.lock().unwrap(),
            [
                ViewerInput::Text('a'),
                ViewerInput::Text(':'),
                ViewerInput::Text('1'),
                key(Key::Enter, true),
                key(Key::Enter, false),
                ViewerInput::Text('ö'),
                key(Key::Tab, true),
                key(Key::Tab, false),
            ]
        );
    }

    /// A control character is refused before anything is typed, so a text is
    /// typed whole or not at all.
    #[test]
    fn untypeable_text_is_refused_whole() {
        let r = Recorder::default();
        for bad in ["ab\u{1b}[2J", "x\0", "\r\n", "del\u{7f}"] {
            let err = r.type_text(bad).unwrap_err();
            assert!(matches!(err, ControlError::Invalid(_)), "{bad:?}: {err}");
        }
        assert!(r.inputs.lock().unwrap().is_empty());
        assert!(r
            .type_text("\u{1b}")
            .unwrap_err()
            .to_string()
            .contains("U+001B"));
    }

    #[test]
    fn turns_are_given_in_degrees_and_sent_in_radians() {
        let r = Recorder::default();
        r.turn_degrees(180.0, -90.0).unwrap();
        let sent = r.inputs.lock().unwrap()[0];
        let ViewerInput::Turn { dyaw, dpitch } = sent else {
            panic!("not a turn: {sent:?}");
        };
        assert!((dyaw - std::f32::consts::PI).abs() < 1e-6);
        assert!((dpitch + std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        for (yaw, pitch) in [(f64::NAN, 0.0), (0.0, f64::INFINITY), (1e300, 0.0)] {
            assert!(r.turn_degrees(yaw, pitch).is_err(), "{yaw} {pitch}");
        }
        assert_eq!(r.inputs.lock().unwrap().len(), 1);
    }

    #[test]
    fn every_key_has_names_that_find_it_again() {
        for key in Key::ALL {
            assert_eq!(key_from_name(key_name(key)), Some(key), "{key:?}");
        }
        for (name, key) in [
            ("w", Key::Forward),
            ("W", Key::Forward),
            ("forward", Key::Forward),
            ("space", Key::Up),
            ("SHIFT", Key::Down),
            ("left", Key::Left),
            ("Enter", Key::Enter),
            ("return", Key::Enter),
            ("esc", Key::Escape),
            ("del", Key::Delete),
            ("caret_left", Key::CaretLeft),
        ] {
            assert_eq!(key_from_name(name), Some(key), "{name}");
        }
        for name in ["", "x", "ww", "ctrl", "arrowleft", "enter "] {
            assert_eq!(key_from_name(name), None, "{name:?}");
        }
        // Each name belongs to exactly one key.
        let names = key_names();
        let mut all: Vec<&str> = names.split(' ').collect();
        let count = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), count, "a name is listed twice");
        assert!(count >= Key::COUNT);
    }

    #[test]
    fn the_state_is_a_flat_json_object_with_stable_names() {
        let mut state = ClientState {
            screen: Screen::Online,
            address: "127.0.0.1:7777".into(),
            shard: Some(2),
            session: Some(41),
            tick: 900,
            handoffs: 1,
            scene: Some("solar".into()),
            scene_revision: 3,
            frame: 12,
            width: 640,
            height: 360,
            fps: 59.5,
            ..ClientState::default()
        };
        state.set_ship([3.0, 4.0, 0.0], [0.0, 3.0, 4.0]);
        state.scene_position = Some([1.0, 2.0, -3.0]);
        state.gravity = 9.81;
        state.landed = true;
        state.bodies = vec![BodyState {
            guid: "1:7".into(),
            visual: "crate".into(),
            visual_id: 1,
            position: [12.0, -1.4, -1.6],
            distance: 2.5,
            speed: 0.25,
            parts: 1,
            in_view: true,
        }];
        state.bodies_in_view = 1;
        state.correction_m = 0.002;
        let json = serde_json::to_value(&state).unwrap();
        assert_eq!(json["gravity"], 9.81);
        assert_eq!(json["landed"], true);
        assert_eq!(json["bodies"][0]["visual"], "crate");
        assert_eq!(
            json["bodies"][0]["position"],
            serde_json::json!([12.0, -1.4, -1.6])
        );
        assert_eq!(json["bodies_in_view"], 1);
        assert_eq!(json["correction_m"], 0.002);
        assert_eq!(json["screen"], "online");
        assert_eq!(json["error"], serde_json::Value::Null);
        assert_eq!(json["address"], "127.0.0.1:7777");
        assert_eq!(json["shard"], 2);
        assert_eq!(json["session"], 41);
        assert_eq!(json["position"], serde_json::json!([3.0, 4.0, 0.0]));
        assert_eq!(json["scene_position"], serde_json::json!([1.0, 2.0, -3.0]));
        assert_eq!(json["speed"], 5.0);
        assert_eq!(json["scene"], "solar");
        assert_eq!(json["scene_revision"], 3);
        assert_eq!(json["retry_in_s"], serde_json::Value::Null);
        assert_eq!(json["frame"], 12);
        assert_eq!(json["width"], 640);
        let back: ClientState = serde_json::from_value(json).unwrap();
        assert_eq!(back, state);

        state.clear_ship();
        assert_eq!((state.position, state.speed), (None, 0.0));
        assert_eq!(state.scene_position, None);
        // A pending automatic attempt is a number of seconds.
        state.retry_in_s = Some(2.5);
        let json = serde_json::to_value(&state).unwrap();
        assert_eq!(json["retry_in_s"], 2.5);
        // The default snapshot: no scene, no retry.
        let fresh = serde_json::to_value(ClientState::default()).unwrap();
        assert_eq!(fresh["scene"], serde_json::Value::Null);
        assert_eq!(fresh["scene_position"], serde_json::Value::Null);
        assert_eq!(fresh["scene_revision"], 0);
        assert_eq!(fresh["retry_in_s"], serde_json::Value::Null);
        assert_eq!(fresh["bodies"], serde_json::json!([]));
        assert_eq!(
            (fresh["gravity"].as_f64(), fresh["landed"].as_bool()),
            (Some(0.0), Some(false))
        );
        // A state from before the physics fields still reads.
        let mut old = serde_json::to_value(ClientState::default()).unwrap();
        for field in [
            "gravity",
            "landed",
            "bodies",
            "bodies_in_view",
            "correction_m",
            "correction_max_m",
        ] {
            old.as_object_mut().unwrap().remove(field);
        }
        assert_eq!(
            serde_json::from_value::<ClientState>(old).unwrap(),
            ClientState::default()
        );
        for screen in [
            Screen::Connect,
            Screen::Connecting,
            Screen::Online,
            Screen::Error,
        ] {
            let json = serde_json::to_value(screen).unwrap();
            assert_eq!(json, screen.as_str());
            assert_eq!(screen.to_string(), screen.as_str());
        }
    }

    #[test]
    fn a_frame_checks_its_size_and_finds_its_pixels() {
        assert!(Frame::new(1, 0, 4, vec![]).is_none());
        assert!(Frame::new(1, 2, 2, vec![0; 15]).is_none());
        let rgba: Vec<u8> = (0..24).collect();
        let frame = Frame::new(7, 3, 2, rgba).unwrap();
        assert_eq!(frame.pixel(0, 0), Some([0, 1, 2, 3]));
        assert_eq!(frame.pixel(2, 1), Some([20, 21, 22, 23]));
        assert_eq!(frame.pixel(3, 0), None);
        assert_eq!(frame.pixel(0, 2), None);
        let debug = format!("{frame:?}");
        assert!(
            debug.contains("seq: 7") && debug.contains("bytes: 24"),
            "{debug}"
        );
    }
}
