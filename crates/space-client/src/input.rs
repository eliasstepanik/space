//! The one input path: what a [`ViewerInput`] means on the screen being shown.
//!
//! The window's keyboard and mouse, the browser viewer, the harness's HTTP API
//! and Lua scripts all produce [`ViewerInput`]s, and all of them end up in
//! [`Controls::handle`]. A sender cannot know which screen is up, so it does
//! not decide what a key press means: a printable key arrives both as its
//! movement key (if it has one) and as [`ViewerInput::Text`], and the screen
//! picks. On the connect screen text and the text-entry keys edit the address
//! and movement keys are ignored; while playing, movement keys fly the ship,
//! a few letters are commands and the rest of the text is ignored.
//!
//! # Brake assist
//!
//! The server's flight model is a spaceship's: six metres per second squared
//! of thrust and almost no drag, so a ship that is let go coasts for most of a
//! minute. That is right for crossing a solar system and useless for looking
//! at a crate from two metres away. With the brake assist on (the default),
//! letting go of every movement key makes the client thrust *against* the
//! ship's velocity until it stands still ([`brake`]) — the "flight computer"
//! a player would otherwise have to be. It is the client's choice of input,
//! nothing the server knows about; `B` switches it off for those who want to
//! coast.
//!
//! # Fast travel
//!
//! `F` switches **fast travel** on and off. Unlike the brake assist it is the
//! server's business: the flag rides in every input
//! (`space_protocol::PlayerInput::fast`) and the shard's flight model scales
//! the ship's speed limit with its distance to the nearest surface
//! (`space_sim::physics::fast_travel_limit`): a second's worth of the
//! distance (less 2 km of the surface being approached), so the Moon is about
//! half a minute away, while near anything (the demo area, a planet's ground)
//! it is normal flight. Letting
//! go of the keys in fast travel stops the ship by itself, so the brake
//! assist sends nothing while the ship is faster than its normal top speed;
//! switching fast travel off drops the ship to that top speed at once.

use space_stream::{Key, ViewerInput};

use crate::connect::{Prompt, PromptAction};

/// The view cannot pitch past this, radians (89 degrees): straight up or down
/// would make the camera's up vector degenerate.
pub const MAX_PITCH: f32 = 1.553_343;

/// Toggles between the chase camera and first person while playing.
pub const KEY_CAMERA: char = 'c';
/// Shows / hides the heads-up display while playing.
pub const KEY_HUD: char = 'h';
/// Switches the brake assist on / off while playing.
pub const KEY_BRAKE: char = 'b';
/// Switches fast travel on / off while playing.
pub const KEY_FAST: char = 'f';

/// The keys, for the hint line of the HUD and `--help`.
pub const KEY_HELP: &str = "W/A/S/D move  Space/Shift up/down  drag mouse: look  \
     C camera  B brake  F fast travel  H hud  Esc disconnect";

/// The normal top speed of a ship, metres/second (the placeholder flight
/// model's): above it the brake assist leaves a ship in fast travel alone.
pub const TOP_SPEED: f32 = 120.0;

/// Above this speed, metres/second, the brake assist brakes at full thrust;
/// below it the thrust falls off with the speed, so the ship eases to a stop
/// instead of hunting around it.
pub const BRAKE_FULL_SPEED: f32 = 1.5;

/// Below this speed, metres/second, the brake assist leaves the ship alone
/// (the server's own settling takes it from there).
const BRAKE_REST_SPEED: f32 = 1.0e-3;

/// The thrust (each axis in `-1..=1`, world frame) that brakes a ship moving
/// at `velocity` metres/second: straight against the velocity, at full power
/// above [`BRAKE_FULL_SPEED`] and proportionally less below. Zero for a ship
/// at rest (or a velocity that is not a number).
#[must_use]
pub fn brake(velocity: [f32; 3]) -> [f32; 3] {
    let speed = velocity.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !speed.is_finite() || speed < BRAKE_REST_SPEED {
        return [0.0; 3];
    }
    let power = (speed / BRAKE_FULL_SPEED).min(1.0);
    velocity.map(|v| -v / speed * power)
}

/// What the player holds and where the player looks.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputState {
    /// Yaw about +Y in radians (0 looks down -Z, positive turns left).
    pub yaw: f32,
    /// Pitch in radians (positive looks up).
    pub pitch: f32,
    /// Held movement keys, indexed by [`Key::index`].
    held: [bool; Key::MOVEMENT_COUNT],
    /// Last pointer position (normalised) and button mask, if any.
    pub pointer: Option<(f32, f32, u8)>,
}

impl InputState {
    /// Looking along `yaw`/`pitch` (radians), nothing held.
    #[must_use]
    pub fn looking(yaw: f32, pitch: f32) -> Self {
        Self {
            yaw: yaw.rem_euclid(std::f32::consts::TAU),
            pitch: pitch.clamp(-MAX_PITCH, MAX_PITCH),
            ..Self::default()
        }
    }

    /// A movement key went down or up (any other key is ignored).
    pub fn set_key(&mut self, key: Key, down: bool) {
        if let Some(held) = self.held.get_mut(key.index()) {
            *held = down;
        }
    }

    /// Turn the view by this many radians. The yaw wraps; the pitch stops
    /// short of straight up and down.
    pub fn turn(&mut self, dyaw: f32, dpitch: f32) {
        if !(dyaw.is_finite() && dpitch.is_finite()) {
            return;
        }
        self.yaw = (self.yaw + dyaw).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + dpitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Release every held movement key (whoever held them can no longer send
    /// the key-ups: a viewer that vanished, a window that lost the focus).
    pub fn release_keys(&mut self) {
        self.held = [false; Key::MOVEMENT_COUNT];
    }

    /// Whether `key` is held.
    #[must_use]
    pub fn is_held(&self, key: Key) -> bool {
        self.held.get(key.index()).copied().unwrap_or(false)
    }

    /// Whether any movement key is held.
    #[must_use]
    pub fn any_held(&self) -> bool {
        self.held.iter().any(|held| *held)
    }

    /// Unit view direction.
    #[must_use]
    pub fn forward(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        [-sy * cp, sp, -cy * cp]
    }

    /// Unit vector to the right of the view, in the horizontal plane.
    #[must_use]
    pub fn right(&self) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        [cy, 0.0, -sy]
    }

    /// Thrust intent for the server, each axis in `-1..=1`, in the world
    /// frame: the held keys along the view's forward and right axes, plus
    /// up/down, normalised so diagonals are not faster. The client never
    /// torques the ship, so the ship's body frame stays the world frame and
    /// the view direction only picks the thrust direction.
    #[must_use]
    pub fn thrust(&self) -> [f32; 3] {
        let axis = |pos: Key, neg: Key| {
            f32::from(i8::from(self.is_held(pos)) - i8::from(self.is_held(neg)))
        };
        let (fwd, right, up) = (
            axis(Key::Forward, Key::Back),
            axis(Key::Right, Key::Left),
            axis(Key::Up, Key::Down),
        );
        let (f, r) = (self.forward(), self.right());
        let mut t = [0.0; 3];
        for i in 0..3 {
            t[i] = fwd * f[i] + right * r[i];
        }
        t[1] += up;
        let len = t.iter().map(|v| v * v).sum::<f32>().sqrt();
        if len > 1.0 {
            t = t.map(|v| v / len);
        }
        t
    }
}

/// Where the camera is while playing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CameraMode {
    /// Behind and above the player's own character.
    #[default]
    Chase,
    /// At the ship itself; the player's own character is not drawn.
    FirstPerson,
}

impl CameraMode {
    /// The other mode.
    #[must_use]
    pub const fn toggled(self) -> Self {
        match self {
            Self::Chase => Self::FirstPerson,
            Self::FirstPerson => Self::Chase,
        }
    }

    /// Lower-case name for the HUD.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Chase => "chase",
            Self::FirstPerson => "first person",
        }
    }
}

/// Which screen an input arrives on (what decides its meaning).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScreenKind {
    /// The connect screen (with or without an error line).
    Connect,
    /// A connection attempt is in flight.
    Connecting,
    /// Playing on a shard.
    Online,
}

/// Something an input asks the application to do (everything else an input
/// can do is a change to the [`Controls`] themselves).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Request {
    /// Connect to the address in the prompt.
    Connect,
    /// Give up the connection attempt, back to the connect screen.
    Cancel,
    /// Leave the shard, back to the connect screen.
    Disconnect,
    /// Esc on the connect screen: step back. The application stops its
    /// retrying, or — where the front end lets its user do that
    /// ([`crate::app::AppConfig::escape_quits`]) — ends.
    Back,
}

/// Everything input changes directly: the address prompt, the view and held
/// keys, the camera mode and the HUD and brake-assist switches.
#[derive(Clone, Debug, PartialEq)]
pub struct Controls {
    /// The connect screen's address field.
    pub prompt: Prompt,
    /// Held movement keys and the view direction.
    pub view: InputState,
    /// Chase camera or first person.
    pub camera: CameraMode,
    /// Whether the heads-up display is drawn.
    pub hud: bool,
    /// Whether the ship is braked while no movement key is held.
    pub brake_assist: bool,
    /// Whether fast travel is on (sent with every input; see the module
    /// docs).
    pub fast: bool,
}

impl Controls {
    /// Controls whose prompt holds `prefill`: chase camera, HUD on, brake
    /// assist on.
    #[must_use]
    pub fn new(prefill: &str) -> Self {
        Self {
            prompt: Prompt::new(prefill),
            view: InputState::default(),
            camera: CameraMode::default(),
            hud: true,
            brake_assist: true,
            fast: false,
        }
    }

    /// The thrust to send for a ship moving at `velocity` (metres/second):
    /// what the held keys ask for, or — with the brake assist on and no key
    /// held — whatever stops the ship.
    #[must_use]
    pub fn thrust(&self, velocity: Option<[f32; 3]>) -> [f32; 3] {
        match velocity {
            // Fast travel stops a ship that is let go by itself.
            Some(velocity)
                if self.fast && velocity.iter().map(|v| v * v).sum::<f32>() > TOP_SPEED.powi(2) =>
            {
                self.view.thrust()
            }
            Some(velocity) if self.brake_assist && !self.view.any_held() => brake(velocity),
            _ => self.view.thrust(),
        }
    }

    /// Apply one input event as the screen `screen` understands it.
    pub fn handle(&mut self, screen: ScreenKind, input: ViewerInput) -> Option<Request> {
        match screen {
            ScreenKind::Connect => self.on_connect_screen(input),
            ScreenKind::Connecting => match input {
                ViewerInput::Key {
                    key: Key::Escape,
                    down: true,
                } => Some(Request::Cancel),
                _ => None,
            },
            ScreenKind::Online => self.while_playing(input),
        }
    }

    fn on_connect_screen(&mut self, input: ViewerInput) -> Option<Request> {
        match input {
            ViewerInput::Text(c) => {
                self.prompt.insert(c);
                None
            }
            // Text-entry keys are strokes: the down acts, the up is ignored.
            ViewerInput::Key { key, down: true } if key.is_text_entry() => {
                self.prompt.stroke(key).map(|action| match action {
                    PromptAction::Submit => Request::Connect,
                    PromptAction::Back => Request::Back,
                })
            }
            // Movement keys (W, A, S, D, Space, ...) type their letter here,
            // which arrives as `Text`; there is nothing to fly or look at.
            ViewerInput::Key { .. } | ViewerInput::Turn { .. } | ViewerInput::Pointer { .. } => {
                None
            }
        }
    }

    fn while_playing(&mut self, input: ViewerInput) -> Option<Request> {
        match input {
            ViewerInput::Key { key, down } if key.is_movement() => {
                self.view.set_key(key, down);
                None
            }
            ViewerInput::Key {
                key: Key::Escape,
                down: true,
            } => Some(Request::Disconnect),
            ViewerInput::Key { .. } => None,
            ViewerInput::Turn { dyaw, dpitch } => {
                self.view.turn(dyaw, dpitch);
                None
            }
            ViewerInput::Pointer { x, y, buttons } => {
                self.view.pointer = Some((x, y, buttons));
                None
            }
            ViewerInput::Text(c) => {
                match c.to_ascii_lowercase() {
                    KEY_CAMERA => self.camera = self.camera.toggled(),
                    KEY_HUD => self.hud = !self.hud,
                    KEY_BRAKE => self.brake_assist = !self.brake_assist,
                    KEY_FAST => self.fast = !self.fast,
                    // W/A/S/D arrive as text too; they are the movement keys.
                    _ => {}
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(key: Key, down: bool) -> ViewerInput {
        ViewerInput::Key { key, down }
    }

    fn turn(dyaw: f32, dpitch: f32) -> ViewerInput {
        ViewerInput::Turn { dyaw, dpitch }
    }

    #[test]
    fn thrust_follows_held_keys_and_view_and_is_normalised() {
        let mut s = InputState::default();
        assert_eq!(s.thrust(), [0.0; 3]);
        assert!(!s.any_held());
        s.set_key(Key::Forward, true);
        assert!(s.any_held() && s.is_held(Key::Forward));
        let t = s.thrust();
        assert!((t[2] + 1.0).abs() < 1e-6 && t[0].abs() < 1e-6, "{t:?}");
        s.set_key(Key::Right, true);
        s.set_key(Key::Up, true);
        let len = s.thrust().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((len - 1.0).abs() < 1e-5, "diagonal thrust is unit: {len}");
        s.release_keys();
        assert_eq!(s.thrust(), [0.0; 3]);
        // Turned left a quarter: forward is -X, right is -Z.
        s.turn(std::f32::consts::FRAC_PI_2, 0.0);
        s.set_key(Key::Forward, true);
        let t = s.thrust();
        assert!((t[0] + 1.0).abs() < 1e-6 && t[2].abs() < 1e-6, "{t:?}");
        // A text-entry key is not something that can be held.
        s.set_key(Key::Enter, true);
        assert!(!s.is_held(Key::Enter));
    }

    #[test]
    fn the_view_wraps_in_yaw_and_stops_in_pitch() {
        let mut s = InputState::default();
        s.turn(0.0, 9.0);
        assert!(s.pitch < std::f32::consts::FRAC_PI_2 && s.pitch > 1.5);
        let mut t = InputState::default();
        for _ in 0..5 {
            t.turn(2.0, -0.1);
        }
        assert!(
            (t.yaw - (10.0 - std::f32::consts::TAU)).abs() < 1e-5,
            "{}",
            t.yaw
        );
        assert!((t.pitch + 0.5).abs() < 1e-6);
        // Nonsense is dropped, not stored.
        t.turn(f32::NAN, 0.0);
        t.turn(0.0, f32::INFINITY);
        assert!(t.yaw.is_finite() && (t.pitch + 0.5).abs() < 1e-6);
        let g = InputState::looking(std::f32::consts::PI, 0.0);
        assert!(g.forward()[2] > 0.99, "{:?}", g.forward());
        assert!(InputState::looking(0.0, 7.0).pitch <= MAX_PITCH);
    }

    /// On the connect screen typing edits the address and nothing flies: the
    /// movement half of a W press is ignored, its text half is typed.
    #[test]
    fn on_the_connect_screen_text_is_typed_and_movement_is_ignored() {
        let mut c = Controls::new("");
        let screen = ScreenKind::Connect;
        // What a browser viewer sends for the W key: both meanings.
        assert_eq!(c.handle(screen, key(Key::Forward, true)), None);
        assert_eq!(c.handle(screen, ViewerInput::Text('w')), None);
        assert_eq!(c.handle(screen, key(Key::Forward, false)), None);
        assert_eq!(c.prompt.text(), "w");
        assert!(!c.view.any_held(), "nothing is held on the connect screen");
        // The command letters are just letters here.
        c.handle(screen, ViewerInput::Text('c'));
        c.handle(screen, ViewerInput::Text('h'));
        c.handle(screen, ViewerInput::Text('b'));
        c.handle(screen, key(Key::Backspace, true));
        c.handle(screen, ViewerInput::Text('f'));
        c.handle(screen, key(Key::Backspace, true));
        assert_eq!(c.prompt.text(), "wch");
        assert_eq!(
            (c.camera, c.hud, c.brake_assist, c.fast),
            (CameraMode::Chase, true, true, false)
        );
        // The arrow keys: the movement half is ignored, the caret half moves.
        c.handle(screen, key(Key::Left, true));
        c.handle(screen, key(Key::CaretLeft, true));
        c.handle(screen, key(Key::CaretLeft, false));
        assert_eq!(c.prompt.caret(), 2);
        c.handle(screen, key(Key::Backspace, true));
        assert_eq!(c.prompt.text(), "wh");
        // Looking around does nothing.
        c.handle(screen, turn(1.0, 0.5));
        assert_eq!((c.view.yaw, c.view.pitch), (0.0, 0.0));
        // Enter asks to connect, Escape to step back; the key-ups ask for
        // nothing.
        assert_eq!(
            c.handle(screen, key(Key::Enter, true)),
            Some(Request::Connect)
        );
        assert_eq!(c.handle(screen, key(Key::Enter, false)), None);
        assert_eq!(
            c.handle(screen, key(Key::Escape, true)),
            Some(Request::Back)
        );
        assert_eq!(c.handle(screen, key(Key::Escape, false)), None);
    }

    /// While playing, movement keys are held, the mouse turns the view, C and
    /// H are commands, other text is ignored and Escape disconnects.
    #[test]
    fn while_playing_keys_fly_and_text_is_commands() {
        let mut c = Controls::new("127.0.0.1:7777");
        let screen = ScreenKind::Online;
        c.handle(screen, key(Key::Forward, true));
        c.handle(screen, ViewerInput::Text('w'));
        assert!(c.view.is_held(Key::Forward));
        assert_eq!(c.prompt.text(), "127.0.0.1:7777", "nothing is typed");
        c.handle(screen, key(Key::Forward, false));
        assert!(!c.view.any_held());

        c.handle(screen, turn(0.5, -0.25));
        assert!((c.view.yaw - 0.5).abs() < 1e-6 && (c.view.pitch + 0.25).abs() < 1e-6);
        c.handle(
            screen,
            ViewerInput::Pointer {
                x: 0.5,
                y: 0.25,
                buttons: 1,
            },
        );
        assert_eq!(c.view.pointer, Some((0.5, 0.25, 1)));

        assert_eq!(c.handle(screen, ViewerInput::Text('c')), None);
        assert_eq!(c.camera, CameraMode::FirstPerson);
        c.handle(screen, ViewerInput::Text('C'));
        assert_eq!(c.camera, CameraMode::Chase);
        c.handle(screen, ViewerInput::Text('h'));
        assert!(!c.hud);
        c.handle(screen, ViewerInput::Text('H'));
        assert!(c.hud);
        c.handle(screen, ViewerInput::Text('b'));
        assert!(!c.brake_assist);
        c.handle(screen, ViewerInput::Text('B'));
        assert!(c.brake_assist);

        // Text-entry keys other than Escape do nothing while playing.
        for k in [
            Key::Enter,
            Key::Backspace,
            Key::Tab,
            Key::Home,
            Key::CaretLeft,
        ] {
            assert_eq!(c.handle(screen, key(k, true)), None, "{k:?}");
        }
        assert_eq!(c.prompt.text(), "127.0.0.1:7777");
        assert_eq!(c.handle(screen, key(Key::Escape, false)), None);
        assert_eq!(
            c.handle(screen, key(Key::Escape, true)),
            Some(Request::Disconnect)
        );
    }

    #[test]
    fn while_connecting_only_escape_does_anything() {
        let mut c = Controls::new("host:1");
        let before = c.clone();
        let screen = ScreenKind::Connecting;
        for input in [
            ViewerInput::Text('x'),
            key(Key::Forward, true),
            key(Key::Enter, true),
            key(Key::Backspace, true),
            turn(1.0, 1.0),
        ] {
            assert_eq!(c.handle(screen, input), None, "{input:?}");
        }
        assert_eq!(c, before);
        assert_eq!(
            c.handle(screen, key(Key::Escape, true)),
            Some(Request::Cancel)
        );
        assert_eq!(c.handle(screen, key(Key::Escape, false)), None);
    }

    /// The brake assist thrusts against the velocity — fully when fast, gently
    /// when slow, not at all at rest — and only while no key is held.
    #[test]
    fn the_brake_assist_stops_a_ship_that_is_let_go() {
        assert_eq!(brake([0.0; 3]), [0.0; 3]);
        assert_eq!(brake([0.0, 0.0, 1.0e-4]), [0.0; 3], "as good as at rest");
        assert_eq!(brake([f32::NAN, 0.0, 0.0]), [0.0; 3]);
        // Fast: full thrust, straight back.
        let fast = brake([30.0, 0.0, -40.0]);
        assert!(
            (fast[0] + 0.6).abs() < 1e-6 && (fast[2] - 0.8).abs() < 1e-6,
            "{fast:?}"
        );
        // Slow: proportionally less, same direction.
        let slow = brake([0.0, BRAKE_FULL_SPEED / 2.0, 0.0]);
        assert!((slow[1] + 0.5).abs() < 1e-6, "{slow:?}");
        // It never asks for more than full thrust and always opposes.
        for v in [[0.3, -0.2, 0.1], [5.0, 5.0, 5.0], [0.0, -120.0, 0.0]] {
            let t = brake(v);
            let len = t.iter().map(|c| c * c).sum::<f32>().sqrt();
            let dot: f32 = t.iter().zip(v).map(|(a, b)| a * b).sum();
            assert!(len <= 1.0 + 1e-6 && dot < 0.0, "{v:?} -> {t:?}");
        }

        let mut c = Controls::new("");
        let moving = Some([4.0, 0.0, 0.0]);
        assert_eq!(c.thrust(moving), [-1.0, 0.0, 0.0], "let go: brake");
        assert_eq!(c.thrust(Some([0.0; 3])), [0.0; 3]);
        assert_eq!(c.thrust(None), [0.0; 3], "no ship yet: nothing to brake");
        // A held key is the player's word: no braking against it.
        c.view.set_key(Key::Forward, true);
        assert_eq!(c.thrust(moving), c.view.thrust());
        c.view.release_keys();
        // Switched off, the ship coasts.
        c.brake_assist = false;
        assert_eq!(c.thrust(moving), [0.0; 3]);
    }

    #[test]
    fn the_camera_modes_toggle_and_have_names() {
        assert_eq!(CameraMode::default(), CameraMode::Chase);
        assert_eq!(CameraMode::Chase.toggled(), CameraMode::FirstPerson);
        assert_eq!(CameraMode::FirstPerson.toggled(), CameraMode::Chase);
        assert_ne!(CameraMode::Chase.name(), CameraMode::FirstPerson.name());
        assert!(KEY_HELP.contains("C camera") && KEY_HELP.contains("Esc"));
        assert!(KEY_HELP.contains("B brake") && !KEY_HELP.contains('\n'));
        assert!(KEY_HELP.contains("F fast travel"));
    }

    /// `F` toggles fast travel while playing; a ship faster than its normal
    /// top speed in fast travel is not braked (fast travel stops it), a slow
    /// one is, as ever.
    #[test]
    fn f_toggles_fast_travel_and_the_brake_assist_leaves_a_fast_ship_alone() {
        let mut c = Controls::new("");
        let screen = ScreenKind::Online;
        assert!(!c.fast);
        c.handle(screen, ViewerInput::Text('f'));
        assert!(c.fast);
        assert_eq!(c.thrust(Some([0.0, 0.0, -5_000.0])), [0.0; 3]);
        assert_ne!(c.thrust(Some([0.0, 0.0, -5.0])), [0.0; 3], "braked");
        c.handle(screen, ViewerInput::Text('F'));
        assert!(!c.fast);
        assert_ne!(c.thrust(Some([0.0, 0.0, -5_000.0])), [0.0; 3], "braked");
    }
}
