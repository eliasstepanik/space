//! [`FakeClient`]: a stand-in client that lives entirely in memory.
//!
//! It is a small state machine behind a [`Hub`], driven by a thread that plays
//! the render loop: it drains requests, steps a toy simulation and publishes
//! state and synthetic frames, the way a real client would. Tests of the Lua
//! API and of the HTTP API run against it; so can tests of code that uses a
//! [`Control`].
//!
//! What it models:
//!
//! * **Connect prompt** ([`Screen::Connect`]): typed characters edit an address
//!   field (Backspace, Delete, Home, End, the caret keys and Escape — clear —
//!   work as in any text field); Enter connects to what is typed.
//!   [`Control::connect`] does the same in one step.
//! * **Connecting**: lasts [`FakeOptions::connect_frames`] frames, then becomes
//!   **online** if the address is one of [`FakeOptions::servers`], else the
//!   **error** screen (`cannot reach ADDRESS: connection refused`), which Enter
//!   or Escape dismiss.
//! * **Online**: the movement keys thrust the ship along the view
//!   ([`ACCELERATION`] m/s², [`TIME_STEP`] s per frame), `Turn` rotates the
//!   view, the tick counts frames, and crossing `z = -`[`HANDOFF_DISTANCE`]
//!   moves the ship to shard 2 (one handoff, a new session).
//! * **Frames**: [`picture`], a pattern that depends on the screen and the
//!   frame number, so a test can tell exactly which frame it was given.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::control::{ClientState, Control, Frame, Key, Screen, ViewerInput};
use crate::hub::{Command, Hub};

/// Thrust acceleration of the fake ship, m/s².
pub const ACCELERATION: f64 = 30.0;
/// Simulated time per frame, seconds (whatever the wall-clock frame interval).
pub const TIME_STEP: f64 = 1.0 / 60.0;
/// The ship changes shard when it has flown this far forward (towards -Z), m.
pub const HANDOFF_DISTANCE: f64 = 50.0;
/// The view cannot pitch past this, degrees.
pub const MAX_PITCH_DEG: f32 = 89.0;

/// How a [`FakeClient`] behaves.
#[derive(Clone, Debug)]
pub struct FakeOptions {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Wall-clock time between frames.
    pub frame_interval: Duration,
    /// The server addresses that exist; connecting to any other fails.
    pub servers: Vec<String>,
    /// How many frames a connection attempt takes.
    pub connect_frames: u64,
    /// Publish every frame (a headless client that has the pixels anyway)
    /// instead of only when one is wanted (a window that reads back on
    /// demand).
    pub publish_every_frame: bool,
}

impl Default for FakeOptions {
    /// 64×48 frames every 2 ms, one server at `127.0.0.1:7777`, five frames to
    /// connect, frames on demand.
    fn default() -> Self {
        Self {
            width: 64,
            height: 48,
            frame_interval: Duration::from_millis(2),
            servers: vec!["127.0.0.1:7777".to_owned()],
            connect_frames: 5,
            publish_every_frame: false,
        }
    }
}

/// What a [`FakeClient`] was asked to do so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    /// Every input event, in order.
    pub inputs: Vec<ViewerInput>,
    /// Every address a connection was attempted to, in order (typed + Enter,
    /// or [`Control::connect`]).
    pub connects: Vec<String>,
    /// How often a disconnect was requested.
    pub disconnects: u32,
    /// How many frames were handed to the hub.
    pub frames_published: u64,
}

impl Record {
    /// The typed characters, in order.
    #[must_use]
    pub fn typed(&self) -> String {
        self.inputs
            .iter()
            .filter_map(|input| match input {
                ViewerInput::Text(c) => Some(*c),
                _ => None,
            })
            .collect()
    }

    /// The key transitions, in order.
    #[must_use]
    pub fn keys(&self) -> Vec<(Key, bool)> {
        self.inputs
            .iter()
            .filter_map(|input| match input {
                ViewerInput::Key { key, down } => Some((*key, *down)),
                _ => None,
            })
            .collect()
    }
}

/// The synthetic picture of frame `seq` on `screen`: pixel `(x, y)` is
/// `[(x + seq) mod 256, y mod 256, c, 255]` with `c` = 32, 96, 160, 224 for the
/// connect, connecting, online and error screens.
#[must_use]
pub fn picture(width: u32, height: u32, screen: Screen, seq: u64) -> Vec<u8> {
    let code = match screen {
        Screen::Connect => 32,
        Screen::Connecting => 96,
        Screen::Online => 160,
        Screen::Error => 224,
    };
    let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            rgba.extend_from_slice(&[(u64::from(x) + seq) as u8, y as u8, code, 255]);
        }
    }
    rgba
}

#[derive(Debug, Default)]
struct Shared {
    stop: AtomicBool,
    paused: AtomicBool,
    quit: AtomicBool,
    record: Mutex<Record>,
}

/// A fake client running on its own thread until dropped (or told to quit).
#[derive(Debug)]
pub struct FakeClient {
    hub: Arc<Hub>,
    shared: Arc<Shared>,
    thread: Option<JoinHandle<()>>,
}

impl FakeClient {
    /// Start a fake client showing its connect prompt.
    ///
    /// # Panics
    ///
    /// If the thread cannot be started, or a frame dimension is zero.
    #[must_use]
    pub fn start(options: FakeOptions) -> Self {
        assert!(
            options.width > 0 && options.height > 0,
            "a frame needs pixels"
        );
        let hub = Hub::new();
        let shared = Arc::new(Shared::default());
        let thread = {
            let (hub, shared) = (Arc::clone(&hub), Arc::clone(&shared));
            std::thread::Builder::new()
                .name("fake-client".into())
                .spawn(move || run(&hub, &shared, &options))
                .expect("cannot start the fake client's thread")
        };
        Self {
            hub,
            shared,
            thread: Some(thread),
        }
    }

    /// The control surface, as front ends take it.
    #[must_use]
    pub fn control(&self) -> Arc<dyn Control> {
        Arc::clone(&self.hub) as Arc<dyn Control>
    }

    /// The hub behind [`control`](Self::control).
    #[must_use]
    pub fn hub(&self) -> &Arc<Hub> {
        &self.hub
    }

    /// Freeze (`true`) or resume the render loop. A frozen client takes no
    /// input, renders nothing and publishes nothing: what a client looks like
    /// from outside when its loop hangs.
    pub fn set_paused(&self, paused: bool) {
        self.shared.paused.store(paused, Ordering::Relaxed);
    }

    /// Everything asked of the client so far.
    #[must_use]
    pub fn record(&self) -> Record {
        self.shared
            .record
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Whether the client exited because it was asked to quit.
    #[must_use]
    pub fn has_quit(&self) -> bool {
        self.shared.quit.load(Ordering::Relaxed)
    }

    /// Wait (at most `timeout`) until `condition` holds for the published
    /// state; returns whether it did.
    pub fn wait_for(&self, timeout: Duration, condition: impl Fn(&ClientState) -> bool) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if condition(&self.hub.state()) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

impl Drop for FakeClient {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The simulated client.
struct Sim {
    servers: Vec<String>,
    connect_frames: u64,
    state: ClientState,
    /// The address field of the connect prompt, and the caret in it.
    field: Vec<char>,
    caret: usize,
    held: [bool; Key::COUNT],
    /// Radians.
    yaw: f32,
    pitch: f32,
    position: [f64; 3],
    velocity: [f64; 3],
    /// The address being connected to and the frames left until the answer.
    connecting: Option<(String, u64)>,
    sessions: u64,
    quit: bool,
}

impl Sim {
    fn new(options: &FakeOptions) -> Self {
        Self {
            servers: options.servers.clone(),
            connect_frames: options.connect_frames,
            state: ClientState {
                width: options.width,
                height: options.height,
                fps: (1.0 / options.frame_interval.as_secs_f64().max(1e-6)) as f32,
                ..ClientState::default()
            },
            field: Vec::new(),
            caret: 0,
            held: [false; Key::COUNT],
            yaw: 0.0,
            pitch: 0.0,
            position: [0.0; 3],
            velocity: [0.0; 3],
            connecting: None,
            sessions: 0,
            quit: false,
        }
    }

    fn apply(&mut self, command: Command, record: &mut Record) {
        match command {
            Command::Input(input) => {
                record.inputs.push(input);
                self.input(input, record);
            }
            Command::Connect(address) => {
                self.field = address.chars().collect();
                self.caret = self.field.len();
                self.begin_connect(record);
            }
            Command::Disconnect => {
                record.disconnects += 1;
                self.show_prompt();
            }
            Command::Quit => self.quit = true,
        }
    }

    fn input(&mut self, input: ViewerInput, record: &mut Record) {
        match input {
            ViewerInput::Key { key, down } => {
                self.held[key.index()] = down;
                if down && key.is_text_entry() {
                    self.stroke(key, record);
                }
            }
            ViewerInput::Text(c) => {
                if self.state.screen == Screen::Connect {
                    self.field.insert(self.caret, c);
                    self.caret += 1;
                }
            }
            ViewerInput::Turn { dyaw, dpitch } => {
                let limit = MAX_PITCH_DEG.to_radians();
                self.yaw = (self.yaw + dyaw).rem_euclid(std::f32::consts::TAU);
                self.pitch = (self.pitch + dpitch).clamp(-limit, limit);
            }
            ViewerInput::Pointer { .. } => {}
        }
    }

    /// A text-entry key went down.
    fn stroke(&mut self, key: Key, record: &mut Record) {
        match self.state.screen {
            Screen::Connect => match key {
                Key::Enter if !self.field.is_empty() => self.begin_connect(record),
                Key::Backspace if self.caret > 0 => {
                    self.caret -= 1;
                    self.field.remove(self.caret);
                }
                Key::Delete if self.caret < self.field.len() => {
                    self.field.remove(self.caret);
                }
                Key::CaretLeft => self.caret = self.caret.saturating_sub(1),
                Key::CaretRight => self.caret = (self.caret + 1).min(self.field.len()),
                Key::Home => self.caret = 0,
                Key::End => self.caret = self.field.len(),
                Key::Escape => {
                    self.field.clear();
                    self.caret = 0;
                }
                _ => {}
            },
            Screen::Error if matches!(key, Key::Enter | Key::Escape) => self.show_prompt(),
            _ => {}
        }
    }

    fn begin_connect(&mut self, record: &mut Record) {
        let address: String = self.field.iter().collect();
        record.connects.push(address.clone());
        self.leave_server();
        self.state.screen = Screen::Connecting;
        self.connecting = Some((address, self.connect_frames));
    }

    fn show_prompt(&mut self) {
        self.leave_server();
        self.state.screen = Screen::Connect;
    }

    fn leave_server(&mut self) {
        self.connecting = None;
        self.state.error = None;
        self.state.shard = None;
        self.state.session = None;
        self.state.tick = 0;
        self.state.handoffs = 0;
        self.state.scene = None;
        self.state.scene_revision = 0;
        self.state.clear_ship();
        self.position = [0.0; 3];
        self.velocity = [0.0; 3];
    }

    /// One frame of simulated time.
    fn step(&mut self) {
        self.state.frame += 1;
        if let Some((address, left)) = self.connecting.take() {
            if left > 1 {
                self.connecting = Some((address, left - 1));
            } else if self.servers.contains(&address) {
                self.sessions += 1;
                self.state.screen = Screen::Online;
                self.state.shard = Some(1);
                self.state.session = Some(self.sessions);
                // What a shard nobody configured shows.
                self.state.scene = Some("demo-area".to_owned());
                self.state.scene_revision = 1;
            } else {
                self.state.screen = Screen::Error;
                self.state.error = Some(format!("cannot reach {address}: connection refused"));
            }
        }
        if self.state.screen == Screen::Online {
            self.fly();
        }
        self.state.address = self.field.iter().collect();
        self.state.yaw_deg = self.yaw.to_degrees();
        self.state.pitch_deg = self.pitch.to_degrees();
    }

    fn fly(&mut self) {
        self.state.tick += 1;
        let axis = |positive: Key, negative: Key| {
            f64::from(i8::from(self.held[positive.index()]) - i8::from(self.held[negative.index()]))
        };
        let (forward, right, up) = (
            axis(Key::Forward, Key::Back),
            axis(Key::Right, Key::Left),
            axis(Key::Up, Key::Down),
        );
        let (yaw, pitch) = (f64::from(self.yaw), f64::from(self.pitch));
        // Yaw 0 looks down -Z; positive yaw turns left, positive pitch up.
        let ahead = [
            -yaw.sin() * pitch.cos(),
            pitch.sin(),
            -yaw.cos() * pitch.cos(),
        ];
        let side = [yaw.cos(), 0.0, -yaw.sin()];
        for i in 0..3 {
            let mut thrust = forward * ahead[i] + right * side[i];
            if i == 1 {
                thrust += up;
            }
            self.velocity[i] += thrust * ACCELERATION * TIME_STEP;
            self.position[i] += self.velocity[i] * TIME_STEP;
        }
        self.state.set_ship(self.position, self.velocity);
        // The fake's world begins at its scene's spawn point.
        self.state.scene_position = Some(self.position);
        let shard = if self.position[2] < -HANDOFF_DISTANCE {
            2
        } else {
            1
        };
        if self.state.shard != Some(shard) {
            self.sessions += 1;
            self.state.shard = Some(shard);
            self.state.session = Some(self.sessions);
            self.state.handoffs += 1;
        }
    }
}

/// The fake render loop.
fn run(hub: &Hub, shared: &Shared, options: &FakeOptions) {
    let mut sim = Sim::new(options);
    hub.publish_state(sim.state.clone());
    while !shared.stop.load(Ordering::Relaxed) {
        if shared.paused.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        let commands = hub.drain();
        let mut record = shared
            .record
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for command in commands {
            sim.apply(command, &mut record);
        }
        sim.step();
        if options.publish_every_frame || hub.frame_wanted() {
            let seq = sim.state.frame;
            let rgba = picture(options.width, options.height, sim.state.screen, seq);
            if let Some(frame) = Frame::new(seq, options.width, options.height, rgba) {
                hub.publish_frame(frame);
                record.frames_published += 1;
            }
        }
        drop(record);
        hub.publish_state(sim.state.clone());
        if sim.quit {
            shared.quit.store(true, Ordering::Relaxed);
            break;
        }
        std::thread::sleep(options.frame_interval);
    }
    hub.close();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::ControlError;

    const LONG: Duration = Duration::from_secs(20);

    fn online(fake: &FakeClient) {
        fake.control().connect("127.0.0.1:7777").unwrap();
        assert!(fake.wait_for(LONG, |s| s.screen == Screen::Online));
    }

    #[test]
    fn typing_an_address_and_enter_connects() {
        let fake = FakeClient::start(FakeOptions::default());
        let control = fake.control();
        assert_eq!(control.state().screen, Screen::Connect);
        // A typo, fixed the way a user would: End, Backspace, caret keys.
        control.type_text("127.0.0.1:777x").unwrap();
        control.stroke(Key::Backspace).unwrap();
        control.type_text("7").unwrap();
        control.stroke(Key::Home).unwrap();
        control.stroke(Key::Delete).unwrap();
        control.type_text("1").unwrap();
        control.stroke(Key::CaretRight).unwrap();
        control.stroke(Key::CaretLeft).unwrap();
        control.stroke(Key::End).unwrap();
        assert!(fake.wait_for(LONG, |s| s.address == "127.0.0.1:7777"));
        control.stroke(Key::Enter).unwrap();
        assert!(fake.wait_for(LONG, |s| s.screen == Screen::Online));
        let state = control.state();
        assert_eq!((state.shard, state.session), (Some(1), Some(1)));
        assert_eq!(state.position, Some([0.0; 3]));
        assert_eq!(fake.record().connects, ["127.0.0.1:7777"]);
    }

    #[test]
    fn an_unknown_server_ends_on_the_error_screen_which_enter_dismisses() {
        let fake = FakeClient::start(FakeOptions::default());
        let control = fake.control();
        control.connect("10.0.0.1:1").unwrap();
        assert!(fake.wait_for(LONG, |s| s.screen == Screen::Error));
        let state = control.state();
        assert_eq!(
            state.error.as_deref(),
            Some("cannot reach 10.0.0.1:1: connection refused")
        );
        assert_eq!(state.session, None);
        control.stroke(Key::Enter).unwrap();
        assert!(fake.wait_for(LONG, |s| s.screen == Screen::Connect && s.error.is_none()));
        // The address is still in the field; Escape clears it.
        assert_eq!(control.state().address, "10.0.0.1:1");
        control.stroke(Key::Escape).unwrap();
        assert!(fake.wait_for(LONG, |s| s.address.is_empty()));
    }

    #[test]
    fn thrust_moves_the_ship_and_flying_far_enough_changes_shard() {
        let fake = FakeClient::start(FakeOptions::default());
        let control = fake.control();
        online(&fake);
        // Typed text is not movement, and does nothing while online.
        control.type_text("w").unwrap();
        assert!(fake.wait_for(LONG, |s| s.tick > 5));
        assert_eq!(control.state().speed, 0.0);

        control.key(Key::Forward, true).unwrap();
        assert!(fake.wait_for(LONG, |s| s.speed > 1.0));
        assert!(fake.wait_for(LONG, |s| s.handoffs == 1));
        control.key(Key::Forward, false).unwrap();
        let state = control.state();
        let (position, velocity) = (state.position.unwrap(), state.velocity.unwrap());
        assert!(position[2] < -HANDOFF_DISTANCE, "{position:?}");
        assert!(
            velocity[2] < 0.0 && velocity[0].abs() < 1e-9,
            "{velocity:?}"
        );
        assert_eq!((state.shard, state.session), (Some(2), Some(2)));
        assert_eq!(state.scene_position, state.position);
        assert_eq!(
            (
                state.scene.as_deref(),
                state.scene_revision,
                state.retry_in_s
            ),
            (Some("demo-area"), 1, None)
        );

        // Turn left a quarter and strafe up: the view is reported in degrees.
        control.turn_degrees(90.0, 200.0).unwrap();
        assert!(fake.wait_for(LONG, |s| (s.yaw_deg - 90.0).abs() < 1e-3));
        assert!((control.state().pitch_deg - MAX_PITCH_DEG).abs() < 1e-3);

        control.disconnect().unwrap();
        assert!(fake.wait_for(LONG, |s| s.screen == Screen::Connect));
        let state = control.state();
        assert_eq!(
            (state.position, state.shard, state.handoffs),
            (None, None, 0)
        );
        assert_eq!((state.scene, state.scene_revision), (None, 0));
        assert_eq!(state.scene_position, None);
        assert_eq!(fake.record().disconnects, 1);
    }

    #[test]
    fn frames_come_on_demand_and_show_which_frame_they_are() {
        let fake = FakeClient::start(FakeOptions::default());
        let control = fake.control();
        assert!(fake.wait_for(LONG, |s| s.frame > 20));
        assert_eq!(fake.record().frames_published, 0, "nobody asked yet");

        let frame = control.frame(None, LONG).unwrap();
        assert_eq!((frame.width, frame.height), (64, 48));
        assert!(frame.seq > 20);
        assert_eq!(*frame.rgba, picture(64, 48, Screen::Connect, frame.seq));
        let next = control.frame(Some(frame.seq), LONG).unwrap();
        assert!(next.seq > frame.seq);
        assert_ne!(next.rgba, frame.rgba);
        let published = fake.record().frames_published;
        assert!(
            (2..=4).contains(&published),
            "{published} frames for two requests"
        );

        let every = FakeClient::start(FakeOptions {
            publish_every_frame: true,
            ..FakeOptions::default()
        });
        assert!(every.wait_for(LONG, |s| s.frame > 20));
        assert!(every.record().frames_published > 15);
    }

    /// A client whose loop hangs: no frames (bounded wait, not a hang), and
    /// input is refused once its queue is full.
    #[test]
    fn a_frozen_client_times_out_instead_of_hanging_its_callers() {
        let fake = FakeClient::start(FakeOptions::default());
        let control = fake.control();
        assert!(fake.wait_for(LONG, |s| s.frame > 2));
        fake.set_paused(true);
        std::thread::sleep(Duration::from_millis(20));
        let frozen_at = control.state().frame;
        assert_eq!(
            control.frame(Some(frozen_at + 1), Duration::from_millis(50)),
            Err(ControlError::Timeout)
        );
        let mut refused = None;
        for i in 0..10_000 {
            if let Err(e) = control.input(ViewerInput::Text('x')) {
                refused = Some((i, e));
                break;
            }
        }
        assert_eq!(
            refused,
            Some((crate::hub::MAX_PENDING_COMMANDS, ControlError::Busy))
        );
        assert_eq!(control.state().frame, frozen_at);
        fake.set_paused(false);
        assert!(fake.wait_for(LONG, |s| s.frame > frozen_at));
    }

    #[test]
    fn quit_ends_the_client_and_closes_its_hub() {
        let fake = FakeClient::start(FakeOptions::default());
        let control = fake.control();
        assert!(!fake.has_quit());
        control.quit();
        let deadline = Instant::now() + LONG;
        while !fake.hub().is_closed() {
            assert!(Instant::now() < deadline, "the fake never quit");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(fake.has_quit());
        assert_eq!(
            control.input(ViewerInput::Text('x')),
            Err(ControlError::Closed)
        );
        assert_eq!(
            control.frame(Some(u64::MAX - 1), LONG),
            Err(ControlError::Closed)
        );
    }

    #[test]
    fn the_picture_is_a_pure_function_of_screen_and_frame_number() {
        let a = picture(4, 2, Screen::Online, 7);
        assert_eq!(a.len(), 4 * 2 * 4);
        assert_eq!(a[..8], [7, 0, 160, 255, 8, 0, 160, 255]);
        assert_eq!(a[16..20], [7, 1, 160, 255]);
        assert_eq!(a, picture(4, 2, Screen::Online, 7));
        assert_ne!(a, picture(4, 2, Screen::Online, 8));
        assert_ne!(a, picture(4, 2, Screen::Error, 7));
        assert_eq!(picture(1, 1, Screen::Connect, 255 + 256)[0], 255);
    }
}
