//! The application core: one state machine, one input path and one scene
//! builder behind every front end.
//!
//! The client is always online: it is either on the **connect screen**
//! (typing a server address), **connecting**, or **playing** on a shard.
//! [`App`] owns that state machine, the [`Controls`] every input ends up in,
//! the player link and the scene that is drawn; it owns no window and no GPU
//! object. A front end — the window, the headless loop, the pixel stream —
//! is a loop around four calls:
//!
//! ```ignore
//! app.update();                         // harness requests, network, scene changes
//! if app.should_quit() { break; }
//! let scene = app.scene(width, height); // the world, the players, the overlay
//! // ... render `scene` with `app.render_options()` ...
//! app.finish_frame(width, height, pixels); // frame counter, harness frame + state
//! ```
//!
//! plus [`App::input`] for whatever input device it has. The harness
//! ([`space_harness::Hub`]) is always attached: [`App::update`] drains its
//! requests into the same input path and [`App::finish_frame`] publishes the
//! frame (when one was read back) and the [`ClientState`].
//!
//! # Where things are
//!
//! The server knows ships at world positions; the client draws a scene
//! ([`World`]) around a spawn point. The two are tied together by the map:
//! the scene's spawn point lies on the map's **anchor** — the centre of its
//! root region ([`crate::link::map_anchor`]), which is where the root shard
//! spawns its players — so a ship `d` metres from the anchor is drawn `d`
//! metres from the spawn point. Every client derives the same anchor from the
//! map its shard sent, so everybody sees the scene in the same place: one
//! world, whichever shard a player joined through. A ship's position is its
//! pilot's eye point; the placeholder character stands [`EYE_HEIGHT`] below
//! it. A player's ship — one a session owns, or one the shard marks with the
//! player visual (a player on a neighbouring shard, seen as its ghost) — is
//! drawn as that character, a ship nobody flies as the ship model.
//!
//! # Moving things, smoothly
//!
//! The shard steps at 30 Hz; frames are drawn at whatever the display does.
//! Everything that moves is drawn by one clock ([`crate::predict::Clock`]):
//! the physics bodies ([`crate::bodies`]) and the other players a couple of
//! ticks in the past, interpolated between the states the shard sent; the
//! player's own ship a few ticks in the future, predicted through the scene's
//! physics from the shard's last word and the inputs it has not seen yet
//! ([`crate::predict::OwnShip`]). The camera follows the predicted ship, the
//! brake assist brakes against its predicted velocity, and the HUD says what
//! the prediction knows of the ship's surroundings: the gravity where it is
//! and whether it stands on something.
//!
//! # Staying connected without a person
//!
//! A client that was given its server on the command line
//! ([`AppConfig::server`]: a container, a kiosk, a test) keeps that
//! connection up by itself: when an attempt fails or the server is lost it
//! shows the reason and tries again after a pause ([`RETRY_DELAYS_S`]), for as
//! long as it takes. The moment somebody edits the address or presses Esc the
//! retrying stops and the connect screen is theirs. Enter (or a harness
//! `connect`) hands the connection back: the address in the field is tried at
//! once, and kept up again from then on. Without a server on the command
//! line nothing is ever retried: a person is there.
//!
//! # Who can end the client
//!
//! Whether a key can end the process depends on whose keys they are
//! ([`AppConfig::escape_quits`]). A window's keyboard belongs to the person
//! who started the client, and Esc on its connect screen quits. A headless
//! or streaming client is somebody's server-side process, and its keys belong
//! to whoever can reach the stream port or the harness's input route: there,
//! no input ends it. Esc still stops the retrying, cancels an attempt and
//! leaves a shard — and that is all it does. Such a client ends when the
//! harness says so ([`Hub::quit`]), when its front end is done (`--frames`,
//! the end of a `--script`), or by a signal.

use std::path::Path;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_harness::{BodyState, ClientState, Command, Frame, Hub, Screen};
use space_protocol::{EntityGuid, SceneState};
use space_render::{
    rotation_y, CameraPose, MeshData, MeshError, ModelInstance, QualityPreset, RenderOptions, Scene,
};
use space_stream::ViewerInput;
use space_world::WorldPos;
use tokio::runtime::Handle;
use tokio::task::JoinHandle;

use crate::address::{dial, ServerAddress, DEFAULT_ADDRESS};
use crate::bodies::{pose_at, BodyRenderer, Drawn, INTERPOLATION_DELAY};
use crate::connect::Prompt;
use crate::input::{CameraMode, Controls, Request, ScreenKind, KEY_HELP};
use crate::kestrel::Detail;
use crate::link::{OtherShip, Replica, ServerLink};
use crate::predict::{Clock, OwnShip, Shown};
use crate::scenes::{character_mesh, metres, relative, ship_mesh, Prop, World, EYE_HEIGHT};
use crate::ui::{self, ConnectView, HudView, Status};

/// The most bodies [`ClientState::bodies`] lists (the nearest ones).
pub const MAX_LISTED_BODIES: usize = 64;

/// The longest a connection attempt may take, name lookup included, before
/// the connect screen is back with an error.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(12);

/// How long a fresh link is given to learn the shard's scene before play
/// starts (the announcement follows the welcome at once; this only avoids a
/// flash of the default scene on a slow link).
const SCENE_PATIENCE: Duration = Duration::from_millis(300);

/// The pauses before the automatic connection attempts, seconds: the first
/// retries come quickly (a shard that is just starting), and from the fourth
/// on it is the last value, for as long as it takes.
pub const RETRY_DELAYS_S: [f32; 4] = [2.0, 3.0, 5.0, 8.0];

/// The pause before the automatic attempt that follows `failures` failed
/// ones in a row.
#[must_use]
pub fn retry_delay(failures: u32) -> Duration {
    let last = RETRY_DELAYS_S.len() - 1;
    let index = usize::try_from(failures).map_or(last, |n| n.min(last));
    Duration::from_secs_f32(RETRY_DELAYS_S[index])
}

/// The chase camera sits this far behind the player's eye point, metres...
pub const CHASE_DISTANCE: f32 = 5.4;
/// ...and this far above it.
pub const CHASE_HEIGHT: f32 = 0.6;
/// The chase camera stays at least this far above a scene's floor, metres.
const CAMERA_CLEARANCE: f64 = 0.3;

/// Another player's character turns to face where it is going once it moves
/// faster than this, metres/second (below, it keeps facing where it faced).
const HEADING_SPEED: f32 = 0.5;

/// The radius a `--glb` model is scaled to when no scale is given, and how
/// many radii in front of the spawn point it is placed.
const MODEL_FIT_RADIUS_M: f32 = 1.5;
const MODEL_DISTANCE_RADII: f32 = 4.0;

/// How often the frame rate readout is refreshed.
const FPS_WINDOW: Duration = Duration::from_millis(500);

/// A client-side test model (`--glb FILE`): shown a few metres in front of
/// the spawn point of whatever scene is drawn. Only this client sees it.
#[derive(Clone, Debug)]
pub struct TestModel {
    mesh: Arc<MeshData>,
    /// Metres per model unit.
    scale: f32,
}

impl TestModel {
    /// Load a `.glb` file. Without `scale` the model is fitted to about three
    /// metres across.
    ///
    /// # Errors
    ///
    /// [`MeshError`] when the file cannot be read or is not a usable model.
    pub fn load(path: &Path, scale: Option<f32>) -> Result<Self, MeshError> {
        Ok(Self::new(MeshData::load(path)?, scale))
    }

    /// Wrap an already built mesh (see [`TestModel::load`] for `scale`).
    #[must_use]
    pub fn new(mesh: MeshData, scale: Option<f32>) -> Self {
        let radius = mesh.radius().max(1e-6);
        Self {
            scale: scale.unwrap_or(MODEL_FIT_RADIUS_M / radius),
            mesh: Arc::new(mesh),
        }
    }

    /// The mesh.
    #[must_use]
    pub fn mesh(&self) -> &Arc<MeshData> {
        &self.mesh
    }

    /// Metres per model unit.
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Where the model stands in `world`: its bounding box centred a few
    /// radii in front of the spawn point (along the view a player starts
    /// with), turned a little so it is not seen dead-on.
    #[must_use]
    pub fn prop(&self, world: &World) -> Prop {
        let (yaw, _) = world.initial_view();
        let ahead = [-yaw.sin(), 0.0, -yaw.cos()];
        let radius = self.mesh.radius().max(1e-6) * self.scale;
        let distance = f64::from(MODEL_DISTANCE_RADII * radius);
        let centre_at = ahead.map(|c| f64::from(c) * distance);
        // The prop's position is where the model's *origin* goes: the
        // bounding-box centre, rotated and scaled, is subtracted so the
        // centre lands on `centre_at`.
        let rotation = rotation_y(0.6);
        let centre = self.mesh.center();
        let origin = [0, 1, 2].map(|r| {
            let rotated: f32 = (0..3).map(|c| rotation[r][c] * centre[c]).sum();
            centre_at[r] - f64::from(rotated * self.scale)
        });
        Prop {
            mesh: Arc::clone(&self.mesh),
            position: WorldPos::from_f64_metres_around(world.spawn(), origin),
            rotation,
            scale: self.scale,
        }
    }
}

/// How an [`App`] starts.
#[derive(Clone, Debug, Default)]
pub struct AppConfig {
    /// `--server ADDR`: connect to this address at once, and keep trying
    /// whenever that fails or the connection is lost (see the module docs,
    /// "Staying connected without a person"). Without it the app starts on
    /// the connect screen and never retries.
    pub server: Option<String>,
    /// The quality preset frames are rendered with while playing.
    pub quality: QualityPreset,
    /// `--glb FILE`: a client-side test model near the spawn point.
    pub model: Option<TestModel>,
    /// Whether Esc on the connect screen ends the client (see the module
    /// docs, "Who can end the client"): `true` for a window, whose keyboard
    /// is its owner's; `false` — the default — for a client without a window,
    /// whose input comes from whoever can reach it.
    pub escape_quits: bool,
}

/// A connection attempt in flight.
struct Attempt {
    /// The address in canonical form, for the screen and the state.
    label: String,
    started: Instant,
    result: Receiver<Result<ServerLink, String>>,
    task: JoinHandle<()>,
}

/// Playing on a shard.
struct Session {
    link: ServerLink,
    /// The address that was dialled, in canonical form.
    label: String,
    /// The replicated view as of this frame.
    replica: Replica,
    /// The estimate of the shard's tick everything is drawn by.
    clock: Clock,
    /// The own ship, predicted.
    own: Option<OwnShip>,
    /// The own ship as drawn this frame.
    shown: Option<Shown>,
}

impl Session {
    /// The frame's bookkeeping: the clock, and the own ship's prediction
    /// (rebuilt for a new ship, shard, scene or anchor).
    fn advance(&mut self, scene: space_protocol::Scene, now: Instant) {
        self.clock.update(&self.replica, now);
        let sent = self.link.take_sent();
        let stale = self
            .own
            .as_ref()
            .is_none_or(|own| !own.is_for(&self.replica, scene));
        if stale {
            match OwnShip::new(&self.replica, scene) {
                Some(mut fresh) => {
                    if let Some(old) = &self.own {
                        fresh.inherit(old, self.shown.map(|s| s.state.pos), now);
                    }
                    self.own = Some(fresh);
                }
                None => self.own = None,
            }
        }
        self.shown = match (self.own.as_mut(), self.clock.now()) {
            (Some(own), Some(tick)) => {
                let at = OwnShip::draw_tick(tick);
                own.update(&self.replica, &sent, at, now);
                Some(own.shown(at, now))
            }
            _ => None,
        };
    }

    /// The own ship's state as drawn: predicted, or the shard's word until
    /// there is a prediction.
    fn ship(&self) -> Option<space_sim::ShipState> {
        self.shown.map(|s| s.state).or(self.replica.state)
    }
}

enum Net {
    /// On the connect screen.
    Idle,
    Connecting(Attempt),
    Online(Box<Session>),
}

/// What the cached world content was built for.
#[derive(Clone, Copy, PartialEq)]
struct BaseKey {
    scene: space_protocol::Scene,
    position: WorldPos,
    forward: [u32; 3],
    aspect: u32,
}

struct Base {
    key: BaseKey,
    /// How many of the scene's instances and models are the world's own (the
    /// rest are this frame's players and are replaced every frame).
    instances: usize,
    models: usize,
}

/// Frames per second over the last half second.
struct FpsMeter {
    since: Instant,
    frames: u32,
    last: Instant,
    fps: f32,
}

impl FpsMeter {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            since: now,
            frames: 0,
            last: now,
            fps: 0.0,
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        self.frames += 1;
        let window = now.duration_since(self.since);
        if window >= FPS_WINDOW {
            self.fps = self.frames as f32 / window.as_secs_f32();
            self.since = now;
            self.frames = 0;
        } else if self.fps == 0.0 {
            // Before the first window is full: the last frame's rate.
            let dt = now.duration_since(self.last).as_secs_f32();
            if dt > 0.0 {
                self.fps = 1.0 / dt;
            }
        }
        self.last = now;
    }
}

/// The application core. See the module docs.
pub struct App {
    runtime: Handle,
    hub: Arc<Hub>,
    quality: QualityPreset,
    model: Option<TestModel>,
    controls: Controls,
    net: Net,
    /// The scene being drawn while playing, and the state it was chosen by.
    world: World,
    drawn: SceneState,
    /// Where the other players' characters last faced (yaw, radians).
    headings: Vec<(EntityGuid, f32)>,
    /// Started with a server address: its connection is kept up unattended.
    unattended: bool,
    /// Automatic reconnection is in force (nobody has taken over).
    reconnect: bool,
    /// Automatic attempts that failed in a row.
    failures: u32,
    /// When the next automatic attempt starts.
    retry_at: Option<Instant>,
    /// Esc on the connect screen ends the client ([`AppConfig::escape_quits`]).
    escape_quits: bool,
    /// What draws the frames ([`App::set_render_device`]): every world is
    /// built for it.
    detail: Detail,
    quit: bool,
    /// Frames finished so far: the number of the last one.
    frame: u64,
    fps: FpsMeter,
    scene: Scene,
    base: Option<Base>,
    /// Draws the physics bodies (and keeps their meshes).
    bodies: BodyRenderer,
    /// The bodies the last frame drew.
    drawn_bodies: Vec<Drawn>,
}

impl std::fmt::Debug for App {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("App")
            .field("screen", &self.screen())
            .field("world", &self.world)
            .field("frame", &self.frame)
            .finish_non_exhaustive()
    }
}

impl App {
    /// A client that connects to `config.server` at once, or shows the
    /// connect screen. `runtime` runs the network tasks; `hub` is the harness
    /// mailbox this client answers (its requests are applied in
    /// [`App::update`]).
    #[must_use]
    pub fn new(config: AppConfig, runtime: Handle, hub: Arc<Hub>) -> Self {
        let prefill = config.server.as_deref().unwrap_or(DEFAULT_ADDRESS);
        let mut app = Self {
            runtime,
            hub,
            quality: config.quality,
            model: config.model,
            controls: Controls::new(prefill),
            net: Net::Idle,
            world: World::default(),
            drawn: SceneState::UNSET,
            headings: Vec::new(),
            unattended: config.server.is_some(),
            reconnect: config.server.is_some(),
            failures: 0,
            retry_at: None,
            escape_quits: config.escape_quits,
            detail: Detail::default(),
            quit: false,
            frame: 0,
            fps: FpsMeter::new(),
            scene: Scene::from_instances(CameraPose::default(), Vec::new()),
            base: None,
            bodies: BodyRenderer::new(),
            drawn_bodies: Vec::new(),
        };
        if config.server.is_some() {
            app.connect();
        }
        app.announce(0, 0);
        app
    }

    /// The harness mailbox.
    #[must_use]
    pub fn hub(&self) -> &Arc<Hub> {
        &self.hub
    }

    /// The controls: the address prompt, the view, the camera mode.
    #[must_use]
    pub fn controls(&self) -> &Controls {
        &self.controls
    }

    /// The scene drawn while playing.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Tell the app what draws its frames, by the device's name
    /// (`RenderContext::device_name`): a front end calls this once it has
    /// its render context. On a software rasteriser the worlds are built
    /// with less detail in the distance ([`Detail`]).
    pub fn set_render_device(&mut self, name: &str) {
        let detail = Detail::for_device(name);
        if detail != self.detail {
            log::info!(
                "space-client: {name} is a software rasteriser: less detail in the distance"
            );
        }
        self.detail = detail;
        self.world = self.world.with_detail(detail);
        self.base = None;
    }

    /// The replicated view of the shard being played on, as of the last
    /// [`App::update`].
    #[must_use]
    pub fn replica(&self) -> Option<&Replica> {
        match &self.net {
            Net::Online(session) => Some(&session.replica),
            _ => None,
        }
    }

    /// Whether the app has been asked to exit: a harness quit, the front end
    /// itself ([`App::request_quit`]), or Esc on the connect screen where
    /// that is the user's to decide ([`AppConfig::escape_quits`]).
    #[must_use]
    pub fn should_quit(&self) -> bool {
        self.quit
    }

    /// Ask the app to exit: the front end's own decision (a closed window),
    /// not input.
    pub fn request_quit(&mut self) {
        self.quit = true;
    }

    /// The number of the last finished frame.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Frames per second, measured.
    #[must_use]
    pub fn fps(&self) -> f32 {
        self.fps.fps
    }

    fn screen_kind(&self) -> ScreenKind {
        match self.net {
            Net::Idle => ScreenKind::Connect,
            Net::Connecting(_) => ScreenKind::Connecting,
            Net::Online(_) => ScreenKind::Online,
        }
    }

    /// The screen being shown, as the harness names it: the connect screen is
    /// [`Screen::Error`] while it shows the error line of a failed attempt.
    #[must_use]
    pub fn screen(&self) -> Screen {
        match self.net {
            Net::Idle if self.controls.prompt.error().is_some() => Screen::Error,
            Net::Idle => Screen::Connect,
            Net::Connecting(_) => Screen::Connecting,
            Net::Online(_) => Screen::Online,
        }
    }

    /// The one input path: a key, a typed character, a view turn, a pointer
    /// position — from the keyboard, a browser viewer or the harness alike.
    pub fn input(&mut self, input: ViewerInput) {
        let screen = self.screen_kind();
        let typed = (screen == ScreenKind::Connect).then(|| self.controls.prompt.text());
        let request = self.controls.handle(screen, input);
        // An edit of the address is somebody taking over: whatever the client
        // was going to retry by itself is no longer what they want.
        if typed.is_some_and(|before| before != self.controls.prompt.text()) {
            self.take_over();
        }
        match request {
            None => {}
            // Enter: now, not when the countdown says — and with it the
            // connection is the client's to keep up again, if it was started
            // to keep one.
            Some(Request::Connect) => {
                self.reconnect = self.unattended;
                self.retry_at = None;
                self.connect();
            }
            Some(Request::Cancel | Request::Disconnect) => {
                self.take_over();
                self.leave(None);
            }
            // Esc on the connect screen. It stops the retrying first...
            Some(Request::Back) if self.retry_at.is_some() => self.take_over(),
            // ...and otherwise ends the client where its user may do that: at
            // a window. Whoever sends keys to a client without one (a viewer
            // of the stream, the harness's input route) cannot end somebody
            // else's process with them; there Esc has nothing left to do.
            Some(Request::Back) if self.escape_quits => self.quit = true,
            Some(Request::Back) => {}
        }
    }

    /// The key hint at the foot of the connect screen as it is drawn now:
    /// what Enter and Esc do there, in this front end ([`ui::connect_keys`]).
    /// `None` while playing.
    #[must_use]
    pub fn connect_keys(&self) -> Option<&'static str> {
        let status = connect_status(&self.net, &self.controls.prompt, self.retry_in())?;
        Some(ui::connect_keys(status, self.escape_quits))
    }

    /// Somebody is at the controls: stop reconnecting on the client's own
    /// initiative (until a connection of theirs succeeds).
    fn take_over(&mut self) {
        self.reconnect = false;
        self.retry_at = None;
        self.failures = 0;
    }

    /// Seconds until the client tries to connect again by itself, when it is
    /// going to.
    #[must_use]
    pub fn retry_in(&self) -> Option<f32> {
        let at = self.retry_at?;
        Some(at.saturating_duration_since(Instant::now()).as_secs_f32())
    }

    /// Release every held movement key (the window lost the focus, the last
    /// viewer left).
    pub fn release_keys(&mut self) {
        self.controls.view.release_keys();
    }

    /// Start connecting to the address in the prompt. An address that cannot
    /// be one is refused on the spot, with the reason under the field.
    fn connect(&mut self) {
        let address = match ServerAddress::parse(&self.controls.prompt.text()) {
            Ok(address) => address,
            Err(e) => {
                self.leave(Some(e.to_string()));
                return;
            }
        };
        self.leave(None);
        let label = address.to_string();
        log::info!("space-client: connecting to {label}");
        let (tx, rx) = mpsc::channel();
        let task = self.runtime.spawn(async move {
            let outcome = match tokio::time::timeout(CONNECT_TIMEOUT, join(&address)).await {
                Ok(outcome) => outcome,
                Err(_) => Err(format!("connecting to {address} took too long")),
            };
            // Nobody listening any more (cancelled): the link closes on drop.
            let _ = tx.send(outcome);
        });
        self.net = Net::Connecting(Attempt {
            label,
            started: Instant::now(),
            result: rx,
            task,
        });
    }

    /// Back to the connect screen: drop the attempt or the link, release the
    /// keys, and show `error` (or clear the error line).
    fn leave(&mut self, error: Option<String>) {
        match std::mem::replace(&mut self.net, Net::Idle) {
            Net::Idle => {}
            Net::Connecting(attempt) => attempt.task.abort(),
            Net::Online(session) => session.link.close(),
        }
        self.controls.view.release_keys();
        self.headings.clear();
        self.base = None;
        self.bodies.forget();
        self.drawn_bodies.clear();
        match error {
            Some(error) => {
                log::warn!("space-client: {error}");
                self.controls.prompt.set_error(error);
            }
            None => self.controls.prompt.clear_error(),
        }
    }

    /// Start playing on `link`.
    fn enter(&mut self, link: ServerLink, label: String) {
        let replica = link.replica();
        log::info!(
            "space-client: connected to {label} as session {} on shard {}",
            replica.session,
            replica
                .shard
                .map_or_else(|| "?".to_owned(), |shard| shard.0.to_string())
        );
        self.drawn = replica.scene;
        self.set_world(replica.scene.scene.or_default());
        self.controls.prompt.clear_error();
        // Connected: from here on an unattended client keeps it that way.
        self.reconnect = self.unattended;
        self.failures = 0;
        self.retry_at = None;
        self.net = Net::Online(Box::new(Session {
            link,
            label,
            replica,
            clock: Clock::new(),
            own: None,
            shown: None,
        }));
    }

    /// Draw `scene` from now on, looking the way a newcomer to it looks.
    fn set_world(&mut self, scene: space_protocol::Scene) {
        self.world = World::new(scene).with_detail(self.detail);
        let (yaw, pitch) = self.world.initial_view();
        self.controls.view.yaw = yaw;
        self.controls.view.pitch = pitch;
        self.base = None;
        // The old scene's bodies are not held over into the new one.
        self.bodies.forget();
    }

    /// Once per frame, before the scene is built: apply the harness's
    /// requests, follow the connection attempt or the link, and notice a
    /// scene change.
    pub fn update(&mut self) {
        for command in self.hub.drain() {
            match command {
                Command::Input(input) => self.input(input),
                // Explicit requests are somebody at the controls, like keys:
                // a new address is an edit, and asking for the connection is
                // what Enter is.
                Command::Connect(address) => {
                    self.take_over();
                    self.reconnect = self.unattended;
                    self.controls.prompt.set_text(&address);
                    self.connect();
                }
                Command::Disconnect => {
                    self.take_over();
                    self.leave(None);
                }
                Command::Quit => self.quit = true,
            }
        }
        if self.hub.quit_requested() {
            self.quit = true;
        }
        /// What the network did since the last frame.
        enum Event {
            Nothing,
            Retry,
            Joined(ServerLink, String),
            Failed(String),
            Scene(SceneState),
        }
        let event = match &mut self.net {
            Net::Idle if self.retry_at.is_some_and(|at| Instant::now() >= at) => Event::Retry,
            Net::Idle => Event::Nothing,
            Net::Connecting(attempt) => match attempt.result.try_recv() {
                Ok(Ok(link)) => Event::Joined(link, attempt.label.clone()),
                Ok(Err(reason)) => Event::Failed(reason),
                Err(TryRecvError::Empty) => Event::Nothing,
                Err(TryRecvError::Disconnected) => {
                    Event::Failed("the connection attempt ended unexpectedly".into())
                }
            },
            Net::Online(session) if session.link.is_closed() => {
                Event::Failed(format!("lost the connection to {}", session.label))
            }
            Net::Online(session) => {
                // The server is authoritative: take its answer, run the own
                // ship ahead of it, send intent (the held keys, or the brake
                // assist's answer to the predicted speed).
                session.replica = session.link.replica();
                session.advance(self.world.scene(), Instant::now());
                let velocity = session.ship().map(|ship| ship.vel);
                session.link.set_thrust(self.controls.thrust(velocity));
                Event::Scene(session.replica.scene)
            }
        };
        match event {
            Event::Nothing => {}
            Event::Retry => {
                self.retry_at = None;
                self.connect();
            }
            Event::Joined(link, label) => self.enter(link, label),
            Event::Failed(reason) => {
                self.leave(Some(reason));
                if self.reconnect {
                    let pause = retry_delay(self.failures);
                    self.failures = self.failures.saturating_add(1);
                    self.retry_at = Some(Instant::now() + pause);
                    log::info!("space-client: trying again in {:.0} s", pause.as_secs_f32());
                }
            }
            Event::Scene(scene) if scene != self.drawn => {
                self.drawn = scene;
                let kind = scene.scene.or_default();
                if kind != self.world.scene() {
                    log::info!("space-client: the scene is now {kind}");
                    self.set_world(kind);
                }
            }
            Event::Scene(_) => {}
        }
    }

    /// The options this frame is rendered with: the chosen preset while
    /// playing; the cheapest one on the connect screen, which is all overlay.
    #[must_use]
    pub fn render_options(&self) -> RenderOptions {
        match self.net {
            Net::Online(_) => self.world.options(self.quality),
            _ => World::default().options(QualityPreset::Ci),
        }
    }

    /// Whether somebody is waiting for a picture ([`Hub::frame_wanted`]): a
    /// front end that does not read its pixels back anyway does so for this
    /// frame.
    #[must_use]
    pub fn frame_wanted(&self) -> bool {
        self.hub.frame_wanted()
    }

    /// Build this frame's scene for a `width` x `height` picture: the connect
    /// screen, or the world around the player with everybody in it and the
    /// HUD on top.
    pub fn scene(&mut self, width: u32, height: u32) -> &Scene {
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        if matches!(self.net, Net::Online(_)) {
            self.world_scene(aspect);
            self.scene.overlay.clear();
            if self.controls.hud {
                let lines = self.hud_lines();
                ui::draw_hud(
                    &mut self.scene.overlay,
                    width,
                    height,
                    &HudView {
                        lines: &lines,
                        hint: KEY_HELP,
                    },
                );
            }
        } else {
            self.connect_scene(width, height, aspect);
        }
        &self.scene
    }

    fn connect_scene(&mut self, width: u32, height: u32, aspect: f32) {
        self.base = None;
        self.scene = Scene::from_instances(
            CameraPose {
                aspect,
                ..CameraPose::default()
            },
            Vec::new(),
        );
        let columns = ui::field_columns(width, height);
        let (text, caret) = self.controls.prompt.window(columns);
        let retry_in = self.retry_in();
        let Some(status) = connect_status(&self.net, &self.controls.prompt, retry_in) else {
            return;
        };
        ui::draw_connect(
            &mut self.scene.overlay,
            width,
            height,
            &ConnectView {
                text: &text,
                caret,
                status,
                escape_quits: self.escape_quits,
            },
        );
    }

    /// Where the ship is from the scene's anchor, metres: what the HUD shows
    /// as the position. Small numbers that show every step, wherever in the
    /// world the cluster's map happens to lie; (0, 0, 0) is the scene's spawn
    /// point. `None` while not playing, and until the shard has shown the
    /// ship.
    #[must_use]
    pub fn position_in_scene(&self) -> Option<[f64; 3]> {
        let replica = self.replica()?;
        let (anchor, ship) = (replica.anchor()?, replica.state?);
        Some(WorldPos::delta(anchor, ship.pos))
    }

    /// Where the player's eye point is in the scene: the ship's place as
    /// drawn (predicted), or the spawn point until the shard has shown the
    /// ship.
    fn eye(&self, session: &Session) -> WorldPos {
        let spawn = self.world.spawn();
        session.ship().map_or(spawn, |ship| {
            scene_place(spawn, session.replica.anchor(), ship.pos)
        })
    }

    /// The camera for this frame, and whether the player's own character is
    /// in the picture.
    fn camera(&self, session: &Session, aspect: f32) -> (CameraPose, WorldPos, bool) {
        let eye = self.eye(session);
        let forward = self.controls.view.forward();
        let position = match self.controls.camera {
            CameraMode::FirstPerson => eye,
            CameraMode::Chase => {
                let mut back = [
                    f64::from(-forward[0] * CHASE_DISTANCE),
                    f64::from(CHASE_HEIGHT - forward[1] * CHASE_DISTANCE),
                    f64::from(-forward[2] * CHASE_DISTANCE),
                ];
                // Looking up swings the camera down behind the player: keep
                // it above the ground the player stands on.
                if let Some(floor) = self.world.floor() {
                    let eye_y = metres(eye)[1];
                    if eye_y > floor + CAMERA_CLEARANCE {
                        back[1] = back[1].max(floor + CAMERA_CLEARANCE - eye_y);
                    }
                }
                WorldPos::from_f64_metres_around(eye, back)
            }
        };
        let camera = self.world.camera(position, forward, aspect);
        (camera, eye, self.controls.camera == CameraMode::Chase)
    }

    fn world_scene(&mut self, aspect: f32) {
        let Net::Online(session) = &self.net else {
            return;
        };
        let replica = &session.replica;
        let (camera, eye, show_self) = self.camera(session, aspect);
        let key = BaseKey {
            scene: self.world.scene(),
            position: camera.position,
            forward: camera.forward.map(f32::to_bits),
            aspect: camera.aspect.to_bits(),
        };
        // The world itself is rebuilt only when the camera has moved (the
        // galaxy's LOD selection is the expensive part); the players are
        // replaced every frame.
        match self.base.as_ref().filter(|base| base.key == key) {
            Some(base) => {
                self.scene.instances.truncate(base.instances);
                self.scene.models.truncate(base.models);
            }
            None => {
                self.scene = self.world.build(&camera);
                if let Some(model) = &self.model {
                    let prop = model.prop(&self.world);
                    self.scene.models.push(prop.instance(&camera));
                }
                self.base = Some(Base {
                    key,
                    instances: self.scene.instances.len(),
                    models: self.scene.models.len(),
                });
            }
        }

        let feet = |eye: WorldPos| {
            WorldPos::from_f64_metres_around(eye, [0.0, -f64::from(EYE_HEIGHT), 0.0])
        };
        if show_self {
            self.scene.models.push(ModelInstance {
                mesh: character_mesh(),
                position: relative(&camera, feet(eye)),
                rotation: rotation_y(self.controls.view.yaw),
                scale: 1.0,
            });
        }
        // Everybody else: where they are relative to the map's anchor is
        // where they are relative to the scene's spawn point, a moment ago
        // (interpolated between the shard's ticks).
        let (spawn, anchor) = (self.world.spawn(), replica.anchor());
        let behind = session.clock.now().map(|t| t - INTERPOLATION_DELAY);
        let mut headings = std::mem::take(&mut self.headings);
        let mut seen: Vec<EntityGuid> = Vec::new();
        for tracked in replica.other_ships() {
            let other = &tracked.ship;
            let pos = behind
                .and_then(|t| pose_at(&tracked.history, t))
                .map_or(other.state.pos, |poses| poses[0].pos);
            let place = scene_place(spawn, anchor, pos);
            seen.push(other.guid);
            // A player (here or, as a ghost, on a neighbouring shard) is a
            // character; a ship nobody flies is the ship model.
            if !tracked.is_player() {
                self.scene.models.push(ModelInstance {
                    mesh: ship_mesh(),
                    position: relative(&camera, place),
                    rotation: quat_rotation(other.state.quat),
                    scale: 1.0,
                });
            } else {
                self.scene.models.push(ModelInstance {
                    mesh: character_mesh(),
                    position: relative(&camera, feet(place)),
                    rotation: rotation_y(heading(&mut headings, other)),
                    scale: 1.0,
                });
            }
        }
        headings.retain(|(guid, _)| seen.contains(guid));
        self.headings = headings;
        // The physics bodies, at the same moment.
        let t = behind.unwrap_or(replica.tick as f64);
        self.drawn_bodies = self
            .bodies
            .draw(replica, t, spawn, Instant::now(), &mut self.scene);
    }

    fn hud_lines(&self) -> Vec<String> {
        let Net::Online(session) = &self.net else {
            return Vec::new();
        };
        let replica = &session.replica;
        let fps = self.fps.fps;
        let ms = if fps > 0.0 { 1000.0 / fps } else { 0.0 };
        let shard = replica
            .shard
            .map_or_else(|| "?".to_owned(), |shard| shard.0.to_string());
        let players = 1 + replica.other_ships().filter(|t| t.is_player()).count();
        let mut lines = vec![
            format!("{fps:.0} fps  {ms:.1} ms"),
            format!(
                "shard {shard}  handoffs {}  players {players}",
                replica.handoffs
            ),
            format!(
                "scene {}  camera {}  brake {}",
                self.world.scene().name(),
                self.controls.camera.name(),
                if self.controls.brake_assist {
                    "on"
                } else {
                    "off"
                }
            ),
        ];
        match (replica.state, self.position_in_scene()) {
            (Some(ship), Some(p)) => {
                let speed = ship.vel.iter().map(|v| v * v).sum::<f32>().sqrt();
                lines.push(hud_motion(speed, p));
            }
            _ => lines.push("waiting for the ship...".to_owned()),
        }
        if let Some(own) = &session.own {
            lines.push(hud_gravity(own.gravity(), own.landed()));
        }
        if let Some(near) = self.nearest_body(session) {
            lines.push(format!(
                "bodies {}  nearest {} {:.1} m  {:.1} m/s",
                self.drawn_bodies.len(),
                space_scene::visual::name(near.0.visual),
                near.1,
                near.0.speed
            ));
        }
        let eye = self.eye(session);
        if let Some(place) = self.world.whereabouts(eye) {
            lines.push(place);
        }
        lines
    }

    /// The body drawn nearest to the player's eye, and how far its root is,
    /// metres.
    fn nearest_body(&self, session: &Session) -> Option<(&Drawn, f64)> {
        let eye = self.eye(session);
        self.drawn_bodies
            .iter()
            .map(|b| (b, WorldPos::distance(eye, b.place)))
            .min_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// The bodies of the last frame as the harness lists them: nearest
    /// first, at most [`MAX_LISTED_BODIES`].
    fn body_states(&self, session: &Session) -> Vec<BodyState> {
        let eye = self.eye(session);
        let spawn = self.world.spawn();
        let mut out: Vec<BodyState> = self
            .drawn_bodies
            .iter()
            .map(|b| BodyState {
                guid: format!("{}", b.guid),
                visual: space_scene::visual::name(b.visual).to_owned(),
                visual_id: b.visual,
                position: WorldPos::delta(spawn, b.place),
                distance: WorldPos::distance(eye, b.place),
                speed: f64::from(b.speed),
                parts: u32::try_from(b.parts).unwrap_or(u32::MAX),
                in_view: b.in_view,
            })
            .collect();
        out.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        out.truncate(MAX_LISTED_BODIES);
        out
    }

    /// The snapshot the harness publishes for a `width` x `height` frame.
    #[must_use]
    pub fn client_state(&self, width: u32, height: u32) -> ClientState {
        let mut state = ClientState {
            screen: self.screen(),
            yaw_deg: self.controls.view.yaw.to_degrees(),
            pitch_deg: self.controls.view.pitch.to_degrees(),
            frame: self.frame,
            width,
            height,
            fps: self.fps.fps,
            ..ClientState::default()
        };
        match &self.net {
            Net::Idle => {
                state.error = self.controls.prompt.error().map(str::to_owned);
                state.address = self.controls.prompt.text();
                state.retry_in_s = self.retry_in();
            }
            Net::Connecting(attempt) => state.address.clone_from(&attempt.label),
            Net::Online(session) => {
                let replica = &session.replica;
                state.address.clone_from(&session.label);
                state.shard = replica.shard.map(|shard| shard.0);
                state.session = Some(replica.session);
                state.tick = replica.tick;
                state.handoffs = replica.handoffs;
                state.scene = Some(self.world.scene().name().to_owned());
                state.scene_revision = self.drawn.revision;
                state.scene_position = self.position_in_scene();
                if let Some(ship) = replica.state {
                    state.set_ship(metres(ship.pos), ship.vel.map(f64::from));
                }
                if let Some(own) = &session.own {
                    let g = own.gravity();
                    state.gravity = g.iter().map(|c| c * c).sum::<f64>().sqrt();
                    state.landed = own.landed();
                    state.correction_m = f64::from(own.correction());
                    state.correction_max_m = f64::from(own.corrections().0);
                }
                state.bodies = self.body_states(session);
                state.bodies_in_view =
                    u32::try_from(self.drawn_bodies.iter().filter(|b| b.in_view).count())
                        .unwrap_or(u32::MAX);
            }
        }
        state
    }

    /// Tell the harness what is on screen before the first frame exists (a
    /// script that starts with the client reads the state at once, and must
    /// find the connect screen and its address there, not an empty default).
    /// `width` x `height` is the size frames are going to have.
    pub fn announce(&self, width: u32, height: u32) {
        self.hub.publish_state(self.client_state(width, height));
    }

    /// The frame built by [`App::scene`] has been rendered: count it, hand
    /// its pixels (RGBA8, top row first — when the front end read them back)
    /// to the harness, then publish the state with the same frame number.
    pub fn finish_frame(&mut self, width: u32, height: u32, pixels: Option<Vec<u8>>) {
        self.frame += 1;
        self.fps.tick();
        if let Some(frame) = pixels.and_then(|rgba| Frame::new(self.frame, width, height, rgba)) {
            self.hub.publish_frame(frame);
        }
        self.hub.publish_state(self.client_state(width, height));
    }

    /// One line for a title bar: frame rate, what is on screen, where.
    #[must_use]
    pub fn title(&self) -> String {
        let fps = self.fps.fps;
        let ms = if fps > 0.0 { 1000.0 / fps } else { 0.0 };
        let what = match &self.net {
            Net::Idle => match self.retry_in() {
                Some(seconds) => format!("retrying in {:.0} s", seconds.ceil()),
                None => "connect".to_owned(),
            },
            Net::Connecting(attempt) => format!("connecting to {}", attempt.label),
            Net::Online(session) => {
                let replica = &session.replica;
                let shard = replica
                    .shard
                    .map_or_else(|| "?".to_owned(), |shard| shard.0.to_string());
                let handoffs = match replica.handoffs {
                    0 => String::new(),
                    n => format!(" after {n} handoff(s)"),
                };
                let place = self
                    .world
                    .whereabouts(self.eye(session))
                    .map_or_else(String::new, |place| format!(" - {place}"));
                format!(
                    "{} on {}, shard {shard}{handoffs}{place}",
                    self.world.scene().name(),
                    session.label
                )
            }
        };
        format!("space - {fps:.0} fps ({ms:.2} ms) - {what}")
    }

    /// The app is going away: leave the server and tell the harness
    /// ([`Hub::close`]), so nothing waits for a frame that will never come.
    pub fn shutdown(&mut self) {
        self.leave(None);
        self.hub.close();
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// What the connect screen shows under its address field: how the attempt is
/// going, why the last one failed and whether another follows by itself.
/// `None` while playing.
fn connect_status<'a>(
    net: &'a Net,
    prompt: &'a Prompt,
    retry_in: Option<f32>,
) -> Option<Status<'a>> {
    match net {
        Net::Online(_) => None,
        Net::Connecting(attempt) => Some(Status::Connecting {
            address: &attempt.label,
            seconds: attempt.started.elapsed().as_secs_f32(),
        }),
        Net::Idle => Some(match (prompt.error(), retry_in) {
            (Some(reason), Some(seconds)) => Status::Retrying { reason, seconds },
            (Some(error), None) => Status::Error(error),
            (None, _) => Status::Idle,
        }),
    }
}

/// Resolve `address`, join the shard and give it a moment to say which scene
/// it shows.
async fn join(address: &ServerAddress) -> Result<ServerLink, String> {
    let link = dial(address).await?;
    let asked = Instant::now();
    while !link.scene().is_set() && asked.elapsed() < SCENE_PATIENCE {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    Ok(link)
}

/// The HUD's line on how the ship moves and where it is from the scene's
/// anchor (metres, one decimal).
fn hud_motion(speed: f32, from_anchor: [f64; 3]) -> String {
    // (A value that rounds to zero is shown as 0.0, not -0.0.)
    let [x, y, z] = from_anchor.map(|v| if v.abs() < 0.05 { 0.0 } else { v });
    format!("speed {speed:.1} m/s  pos {x:.1} {y:.1} {z:.1}")
}

/// The HUD's line on the gravity at the ship and whether it stands on
/// something.
fn hud_gravity(gravity: [f64; 3], landed: bool) -> String {
    let g = gravity.iter().map(|c| c * c).sum::<f64>().sqrt();
    if g < 0.005 {
        "weightless".to_owned()
    } else {
        format!(
            "gravity {g:.2} m/s2{}",
            if landed { "  landed" } else { "" }
        )
    }
}

/// Where `position` — a place in the server's world — is in a scene whose
/// spawn point is `spawn`: as far from the spawn point as it is from the
/// map's `anchor`. (Without an anchor — no map and no ship yet — everything
/// is at the spawn point; it lasts a frame or two.)
fn scene_place(spawn: WorldPos, anchor: Option<WorldPos>, position: WorldPos) -> WorldPos {
    anchor.map_or(spawn, |anchor| {
        WorldPos::from_f64_metres_around(spawn, WorldPos::delta(anchor, position))
    })
}

/// The yaw another player's character is drawn with: along its horizontal
/// velocity while it moves, where it last faced otherwise.
fn heading(headings: &mut Vec<(EntityGuid, f32)>, ship: &OtherShip) -> f32 {
    let [vx, _, vz] = ship.state.vel;
    let moving = (vx * vx + vz * vz).sqrt() > HEADING_SPEED;
    let slot = headings.iter_mut().find(|(guid, _)| *guid == ship.guid);
    match (slot, moving) {
        (Some(slot), true) => {
            slot.1 = (-vx).atan2(-vz);
            slot.1
        }
        (Some(slot), false) => slot.1,
        (None, _) => {
            let yaw = if moving { (-vx).atan2(-vz) } else { 0.0 };
            headings.push((ship.guid, yaw));
            yaw
        }
    }
}

/// The rotation matrix (rows) of a unit quaternion `(w, x, y, z)`; the
/// identity for one that is not finite or has no length.
fn quat_rotation(quat: [f32; 4]) -> [[f32; 3]; 3] {
    let length = quat.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !length.is_finite() || length < 1e-6 {
        return space_render::IDENTITY_ROTATION;
    }
    let [w, x, y, z] = quat.map(|v| v / length);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_harness::Control;
    use space_protocol::ShardId;
    use space_sim::ShipState;
    use space_stream::Key;

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime")
    }

    /// The core as the window runs it: its user may quit with Esc.
    fn app(rt: &tokio::runtime::Runtime, server: Option<&str>) -> App {
        core(rt, server, true)
    }

    /// The core as `--headless` and `--stream` run it: no input ends it.
    fn served(rt: &tokio::runtime::Runtime, server: Option<&str>) -> App {
        core(rt, server, false)
    }

    fn core(rt: &tokio::runtime::Runtime, server: Option<&str>, escape_quits: bool) -> App {
        App::new(
            AppConfig {
                server: server.map(str::to_owned),
                quality: QualityPreset::Ci,
                model: None,
                escape_quits,
            },
            rt.handle().clone(),
            Hub::new(),
        )
    }

    fn stroke(app: &mut App, key: Key) {
        app.input(ViewerInput::Key { key, down: true });
        app.input(ViewerInput::Key { key, down: false });
    }

    fn type_text(app: &mut App, text: &str) {
        for c in text.chars() {
            app.input(ViewerInput::Text(c));
        }
    }

    /// Run frames (no rendering) until `done`, for at most `seconds`.
    fn frames_until(app: &mut App, seconds: u64, mut done: impl FnMut(&App) -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        loop {
            app.update();
            let _ = app.scene(320, 240);
            app.finish_frame(320, 240, None);
            if done(app) {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// Without `--server` the client starts on the connect screen, prefilled,
    /// and typing edits the address — through the one input path.
    #[test]
    fn the_client_starts_on_the_connect_screen_and_takes_typing() {
        let rt = runtime();
        let mut app = app(&rt, None);
        assert_eq!(app.screen(), Screen::Connect);
        assert_eq!(app.controls().prompt.text(), DEFAULT_ADDRESS);
        for _ in 0..4 {
            stroke(&mut app, Key::Backspace);
        }
        type_text(&mut app, "7778");
        assert_eq!(app.controls().prompt.text(), "127.0.0.1:7778");
        // W is a letter here, not thrust.
        app.input(ViewerInput::Key {
            key: Key::Forward,
            down: true,
        });
        assert!(!app.controls().view.any_held());

        // The state says what is on screen; the frame number counts frames.
        app.update();
        let scene = app.scene(640, 480);
        assert!(!scene.overlay.is_empty(), "the connect screen is overlay");
        assert!(scene.instances.is_empty() && scene.models.is_empty());
        app.finish_frame(640, 480, None);
        let state = app.hub().state();
        assert_eq!(state.screen, Screen::Connect);
        assert_eq!(state.address, "127.0.0.1:7778");
        assert_eq!((state.frame, state.width, state.height), (1, 640, 480));
        assert_eq!(
            (state.error, state.shard, state.position),
            (None, None, None)
        );
        // Before any frame the harness already knows what is on screen.
        let fresh = Hub::new();
        let unborn = App::new(
            AppConfig::default(),
            rt.handle().clone(),
            Arc::clone(&fresh),
        );
        assert_eq!(fresh.state().address, DEFAULT_ADDRESS);
        assert_eq!(
            (fresh.state().screen, fresh.state().frame),
            (Screen::Connect, 0)
        );
        unborn.announce(800, 600);
        assert_eq!((fresh.state().width, fresh.state().height), (800, 600));
        drop(unborn);
        assert_eq!(app.render_options().preset, QualityPreset::Ci);
        assert!(app.title().contains("connect"), "{}", app.title());

        // Escape on the connect screen quits — this is the window's core —
        // and the screen says so.
        assert_eq!(app.connect_keys(), Some("Enter connects    Esc quits"));
        assert!(!app.should_quit());
        stroke(&mut app, Key::Escape);
        assert!(app.should_quit());
    }

    /// An address that cannot be one is refused at once: the error line says
    /// why, the screen reports `error`, the text stays, and editing clears it.
    #[test]
    fn a_malformed_address_is_refused_on_the_spot() {
        let rt = runtime();
        let mut app = app(&rt, None);
        for _ in 0..DEFAULT_ADDRESS.len() {
            stroke(&mut app, Key::Backspace);
        }
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Error, "an empty address");
        type_text(&mut app, "host:port");
        assert_eq!(app.screen(), Screen::Connect, "editing clears the error");
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Error);
        let state = app.client_state(320, 240);
        assert_eq!(state.address, "host:port", "the typed address is kept");
        assert!(state.error.is_some_and(|e| e.contains("port")));
        // The error is drawn.
        app.update();
        assert!(!app.scene(320, 240).overlay.is_empty());
    }

    /// A connection nobody answers, made by hand (no `--server`): "Connecting"
    /// for a bounded time, then the connect screen again with a readable
    /// reason and the address kept — and nothing is retried: a person is
    /// there. Escape cancels an attempt.
    #[test]
    fn a_failed_attempt_returns_to_the_connect_screen_with_a_reason() {
        let rt = runtime();
        // A UDP port nobody listens on (bound and released just now).
        let dead = std::net::UdpSocket::bind("127.0.0.1:0")
            .and_then(|socket| socket.local_addr())
            .expect("a free port");
        let mut app = app(&rt, None);
        for _ in 0..DEFAULT_ADDRESS.len() {
            stroke(&mut app, Key::Backspace);
        }
        type_text(&mut app, &dead.to_string());
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Connecting);
        assert_eq!(app.client_state(1, 1).address, dead.to_string());
        // Typing while connecting changes nothing.
        type_text(&mut app, "xyz");
        assert_eq!(app.controls().prompt.text(), dead.to_string());

        let started = Instant::now();
        assert!(
            frames_until(&mut app, 20, |app| app.screen() != Screen::Connecting),
            "the attempt never ended"
        );
        assert!(started.elapsed() < CONNECT_TIMEOUT + Duration::from_secs(2));
        assert_eq!(app.screen(), Screen::Error);
        let state = app.hub().state();
        let error = state.error.expect("an error line");
        assert!(
            error.contains(&dead.to_string()) && error.contains("no answer"),
            "{error}"
        );
        assert_eq!(state.address, dead.to_string(), "the address is kept");
        assert_eq!(state.retry_in_s, None, "nothing is retried by itself");
        assert!(!app.should_quit());
        // ...and it stays that way.
        let waited = Instant::now();
        while waited.elapsed() < retry_delay(0) + Duration::from_millis(500) {
            app.update();
            assert_eq!(app.screen(), Screen::Error);
            std::thread::sleep(Duration::from_millis(20));
        }

        // Enter tries again; Escape cancels and leaves no error.
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Connecting);
        stroke(&mut app, Key::Escape);
        assert_eq!(app.screen(), Screen::Connect);
        assert!(!app.should_quit(), "Escape cancelled, it did not quit");

        // The harness asks for a connection the same way.
        app.hub().connect("no-such-shard.invalid:7777").unwrap();
        app.update();
        assert_eq!(app.screen(), Screen::Connecting);
        assert_eq!(
            app.controls().prompt.text(),
            "no-such-shard.invalid:7777",
            "the address field shows what is being dialled"
        );
        assert!(frames_until(&mut app, 20, |app| app.screen() == Screen::Error));
        let error = app.hub().state().error.expect("an error line");
        assert!(error.contains("no-such-shard.invalid"), "{error}");
        assert_eq!(app.retry_in(), None);
        app.hub().disconnect().unwrap();
        app.update();
        assert_eq!(app.screen(), Screen::Connect);
        app.hub().quit();
        app.update();
        assert!(app.should_quit());
    }

    #[test]
    fn the_pauses_between_automatic_attempts_grow_to_a_cap() {
        let seconds: Vec<f32> = (0..7).map(|n| retry_delay(n).as_secs_f32()).collect();
        assert_eq!(seconds, [2.0, 3.0, 5.0, 8.0, 8.0, 8.0, 8.0]);
        assert_eq!(retry_delay(u32::MAX), Duration::from_secs(8));
        assert!(RETRY_DELAYS_S.windows(2).all(|w| w[0] < w[1]));
    }

    /// With a server on the command line the client does not wait for a key:
    /// a failed attempt is followed by another, after a pause the screen and
    /// the harness count down. Whoever touches the address field or presses
    /// Escape takes over, and from then on nothing happens by itself — until
    /// Enter hands the connection back.
    #[test]
    fn with_a_server_given_the_client_retries_until_somebody_takes_over() {
        let rt = runtime();
        // A name that cannot resolve fails at once, attempt after attempt.
        let nowhere = "no-such-shard.invalid:7777";
        let fails = |app: &mut App| {
            assert!(
                frames_until(app, 20, |app| app.screen() == Screen::Error),
                "the attempt never failed"
            );
        };

        let mut app = app(&rt, Some(nowhere));
        assert_eq!(
            app.screen(),
            Screen::Connecting,
            "--server connects at once"
        );
        fails(&mut app);
        // The reason is on the screen, and so is the countdown.
        let state = app.hub().state();
        assert!(state
            .error
            .is_some_and(|e| e.contains("no-such-shard.invalid")));
        assert_eq!(state.address, nowhere);
        let wait = state.retry_in_s.expect("a retry is pending");
        assert!(wait > 0.0 && wait <= 2.0, "{wait}");
        assert!(app.title().contains("retrying in"), "{}", app.title());
        assert!(!app.scene(640, 360).overlay.is_empty());
        // It counts down, and then the client tries again by itself.
        let started = Instant::now();
        assert!(
            frames_until(&mut app, 20, |app| app.screen() == Screen::Connecting),
            "no second attempt"
        );
        let waited = started.elapsed().as_secs_f32();
        assert!(waited > 1.0 && waited < 3.5, "retried after {waited} s");
        assert_eq!(app.retry_in(), None, "an attempt is running");
        // The second failure waits longer than the first.
        fails(&mut app);
        let second = app.retry_in().expect("a retry is pending");
        assert!(second > 2.0 && second <= 3.0, "{second}");

        // Enter does not wait for the countdown (and the client goes on
        // retrying after it).
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Connecting);
        fails(&mut app);
        assert!(app.retry_in().is_some());

        // Moving the caret is not an edit...
        stroke(&mut app, Key::Home);
        stroke(&mut app, Key::End);
        assert!(app.retry_in().is_some(), "looking is not taking over");
        // ...typing is: the retrying stops, the error stays, and the client
        // waits for the user like any other.
        type_text(&mut app, "1");
        assert_eq!(app.retry_in(), None, "an edit takes over");
        assert_eq!(app.client_state(1, 1).retry_in_s, None);
        assert_eq!(app.client_state(1, 1).screen, Screen::Connect);
        let waited = Instant::now();
        while waited.elapsed() < retry_delay(2) + Duration::from_millis(500) {
            app.update();
            assert_ne!(app.screen(), Screen::Connecting, "retried after an edit");
            std::thread::sleep(Duration::from_millis(20));
        }
        // Enter hands the connection back to the client: the attempt the
        // user starts is retried like any other when it fails, the pauses
        // starting over...
        stroke(&mut app, Key::Backspace);
        stroke(&mut app, Key::Enter);
        fails(&mut app);
        let again = app.retry_in().expect("Enter resumed the retrying");
        assert!(again > 0.0 && again <= 2.0, "{again}");
        // ...and so does a connection the harness asks for.
        stroke(&mut app, Key::Escape);
        assert_eq!(app.retry_in(), None);
        app.hub().connect(nowhere).unwrap();
        fails(&mut app);
        assert!(app.retry_in().is_some());
        assert!(!app.should_quit());

        // Escape while the countdown runs stops the retrying and does not
        // quit; the next Escape does (this is the window's core).
        let mut app = core(&rt, Some(nowhere), true);
        fails(&mut app);
        assert!(app.retry_in().is_some());
        assert_eq!(
            app.connect_keys(),
            Some("Enter retries now    Esc stops retrying")
        );
        stroke(&mut app, Key::Escape);
        assert_eq!(app.retry_in(), None);
        assert!(
            !app.should_quit(),
            "the first Escape only stops the retrying"
        );
        assert_eq!(
            app.screen(),
            Screen::Error,
            "the reason stays on the screen"
        );
        assert_eq!(app.connect_keys(), Some("Enter tries again    Esc quits"));
        stroke(&mut app, Key::Escape);
        assert!(app.should_quit());

        // Escape during an automatic attempt cancels it and the retrying.
        let mut app = served(&rt, Some("127.0.0.1:9"));
        assert_eq!(app.screen(), Screen::Connecting);
        assert_eq!(app.connect_keys(), Some("Esc cancels"));
        stroke(&mut app, Key::Escape);
        assert_eq!((app.screen(), app.retry_in()), (Screen::Connect, None));
        let waited = Instant::now();
        while waited.elapsed() < Duration::from_secs(3) {
            app.update();
            assert_eq!(app.screen(), Screen::Connect);
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    /// A client without a window is somebody's server-side process: no input
    /// ends it, however often Escape is pressed and whatever the connect
    /// screen shows at the time, and it goes on doing what is asked of it.
    /// Its key hint never says "Esc quits". The harness's quit, which is not
    /// input, still ends it.
    #[test]
    fn no_key_ends_a_client_without_a_window() {
        let rt = runtime();
        assert!(!AppConfig::default().escape_quits, "the safe default");
        let nowhere = "no-such-shard.invalid:7777";
        let fails = |app: &mut App| {
            assert!(
                frames_until(app, 20, |app| app.screen() == Screen::Error),
                "the attempt never failed"
            );
        };

        // On the idle connect screen: once, and again and again.
        let mut app = served(&rt, None);
        assert_eq!(app.connect_keys(), Some("Enter connects"));
        for _ in 0..5 {
            stroke(&mut app, Key::Escape);
            assert!(!app.should_quit());
            assert_eq!(app.screen(), Screen::Connect);
        }
        // Through the harness's input route it is a key like any other.
        app.hub().stroke(Key::Escape).unwrap();
        app.update();
        assert!(!app.should_quit());
        // Still listening: the address can be edited, Enter connects, and
        // Escape cancels the attempt — and only that.
        for _ in 0..DEFAULT_ADDRESS.len() {
            stroke(&mut app, Key::Backspace);
        }
        type_text(&mut app, nowhere);
        assert_eq!(app.controls().prompt.text(), nowhere);
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Connecting);
        assert_eq!(app.connect_keys(), Some("Esc cancels"));
        stroke(&mut app, Key::Escape);
        assert_eq!(app.screen(), Screen::Connect);
        assert!(!app.should_quit());
        // With an error on the screen (nothing is retried: no `--server`).
        stroke(&mut app, Key::Enter);
        fails(&mut app);
        assert_eq!(app.retry_in(), None);
        assert_eq!(app.connect_keys(), Some("Enter tries again"));
        for _ in 0..3 {
            stroke(&mut app, Key::Escape);
        }
        assert!(!app.should_quit());
        assert_eq!(app.screen(), Screen::Error, "Escape changed nothing");

        // While the client retries by itself: the first Escape stops that,
        // the following ones do nothing at all.
        let mut app = served(&rt, Some(nowhere));
        fails(&mut app);
        assert!(app.retry_in().is_some());
        assert_eq!(
            app.connect_keys(),
            Some("Enter retries now    Esc stops retrying")
        );
        stroke(&mut app, Key::Escape);
        assert_eq!((app.retry_in(), app.screen()), (None, Screen::Error));
        assert_eq!(app.connect_keys(), Some("Enter tries again"));
        for _ in 0..5 {
            stroke(&mut app, Key::Escape);
        }
        assert!(!app.should_quit());
        let waited = Instant::now();
        while waited.elapsed() < retry_delay(0) + Duration::from_millis(500) {
            app.update();
            assert_eq!(app.screen(), Screen::Error, "retried after Escape");
            std::thread::sleep(Duration::from_millis(20));
        }
        // The way back is Enter: the address in the field is tried at once,
        // and the client keeps at it by itself again.
        stroke(&mut app, Key::Enter);
        assert_eq!(app.screen(), Screen::Connecting);
        fails(&mut app);
        assert!(app.retry_in().is_some(), "Enter resumed the retrying");
        assert!(!app.should_quit());

        // What is not input still ends it: the harness's quit...
        app.hub().quit();
        app.update();
        assert!(app.should_quit());
        // ...and the front end's own decision (`--frames`, a closed window).
        let mut app = served(&rt, None);
        app.request_quit();
        assert!(app.should_quit());
    }

    #[test]
    fn the_harness_is_told_when_the_app_goes_away() {
        let rt = runtime();
        let hub = Hub::new();
        let app = App::new(AppConfig::default(), rt.handle().clone(), Arc::clone(&hub));
        assert!(!hub.is_closed());
        drop(app);
        assert!(hub.is_closed());
    }

    #[test]
    fn other_players_face_where_they_are_going() {
        let guid = EntityGuid::new(ShardId::new(1), 9);
        let ship = |vel: [f32; 3]| OtherShip {
            guid,
            owner: 3,
            state: ShipState {
                vel,
                ..ShipState::IDENTITY
            },
        };
        let mut headings = Vec::new();
        assert_eq!(heading(&mut headings, &ship([0.0; 3])), 0.0, "at rest: -Z");
        // Moving along +X: a quarter turn to the right (negative yaw).
        let right = heading(&mut headings, &ship([4.0, 0.0, 0.0]));
        assert!(
            (right + std::f32::consts::FRAC_PI_2).abs() < 1e-5,
            "{right}"
        );
        // Slowing to a stop keeps the facing; climbing does not turn it.
        assert_eq!(heading(&mut headings, &ship([0.1, 0.0, 0.0])), right);
        assert_eq!(heading(&mut headings, &ship([0.0, 9.0, 0.0])), right);
        let back = heading(&mut headings, &ship([0.0, 0.0, 3.0]));
        assert!((back.abs() - std::f32::consts::PI).abs() < 1e-5, "{back}");
        assert_eq!(headings.len(), 1);
    }

    /// The HUD shows where the ship is from the scene's anchor, to a
    /// decimal: a step is visible wherever the cluster's map lies.
    #[test]
    fn the_hud_position_is_relative_to_the_anchor() {
        assert_eq!(
            hud_motion(0.0, [0.0, 0.0, 0.0]),
            "speed 0.0 m/s  pos 0.0 0.0 0.0"
        );
        assert_eq!(
            hud_motion(6.04, [1.26, -0.04, -512.0]),
            "speed 6.0 m/s  pos 1.3 0.0 -512.0"
        );
        // A ship 1.3 m from an anchor 8.6e9 m from the world origin: the
        // scene place is exact, not rounded to the 1024 m an f32 would give.
        let anchor = WorldPos::from_f64_metres_around(WorldPos::ZERO, [8_589_934_080.0; 3]);
        let ship = WorldPos::from_f64_metres_around(anchor, [1.3, 0.0, -2.5]);
        let spawn = World::default().spawn();
        let place = scene_place(spawn, Some(anchor), ship);
        let d = WorldPos::delta(spawn, place);
        assert!(
            (d[0] - 1.3).abs() < 1e-4 && (d[2] + 2.5).abs() < 1e-4,
            "{d:?}"
        );
        // No anchor yet: at the spawn point.
        assert_eq!(scene_place(spawn, None, ship), spawn);
    }

    /// The HUD says what pulls on the ship and whether it stands.
    #[test]
    fn the_hud_tells_gravity_and_landing() {
        assert_eq!(hud_gravity([0.0; 3], false), "weightless");
        assert_eq!(hud_gravity([0.0, -9.81, 0.0], false), "gravity 9.81 m/s2");
        assert_eq!(
            hud_gravity([0.0, -9.81, 0.0], true),
            "gravity 9.81 m/s2  landed"
        );
        // A touch without a pull is not landing.
        assert_eq!(hud_gravity([0.0, 0.0, 0.001], true), "weightless");
    }

    #[test]
    fn quaternions_become_rotation_matrices() {
        assert_eq!(
            quat_rotation([1.0, 0.0, 0.0, 0.0]),
            space_render::IDENTITY_ROTATION
        );
        assert_eq!(quat_rotation([0.0; 4]), space_render::IDENTITY_ROTATION);
        assert_eq!(
            quat_rotation([f32::NAN, 0.0, 0.0, 0.0]),
            space_render::IDENTITY_ROTATION
        );
        // A turn of `a` about +Y is `rotation_y(a)`.
        let a = 0.7f32;
        let got = quat_rotation([(a / 2.0).cos(), 0.0, (a / 2.0).sin(), 0.0]);
        let want = rotation_y(a);
        for r in 0..3 {
            for c in 0..3 {
                assert!((got[r][c] - want[r][c]).abs() < 1e-6, "{got:?} vs {want:?}");
            }
        }
    }

    /// The `--glb` model stands in front of the spawn point of every scene,
    /// along the way a newcomer looks, its bounding box centred there.
    #[test]
    fn the_test_model_stands_in_front_of_every_spawn_point() {
        let mesh = MeshData::cuboid([1.0, 2.0, 0.5], 1.0, space_render::MeshMaterial::default());
        let radius = mesh.radius();
        let model = TestModel::new(mesh, None);
        assert!((model.scale() * radius - MODEL_FIT_RADIUS_M).abs() < 1e-4);
        assert!((TestModel::new((**model.mesh()).clone(), Some(2.5)).scale() - 2.5).abs() < 1e-6);
        for kind in space_protocol::Scene::ALL {
            let world = World::new(kind);
            let prop = model.prop(&world);
            let d = WorldPos::delta(world.spawn(), prop.position);
            let (yaw, _) = world.initial_view();
            let ahead = [-f64::from(yaw.sin()), 0.0, -f64::from(yaw.cos())];
            let along: f64 = (0..3).map(|i| d[i] * ahead[i]).sum();
            let distance = f64::from(MODEL_DISTANCE_RADII * MODEL_FIT_RADIUS_M);
            assert!((along - distance).abs() < 1e-2, "{kind}: {along}");
            assert!(d[1].abs() < 1e-2, "{kind}: centred on eye height");
        }
    }
}
