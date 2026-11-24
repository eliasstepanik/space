//! The client end to end, against a real in-process shard over QUIC on
//! loopback (the same boot wiring as the `space-server` binary, as in
//! `handoff.rs`):
//!
//! * a headless client that is driven **only through the harness's `Hub`** —
//!   it types the address on the connect screen, connects, flies, and its
//!   frames show the connect screen and a plausibly lit demo area with the
//!   HUD on top;
//! * the scene switching live when the shard's scene changes, and a Lua
//!   script waiting for the switch through `client.state().scene`;
//! * two clients seeing each other's characters, and an unowned ship;
//! * one world for everyone: clients that joined through different shards
//!   agree where the scene is, and a split while playing does not move it;
//! * a client that was given `--server` reconnecting by itself: before its
//!   shard exists, and when the shard restarts — until somebody types;
//! * who can end the client: no key ends a headless or streaming one, however
//!   often a viewer presses Escape, and it goes on listening;
//! * the pixel stream: a browser stand-in types the address and gets frames;
//! * the real binary's `--script` path: a passing Lua script exits 0, a
//!   failing one exits 1 with its message; `--harness` refuses a public
//!   address without a token.
//!
//! The tests that render need a Vulkan device and skip loudly without one
//! (`SPACE_REQUIRE_GPU=1` turns the skip into a failure).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_client::app::{App, AppConfig, CHASE_DISTANCE, CHASE_HEIGHT};
use space_client::headless::{run_headless, HeadlessConfig, HeadlessReport};
use space_client::scenes::{character_mesh, ship_mesh, World};
use space_harness::{ClientState, Control, Frame, Hub, Key, Screen, ViewerInput};
use space_protocol::{Scene, SceneId, SceneState, ShardId, ShardMap, ShardRegion};
use space_render::{ContextOptions, QualityPreset, RenderContext};
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::{NodeRef, WorldPos};
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

const LONG: Duration = Duration::from_secs(30);

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("runtime")
}

struct Running {
    net: Arc<ShardNet>,
    shard: Arc<AsyncMutex<Shard>>,
    tasks: Vec<JoinHandle<()>>,
}

impl Running {
    fn addr(&self) -> SocketAddr {
        self.net.local_addr().expect("bound address")
    }

    fn stop(self) {
        for task in &self.tasks {
            task.abort();
        }
        self.net.close();
    }
}

/// Bind a shard on loopback and start serving and ticking it at 30 Hz (the
/// region is the single cell a standalone dev shard owns). Must be called
/// inside the runtime.
fn start_shard() -> Running {
    start_shard_at(
        1,
        NodeRef::new(64, [0, 0, 0]),
        SocketAddr::from(([127, 0, 0, 1], 0)),
    )
}

/// The same for shard `id` owning `region`, bound to `bind`.
fn start_shard_at(id: u32, region: NodeRef, bind: SocketAddr) -> Running {
    serve_shard(id, region, ShardNet::bind(bind).expect("bind"))
}

/// Serve and tick shard `id`, owning `region`, on the bound `net`.
fn serve_shard(id: u32, region: NodeRef, net: ShardNet) -> Running {
    let net = Arc::new(net);
    let shard = Arc::new(AsyncMutex::new(Shard::new(ShardConfig::new(
        ShardId::new(id),
        net.local_addr().expect("addr"),
        region,
    ))));
    let serve = tokio::spawn({
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        async move {
            let _ = net.serve(shard).await;
        }
    });
    let ticker = tokio::spawn({
        let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
        async move {
            let mut clock =
                tokio::time::interval(Duration::from_secs_f64(f64::from(space_sim::dt())));
            clock.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                clock.tick().await;
                shard.lock().await.tick();
                net.flush(&shard).await;
            }
        }
    });
    Running {
        net,
        shard,
        tasks: vec![serve, ticker],
    }
}

/// A usable Vulkan device, or `None` after skipping loudly (or failing, with
/// `SPACE_REQUIRE_GPU=1`).
fn gpu() -> Option<RenderContext> {
    match RenderContext::new(&ContextOptions::default()) {
        Ok(ctx) => {
            // With the variable set the layer must really be active, or a
            // `validation_messages == 0` assertion is vacuous.
            assert!(
                std::env::var("SPACE_RENDER_VALIDATION").as_deref() != Ok("1")
                    || ctx.validation_active(),
                "SPACE_RENDER_VALIDATION=1 but VK_LAYER_KHRONOS_validation is not active"
            );
            Some(ctx)
        }
        Err(e) => {
            space_render::skip_or_require_gpu(&e);
            None
        }
    }
}

/// A headless client on its own thread: exactly what `--headless` runs.
struct Headless {
    hub: Arc<Hub>,
    stop: Arc<AtomicBool>,
    thread: std::thread::JoinHandle<HeadlessReport>,
}

impl Headless {
    fn start(rt: &tokio::runtime::Runtime, server: Option<String>, config: HeadlessConfig) -> Self {
        let hub = Hub::new();
        let stop = Arc::new(AtomicBool::new(false));
        let (handle, hub_for_app, stopper) = (rt.handle().clone(), Arc::clone(&hub), stop.clone());
        let thread = std::thread::spawn(move || {
            let ctx = RenderContext::new(&ContextOptions::default()).expect("render context");
            let mut app = App::new(
                AppConfig {
                    server,
                    quality: QualityPreset::High,
                    model: None,
                    escape_quits: false,
                },
                handle,
                hub_for_app,
            );
            let report = run_headless(&ctx, &mut app, &config, None, &stopper).expect("rendering");
            drop(app);
            report
        });
        Self { hub, stop, thread }
    }

    fn wait(&self, what: &str, condition: impl Fn(&ClientState) -> bool) -> ClientState {
        let deadline = Instant::now() + LONG;
        loop {
            let state = self.hub.state();
            if condition(&state) {
                return state;
            }
            assert!(Instant::now() < deadline, "never {what}: {state:?}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A frame rendered after everything requested so far has been applied.
    fn fresh_frame(&self) -> Frame {
        let seen = self.hub.state().frame;
        self.hub
            .frame(Some(seen + 1), LONG)
            .expect("a frame from the client")
    }

    fn finish(self) -> HeadlessReport {
        self.hub.quit();
        let report = self.thread.join().expect("the client thread");
        self.stop.store(true, Ordering::Relaxed);
        report
    }
}

/// What a picture looks like, in numbers.
#[derive(Debug)]
struct Look {
    /// Mean of the per-pixel maximum channel, 0..255.
    mean: f64,
    /// Fraction of pixels darker than 12 in every channel.
    black: f64,
    /// Fraction of pixels brighter than 250 in every channel.
    white: f64,
    /// Fraction of pixels that are clearly coloured (a channel well above
    /// another).
    coloured: f64,
}

fn look(frame: &Frame, region: Option<[u32; 4]>) -> Look {
    let [x0, y0, x1, y1] = region.unwrap_or([0, 0, frame.width, frame.height]);
    let (mut sum, mut black, mut white, mut coloured, mut count) = (0u64, 0u64, 0u64, 0u64, 0u64);
    for y in y0..y1 {
        for x in x0..x1 {
            let [r, g, b, _] = frame.pixel(x, y).expect("inside the frame");
            let (max, min) = (r.max(g).max(b), r.min(g).min(b));
            sum += u64::from(max);
            black += u64::from(max < 12);
            white += u64::from(min > 250);
            coloured += u64::from(max - min > 60);
            count += 1;
        }
    }
    let part = |n: u64| n as f64 / count.max(1) as f64;
    Look {
        mean: sum as f64 / count.max(1) as f64,
        black: part(black),
        white: part(white),
        coloured: part(coloured),
    }
}

/// How many pixels of `region` are as bright as overlay text.
fn bright_pixels(frame: &Frame, region: [u32; 4]) -> usize {
    let [x0, y0, x1, y1] = region;
    let mut count = 0;
    for y in y0..y1.min(frame.height) {
        for x in x0..x1.min(frame.width) {
            let [r, g, b, _] = frame.pixel(x, y).expect("inside the frame");
            count += usize::from(r.min(g).min(b) > 170);
        }
    }
    count
}

/// The whole user journey through nothing but the `Hub`: the connect screen is
/// shown, the address is typed and Enter connects, W flies the ship (the
/// authoritative one, on the shard), letting go stops it, and the frames are
/// what a person would expect to see.
#[test]
fn a_headless_client_driven_through_the_hub_connects_flies_and_shows_the_demo_area() {
    if gpu().is_none() {
        return;
    }
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let (width, height) = (640, 360);
    let client = Headless::start(
        &rt,
        None,
        HeadlessConfig {
            width,
            height,
            fps: 60,
            ..HeadlessConfig::default()
        },
    );
    let hub = &client.hub;

    // --- the connect screen
    let state = client.wait("showed the connect screen", |s| s.frame > 2);
    assert_eq!(state.screen, Screen::Connect);
    assert_eq!(state.address, "127.0.0.1:7777", "the prefilled address");
    assert_eq!((state.width, state.height), (width, height));
    assert_eq!(
        (state.shard, state.session, state.position),
        (None, None, None)
    );
    // No scene before a server said one, and nothing is retried: the client
    // was started without `--server`.
    assert_eq!((state.scene.as_deref(), state.scene_revision), (None, 0));
    assert_eq!(state.retry_in_s, None);
    let screen = client.fresh_frame();
    assert_eq!((screen.width, screen.height), (width, height));
    let all = look(&screen, None);
    assert!(
        all.black < 0.2,
        "the connect screen has a backdrop: {all:?}"
    );
    assert!(
        all.mean > 8.0 && all.mean < 90.0,
        "calm, not glaring: {all:?}"
    );
    // The panel is in the middle and has text in it; the corners have none.
    let panel = [
        width / 2 - 150,
        height / 2 - 75,
        width / 2 + 150,
        height / 2 + 75,
    ];
    assert!(bright_pixels(&screen, panel) > 300, "no text in the panel");
    assert!(
        bright_pixels(&screen, [0, 0, 100, 60]) < 20,
        "text in a corner"
    );

    // --- type the shard's address over the prefilled one, as a user would
    let address = shard.addr().to_string();
    hub.stroke(Key::End).unwrap();
    for _ in 0..state.address.len() {
        hub.stroke(Key::Backspace).unwrap();
    }
    client.wait("emptied the field", |s| s.address.is_empty());
    hub.type_text(&address).unwrap();
    let typed = client.wait("typed the address", |s| s.address == address);
    assert_eq!(typed.screen, Screen::Connect);
    // Typing changed the picture (the field shows the new text).
    let edited = client.fresh_frame();
    assert!(edited.rgba != screen.rgba, "the typed text is not drawn");
    hub.stroke(Key::Enter).unwrap();

    // --- online
    let online = client.wait("came online", |s| {
        s.screen == Screen::Online && s.position.is_some() && s.tick > 0
    });
    assert_eq!(online.address, address);
    assert_eq!(online.shard, Some(1));
    assert_eq!(online.error, None);
    // The scene the shard chose (nobody set one: the default, revision 0).
    assert_eq!(online.scene.as_deref(), Some("demo-area"));
    assert_eq!((online.scene_revision, online.retry_in_s), (0, None));
    let session = online.session.expect("a session");
    let start = online.position.expect("a ship");
    assert!(online.speed < 0.01, "the ship spawns at rest: {online:?}");
    // A standalone cell's spawn point is the world origin — and the scene's
    // spawn point, which is where the position in the scene counts from.
    assert!(start.iter().all(|c| c.abs() < 1e-3), "{start:?}");
    assert_eq!(online.scene_position, online.position);

    // --- the demo area, as a newcomer sees it
    let arrival = client.fresh_frame();
    let seen = look(&arrival, None);
    assert!(seen.mean > 45.0, "the demo area is dark: {seen:?}");
    assert!(seen.black < 0.55, "mostly black: {seen:?}");
    assert!(seen.white < 0.05, "blown out: {seen:?}");
    assert!(seen.coloured > 0.01, "no colour anywhere: {seen:?}");
    // The lower half is the lit floor and what stands on it.
    let ground = look(&arrival, Some([0, height / 2, width, height]));
    assert!(ground.black < 0.2 && ground.mean > 70.0, "{ground:?}");
    // The HUD is in the top-left corner, and H switches it off and on.
    let corner = [0, 0, 330, 70];
    let with_hud = bright_pixels(&arrival, corner);
    assert!(with_hud > 250, "no HUD text: {with_hud} bright pixels");
    hub.type_text("h").unwrap();
    let without = client.fresh_frame();
    assert!(
        bright_pixels(&without, corner) * 4 < with_hud,
        "H did not hide the HUD"
    );
    hub.type_text("h").unwrap();
    assert!(bright_pixels(&client.fresh_frame(), corner) > 250);

    // --- fly: W is thrust along the view (-Z), on the authoritative ship
    hub.key(Key::Forward, true).unwrap();
    let moving = client.wait("moved forward", |s| {
        s.position.is_some_and(|p| p[2] < start[2] - 2.0)
    });
    assert!(moving.speed > 1.0, "{moving:?}");
    assert!(moving.velocity.expect("velocity")[2] < -1.0);
    let server = rt.block_on(async {
        let shard = shard.shard.lock().await;
        let ship = shard.ship_of(session).expect("the shard knows our ship");
        shard.sim().ship_state(ship).expect("the ship is alive")
    });
    assert!(
        server.vel[2] < 0.0,
        "the shard applied the thrust: {server:?}"
    );
    assert!(WorldPos::delta(WorldPos::ZERO, server.pos)[2] < -1.0);
    // Letting go: the brake assist brings the ship to a stop.
    hub.key(Key::Forward, false).unwrap();
    let stopped = client.wait("stopped", |s| s.speed < 0.02);
    let end = stopped.position.expect("a ship");
    assert!(end[2] < start[2] - 2.0 && end[0].abs() < 0.5, "{end:?}");

    // --- look around: a turn changes the view angles and the picture
    let before = client.fresh_frame();
    hub.turn_degrees(40.0, -10.0).unwrap();
    let turned = client.wait("turned", |s| (s.yaw_deg - 40.0).abs() < 0.5);
    assert!((turned.pitch_deg - (online.pitch_deg - 10.0)).abs() < 0.5);
    assert!(client.fresh_frame().rgba != before.rgba);
    // First person: the player's own character leaves the picture.
    hub.type_text("c").unwrap();
    let first_person = client.fresh_frame();
    assert!(look(&first_person, None).mean > 30.0);

    // --- Escape leaves the shard; the address is still there
    hub.stroke(Key::Escape).unwrap();
    let left = client.wait("returned to the connect screen", |s| {
        s.screen == Screen::Connect
    });
    assert_eq!(left.address, address);
    assert_eq!((left.shard, left.position, left.speed), (None, None, 0.0));
    assert_eq!((left.scene, left.retry_in_s), (None, None));
    assert_eq!(left.scene_position, None);
    rt.block_on(async {
        for _ in 0..100 {
            if shard.shard.lock().await.ship_of(session).is_none() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!("the shard still has the ship of a client that left");
    });

    let hub = Arc::clone(&client.hub);
    let report = client.finish();
    assert!(report.frames > 20, "{report:?}");
    assert_eq!(report.validation_messages, 0, "{report:?}");
    assert!(
        hub.is_closed(),
        "the harness is told that the client is gone"
    );
    shard.stop();
}

/// An app run without a GPU: frames are built but not rendered.
struct Dry {
    app: App,
}

impl Dry {
    /// The core as `--headless` and `--stream` run it: no key ends it.
    fn new(rt: &tokio::runtime::Runtime, server: Option<String>) -> Self {
        Self::core(rt, server, false)
    }

    /// The core as the window runs it: Esc on the connect screen quits.
    fn windowed(rt: &tokio::runtime::Runtime, server: Option<String>) -> Self {
        Self::core(rt, server, true)
    }

    fn core(rt: &tokio::runtime::Runtime, server: Option<String>, escape_quits: bool) -> Self {
        Self {
            app: App::new(
                AppConfig {
                    server,
                    quality: QualityPreset::High,
                    model: None,
                    escape_quits,
                },
                rt.handle().clone(),
                Hub::new(),
            ),
        }
    }

    fn frame(&mut self) {
        self.app.update();
        let _ = self.app.scene(640, 360);
        self.app.finish_frame(640, 360, None);
    }

    fn until(&mut self, what: &str, done: impl Fn(&App) -> bool) {
        let deadline = Instant::now() + LONG;
        loop {
            self.frame();
            if done(&self.app) {
                return;
            }
            assert!(Instant::now() < deadline, "never {what}: {:?}", self.app);
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A client that was given its server on the command line, once it is
    /// playing there.
    fn online(rt: &tokio::runtime::Runtime, addr: SocketAddr) -> Self {
        let mut dry = Self::new(rt, Some(addr.to_string()));
        dry.until("came online", |app| {
            app.screen() == Screen::Online && app.replica().is_some_and(|r| r.state.is_some())
        });
        dry
    }

    /// A window client that was started without a server and connected from
    /// its connect screen (here: asked to through the harness), once it is
    /// playing.
    fn by_hand(rt: &tokio::runtime::Runtime, addr: SocketAddr) -> Self {
        let mut dry = Self::windowed(rt, None);
        assert_eq!(dry.app.screen(), Screen::Connect);
        dry.app
            .hub()
            .connect(&addr.to_string())
            .expect("the client takes the request");
        dry.until("came online", |app| {
            app.screen() == Screen::Online && app.replica().is_some_and(|r| r.state.is_some())
        });
        dry
    }
}

fn set_scene(rt: &tokio::runtime::Runtime, shard: &Running, state: SceneState) {
    assert!(rt.block_on(async { shard.shard.lock().await.set_scene(state) }));
}

/// The scene is the shard's: a client draws the default until told, follows
/// every change while it is playing — the world is rebuilt, the ship stays
/// where the server has it — and draws the default for a scene it does not
/// know.
#[test]
fn the_scene_switches_live_when_the_shards_scene_changes() {
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let mut client = Dry::online(&rt, shard.addr());
    // Nobody set a scene (revision 0): the default, not nothing.
    assert_eq!(client.app.world().scene(), Scene::DemoArea);
    let demo_models = client.app.scene(640, 360).models.len();
    assert!(demo_models > 20, "the demo area's props: {demo_models}");

    // Fly a little, so "the ship is unaffected" means something.
    client.app.input(ViewerInput::Key {
        key: Key::Right,
        down: true,
    });
    client.until("moved", |app| {
        app.client_state(1, 1).position.is_some_and(|p| p[0] > 1.0)
    });
    client.app.input(ViewerInput::Key {
        key: Key::Right,
        down: false,
    });
    client.until("stopped", |app| app.client_state(1, 1).speed < 0.01);
    let parked = client.app.client_state(1, 1).position.expect("a ship");

    let mut revision = 0;
    for (scene, yaw_deg) in [
        (Scene::Solar, 0.0),
        (Scene::Galaxy, 180.0),
        (Scene::Materials, 0.0),
        (Scene::DemoArea, 0.0),
    ] {
        revision += 1;
        // Look somewhere else first: a new scene starts with its own view.
        client.app.input(ViewerInput::Turn {
            dyaw: 1.0,
            dpitch: 0.3,
        });
        set_scene(&rt, &shard, SceneState::new(scene.id(), revision));
        client.until("switched scene", |app| app.world().scene() == scene);
        let state = client.app.client_state(640, 360);
        assert_eq!(state.screen, Screen::Online, "{scene}: still playing");
        assert!((state.yaw_deg - yaw_deg).abs() < 1e-3, "{scene}: {state:?}");
        let ship = state.position.expect("a ship");
        let drift: f64 = (0..3).map(|i| (ship[i] - parked[i]).abs()).sum();
        assert!(drift < 0.01, "{scene}: the ship moved {drift} m");
        assert!(client.app.title().contains(scene.name()), "{scene}");
        assert_eq!(state.scene.as_deref(), Some(scene.name()));
        assert_eq!(state.scene_revision, revision, "{scene}");
        // The world that is drawn is the new scene's, around the same ship.
        let world = World::new(scene);
        let expected = world.build(&client.app.scene(640, 360).camera);
        let bodies = client.app.client_state(640, 360).bodies.len();
        let drawn = client.app.scene(640, 360);
        assert_eq!(drawn.lights.len(), expected.lights.len(), "{scene}");
        if bodies == 0 {
            assert_eq!(drawn.instances.len(), expected.instances.len(), "{scene}");
            // ...plus the player's own character (chase camera).
            assert_eq!(drawn.models.len(), expected.models.len() + 1, "{scene}");
        } else {
            // ...plus the scene's physics bodies (the demo area's showcase):
            // a model per part, and coils of beads for their springs.
            assert!(drawn.instances.len() > expected.instances.len(), "{scene}");
            assert!(
                drawn.models.len() > expected.models.len() + 1 + bodies,
                "{scene}"
            );
        }
        assert!(!drawn.overlay.is_empty(), "{scene}: the HUD");
        // The camera is where the ship is in that scene: the spawn point
        // plus what the ship has flown, plus the chase offset behind it.
        let from_spawn = WorldPos::delta(world.spawn(), drawn.camera.position);
        assert!(
            (from_spawn[0] - parked[0]).abs() < 6.0 && from_spawn[1].abs() < 6.0,
            "{scene}: camera {from_spawn:?} for a ship at {parked:?}"
        );
    }

    // The same scene at a newer revision is not a change: the view stays.
    client.app.input(ViewerInput::Turn {
        dyaw: 0.5,
        dpitch: 0.0,
    });
    let yaw = client.app.client_state(1, 1).yaw_deg;
    set_scene(
        &rt,
        &shard,
        SceneState::new(Scene::DemoArea.id(), revision + 1),
    );
    for _ in 0..30 {
        client.frame();
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(client.app.world().scene(), Scene::DemoArea);
    assert_eq!(client.app.client_state(1, 1).yaw_deg, yaw);

    // A scene this build has never heard of: the default is drawn.
    set_scene(
        &rt,
        &shard,
        SceneState::new(Scene::Solar.id(), revision + 2),
    );
    client.until("switched to solar", |app| {
        app.world().scene() == Scene::Solar
    });
    set_scene(&rt, &shard, SceneState::new(SceneId(4_000), revision + 3));
    client.until("fell back to the default", |app| {
        app.world().scene() == Scene::DEFAULT
    });

    // A client that joins now is told the scene at once.
    set_scene(
        &rt,
        &shard,
        SceneState::new(Scene::Materials.id(), revision + 4),
    );
    let late = Dry::online(&rt, shard.addr());
    assert_eq!(late.app.world().scene(), Scene::Materials);

    drop(client);
    drop(late);
    shard.stop();
}

fn count_mesh(scene: &space_render::Scene, mesh: &Arc<space_render::MeshData>) -> usize {
    scene
        .models
        .iter()
        .filter(|model| Arc::ptr_eq(&model.mesh, mesh))
        .count()
}

/// Players see each other: a second client's character appears in the first
/// one's scene where the server has it, moves when it flies, and is gone when
/// it leaves. A ship nobody owns is drawn as the ship model. A player's own
/// character is drawn in the chase camera and not in first person.
#[test]
fn players_see_each_other_as_characters_and_unowned_ships_as_ships() {
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    // The materials scene has no models of its own: every model is a player.
    set_scene(&rt, &shard, SceneState::new(Scene::Materials.id(), 1));
    let mut alice = Dry::online(&rt, shard.addr());
    assert_eq!(alice.app.world().scene(), Scene::Materials);
    let (character, ship) = (character_mesh(), ship_mesh());
    assert_eq!(
        count_mesh(alice.app.scene(640, 360), &character),
        1,
        "herself"
    );

    let mut bob = Dry::online(&rt, shard.addr());
    alice.until("saw the other player", |app| {
        app.replica().is_some_and(|r| r.others().count() == 1)
    });
    bob.until("saw the first player", |app| {
        app.replica().is_some_and(|r| r.others().count() == 1)
    });
    for client in [&mut alice, &mut bob] {
        let scene = client.app.scene(640, 360);
        assert_eq!(count_mesh(scene, &character), 2, "both characters");
        assert_eq!(count_mesh(scene, &ship), 0);
    }
    // First person: one's own character is not drawn, the other one's is.
    alice.app.input(ViewerInput::Text('c'));
    assert_eq!(count_mesh(alice.app.scene(640, 360), &character), 1);

    // Bob flies up; Alice sees his character rise by what the server says.
    let seen_at = |app: &mut App| {
        let scene = app.scene(640, 360);
        let character = character_mesh();
        scene
            .models
            .iter()
            .find(|model| Arc::ptr_eq(&model.mesh, &character))
            .map(|model| model.position)
            .expect("the other player's character")
    };
    alice.frame();
    let before = seen_at(&mut alice.app);
    // Each joined on a free spawn point of its own: his feet are where the
    // server has him, seen from her eyes.
    let (her, his) = (
        alice.app.client_state(1, 1).position.expect("her ship"),
        bob.app.client_state(1, 1).position.expect("his ship"),
    );
    let apart: f64 = (0..3)
        .map(|i| (his[i] - her[i]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(apart > 0.6, "two solid ships {apart} m apart");
    assert!(
        (f64::from(before[0]) - (his[0] - her[0])).abs() < 0.01
            && (f64::from(before[2]) - (his[2] - her[2])).abs() < 0.01,
        "{before:?} for {his:?} seen from {her:?}"
    );
    assert!(
        (f64::from(before[1]) - (his[1] - her[1]) + f64::from(space_client::scenes::EYE_HEIGHT))
            .abs()
            < 0.01
    );
    bob.app.input(ViewerInput::Key {
        key: Key::Up,
        down: true,
    });
    bob.until("climbed", |app| {
        app.client_state(1, 1)
            .position
            .is_some_and(|p| p[1] > his[1] + 3.0)
    });
    bob.app.input(ViewerInput::Key {
        key: Key::Up,
        down: false,
    });
    bob.until("stopped", |app| app.client_state(1, 1).speed < 0.01);
    let height = bob.app.client_state(1, 1).position.expect("a ship")[1] - his[1];
    let deadline = Instant::now() + LONG;
    loop {
        alice.frame();
        let now = seen_at(&mut alice.app);
        if (f64::from(now[1] - before[1]) - height).abs() < 0.05 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the other player is drawn at {now:?}, the server has it {height} m up"
        );
        std::thread::sleep(Duration::from_millis(10));
    }

    // A ship nobody flies: the ship model.
    rt.block_on(async {
        let mut shard = shard.shard.lock().await;
        let pos = WorldPos::from_f64_metres_around(WorldPos::ZERO, [6.0, 0.0, -8.0]);
        shard.spawn_ship_for(0, pos);
    });
    alice.until("saw the unowned ship", |app| {
        app.replica()
            .is_some_and(|r| r.others().any(|s| s.owner == 0))
    });
    let scene = alice.app.scene(640, 360);
    assert_eq!(count_mesh(scene, &ship), 1);
    assert_eq!(
        count_mesh(scene, &character),
        1,
        "first person: only the other"
    );

    // Bob leaves: his character goes.
    drop(bob);
    alice.until("saw the other player leave", |app| {
        app.replica()
            .is_some_and(|r| r.others().all(|s| s.owner == 0))
    });
    assert_eq!(count_mesh(alice.app.scene(640, 360), &character), 0);
    drop(alice);
    shard.stop();
}

/// Losing the server while playing returns to the connect screen with a
/// message, the address still in the field. The connection was made by hand
/// (no `--server`), so a person is there: nothing is retried, however long
/// the client is left alone.
#[test]
fn losing_the_server_returns_to_the_connect_screen_with_a_message() {
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let addr = shard.addr();
    let mut client = Dry::by_hand(&rt, addr);
    // The shard goes away without a word.
    shard.stop();
    client.until("noticed", |app| app.screen() != Screen::Online);
    let state = client.app.client_state(640, 360);
    assert_eq!(state.screen, Screen::Error);
    assert_eq!(state.address, addr.to_string());
    let error = state.error.expect("a message");
    assert!(
        error.contains("lost the connection") && error.contains(&addr.to_string()),
        "{error}"
    );
    assert_eq!(
        (state.shard, state.session, state.position),
        (None, None, None)
    );
    assert!(!client.app.should_quit());
    // Longer than the first pause of an unattended client: still there.
    let watched = Instant::now();
    while watched.elapsed() < Duration::from_secs(3) {
        client.frame();
        let state = client.app.client_state(640, 360);
        assert_eq!((state.screen, state.retry_in_s), (Screen::Error, None));
        std::thread::sleep(Duration::from_millis(20));
    }
    // Enter tries again (and fails: nobody is there any more); Esc quits —
    // this is a window — and the foot of the screen says both.
    assert_eq!(
        client.app.connect_keys(),
        Some("Enter tries again    Esc quits")
    );
    client.app.input(ViewerInput::Key {
        key: Key::Enter,
        down: true,
    });
    assert_eq!(client.app.screen(), Screen::Connecting);
    client.app.input(ViewerInput::Key {
        key: Key::Escape,
        down: true,
    });
    assert_eq!(client.app.screen(), Screen::Connect);
    client.app.input(ViewerInput::Key {
        key: Key::Escape,
        down: true,
    });
    assert!(client.app.should_quit());
}

/// Every scene renders through the application's own path — the world, a
/// player's character in it, the HUD — at every preset, with lit pixels and
/// zero validation messages.
#[test]
fn every_scene_renders_with_a_player_in_it_at_every_preset() {
    let Some(ctx) = gpu() else {
        return;
    };
    let (width, height) = (320, 180);
    let mut pictures = Vec::new();
    for scene in Scene::ALL {
        let world = World::new(scene);
        let (yaw, pitch) = world.initial_view();
        let view = space_client::input::InputState::looking(yaw, pitch);
        // The chase camera's place: behind and above the spawn point.
        let forward = view.forward();
        let back = [
            f64::from(-forward[0] * CHASE_DISTANCE),
            f64::from(CHASE_HEIGHT - forward[1] * CHASE_DISTANCE),
            f64::from(-forward[2] * CHASE_DISTANCE),
        ];
        let camera = world.camera(
            WorldPos::from_f64_metres_around(world.spawn(), back),
            forward,
            width as f32 / height as f32,
        );
        for preset in QualityPreset::ALL {
            let mut frame = world.build(&camera);
            let feet = WorldPos::from_f64_metres_around(
                world.spawn(),
                [0.0, -f64::from(space_client::scenes::EYE_HEIGHT), 0.0],
            );
            frame.models.push(space_render::ModelInstance {
                mesh: character_mesh(),
                position: space_client::scenes::relative(&camera, feet),
                rotation: space_render::rotation_y(yaw),
                scale: 1.0,
            });
            let out = ctx
                .render_offscreen(&frame, width, height, &world.options(preset))
                .unwrap_or_else(|e| panic!("{scene} at {preset}: {e}"));
            assert_eq!(out.validation_messages, 0, "{scene} at {preset}");
            let picture = Frame::new(1, out.width, out.height, out.pixels).expect("a frame");
            let seen = look(&picture, None);
            assert!(seen.mean > 4.0, "{scene} at {preset} is black: {seen:?}");
            assert!(
                seen.white < 0.2,
                "{scene} at {preset} is blown out: {seen:?}"
            );
            // The character stands in the middle of the lower half, lit.
            let middle = [width / 2 - 20, height / 2, width / 2 + 20, height - 10];
            let player = look(&picture, Some(middle));
            assert!(
                player.mean > 25.0,
                "{scene} at {preset}: the player's character is in the dark: {player:?}"
            );
            if preset == QualityPreset::High {
                pictures.push(picture);
            }
        }
    }
    // Four scenes, four different pictures.
    for (i, a) in pictures.iter().enumerate() {
        for b in &pictures[i + 1..] {
            assert!(a.rgba != b.rgba, "two scenes render the same picture");
        }
    }
    // The demo area again: the meshes are still on the device (one upload
    // each, however many frames drew them).
    let stats = ctx.model_cache_stats();
    let world = World::default();
    let camera = world.camera(world.spawn(), [0.0, 0.0, -1.0], 16.0 / 9.0);
    for _ in 0..3 {
        ctx.render_offscreen(
            &world.build(&camera),
            width,
            height,
            &world.options(QualityPreset::High),
        )
        .expect("demo area");
    }
    let after = ctx.model_cache_stats();
    let meshes = {
        let mut seen: Vec<*const space_render::MeshData> = world
            .build(&camera)
            .models
            .iter()
            .map(|model| Arc::as_ptr(&model.mesh))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        seen.len() as u64
    };
    assert!(
        after.uploads - stats.uploads <= meshes,
        "three frames of the demo area uploaded {} models for {meshes} meshes",
        after.uploads - stats.uploads
    );
}

/// The pixel stream: a browser stand-in connects over WebSocket, types the
/// shard's address into the connect screen and presses Enter — and the client
/// goes online. Frames arrive as JPEG; when the viewer vanishes while holding
/// a key, the key is released.
#[test]
fn a_stream_viewer_types_the_address_and_plays() {
    use futures_util::{SinkExt, StreamExt};
    use space_stream::protocol::{decode_frame, FLAG_DOWNSCALED, FLAG_KEYFRAME};
    use space_stream::{jpeg_dimensions, StreamServer};
    use tokio_tungstenite::tungstenite::Message;

    if gpu().is_none() {
        return;
    }
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let (server, stream) = rt
        .block_on(StreamServer::bind("127.0.0.1:0"))
        .expect("stream bind");
    let stream = Arc::new(stream);
    let hub = Hub::new();
    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let (handle, hub, stop, stream) = (
            rt.handle().clone(),
            Arc::clone(&hub),
            Arc::clone(&stop),
            Arc::clone(&stream),
        );
        std::thread::spawn(move || {
            let ctx = RenderContext::new(&ContextOptions::default()).expect("render context");
            let mut app = App::new(
                AppConfig {
                    server: None,
                    quality: QualityPreset::Ci,
                    model: None,
                    escape_quits: false,
                },
                handle,
                hub,
            );
            let config = HeadlessConfig {
                width: 64,
                height: 48,
                fps: 60,
                frames: None,
                // No harness is "attached": nothing is rendered unwatched.
                idle_unwatched: true,
            };
            run_headless(&ctx, &mut app, &config, Some(&stream), &stop).expect("streaming")
        })
    };
    // Unwatched, the client idles: no frame is rendered.
    std::thread::sleep(Duration::from_millis(300));
    assert_eq!(hub.state().frame, 0, "rendered with nobody watching");

    let ws_addr = server.local_addr();
    let address = shard.addr().to_string();
    let frames = rt.block_on(async {
        let tcp = tokio::net::TcpStream::connect(ws_addr).await.unwrap();
        let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://{ws_addr}/"), tcp)
            .await
            .unwrap();
        let send = |input: ViewerInput| Message::Binary(input.encode().into());
        let stroke = |key| {
            [
                send(ViewerInput::Key { key, down: true }),
                send(ViewerInput::Key { key, down: false }),
            ]
        };
        // Clear the prefilled address, type the shard's, press Enter.
        for _ in 0.."127.0.0.1:7777".len() {
            for message in stroke(Key::Backspace) {
                ws.send(message).await.unwrap();
            }
        }
        for c in address.chars() {
            ws.send(send(ViewerInput::Text(c))).await.unwrap();
        }
        for message in stroke(Key::Enter) {
            ws.send(message).await.unwrap();
        }
        let mut frames = Vec::new();
        let mut holding = false;
        let deadline = Instant::now() + LONG;
        loop {
            let message = tokio::time::timeout(LONG, ws.next())
                .await
                .expect("timed out waiting for a frame")
                .expect("the stream ended")
                .expect("a message");
            if let Message::Binary(bytes) = message {
                frames.push(bytes.to_vec());
            }
            let state = hub.state();
            // Once the picture shows the world: hold W, as a player would.
            if state.screen == Screen::Online && !holding {
                holding = true;
                ws.send(send(ViewerInput::Key {
                    key: Key::Forward,
                    down: true,
                }))
                .await
                .unwrap();
                // (What a browser sends with it: the letter. Not a command.)
                ws.send(send(ViewerInput::Text('w'))).await.unwrap();
            }
            let flying = state.position.is_some_and(|p| p[2] < -1.0);
            if flying && frames.len() >= 3 {
                break;
            }
            assert!(Instant::now() < deadline, "never flew: {state:?}");
        }
        // The viewer vanishes without sending the key-up.
        drop(ws);
        frames
    });
    for frame in &frames {
        let (header, jpeg) = decode_frame(frame).expect("a stream frame");
        assert_eq!(header.flags, FLAG_KEYFRAME | FLAG_DOWNSCALED);
        assert_eq!(jpeg_dimensions(jpeg), Some((32, 24)), "half of 64x48");
    }
    // Its key is released: the brake assist stops the ship. (The harness asks
    // for frames, so the client keeps running without a viewer.)
    let deadline = Instant::now() + LONG;
    loop {
        let _ = hub.frame(Some(hub.state().frame), Duration::from_secs(5));
        let state = hub.state();
        if state.screen == Screen::Online && state.speed < 0.02 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "the ship of a viewer that left keeps flying: {state:?}"
        );
    }
    stop.store(true, Ordering::Relaxed);
    let report = worker.join().expect("the stream thread");
    assert!(report.streamed >= 3 && report.frames >= report.streamed);
    assert_eq!(report.validation_messages, 0);
    drop(server);
    shard.stop();
}

// ---- one world for everyone: the anchor ------------------------------------------

/// Two neighbouring shards as a split leaves them (the handoff test's
/// topology): A owns a two-cell-wide node and B the far octant carved out of
/// it — the cell the scene's spawn point lies in. Both know the same map.
fn neighbours(rt: &tokio::runtime::Runtime) -> (Running, Running) {
    let _inside = rt.enter();
    let region_a = NodeRef::new(63, [0, 0, 0]);
    let region_b = NodeRef::new(64, [1, 1, 1]);
    let a = start_shard_at(1, region_a, SocketAddr::from(([127, 0, 0, 1], 0)));
    let b = start_shard_at(2, region_b, SocketAddr::from(([127, 0, 0, 1], 0)));
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(region_a, ShardId::new(1), a.addr()),
            ShardRegion::new(region_b, ShardId::new(2), b.addr()),
        ],
    );
    rt.block_on(async {
        a.shard.lock().await.set_shard_map(map.clone());
        b.shard.lock().await.set_shard_map(map);
    });
    (a, b)
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    (0..3).map(|i| (a[i] - b[i]).powi(2)).sum::<f32>().sqrt()
}

/// Players who dial different shards see one world — and start in one place.
/// Both derive the same anchor from the map they were sent, so the scene's
/// objects, and each other, are in the same places for both. The one who
/// dials the shard that owns the scene's spawn point is welcomed there; the
/// one who dials the **other** shard is sent on to it (a `JoinRedirect`: a
/// new player starts on the spawn point, whatever address was typed) and
/// arrives next to the first — not in the middle of the cell of the shard
/// she dialed, and without a handoff.
///
/// (A shard that did not own the spawn point used to put the player on the
/// centre of its own region: half a cell below the scene here, on ground of
/// the other shard, to which she was handed at once. That is what this test
/// used to show for Alice.)
#[test]
fn two_clients_who_dial_two_shards_start_together_and_agree_where_the_scene_is() {
    let rt = runtime();
    let (a, b) = neighbours(&rt);
    // The material chart: no models of its own, so every model is a player,
    // and its first sphere is a landmark both clients can be asked about.
    for shard in [&a, &b] {
        set_scene(&rt, shard, SceneState::new(Scene::Materials.id(), 1));
    }
    let root_centre = NodeRef::new(63, [0, 0, 0]).center();
    // The scene's place in the root: 6 m, half a cell up and 4 m from the
    // corner where its cells meet — inside B's cell.
    let off_the_corner = [6.0, 512.0, 4.0];
    let scene_at = WorldPos::from_f64_metres_around(root_centre, off_the_corner);
    assert_eq!(scene_at.cell, [1, 1, 1], "in the octant B owns");

    // Alice dials A, which does not own the spawn point; Bob dials B, which
    // does. Alice first, so the spawn point itself is hers.
    let mut alice = Dry::online(&rt, a.addr());
    let mut bob = Dry::online(&rt, b.addr());
    for client in [&mut alice, &mut bob] {
        client.until("saw the other player", |app| {
            app.replica().is_some_and(|r| r.others().count() == 1)
        });
    }
    // Both play on B. Alice got there by being sent on — no handoff — and
    // her client says both addresses: the one she dialed, and the shard's.
    for (name, client) in [("Alice", &alice), ("Bob", &bob)] {
        let replica = client.app.replica().expect("online");
        assert_eq!(replica.shard, Some(ShardId::new(2)), "{name}");
        assert_eq!(replica.handoffs, 0, "{name}: a join redirect is no handoff");
    }
    let told = alice.app.client_state(640, 360);
    assert_eq!(told.address, a.addr().to_string(), "what she dialed");
    assert_eq!(told.shard_address, Some(b.addr().to_string()));
    let told = bob.app.client_state(640, 360);
    assert_eq!(told.address, b.addr().to_string());
    assert_eq!(told.shard_address, Some(b.addr().to_string()));
    rt.block_on(async {
        assert_eq!(a.shard.lock().await.stats().clients, 0, "A let nobody in");
        assert_eq!(b.shard.lock().await.stats().clients, 2);
    });

    // The same anchor: the scene's place in the root region.
    let anchor = |client: &Dry| client.app.replica().and_then(|r| r.anchor());
    assert_eq!(anchor(&alice), Some(scene_at));
    assert_eq!(anchor(&bob), Some(scene_at), "whichever shard was dialed");
    // Alice arrived on the scene's spawn point, Bob on the free point next to
    // it: a ship's width away on the same floor — not half a cell below.
    let from_scene = |client: &Dry| client.app.position_in_scene().expect("a ship");
    assert!(
        from_scene(&alice).iter().all(|c| c.abs() < 0.05),
        "{:?}",
        from_scene(&alice)
    );
    let beside = from_scene(&bob);
    let apart = (beside[0] * beside[0] + beside[2] * beside[2]).sqrt();
    assert!(
        beside[1].abs() < 0.05 && (1.0..16.0).contains(&apart),
        "Bob starts beside the spawn point: {beside:?}"
    );
    let _ = off_the_corner;
    // The harness is told both: where the ship is in the world, and where it
    // is in the scene — the same place, measured from the world's origin and
    // from the anchor.
    let origin = WorldPos::delta(WorldPos::ZERO, scene_at);
    for client in [&bob, &alice] {
        let told = client.app.client_state(640, 360);
        assert_eq!(told.scene_position, Some(from_scene(client)));
        let world = told.position.expect("a ship");
        for i in 0..3 {
            assert!(
                (world[i] - origin[i] - from_scene(client)[i]).abs() < 0.05,
                "{world:?} vs {:?}",
                from_scene(client)
            );
        }
    }

    // One world: where a player stands relative to the chart's first sphere
    // is the same in the player's own picture (the character in the chase
    // camera, model 0) and in the other one's (model 1).
    let from_landmark = |client: &mut Dry, model: usize| {
        let scene = client.app.scene(640, 360);
        assert_eq!(scene.models.len(), 2, "two players, nothing else");
        let landmark = scene.instances[0].center;
        let player = scene.models[model].position;
        [0, 1, 2].map(|i| player[i] - landmark[i])
    };
    let bob_by_alice = from_landmark(&mut alice, 1);
    let bob_by_himself = from_landmark(&mut bob, 0);
    assert!(
        distance(bob_by_alice, bob_by_himself) < 0.05,
        "Alice has Bob at {bob_by_alice:?} from the landmark, Bob has himself at {bob_by_himself:?}"
    );
    let alice_by_herself = from_landmark(&mut alice, 0);
    let alice_by_bob = from_landmark(&mut bob, 1);
    assert!(
        distance(alice_by_herself, alice_by_bob) < 0.05,
        "{alice_by_herself:?} / {alice_by_bob:?}"
    );
    // Both stand by the chart.
    assert!(distance(bob_by_himself, [0.0; 3]) < 30.0);
    assert!(distance(alice_by_herself, [0.0; 3]) < 30.0);

    drop(alice);
    drop(bob);
    a.stop();
    b.stop();
}

/// Players on neighbouring shards see each other as characters: each is
/// near the border on its own shard, and the other one arrives as the ghost
/// the neighbour sends — owner 0 there, but marked with the player visual —
/// and is drawn as a character, not as the ship model of a ship nobody flies.
///
/// Needs the shard to put `Visual(PLAYER)` on every player's ship (phase 2 of
/// the physics work).
#[test]
fn players_on_neighbouring_shards_see_each_other_as_characters() {
    let rt = runtime();
    let (a, b) = {
        let _inside = rt.enter();
        let region_a = NodeRef::new(64, [0, 0, 0]);
        let region_b = NodeRef::new(64, [1, 0, 0]);
        let a = start_shard_at(1, region_a, SocketAddr::from(([127, 0, 0, 1], 0)));
        let b = start_shard_at(2, region_b, SocketAddr::from(([127, 0, 0, 1], 0)));
        let map = ShardMap::new(
            1,
            vec![
                ShardRegion::new(region_a, ShardId::new(1), a.addr()),
                ShardRegion::new(region_b, ShardId::new(2), b.addr()),
            ],
        );
        rt.block_on(async {
            a.shard.lock().await.set_shard_map(map.clone());
            b.shard.lock().await.set_shard_map(map);
        });
        (a, b)
    };
    // The material chart: no models of its own, every model is a player.
    for shard in [&a, &b] {
        set_scene(&rt, shard, SceneState::new(Scene::Materials.id(), 1));
    }
    // Both start on A, which owns the spawn point — Bob although he dials B.
    let mut alice = Dry::online(&rt, a.addr());
    let mut bob = Dry::online(&rt, b.addr());
    assert_eq!(
        bob.app.replica().expect("online").shard,
        Some(ShardId::new(1)),
        "B sent Bob on to the spawn point's shard"
    );
    // Both a few metres from the border between the two cells (x = 512 m),
    // each on its own side: Alice stays on A; Bob is put across the border,
    // and A hands his ship to B (that is how a player gets onto B).
    let place = |session: u64, x: f64| {
        rt.block_on(async {
            let mut shard = a.shard.lock().await;
            let ship = shard.ship_of(session).expect("the player's ship");
            let at = WorldPos::from_f64_metres_around(WorldPos::ZERO, [x, 0.0, 0.0]);
            assert!(shard.set_state(ship, space_sim::ShipState::at_rest(at)));
        });
    };
    let session = |client: &Dry| client.app.replica().expect("online").session;
    place(session(&alice), 505.0);
    place(session(&bob), 519.0);
    bob.until("was handed to B", |app| {
        app.replica()
            .is_some_and(|r| r.shard == Some(ShardId::new(2)) && r.handoffs == 1)
    });
    for client in [&mut alice, &mut bob] {
        client.until("saw the player across the border", |app| {
            app.replica()
                .is_some_and(|r| r.other_ships().any(|t| t.is_player()))
        });
    }
    let (character, ship) = (character_mesh(), ship_mesh());
    for (name, client, shard, handoffs) in [("Alice", &mut alice, 1, 0), ("Bob", &mut bob, 2, 1)] {
        let replica = client.app.replica().unwrap();
        assert_eq!(
            (replica.shard, replica.handoffs),
            (Some(ShardId::new(shard)), handoffs),
            "{name} is on its own side of the border"
        );
        let scene = client.app.scene(640, 360);
        assert_eq!(count_mesh(scene, &character), 2, "{name}: both characters");
        assert_eq!(count_mesh(scene, &ship), 0, "{name}: no ship model");
        assert_eq!(client.app.client_state(1, 1).bodies.len(), 0, "{name}");
    }
    drop(alice);
    drop(bob);
    a.stop();
    b.stop();
}

/// A split (and the merge that undoes it) while a client is playing does not
/// move its scene, and a client that joins after the split derives the very
/// same anchor from the new map. The cluster's root lies far from the world
/// origin, as an orchestrated one does: the position in the scene (what the
/// HUD shows) is the ship's place on the floor, not that distance.
#[test]
fn a_split_while_connected_does_not_move_the_scene() {
    let rt = runtime();
    let root = NodeRef::new(40, [3, -2, 5]);
    let shard = {
        let _inside = rt.enter();
        start_shard_at(1, root, SocketAddr::from(([127, 0, 0, 1], 0)))
    };
    let mut early = Dry::online(&rt, shard.addr());
    let anchor = early.app.replica().and_then(|r| r.anchor());
    // The scene's place in a root of many cells: 6 m, half a cell up and 4 m
    // from the corner where its cells meet, clear of every border a split
    // can draw.
    assert_eq!(
        anchor,
        Some(WorldPos::from_f64_metres_around(
            root.center(),
            [6.0, 512.0, 4.0]
        ))
    );
    // Far from the world origin in absolute terms; on the spawn point in the
    // scene's.
    let absolute = early.app.client_state(1, 1).position.expect("a ship");
    assert!(absolute.iter().all(|c| c.abs() > 1.0e9), "{absolute:?}");
    let arrived = early.app.position_in_scene().expect("a ship");
    assert!(arrived.iter().all(|c| c.abs() < 0.01), "{arrived:?}");
    assert_eq!(early.app.client_state(1, 1).scene_position, Some(arrived));
    // Walk a few metres, so "nothing moved" is not about the spawn point only.
    early.app.input(ViewerInput::Key {
        key: Key::Right,
        down: true,
    });
    early.until("moved", |app| {
        app.position_in_scene().is_some_and(|p| p[0] > 2.0)
    });
    early.app.input(ViewerInput::Key {
        key: Key::Right,
        down: false,
    });
    early.until("stopped", |app| app.client_state(1, 1).speed < 0.01);
    let parked = early.app.position_in_scene().expect("a ship");
    assert!(parked[0] > 2.0 && parked[1].abs() < 0.01, "{parked:?}");
    // The absolute position moved by the same few metres.
    let now = early.app.client_state(1, 1).position.expect("a ship");
    assert!(((now[0] - absolute[0]) - parked[0]).abs() < 0.01);
    let prop_before = early.app.scene(640, 360).models[0].position;

    // The split: an octant of the root goes to a new shard. (Not the octant
    // the players stand in — nobody is handed over; the map is what changes.)
    let child = NodeRef::new(41, [6, -4, 10]);
    let whole = ShardRegion::new(root, ShardId::new(1), shard.addr());
    let carved = ShardRegion::new(
        child,
        ShardId::new(2),
        SocketAddr::from(([127, 0, 0, 1], 9)),
    );
    let set_map = |map: ShardMap| {
        rt.block_on(async { shard.shard.lock().await.set_shard_map(map) });
    };
    // The new shard is listed first: the order of the map does not matter.
    set_map(ShardMap::new(2, vec![carved, whole.clone()]));
    let unmoved = |client: &mut Dry, when: &str| {
        for _ in 0..20 {
            client.frame();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            client.app.replica().and_then(|r| r.anchor()),
            anchor,
            "{when}"
        );
        let now = client.app.position_in_scene().expect("a ship");
        assert!(
            (0..3).all(|i| (now[i] - parked[i]).abs() < 0.05),
            "{when}: {now:?}"
        );
        let prop = client.app.scene(640, 360).models[0].position;
        assert!(distance(prop, prop_before) < 0.05, "{when}: {prop:?}");
        assert_eq!(client.app.screen(), Screen::Online, "{when}");
    };
    unmoved(&mut early, "after the split");

    // A client that joins now is given the new map — and the same anchor.
    let mut late = Dry::online(&rt, shard.addr());
    assert_eq!(late.app.replica().and_then(|r| r.anchor()), anchor);
    // Both stand on the same floor: the newcomer on the spawn point, the
    // early one where it walked to, and the newcomer draws it there.
    late.until("saw the early player", |app| {
        app.replica().is_some_and(|r| r.others().count() == 1)
    });
    let other = {
        let scene = late.app.scene(640, 360);
        let character = character_mesh();
        let players: Vec<[f32; 3]> = scene
            .models
            .iter()
            .filter(|model| Arc::ptr_eq(&model.mesh, &character))
            .map(|model| model.position)
            .collect();
        // The demo area's own characters come first, then the newcomer's
        // (chase camera), then the other player's.
        let own = players[players.len() - 2];
        let early_one = players[players.len() - 1];
        [0, 1, 2].map(|i| f64::from(early_one[i] - own[i]))
    };
    assert!(
        (0..3).all(|i| (other[i] - parked[i]).abs() < 0.05),
        "the early player is drawn {other:?} from the spawn point, it stands at {parked:?}"
    );

    // The merge: the octant goes back. Nothing moves for anybody.
    set_map(ShardMap::new(3, vec![whole]));
    unmoved(&mut early, "after the merge");
    assert_eq!(late.app.replica().and_then(|r| r.anchor()), anchor);

    drop(early);
    drop(late);
    shard.stop();
}

// ---- staying connected without a person ------------------------------------------

/// A UDP port on loopback that nothing is bound to right now.
fn free_port() -> SocketAddr {
    std::net::UdpSocket::bind("127.0.0.1:0")
        .and_then(|socket| socket.local_addr())
        .expect("a free port")
}

/// Start a shard on exactly `addr` (it may take the system a moment to give a
/// port back that another shard has just closed).
fn start_shard_on(rt: &tokio::runtime::Runtime, addr: SocketAddr) -> Running {
    let _inside = rt.enter();
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match ShardNet::bind(addr) {
            Ok(net) => return serve_shard(1, NodeRef::new(64, [0, 0, 0]), net),
            Err(e) => {
                assert!(Instant::now() < deadline, "cannot bind {addr}: {e:?}");
                std::thread::sleep(Duration::from_millis(100));
            }
        }
    }
}

/// A client that was given its server on the command line — a container that
/// starts before its shard — does not sit on the connect screen: it shows why
/// the attempt failed, counts down, tries again, and is in the game as soon
/// as the shard is there. When the shard goes away and comes back, so does
/// the client. Nobody presses a key — and when somebody does type, the client
/// stops retrying.
#[test]
fn a_client_started_before_its_shard_connects_when_the_shard_appears() {
    let rt = runtime();
    let addr = free_port();
    let mut client = Dry::new(&rt, Some(addr.to_string()));
    assert_eq!(client.app.screen(), Screen::Connecting);
    let online = |app: &App| {
        app.screen() == Screen::Online && app.replica().is_some_and(|r| r.state.is_some())
    };

    // Nobody is there: the reason, and a countdown to the next attempt.
    client.until("failed the first time", |app| app.screen() == Screen::Error);
    let state = client.app.client_state(640, 360);
    assert!(
        state
            .error
            .as_deref()
            .is_some_and(|e| e.contains("no answer")),
        "{state:?}"
    );
    assert_eq!(state.address, addr.to_string());
    assert!(
        state.retry_in_s.is_some_and(|s| s > 0.0 && s <= 2.0),
        "{state:?}"
    );
    assert_eq!((state.scene, state.shard), (None, None));
    assert!(client.app.title().contains("retrying in"));

    // The shard appears. The client joins it by itself.
    let shard = start_shard_on(&rt, addr);
    client.until("joined the shard that appeared", online);
    let joined = client.app.client_state(640, 360);
    assert_eq!((joined.retry_in_s, joined.error.as_deref()), (None, None));
    assert_eq!(joined.scene.as_deref(), Some("demo-area"));
    assert!(joined.session.is_some());

    // The shard restarts: gone without a word, back a moment later.
    shard.stop();
    client.until("noticed the loss", |app| app.screen() != Screen::Online);
    let lost = client.app.client_state(640, 360);
    assert!(
        lost.error
            .as_deref()
            .is_some_and(|e| e.contains("lost the connection")),
        "{lost:?}"
    );
    assert!(lost.retry_in_s.is_some(), "a lost server is retried too");
    let shard = start_shard_on(&rt, addr);
    client.until("rejoined the shard that came back", online);
    let back = client.app.client_state(640, 360);
    assert_eq!((back.retry_in_s, back.error.as_deref()), (None, None));
    // (A fresh shard numbers its sessions from 1 again; what matters is that
    // the ship is there, at rest on the spawn point.)
    assert!(back.session.is_some() && back.position.is_some());
    assert!(back.speed < 0.01, "{back:?}");
    assert!(!client.app.should_quit());

    // The shard goes away for good, and somebody is at the keyboard now:
    // one typed character, and the client leaves the address alone.
    shard.stop();
    client.until("noticed the second loss", |app| {
        app.screen() == Screen::Error
    });
    assert!(client.app.client_state(640, 360).retry_in_s.is_some());
    client.app.input(ViewerInput::Text('1'));
    let typed = client.app.client_state(640, 360);
    assert_eq!(typed.retry_in_s, None, "typing takes over");
    assert_eq!(typed.address, format!("{addr}1"));
    // Longer than the first pause: no attempt is made.
    let watched = Instant::now();
    while watched.elapsed() < Duration::from_secs(3) {
        client.frame();
        let state = client.app.client_state(640, 360);
        assert!(
            state.screen != Screen::Connecting && state.retry_in_s.is_none(),
            "{state:?}"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(!client.app.should_quit());
}

// ---- the scene in the harness state ----------------------------------------------

/// A script can see which scene is drawn and wait for a switch:
/// `client.state().scene` and `.scene_revision`.
#[test]
fn a_lua_script_waits_for_a_scene_switch() {
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let mut client = Dry::new(&rt, None);
    let control: Arc<dyn Control> = client.app.hub().clone();
    let address = shard.addr();
    let source = format!(
        r#"local s = client.state()
assert(s.scene == nil and s.scene_revision == 0, "a scene before any server")
assert(s.retry_in_s == nil, "retrying without a server on the command line")
client.connect("{address}", 20)
client.wait_until(function() return client.state().position ~= nil end, 10)
s = client.state()
assert(s.scene == "demo-area", "expected the default scene, got " .. tostring(s.scene))
assert(#s.scene_position == 3 and math.abs(s.scene_position[1]) < 0.01, "not on the spawn point")
local before = s.scene_revision
client.log("drawing " .. s.scene .. " at revision " .. before)
-- The cue for the operator (the test) to switch the scene.
client.turn(10, 0)
client.wait_until(function() return client.state().scene == "solar" end, 20)
s = client.state()
assert(s.scene_revision > before, "the revision did not go up")
assert(s.screen == "online" and s.position ~= nil, "the switch dropped the player")
client.log("now " .. s.scene .. " at revision " .. s.scene_revision)
client.disconnect()
client.wait_until(function() return client.state().screen == "connect" end, 5)
assert(client.state().scene == nil, "a scene after leaving")
"#
    );
    let script = std::thread::spawn(move || {
        space_harness::run_script(
            control,
            &source,
            &space_harness::ScriptConfig::default(),
            &space_harness::StopHandle::new(),
        )
    });
    // The client's loop, and the operator: when the script gives its cue (it
    // has read the first scene), switch to the solar system.
    let mut switched = false;
    let deadline = Instant::now() + Duration::from_secs(60);
    while !script.is_finished() {
        client.frame();
        let state = client.app.client_state(640, 360);
        if !switched && state.scene.is_some() && (state.yaw_deg - 10.0).abs() < 0.5 {
            set_scene(&rt, &shard, SceneState::new(Scene::Solar.id(), 7));
            switched = true;
        }
        assert!(Instant::now() < deadline, "the script never ended");
        std::thread::sleep(Duration::from_millis(5));
    }
    let report = script.join().expect("the script thread");
    assert!(report.ok, "{:?}\n{:?}", report.error, report.output);
    assert_eq!(
        report.output,
        vec![
            "drawing demo-area at revision 0".to_owned(),
            "now solar at revision 7".to_owned(),
        ]
    );
    drop(client);
    shard.stop();
}

// ---- who can end the client ------------------------------------------------------

/// A headless client is somebody's server-side process: Escape on its connect
/// screen — once, repeatedly, with and without a retry loop running — never
/// ends it, and it stays responsive: the address typed afterwards and Enter
/// connect. It ends when the harness says so.
#[test]
fn escape_never_ends_a_headless_client() {
    if gpu().is_none() {
        return;
    }
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let address = shard.addr().to_string();
    let config = HeadlessConfig {
        width: 320,
        height: 180,
        fps: 60,
        ..HeadlessConfig::default()
    };
    let retype = |client: &Headless, text: &str| {
        let hub = &client.hub;
        hub.stroke(Key::End).unwrap();
        for _ in 0..hub.state().address.len() {
            hub.stroke(Key::Backspace).unwrap();
        }
        client.wait("emptied the field", |s| s.address.is_empty());
        hub.type_text(text).unwrap();
        client.wait("typed the address", |s| s.address == text);
    };
    // Press Escape `times` times and let the client take a few frames over it.
    let escape = |client: &Headless, times: usize| {
        for _ in 0..times {
            client.hub.stroke(Key::Escape).unwrap();
        }
        let seen = client.hub.state().frame;
        client.wait("kept rendering", |s| s.frame >= seen + 3);
        assert!(!client.thread.is_finished(), "Escape ended the client");
    };
    let online = |s: &ClientState| s.screen == Screen::Online && s.position.is_some();

    // --- started on the connect screen (no `--server`): nothing is retried
    let client = Headless::start(&rt, None, config);
    client.wait("showed the connect screen", |s| {
        s.frame > 2 && s.screen == Screen::Connect
    });
    escape(&client, 1);
    escape(&client, 6);
    assert_eq!(client.hub.state().screen, Screen::Connect);
    // Still listening: the typed address and Enter connect.
    retype(&client, &address);
    client.hub.stroke(Key::Enter).unwrap();
    client.wait("came online", online);
    // Escape while playing leaves the shard; on the connect screen it is
    // back to doing nothing, and Enter reconnects with what is in the field.
    escape(&client, 1);
    let left = client.wait("left the shard", |s| s.screen == Screen::Connect);
    assert_eq!(
        (left.address.as_str(), left.retry_in_s),
        (address.as_str(), None)
    );
    escape(&client, 4);
    assert_eq!(client.hub.state().screen, Screen::Connect);
    client.hub.stroke(Key::Enter).unwrap();
    client.wait("came back online", online);
    // The harness's quit is what ends it.
    let report = client.finish();
    assert!(report.frames > 20, "{report:?}");
    assert_eq!(report.validation_messages, 0, "{report:?}");

    // --- started with `--server` before its shard: the retry loop is running
    let dead = free_port();
    let client = Headless::start(&rt, Some(dead.to_string()), config);
    let retrying = client.wait("counted down to a retry", |s| {
        s.screen == Screen::Error && s.retry_in_s.is_some()
    });
    assert!(retrying.error.is_some_and(|e| e.contains("no answer")));
    // The first Escape stops the retrying; the reason stays on the screen.
    escape(&client, 1);
    let stopped = client.hub.state();
    assert_eq!((stopped.screen, stopped.retry_in_s), (Screen::Error, None));
    escape(&client, 6);
    // Longer than the first pause: nothing was retried, and nothing ended.
    std::thread::sleep(Duration::from_millis(2_500));
    let still = client.hub.state();
    assert_eq!((still.screen, still.retry_in_s), (Screen::Error, None));
    assert!(!client.thread.is_finished());
    // The way back is Enter: the address in the field is tried again, and
    // the client keeps at it by itself from then on.
    client.hub.stroke(Key::Enter).unwrap();
    client.wait("tried again", |s| s.screen == Screen::Connecting);
    client.wait("is retrying again", |s| {
        s.screen == Screen::Error && s.retry_in_s.is_some()
    });
    // Escape during an attempt cancels it, and that is all.
    client.wait("started the next attempt", |s| {
        s.screen == Screen::Connecting
    });
    escape(&client, 1);
    let cancelled = client.hub.state();
    assert_eq!(
        (cancelled.screen, cancelled.retry_in_s),
        (Screen::Connect, None)
    );
    // Somebody who knows where the shard is types it: online.
    retype(&client, &address);
    client.hub.stroke(Key::Enter).unwrap();
    client.wait("came online", online);
    let report = client.finish();
    assert_eq!(report.validation_messages, 0, "{report:?}");
    shard.stop();
}

/// A browser viewer of the pixel stream, as far as the client can tell.
type Viewer = tokio_tungstenite::WebSocketStream<tokio::net::TcpStream>;

/// What a viewer sends for `times` presses of a text-entry key.
async fn viewer_strokes(ws: &mut Viewer, key: Key, times: usize) {
    use futures_util::SinkExt;
    use tokio_tungstenite::tungstenite::Message;
    for _ in 0..times {
        for down in [true, false] {
            let input = ViewerInput::Key { key, down };
            ws.send(Message::Binary(input.encode().into()))
                .await
                .expect("the stream takes the key");
        }
    }
}

/// Watch the stream (a viewer that does not read stalls it) until `done` says
/// so about the client's state.
async fn viewer_until(
    ws: &mut Viewer,
    hub: &Hub,
    what: &str,
    done: impl Fn(&ClientState) -> bool,
) -> ClientState {
    use futures_util::StreamExt;
    let deadline = Instant::now() + LONG;
    loop {
        let state = hub.state();
        if done(&state) {
            return state;
        }
        assert!(Instant::now() < deadline, "never {what}: {state:?}");
        tokio::time::timeout(LONG, ws.next())
            .await
            .expect("timed out waiting for a frame")
            .expect("the stream ended")
            .expect("a message");
    }
}

/// The pixel stream is open to whoever can reach its port. Such a viewer can
/// press Escape as often as it likes — while the client retries by itself,
/// on the error left behind, on the idle connect screen: it stops the
/// retrying, and nothing else. The client keeps running, keeps streaming and
/// keeps listening, Enter brings the retrying back, and the address the
/// viewer types, with Enter, connects.
#[test]
fn no_viewer_can_end_a_stream_client_with_escape() {
    use futures_util::SinkExt;
    use space_stream::StreamServer;
    use tokio_tungstenite::tungstenite::Message;

    if gpu().is_none() {
        return;
    }
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let dead = free_port();
    let (server, stream) = rt
        .block_on(StreamServer::bind("127.0.0.1:0"))
        .expect("stream bind");
    let stream = Arc::new(stream);
    let hub = Hub::new();
    let stop = Arc::new(AtomicBool::new(false));
    let worker = {
        let (handle, hub, stop, stream) = (
            rt.handle().clone(),
            Arc::clone(&hub),
            Arc::clone(&stop),
            Arc::clone(&stream),
        );
        std::thread::spawn(move || {
            let ctx = RenderContext::new(&ContextOptions::default()).expect("render context");
            // What `--stream ADDR --server ADDR` runs, its shard not being
            // there: a client that keeps trying by itself.
            let mut app = App::new(
                AppConfig {
                    server: Some(dead.to_string()),
                    quality: QualityPreset::Ci,
                    model: None,
                    escape_quits: false,
                },
                handle,
                hub,
            );
            let config = HeadlessConfig {
                width: 64,
                height: 48,
                fps: 60,
                frames: None,
                idle_unwatched: true,
            };
            run_headless(&ctx, &mut app, &config, Some(&stream), &stop).expect("streaming")
        })
    };
    let ws_addr = server.local_addr();
    let address = shard.addr().to_string();
    rt.block_on(async {
        let tcp = tokio::net::TcpStream::connect(ws_addr).await.unwrap();
        let (mut ws, _) = tokio_tungstenite::client_async(format!("ws://{ws_addr}/"), tcp)
            .await
            .unwrap();
        let ws = &mut ws;
        // A few frames later the client has taken everything sent so far.
        async fn settle(ws: &mut Viewer, hub: &Hub) -> ClientState {
            let seen = hub.state().frame;
            viewer_until(ws, hub, "kept streaming", |s| s.frame >= seen + 3).await
        }
        let online = |s: &ClientState| s.screen == Screen::Online && s.position.is_some();

        // --- the client is retrying by itself
        let retrying = viewer_until(ws, &hub, "counted down to a retry", |s| {
            s.screen == Screen::Error && s.retry_in_s.is_some()
        })
        .await;
        assert_eq!(retrying.address, dead.to_string());
        // One Escape: the retrying stops, the reason stays.
        viewer_strokes(ws, Key::Escape, 1).await;
        let stopped = settle(ws, &hub).await;
        assert_eq!((stopped.screen, stopped.retry_in_s), (Screen::Error, None));
        // Escape again and again: nothing — for longer than the first pause.
        viewer_strokes(ws, Key::Escape, 6).await;
        let watched = Instant::now();
        while watched.elapsed() < Duration::from_millis(2_500) {
            let state = settle(ws, &hub).await;
            assert_eq!((state.screen, state.retry_in_s), (Screen::Error, None));
        }
        assert!(!worker.is_finished(), "Escape ended the client");

        // --- the way back: Enter, and the client keeps trying again
        viewer_strokes(ws, Key::Enter, 1).await;
        viewer_until(ws, &hub, "tried again", |s| s.screen == Screen::Connecting).await;
        viewer_until(ws, &hub, "is retrying again", |s| {
            s.screen == Screen::Error && s.retry_in_s.is_some()
        })
        .await;

        // --- the viewer types where the shard really is, and Enter
        viewer_strokes(ws, Key::End, 1).await;
        viewer_strokes(ws, Key::Backspace, dead.to_string().len()).await;
        viewer_until(ws, &hub, "emptied the field", |s| s.address.is_empty()).await;
        for c in address.chars() {
            ws.send(Message::Binary(ViewerInput::Text(c).encode().into()))
                .await
                .unwrap();
        }
        let typed = viewer_until(ws, &hub, "typed the address", |s| s.address == address).await;
        assert_eq!(typed.retry_in_s, None, "typing took over");
        viewer_strokes(ws, Key::Enter, 1).await;
        viewer_until(ws, &hub, "came online", online).await;

        // --- playing: Escape leaves the shard; then, on the idle connect
        // screen, it does nothing, and Enter reconnects
        viewer_strokes(ws, Key::Escape, 1).await;
        let left = viewer_until(ws, &hub, "left the shard", |s| s.screen == Screen::Connect).await;
        assert_eq!(
            (left.address.as_str(), left.retry_in_s),
            (address.as_str(), None)
        );
        viewer_strokes(ws, Key::Escape, 5).await;
        assert_eq!(settle(ws, &hub).await.screen, Screen::Connect);
        assert!(!worker.is_finished(), "Escape ended the client");
        viewer_strokes(ws, Key::Enter, 1).await;
        viewer_until(ws, &hub, "came back online", online).await;
    });
    // The viewer is gone; the client is still there, and ends when its front
    // end says so.
    std::thread::sleep(Duration::from_millis(300));
    assert!(!worker.is_finished());
    stop.store(true, Ordering::Relaxed);
    let report = worker.join().expect("the stream thread");
    assert!(report.streamed >= 20, "{report:?}");
    assert_eq!(report.validation_messages, 0, "{report:?}");
    drop(server);
    shard.stop();
}

// ---- the real binary -------------------------------------------------------------

fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("space-client-test-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    dir
}

struct Finished {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

/// Run the real `space-client` binary to its end (killed after two minutes).
fn run_client(args: &[&str], env_remove: &[&str]) -> Finished {
    let mut command = std::process::Command::new(env!("CARGO_BIN_EXE_space-client"));
    command
        .args(args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    for name in env_remove {
        command.env_remove(name);
    }
    let mut child = command.spawn().expect("start space-client");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        match child.try_wait().expect("wait for space-client") {
            Some(_) => break,
            None if Instant::now() >= deadline => {
                let _ = child.kill();
                break;
            }
            None => std::thread::sleep(Duration::from_millis(20)),
        }
    }
    let output = child.wait_with_output().expect("collect the output");
    Finished {
        code: output.status.code(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// `--script FILE` through the real binary: a script that types the address,
/// connects, flies and takes a screenshot passes (exit 0, the screenshot is
/// written under `--out-dir`); one whose assertion fails exits 1 with the
/// message and the line on stderr.
#[test]
fn the_script_flag_runs_a_lua_script_and_its_outcome_is_the_exit_code() {
    if gpu().is_none() {
        return;
    }
    let rt = runtime();
    let shard = {
        let _inside = rt.enter();
        start_shard()
    };
    let dir = scratch("script");
    let out_dir = dir.join("shots");
    let address = shard.addr().to_string();

    let passing = dir.join("pass.lua");
    let source = format!(
        r#"-- Connect through the connect screen, fly, look, take a picture.
local s = client.state()
assert(s.screen == "connect", "expected the connect screen, got " .. s.screen)
client.press("end")
for _ = 1, #s.address do client.press("backspace") end
client.type("{address}")
client.press("enter")
client.wait_until(function() return client.state().screen == "online" end, 20)
client.wait_until(function() return client.state().position ~= nil end, 10)
local start = client.state().position
client.hold("w", 1.0)
client.turn(25, -5)
client.wait_until(function() return client.state().speed < 0.05 end, 20)
local now = client.state()
assert(now.position[3] < start[3] - 1, "the ship did not move forward")
assert(math.abs(now.yaw_deg - 25) < 0.5, "the view did not turn")
print("moved " .. string.format("%.1f", start[3] - now.position[3]) .. " m")
local path = client.screenshot("arrived.png")
local r, g, b = client.pixel(math.floor(now.width / 2), now.height - 4)
assert(r + g + b > 60, "the floor in front of the camera is black")
"#
    );
    std::fs::write(&passing, source).expect("write the script");
    let base = [
        "--headless",
        "--size",
        "320x180",
        "--fps",
        "60",
        "--quality",
        "high",
    ];
    let mut args = base.to_vec();
    let (script, out) = (passing.to_string_lossy(), out_dir.to_string_lossy());
    args.extend(["--script", &script, "--out-dir", &out]);
    let passed = run_client(&args, &[]);
    assert_eq!(
        passed.code,
        Some(0),
        "stdout:\n{}\nstderr:\n{}",
        passed.stdout,
        passed.stderr
    );
    assert!(passed.stdout.contains("moved "), "{}", passed.stdout);
    assert!(passed.stderr.contains("pass.lua: ok"), "{}", passed.stderr);
    assert!(
        passed.stderr.contains(" 0 validation messages"),
        "{}",
        passed.stderr
    );
    let picture = std::fs::read(out_dir.join("arrived.png")).expect("the screenshot");
    assert!(picture.starts_with(b"\x89PNG\r\n\x1a\n"), "not a PNG");
    assert!(picture.len() > 320 * 180, "{} bytes", picture.len());

    // A failing script: exit 1, the message and its line on stderr.
    let failing = dir.join("fail.lua");
    std::fs::write(
        &failing,
        "local s = client.state()\n\
         client.wait_frames(2)\n\
         assert(s.screen == \"online\", \"deliberately failing: the screen is \" .. s.screen)\n\
         client.screenshot(\"never.png\")\n",
    )
    .expect("write the script");
    let mut args = base.to_vec();
    let script = failing.to_string_lossy();
    args.extend(["--script", &script, "--out-dir", &out]);
    let failed = run_client(&args, &[]);
    assert_eq!(
        failed.code,
        Some(1),
        "stdout:\n{}\nstderr:\n{}",
        failed.stdout,
        failed.stderr
    );
    assert!(
        failed
            .stderr
            .contains("fail.lua:3: deliberately failing: the screen is connect"),
        "{}",
        failed.stderr
    );
    assert!(failed.stderr.contains("FAILED"), "{}", failed.stderr);
    assert!(!out_dir.join("never.png").exists());

    // A script that cannot be read is a usage error, before anything starts.
    let missing = run_client(&["--headless", "--script", "no-such-script.lua"], &[]);
    assert_eq!(missing.code, Some(2), "{}", missing.stderr);
    assert!(missing.stderr.contains("no-such-script.lua"));

    shard.stop();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The harness API is remote control: on an address that is not loopback it
/// needs a token, and without one the client refuses to start — saying how to
/// give it one.
#[test]
fn the_harness_refuses_a_public_address_without_a_token() {
    let refused = run_client(
        &["--headless", "--harness", "0.0.0.0:0"],
        &["SPACE_HARNESS_TOKEN"],
    );
    assert_eq!(refused.code, Some(2), "{}", refused.stderr);
    assert!(
        refused.stderr.contains("without a token")
            && refused.stderr.contains("SPACE_HARNESS_TOKEN"),
        "{}",
        refused.stderr
    );
    // An address that cannot be bound at all is said so too.
    let unbindable = run_client(&["--headless", "--harness", "not-an-address"], &[]);
    assert_eq!(unbindable.code, Some(2), "{}", unbindable.stderr);
    assert!(unbindable.stderr.contains("--harness not-an-address"));
    // An old scene flag is ignored with a pointer to where its job went (a
    // deployment that still passes one must not lose its client): parsing
    // goes on past it, which the unknown argument after it shows. Checked on
    // the command line only, so this needs no Vulkan device - the CI runners
    // have none, and a client that had to start would exit 1 there. ...
    let ignored = run_client(&["--galaxy", "--no-such-flag"], &[]);
    assert_eq!(ignored.code, Some(2), "{}", ignored.stderr);
    assert!(
        ignored.stderr.contains("--galaxy is ignored")
            && ignored.stderr.contains("--scene")
            && ignored.stderr.contains("--no-such-flag"),
        "{}",
        ignored.stderr
    );
    // ... the other flags of the offline client are refused.
    let gone = run_client(&["--window"], &[]);
    assert_eq!(gone.code, Some(2));
    assert!(gone.stderr.contains("is gone"), "{}", gone.stderr);
    let help = run_client(&["--help"], &[]);
    assert_eq!(help.code, Some(0));
    assert!(help.stdout.contains("--harness") && help.stdout.contains("--script"));
}
