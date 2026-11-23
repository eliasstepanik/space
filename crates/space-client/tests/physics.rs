//! The client's side of the physics, end to end against a real in-process
//! shard over QUIC on loopback that runs the demo area's physics showcase:
//!
//! * the showcase's bodies arrive, are listed in the harness state with
//!   their models, are drawn every part, and move smoothly at a frame rate
//!   far above the shard's 30 Hz (interpolated, not stepped);
//! * a player walks into the gravity room, lands on its floor and stands
//!   there without a jitter, the prediction agreeing with the shard to a
//!   centimetre, and lifts off again with the lift engines;
//! * a player flies into a floating crate in the zero-g corner and it drifts
//!   off.
//!
//! None of these render (the frames are built, not drawn): no GPU needed.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_client::app::{App, AppConfig};
use space_harness::{ClientState, Hub, Key, Screen, ViewerInput};
use space_protocol::{Scene, SceneState, ShardId};
use space_render::QualityPreset;
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::NodeRef;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

const LONG: Duration = Duration::from_secs(40);

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

/// A standalone shard on loopback showing the demo area, ticking at 30 Hz.
/// Must be called inside the runtime.
fn demo_shard(rt: &tokio::runtime::Runtime) -> Running {
    let running = {
        let _inside = rt.enter();
        let net = Arc::new(ShardNet::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind"));
        let shard = Arc::new(AsyncMutex::new(Shard::new(ShardConfig::new(
            ShardId::new(1),
            net.local_addr().expect("addr"),
            NodeRef::new(64, [0, 0, 0]),
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
    };
    assert!(rt.block_on(async {
        running
            .shard
            .lock()
            .await
            .set_scene(SceneState::new(Scene::DemoArea.id(), 1))
    }));
    running
}

/// A client core without a renderer, playing on `addr`.
struct Player {
    app: App,
}

impl Player {
    fn online(rt: &tokio::runtime::Runtime, addr: SocketAddr) -> Self {
        let mut player = Self {
            app: App::new(
                AppConfig {
                    server: Some(addr.to_string()),
                    quality: QualityPreset::Ci,
                    model: None,
                    escape_quits: false,
                },
                rt.handle().clone(),
                Hub::new(),
            ),
        };
        player.until("came online", |s| {
            s.screen == Screen::Online && s.scene_position.is_some()
        });
        player
    }

    /// One frame (built, not rendered); the state it leaves.
    fn frame(&mut self) -> ClientState {
        self.app.update();
        let _ = self.app.scene(640, 360);
        self.app.finish_frame(640, 360, None);
        self.app.client_state(640, 360)
    }

    fn until(&mut self, what: &str, done: impl Fn(&ClientState) -> bool) -> ClientState {
        let deadline = Instant::now() + LONG;
        loop {
            let state = self.frame();
            if done(&state) {
                return state;
            }
            assert!(Instant::now() < deadline, "never {what}: {state:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
    }

    /// Run frames for `seconds`, about every `every`.
    fn run(&mut self, seconds: f64, every: Duration) -> Vec<ClientState> {
        let end = Instant::now() + Duration::from_secs_f64(seconds);
        let mut states = Vec::new();
        while Instant::now() < end {
            states.push(self.frame());
            std::thread::sleep(every);
        }
        states
    }

    fn key(&mut self, key: Key, down: bool) {
        self.app.input(ViewerInput::Key { key, down });
    }

    /// Turn the view to look along +X (a quarter turn right).
    fn face_east(&mut self) {
        let yaw = self.app.controls().view.yaw;
        self.app.input(ViewerInput::Turn {
            dyaw: -std::f32::consts::FRAC_PI_2 - yaw,
            dpitch: 0.0,
        });
    }
}

/// The showcase arrives whole: every body of the scene definition, listed
/// nearest first with its model's name, every part drawn — and a body in
/// motion is drawn at a new place nearly every frame of a client running far
/// faster than the shard's 30 Hz (interpolated between ticks, not stepped
/// from one to the next).
#[test]
fn the_showcase_arrives_and_moves_smoothly() {
    let rt = runtime();
    let shard = demo_shard(&rt);
    let mut player = Player::online(&rt, shard.addr());
    let expected = space_scene::body_defs(Scene::DemoArea).len();
    let state = player.until("saw every body of the showcase", |s| {
        s.bodies.len() == expected
    });
    assert!(
        state
            .bodies
            .windows(2)
            .all(|w| w[0].distance <= w[1].distance),
        "nearest first"
    );
    for name in ["crate", "ball", "plank", "door", "chain", "pendulum"] {
        assert!(
            state.bodies.iter().any(|b| b.visual == name),
            "no {name} among {:?}",
            state.bodies
        );
    }
    let chain = state.bodies.iter().find(|b| b.visual == "chain").unwrap();
    assert_eq!(chain.parts, 6);
    assert_eq!(state.gravity, 0.0, "the spawn point is weightless");
    // Every part is a model of the frame (on top of the scene's own).
    let parts: u32 = state.bodies.iter().map(|b| b.parts).sum();
    let models = player.app.scene(640, 360).models.len();
    assert!(
        models > parts as usize + 50,
        "{models} models for {parts} parts"
    );

    // The spring pair oscillates for minutes in weightless space: follow one
    // of its crates at ~120 frames a second.
    let pair = state
        .bodies
        .iter()
        .filter(|b| b.visual == "crate" && b.position[0] < -12.0 && b.position[2] > 2.0)
        .map(|b| b.guid.clone())
        .next()
        .expect("a crate of the spring pair");
    let samples = player.run(1.5, Duration::from_millis(8));
    let xs: Vec<f64> = samples
        .iter()
        .filter_map(|s| s.bodies.iter().find(|b| b.guid == pair))
        .map(|b| b.position[0])
        .collect();
    let moves = xs.windows(2).filter(|w| w[0] != w[1]).count();
    let travel =
        xs.iter().fold(f64::MIN, |a, b| a.max(*b)) - xs.iter().fold(f64::MAX, |a, b| a.min(*b));
    println!(
        "{} frames, {} with the crate moved, travel {travel:.3} m",
        xs.len(),
        moves
    );
    assert!(xs.len() > 60, "{} frames in 1.5 s", xs.len());
    assert!(travel > 0.05, "the spring pair is still: {travel}");
    assert!(
        moves * 10 > xs.len() * 8,
        "drawn at a new place in only {moves} of {} frames: stepped at the tick rate",
        xs.len()
    );
    drop(player);
    shard.stop();
}

/// A player walks into the gravity room: the ship falls onto the floor and
/// stands there — the HUD's gravity and "landed" — without a jitter, the
/// prediction within a centimetre of the shard; the lift engines take it up
/// again.
#[test]
fn a_player_lands_in_the_gravity_room_and_lifts_off() {
    let rt = runtime();
    let shard = demo_shard(&rt);
    let mut player = Player::online(&rt, shard.addr());
    let start = player.frame().scene_position.unwrap();
    println!("spawned at {start:?}");
    player.face_east();
    // Up to speed, then let go: the brake assist (and the floor's friction
    // once inside) stops the ship a few metres into the room.
    player.key(Key::Forward, true);
    player.until("got going", |s| {
        s.scene_position.is_some_and(|p| p[0] > 6.0)
    });
    player.key(Key::Forward, false);
    let landed = player.until("landed", |s| s.landed && s.speed < 0.05);
    assert!((landed.gravity - 9.81).abs() < 0.05, "{}", landed.gravity);
    let p = landed.scene_position.unwrap();
    assert!(p[0] > 10.5 && p[0] < 18.5, "inside the room: {p:?}");
    // Standing: the ship (and with it the first-person camera) stays put to
    // the millimetre, and the prediction agrees with the shard.
    player.app.input(ViewerInput::Text('c'));
    let mut heights = Vec::new();
    for _ in 0..60 {
        player.frame();
        heights.push(player.app.scene(640, 360).camera.position);
        std::thread::sleep(Duration::from_millis(8));
    }
    let ys: Vec<f64> = heights
        .iter()
        .map(|p| space_world::WorldPos::delta(heights[0], *p)[1])
        .collect();
    let jitter = ys.iter().fold(0.0f64, |a, y| a.max(y.abs()));
    let state = player.frame();
    println!(
        "standing: camera moved {jitter:.5} m; corrections last {:.4} m, largest {:.4} m",
        state.correction_m, state.correction_max_m
    );
    assert!(jitter < 0.002, "the landed ship jitters by {jitter} m");
    assert!(state.landed);
    assert!(
        state.correction_max_m < 0.01,
        "the prediction is {} m off while standing",
        state.correction_max_m
    );
    // Lift-off.
    player.key(Key::Up, true);
    let up = player.until("lifted off", |s| {
        s.scene_position.is_some_and(|q| q[1] > p[1] + 1.0)
    });
    player.key(Key::Up, false);
    assert!(!up.landed);
    let gone = player.until("stopped climbing", |s| {
        s.velocity.is_some_and(|v| v[1] < 0.0)
    });
    println!(
        "lifted off to {:?}; corrections: largest {:.4} m",
        gone.scene_position, gone.correction_max_m
    );
    drop(player);
    shard.stop();
}

/// In the zero-g corner a floating crate drifts off when a ship flies into
/// it.
#[test]
fn a_ship_pushes_a_floating_crate() {
    let rt = runtime();
    let shard = demo_shard(&rt);
    let mut player = Player::online(&rt, shard.addr());
    let expected = space_scene::body_defs(Scene::DemoArea).len();
    let state = player.until("saw the bodies", |s| s.bodies.len() == expected);
    // The floating crate nearest the spawn point: (-12.5, -0.4, -0.5).
    let target = state
        .bodies
        .iter()
        .filter(|b| {
            b.visual == "crate"
                && b.position[0] < -10.0
                && b.position[2] < 0.0
                && b.position[2] > -1.0
        })
        .min_by(|a, b| a.distance.total_cmp(&b.distance))
        .expect("the floating crate")
        .clone();
    let me = state.scene_position.unwrap();
    // Line up with it: same height and z, then fly west into it.
    let yaw = (-(target.position[0] - me[0])).atan2(-(target.position[2] - me[2])) as f32;
    let now = player.app.controls().view.yaw;
    player.app.input(ViewerInput::Turn {
        dyaw: yaw - now,
        dpitch: 0.0,
    });
    player.key(Key::Forward, true);
    let pushed = player.until("pushed the crate", |s| {
        s.bodies
            .iter()
            .find(|b| b.guid == target.guid)
            .is_some_and(|b| b.speed > 0.3)
    });
    player.key(Key::Forward, false);
    let after = pushed
        .bodies
        .iter()
        .find(|b| b.guid == target.guid)
        .unwrap();
    println!("the crate drifts at {:.2} m/s", after.speed);
    let moved = player.until("saw it drift", |s| {
        s.bodies
            .iter()
            .find(|b| b.guid == target.guid)
            .is_some_and(|b| {
                let d: f64 = (0..3)
                    .map(|k| (b.position[k] - target.position[k]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                d > 0.5
            })
    });
    assert!(moved.bodies.iter().any(|b| b.guid == target.guid));
    drop(player);
    shard.stop();
}
