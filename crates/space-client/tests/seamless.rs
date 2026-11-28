//! Seamless crossing, as a player sees it: four in-process shards on the
//! cluster's clock (the wiring of the `space-server` binary: `clock::run`),
//! meeting at the scene's `x = -6` and `z = -4`, and the real client core
//! (`App`, no renderer) flown across their borders frame by frame.
//!
//! What every frame is checked for is what a player would notice
//! (`docs/plan-sharding.md` §5):
//!
//! * **the own ship as drawn** — predicted, interpolated — is where the frame
//!   before, its velocity and the time between them put it: no step, no
//!   rubber band; its velocity has no step either;
//! * **the clock everything is drawn by** runs on through a change of shard;
//! * **the prediction** is not corrected by more than a hair when the new
//!   shard's first word arrives;
//! * **no input is lost and none applied twice**: the shards' own records of
//!   the inputs they applied name every tick once across every switch, and a
//!   key pressed while the handoff is in flight moves the ship as soon as one
//!   pressed anywhere else;
//! * **everything else in sight** — bodies, other players — is drawn in every
//!   frame it was drawn in before and after, once, and does not jump;
//! * **one handoff per crossing**, none for standing on a border or walking
//!   along it, and none for a player who stands still while another flies
//!   past.
//!
//! None of these render (the frames are built, not drawn): no GPU needed.
//!
//! # How the bounds are taken
//!
//! These tests run real loops in real time, on whatever machine runs them —
//! a CI runner with two cores and other tests beside them included. So:
//!
//! * **Nothing is measured against the test's own clock.** A frame's time is
//!   the instant the app advanced it to ([`DrawnShip::at`]) and the tick it
//!   was drawn at; when this thread gets to look is not part of any number.
//! * **What a long frame explains is not held against it.** The ship
//!   accelerates at up to [`ACCEL`], so between two frames `t` apart its
//!   drawn position may differ from "the frame before plus velocity times
//!   `t`" by `ACCEL · t² / 2` and its velocity by `ACCEL · t`: a frame that
//!   took 100 ms (the thread was kept waiting) is allowed what 100 ms allow.
//! * **A switch is compared with ordinary flight.** Every bound is taken on
//!   the frames **around the changes of shard** (from [`BEFORE_SWITCH`]
//!   before one — the second of the handoff — to [`AFTER_SWITCH`] after it)
//!   and passes if it is under a **floor** — the size of defect the test is
//!   there to catch, given with each constant — or no worse than
//!   [`NOISE_FACTOR`] times what the 99.5th percentile of the frames **away
//!   from any switch** shows in the same flight: on a machine where ordinary
//!   flight is rough (inputs arriving late, frames delayed), a switch may be
//!   as rough, and no rougher. A maximum over a few samples is never the
//!   yardstick.
//!
//! Set `SPACE_SKIP_TIMING=1` to skip the whole file (the CI's second,
//! renderer-only pass over this package does; the tests have run by then).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_client::app::{App, AppConfig, DrawnShip};
use space_harness::{ClientState, Hub, Key, Screen, ViewerInput};
use space_protocol::{EntityGuid, Scene, SceneClock, SceneState, ShardId, ShardMap, ShardRegion};
use space_render::QualityPreset;
use space_server::clock::TickClock;
use space_server::net::ShardNet;
use space_server::{Shard, ShardConfig};
use space_world::NodeRef;
use tokio::sync::Mutex as AsyncMutex;
use tokio::task::JoinHandle;

/// Seconds per tick.
fn dt() -> f64 {
    f64::from(space_sim::dt())
}

/// The window around a change of shard that is held to the bounds: from the
/// offer (the shards give the client a second to connect) ...
const BEFORE_SWITCH: f64 = 1.5;
/// ... until the new shard's first words have been absorbed, seconds.
const AFTER_SWITCH: f64 = 0.7;

/// A switch may be this many times as rough as the 99.5th percentile of
/// ordinary flight (see the module docs).
const NOISE_FACTOR: f64 = 3.0;

/// The floor for how far the drawn ship may be from where the frame before,
/// the drawn velocity and the drawn time between the two frames put it,
/// metres, beyond what the frame's own length explains. 2.5 cm is 2 ms of
/// flight at [`CRUISE`]. **Catches:** a tick of flight lost or flown twice
/// (0.4 m at 12 m/s), a prediction rebuilt at the switch (two to three
/// ticks: 0.8 m — what every crossing showed before the cluster clock), a
/// correction of more than a few centimetres absorbed at once.
const JUMP_FLOOR: f64 = 0.025;

/// The floor for a change of the drawn velocity between two frames, m/s,
/// beyond `ACCEL · t`. **Catches:** a ship frozen or replayed for two ticks
/// or more under thrust (0.2 m/s a tick), a velocity taken from a stale
/// state.
const DV_FLOOR: f64 = 0.3;

/// The floor for how far the drawn tick may run ahead of or behind the
/// app's own frame time between two frames, ticks. The clock steers gently
/// (a few hundredths of a tick a frame). **Catches:** a clock restarted or
/// snapped at the switch by a tick or more — between shards that count
/// from their own starts it stepped by 4 to 12 ticks here.
const CLOCK_FLOOR: f64 = 0.35;

/// The floor for a correction of the prediction, metres. **Catches:** an
/// input the new shard never applied or applied twice while the pilot holds
/// thrust (7 mm after one tick, 3 cm after two and growing), a first word
/// of the new shard that is a tick off (0.4 m).
const CORRECTION_FLOOR: f64 = 0.02;

/// The floor for how far a body or another player may be drawn from where
/// its last frame's motion puts it, metres (they are drawn from a 30 Hz
/// history, between two frames that are not a tick apart). **Catches:** a
/// history started over at the switch (a moving body jumps by the two ticks
/// it is drawn behind: 0.4 m at 6 m/s), an entity drawn at another shard's
/// stale word.
const ENTITY_FLOOR: f64 = 0.12;

/// How long a handoff may take from the frame the drawn ship is
/// [`HYSTERESIS`] across a border until the client plays on the other
/// shard, in seconds of the draw clock: the second the shards give the
/// client to connect, and as much again for a machine on which connecting
/// is slow. **Catches:** a handoff that is refused and offered again (the
/// re-offer pause is two seconds: 3.1 s for the crossing).
const HANDOFF_BUDGET: f64 = 2.0;

/// `space_server::border::SHIP_HYSTERESIS`.
const HYSTERESIS: f64 = 1.0;

/// The speed the pilot flies at, m/s.
const CRUISE: f64 = 12.0;

/// The ship's acceleration under full thrust, m/s².
const ACCEL: f64 = 6.0;

/// Says how long a test took when it ends (`--nocapture`): these tests fly
/// in real time, and what they cost is worth seeing.
struct Timed(&'static str, Instant);

impl Timed {
    fn start(name: &'static str) -> Self {
        Self(name, Instant::now())
    }
}

impl Drop for Timed {
    fn drop(&mut self) {
        eprintln!("[{}] took {:.1} s", self.0, self.1.elapsed().as_secs_f64());
    }
}

/// Whether this file is to be skipped (see the module docs).
fn skipped() -> bool {
    let skip = std::env::var_os("SPACE_SKIP_TIMING").is_some_and(|v| v == "1");
    if skip {
        eprintln!(
            "SPACE_SKIP_TIMING=1: not run here (this pass is for the renderer; the crossing \
             tests run in `cargo run -p xtask -- ci`)"
        );
    }
    skip
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(6)
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
}

/// Four shards whose cells meet at the scene's `x = -6` / `z = -4`: shard 1
/// owns the root `63:0:0:0` — the spawn point's cell and the layer below —
/// and shards 2 (`x < -6`, `z > -4`), 3 (`x > -6`, `z < -4`) and 4 (both
/// smaller) one cell each. Each is started a good four ticks after the one
/// before it and runs on the cluster clock, like the binary's.
struct Cluster {
    shards: Vec<Running>,
}

impl Cluster {
    fn start(rt: &tokio::runtime::Runtime, scene: Option<Scene>) -> Self {
        let _inside = rt.enter();
        let regions = [
            (1, NodeRef::new(63, [0, 0, 0])),
            (2, NodeRef::new(64, [0, 1, 1])),
            (3, NodeRef::new(64, [1, 1, 0])),
            (4, NodeRef::new(64, [0, 1, 0])),
        ];
        let mut shards = Vec::new();
        for (id, region) in regions {
            let net =
                Arc::new(ShardNet::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind"));
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
            let ticker = tokio::spawn(space_server::clock::run(
                Arc::clone(&shard),
                Arc::clone(&net),
                TickClock::new(SceneClock::DEFAULT_ORIGIN_US)
                    .expect("the default origin is in the past"),
            ));
            shards.push(Running {
                net,
                shard,
                tasks: vec![serve, ticker],
            });
            std::thread::sleep(Duration::from_millis(137));
        }
        let map = ShardMap::new(
            1,
            regions
                .iter()
                .zip(&shards)
                .map(|((id, region), s)| ShardRegion::new(*region, ShardId::new(*id), s.addr()))
                .collect(),
        );
        rt.block_on(async {
            for s in &shards {
                let mut shard = s.shard.lock().await;
                shard.set_shard_map(map.clone());
                if let Some(scene) = scene {
                    shard.set_scene(SceneState::new(scene.id(), 1));
                }
            }
        });
        Self { shards }
    }

    fn entry(&self) -> SocketAddr {
        self.shards[0].addr()
    }

    /// Every shard's counters and the transfers it called off, and whether
    /// any of those was a player's ship.
    fn report(&self, rt: &tokio::runtime::Runtime) -> usize {
        rt.block_on(async {
            let mut player_aborts = 0;
            for s in &self.shards {
                let shard = s.shard.lock().await;
                eprintln!("    {:?}", shard.stats());
                for note in shard.recent_aborts() {
                    if note.player {
                        player_aborts += 1;
                        eprintln!("        {note:?}");
                    }
                }
            }
            player_aborts
        })
    }

    /// The shards' own records of the inputs they applied to `ship`: the
    /// shard that applied each tick's (`SimWorld::applied_inputs`).
    fn applied(&self, rt: &tokio::runtime::Runtime, ship: EntityGuid) -> Vec<(u64, u32)> {
        rt.block_on(async {
            let mut all = Vec::new();
            for s in &self.shards {
                let shard = s.shard.lock().await;
                let id = shard.shard().0;
                all.extend(
                    shard
                        .sim()
                        .applied_inputs()
                        .filter(|(_, guid)| *guid == ship)
                        .map(|(tick, _)| (tick, id)),
                );
            }
            all.sort_unstable();
            all
        })
    }

    fn stop(self) {
        for s in self.shards {
            for task in &s.tasks {
                task.abort();
            }
            s.net.close();
        }
    }
}

/// The exactly-once check, on the shards' own records of the real send
/// path (`link::inputs_for` gives every tick one input; the real
/// connections carry it; each shard records the ticks it applied one on):
///
/// * **no tick's input is applied twice** — by two shards, or twice by one
///   — anywhere in the flight;
/// * the ticks either side of **every change of the applying shard** are
///   consecutive, and within ten ticks of one no tick is missing — unless
///   ticks are missing just as often away from every switch (a machine so
///   loaded that inputs arrive after their tick: then a switch may lose as
///   many as ordinary flight does, and no more).
///
/// Mutation this catches: inputs sent to the old shard only (the ticks
/// after the commit are applied by nobody), the accepting shard dropping
/// what it is sent for a staged ship, an input-ack floor or an install tick
/// that makes both shards apply one tick. Returns the number of changes of
/// the applying shard.
fn inputs_applied_once(name: &str, applied: &[(u64, u32)]) -> usize {
    let record = Record::of(applied);
    eprintln!(
        "[{name}] inputs applied: {} ticks ({}..={}), {} changes of shard at {:?}; applied twice \
         {:?}; missing near a switch {:?}, elsewhere {}",
        applied.len(),
        applied.first().map_or(0, |a| a.0),
        applied.last().map_or(0, |a| a.0),
        record.switches.len(),
        record.switches,
        record.twice,
        record.lost_at_switch,
        record.lost_elsewhere
    );
    assert!(applied.len() > 30, "[{name}] the shards applied inputs");
    assert!(
        record.twice.is_empty(),
        "[{name}] an input was applied twice: {:?}",
        record.twice
    );
    assert!(
        record.lost_at_switch.len() <= record.lost_elsewhere,
        "[{name}] inputs were lost at a change of shard: ticks {:?} ({} lost elsewhere)",
        record.lost_at_switch,
        record.lost_elsewhere
    );
    record.switches.len()
}

/// What the shards' records of applied inputs say (see
/// [`inputs_applied_once`]).
#[derive(Debug, Default, PartialEq)]
struct Record {
    /// Ticks an input was applied on twice.
    twice: Vec<u64>,
    /// The last tick one shard applied and the first the next did.
    switches: Vec<(u64, u64)>,
    /// Ticks without an input within ten ticks of a switch.
    lost_at_switch: Vec<u64>,
    /// How many ticks without an input there are elsewhere.
    lost_elsewhere: usize,
}

impl Record {
    fn of(applied: &[(u64, u32)]) -> Self {
        let mut out = Self::default();
        let mut missing = Vec::new();
        for w in applied.windows(2) {
            let ((a, on_a), (b, on_b)) = (w[0], w[1]);
            if a == b {
                out.twice.push(a);
                continue;
            }
            if on_a != on_b {
                out.switches.push((a, b));
            }
            missing.extend(a + 1..b);
        }
        for tick in missing {
            let near = out
                .switches
                .iter()
                .any(|(a, b)| tick + 10 >= *a && tick <= *b + 10);
            if near {
                out.lost_at_switch.push(tick);
            } else {
                out.lost_elsewhere += 1;
            }
        }
        out
    }
}

/// One frame of a player, as drawn.
#[derive(Clone, Debug)]
struct Frame {
    /// The instant the app advanced the frame to (the test's own, while no
    /// ship is drawn).
    at: Instant,
    state: ClientState,
    ship: Option<DrawnShip>,
    others: Vec<(EntityGuid, [f64; 3])>,
    bodies: Vec<(EntityGuid, [f64; 3])>,
}

struct Player {
    app: App,
    log: Vec<Frame>,
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
            log: Vec::new(),
        };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            let f = player.frame();
            if f.state.screen == Screen::Online && f.ship.is_some() {
                break;
            }
            assert!(Instant::now() < deadline, "never online: {:?}", f.state);
            std::thread::sleep(Duration::from_millis(5));
        }
        // The pilot flies by hand: no brake assist.
        player.app.input(ViewerInput::Text('b'));
        player.log.clear();
        player
    }

    /// One frame (built, not rendered), logged.
    fn frame(&mut self) -> Frame {
        self.app.update();
        let _ = self.app.scene(640, 360);
        self.app.finish_frame(640, 360, None);
        let ship = self.app.drawn_ship();
        let f = Frame {
            at: ship.map_or_else(Instant::now, |s| s.at),
            state: self.app.client_state(640, 360),
            ship,
            others: self.app.drawn_others(),
            bodies: self.app.drawn_bodies(),
        };
        self.log.push(f.clone());
        f
    }

    fn key(&mut self, key: Key, down: bool) {
        self.app.input(ViewerInput::Key { key, down });
    }

    /// Look along the direction (x, z) (yaw 0 looks along -z).
    fn face(&mut self, x: f64, z: f64) {
        let want = (-x as f32).atan2(-z as f32);
        let yaw = self.app.controls().view.yaw;
        self.app.input(ViewerInput::Turn {
            dyaw: want - yaw,
            dpitch: 0.0,
        });
    }

    fn ship(&self) -> DrawnShip {
        self.log.last().and_then(|f| f.ship).expect("a drawn ship")
    }

    fn guid(&self) -> EntityGuid {
        self.app.replica().and_then(|r| r.ship).expect("a ship")
    }

    fn handoffs(&self) -> u64 {
        self.log.last().map_or(0, |f| f.state.handoffs)
    }

    /// One frame's steering towards (x, z) at up to `cruise` m/s, stopping
    /// there: thrust along the velocity error. Returns the distance left.
    fn steer(&mut self, x: f64, z: f64, cruise: f64) -> f64 {
        let s = self.ship();
        let (dx, dz) = (x - s.position[0], z - s.position[2]);
        let dist = (dx * dx + dz * dz).sqrt();
        let want = cruise.min((2.0 * 0.6 * ACCEL * dist).sqrt());
        let (wx, wz) = if dist > 1e-6 {
            (dx / dist * want, dz / dist * want)
        } else {
            (0.0, 0.0)
        };
        let (ex, ez) = (wx - f64::from(s.velocity[0]), wz - f64::from(s.velocity[2]));
        if (ex * ex + ez * ez).sqrt() > 0.4 {
            self.face(ex, ez);
            self.key(Key::Forward, true);
        } else {
            self.key(Key::Forward, false);
        }
        dist
    }

    /// Fly to (x, z) at up to `cruise` m/s, until within `near` metres;
    /// `others` draw their frames alongside.
    fn fly_to(&mut self, others: &mut [&mut Player], x: f64, z: f64, cruise: f64, near: f64) {
        let deadline = Instant::now() + Duration::from_secs(90);
        loop {
            self.frame();
            for o in others.iter_mut() {
                o.frame();
            }
            if self.steer(x, z, cruise) < near {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "never got to {x} {z}: {:?}",
                self.ship()
            );
            std::thread::sleep(Duration::from_millis(7));
        }
        self.key(Key::Forward, false);
    }

    fn idle(&mut self, others: &mut [&mut Player], seconds: f64) {
        let end = Instant::now() + Duration::from_secs_f64(seconds);
        while Instant::now() < end {
            self.frame();
            for o in others.iter_mut() {
                o.frame();
            }
            std::thread::sleep(Duration::from_millis(7));
        }
    }

    /// Frames until `done` says so, for at most `seconds`; whether it did.
    fn until(
        &mut self,
        others: &mut [&mut Player],
        seconds: f64,
        mut done: impl FnMut(&Player) -> bool,
    ) -> bool {
        let end = Instant::now() + Duration::from_secs_f64(seconds);
        loop {
            self.frame();
            for o in others.iter_mut() {
                o.frame();
            }
            if done(self) {
                return true;
            }
            if Instant::now() >= end {
                return false;
            }
            std::thread::sleep(Duration::from_millis(7));
        }
    }

    /// Brake to a stop (the brake assist, switched on for it): until the
    /// drawn ship all but stands (the assist eases off below 1.5 m/s, so the
    /// last of it takes a second), three seconds at most.
    fn stop(&mut self, others: &mut [&mut Player]) {
        self.slow(others, 0.005);
    }

    /// Brake (the brake assist, switched on for it) until the drawn ship is
    /// slower than `below` m/s, three seconds at most.
    fn slow(&mut self, others: &mut [&mut Player], below: f64) {
        self.key(Key::Forward, false);
        self.app.input(ViewerInput::Text('b'));
        self.until(others, 3.0, |p| {
            norm(p.ship().velocity.map(f64::from)) < below
        });
        self.idle(others, 0.1);
        self.app.input(ViewerInput::Text('b'));
    }
}

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// What two consecutive frames show (see the module docs, "How the bounds
/// are taken").
#[derive(Clone, Copy, Debug, Default)]
struct Step {
    /// How far the drawn ship is from where the frame before, the drawn
    /// velocity and the drawn time between them put it, metres — beyond
    /// what the frame's length explains (`ACCEL · t² / 2`).
    jump: f64,
    /// The change of the drawn velocity, m/s, beyond `ACCEL · t`.
    dv: f64,
    /// Drawn ticks between the two frames, minus the app's own frame time
    /// in ticks.
    clock: f64,
    /// The drawn speed, m/s.
    speed: f64,
    /// The time between the two frames by the draw clock, seconds.
    span: f64,
}

fn step(a: &Frame, b: &Frame) -> Option<Step> {
    let (sa, sb) = (a.ship?, b.ship?);
    let span = (sb.tick - sa.tick) * dt();
    let frame = sb.at.saturating_duration_since(sa.at).as_secs_f64();
    let mid = [0, 1, 2].map(|k| 0.5 * f64::from(sa.velocity[k] + sb.velocity[k]));
    let moved = [0, 1, 2].map(|k| sb.position[k] - sa.position[k]);
    let off = norm([0, 1, 2].map(|k| moved[k] - mid[k] * span));
    let dv = norm([0, 1, 2].map(|k| f64::from(sb.velocity[k] - sa.velocity[k])));
    let t = span.abs().max(frame);
    Some(Step {
        jump: (off - 0.5 * ACCEL * t * t).max(0.0),
        dv: (dv - ACCEL * t).max(0.0),
        clock: ((sb.tick - sa.tick) - frame / dt()).abs(),
        speed: norm(mid),
        span,
    })
}

/// One measure of a flight: its worst around the changes of shard, and the
/// 99.5th percentile of the frames away from all of them.
#[derive(Clone, Copy, Debug, Default)]
struct Split {
    near: f64,
    away: f64,
    away_frames: usize,
}

impl Split {
    fn of(values: &[(bool, f64)]) -> Self {
        let near = values
            .iter()
            .filter(|(near, _)| *near)
            .map(|(_, v)| *v)
            .fold(0.0, f64::max);
        let mut away: Vec<f64> = values
            .iter()
            .filter(|(near, _)| !*near)
            .map(|(_, v)| *v)
            .collect();
        away.sort_by(f64::total_cmp);
        Self {
            near,
            // (With too few frames a percentile says nothing: the floor
            // alone decides.)
            away: if away.len() >= 100 {
                away[away.len() * 995 / 1000]
            } else {
                0.0
            },
            away_frames: away.len(),
        }
    }

    /// The bound: under the floor, or no rougher than ordinary flight.
    fn holds(&self, floor: f64) -> bool {
        self.near <= floor.max(NOISE_FACTOR * self.away)
    }
}

/// What a flight's frames show, around the changes of shard and away from
/// them.
#[derive(Debug, Default)]
struct Flight {
    handoffs: usize,
    jump: Split,
    dv: Split,
    clock: Split,
    correction: Split,
    /// How far a body or another player is drawn from where its last
    /// frame's motion puts it, metres (the worst of a frame).
    entity: Split,
    /// Entities drawn before and after a switch but missing in a frame
    /// between, and entities drawn twice in a frame.
    gaps: usize,
    doubles: usize,
    /// The fewest entities that were in sight across a switch.
    carried: usize,
}

impl Flight {
    /// The bounds every crossing flight is held to (see the constants).
    fn assert_seamless(&self, name: &str) {
        assert!(
            self.jump.holds(JUMP_FLOOR),
            "[{name}] the drawn ship jumped at a change of shard: {self:?}"
        );
        assert!(
            self.dv.holds(DV_FLOOR),
            "[{name}] the drawn velocity stepped at a change of shard: {self:?}"
        );
        assert!(
            self.clock.holds(CLOCK_FLOOR),
            "[{name}] the draw clock stepped at a change of shard: {self:?}"
        );
        assert!(
            self.correction.holds(CORRECTION_FLOOR),
            "[{name}] the prediction was corrected at a change of shard: {self:?}"
        );
    }
}

/// Go through a log: every frame's step, split into the frames around a
/// change of shard and the others, and around every change what was in
/// sight. Prints one line per switch (`--nocapture`).
fn examine(name: &str, log: &[Frame]) -> Flight {
    let mut out = Flight {
        carried: usize::MAX,
        ..Flight::default()
    };
    let switches: Vec<usize> = (1..log.len())
        .filter(|&i| log[i].state.handoffs != log[i - 1].state.handoffs)
        .collect();
    out.handoffs = switches.len();
    let near = |i: usize| {
        switches.iter().any(|&s| {
            let (at, sw) = (log[i].at, log[s].at);
            if at <= sw {
                sw.duration_since(at).as_secs_f64() <= BEFORE_SWITCH
            } else {
                at.duration_since(sw).as_secs_f64() <= AFTER_SWITCH
            }
        })
    };
    // The own ship.
    let steps: Vec<(usize, Step)> = (1..log.len())
        .filter_map(|i| step(&log[i - 1], &log[i]).map(|s| (i, s)))
        .collect();
    let split = |f: &dyn Fn(&Step) -> f64| {
        Split::of(
            &steps
                .iter()
                .map(|(i, s)| (near(*i), f(s)))
                .collect::<Vec<_>>(),
        )
    };
    out.jump = split(&|s| s.jump);
    out.dv = split(&|s| s.dv);
    out.clock = split(&|s| s.clock);
    out.correction = Split::of(
        &(0..log.len())
            .map(|i| (near(i), log[i].state.correction_m))
            .collect::<Vec<_>>(),
    );
    // Everything else in sight: against the velocity it was drawn moving at
    // the frame before, over the app's own frame time.
    let mut entity = Vec::with_capacity(log.len());
    let mut last: HashMap<EntityGuid, ([f64; 3], [f64; 3], Instant)> = HashMap::new();
    for (i, frame) in log.iter().enumerate() {
        let mut worst = 0.0f64;
        for (g, p) in frame.bodies.iter().chain(&frame.others) {
            let seen = last.get(g).copied();
            let span = seen.map_or(0.0, |(_, _, at)| {
                frame.at.saturating_duration_since(at).as_secs_f64()
            });
            let v = match seen {
                Some((q, _, _)) if span > 0.0 => [0, 1, 2].map(|k| (p[k] - q[k]) / span),
                _ => [0.0; 3],
            };
            if let Some((q, w, _)) = seen {
                if span < 0.25 && (norm(w) > 0.0 || norm(v) > 0.0) {
                    let off = norm([0, 1, 2].map(|k| p[k] - q[k] - w[k] * span));
                    worst = worst.max(off);
                }
            }
            last.insert(*g, (*p, v, frame.at));
        }
        entity.push((near(i), worst));
    }
    out.entity = Split::of(&entity);
    for &i in &switches {
        let (lo, hi) = (i.saturating_sub(60), (i + 60).min(log.len() - 1));
        let all = |f: &Frame| -> HashMap<EntityGuid, [f64; 3]> {
            f.bodies.iter().chain(&f.others).copied().collect()
        };
        let (before, after) = (all(&log[lo]), all(&log[hi]));
        let through: Vec<EntityGuid> = before
            .keys()
            .filter(|g| after.contains_key(g))
            .copied()
            .collect();
        out.carried = out.carried.min(through.len());
        let (mut gaps, mut doubles) = (0usize, 0usize);
        for frame in &log[lo..=hi] {
            let mut seen: HashMap<EntityGuid, usize> = HashMap::new();
            for (g, _) in frame.bodies.iter().chain(&frame.others) {
                *seen.entry(*g).or_default() += 1;
            }
            doubles += seen.values().filter(|n| **n > 1).count();
            gaps += through.iter().filter(|g| !seen.contains_key(g)).count();
        }
        out.gaps += gaps;
        out.doubles += doubles;
        let around: Vec<&Step> = steps
            .iter()
            .filter(|(k, _)| (lo..=hi).contains(k))
            .map(|(_, s)| s)
            .collect();
        let worst = |f: &dyn Fn(&Step) -> f64| around.iter().map(|s| f(s)).fold(0.0f64, f64::max);
        let f = &log[i];
        eprintln!(
            "[{name}]   handoff {} -> shard {:?} at {:.1?}, {:.1} m/s: jump {:.4} m, dv {:.3} m/s, \
             clock {:.3} ticks, longest frame {:.0} ms, correction {:.4} m; in sight throughout {} \
             (before {}, after {}), gaps {gaps}, doubles {doubles}",
            f.state.handoffs,
            f.state.shard,
            f.ship.map(|s| s.position).unwrap_or([0.0; 3]),
            worst(&|s| s.speed),
            worst(&|s| s.jump),
            worst(&|s| s.dv),
            worst(&|s| s.clock),
            1e3 * worst(&|s| s.span),
            log[lo..=hi]
                .iter()
                .map(|f| f.state.correction_m)
                .fold(0.0, f64::max),
            through.len(),
            before.len(),
            after.len(),
        );
    }
    eprintln!(
        "[{name}] {} frames, {} handoffs; at the switches / 99.5 % of the {} frames away from \
         them: jump {:.4} / {:.4} m, dv {:.3} / {:.3} m/s, clock {:.3} / {:.3} ticks, correction \
         {:.4} / {:.4} m, entities {:.3} / {:.3} m; gaps {}, doubles {}",
        log.len(),
        out.handoffs,
        out.jump.away_frames,
        out.jump.near,
        out.jump.away,
        out.dv.near,
        out.dv.away,
        out.clock.near,
        out.clock.away,
        out.correction.near,
        out.correction.away,
        out.entity.near,
        out.entity.away,
        out.gaps,
        out.doubles,
    );
    out
}

/// What `log` shows of the other entity `guid`: how far it is drawn from
/// where its motion of the frame before puts it, metres — around the
/// instants in `switched` and away from them.
fn watch(log: &[Frame], guid: EntityGuid, switched: &[Instant]) -> Split {
    let near = |at: Instant| {
        switched.iter().any(|sw| {
            if at <= *sw {
                sw.duration_since(at).as_secs_f64() <= BEFORE_SWITCH
            } else {
                at.duration_since(*sw).as_secs_f64() <= AFTER_SWITCH
            }
        })
    };
    let mut values = Vec::new();
    let mut last: Option<([f64; 3], [f64; 3], Instant)> = None;
    for f in log {
        let Some((_, p)) = f.others.iter().find(|(g, _)| *g == guid) else {
            continue;
        };
        if let Some((q, w, at)) = last {
            let span = f.at.saturating_duration_since(at).as_secs_f64();
            if span > 0.0 {
                if span < 0.25 {
                    let off = norm([0, 1, 2].map(|k| p[k] - q[k] - w[k] * span));
                    values.push((near(f.at), off));
                }
                last = Some((*p, [0, 1, 2].map(|k| (p[k] - q[k]) / span), f.at));
            }
        } else {
            last = Some((*p, [0.0; 3], f.at));
        }
    }
    Split::of(&values)
}

/// The longest a handoff took, in seconds of the draw clock, from the frame
/// the drawn ship was a hysteresis inside another cell than before (clear of
/// both borders) to the frame the client was on another shard.
fn slowest_handoff(log: &[Frame]) -> f64 {
    let cell = |s: &DrawnShip| {
        (
            s.position[0] < -6.0 - HYSTERESIS,
            s.position[0] > -6.0 + HYSTERESIS,
            s.position[2] < -4.0 - HYSTERESIS,
            s.position[2] > -4.0 + HYSTERESIS,
        )
    };
    let mut slowest = 0.0f64;
    let mut crossed: Option<f64> = None;
    let mut inside = log.first().and_then(|f| f.ship).map(|s| cell(&s));
    for w in log.windows(2) {
        let Some(s) = w[1].ship else { continue };
        let now = cell(&s);
        if (now.0 || now.1) && (now.2 || now.3) {
            if inside.is_some_and(|was| was != now) && crossed.is_none() {
                crossed = Some(s.tick);
            }
            inside = Some(now);
        }
        if w[1].state.handoffs != w[0].state.handoffs {
            if let Some(at) = crossed.take() {
                slowest = slowest.max((s.tick - at) * dt());
            }
        }
    }
    slowest
}

/// A loop through all four cells and over both borders, both ways, at flying
/// speed: nothing a frame shows gives a change of shard away, and the shards
/// applied every tick's input exactly once.
///
/// Mutations this catches (each was applied, and the test failed): shards
/// counting from their own start (the clock steps by their difference, 4 to
/// 12 ticks here, and the ship with it); a client that starts over at every
/// switch — clock, prediction, sent inputs (a jump of the two or three ticks
/// the new prediction lacks: about 0.8 m at this speed); inputs sent to the
/// old shard only (ticks nobody applied, at every switch).
#[test]
fn a_loop_through_all_four_cells_is_seamless() {
    if skipped() {
        return;
    }
    let _timed = Timed::start("loop");
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut p = Player::online(&rt, cluster.entry());
    let mut crossings = 0;
    for (x, z) in [(-21.0, 10.0), (-21.0, -19.0), (9.0, -19.0), (9.0, 10.0)] {
        let h = p.handoffs();
        let from = p.frame().state.shard;
        p.fly_to(&mut [], x, z, CRUISE, 3.0);
        // One border lies on every leg: one handoff, to the cell's shard.
        p.until(&mut [], 4.0, |p| p.handoffs() != h);
        assert_eq!(p.handoffs(), h + 1, "one handoff on the leg to ({x}, {z})");
        assert_ne!(p.frame().state.shard, from);
        crossings += 1;
    }
    p.idle(&mut [], 0.8);
    let slowest = slowest_handoff(&p.log);
    let flight = examine("loop", &p.log);
    eprintln!("[loop] slowest handoff from a hysteresis past the border: {slowest:.2} s");
    assert_eq!(cluster.report(&rt), 0, "a player's handoff was called off");
    assert_eq!(flight.handoffs, crossings, "one handoff per crossing");
    assert_eq!(crossings, 4, "both borders, both ways");
    flight.assert_seamless("loop");
    assert!(slowest <= HANDOFF_BUDGET, "a handoff took {slowest:.2} s");
    let changes = inputs_applied_once("loop", &cluster.applied(&rt, p.guid()));
    assert_eq!(
        changes, crossings,
        "the applying shard changed once a crossing"
    );
    let replica = p.app.replica().expect("online");
    assert_eq!(replica.tick_epoch, 0, "the ticks never started over");
    drop(p);
    cluster.stop();
}

/// A key pressed while a handoff is in flight — before the commit, just
/// before it, just after it, after the switch — moves the ship as soon as a
/// key pressed far from any border does, and **every tick's input is applied
/// exactly once** by the shards (their own records, [`inputs_applied_once`]).
///
/// The ship coasts over `x = -6`; at a set time after it is a hysteresis
/// across (the offer goes out there, the commit follows a second later) the
/// pilot pushes up or down, and the ticks until the **shard's** word shows
/// the ship rising are counted: from the draw clock's tick at the press to
/// the tick of the shard's state that shows it. The same far from the border
/// is the usual latency: the input's lead and the frame's way back. The
/// latency is a coarse measure (a few ticks either way on a busy machine) and
/// is held loosely; what is exact is the record.
///
/// Mutation this catches: inputs sent to the old shard only (the ticks after
/// the commit are applied by nobody: the record has a hole at every switch),
/// an input-ack floor in the wrong clock (the new shard's word is taken for
/// covering inputs it never applied: the prediction drops them and the ship
/// stalls), inputs for a staged ship dropped by the accepting shard, a tick
/// both shards apply.
#[test]
fn an_input_during_the_handoff_is_not_lost() {
    if skipped() {
        return;
    }
    let _timed = Timed::start("input");
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut p = Player::online(&rt, cluster.entry());
    // Until the shard's word says the ship climbs (or sinks): ticks from the
    // key press, for the drawn ship and for the shard's word.
    let respond = |p: &mut Player, up: bool| -> (f64, f64) {
        let key = if up { Key::Up } else { Key::Down };
        let vy0 = p.frame().state.velocity.map_or(0.0, |v| v[1]);
        let (drawn0, pressed) = (f64::from(p.ship().velocity[1]), p.ship().tick);
        let started = Instant::now();
        p.key(key, true);
        let (mut drawn, mut shard) = (None, None);
        while shard.is_none() {
            let f = p.frame();
            let sign = if up { 1.0 } else { -1.0 };
            if drawn.is_none() && sign * (f64::from(p.ship().velocity[1]) - drawn0) > 0.05 {
                drawn = Some(p.ship().tick - pressed);
            }
            if sign * (f.state.velocity.map_or(0.0, |v| v[1]) - vy0) > 0.05 {
                shard = Some(f.state.tick as f64 - pressed);
            }
            assert!(
                started.elapsed() < Duration::from_secs(3),
                "no answer at all"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        // Held for a moment, then let go.
        p.idle(&mut [], 0.15);
        p.key(key, false);
        (drawn.unwrap_or(f64::NAN), shard.unwrap_or(f64::NAN))
    };
    // The usual latency, inside shard 1 and at rest.
    p.fly_to(&mut [], 6.0, 8.0, CRUISE, 1.0);
    p.stop(&mut []);
    let mut usual = Vec::new();
    for i in 0..4 {
        usual.push(respond(&mut p, i % 2 == 0));
        p.idle(&mut [], 0.25);
    }
    let median = |values: &mut Vec<f64>| {
        values.sort_by(f64::total_cmp);
        values[values.len() / 2]
    };
    let usual_shard = median(&mut usual.iter().map(|u| u.1).collect());
    let usual_drawn = median(&mut usual.iter().map(|u| u.0).collect());
    eprintln!("[input] the usual latency in ticks (drawn, shard's word): {usual:.1?}");
    p.slow(&mut [], 0.3);
    // Over the border, to and fro, pressing at a set time after the offer
    // went out.
    let mut during = Vec::new();
    let mut west = true;
    for (i, after) in [0.5, 0.95, 1.03, 1.25].into_iter().enumerate() {
        let h = p.handoffs();
        let past = |x: f64| {
            if west {
                x < -6.0 - HYSTERESIS
            } else {
                x > -6.0 + HYSTERESIS
            }
        };
        // Up to speed towards the border, then coast across it.
        let heading = if west { -1.0 } else { 1.0 };
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            p.frame();
            let s = p.ship();
            if past(s.position[0]) {
                break;
            }
            if heading * f64::from(s.velocity[0]) < 7.0 {
                p.face(heading, 0.0);
                p.key(Key::Forward, true);
            } else {
                p.key(Key::Forward, false);
            }
            assert!(Instant::now() < deadline, "never crossed: {s:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
        p.key(Key::Forward, false);
        p.idle(&mut [], after);
        let before = p.handoffs();
        let answer = respond(&mut p, i % 2 == 0);
        let switched_during = p.handoffs() != before;
        p.until(&mut [], 3.0, |p| p.handoffs() != h);
        p.slow(&mut [], 0.3);
        assert_eq!(p.handoffs(), h + 1, "one handoff per crossing");
        eprintln!(
            "[input] pressed {after:.2} s after the offer ({}): drawn {:.1} ticks, shard's word {:.1} ticks",
            if before != h {
                "after the switch"
            } else if switched_during {
                "the switch came before the answer"
            } else {
                "before the switch"
            },
            answer.0,
            answer.1
        );
        during.push(answer);
        west = !west;
    }
    let worst_shard = during.iter().map(|u| u.1).fold(0.0, f64::max);
    let worst_drawn = during.iter().map(|u| u.0).fold(0.0, f64::max);
    // Four ticks of slack on the usual (median) latency: this catches an
    // input that waits for the switch or for a re-offer (ten ticks and
    // more), not one that is a tick late — the record below does that.
    assert!(
        worst_shard <= usual_shard + 4.0,
        "an input during the handoff was answered late: {worst_shard:.1} ticks, usually {usual_shard:.1}"
    );
    assert!(
        worst_drawn <= usual_drawn + 4.0,
        "the drawn ship answered late: {worst_drawn:.1} ticks, usually {usual_drawn:.1}"
    );
    let flight = examine("input", &p.log);
    assert!(flight.correction.holds(CORRECTION_FLOOR), "{flight:?}");
    assert_eq!(cluster.report(&rt), 0);
    let changes = inputs_applied_once("input", &cluster.applied(&rt, p.guid()));
    assert_eq!(
        changes,
        during.len(),
        "one change of the applying shard a crossing"
    );
    drop(p);
    cluster.stop();
}

/// Standing on a border, and walking along it: no handoff. Stopping just
/// across it: one, three seconds later, and no more. (Two players at once:
/// one stands on `x = -6` and then walks along it, the other stops half a
/// metre across it further along and stays.)
///
/// Mutation this catches: no hysteresis for ships (walking along `x = -6`
/// and straying 0.3 m to either side was 13 handoffs in 14 legs), a
/// hysteresis without the settle rule (a player who stops half a metre
/// across stays on the wrong shard for good), a settle rule that hands a
/// ship back and forth.
#[test]
fn standing_on_a_border_and_walking_along_it_is_no_ping_pong() {
    if skipped() {
        return;
    }
    let _timed = Timed::start("border");
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut p = Player::online(&rt, cluster.entry());
    p.fly_to(&mut [], -6.0, 9.0, 4.0, 0.1);
    p.stop(&mut []);
    assert_eq!(
        p.handoffs(),
        0,
        "it never was a metre across: {:?}",
        p.ship()
    );
    assert!((p.ship().position[0] + 6.0).abs() < 0.5, "{:?}", p.ship());
    // The second player: half a metre into shard 2's ground, 16 m further
    // along the border, and stays there. Meanwhile the first stands.
    let mut q = Player::online(&rt, cluster.entry());
    let stood = Instant::now();
    q.fly_to(&mut [&mut p], -6.5, 24.0, 6.0, 0.1);
    q.stop(&mut [&mut p]);
    assert!(
        (q.ship().position[0] + 6.5).abs() < 0.4,
        "half a metre across: {:?}",
        q.ship()
    );
    let settled = Instant::now();
    // Standing on the border: if it stopped a hair across, it is taken over
    // once it has settled there (three seconds) — once — and from then on
    // it stays.
    let rest = (5.5 - stood.elapsed().as_secs_f64()).max(0.1);
    p.idle(&mut [&mut q], rest);
    let h0 = p.handoffs();
    let across = p.ship().position[0] < -6.0;
    assert_eq!(h0, u64::from(across), "at {:.3?}", p.ship().position);
    eprintln!(
        "[border] stood at {:.3?} for {:.1} s: {h0} handoff(s)",
        p.ship().position,
        stood.elapsed().as_secs_f64()
    );
    // Along it for 7 s, weaving 0.3 m to either side.
    let end = Instant::now() + Duration::from_secs(7);
    let (mut legs, mut furthest) = (0, 0.0f64);
    while Instant::now() < end {
        let x = if legs % 2 == 0 { -6.3 } else { -5.7 };
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            p.frame();
            q.frame();
            furthest = furthest.max((p.ship().position[0] + 6.0).abs());
            if p.steer(x, 10.5 + f64::from(legs) * 1.5, 2.0) < 0.3 {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(7));
        }
        legs += 1;
    }
    p.stop(&mut [&mut q]);
    eprintln!(
        "[border] walked {legs} legs along it, at most {furthest:.2} m from it, to {:.2?}: {} handoffs",
        p.ship().position,
        p.handoffs() - h0
    );
    assert!(legs >= 3 && furthest < 0.9, "{legs} legs, {furthest} m");
    // Where it came to rest at the end may be a hair across: wait the settle
    // rule out, so that whatever it does to the walker is counted.
    p.idle(&mut [&mut q], 3.3);
    // Never a metre across, and never three seconds in a row on the other
    // side: nothing to hand over. (One is allowed for where it came to rest
    // at the end.)
    assert!(
        p.handoffs() - h0 <= 1,
        "walking along the border: {} handoffs",
        p.handoffs() - h0
    );
    // The one who stopped half a metre across: taken over once it had
    // settled there, and — more than three seconds later — not handed back.
    let stayed = settled.elapsed().as_secs_f64();
    assert!(stayed > 6.5, "it stayed {stayed:.1} s");
    assert_eq!(q.handoffs(), 1, "settled across the border: once");
    assert_eq!(q.frame().state.shard, Some(2));
    assert!((q.ship().position[0] + 6.5).abs() < 0.4, "it stood still");
    for (name, log) in [("border", &p.log), ("settled", &q.log)] {
        let flight = examine(name, log);
        assert!(
            flight.jump.holds(JUMP_FLOOR) && flight.correction.holds(CORRECTION_FLOOR),
            "{flight:?}"
        );
    }
    assert_eq!(cluster.report(&rt), 0);
    drop(p);
    drop(q);
    cluster.stop();
}

/// Two players stand — a metre and a half beside the path of a third, and
/// nine metres beside it — while the third flies past and across the border
/// and back, again and again: the ones who stand are never handed anywhere,
/// keep the sessions they had, and see the other in every frame, once,
/// moving without a jump.
///
/// The one who flies joined first: its ship has the lower guid and is the
/// key of the island the ships form whenever its second of motion reaches
/// another.
///
/// Mutation this catches: a player's ship moved with its island (the one who
/// stands went to the neighbour shard and back at every pass), a ghost of a
/// player taken for something an island's bodies need not follow, the other
/// player's history started over when it changed from own entity to ghost.
#[test]
fn a_standing_player_is_not_handed_over_because_another_flies_past() {
    if skipped() {
        return;
    }
    let _timed = Timed::start("stander");
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut mover = Player::online(&rt, cluster.entry());
    mover.fly_to(&mut [], 8.0, 0.0, 8.0, 0.5);
    mover.slow(&mut [], 0.05);
    let mover_guid = mover.guid();
    // Beside the mover's path (z = 0), near the border, on shard 1's ground.
    let besides = [1.5, 9.0];
    let mut near = Player::online(&rt, cluster.entry());
    near.fly_to(&mut [&mut mover], -3.0, besides[0], 5.0, 0.3);
    near.stop(&mut [&mut mover]);
    let mut far = Player::online(&rt, cluster.entry());
    far.fly_to(&mut [&mut mover, &mut near], -3.0, besides[1], 6.0, 0.3);
    far.stop(&mut [&mut mover, &mut near]);
    let began = [near.frame(), far.frame()];
    let stood = [near.ship().position, far.ship().position];
    let m0 = mover.handoffs();
    near.log.clear();
    far.log.clear();
    let passes = 2;
    for _ in 0..passes {
        for x in [-22.0, 9.0] {
            let h = mover.handoffs();
            mover.fly_to(&mut [&mut near, &mut far], x, 0.0, CRUISE, 3.0);
            mover.until(&mut [&mut near, &mut far], 4.0, |m| m.handoffs() != h);
        }
    }
    mover.slow(&mut [&mut near, &mut far], 0.3);
    assert_eq!(mover.handoffs() - m0, 2 * passes, "the mover's crossings");
    // When the mover changed shard.
    let switched: Vec<Instant> = mover
        .log
        .windows(2)
        .filter(|w| w[1].state.handoffs != w[0].state.handoffs)
        .map(|w| w[1].at)
        .collect();
    for (k, stander) in [&mut near, &mut far].into_iter().enumerate() {
        let (beside, at) = (besides[k], &began[k]);
        let s = stander.frame();
        eprintln!(
            "[stander {beside} m] stood at {:.2?}: {} handoffs, session {:?} -> {:?}; the other \
             crossed {} times",
            stood[k],
            s.state.handoffs - at.state.handoffs,
            at.state.session,
            s.state.session,
            mover.handoffs() - m0,
        );
        assert_eq!(
            s.state.handoffs, at.state.handoffs,
            "the one who stands was handed over"
        );
        assert_eq!(s.state.session, at.state.session, "it keeps its session");
        assert_eq!(s.state.shard, Some(1));
        let moved = norm([0, 1, 2].map(|i| stander.ship().position[i] - stood[k][i]));
        // (It was all but at rest when the passes began: centimetres of
        // drift in half a minute. Pushed or handed about it would be metres
        // off.)
        assert!(moved < 0.2, "it did not move: {moved} m");
        // What it saw of the other: in every frame, once ...
        for f in &stander.log {
            let seen = f.others.iter().filter(|(g, _)| *g == mover_guid).count();
            assert_eq!(seen, 1, "the other player, once in every frame");
        }
        // ... and without a jump: around the moments the **other** changed
        // shard (for this client it turns from an entity of its own shard
        // into a neighbour's ghost, and back) it is no further from where
        // its motion puts it than half a tick of flight — it is drawn from
        // a 30 Hz history, between two frames that are not a tick apart —
        // or than ordinary watching shows.
        let watched = watch(&stander.log, mover_guid, &switched);
        eprintln!(
            "[stander {beside} m] the other's step beyond its motion: {:.3} m around its {} \
             changes of shard, 99.5 % of the other {} frames {:.3} m",
            watched.near,
            switched.len(),
            watched.away_frames,
            watched.away
        );
        assert!(
            watched.holds(0.3),
            "the other player jumped when it changed shard: {watched:?}"
        );
    }
    assert_eq!(cluster.report(&rt), 0);
    drop(near);
    drop(far);
    drop(mover);
    cluster.stop();
}

/// From the spawn point over `x = -6` through the demo area's zero-g corner
/// and back, twice: every crossing is one handoff within the budget and
/// no player's transfer is called off — and every body in sight before and
/// after a change of shard is drawn in every frame between, once, where its
/// motion puts it.
///
/// This is the crossing that took three seconds: the ship went in a group
/// with the bodies its second of motion reached, the corner's shard was at
/// that moment sending one of them the other way, refused the whole group as
/// a "crossing island transfer", and the ship waited out the two seconds of
/// the re-offer pause.
///
/// Mutation this catches: a ship offered in a group with bodies, the
/// crossing rule applied to a player's ship, bodies' histories started over
/// at the switch (every moving body would jump by the two ticks it is drawn
/// behind).
#[test]
fn crossing_beside_the_showcase_takes_one_prompt_handoff_and_nothing_pops() {
    if skipped() {
        return;
    }
    let _timed = Timed::start("demo");
    let rt = runtime();
    let cluster = Cluster::start(&rt, Some(Scene::DemoArea));
    let mut p = Player::online(&rt, cluster.entry());
    let expected = space_scene::body_defs(Scene::DemoArea).len();
    let deadline = Instant::now() + Duration::from_secs(30);
    while p.frame().bodies.len() != expected {
        assert!(
            Instant::now() < deadline,
            "the showcase never arrived whole"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    // Up a little: over the corner's floating crates (a ship that runs into
    // one is pushed about, which is physics and not this test's business),
    // and well within the reach of its second of motion.
    // (By height, not by time: how long a second of holding a key is
    // depends on how busy the machine is.)
    p.key(Key::Up, true);
    p.until(&mut [], 5.0, |p| p.ship().position[1] > 3.0);
    p.key(Key::Up, false);
    p.slow(&mut [], 0.05);
    let height = p.ship().position[1];
    assert!((3.0..10.0).contains(&height), "at height {height}");
    p.log.clear();
    let mut times = Vec::new();
    for _ in 0..2 {
        for (x, z) in [(-20.0, 0.0), (0.0, 0.0)] {
            let h = p.handoffs();
            let west = x < -6.0;
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut crossed: Option<f64> = None;
            loop {
                let f = p.frame();
                p.steer(x, z, 9.0);
                let s = p.ship();
                let past = if west {
                    s.position[0] < -6.0 - HYSTERESIS
                } else {
                    s.position[0] > -6.0 + HYSTERESIS
                };
                if past && crossed.is_none() {
                    crossed = Some(s.tick);
                }
                if f.state.handoffs != h {
                    times.push((s.tick - crossed.expect("crossed first")) * dt());
                    break;
                }
                assert!(Instant::now() < deadline, "no handoff: {:?}", f.state);
                std::thread::sleep(Duration::from_millis(7));
            }
            p.fly_to(&mut [], x, z, 9.0, 1.0);
            p.slow(&mut [], 0.3);
        }
    }
    eprintln!("[demo] from a hysteresis past the border to the switch, s: {times:.2?}");
    let flight = examine("demo", &p.log);
    assert_eq!(cluster.report(&rt), 0, "a player's handoff was called off");
    assert_eq!(flight.handoffs, 4);
    for took in &times {
        assert!(
            *took <= HANDOFF_BUDGET,
            "a crossing took {took:.2} s: {times:.2?}"
        );
    }
    assert!(
        flight.carried >= expected - 2,
        "in sight throughout: {flight:?}"
    );
    assert_eq!(flight.gaps, 0, "a body vanished for a frame: {flight:?}");
    assert_eq!(flight.doubles, 0, "a body was drawn twice: {flight:?}");
    assert!(
        flight.entity.holds(ENTITY_FLOOR),
        "a body jumped at a change of shard: {flight:?}"
    );
    assert!(
        flight.jump.holds(JUMP_FLOOR),
        "the drawn ship jumped: {flight:?}"
    );
    assert!(flight.clock.holds(CLOCK_FLOOR), "{flight:?}");
    let changes = inputs_applied_once("demo", &cluster.applied(&rt, p.guid()));
    assert_eq!(changes, 4);
    drop(p);
    cluster.stop();
}

/// A loop at fast-travel speed — some hundred metres a second over the demo
/// area, metres a tick — through all four cells, high above the showcase:
/// the frames around a change of shard show nothing the frames anywhere
/// else do not.
///
/// Fast travel is not smooth flight: the speed limit follows the distance to
/// the nearest surface, and the velocity follows the limit, so the drawn
/// velocity changes by metres per second from frame to frame all the time.
/// The measure is therefore relative — how far the drawn ship is from where
/// the frame before puts it, **in seconds of flight at the speed it is drawn
/// at** — and so is the bound: under [`FAST_FLOOR_S`], or no worse than
/// three times the 99.5th percentile of the frames away from every switch.
/// A clock that restarted, a prediction that was rebuilt or a ship that lost
/// a tick would be off by a tick of flight (33 ms) or more.
///
/// Mutation this catches: as for the loop at flying speed, at a speed where
/// a tick is metres.
#[test]
fn a_fast_loop_over_the_demo_area_is_seamless_at_the_borders() {
    /// See the test's docs: 15 ms of flight — not quite half a tick (33 ms).
    /// **Catches** a tick of flight lost or flown twice, a prediction rebuilt
    /// at the switch (two ticks and more). It does not tell a defect of
    /// less than half a tick from fast travel itself: on a quiet machine a
    /// frame of it is off by up to 2 ms of flight, and with the test pinned
    /// to two busy CPUs single frames reached 8 ms — where the speed limit
    /// kinks while a frame takes three times as long as usual.
    const FAST_FLOOR_S: f64 = 0.015;
    /// The speed from which a ship in fast travel counts as in flight, m/s.
    const FAST_FLIGHT: f64 = 60.0;
    if skipped() {
        return;
    }
    let _timed = Timed::start("fast");
    let rt = runtime();
    let cluster = Cluster::start(&rt, Some(Scene::DemoArea));
    let mut p = Player::online(&rt, cluster.entry());
    // Up, above the showcase.
    p.key(Key::Up, true);
    p.idle(&mut [], 2.2);
    p.key(Key::Up, false);
    p.slow(&mut [], 0.1);
    assert!(
        p.ship().position[1] > 20.0,
        "above the showcase: {:?}",
        p.ship()
    );
    p.log.clear();
    p.app.input(ViewerInput::Text('f'));
    let mut crossings = 0;
    // (The first border is 6 m from the spawn point: the ship is still being
    // flung up to speed when it crosses it. The other three are crossed in
    // flight.) Each leg heads for a far waypoint and ends a second and a half
    // after the change of shard — full ahead all the way, so that the
    // switch, the second before it and the second after it are flight.
    for (x, z) in [
        (-400.0, 300.0),
        (-400.0, -400.0),
        (300.0, -400.0),
        (300.0, 300.0),
    ] {
        let h = p.handoffs();
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut switched: Option<Instant> = None;
        loop {
            let f = p.frame();
            let s = p.ship();
            let (dx, dz) = (x - s.position[0], z - s.position[2]);
            if f.state.handoffs != h && switched.is_none() {
                switched = Some(Instant::now());
            }
            let flown_on = switched.is_some_and(|at| at.elapsed() >= Duration::from_millis(1_500));
            if flown_on || (dx * dx + dz * dz).sqrt() < 110.0 {
                break;
            }
            p.face(dx, dz);
            p.key(Key::Forward, true);
            assert!(Instant::now() < deadline, "never got to {x} {z}: {s:?}");
            std::thread::sleep(Duration::from_millis(7));
        }
        p.key(Key::Forward, false);
        p.until(&mut [], 3.0, |p| p.handoffs() != h);
        p.idle(&mut [], 0.8);
        assert_eq!(p.handoffs(), h + 1, "one handoff on the leg to ({x}, {z})");
        crossings += 1;
    }
    p.app.input(ViewerInput::Text('f'));
    p.idle(&mut [], 0.6);
    let flight = examine("fast", &p.log);
    // How far off, in seconds of flight at the speed it was drawn at: the
    // raw position error (fast travel does not accelerate by `ACCEL`, so
    // nothing is taken off it). Frames in which the ship is not in flight
    // yet — below [`FAST_FLIGHT`], where it is being flung up to speed at
    // hundreds of m/s² — are left out, around a switch and away from one.
    let switches: Vec<usize> = (1..p.log.len())
        .filter(|&i| p.log[i].state.handoffs != p.log[i - 1].state.handoffs)
        .collect();
    let mut off_s = Vec::new();
    let mut fastest = 0.0f64;
    for i in 1..p.log.len() {
        let (Some(a), Some(b)) = (p.log[i - 1].ship, p.log[i].ship) else {
            continue;
        };
        let span = (b.tick - a.tick) * dt();
        let mid = [0, 1, 2].map(|k| 0.5 * f64::from(a.velocity[k] + b.velocity[k]));
        let off = norm([0, 1, 2].map(|k| b.position[k] - a.position[k] - mid[k] * span));
        let near = switches.iter().any(|&s| {
            let (at, sw) = (p.log[i].at, p.log[s].at);
            if at <= sw {
                sw.duration_since(at).as_secs_f64() <= BEFORE_SWITCH
            } else {
                at.duration_since(sw).as_secs_f64() <= AFTER_SWITCH
            }
        });
        if norm(mid) < FAST_FLIGHT {
            continue;
        }
        if near {
            fastest = fastest.max(norm(mid));
        }
        off_s.push((near, off / norm(mid)));
    }
    let off = Split::of(&off_s);
    eprintln!(
        "[fast] around the {} switches (up to {fastest:.0} m/s): {:.2} ms of flight off; 99.5 % of \
         the {} frames elsewhere: {:.2} ms",
        switches.len(),
        1e3 * off.near,
        off.away_frames,
        1e3 * off.away,
    );
    assert_eq!(cluster.report(&rt), 0, "a player's handoff was called off");
    assert_eq!(flight.handoffs, crossings);
    assert_eq!(crossings, 4, "both borders, both ways");
    assert!(fastest > 100.0, "it was fast there: {fastest} m/s");
    // Three of the four switches were flown through at speed: frames in
    // flight right at them, which is what the measure above is taken on.
    let at_speed = switches
        .iter()
        .filter(|&&sw| {
            p.log[sw.saturating_sub(3)..=(sw + 3).min(p.log.len() - 1)]
                .iter()
                .filter_map(|f| f.ship)
                .all(|s| norm(s.velocity.map(f64::from)) >= FAST_FLIGHT)
        })
        .count();
    assert!(at_speed >= 3, "{at_speed} of the switches were in flight");
    assert!(
        off.holds(FAST_FLOOR_S),
        "the drawn ship jumped at a change of shard: {:.2} ms of flight (elsewhere {:.2} ms)",
        1e3 * off.near,
        1e3 * off.away
    );
    assert!(
        flight.clock.holds(CLOCK_FLOOR),
        "the clock stepped: {flight:?}"
    );
    assert_eq!(flight.gaps + flight.doubles, 0, "{flight:?}");
    let changes = inputs_applied_once("fast", &cluster.applied(&rt, p.guid()));
    assert_eq!(changes, crossings);
    let replica = p.app.replica().expect("online");
    assert_eq!(replica.tick_epoch, 0, "the ticks never started over");
    drop(p);
    cluster.stop();
}

/// The yardsticks themselves: the split takes its percentile from the frames
/// away from a switch and its worst from the ones around one, and a bound
/// holds under the floor or within the noise — not otherwise.
#[test]
fn the_bounds_compare_a_switch_with_ordinary_flight() {
    // A quiet flight: a millimetre everywhere, 0.8 m at the switch.
    let mut values: Vec<(bool, f64)> = (0..1_000).map(|_| (false, 0.001)).collect();
    values.push((true, 0.8));
    let split = Split::of(&values);
    assert_eq!((split.near, split.away), (0.8, 0.001));
    assert!(!split.holds(JUMP_FLOOR), "a rebuilt prediction is caught");
    // A rough flight: a centimetre in two frames of a hundred, and three
    // centimetres at the switch — as rough as flight is there, not rougher.
    let rough: Vec<(bool, f64)> = (0..1_000)
        .map(|i| (false, if i % 50 == 0 { 0.011 } else { 0.001 }))
        .chain([(true, 0.03)])
        .collect();
    let split = Split::of(&rough);
    assert_eq!(split.away, 0.011);
    assert!(split.holds(JUMP_FLOOR));
    // The same flight with a tick of flight lost at the switch is caught.
    let lost: Vec<(bool, f64)> = rough.iter().copied().chain([(true, 0.4)]).collect();
    assert!(!Split::of(&lost).holds(JUMP_FLOOR));
    // A single outlier away from the switches does not move the yardstick.
    let outlier: Vec<(bool, f64)> = (0..1_000)
        .map(|i| (false, if i == 7 { 5.0 } else { 0.001 }))
        .chain([(true, 0.4)])
        .collect();
    assert!(!Split::of(&outlier).holds(JUMP_FLOOR));
    // Too few ordinary frames: the floor alone decides.
    let few: Vec<(bool, f64)> = (0..50).map(|_| (false, 1.0)).chain([(true, 0.1)]).collect();
    assert!(!Split::of(&few).holds(JUMP_FLOOR));
    // The record of applied inputs: a hole at a switch is caught, a tick
    // applied twice is caught, a clean record passes.
    let clean: Vec<(u64, u32)> = (100..200)
        .map(|t| (t, if t < 150 { 1 } else { 2 }))
        .collect();
    assert_eq!(inputs_applied_once("clean", &clean), 1);
    let holed: Vec<(u64, u32)> = clean.iter().copied().filter(|(t, _)| *t != 151).collect();
    let record = Record::of(&holed);
    assert_eq!(
        (record.lost_at_switch.clone(), record.lost_elsewhere),
        (vec![151], 0)
    );
    let far: Vec<(u64, u32)> = clean.iter().copied().filter(|(t, _)| *t != 120).collect();
    let record = Record::of(&far);
    assert_eq!((record.lost_at_switch.len(), record.lost_elsewhere), (0, 1));
    let mut twice = clean.clone();
    twice.push((149, 2));
    twice.sort_unstable();
    assert_eq!(Record::of(&twice).twice, vec![149]);
}
