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
//!   rubber band ([`MAX_JUMP`]); its velocity has no step either
//!   ([`MAX_DV`]);
//! * **the clock everything is drawn by** runs on through a change of shard
//!   ([`MAX_CLOCK_STEP`]);
//! * **the prediction** is not corrected by more than a hair when the new
//!   shard's first word arrives ([`MAX_CORRECTION`]);
//! * **no input is lost**: a key pressed while the handoff is in flight
//!   moves the ship as soon as one pressed anywhere else;
//! * **everything else in sight** — bodies, other players — is drawn in every
//!   frame it was drawn in before and after, once, and does not jump;
//! * **one handoff per crossing**, none for standing on a border or walking
//!   along it, and none for a player who stands still while another flies
//!   past.
//!
//! None of these render (the frames are built, not drawn): no GPU needed.

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

/// How far the drawn ship may be from where the frame before, the drawn
/// velocity and the drawn time between the two frames put it, metres — at
/// any speed up to [`CRUISE`]. A ship at 15 m/s moves half a metre a tick;
/// this is a twentieth of that, 1.7 ms of flight: below what a frame shows.
/// (Before the cluster clock the drawn ship was off by about two ticks of
/// flight at every change of shard — 0.7 to 0.9 m at 13 m/s — and glided
/// back over a third of a second.)
const MAX_JUMP: f64 = 0.025;

/// How much the drawn velocity may change between two frames, m/s: what full
/// thrust (6 m/s²) does in two ticks, and a little. A ship that was frozen
/// for the handoff, or replayed from a stale state, would lose or gain
/// several ticks' worth at once.
const MAX_DV: f64 = 0.5;

/// How far the drawn tick may run ahead of or behind the wall clock between
/// two frames, ticks. The clock steers gently; a change of shard that
/// restarted it would step it by whatever the two shards' ticks differ.
const MAX_CLOCK_STEP: f64 = 0.35;

/// The largest correction the prediction may need, metres, the first word of
/// a new shard included: that of a ship in flight without a handoff.
const MAX_CORRECTION: f64 = 0.02;

/// How long a handoff may take from the moment the drawn ship is
/// [`HYSTERESIS`] across a border until the client plays on the other shard:
/// the second the shards give the client to connect, and a little.
const HANDOFF_BUDGET: Duration = Duration::from_millis(1_600);

/// `space_server::border::SHIP_HYSTERESIS`.
const HYSTERESIS: f64 = 1.0;

/// The speed the pilot flies at, m/s.
const CRUISE: f64 = 15.0;

/// The ship's acceleration under full thrust, m/s².
const ACCEL: f64 = 6.0;

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
                TickClock::new(SceneClock::DEFAULT_ORIGIN_US),
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

    fn stop(self) {
        for s in self.shards {
            for task in &s.tasks {
                task.abort();
            }
            s.net.close();
        }
    }
}

/// One frame of a player, as drawn.
#[derive(Clone, Debug)]
struct Frame {
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
        let at = Instant::now();
        self.app.update();
        let _ = self.app.scene(640, 360);
        self.app.finish_frame(640, 360, None);
        let f = Frame {
            at,
            state: self.app.client_state(640, 360),
            ship: self.app.drawn_ship(),
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

    /// Brake to a stop (the brake assist, switched on for it).
    fn stop(&mut self, others: &mut [&mut Player]) {
        self.key(Key::Forward, false);
        self.app.input(ViewerInput::Text('b'));
        self.idle(others, 2.5);
        self.app.input(ViewerInput::Text('b'));
    }
}

fn norm(v: [f64; 3]) -> f64 {
    v.iter().map(|c| c * c).sum::<f64>().sqrt()
}

/// What two consecutive frames show of the own ship.
#[derive(Clone, Copy, Debug)]
struct Step {
    /// How far the drawn ship is from where the frame before, the drawn
    /// velocity and the drawn time between them put it, metres.
    jump: f64,
    /// The same, with the wall clock's time between the frames.
    jump_wall: f64,
    /// The change of the drawn velocity, m/s.
    dv: f64,
    /// Drawn ticks between the two frames, minus the wall clock's.
    clock: f64,
    /// The drawn speed, m/s.
    speed: f64,
}

fn step(a: &Frame, b: &Frame) -> Option<Step> {
    let (sa, sb) = (a.ship?, b.ship?);
    let wall = b.at.duration_since(a.at).as_secs_f64();
    let mid = [0, 1, 2].map(|k| 0.5 * f64::from(sa.velocity[k] + sb.velocity[k]));
    let moved = [0, 1, 2].map(|k| sb.position[k] - sa.position[k]);
    let off = |seconds: f64| norm([0, 1, 2].map(|k| moved[k] - mid[k] * seconds));
    Some(Step {
        jump: off((sb.tick - sa.tick) * dt()),
        jump_wall: off(wall),
        dv: norm([0, 1, 2].map(|k| f64::from(sb.velocity[k] - sa.velocity[k]))),
        clock: (sb.tick - sa.tick) - wall / dt(),
        speed: norm(mid),
    })
}

/// The worst of a flight's frames, overall and around each change of shard.
#[derive(Debug, Default)]
struct Flight {
    handoffs: usize,
    jump: f64,
    jump_wall: f64,
    dv: f64,
    clock: f64,
    correction: f64,
    /// Entities drawn before and after a switch but missing in a frame
    /// between, entities drawn twice in a frame, and the largest move of one
    /// between two frames beyond what its speed explains, metres.
    gaps: usize,
    doubles: usize,
    entity_jump: f64,
    /// The fewest entities that were in sight across a switch.
    carried: usize,
}

/// Go through a log: every frame's step, and around every change of shard
/// what was in sight. Prints one line per switch (`--nocapture`).
fn examine(name: &str, log: &[Frame]) -> Flight {
    let mut out = Flight {
        carried: usize::MAX,
        ..Flight::default()
    };
    let steps: Vec<Option<Step>> = (1..log.len()).map(|i| step(&log[i - 1], &log[i])).collect();
    for s in steps.iter().flatten() {
        out.jump = out.jump.max(s.jump);
        out.jump_wall = out.jump_wall.max(s.jump_wall);
        out.dv = out.dv.max(s.dv);
        out.clock = out.clock.max(s.clock.abs());
    }
    out.correction = log.iter().map(|f| f.state.correction_m).fold(0.0, f64::max);
    let switches: Vec<usize> = (1..log.len())
        .filter(|&i| log[i].state.handoffs != log[i - 1].state.handoffs)
        .collect();
    out.handoffs = switches.len();
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
        let (mut gaps, mut doubles, mut entity_jump) = (0usize, 0usize, 0.0f64);
        let mut last: HashMap<EntityGuid, ([f64; 3], [f64; 3], Instant)> = HashMap::new();
        for frame in &log[lo..=hi] {
            let mut seen: HashMap<EntityGuid, usize> = HashMap::new();
            for (g, p) in frame.bodies.iter().chain(&frame.others) {
                *seen.entry(*g).or_default() += 1;
                let wall = last
                    .get(g)
                    .map_or(0.0, |(_, _, at)| frame.at.duration_since(*at).as_secs_f64());
                let v = match last.get(g) {
                    Some((q, _, _)) if wall > 0.0 => [0, 1, 2].map(|k| (p[k] - q[k]) / wall),
                    _ => [0.0; 3],
                };
                if let Some((q, w, _)) = last.get(g) {
                    // Against the velocity it was drawn moving at before.
                    let off = norm([0, 1, 2].map(|k| p[k] - q[k] - w[k] * wall));
                    if norm(*w) > 0.0 || norm(v) > 0.0 {
                        entity_jump = entity_jump.max(off);
                    }
                }
                last.insert(*g, (*p, v, frame.at));
            }
            doubles += seen.values().filter(|n| **n > 1).count();
            gaps += through.iter().filter(|g| !seen.contains_key(g)).count();
        }
        out.gaps += gaps;
        out.doubles += doubles;
        out.entity_jump = out.entity_jump.max(entity_jump);
        let near: Vec<&Step> = steps[lo..hi.min(steps.len())].iter().flatten().collect();
        let worst = |f: &dyn Fn(&Step) -> f64| near.iter().map(|s| f(s)).fold(0.0f64, f64::max);
        let f = &log[i];
        eprintln!(
            "[{name}]   handoff {} -> shard {:?} at {:.1?}, {:.1} m/s: jump {:.4} m (wall {:.4}), \
             dv {:.3} m/s, clock {:.2} ticks, correction {:.4} m; in sight throughout {} \
             (before {}, after {}), gaps {gaps}, doubles {doubles}, entity jump {:.3} m",
            f.state.handoffs,
            f.state.shard,
            f.ship.map(|s| s.position).unwrap_or([0.0; 3]),
            worst(&|s| s.speed),
            worst(&|s| s.jump),
            worst(&|s| s.jump_wall),
            worst(&|s| s.dv),
            worst(&|s| s.clock.abs()),
            log[lo..=hi]
                .iter()
                .map(|f| f.state.correction_m)
                .fold(0.0, f64::max),
            through.len(),
            before.len(),
            after.len(),
            entity_jump,
        );
    }
    eprintln!(
        "[{name}] {} frames, {} handoffs: worst jump {:.4} m (wall {:.4}), dv {:.3} m/s, \
         clock {:.2} ticks, correction {:.4} m; gaps {}, doubles {}, entity jump {:.3} m",
        log.len(),
        out.handoffs,
        out.jump,
        out.jump_wall,
        out.dv,
        out.clock,
        out.correction,
        out.gaps,
        out.doubles,
        out.entity_jump,
    );
    out
}

/// A loop through all four cells and over both borders, both ways, at flying
/// speed: nothing a frame shows gives a change of shard away.
///
/// Mutations this catches (each was applied, and the test failed): shards
/// counting from their own start (the clock steps by their difference, 4 to
/// 12 ticks here, and the ship with it); a client that starts over at every
/// switch — clock, prediction, sent inputs (a jump of the two or three ticks
/// the new prediction lacks: about 0.8 m at this speed). (Inputs sent to the
/// old shard only do not show here — the pilot coasts most of the way —
/// and are `an_input_during_the_handoff_is_not_lost`'s to catch.)
#[test]
fn a_loop_through_all_four_cells_is_seamless() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut p = Player::online(&rt, cluster.entry());
    let mut crossings = 0;
    let mut slowest = Duration::ZERO;
    let loops = 2;
    for _ in 0..loops {
        for (x, z) in [(-40.0, 20.0), (-40.0, -40.0), (30.0, -40.0), (30.0, 20.0)] {
            let h = p.handoffs();
            let from = p.frame().state.shard;
            p.fly_to(&mut [], x, z, CRUISE, 3.0);
            // One border lies on every leg: one handoff, to the cell's shard.
            let deadline = Instant::now() + Duration::from_secs(5);
            while p.handoffs() == h && Instant::now() < deadline {
                p.idle(&mut [], 0.05);
            }
            assert_eq!(p.handoffs(), h + 1, "one handoff on the leg to ({x}, {z})");
            assert_ne!(p.frame().state.shard, from);
            crossings += 1;
        }
    }
    p.stop(&mut []);
    // From the frame the drawn ship was a hysteresis across a border to the
    // frame the client was on the other shard.
    let cell = |s: &DrawnShip, margin: f64| {
        (
            s.position[0] < -6.0 - margin,
            s.position[0] > -6.0 + margin,
            s.position[2] < -4.0 - margin,
            s.position[2] > -4.0 + margin,
        )
    };
    let mut crossed: Option<Instant> = None;
    let mut inside = p.log[0].ship.map(|s| cell(&s, HYSTERESIS));
    for w in p.log.windows(2) {
        let Some(s) = w[1].ship else { continue };
        let now = cell(&s, HYSTERESIS);
        // Clear of both borders, and in another cell than before.
        if (now.0 || now.1) && (now.2 || now.3) {
            if inside.is_some_and(|was| was != now) && crossed.is_none() {
                crossed = Some(w[1].at);
            }
            inside = Some(now);
        }
        if w[1].state.handoffs != w[0].state.handoffs {
            if let Some(at) = crossed.take() {
                slowest = slowest.max(w[1].at.duration_since(at));
            }
        }
    }
    let flight = examine("loop", &p.log);
    eprintln!("[loop] slowest handoff from a hysteresis past the border: {slowest:?}");
    assert_eq!(cluster.report(&rt), 0, "a player's handoff was called off");
    assert_eq!(flight.handoffs, crossings, "one handoff per crossing");
    assert_eq!(crossings, 4 * loops);
    assert!(flight.jump <= MAX_JUMP, "the drawn ship jumped: {flight:?}");
    assert!(
        flight.jump_wall <= 2.0 * MAX_JUMP,
        "by the wall clock: {flight:?}"
    );
    assert!(
        flight.dv <= MAX_DV,
        "the drawn velocity stepped: {flight:?}"
    );
    assert!(
        flight.clock <= MAX_CLOCK_STEP,
        "the clock stepped: {flight:?}"
    );
    assert!(
        flight.correction <= MAX_CORRECTION,
        "the prediction was corrected: {flight:?}"
    );
    assert!(slowest <= HANDOFF_BUDGET, "a handoff took {slowest:?}");
    let replica = p.app.replica().expect("online");
    assert_eq!(replica.tick_epoch, 0, "the ticks never started over");
    drop(p);
    cluster.stop();
}

/// A key pressed while a handoff is in flight — before the commit, at it,
/// just after it — moves the ship as soon as a key pressed far from any
/// border does.
///
/// The ship coasts west over `x = -6`; at a set time after it is a
/// hysteresis across (the offer goes out there, the commit follows a second
/// later) the pilot pushes up, and the time until the **shard's** word shows
/// the ship rising is taken. The same far from the border is the usual
/// latency: the input's lead and the frame's way back.
///
/// Mutation this catches: inputs sent to the old shard only (one pressed in
/// the ticks around the commit is applied by nobody until the switch: two to
/// four ticks late), an input-ack floor in the wrong clock (the new shard's
/// word is taken for covering inputs it never applied: the prediction drops
/// them and the ship stalls), inputs for a staged ship dropped by the
/// accepting shard.
#[test]
fn an_input_during_the_handoff_is_not_lost() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut p = Player::online(&rt, cluster.entry());
    // Until the shard's word says the ship climbs (or sinks): seconds from
    // the key press.
    let respond = |p: &mut Player, up: bool| -> (f64, f64) {
        let key = if up { Key::Up } else { Key::Down };
        let vy0 = p.frame().state.velocity.map_or(0.0, |v| v[1]);
        let drawn0 = f64::from(p.ship().velocity[1]);
        let pressed = Instant::now();
        p.key(key, true);
        let (mut drawn, mut shard) = (None, None);
        while shard.is_none() {
            let f = p.frame();
            let sign = if up { 1.0 } else { -1.0 };
            if drawn.is_none() && sign * (f64::from(p.ship().velocity[1]) - drawn0) > 0.05 {
                drawn = Some(pressed.elapsed().as_secs_f64());
            }
            if sign * (f.state.velocity.map_or(0.0, |v| v[1]) - vy0) > 0.05 {
                shard = Some(pressed.elapsed().as_secs_f64());
            }
            assert!(
                pressed.elapsed() < Duration::from_secs(3),
                "no answer at all"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        // Held for a moment, then let go.
        p.idle(&mut [], 0.15);
        p.key(key, false);
        (drawn.unwrap_or(f64::NAN), shard.unwrap_or(f64::NAN))
    };
    // The usual latency, well inside shard 1.
    p.fly_to(&mut [], 20.0, 20.0, CRUISE, 1.0);
    p.stop(&mut []);
    let mut usual = Vec::new();
    for i in 0..4 {
        usual.push(respond(&mut p, i % 2 == 0));
        p.idle(&mut [], 0.4);
    }
    let usual_shard = usual.iter().map(|u| u.1).fold(0.0, f64::max);
    let usual_drawn = usual.iter().map(|u| u.0).fold(0.0, f64::max);
    eprintln!("[input] the usual latency (drawn, shard's word): {usual:.3?}");
    // Over the border, pressing at a set time after the offer went out.
    let mut during = Vec::new();
    for (i, after) in [0.5, 0.9, 0.97, 1.03, 1.1, 1.25].into_iter().enumerate() {
        let (east, west) = (6.0, -30.0);
        let (from, to) = if i % 2 == 0 {
            (east, west)
        } else {
            (west, east)
        };
        p.fly_to(&mut [], from, 20.0, CRUISE, 1.0);
        p.stop(&mut []);
        let h = p.handoffs();
        // Up to speed towards the border, then coast across it.
        let deadline = Instant::now() + Duration::from_secs(30);
        let past = |x: f64| {
            if to < from {
                x < -6.0 - HYSTERESIS
            } else {
                x > -6.0 + HYSTERESIS
            }
        };
        loop {
            p.frame();
            let s = p.ship();
            if past(s.position[0]) {
                break;
            }
            if norm(s.velocity.map(f64::from)) < 10.0 {
                p.face(to - from, 0.0);
                p.key(Key::Forward, true);
            } else {
                p.key(Key::Forward, false);
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(5));
        }
        p.key(Key::Forward, false);
        p.idle(&mut [], after);
        let before = p.handoffs();
        let answer = respond(&mut p, i % 2 == 0);
        let switched_during = p.handoffs() != before;
        p.idle(&mut [], 1.0);
        assert_eq!(p.handoffs(), h + 1, "one handoff per crossing");
        eprintln!(
            "[input] pressed {after:.2} s after the offer ({}): drawn {:.3} s, shard's word {:.3} s",
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
    }
    let worst_shard = during.iter().map(|u| u.1).fold(0.0, f64::max);
    let worst_drawn = during.iter().map(|u| u.0).fold(0.0, f64::max);
    // Two ticks of slack: the shard's word arrives once a tick, and so do the
    // inputs leave.
    assert!(
        worst_shard <= usual_shard + 2.0 * dt(),
        "an input during the handoff was answered late: {worst_shard:.3} s, usually {usual_shard:.3} s"
    );
    assert!(
        worst_drawn <= usual_drawn + 2.0 * dt(),
        "the drawn ship answered late: {worst_drawn:.3} s, usually {usual_drawn:.3} s"
    );
    let flight = examine("input", &p.log);
    assert!(flight.correction <= MAX_CORRECTION, "{flight:?}");
    assert_eq!(cluster.report(&rt), 0);
    drop(p);
    cluster.stop();
}

/// Standing on a border, and walking along it: no handoff. Stopping just
/// across it: one, three seconds later, and no more.
///
/// Mutation this catches: no hysteresis for ships (walking along `x = -6`
/// and straying 0.3 m to either side was 13 handoffs in 14 legs), a
/// hysteresis without the settle rule (a player who stops half a metre
/// across stays on the wrong shard for good), a settle rule that hands a
/// ship back and forth.
#[test]
fn standing_on_a_border_and_walking_along_it_is_no_ping_pong() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut p = Player::online(&rt, cluster.entry());
    p.fly_to(&mut [], -6.0, 10.0, 3.0, 0.1);
    p.stop(&mut []);
    assert_eq!(
        p.handoffs(),
        0,
        "it never was a metre across: {:?}",
        p.ship()
    );
    assert!((p.ship().position[0] + 6.0).abs() < 0.5, "{:?}", p.ship());
    // Standing there: if it stopped a hair across, it is taken over once it
    // has settled there (three seconds) — once — and from then on it stays.
    p.idle(&mut [], 5.0);
    let h0 = p.handoffs();
    let across = p.ship().position[0] < -6.0;
    assert_eq!(h0, u64::from(across), "at {:.3?}", p.ship().position);
    p.idle(&mut [], 6.0);
    eprintln!(
        "[border] stood at {:.3?} for 11 s: {h0} handoff(s) in the first 5 s, {} after",
        p.ship().position,
        p.handoffs() - h0
    );
    assert_eq!(p.handoffs(), h0, "standing on the border");
    // Along it for 20 s, weaving 0.3 m to either side.
    let end = Instant::now() + Duration::from_secs(20);
    let (mut legs, mut furthest) = (0, 0.0f64);
    while Instant::now() < end {
        let x = if legs % 2 == 0 { -6.3 } else { -5.7 };
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            p.frame();
            furthest = furthest.max((p.ship().position[0] + 6.0).abs());
            if p.steer(x, 12.0 + f64::from(legs) * 3.0, 2.0) < 0.3 {
                break;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(7));
        }
        legs += 1;
    }
    p.stop(&mut []);
    eprintln!(
        "[border] walked {legs} legs along it, at most {furthest:.2} m from it, to {:.2?}: {} handoffs",
        p.ship().position,
        p.handoffs() - h0
    );
    assert!(legs >= 6 && furthest < 0.9, "{legs} legs, {furthest} m");
    // Never a metre across, and never three seconds in a row on the other
    // side: nothing to hand over. (One is allowed for where it came to rest
    // at the end, a hair across.)
    assert!(
        p.handoffs() - h0 <= 1,
        "walking along the border: {} handoffs",
        p.handoffs() - h0
    );
    p.idle(&mut [], 4.0);
    let h1 = p.handoffs();
    // Half a metre into the other shard's ground, and stay: taken over once
    // it has settled there, and not handed back.
    let on = p.frame().state.shard.expect("on a shard");
    let (target, other) = if on == 1 { (-6.5, 2) } else { (-5.5, 1) };
    let z = p.ship().position[2];
    p.fly_to(&mut [], target, z, 2.0, 0.1);
    p.stop(&mut []);
    assert!(
        (p.ship().position[0] - target).abs() < 0.4,
        "half a metre across: {:?}",
        p.ship()
    );
    p.idle(&mut [], 4.0);
    assert_eq!(p.handoffs(), h1 + 1, "settled across the border");
    assert_eq!(p.frame().state.shard, Some(other));
    p.idle(&mut [], 5.0);
    assert_eq!(p.handoffs(), h1 + 1, "and stays there");
    let flight = examine("border", &p.log);
    assert!(
        flight.jump <= MAX_JUMP && flight.correction <= MAX_CORRECTION,
        "{flight:?}"
    );
    assert_eq!(cluster.report(&rt), 0);
    drop(p);
    cluster.stop();
}

/// One player stands, another flies past and across the border and back,
/// again and again — a metre and a half beside the one who stands, then nine
/// metres: the one who stands is never handed anywhere, keeps the session it
/// had, and sees the other in every frame, once, moving without a jump.
///
/// The one who flies joined first: its ship has the lower guid and is the
/// key of the island the two ships form whenever its second of motion
/// reaches the other.
///
/// Mutation this catches: a player's ship moved with its island (the one who
/// stands went to the neighbour shard and back at every pass), a ghost of a
/// player taken for something an island's bodies need not follow, the other
/// player's history started over when it changed from own entity to ghost.
#[test]
fn a_standing_player_is_not_handed_over_because_another_flies_past() {
    let rt = runtime();
    let cluster = Cluster::start(&rt, None);
    let mut mover = Player::online(&rt, cluster.entry());
    mover.fly_to(&mut [], 8.0, 0.0, 6.0, 0.5);
    mover.stop(&mut []);
    let mover_guid = mover.app.replica().and_then(|r| r.ship).expect("a ship");
    for beside in [1.5, 9.0] {
        let mut stander = Player::online(&rt, cluster.entry());
        // `beside` metres beside the mover's path (z = 0), near the border.
        stander.fly_to(&mut [&mut mover], -3.0, beside, 4.0, 0.3);
        stander.stop(&mut [&mut mover]);
        let at = stander.frame();
        let stood = stander.ship().position;
        let (h0, m0) = (at.state.handoffs, mover.handoffs());
        stander.log.clear();
        let passes = 4;
        for _ in 0..passes {
            mover.fly_to(&mut [&mut stander], -30.0, 0.0, CRUISE, 3.0);
            mover.fly_to(&mut [&mut stander], 12.0, 0.0, CRUISE, 3.0);
        }
        mover.stop(&mut [&mut stander]);
        let s = stander.frame();
        eprintln!(
            "[stander {beside} m] stood at {stood:.2?}: {} handoffs, session {:?} -> {:?}; the \
             other crossed {} times",
            s.state.handoffs - h0,
            at.state.session,
            s.state.session,
            mover.handoffs() - m0,
        );
        assert_eq!(mover.handoffs() - m0, 2 * passes, "the mover's crossings");
        assert_eq!(s.state.handoffs, h0, "the one who stands was handed over");
        assert_eq!(s.state.session, at.state.session, "it keeps its session");
        assert_eq!(s.state.shard, Some(1));
        let moved = norm([0, 1, 2].map(|k| stander.ship().position[k] - stood[k]));
        assert!(moved < 0.05, "it did not move: {moved} m");
        // What it saw of the other: in every frame, once, without a jump.
        let mut worst = 0.0f64;
        let mut last: Option<([f64; 3], [f64; 3], Instant)> = None;
        for f in &stander.log {
            let seen: Vec<&[f64; 3]> = f
                .others
                .iter()
                .filter(|(g, _)| *g == mover_guid)
                .map(|(_, p)| p)
                .collect();
            assert_eq!(seen.len(), 1, "the other player, once in every frame");
            let p = *seen[0];
            if let Some((q, w, at)) = last {
                let wall = f.at.duration_since(at).as_secs_f64();
                if wall > 0.0 {
                    worst = worst.max(norm([0, 1, 2].map(|k| p[k] - q[k] - w[k] * wall)));
                    last = Some((p, [0, 1, 2].map(|k| (p[k] - q[k]) / wall), f.at));
                }
            } else {
                last = Some((p, [0.0; 3], f.at));
            }
        }
        eprintln!("[stander {beside} m] the other's worst step beyond its motion: {worst:.3} m");
        // Half a tick of flight: it is drawn from a 30 Hz history, between
        // two frames that are not a tick apart.
        assert!(worst <= 0.3, "the other player jumped by {worst} m");
        drop(stander);
        mover.idle(&mut [], 0.5);
    }
    assert_eq!(cluster.report(&rt), 0);
    drop(mover);
    cluster.stop();
}

/// From the spawn point over `x = -6` through the demo area's zero-g corner
/// and back, three times: every crossing is one handoff within the budget and
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
    p.key(Key::Up, true);
    p.idle(&mut [], 1.2);
    p.key(Key::Up, false);
    p.stop(&mut []);
    let height = p.ship().position[1];
    assert!((2.5..9.0).contains(&height), "at height {height}");
    p.log.clear();
    let mut times = Vec::new();
    for _ in 0..3 {
        for (x, z) in [(-22.0, 0.0), (0.0, 0.0)] {
            let h = p.handoffs();
            let west = x < -6.0;
            let deadline = Instant::now() + Duration::from_secs(60);
            let mut crossed: Option<Instant> = None;
            loop {
                let f = p.frame();
                p.steer(x, z, 10.0);
                let px = p.ship().position[0];
                let past = if west {
                    px < -6.0 - HYSTERESIS
                } else {
                    px > -6.0 + HYSTERESIS
                };
                if past && crossed.is_none() {
                    crossed = Some(f.at);
                }
                if f.state.handoffs != h {
                    times.push(f.at.duration_since(crossed.expect("crossed first")));
                    break;
                }
                assert!(Instant::now() < deadline, "no handoff: {:?}", f.state);
                std::thread::sleep(Duration::from_millis(7));
            }
            p.fly_to(&mut [], x, z, 10.0, 1.0);
            p.stop(&mut []);
        }
    }
    eprintln!("[demo] from a hysteresis past the border to the switch: {times:.2?}");
    let flight = examine("demo", &p.log);
    assert_eq!(cluster.report(&rt), 0, "a player's handoff was called off");
    assert_eq!(flight.handoffs, 6);
    for took in &times {
        assert!(
            *took <= HANDOFF_BUDGET,
            "a crossing took {took:?}: {times:?}"
        );
    }
    assert!(
        flight.carried >= expected - 2,
        "in sight throughout: {flight:?}"
    );
    assert_eq!(flight.gaps, 0, "a body vanished for a frame: {flight:?}");
    assert_eq!(flight.doubles, 0, "a body was drawn twice: {flight:?}");
    assert!(flight.entity_jump <= 0.12, "a body jumped: {flight:?}");
    assert!(flight.jump <= MAX_JUMP, "the drawn ship jumped: {flight:?}");
    assert!(flight.clock <= MAX_CLOCK_STEP, "{flight:?}");
    drop(p);
    cluster.stop();
}

/// A loop at fast-travel speed — up to some 300 m/s over the demo area, ten
/// metres a tick — through all four cells, high above the showcase: the
/// frames in which the shard changes show nothing a frame anywhere else does
/// not.
///
/// Fast travel is not smooth flight: the speed limit follows the distance to
/// the nearest surface, and the velocity follows the limit, so the drawn
/// velocity changes by metres per second from frame to frame all the time.
/// The bound is therefore relative: at a change of shard the drawn ship is
/// within [`FAST_JUMP_S`] of flight of where the frame before puts it. A
/// clock that restarted, a prediction that was rebuilt or a ship that lost a
/// tick would be off by a tick of flight or more, over ten times that.
///
/// Mutation this catches: as for the loop at flying speed, at a speed where
/// a tick is ten metres.
#[test]
fn a_fast_loop_over_the_demo_area_is_seamless_at_the_borders() {
    /// See the test's docs: 2.5 ms of flight (a tick is 33 ms).
    const FAST_JUMP_S: f64 = 0.0025;
    let rt = runtime();
    let cluster = Cluster::start(&rt, Some(Scene::DemoArea));
    let mut p = Player::online(&rt, cluster.entry());
    // Up, above the showcase.
    p.key(Key::Up, true);
    p.idle(&mut [], 4.0);
    p.key(Key::Up, false);
    p.stop(&mut []);
    p.log.clear();
    p.app.input(ViewerInput::Text('f'));
    let mut crossings = 0;
    for _ in 0..2 {
        for (x, z) in [
            (-400.0, 300.0),
            (-400.0, -400.0),
            (300.0, -400.0),
            (300.0, 300.0),
        ] {
            let h = p.handoffs();
            // Steady: full ahead at the waypoint, let go near it.
            let deadline = Instant::now() + Duration::from_secs(60);
            loop {
                p.frame();
                let s = p.ship();
                let (dx, dz) = (x - s.position[0], z - s.position[2]);
                if (dx * dx + dz * dz).sqrt() < 110.0 {
                    break;
                }
                p.face(dx, dz);
                p.key(Key::Forward, true);
                assert!(Instant::now() < deadline, "never got to {x} {z}: {s:?}");
                std::thread::sleep(Duration::from_millis(7));
            }
            p.key(Key::Forward, false);
            p.idle(&mut [], 1.5);
            assert_eq!(p.handoffs(), h + 1, "one handoff on the leg to ({x}, {z})");
            crossings += 1;
        }
    }
    p.app.input(ViewerInput::Text('f'));
    p.idle(&mut [], 1.0);
    // The frames in which the shard changed (and the two either side),
    // against all the others.
    let switches: Vec<usize> = (1..p.log.len())
        .filter(|&i| p.log[i].state.handoffs != p.log[i - 1].state.handoffs)
        .collect();
    let (mut at_switch, mut elsewhere) = (Vec::new(), Vec::new());
    for i in 1..p.log.len() {
        let Some(s) = step(&p.log[i - 1], &p.log[i]) else {
            continue;
        };
        if switches.iter().any(|sw| i.abs_diff(*sw) <= 2) {
            at_switch.push(s);
        } else {
            elsewhere.push(s);
        }
    }
    let worst =
        |steps: &[Step], f: &dyn Fn(&Step) -> f64| steps.iter().map(f).fold(0.0f64, f64::max);
    // How far off, in seconds of flight at the speed it was drawn at.
    let off_s = |s: &Step| s.jump / s.speed.max(CRUISE);
    let mut sorted: Vec<f64> = elsewhere.iter().map(off_s).collect();
    sorted.sort_by(f64::total_cmp);
    let flight = examine("fast", &p.log);
    eprintln!(
        "[fast] at the {} switches (speed up to {:.0} m/s): jump {:.3} m = {:.2} ms of flight, \
         dv {:.2} m/s, clock {:.2} ticks; elsewhere: jump {:.3} m, median {:.2} ms / 99 % {:.2} ms \
         / worst {:.2} ms of flight, dv {:.2} m/s",
        switches.len(),
        worst(&at_switch, &|s| s.speed),
        worst(&at_switch, &|s| s.jump),
        1e3 * worst(&at_switch, &off_s),
        worst(&at_switch, &|s| s.dv),
        worst(&at_switch, &|s| s.clock.abs()),
        worst(&elsewhere, &|s| s.jump),
        1e3 * sorted[sorted.len() / 2],
        1e3 * sorted[sorted.len() * 99 / 100],
        1e3 * sorted[sorted.len() - 1],
        worst(&elsewhere, &|s| s.dv),
    );
    assert_eq!(cluster.report(&rt), 0, "a player's handoff was called off");
    assert_eq!(flight.handoffs, crossings);
    assert!(worst(&at_switch, &|s| s.speed) > 100.0, "it was fast there");
    assert!(
        worst(&at_switch, &off_s) <= FAST_JUMP_S,
        "the drawn ship jumped at a change of shard: {:.2} ms of flight",
        1e3 * worst(&at_switch, &off_s)
    );
    assert!(
        flight.clock <= MAX_CLOCK_STEP,
        "the clock stepped: {flight:?}"
    );
    assert_eq!(flight.gaps + flight.doubles, 0, "{flight:?}");
    let replica = p.app.replica().expect("online");
    assert_eq!(replica.tick_epoch, 0, "the ticks never started over");
    drop(p);
    cluster.stop();
}
