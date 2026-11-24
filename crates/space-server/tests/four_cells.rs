//! Physics on four shards (`docs/plan-sharding.md` §7): the demo area cut
//! into the four shard areas a `--min-shards 4` cluster gives it — the
//! borders are the planes `x = −6` and `z = −4` of the scene, and the line
//! where they cross — against the same thing on **one** shard.
//!
//! The fixture is four [`Shard`]s in one process on one cluster clock, their
//! messages carried between them by the test once per tick (what
//! `tests/crossing.rs` does for two): deterministic, no wall clock. Every
//! scenario is a function of a [`Cluster`], so it runs on the four-cell map
//! and on a single shard that owns all four cells, and the two runs are
//! compared body by body, tick by tick.
//!
//! ```text
//!            x = −6
//!   shard 4    |   shard 2          −z
//!   planetoid  |   the bowl end      ^
//!   -----------+----------- z = −4   |
//!   shard 3    |   shard 1           +--> +x
//!   zero-g     |   spawn point, most of the gravity room
//! ```
//!
//! (Further out, one piece of the scene's far content in each of the three
//! cells beside the root's: the giant rock on shard 4, the drifters on
//! shard 2, the far rock on shard 3.)
//!
//! # What "the same as on one shard" can mean
//!
//! * A body that is **never handed over**, and touches nothing that was, is
//!   **bit-identical** to the one-shard run at every tick.
//! * A body handed over **in free flight** — weightless, in orbit, falling —
//!   is bit-identical too: a handoff moves the whole state, and the shard
//!   that takes it steps it on the tick after the one the state describes.
//! * A body handed over **in contact** is not: the solver's memory of a
//!   contact (its warm start, the order of its constraints) stays behind,
//!   and the receiving shard solves the contact cold on the first tick. The
//!   difference is what one cold solve differs from a warm one — measured
//!   here as tens of micrometres for a crate at rest and under a millimetre a
//!   tick for one sliding — and it is then carried on by whatever the body
//!   does: nothing for a body that rests, centimetres for three balls
//!   colliding in a bowl (a collision amplifies any difference). The tests
//!   pin the bound per case and say which case it is.
//! * Two bodies on two shards that are tied by a **spring** each see the
//!   other end as a ghost, one tick stale and moved on along its velocity:
//!   the force is evaluated from a position that is off by half a tick's
//!   acceleration. Pinned in [`a_spring_across_the_corner_swings_as_on_one_shard`].
//!
//! | test | what it pins |
//! |---|---|
//! | [`the_far_content_is_on_its_shards_once_and_moves_as_on_one_shard`] | the far content (171 m, 706 m and 1.2 km out): spawned once, one handoff each to the cell's shard, nobody's ghost; three minutes of orbits, swing and tumbling bit-identical to one shard |
//! | [`the_showcase_rests_for_a_minute_on_four_shards`] | every object of the showcase at rest for 60 s: one handoff per body to its shard at the start, then none; nothing called off; nothing creeps; against one shard |
//! | [`bodies_pushed_across_each_border_and_through_the_corner_fly_as_on_one_shard`] | free flight across `x = −6`, `z = −4`, the corner, along a border: bit-identical |
//! | [`a_crate_slides_and_a_crate_falls_across_the_border_in_gravity`] | handed over in contact and while falling: no sink, no hop |
//! | [`a_late_commit_does_not_sink_or_hop_a_falling_or_a_sliding_crate`] | the catch-up of a body |
//! | [`a_stack_on_the_border_stands_and_is_pushed_over_across_it`] | a stack whose crates lie on both sides; a player pushing it over |
//! | [`bodies_resting_against_and_on_a_ghost_do_not_move`] | body against ghost, stack on a ghost, for the ticks before they are consolidated |
//! | [`a_ship_set_down_on_a_slab_of_the_neighbour_stands_still`] | a ship on a ghost body |
//! | [`a_spring_across_the_corner_swings_as_on_one_shard`] | a spring whose ends are on two shards |
//! | [`a_pendulum_and_a_chain_swinging_over_the_border_stay_on_their_shard`] | hysteresis for what swings across a border |
//! | [`a_player_passing_the_showcase_moves_no_body_it_does_not_reach`] | walking and flying past: 0 body handoffs |
//! | [`an_island_waits_for_a_member_that_is_on_its_way_before_it_moves_on`] | two shards never offer each other members of one island |
//! | [`a_body_ahead_of_a_player_who_is_changing_shard_waits_where_it_is`] | bodies do not follow a player back to the shard it is leaving |
//! | [`the_floating_crate_on_the_border_is_struck_the_same_on_one_shard`] | a ship striking a floating crate: the collider, not the border |
//! | [`the_same_script_on_one_shard_and_on_four`] | bodies dropped and pushed, a player walking through: body by body |

use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_protocol::{
    EntityGuid, InputTick, Message, NeighbourStatus, PlayerInput, Scene, SceneClock, SceneState,
    ShardId, ShardMap, ShardRegion,
};
use space_server::border;
use space_server::shard::{Destination, Shard};
use space_server::ShardConfig;
use space_sim::body::{Body, MassSpec};
use space_sim::shape::ShapeRef;
use space_sim::spring::{Spring, SpringMode, SpringTarget, Springs};
use space_sim::{BodySpawn, ShipState, SimTick};
use space_world::{NodeRef, WorldPos};

// ------------------------------------------------------------------ fixture

/// The cluster's tick origin in these tests.
const ORIGIN: i64 = 1_763_900_000_000_000;
/// The tick every shard is put on the clock at.
const START: SimTick = 1_000;
/// How far ahead of the newest tick a client tags its inputs
/// (`space_client::link::INPUT_LEAD`).
const INPUT_LEAD: SimTick = 3;
/// The border planes through the scene, anchor-relative metres.
const BORDER_X: f64 = -6.0;
const BORDER_Z: f64 = -4.0;
/// The floor of the demo area.
const FLOOR: f64 = space_scene::demo::FLOOR;
/// A lane of the gravity room's floor that is clear across `z = −4`: between
/// the west curb and the crate stack.
const LANE_X: f64 = 10.9;

/// The scene bodies the root's shard — which spawns the scene — hands to the
/// shards whose ground they lie on, at the start of every four-shard run:
/// 16 of the showcase's 35 and the far content's 11 (`space_scene::demo`,
/// "Far content").
const SCENE_HANDOFFS: usize = 16 + 11;

const ROOT: ShardId = ShardId::new(1);
/// `x > −6`, `z < −4`: the bowl end of the gravity room.
const NORTH: ShardId = ShardId::new(2);
/// `x < −6`, `z > −4`: the zero-g corner.
const WEST: ShardId = ShardId::new(3);
/// `x < −6`, `z < −4`: the planetoid.
const DIAGONAL: ShardId = ShardId::new(4);

fn listening(n: u32) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_300 + n as u16)
}

/// Shards stepped by hand on one cluster clock, running the demo area.
struct Cluster {
    shards: Vec<Shard>,
    /// Frames for clients: `(sending shard, session, message)`.
    frames: Vec<(ShardId, u64, Message)>,
    /// Shard-to-shard messages held back: `(deliver at round, from, to, message)`.
    held: Vec<(u64, ShardId, ShardId, Message)>,
    /// How many rounds a `HandoffCommit` is held back.
    commit_delay: u64,
    round: u64,
    /// The scene's anchor: what positions are measured from.
    anchor: WorldPos,
    /// The scene's bodies, in definition order.
    names: Vec<(&'static str, EntityGuid)>,
    /// Bodies a test spawned, in the order it did.
    extra: Vec<EntityGuid>,
}

impl Cluster {
    /// The orchestrator's layout for four shards: the root's shard and the
    /// three cells carved out beside the spawn point's.
    fn four() -> Self {
        Self::of(&[
            (1, NodeRef::new(63, [0, 0, 0])),
            (2, NodeRef::new(64, [1, 1, 0])),
            (3, NodeRef::new(64, [0, 1, 1])),
            (4, NodeRef::new(64, [0, 1, 0])),
        ])
    }

    /// One shard that owns all of it: where nothing is ever handed over.
    fn one() -> Self {
        Self::of(&[(1, NodeRef::new(63, [0, 0, 0]))])
    }

    fn of(regions: &[(u32, NodeRef)]) -> Self {
        let map = ShardMap::new(
            1,
            regions
                .iter()
                .map(|(id, node)| ShardRegion::new(*node, ShardId::new(*id), listening(*id)))
                .collect(),
        );
        let mut shards: Vec<Shard> = regions
            .iter()
            .map(|(id, node)| {
                let mut s = Shard::new(ShardConfig::new(ShardId::new(*id), listening(*id), *node));
                s.start_clock(SceneClock::cluster(ORIGIN), START);
                s.set_shard_map(map.clone());
                s
            })
            .collect();
        for s in &mut shards {
            assert!(s.set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
        }
        let anchor = shards[0].installed_scene().expect("the scene is built").1;
        let names = space_scene::body_defs(Scene::DemoArea)
            .iter()
            .enumerate()
            .map(|(i, d)| (d.name, border::scene_guid(ROOT, 1, i)))
            .collect();
        let c = Self {
            shards,
            frames: Vec::new(),
            held: Vec::new(),
            commit_delay: 0,
            round: 0,
            anchor,
            names,
            extra: Vec::new(),
        };
        if regions.len() == 4 {
            // The map is the one the picture in the module docs shows.
            assert_eq!(c.owner([0.0, 0.0, 0.0]), ROOT, "the spawn point");
            assert_eq!(c.owner([BORDER_X + 1.0, 0.0, BORDER_Z - 1.0]), NORTH);
            assert_eq!(c.owner([BORDER_X - 1.0, 0.0, BORDER_Z + 1.0]), WEST);
            assert_eq!(c.owner([BORDER_X - 1.0, 0.0, BORDER_Z - 1.0]), DIAGONAL);
        }
        c
    }

    fn shard(&self, id: ShardId) -> &Shard {
        self.shards.iter().find(|s| s.shard() == id).expect("shard")
    }

    fn shard_mut(&mut self, id: ShardId) -> &mut Shard {
        self.shards
            .iter_mut()
            .find(|s| s.shard() == id)
            .expect("shard")
    }

    /// Collect what every shard queued; deliver what is due between shards.
    fn deliver(&mut self) {
        for shard in &mut self.shards {
            let from = shard.shard();
            for out in shard.take_outbox() {
                match out.dest {
                    Destination::Client(session) => self.frames.push((from, session, out.message)),
                    Destination::Shard(to) => {
                        let delay = if matches!(out.message, Message::HandoffCommit(_)) {
                            self.commit_delay
                        } else {
                            0
                        };
                        self.held.push((self.round + delay, from, to, out.message));
                    }
                    Destination::Orchestrator => {}
                }
            }
        }
        let round = self.round;
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.held)
            .into_iter()
            .partition(|(at, ..)| *at <= round);
        self.held = later;
        for (_, from, to, message) in due {
            self.shard_mut(to).peer_message(from, message);
        }
    }

    /// One tick of every shard, in id order, then one round of delivery —
    /// and the one thing that must hold after every tick of every test: no
    /// entity is simulated by two shards.
    fn tick(&mut self) {
        for shard in &mut self.shards {
            shard.tick();
        }
        self.round += 1;
        self.deliver();
        if self.shards.len() > 1 {
            let mut seen = BTreeSet::new();
            for shard in &self.shards {
                for body in shard.body_list() {
                    assert!(
                        body.ghost || seen.insert(body.guid),
                        "{} is simulated by two shards at once",
                        body.guid
                    );
                }
            }
        }
    }

    fn tick_n(&mut self, n: usize) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// The shards that simulate `guid` now.
    fn simulators(&self, guid: EntityGuid) -> Vec<ShardId> {
        self.shards
            .iter()
            .filter(|s| s.sim().ship_state(guid).is_some())
            .map(Shard::shard)
            .collect()
    }

    /// `guid`'s state where it is simulated — or, on the one tick it is in
    /// flight between two shards (committed by one, not yet installed by the
    /// other), the state it was committed with: what the shard that let it
    /// go keeps as its ghost.
    fn state(&self, guid: EntityGuid) -> Option<ShipState> {
        self.shards
            .iter()
            .find_map(|s| s.sim().ship_state(guid))
            .or_else(|| self.shards.iter().find_map(|s| s.sim().state_of(guid)))
    }

    /// Where `guid` is, in metres from the scene's spawn point.
    fn rel(&self, guid: EntityGuid) -> [f64; 3] {
        let state = self.state(guid).expect("the entity exists");
        WorldPos::delta(self.anchor, state.pos)
    }

    fn at(&self, rel: [f64; 3]) -> WorldPos {
        WorldPos::from_f64_metres_around(self.anchor, rel)
    }

    /// The shard the map gives the point `rel`.
    fn owner(&self, rel: [f64; 3]) -> ShardId {
        self.shards[0]
            .shard_map()
            .lookup(&self.at(rel))
            .expect("every point of the scene has an owner")
    }

    /// Spawn `body` at `rel` moving at `vel`, on the shard that owns the
    /// point (on one shard: there). Remembered in [`Cluster::extra`].
    fn spawn(&mut self, rel: [f64; 3], vel: [f32; 3], body: Body) -> EntityGuid {
        let state = ShipState {
            vel,
            ..ShipState::at_rest(self.at(rel))
        };
        self.spawn_as(BodySpawn::new(state, body))
    }

    fn spawn_as(&mut self, spawn: BodySpawn) -> EntityGuid {
        let owner = self.shards[0]
            .shard_map()
            .lookup(&spawn.state.pos)
            .expect("owned");
        let guid = self.shard_mut(owner).spawn_body(spawn).expect("it spawns");
        self.extra.push(guid);
        guid
    }

    /// Spawn `body` at `rel` on the **root's** shard, whoever owns the
    /// point: a body that finds itself on a neighbour's ground from its
    /// first tick, and is offered to the neighbour at once.
    fn spawn_on_root(&mut self, rel: [f64; 3], vel: [f32; 3], body: Body) -> EntityGuid {
        let state = ShipState {
            vel,
            ..ShipState::at_rest(self.at(rel))
        };
        let guid = self
            .shard_mut(ROOT)
            .spawn_body(BodySpawn::new(state, body))
            .expect("it spawns");
        self.extra.push(guid);
        guid
    }

    /// One of the scene's own bodies — by its name in the scene's
    /// definition — built as the scene builds it, but `shift` metres from
    /// where the scene has it.
    fn scene_body_shifted(&mut self, name: &str, shift: [f64; 3]) -> EntityGuid {
        let anchor = self.at(shift);
        let mut unused = 0u64;
        let body = space_scene::bodies(Scene::DemoArea, anchor, &mut || {
            unused += 1;
            EntityGuid::new(ShardId::new(999), unused)
        })
        .into_iter()
        .find(|b| b.name == name)
        .expect("a body of the demo area");
        self.spawn_as(body.spawn)
    }

    /// Where `guid` is on the shard that simulates it — and not a number
    /// on the one tick it is in flight between two shards (committed by
    /// one, installed by the other before its next step): the comparisons
    /// skip that sample. The state it is in flight with is the one the next
    /// tick starts from, so nothing escapes them by it.
    fn sample(&self, guid: EntityGuid) -> [f64; 3] {
        self.shards
            .iter()
            .find_map(|s| s.sim().ship_state(guid))
            .map_or([f64::NAN; 3], |s| WorldPos::delta(self.anchor, s.pos))
    }

    /// A [`Cluster::sample`] of every extra and every scene body, extras
    /// first.
    fn track(&self) -> Vec<[f64; 3]> {
        self.extra
            .iter()
            .copied()
            .chain(self.names.iter().map(|(_, g)| *g))
            .map(|g| self.sample(g))
            .collect()
    }

    /// The scene bodies called `name`.
    fn named(&self, name: &str) -> Vec<EntityGuid> {
        self.names
            .iter()
            .filter(|(n, _)| *n == name)
            .map(|(_, g)| *g)
            .collect()
    }

    /// Entities handed over so far, cluster-wide (ships and bodies).
    fn handed_off(&self) -> usize {
        self.shards.iter().map(|s| s.stats().handed_off).sum()
    }

    /// Ticks handed-over entities were brought forward at their install.
    fn caught_up(&self) -> SimTick {
        self.shards.iter().map(|s| s.stats().caught_up).sum()
    }

    /// No transfer was called off, anywhere.
    fn assert_nothing_called_off(&self) {
        for shard in &self.shards {
            assert_eq!(
                shard.stats().aborted,
                0,
                "{} called transfers off: {:#?}",
                shard.shard(),
                shard.recent_aborts()
            );
        }
    }

    /// Frames sent to `(shard, session)`, draining them.
    fn frames_for(&mut self, shard: ShardId, session: u64) -> Vec<Message> {
        let (taken, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.frames)
            .into_iter()
            .partition(|f| f.0 == shard && f.1 == session);
        self.frames = keep;
        taken.into_iter().map(|(_, _, m)| m).collect()
    }
}

/// A player as the client link plays it: one session on the shard that
/// simulates its ship, a second on the shard a redirect names, and every
/// input — tagged [`INPUT_LEAD`] ticks ahead — sent to both.
struct Pilot {
    token: String,
    ship: EntityGuid,
    on: ShardId,
    session: u64,
    next: Option<(ShardId, u64)>,
    /// How often the ship changed shard.
    switches: usize,
}

impl Pilot {
    /// Join: the root's shard owns the spawn point on every map here.
    fn join(c: &mut Cluster, token: &str) -> Self {
        let welcome = c
            .shard_mut(ROOT)
            .join(&space_net::hello(token))
            .expect("the root's shard owns the spawn point");
        let ship = c.shard(ROOT).ship_of(welcome.session).expect("a ship");
        Self {
            token: token.into(),
            ship,
            on: ROOT,
            session: welcome.session,
            next: None,
            switches: 0,
        }
    }

    /// Put the ship somewhere by hand (the shard that simulates it is told).
    fn put(&self, c: &mut Cluster, rel: [f64; 3], vel: [f32; 3]) {
        let state = ShipState {
            vel,
            ..ShipState::at_rest(c.at(rel))
        };
        assert!(c.shard_mut(self.on).set_state(self.ship, state));
    }

    /// One client turn and one tick of the cluster: read the frames, follow
    /// a redirect, notice a switch, send the input for the tick
    /// [`INPUT_LEAD`] ahead.
    fn step(&mut self, c: &mut Cluster, thrust: [f32; 3]) {
        for m in c.frames_for(self.on, self.session) {
            let Message::Redirect(r) = m else { continue };
            if r.entity != self.ship || self.next.is_some() {
                continue;
            }
            let welcome = c
                .shard_mut(r.new_shard)
                .join(&space_net::hello(&self.token))
                .expect("the accepting shard admits the ship's player");
            c.shard_mut(r.new_shard).client_status(
                welcome.session,
                NeighbourStatus {
                    entity: self.ship,
                    shard: r.new_shard,
                    connected: true,
                },
            );
            self.next = Some((r.new_shard, welcome.session));
        }
        if let Some((to, session)) = self.next {
            let _ = c.frames_for(to, session);
            if c.shard(to).sim().ship_state(self.ship).is_some() {
                c.shard_mut(self.on).leave(self.session);
                self.on = to;
                self.session = session;
                self.next = None;
                self.switches += 1;
            }
        }
        let tick = c.shard(self.on).sim().tick() + INPUT_LEAD;
        let batch = InputTick {
            tick,
            seq: tick,
            inputs: vec![PlayerInput {
                entity: self.ship,
                thrust,
                torque: [0.0; 3],
                tick,
                fast: false,
            }],
            ack_tick: 0,
        };
        c.shard_mut(self.on).apply_input(self.session, &batch);
        if let Some((to, session)) = self.next {
            c.shard_mut(to).apply_input(session, &batch);
        }
        c.tick();
    }

    fn steps(&mut self, c: &mut Cluster, thrust: [f32; 3], ticks: usize) {
        for _ in 0..ticks {
            self.step(c, thrust);
        }
    }
}

/// A crate as the demo area's are.
fn crate_body(size: f32) -> Body {
    let mut b = Body::dynamic(ShapeRef::cuboid([size; 3]));
    b.mass = MassSpec::Density(500.0);
    b.friction = 0.7;
    b.restitution = 0.05;
    b
}

fn ball_body(radius: f32) -> Body {
    let mut b = Body::dynamic(ShapeRef::Ball { radius });
    b.mass = MassSpec::Density(800.0);
    b.friction = 0.6;
    b.restitution = 0.2;
    b
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// Where every tracked entity was after every tick of a run (not a number
/// where it was in flight: [`Cluster::sample`]).
type Tracks = Vec<Vec<[f64; 3]>>;

/// The same scenario on the four-cell map and on one shard: both runs'
/// tracks, and the four-shard cluster as the scenario left it.
fn on_four_and_on_one(scenario: impl Fn(&mut Cluster) -> Tracks) -> (Tracks, Tracks, Cluster) {
    let mut four = Cluster::four();
    let mut one = Cluster::one();
    let a = scenario(&mut four);
    let b = scenario(&mut one);
    assert_eq!(a.len(), b.len(), "the scenario is the same length on both");
    assert_eq!(one.handed_off(), 0, "one shard hands nothing over");
    (a, b, four)
}

/// The largest distance between the two runs' positions of entity `i`, and
/// the tick it was at (samples in flight are skipped: a comparison with one
/// is false).
fn worst_apart(a: &Tracks, b: &Tracks, i: usize) -> (f64, usize) {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| dist(ra[i], rb[i]))
        .enumerate()
        .fold((0.0, 0), |w, (t, d)| if d > w.0 { (d, t) } else { w })
}

/// How far entity `i` strayed from where it was at tick `from`, at most,
/// from then on.
fn strayed(a: &Tracks, i: usize, from: usize) -> f64 {
    assert!(a[from][i][0].is_finite(), "in flight at tick {from}");
    a[from..]
        .iter()
        .map(|row| dist(row[i], a[from][i]))
        .fold(0.0, f64::max)
}

// -------------------------------------------------------------------- at rest

/// The whole showcase left alone for a minute on four shards.
///
/// * At the start the root's shard, which spawns the scene, hands every body
///   of another shard's ground to that shard — one handoff each, all on the
///   same tick — and after that **nothing changes shard**: every body is
///   simulated by the shard the map names for where it is.
/// * **No transfer is called off**, from the first tick on.
/// * Nothing that rests creeps: not the crate floating exactly on `z = −4`,
///   not the stack and the pyramid 2.4 m from it, not the door on its hinge.
/// * Against one shard: every body the root's shard keeps is bit-identical
///   at every tick, and so is every body that was handed over weightless or
///   in orbit. The crates resting on the planetoid were handed over in
///   contact and settle within 0.1 mm of where one shard has them; the three
///   balls rolling into the bowl were handed over rolling and collide at its
///   bottom, and end within 2 cm.
///
/// Mutations this catches: an island that reaches as far as a body's speed
/// in **every** direction (the balls rolling down the ramp, away from the
/// bowl, were then one island with the balls rolling into it: the two shards
/// offered each other its members and one offer was called off, at tick 11
/// of every start); the distance hysteresis holding a body the spawner put
/// on a neighbour's ground (the third bowl ball lies 1.9 m from the border:
/// it followed nine ticks later, after the three had met as body and ghost,
/// and ended 18 cm from where one shard has it).
#[test]
fn the_showcase_rests_for_a_minute_on_four_shards() {
    const MINUTE: usize = 1_800;
    let (four, one, c) = on_four_and_on_one(|c| {
        let mut out = Tracks::new();
        for _ in 0..MINUTE {
            c.tick();
            out.push(c.track());
        }
        out
    });
    let foreign: Vec<EntityGuid> = c
        .names
        .iter()
        .map(|(_, g)| *g)
        .filter(|g| c.simulators(*g) != vec![ROOT])
        .collect();
    assert_eq!(
        foreign.len(),
        SCENE_HANDOFFS,
        "the bodies off the root's ground"
    );
    assert_eq!(
        c.handed_off(),
        foreign.len(),
        "each went to its shard once, and nothing else moved"
    );
    c.assert_nothing_called_off();
    assert_eq!(c.caught_up(), 0);
    for (i, (name, guid)) in c.names.iter().enumerate() {
        let here = c.rel(*guid);
        assert_eq!(
            c.simulators(*guid),
            vec![c.owner(here)],
            "{name} is on the shard that owns {here:?}"
        );
        let (apart, at) = worst_apart(&four, &one, i);
        let bound = match *name {
            "bowl ball" => 0.02,
            "planetoid crate" => 1.0e-4,
            _ => 0.0,
        };
        assert!(
            apart <= bound,
            "{name}: {apart:.6} m from the one-shard run at tick {at} (bound {bound})"
        );
        // What rests does not creep (the last 50 of the 60 seconds).
        let rests = !matches!(
            *name,
            "moon"
                | "swinging pendulum"
                | "bobbing pendulum"
                | "chain"
                | "ramp ball"
                | "far moon"
                | "giant moon"
        ) && !name.starts_with("spring pair")
            && !name.starts_with("drift pair");
        if rests {
            let crept = strayed(&four, i, 300);
            assert!(crept < 1.0e-3, "{name} crept {crept:.5} m");
        }
    }
    // The crate that floats exactly on the border did not move at all.
    let on_border = c
        .named("floating crate")
        .into_iter()
        .find(|g| c.rel(*g)[2] == BORDER_Z)
        .expect("the demo area floats a crate on z = -4");
    let i = c.names.iter().position(|(_, g)| *g == on_border).unwrap();
    assert_eq!(strayed(&four, i, 0), 0.0, "the crate on the border moved");
}

/// The far content — the drifters 171 m out, the far rock's moons 706 m
/// out, the giant rock's 1.2 km out (`space_scene::demo`, "Far content") —
/// over three minutes on four shards and on one.
///
/// * **Spawned once per cluster.** The four shards together simulate
///   exactly the bodies one shard does, each on one shard; the far content
///   is on the shard of its cell — the drifters on the north shard, the far
///   moons on the west one, the giant moons on the diagonal one — after one
///   handoff each from the root's shard, which spawned them, and never
///   changes shard again. Nothing is called off.
/// * **Nobody's ghost.** No shard ever holds a ghost of any of it: it lies
///   beyond the 64 m (plus three seconds of motion) of every border, which
///   is what makes it a far replica for a player on another shard
///   (`tests/far.rs`).
/// * **The same as on one shard, to the bit, at every tick**: orbits, swing
///   and tumbling are free flight, and a handoff in free flight is exact.
///   The moons stay within 1 % of their orbits' radii for the three minutes
///   (four to sixteen times round), on four shards as on one.
///
/// Mutations this catches (each was run): every shard spawning the scene's
/// bodies instead of the one spawner (`replace_scene_bodies` without the
/// spawner check: two shards simulate one guid at once — the fixture's own
/// assertion after every tick); the far rock 60 m from `x = −6` (its moons
/// are the west shard's ghosts on the root's shard); the giant rock at
/// +840, 320, +840, in the root's own cell (24 handoffs, and its moons on
/// the root's shard); a body brought forward at its install by a tick it
/// did not miss (`install_body`: 27 ticks caught up where there are none,
/// and every moon off the one-shard orbit).
#[test]
fn the_far_content_is_on_its_shards_once_and_moves_as_on_one_shard() {
    const TICKS: usize = 3 * 60 * 30;
    let far: [(&str, ShardId); 5] = [
        ("drift pair a", NORTH),
        ("drift pair b", NORTH),
        ("drift crate", NORTH),
        ("far moon", WEST),
        ("giant moon", DIAGONAL),
    ];
    let (four, one, c) = on_four_and_on_one(|c| {
        let content: Vec<EntityGuid> = far.iter().flat_map(|(name, _)| c.named(name)).collect();
        assert_eq!(content.len(), 11);
        let mut out = Tracks::new();
        for tick in 0..TICKS {
            c.tick();
            out.push(c.track());
            // (For the dozen ticks the handoff from the spawner takes, the
            // shard a body is going to holds it as the root's ghost — it
            // lies on that shard's own ground. Measured: gone by tick 12.)
            if tick < 30 {
                continue;
            }
            for shard in &c.shards {
                for guid in &content {
                    assert_eq!(
                        shard.ghost_owner(*guid),
                        None,
                        "{} holds a ghost of the far content's {guid} at tick {tick}",
                        shard.shard()
                    );
                }
            }
        }
        out
    });
    // Once per cluster: the four shards' own bodies are the one shard's.
    let own = |c: &Cluster| -> Vec<EntityGuid> {
        let mut all: Vec<EntityGuid> = c
            .shards
            .iter()
            .flat_map(|s| s.body_list())
            .filter(|b| !b.ghost)
            .map(|b| b.guid)
            .collect();
        all.sort();
        all
    };
    let scene: Vec<EntityGuid> = c.names.iter().map(|(_, g)| *g).collect();
    assert_eq!(
        scene.len(),
        46,
        "the showcase's 35 and the far content's 11"
    );
    assert_eq!(own(&c), scene, "every scene body once, and nothing else");
    assert_eq!(own(&c), own(&Cluster::one()));
    assert_eq!(
        c.handed_off(),
        SCENE_HANDOFFS,
        "one handoff each, at the start"
    );
    c.assert_nothing_called_off();
    assert_eq!(c.caught_up(), 0);
    let len = |v: [f64; 3]| dist(v, [0.0; 3]);
    for (name, shard) in far {
        for guid in c.named(name) {
            assert_eq!(c.simulators(guid), vec![shard], "{name}");
            assert_eq!(c.owner(c.rel(guid)), shard, "{name} lies in its cell");
            let i = c.names.iter().position(|(_, g)| *g == guid).unwrap();
            let (apart, at) = worst_apart(&four, &one, i);
            assert_eq!(apart, 0.0, "{name}: off the one-shard run at tick {at}");
            // It was handed over once, within the first ticks, and is
            // there in every sample but that one.
            let in_flight = four.iter().filter(|row| row[i][0].is_nan()).count();
            assert!(in_flight <= 1, "{name} was in flight {in_flight} times");
        }
    }
    use space_scene::demo;
    for (name, centre, orbits) in [
        ("far moon", demo::FAR_ROCK_CENTER, demo::FAR_ROCK_ORBITS),
        (
            "giant moon",
            demo::GIANT_ROCK_CENTER,
            demo::GIANT_ROCK_ORBITS,
        ),
    ] {
        for (guid, radius) in c.named(name).into_iter().zip(orbits) {
            let i = c.names.iter().position(|(_, g)| *g == guid).unwrap();
            let mut swept = 0.0;
            let mut last: Option<[f64; 3]> = None;
            for row in &four {
                if row[i][0].is_nan() {
                    continue;
                }
                let d = [0, 1, 2].map(|k| row[i][k] - centre[k]);
                assert!(
                    (len(d) - radius).abs() < radius * 0.01,
                    "{name} is {} m from its rock, not {radius}",
                    len(d)
                );
                if let Some(l) = last {
                    let cos = (0..3).map(|k| d[k] * l[k]).sum::<f64>() / (len(d) * len(l));
                    swept += cos.clamp(-1.0, 1.0).acos();
                }
                last = Some(d);
            }
            assert!(
                swept > 2.0 * std::f64::consts::TAU,
                "{name} went {swept} rad round in three minutes"
            );
        }
    }
    // The pair swings (4 m either way between its ends) and the tumbling
    // crates stay where they are.
    let (a, b) = (c.named("drift pair a")[0], c.named("drift pair b")[0]);
    let index = |g: EntityGuid| c.names.iter().position(|(_, n)| *n == g).unwrap();
    let apart: Vec<f64> = four
        .iter()
        .filter(|row| row[index(a)][0].is_finite() && row[index(b)][0].is_finite())
        .map(|row| dist(row[index(a)], row[index(b)]))
        .collect();
    let (lo, hi) = apart
        .iter()
        .fold((f64::MAX, f64::MIN), |(l, h), d| (l.min(*d), h.max(*d)));
    assert!(
        (lo - 12.0).abs() < 0.05 && (hi - 20.0).abs() < 0.05,
        "the pair swings between {lo} and {hi} m apart"
    );
    for guid in c.named("drift crate") {
        assert_eq!(
            strayed(&four, index(guid), 30),
            0.0,
            "a tumbling crate moved"
        );
    }
}

// ------------------------------------------------------------- free flight

/// Bodies pushed across each border, through the corner where the four cells
/// meet, along a border, and fast: weightless, so nothing but the handoff
/// could change their flight — and nothing does. Each is bit-identical to
/// the same push on one shard at every tick, is handed over exactly as often
/// as it crosses into a new shard's ground, and ends on the shard that owns
/// where it is.
///
/// Mutations this catches: a handed-over body stepped twice or not at all on
/// the tick it changes shard; a body measured against the wrong shard's
/// ground at the corner (it then stays with a shard it has left for good, or
/// is handed over twice).
#[test]
fn bodies_pushed_across_each_border_and_through_the_corner_fly_as_on_one_shard() {
    let (four, one, c) = on_four_and_on_one(|c| {
        // Across x = -6 (root -> west), spinning.
        let g = c.spawn([-3.0, 6.0, 10.0], [-2.0, 0.0, 0.0], crate_body(0.6));
        let on = c.simulators(g)[0];
        let mut spin = c.state(g).unwrap();
        spin.ang_vel = [0.4, 0.2, -0.3];
        assert!(c.shard_mut(on).set_state(g, spin));
        // Across z = -4 (west -> diagonal).
        c.spawn([-30.0, 6.0, -1.0], [0.0, 0.0, -2.0], crate_body(0.6));
        // Through the corner line itself (root -> diagonal).
        c.spawn([-3.0, 6.0, -1.0], [-2.0, 0.0, -2.0], ball_body(0.3));
        // Along x = -6, half a metre inside the west cell, drifting across
        // the border slowly and on.
        c.spawn([-6.5, 9.0, 10.0], [0.3, 0.0, 0.0], crate_body(0.6));
        // Fast, north -> root: 20 m/s is two thirds of a metre a tick.
        c.spawn([30.0, 6.0, -8.0], [0.0, 0.0, 20.0], crate_body(0.6));
        let mut out = Tracks::new();
        for _ in 0..360 {
            c.tick();
            out.push(c.track());
        }
        out
    });
    let expect = [WEST, DIAGONAL, DIAGONAL, ROOT, ROOT];
    for (i, guid) in c.extra.iter().enumerate() {
        let (apart, at) = worst_apart(&four, &one, i);
        assert_eq!(apart, 0.0, "body {i} left the one-shard path at tick {at}");
        assert_eq!(c.simulators(*guid), vec![expect[i]], "body {i}");
        assert_eq!(c.simulators(*guid), vec![c.owner(c.rel(*guid))]);
    }
    // One handoff per body (the scene's sixteen at the start aside): also
    // for the one that went through the corner, which never was a body of
    // the two cells beside its path.
    assert_eq!(c.handed_off(), SCENE_HANDOFFS + c.extra.len());
    c.assert_nothing_called_off();
}

// ------------------------------------------------------------------ gravity

/// The floor height a crate of `size` rests at.
fn resting(size: f64) -> f64 {
    FLOOR + size / 2.0
}

/// A crate sliding along the lane across `z = −4`, and one dropped from six
/// metres up, three metres behind the border, **by the root's shard**: it
/// is on the neighbour's ground, is offered at once and changes shard a
/// quarter of a second later, in mid-fall, at 2.7 m/s. Spawned at tick 0,
/// tracked as extras 0 and 1.
fn slide_and_drop(c: &mut Cluster) {
    // 9 m/s on a floor of friction 0.7: it slides six metres, from a metre
    // before the border to five behind it.
    c.spawn(
        [LANE_X, resting(0.6) + 0.005, -1.0],
        [0.0, 0.0, -9.0],
        crate_body(0.6),
    );
    // (East of the bowl: the slide ends in the lane, three metres behind
    // the border too.)
    c.spawn_on_root(
        [18.3, FLOOR + 6.0, BORDER_Z - 3.0],
        [0.0; 3],
        crate_body(0.4),
    );
}

/// A crate that is handed over **while it slides on the floor** and one that
/// is handed over **while it falls**, in the gravity room, across `z = −4`.
///
/// Neither sinks into the floor nor hops off it at the handoff: on every
/// tick the height of each is within a millimetre of the same slide and
/// drop on one shard (measured: 0.02 mm), and both stay within 2 cm of it
/// altogether (measured: 0.7 mm — the slide is handed over in contact, so
/// it is not bit-exact; see the module docs) and end at rest on the floor on
/// the far side.
///
/// Mutations this catches: gravity or contact skipped on the tick of the
/// handoff (the crate hops by `g·dt²`, 1 cm, or sinks by it); the body's
/// position residual dropped in the transfer.
#[test]
fn a_crate_slides_and_a_crate_falls_across_the_border_in_gravity() {
    let (four, one, c) = on_four_and_on_one(|c| {
        slide_and_drop(c);
        let mut out = Tracks::new();
        for _ in 0..240 {
            c.tick();
            out.push(c.track());
        }
        out
    });
    for (i, name) in ["sliding crate", "dropped crate"].iter().enumerate() {
        let guid = c.extra[i];
        let end = c.rel(guid);
        assert!(end[2] < BORDER_Z - 0.5, "the {name} crossed: {end:?}");
        assert_eq!(c.simulators(guid), vec![NORTH], "the {name}");
        let (apart, at) = worst_apart(&four, &one, i);
        println!("{name}: {apart:.6} m from one shard at tick {at}");
        assert!(apart < 0.02, "the {name}: {apart:.4} m at tick {at}");
        let v = c.state(guid).unwrap().vel;
        assert!(v.iter().all(|x| x.abs() < 0.01), "the {name} rests: {v:?}");
    }
    // Neither sinks nor hops: the heights, tick by tick, against one shard
    // (a crate braking from 9 m/s leans forward and rides on its front
    // edge for a while, on one shard as on four: its height is not its
    // resting height while it slides).
    for (i, name) in ["sliding crate", "dropped crate"].iter().enumerate() {
        let hop = four
            .iter()
            .zip(&one)
            .map(|(a, b)| (a[i][1] - b[i][1]).abs())
            .fold(0.0, f64::max);
        println!("{name}: height within {hop:.6} m of one shard");
        assert!(hop < 1.0e-3, "the {name} hopped or sank by {hop:.5} m");
    }
    assert!((c.rel(c.extra[1])[1] - resting(0.4)).abs() < 0.002);
    c.assert_nothing_called_off();
    assert_eq!(c.caught_up(), 0, "every commit was on time");
}

/// The same slide and throw with every commit **two ticks late**: the shard
/// that takes a body then has to bring it forward by the ticks it missed,
/// and does so along the body's velocity — without the pull of gravity,
/// which it cannot tell from the floor's push for a body it has no contacts
/// for yet (`handoff.rs`, guarantee 4).
///
/// What that costs is bounded and stated here: a body that was **falling**
/// is short of two ticks of gravity — 0.65 m/s of speed — from the install
/// on, so it falls behind the on-time run by up to half a metre and lands a
/// tick or two later, in the same place; a body **sliding on the floor** is
/// moved along the floor and neither sinks into it nor leaves it (its
/// height stays within 2 mm of the on-time run's on every tick).
///
/// Mutation this catches: a catch-up that applies gravity to a body on the
/// floor (it is installed `g·(n·dt)²/2` inside the floor and thrown out), or
/// none of the motion (it stands still for the ticks it missed).
#[test]
fn a_late_commit_does_not_sink_or_hop_a_falling_or_a_sliding_crate() {
    let run = |late: u64| {
        let mut c = Cluster::four();
        c.commit_delay = late;
        slide_and_drop(&mut c);
        let mut out = Tracks::new();
        for _ in 0..240 {
            c.tick();
            out.push(c.track());
        }
        (out, c)
    };
    let (on_time, _) = run(0);
    let (late, c) = run(2);
    assert!(
        c.caught_up() >= 4,
        "two bodies, two ticks each: {}",
        c.caught_up()
    );
    c.assert_nothing_called_off();
    let hop = late
        .iter()
        .zip(&on_time)
        .map(|(a, b)| (a[0][1] - b[0][1]).abs())
        .fold(0.0, f64::max);
    println!("late sliding crate: height within {hop:.6} m of the on-time run");
    assert!(
        hop < 2.0e-3,
        "the sliding crate hopped or sank by {hop:.5} m"
    );
    let (slide, at) = worst_apart(&late, &on_time, 0);
    println!("late sliding crate: {slide:.6} m from the on-time run at tick {at}");
    assert!(slide < 0.05, "the slide: {slide:.4} m at tick {at}");
    let (drop, at) = worst_apart(&late, &on_time, 1);
    println!("late dropped crate: {drop:.6} m from the on-time run at tick {at}");
    assert!(drop < 0.5, "the drop: {drop:.4} m at tick {at}");
    for i in 0..2 {
        assert_eq!(c.simulators(c.extra[i]), vec![NORTH]);
        let v = c.state(c.extra[i]).unwrap().vel;
        assert!(v.iter().all(|x| x.abs() < 0.01), "body {i} rests: {v:?}");
        let last = late.len() - 1;
        let ends = dist(late[last][i], on_time[last][i]);
        println!("late commit: body {i} ends {ends:.6} m from where it ends on time");
        assert!(ends < 0.05, "body {i} ends {ends:.4} m off");
    }
    assert!((c.rel(c.extra[1])[1] - resting(0.4)).abs() < 0.002);
}

// -------------------------------------------------------------- on a border

/// A stack standing **on** the border — three crates in the lane, `z = −4`
/// through the middle of them — and a player who walks into it and pushes it
/// over across the border.
///
/// Standing, it is one island on one shard: nothing is handed over, nothing
/// creeps, and it is bit-identical to the same stack on one shard. Pushed
/// over, its crates and the player change shard together: no crate is ever
/// under the floor, none is left on a shard that does not own where it came
/// to rest, and no transfer is called off.
///
/// Mutations this catches: an island split by the border (the crates of one
/// stack on two solvers: the upper ones sink into the ghost of the lowest);
/// bodies that do not follow the player who pushes them (each shard then
/// sees the push as a ghost's, with no weight behind it).
#[test]
fn a_stack_on_the_border_stands_and_is_pushed_over_across_it() {
    const STANDING: usize = 300;
    let (four, one, c) = on_four_and_on_one(|c| {
        for i in 0..3 {
            c.spawn(
                [LANE_X, resting(0.6) + 0.6 * f64::from(i), BORDER_Z],
                [0.0; 3],
                crate_body(0.6),
            );
        }
        let mut out = Tracks::new();
        for _ in 0..STANDING {
            c.tick();
            out.push(c.track());
        }
        assert_eq!(
            c.handed_off(),
            if c.shards.len() == 4 {
                SCENE_HANDOFFS
            } else {
                0
            }
        );
        // The player lands in the lane two and a half metres before the
        // stack, walks into it and stops.
        let mut pilot = Pilot::join(c, "pusher");
        pilot.put(c, [LANE_X, FLOOR + 1.7, BORDER_Z + 2.5], [0.0; 3]);
        for (thrust, ticks) in [([0.0; 3], 45), ([0.0, 0.0, -1.0], 38), ([0.0; 3], 360)] {
            for _ in 0..ticks {
                pilot.step(c, thrust);
                out.push(c.track());
            }
        }
        if c.shards.len() == 4 {
            assert_eq!(pilot.on, NORTH, "the player ended behind the border");
        }
        out
    });
    for i in 0..3 {
        let standing = (0..STANDING)
            .map(|t| dist(four[t][i], one[t][i]))
            .fold(0.0, f64::max);
        assert_eq!(standing, 0.0, "crate {i}, standing");
        let crept = (60..STANDING)
            .map(|t| dist(four[t][i], four[60][i]))
            .fold(0.0, f64::max);
        assert!(crept < 1.0e-4, "crate {i} crept {crept:.6} m");
    }
    let lowest = |t: &Tracks| {
        t.iter()
            .flat_map(|row| row[..3].iter().map(|p| p[1]))
            .fold(f64::INFINITY, f64::min)
    };
    println!(
        "pushed stack: lowest crate centre {:.4} on four, {:.4} on one (rest {:.4})",
        lowest(&four),
        lowest(&one),
        resting(0.6)
    );
    assert!(
        lowest(&four) > lowest(&one) - 0.002 && lowest(&four) > resting(0.6) - 0.02,
        "a crate was pushed into the floor: {:.3} (on one shard {:.3})",
        lowest(&four),
        lowest(&one)
    );
    for (i, g) in c.extra.iter().enumerate() {
        let here = c.rel(*g);
        println!(
            "pushed stack: crate {i} ends at {here:?} on {:?}",
            c.simulators(*g)
        );
        assert!(here[2] < BORDER_Z, "crate {i} was pushed across: {here:?}");
        assert_eq!(
            c.simulators(*g),
            vec![c.owner(here)],
            "crate {i} at {here:?}"
        );
        let v = c.state(*g).unwrap().vel;
        assert!(v.iter().all(|x| x.abs() < 0.01), "crate {i} rests: {v:?}");
    }
    c.assert_nothing_called_off();
}

/// Bodies that **touch across the border from their first tick**, each on
/// its own shard: for the ticks until the island rule has brought them onto
/// one shard, each rests against — or on — the other's **ghost**.
///
/// * two crates side by side on the floor, their faces a millimetre apart
///   on `z = −4`;
/// * a stack of two whose lower crate lies on the neighbour's ground and
///   whose upper crate on ours (a stack whose lower half is a ghost).
///
/// (The neighbour's crates are spawned three ticks before ours, so that
/// their ghosts are there when ours appear. Two bodies put into the world
/// touching on the very same tick by two shards do not see each other on
/// that tick — nothing in the game does that: a scene's bodies are all
/// spawned by one shard.)
///
/// No crate sinks into a ghost or is pushed out of one: every height stays
/// within 3 mm on every tick — a ghost at rest is exactly where its owner
/// has the body, so resting on it is resting on the body. Each pair ends on
/// one shard after one handoff, eleven ticks after it formed. (Sideways the
/// off-centre stack creeps while it settles, on one shard too: 3 mm there,
/// 8 mm here, where the handoff makes the solver start its contact afresh.)
///
/// Mutations this catches: a ghost held a tick behind or dead-reckoned while
/// at rest (what rests on it is pushed out or dropped by the difference);
/// consolidation that takes both halves (two handoffs per pair, or a
/// crossing that is called off); consolidation on the first pass, before
/// the other side's key is in (the handoff is two ticks earlier).
#[test]
fn bodies_resting_against_and_on_a_ghost_do_not_move() {
    let scenario = |c: &mut Cluster| {
        // Let the scene's own bodies get to their shards first.
        c.tick_n(30);
        // The neighbour's two first: a crate with its face a hair behind
        // the border, and the lower crate of a stack.
        c.spawn(
            [LANE_X - 0.4, resting(0.6), BORDER_Z - 0.3],
            [0.0; 3],
            crate_body(0.6),
        );
        c.spawn(
            [LANE_X + 0.4, resting(0.6), BORDER_Z - 0.05],
            [0.0; 3],
            crate_body(0.6),
        );
        // Three ticks later — their ghosts are on our side by then — ours:
        // a crate against the first, and one on top of the second, 10 cm
        // off its centre, so that it lies on our side of the border.
        c.tick_n(3);
        c.spawn(
            [LANE_X - 0.4, resting(0.6), BORDER_Z + 0.301],
            [0.0; 3],
            crate_body(0.6),
        );
        c.spawn(
            [LANE_X + 0.4, resting(0.6) + 0.6, BORDER_Z + 0.05],
            [0.0; 3],
            crate_body(0.6),
        );
        let mut out = Tracks::new();
        for _ in 0..300 {
            c.tick();
            out.push(c.track());
        }
        out
    };
    let (four, one, c) = on_four_and_on_one(scenario);
    assert_eq!(c.owner(c.rel(c.extra[0])), NORTH);
    assert_eq!(c.owner(c.rel(c.extra[1])), NORTH);
    assert_eq!(c.owner(c.rel(c.extra[2])), ROOT);
    assert_eq!(c.owner(c.rel(c.extra[3])), ROOT);
    for i in 0..4 {
        let moved = strayed(&four, i, 0);
        let sank = four
            .iter()
            .map(|row| (row[i][1] - four[0][i][1]).abs())
            .fold(0.0, f64::max);
        let (apart, at) = worst_apart(&four, &one, i);
        println!(
            "ghost rest: body {i} moved {moved:.6} m (on one shard {:.6}), its height {sank:.6} m; {apart:.6} m from one shard at tick {at}",
            strayed(&one, i, 0)
        );
        assert!(sank < 3.0e-3, "body {i} sank or rose by {sank:.5} m");
        assert!(moved < 0.015, "body {i} moved {moved:.5} m");
        assert!(
            apart < 0.01,
            "body {i}: {apart:.5} m from one shard at tick {at}"
        );
        assert_eq!(
            c.simulators(c.extra[i]),
            vec![ROOT],
            "consolidated on the root"
        );
    }
    // The neighbour's crates changed shard on the tick the rule says: the
    // island is two passes old (`CONFIRM_TICKS`: the other side's key of
    // the tick it formed on is in by then), plus the lead of a body's
    // handoff, plus the tick it is in flight.
    for i in 0..2 {
        let in_flight: Vec<usize> = (0..four.len())
            .filter(|t| four[*t][i][0].is_nan())
            .collect();
        assert_eq!(
            in_flight,
            vec![(border::CONFIRM_TICKS + border::BODY_HANDOFF_LEAD + 1) as usize],
            "body {i}"
        );
    }
    assert_eq!(c.handed_off(), SCENE_HANDOFFS + 2, "one handoff per pair");
    c.assert_nothing_called_off();
}

/// A player's ship set down on the end of a slab that lies on the
/// neighbour's ground: the ship is the root's, the slab the north shard's,
/// so the ship first stands on a **ghost**. The slab follows the player —
/// one handoff — and the ship stands still throughout: its height does not
/// change by a millimetre once it has landed, as on one shard.
///
/// Mutation this catches: bodies that go to the home of their island's
/// lowest guid without regard to who flies it (the ship's guid is the
/// lowest: the player would be handed to the slab's shard for standing on
/// it), or a ghost that gives way under a ship.
#[test]
fn a_ship_set_down_on_a_slab_of_the_neighbour_stands_still() {
    let scenario = |c: &mut Cluster| {
        c.tick_n(30);
        let mut slab = Body::dynamic(ShapeRef::cuboid([1.2, 0.4, 2.0]));
        slab.mass = MassSpec::Density(500.0);
        slab.friction = 0.7;
        slab.restitution = 0.0;
        // Its centre 0.6 m behind the border: 0.4 m of it reaches onto the
        // root's ground.
        c.spawn([LANE_X, resting(0.4), BORDER_Z - 0.6], [0.0; 3], slab);
        c.tick_n(1);
        let mut pilot = Pilot::join(c, "stander");
        pilot.put(c, [LANE_X, FLOOR + 0.4 + 1.72, BORDER_Z + 0.2], [0.0; 3]);
        let mut out = Tracks::new();
        for _ in 0..300 {
            pilot.step(c, [0.0; 3]);
            let mut row = c.track();
            row.insert(0, c.sample(pilot.ship));
            out.push(row);
        }
        assert_eq!(pilot.switches, 0, "the player stayed on its shard");
        out
    };
    let (four, one, c) = on_four_and_on_one(scenario);
    // Landing: on one shard the slab gives a little under the ship and the
    // two bounce by a centimetre for a third of a second; on four the ship
    // lands on the slab's ghost, which does not give, and stands at once.
    // From a second and a half after it was set down the ship does not
    // move, and stands where one shard has it.
    let (landing, at) = worst_apart(&four, &one, 0);
    let moved = strayed(&four, 0, 45);
    let apart = four[45..]
        .iter()
        .zip(&one[45..])
        .map(|(a, b)| dist(a[0], b[0]))
        .fold(0.0, f64::max);
    println!(
        "ship on slab: landing {landing:.6} m from one shard at tick {at}; then strayed {moved:.6} m, {apart:.6} m from one shard"
    );
    assert!(landing < 0.025, "the landing: {landing:.4} m at tick {at}");
    assert!(moved < 1.0e-3, "the ship moved {moved:.5} m on the slab");
    assert!(
        apart < 6.0e-3,
        "the ship stands {apart:.5} m from one shard's"
    );
    let slab_moved = strayed(&four, 1, 0);
    assert!(slab_moved < 5.0e-3, "the slab moved {slab_moved:.5} m");
    assert_eq!(
        c.simulators(c.extra[0]),
        vec![ROOT],
        "the slab followed the player"
    );
    assert_eq!(c.handed_off(), SCENE_HANDOFFS + 1);
    c.assert_nothing_called_off();
}

/// Two shards never offer each other members of one island: an island whose
/// home changes while one of its bodies is still on its way between the two
/// shards waits for that body, and then goes on together.
///
/// The scene that showed it: a crate slides across the border towards the
/// spot a second crate is falling onto, a moment after the falling crate was
/// handed to the border's far side — with every commit two ticks late, as
/// on a loaded host. The two are one island; its key is the sliding crate
/// (the lower guid), whose home is the near side until it is two metres
/// across. So the far side sends the falling crate **back**, to the key's
/// home — and while that transfer is under way the sliding crate gets its
/// two metres across, its home becomes the far side, and the near side
/// offered it there: one island's members offered both ways at once. The
/// far side refused ("crossing island transfer; the lower shard id takes
/// it"), and the sliding crate waited out the pause after a refusal.
///
/// Mutation this catches: offering an island's bodies while another body of
/// the island is staged to arrive from the shard they would go to.
#[test]
fn an_island_waits_for_a_member_that_is_on_its_way_before_it_moves_on() {
    let mut c = Cluster::four();
    c.commit_delay = 2;
    c.spawn(
        [LANE_X, resting(0.6) + 0.005, -1.0],
        [0.0, 0.0, -9.0],
        crate_body(0.6),
    );
    c.spawn_on_root(
        [LANE_X + 0.2, FLOOR + 6.0, BORDER_Z - 3.0],
        [0.0; 3],
        crate_body(0.4),
    );
    c.tick_n(300);
    c.assert_nothing_called_off();
    for (i, guid) in c.extra.iter().enumerate() {
        let here = c.rel(*guid);
        assert_eq!(
            c.simulators(*guid),
            vec![c.owner(here)],
            "crate {i} at {here:?}"
        );
    }
    println!(
        "crossing: {} handoffs beyond the scene's sixteen",
        c.handed_off() - SCENE_HANDOFFS
    );
}

// ------------------------------------------------------------------ springs

/// A spring of rest length `rest` to `guid`.
fn spring_to(guid: EntityGuid, rest: f32) -> Springs {
    Springs(vec![Spring {
        part: 0,
        anchor: [0.0; 3],
        target: SpringTarget::Entity {
            guid,
            part: 0,
            anchor: [0.0; 3],
        },
        rest_length: rest,
        stiffness: 40.0,
        damping: 0.0,
        max_force: 0.0,
        mode: SpringMode::Spring,
    }])
}

/// Two crates on a spring **across the corner**: one in the root's cell, one
/// in the diagonal cell, 5.7 m apart — the spring is the only thing between
/// them, so they are never one island and each stays on its shard. One is
/// given a push; they swing for twenty seconds.
///
/// Each shard computes the spring's force on its end from the other end's
/// ghost: the owner's state of a tick ago, moved on one tick along its
/// velocity. Measured against one shard: the two ends are never more than
/// 2 cm from where one shard has them over twenty seconds and a dozen
/// swings, and the swing neither grows nor dies (its largest stretch in the
/// last five seconds is within 2 % of that of the first five).
///
/// Mutations this catches: a ghost that is not moved on between updates (the
/// force lags a full tick: the swing gains energy every period and the ends
/// drift apart from the one-shard run by decimetres); spring ends that are
/// not ghosted to each other's shard beyond the ghost margin; a spring that
/// makes an island (the two crates would be pulled onto one shard).
#[test]
fn a_spring_across_the_corner_swings_as_on_one_shard() {
    let a_at = [-4.0, 5.0, -2.0];
    let b_at = [-8.0, 5.0, -6.0];
    let rest = dist(a_at, b_at);
    let scenario = |c: &mut Cluster| {
        let mut body = crate_body(0.5);
        body.linear_damping = 0.0;
        let b = c.spawn(b_at, [0.0; 3], body.clone());
        let state = ShipState::at_rest(c.at(a_at));
        let a = c.spawn_as(BodySpawn::new(state, body).with_springs(spring_to(b, rest as f32)));
        // At rest length nothing pulls: let the ghosts be there, then push.
        c.tick_n(30);
        // (Apart, each as hard as the other: the pair as a whole stays
        // where it is.)
        for (guid, v) in [(a, 0.3f32), (b, -0.3f32)] {
            let on = c.simulators(guid)[0];
            let mut pushed = c.state(guid).unwrap();
            pushed.vel = [v, 0.0, v];
            assert!(c.shard_mut(on).set_state(guid, pushed));
        }
        let mut out = Tracks::new();
        for _ in 0..600 {
            c.tick();
            out.push(c.track());
        }
        out
    };
    let (four, one, c) = on_four_and_on_one(scenario);
    assert_eq!(c.simulators(c.extra[0]), vec![DIAGONAL]);
    assert_eq!(c.simulators(c.extra[1]), vec![ROOT]);
    assert_eq!(c.handed_off(), SCENE_HANDOFFS, "neither end changed shard");
    c.assert_nothing_called_off();
    for i in 0..2 {
        let (apart, at) = worst_apart(&four, &one, i);
        println!("spring end {i}: {apart:.6} m from one shard at tick {at}");
        assert!(apart < 0.02, "end {i}: {apart:.4} m at tick {at}");
    }
    let stretch = |t: &Tracks, range: std::ops::Range<usize>| {
        t[range]
            .iter()
            .map(|row| (dist(row[0], row[1]) - rest).abs())
            .fold(0.0, f64::max)
    };
    let (early, late) = (stretch(&four, 0..150), stretch(&four, 450..600));
    println!(
        "spring stretch: {early:.4} m early, {late:.4} m late (one shard {:.4}, {:.4})",
        stretch(&one, 0..150),
        stretch(&one, 450..600)
    );
    assert!(early > 0.3, "it swings: {early:.3} m");
    assert!(
        (late / early - 1.0).abs() < 0.02,
        "the swing changed: {early:.4} -> {late:.4}"
    );
}

/// The demo area's own swinging pendulum and its chain, hung over the border
/// instead of under the gantry: each swings to and fro across `z = −4`.
///
/// Neither changes shard — the distance hysteresis is for exactly this — so
/// both are bit-identical to the same pendulum and chain on one shard, and
/// no transfer is offered at all.
///
/// Mutation this catches: no hysteresis (a handoff at every swing), or one
/// measured from the wrong side.
#[test]
fn a_pendulum_and_a_chain_swinging_over_the_border_stay_on_their_shard() {
    let (four, one, c) = on_four_and_on_one(|c| {
        // The scene has the pendulum's anchor at (14.8, ., 3.5) and the
        // chain's at (17.6, ., 3.5): moved to z = -4, the pendulum over the
        // lane at the room's west side and the chain to its east side —
        // either more than a second of its swing from the balls in the
        // bowl between them (what swings towards a body of another shard
        // closer than that is one island with it, rightly, and is brought
        // onto one shard with it).
        c.scene_body_shifted("swinging pendulum", [10.6 - 14.8, 0.0, BORDER_Z - 3.5]);
        c.scene_body_shifted("chain", [18.6 - 17.6, 0.0, BORDER_Z - 3.5]);
        let mut out = Tracks::new();
        for _ in 0..600 {
            c.tick();
            out.push(c.track());
        }
        out
    });
    for (i, name) in ["pendulum", "chain"].iter().enumerate() {
        let z: Vec<f64> = four.iter().map(|row| row[i][2] - BORDER_Z).collect();
        let (lo, hi) = z
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), v| (l.min(*v), h.max(*v)));
        let crossings = z
            .windows(2)
            .filter(|w| (w[0] < 0.0) != (w[1] < 0.0))
            .count();
        println!("{name}: swings {lo:.2}..{hi:.2} m about the border, {crossings} crossings");
        assert!(
            crossings >= 6,
            "the {name} swings across the border: {crossings}"
        );
        assert!(lo > -border::BODY_HYSTERESIS && hi < border::BODY_HYSTERESIS);
        assert_eq!(worst_apart(&four, &one, i).0, 0.0, "the {name}");
        assert_eq!(c.simulators(c.extra[i]), vec![ROOT], "the {name}");
    }
    assert_eq!(c.handed_off(), SCENE_HANDOFFS);
    c.assert_nothing_called_off();
}

// ------------------------------------------------------------------ players

/// How close (centre to centre, metres) the ship is to the nearest scene
/// body.
fn nearest_body(c: &Cluster, ship: EntityGuid) -> f64 {
    let at = c.rel(ship);
    c.names
        .iter()
        .map(|(_, g)| dist(at, c.rel(*g)))
        .fold(f64::INFINITY, f64::min)
}

/// A player who passes the showcase without coming within reach of anything
/// — flying along both borders five metres from the nearest body, across
/// each of them and back, then ten times over the whole of it at 120 m/s a
/// hundred metres up — changes shard a dozen times and takes **no body**
/// with it: every handoff of the run is the ship's own.
///
/// Mutation this catches: an island that reaches as far as a body's speed in
/// every direction, behind it and beside it too. At 120 m/s that is 120 m:
/// every pass a hundred metres overhead made the whole showcase one island
/// with the ship and sent it to the ship's shard and back — 73 body handoffs
/// a pass in this very run, 446 in six passes, where there are none now.
#[test]
fn a_player_passing_the_showcase_moves_no_body_it_does_not_reach() {
    let mut c = Cluster::four();
    let mut pilot = Pilot::join(&mut c, "passer-by");
    pilot.steps(&mut c, [0.0; 3], 60);
    let settled = c.handed_off();
    assert_eq!(settled, SCENE_HANDOFFS);
    let mut nearest = f64::INFINITY;
    let mut fly = |c: &mut Cluster, pilot: &mut Pilot, thrust: [f32; 3], ticks: usize| {
        for _ in 0..ticks {
            pilot.step(c, thrust);
            nearest = nearest.min(nearest_body(c, pilot.ship));
        }
    };
    // Along z = 12 (the zero-g corner's bodies end at z = 5), over x = -6
    // and back; then down x = -36 over z = -4 and back.
    pilot.put(&mut c, [4.0, 0.0, 12.0], [0.0; 3]);
    for (thrust, ticks) in [
        ([-1.0, 0.0, 0.0], 60),
        ([0.0; 3], 90),
        ([1.0, 0.0, 0.0], 120),
        ([0.0; 3], 90),
        ([-1.0, 0.0, 0.0], 60),
        ([0.0; 3], 60),
    ] {
        fly(&mut c, &mut pilot, thrust, ticks);
    }
    pilot.put(&mut c, [-36.0, 0.0, 8.0], [0.0; 3]);
    for (thrust, ticks) in [
        ([0.0, 0.0, -1.0], 60),
        ([0.0; 3], 90),
        ([0.0, 0.0, 1.0], 120),
        ([0.0; 3], 90),
        ([0.0, 0.0, -1.0], 60),
        ([0.0; 3], 60),
    ] {
        fly(&mut c, &mut pilot, thrust, ticks);
    }
    let walked = pilot.switches;
    println!("passer-by: {walked} switches on the slow legs, nearest body {nearest:.2} m");
    assert!(walked >= 4, "the slow legs crossed both borders: {walked}");
    assert!(nearest > 5.0, "the slow legs kept clear: {nearest:.1} m");
    assert_eq!(
        c.handed_off() - settled,
        walked,
        "only the ship changed shard"
    );
    // Ten passes at top speed, a hundred metres up, over the middle of it.
    for pass in 0..10 {
        let dir: f32 = if pass % 2 == 0 { -1.0 } else { 1.0 };
        let z = if pass % 4 < 2 { -2.0 } else { -6.0 };
        pilot.put(
            &mut c,
            [-400.0 * f64::from(dir), 100.0, z],
            [120.0 * dir, 0.0, 0.0],
        );
        pilot.steps(&mut c, [dir, 0.0, 0.0], 200);
    }
    pilot.steps(&mut c, [0.0; 3], 150);
    println!("passer-by: {} switches in all", pilot.switches);
    assert!(pilot.switches >= walked + 10, "every pass crossed x = -6");
    assert_eq!(
        c.handed_off() - settled,
        pilot.switches,
        "only the ship changed shard: {} handoffs for {} of the ship's",
        c.handed_off() - settled,
        pilot.switches
    );
    c.assert_nothing_called_off();
    for (name, guid) in &c.names {
        assert_eq!(c.simulators(*guid), vec![c.owner(c.rel(*guid))], "{name}");
    }
}

/// A body **ahead** of a player who is changing shard waits where it is:
/// the player will be on its shard within the second.
///
/// A player flies across `x = −6` at 6 m/s towards a crate that floats
/// eleven metres behind the border. The crate comes within the ship's reach
/// — a second ahead of it — when the ship is a metre and a half into the
/// crate's shard, and so already being handed to it. The crate is not sent
/// to the shard the ship is leaving: it never changes shard at all, the
/// ship arrives on its shard a third of a second before it strikes it, and
/// the whole thing — the flight, the blow, where the crate goes — is
/// **bit-identical** to the same on one shard.
///
/// Mutation this catches: the crate following the player back (it was
/// offered to the ship's old shard while the ship's own transfer went the
/// other way: sent there and back again around the blow — two handoffs more
/// and a collision solved across them — or, with the shard ids the other
/// way round, refused as a "crossing island transfer").
#[test]
fn a_body_ahead_of_a_player_who_is_changing_shard_waits_where_it_is() {
    let scenario = |c: &mut Cluster| {
        let mut pilot = Pilot::join(c, "flyer");
        pilot.steps(c, [0.0; 3], 30);
        c.spawn([BORDER_X - 11.0, 3.0, 10.0], [0.0; 3], crate_body(0.7));
        pilot.put(c, [0.0, 3.0, 10.0], [-6.0, 0.0, 0.0]);
        let mut out = Tracks::new();
        for _ in 0..150 {
            pilot.step(c, [0.0; 3]);
            out.push(vec![c.sample(c.extra[0]), c.sample(pilot.ship)]);
        }
        if c.shards.len() == 4 {
            assert_eq!(pilot.switches, 1, "the ship changed shard once");
            assert_eq!(pilot.on, WEST);
        }
        out
    };
    let (four, one, c) = on_four_and_on_one(scenario);
    let crate_moved = dist(four[0][0], four[four.len() - 1][0]);
    assert!(
        crate_moved > 1.0,
        "the ship struck the crate: {crate_moved:.2} m"
    );
    assert_eq!(
        c.handed_off(),
        SCENE_HANDOFFS + 1,
        "the ship's own handoff and no other"
    );
    c.assert_nothing_called_off();
    for (i, what) in ["crate", "ship"].iter().enumerate() {
        let (apart, at) = worst_apart(&four, &one, i);
        assert_eq!(
            apart, 0.0,
            "the {what} left the one-shard path at tick {at}"
        );
    }
    // The tick that matters is the one on which the ship's own shard offers
    // it: the crate's shard has not heard of that offer yet. So the same
    // flight with the crate at every distance that puts "within reach" on a
    // tick around that one: never a handoff but the ship's.
    // (Nearer than 9.7 m the crate is within reach before the ship is a
    // metre across, and goes to the ship's shard and back with it: that is
    // "bodies follow the player". Without the rule the crates at 9.7 m and
    // 9.8 m did too.)
    for step in 0..8 {
        let behind = 9.7 + 0.1 * f64::from(step);
        let mut c = Cluster::four();
        let mut pilot = Pilot::join(&mut c, "flyer");
        pilot.steps(&mut c, [0.0; 3], 30);
        c.spawn([BORDER_X - behind, 3.0, 10.0], [0.0; 3], crate_body(0.7));
        pilot.put(&mut c, [0.0, 3.0, 10.0], [-6.0, 0.0, 0.0]);
        pilot.steps(&mut c, [0.0; 3], 75);
        assert_eq!(
            c.handed_off(),
            SCENE_HANDOFFS + pilot.switches,
            "a crate {behind:.1} m behind the border changed shard"
        );
        c.assert_nothing_called_off();
    }
}

/// The crate the demo area floats on the border, struck by a player who
/// flies into it: is it "knocked up and forward rather than pushed", as a
/// look at the live stack had it — and is that the border?
///
/// It is knocked up and forward, and it is not the border: the same flight
/// on one shard sends the crate off the same way, more than half as fast
/// upwards as forwards. The crate floats at head height (its underside
/// 25 cm above a player's eye line) and a ship is a capsule: what meets the
/// crate is the rounded top of the capsule, and with no friction on a ship
/// the push is along the contact's normal — up and forward.
///
/// Against one shard, by when the ship changes shard on its way:
///
/// * from 5.7 m away it has changed shard before it is within a second of
///   the crate, and the collision is the neighbour's own: the crate leaves
///   on the one-shard path to the bit;
/// * from 3.2 m away the ship strikes the crate **on the very ticks it is
///   handed over** with it (the crate was brought to the ship's shard on
///   the way, and the two go on together): the collision is solved by two
///   solvers, one after the other, and comes out 2 % different — the crate
///   is 19 cm from the one-shard path five seconds and ten metres on.
///
/// Mutation this catches: a body struck across a border by a ship that the
/// other shard holds as a ghost (the crate is then pushed by something of
/// endless mass and leaves a sixth faster than it should — a metre off
/// after these five seconds).
#[test]
fn the_floating_crate_on_the_border_is_struck_the_same_on_one_shard() {
    for (away, bound) in [(5.7, 0.0), (3.2, 0.25)] {
        let scenario = |c: &mut Cluster| {
            let mut pilot = Pilot::join(c, "striker");
            pilot.steps(c, [0.0; 3], 30);
            let on_border = c
                .named("floating crate")
                .into_iter()
                .find(|g| c.rel(*g)[2] == BORDER_Z)
                .expect("the demo area floats a crate on z = -4");
            // A quarter of a metre above the ship's own eye line, at 3 m/s.
            let x = c.rel(on_border)[0] + away;
            pilot.put(c, [x, 0.25, BORDER_Z], [-3.0, 0.0, 0.0]);
            let mut out = Tracks::new();
            for _ in 0..180 {
                pilot.step(c, [0.0; 3]);
                out.push(vec![c.sample(on_border), c.sample(pilot.ship)]);
            }
            out
        };
        let (four, one, c) = on_four_and_on_one(scenario);
        c.assert_nothing_called_off();
        let (apart, at) = worst_apart(&four, &one, 0);
        let moved = |t: &Tracks| {
            let (a, b) = (t[0][0], t[t.len() - 1][0]);
            [b[0] - a[0], b[1] - a[1], b[2] - a[2]]
        };
        let (m4, m1) = (moved(&four), moved(&one));
        println!(
            "struck crate from {away} m: moved {m4:?} on four, {m1:?} on one; {apart:.6} m apart at tick {at}"
        );
        assert!(m1[0] < -1.0, "struck and sent on: {m1:?}");
        assert!(
            m1[1] > 0.5 * m1[0].abs(),
            "on one shard too it goes up as well as forward: {m1:?}"
        );
        assert!(
            apart <= bound,
            "from {away} m: the crate is {apart:.4} m from one shard at tick {at}"
        );
    }
}

/// One script — bodies dropped and pushed, a player walking through the
/// gravity room across `z = −4`, pushing a crate on and walking back — on
/// one shard and on four, compared body by body at every tick.
///
/// * A crate dropped onto the border from two metres and one pushed across
///   `x = −6` weightless: **bit-identical**.
/// * The player, handed over twice while it walks (standing on the floor:
///   in contact): within 3 cm at every tick of the walk.
/// * The crate the player walks into and pushes 2.7 m on, handed over with
///   the player: within 4 cm.
/// * The scene's bodies: the crate stack beside the player's path — a metre
///   and a half off, within reach by the island rule, so it is brought to
///   the player's shard and back — settles within 1.5 cm of where one shard
///   has it; the bowl's balls and the planetoid's crates as at rest (2 cm,
///   0.1 mm); every other body bit-identical.
///
/// Mutation this catches: anything that makes a border felt by what crosses
/// it or is pushed across it — a lost tick, a lost input, a body left
/// behind as a ghost while it is pushed (the ship then meets something that
/// does not give: the two runs part by decimetres).
#[test]
fn the_same_script_on_one_shard_and_on_four() {
    let walk_x = LANE_X - 0.3;
    let scenario = |c: &mut Cluster| {
        let mut pilot = Pilot::join(c, "walker");
        pilot.steps(c, [0.0; 3], 30);
        // Dropped onto the border from two metres, east of the bowl.
        c.spawn([18.3, FLOOR + 2.3, BORDER_Z], [0.0; 3], crate_body(0.5));
        // Pushed across x = -6, weightless.
        c.spawn([-2.0, 4.0, 14.0], [-1.5, 0.0, 0.0], crate_body(0.7));
        // In the player's way, a metre and a half behind the border.
        c.spawn(
            [walk_x, resting(0.6), BORDER_Z - 1.5],
            [0.0; 3],
            crate_body(0.6),
        );
        pilot.put(c, [walk_x, FLOOR + 1.7, BORDER_Z + 1.5], [0.0; 3]);
        let mut out = Tracks::new();
        for (thrust, ticks) in [
            ([0.0; 3], 45),
            // Walk -z: over the border, into the crate, pushing it on. (A
            // ship that stands on something stops by itself when its
            // player lets go.)
            ([0.0, 0.0, -1.0], 40),
            ([0.0; 3], 110),
            // And back over the border.
            ([0.0, 0.0, 1.0], 40),
            ([0.0; 3], 150),
        ] {
            for _ in 0..ticks {
                pilot.step(c, thrust);
                let mut row = c.track();
                row.insert(0, c.sample(pilot.ship));
                out.push(row);
            }
        }
        if c.shards.len() == 4 {
            assert_eq!(pilot.switches, 2, "over the border and back");
            assert!(
                c.rel(c.extra[2])[2] < BORDER_Z - 3.0,
                "the crate was pushed on"
            );
        }
        out
    };
    let (four, one, c) = on_four_and_on_one(scenario);
    c.assert_nothing_called_off();
    let (ship, at) = worst_apart(&four, &one, 0);
    println!("script: the ship {ship:.6} m from one shard at tick {at}");
    assert!(ship < 0.03, "the ship: {ship:.4} m at tick {at}");
    for (i, (what, bound)) in [
        ("the dropped crate", 0.0),
        ("the weightless crate", 0.0),
        ("the pushed crate", 0.04),
    ]
    .iter()
    .enumerate()
    {
        let (apart, at) = worst_apart(&four, &one, i + 1);
        println!("script: {what} {apart:.6} m from one shard at tick {at}");
        assert!(apart <= *bound, "{what}: {apart:.5} m at tick {at}");
    }
    for (i, (name, _)) in c.names.iter().enumerate() {
        let (apart, at) = worst_apart(&four, &one, i + 4);
        let bound = match *name {
            "bowl ball" => 0.02,
            // Within a metre and a half of the player's path: brought to
            // the player's shard and back, and settled anew each time.
            "stack crate" => 0.015,
            "planetoid crate" => 1.0e-4,
            _ => 0.0,
        };
        assert!(
            apart <= bound,
            "{name}: {apart:.6} m from the one-shard run at tick {at} (bound {bound})"
        );
    }
}

/// **A shard whose region is merged away drains** (`docs/plan-sharding.md`
/// §3): the map changes under it, it owns nothing, and everything it
/// simulates — the scene's bodies on what was its ground, a player standing
/// there — is handed to the shard that owns the ground now by the ordinary
/// handoff, until it holds nothing and reports so. The others still hear it
/// although their map no longer names it.
///
/// Mutations this catches: `Shard::knows_shard` naming only the shards of
/// the map (the offers of the drained shard are dropped: after ten seconds it
/// still holds every body and its player); `peer_endpoint` without the former
/// shard's address is the same over real sockets (not seen by this fixture,
/// which delivers by id).
#[test]
fn a_shard_whose_region_is_merged_away_hands_everything_over_and_is_empty() {
    let mut c = Cluster::four();
    c.tick_n(90);
    // A player walks onto the diagonal cell's ground and stands there.
    let mut pilot = Pilot::join(&mut c, "drained");
    let stand = [BORDER_X - 9.0, 3.0, BORDER_Z - 9.0];
    assert_eq!(c.owner(stand), DIAGONAL);
    pilot.put(&mut c, stand, [0.0; 3]);
    for _ in 0..240 {
        pilot.step(&mut c, [0.0; 3]);
        if pilot.on == DIAGONAL {
            break;
        }
    }
    assert_eq!(pilot.on, DIAGONAL, "the player is on the diagonal shard");
    let own = |c: &Cluster| {
        c.shard(DIAGONAL)
            .body_list()
            .iter()
            .filter(|body| !body.ghost)
            .count()
    };
    let before = c.shard_mut(DIAGONAL).load_report();
    assert_eq!(before.players, 1);
    assert!(
        before.entities >= 2,
        "the ship and scene bodies: {before:?}"
    );
    assert_eq!(before.entities as usize, own(&c));
    assert_eq!(
        (before.hot_cell, before.hot_cell_players),
        (c.at(stand).cell, 1)
    );
    let bodies: Vec<EntityGuid> = c.names.iter().map(|(_, g)| *g).collect();
    let where_before: Vec<[f64; 3]> = bodies.iter().map(|g| c.rel(*g)).collect();
    let resting: Vec<bool> = bodies
        .iter()
        .map(|g| c.state(*g).is_some_and(|s| s.vel == [0.0; 3]))
        .collect();

    // The merge commits: the diagonal cell goes back to the root's shard.
    let mut merged = c.shard(ROOT).shard_map().clone();
    merged.regions.retain(|region| region.shard != DIAGONAL);
    merged.epoch += 1;
    for shard in &mut c.shards {
        shard.apply_epoch(merged.clone());
    }
    assert_eq!(c.owner(stand), ROOT);
    let aborted_before: usize = c.shards.iter().map(|s| s.stats().aborted).sum();

    let mut drained_after = None;
    for tick in 0..300 {
        pilot.step(&mut c, [0.0; 3]);
        if own(&c) == 0 && c.shard(DIAGONAL).stats().clients == 0 {
            drained_after = Some(tick);
            break;
        }
    }
    let ticks = drained_after.unwrap_or_else(|| {
        panic!(
            "the diagonal shard still holds {} entities and {} client(s) ten seconds after its \
             region was merged away",
            own(&c),
            c.shard(DIAGONAL).stats().clients
        )
    });
    println!("drained {} entities in {ticks} ticks", before.entities);
    let after = c.shard_mut(DIAGONAL).load_report();
    assert_eq!((after.players, after.entities), (0, 0), "{after:?}");
    assert_eq!(pilot.on, ROOT, "the player went to the owner of the ground");
    let aborted_after: usize = c.shards.iter().map(|s| s.stats().aborted).sum();
    assert_eq!(aborted_after, aborted_before, "no transfer was called off");
    // Every body is simulated once, by a shard of the map, and none of the
    // ones at rest moved by the change of shard.
    c.tick_n(30);
    for ((guid, was), rests) in bodies.iter().zip(&where_before).zip(&resting) {
        let on = c.simulators(*guid);
        assert_eq!(on.len(), 1, "{guid} is simulated by {on:?}");
        assert_ne!(on[0], DIAGONAL);
        if *rests {
            let moved = dist(c.rel(*guid), *was);
            assert!(
                moved < 0.01,
                "{guid} moved {moved:.4} m while it was handed over"
            );
        }
    }
}

/// An orchestrated shard reports its load once a second: who it holds and
/// where most of its players are.
///
/// Mutation this catches: `Shard::tick` not queueing the report (the
/// orchestrator would scale on nothing), or queueing one every tick.
#[test]
fn a_shard_reports_its_load_once_a_second() {
    let mut c = Cluster::one();
    let pilot = Pilot::join(&mut c, "counted");
    c.shard_mut(ROOT).report_load();
    let mut reports = Vec::new();
    for _ in 0..95 {
        c.shard_mut(ROOT).tick();
        for out in c.shard_mut(ROOT).take_outbox() {
            if let (Destination::Orchestrator, Message::LoadReport(report)) =
                (out.dest, out.message)
            {
                reports.push(report);
            }
        }
    }
    assert_eq!(reports.len(), 3, "one a second for three seconds");
    let report = reports[2];
    assert_eq!(report.shard, ROOT);
    assert_eq!(report.players, 1);
    let ship = c
        .shard(ROOT)
        .sim()
        .ship_state(pilot.ship)
        .expect("the ship");
    assert_eq!(
        (report.hot_cell, report.hot_cell_players),
        (ship.pos.cell, 1)
    );
    let scene = space_scene::body_defs(Scene::DemoArea).len();
    assert_eq!(
        report.entities as usize,
        scene + 1,
        "the scene and the ship"
    );
    assert!(report.tick_mean_ms > 0.0 && report.tick_mean_ms <= report.tick_max_ms);
    assert!(report.tick_p95_ms <= report.tick_max_ms);
}
