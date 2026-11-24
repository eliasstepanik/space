//! Physics across shards: bodies handed off at borders, ghosts, island
//! consolidation, springs across a border, scene bodies cluster-wide, a split
//! handing its octant's bodies to the child, the same far from the origin,
//! bandwidth, and the protocol bump — see `src/border.rs` for the rules and
//! `src/handoff.rs` for the guarantee table these tests check for bodies.
//!
//! Most tests run real [`Shard`]s side by side in one process and move their
//! outboxes by hand (the `tests/acceptance.rs` pattern): deterministic, no
//! wall clock, and a single-world [`SimWorld`] can run the same physics as a
//! reference. The last tests run real shards over loopback QUIC, paced at the
//! real tick rate, with a raw client watching.
//!
//! Where a test cuts the demo area, the borders are where a scene has them:
//! a root of more than one cell puts the scene 6 m, 512 m up and 4 m from the
//! corner where its cells meet (`space_scene::ANCHOR_OFFSET`), so the planes
//! through the scene are `x = −6` and `z = −4` ([`BORDER_X`], [`BORDER_Z`])
//! and none is horizontal.
//!
//! | test | what |
//! |---|---|
//! | [`a_body_flying_across_a_border_is_seen_without_a_gap_or_a_jump`] | handoff of an unowned body tracked by clients on both sides; ghosts appear and disappear at the margin |
//! | [`an_unowned_body_handoff_keeps_h1_to_h8`] | H1, H2, H4 (bit-exact against one world), H3 for bodies |
//! | [`an_unanswered_body_offer_times_out_and_nothing_moved`] | H6, H8 |
//! | [`a_refused_body_offer_leaves_the_body_where_it_was`] | H6 (refusal), the scene-revision check |
//! | [`an_epoch_change_aborts_a_body_transfer`] | H7 |
//! | [`a_player_pushes_a_crate_across_the_border_and_keeps_pushing`] | island consolidation with a ship, redirect, group commit |
//! | [`bodies_colliding_head_on_at_a_border_match_one_shard`] | consolidation before contact; momentum against one world |
//! | [`a_spring_across_a_border_keeps_its_period`] | springs through ghosts |
//! | [`a_scene_change_replaces_the_bodies_cluster_wide`] | scene bodies spawned once per cluster, replaced on change |
//! | [`a_crate_falls_through_the_gravity_room_across_a_border`] | gravity, a falling crate crossing a border, a player standing there |
//! | [`a_split_hands_the_carved_octants_bodies_to_the_child`] | the orchestrator's split falls out of the handoff rule |
//! | [`joining_players_get_distinct_free_spawn_points_and_ships_are_solid`] | spawn search, ship-ship contact |
//! | [`a_busy_spawn_point_never_puts_a_ship_across_a_border`] | the spawn grid stays on its shard's ground when the scene is cut into four shard areas |
//! | [`replication_bandwidth_of_the_moving_showcase`] | bytes per tick for a client in the demo area, and for the chain |
//! | [`over_quic_a_body_crosses_between_two_real_shards`] | the same handoff through the real front end, paced at 30 Hz |
//! | [`an_old_peer_is_refused_with_both_versions`] | protocol 7 |

use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use space_protocol::{
    EntityGuid, EntitySnapshot, Hello, InputTick, Message, NeighbourStatus, PlayerInput, Scene,
    SceneState, ShardId, ShardMap, ShardRegion, PROTOCOL_VERSION,
};
use space_server::border;
use space_server::shard::{Destination, Shard};
use space_server::ShardConfig;
use space_sim::body::{Body, MassSpec};
use space_sim::shape::ShapeRef;
use space_sim::spring::{Spring, SpringMode, SpringTarget, Springs};
use space_sim::{BodySpawn, ShipState, SimWorld};
use space_world::{NodeRef, WorldPos};

// ------------------------------------------------------------------ harness

fn addr() -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)
}

/// Where shard `id` of a hand-wired [`Cluster`] would listen (nothing does:
/// the shards are driven in process).
fn listening(id: u32) -> SocketAddr {
    SocketAddr::new(
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        7_100 + u16::try_from(id).expect("a small shard id"),
    )
}

/// Shards in one process, sharing one map, wired together by hand.
struct Cluster {
    shards: Vec<Shard>,
    /// Frames for clients: `(shard, session, message)`.
    frames: Vec<(ShardId, u64, Message)>,
    /// Drop every shard-to-shard message (a dead network).
    cut: bool,
    /// Encoded bytes of the ghost frames delivered between shards.
    ghost_bytes: usize,
}

impl Cluster {
    /// One shard per `(id, region)`, all on the same epoch-1 map.
    fn new(regions: &[(u32, NodeRef)]) -> Self {
        let map = ShardMap::new(
            1,
            regions
                .iter()
                // (An address a shard could be dialed at: a map naming port 0
                // for a shard says "not known yet", and nobody is sent there.)
                .map(|(id, node)| ShardRegion::new(*node, ShardId::new(*id), listening(*id)))
                .collect(),
        );
        let shards = regions
            .iter()
            .map(|(id, node)| {
                let mut s = Shard::new(ShardConfig::new(ShardId::new(*id), addr(), *node));
                s.set_shard_map(map.clone());
                s
            })
            .collect();
        Self {
            shards,
            frames: Vec::new(),
            cut: false,
            ghost_bytes: 0,
        }
    }

    fn shard(&self, id: u32) -> &Shard {
        self.shards
            .iter()
            .find(|s| s.shard() == ShardId::new(id))
            .expect("a shard of the cluster")
    }

    fn shard_mut(&mut self, id: u32) -> &mut Shard {
        self.shards
            .iter_mut()
            .find(|s| s.shard() == ShardId::new(id))
            .expect("a shard of the cluster")
    }

    /// Deliver one round of shard messages; collect client frames.
    fn deliver(&mut self) {
        let mut pending: Vec<(ShardId, ShardId, Message)> = Vec::new();
        for shard in &mut self.shards {
            let from = shard.shard();
            for out in shard.take_outbox() {
                match out.dest {
                    Destination::Client(session) => self.frames.push((from, session, out.message)),
                    Destination::Shard(to) => pending.push((from, to, out.message)),
                    Destination::Orchestrator => {}
                }
            }
        }
        if self.cut {
            return;
        }
        for (from, to, message) in pending {
            if matches!(message, Message::GhostUpdate(_)) {
                self.ghost_bytes += space_protocol::encode_frame(&message).len();
            }
            if let Some(target) = self.shards.iter_mut().find(|s| s.shard() == to) {
                target.peer_message(from, message);
            }
        }
    }

    fn tick(&mut self) {
        for shard in &mut self.shards {
            shard.tick();
        }
        self.deliver();
    }

    fn tick_n(&mut self, n: usize) {
        for _ in 0..n {
            self.tick();
        }
    }

    /// The shards simulating `guid` (not as a ghost).
    fn simulators(&self, guid: EntityGuid) -> Vec<ShardId> {
        self.shards
            .iter()
            .filter(|s| s.sim().ship_state(guid).is_some())
            .map(Shard::shard)
            .collect()
    }

    /// `guid`'s state where it is simulated.
    fn state(&self, guid: EntityGuid) -> Option<ShipState> {
        self.shards.iter().find_map(|s| s.sim().ship_state(guid))
    }

    /// Frames sent to `(shard, session)`, draining them.
    fn frames_for(&mut self, shard: u32, session: u64) -> Vec<Message> {
        let id = ShardId::new(shard);
        let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut self.frames)
            .into_iter()
            .partition(|(s, c, _)| *s == id && *c == session);
        self.frames = rest;
        mine.into_iter().map(|(_, _, m)| m).collect()
    }
}

/// A client's view of one entity across frames (what a real client keeps).
#[derive(Default)]
struct View {
    /// `(tick, position)` of every frame that carried it.
    seen: Vec<(u64, WorldPos)>,
    removed_at: Vec<u64>,
    /// Ticks of every frame received.
    ticks: Vec<u64>,
    /// Components as the client holds them.
    components: Vec<(u16, Vec<u8>)>,
}

impl View {
    fn apply(&mut self, guid: EntityGuid, message: &Message) {
        let (tick, entities, removed): (u64, &[EntitySnapshot], &[EntityGuid]) = match message {
            Message::Snapshot(s) => (s.tick, &s.entities, &[]),
            Message::Delta(d) => (d.tick, &d.entities, &d.removed),
            _ => return,
        };
        self.ticks.push(tick);
        for e in entities.iter().filter(|e| e.guid == guid) {
            for (id, bytes) in &e.components {
                self.components.retain(|(i, _)| i != id);
                if !bytes.is_empty() {
                    self.components.push((*id, bytes.clone()));
                }
            }
            let state = e
                .components
                .iter()
                .find(|(id, _)| *id == 0)
                .map(|(_, b)| space_sim::wire::decode_ship_state(b).expect("a ship state"))
                .expect("every entry carries the ShipState");
            self.seen.push((tick, state.pos));
        }
        if removed.contains(&guid) {
            self.removed_at.push(tick);
        }
    }

    /// The largest per-tick displacement between two consecutive sightings.
    fn largest_step(&self) -> f64 {
        self.seen
            .windows(2)
            .map(|w| WorldPos::distance(w[0].1, w[1].1) / (w[1].0 - w[0].0).max(1) as f64)
            .fold(0.0, f64::max)
    }

    fn has(&self, id: u16) -> bool {
        self.components.iter().any(|(i, _)| *i == id)
    }
}

fn join(shard: &mut Shard, token: &str) -> (u64, EntityGuid) {
    let welcome = shard
        .join(&Hello {
            protocol: PROTOCOL_VERSION,
            player_token: token.into(),
        })
        .expect("join");
    let ship = shard.ship_of(welcome.session).expect("a ship");
    (welcome.session, ship)
}

/// Put `token` on shard `id` as a player who is **already there**.
///
/// A new player starts on the scene's spawn point, so only the shard that
/// owns it takes a fresh join; every other shard answers with a
/// `JoinRedirect` (`a_fresh_join_is_taken_only_by_the_shard_that_owns_the_spawn_point`
/// in `acceptance.rs`). In a running cluster a player gets onto such a shard
/// by flying there. A test that needs somebody on it from its first tick — a
/// watcher, a player it then places by hand — lets that player in while the
/// shard runs the map of its own regions alone, in which the spawn point is
/// its own, and gives it the cluster's map back before anything ticks.
fn resident(c: &mut Cluster, id: u32, token: &str) -> (u64, EntityGuid) {
    let shard = c.shard_mut(id);
    let cluster_map = shard.shard_map().clone();
    let own: Vec<ShardRegion> = cluster_map
        .regions
        .iter()
        .filter(|r| r.shard == shard.shard())
        .cloned()
        .collect();
    shard.set_shard_map(ShardMap::new(cluster_map.epoch, own));
    let joined = join(shard, token);
    shard.set_shard_map(cluster_map);
    joined
}

/// A ball of `radius`, `mass` kg, no damping, no gravity, a lively bounce.
fn ball(radius: f32, mass: f32) -> Body {
    let mut b = Body::dynamic(ShapeRef::Ball { radius });
    b.mass = MassSpec::Mass(mass);
    b.linear_damping = 0.0;
    b.angular_damping = 0.0;
    b.restitution = 0.5;
    b.friction = 0.0;
    b
}

fn moving(pos: WorldPos, vel: [f32; 3]) -> ShipState {
    ShipState {
        vel,
        ..ShipState::at_rest(pos)
    }
}

/// The border between cell `x0` (shard 1) and cell `x0 + 1` (shard 2),
/// offset by `dx` metres along x (and `y`, `z`).
fn at_border(x0: i64, d: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(WorldPos::new([x0 + 1, 0, 0], [-512.0, 0.0, 0.0]), d)
}

/// Two side-by-side one-cell shards at cell `x0` (1) and `x0 + 1` (2).
fn side_by_side(x0: i64) -> Cluster {
    Cluster::new(&[
        (1, NodeRef::new(64, [x0, 0, 0])),
        (2, NodeRef::new(64, [x0 + 1, 0, 0])),
    ])
}

/// x of `p` relative to the border at `x0`.
fn border_x(x0: i64, p: WorldPos) -> f64 {
    WorldPos::delta(at_border(x0, [0.0; 3]), p)[0]
}

/// 1e13 m from the origin, in cells.
const FAR: i64 = 9_765_625_000;

// ------------------------------------------------------- a body crossing

/// A ball flies across the border from shard 1 into shard 2. A client on
/// each side watches it the whole time: on shard 1 it is first a body, then
/// (after the handoff) a ghost of shard 2's body; on shard 2 a ghost, then a
/// body. Neither client is ever told it went away, no frame shows it jump,
/// and each side drops its ghost once the ball is past the ghost margin —
/// shard 1's client then goes on seeing it as a **far replica** (it used to
/// be told the ball had left: what another shard simulates beyond its
/// ghosts' reach was invisible).
///
/// (The watcher on shard 1 stands 20 m from the border, beside the ball's
/// path: within 100 m a moving thing is sent every tick, further off less
/// often — `space_protocol::far` — and this test is about every tick.)
fn crossing(x0: i64) {
    let mut c = side_by_side(x0);
    let (session_a, watcher_a) = join(c.shard_mut(1), "watcher-a");
    assert!(c.shard_mut(1).set_state(
        watcher_a,
        ShipState::at_rest(at_border(x0, [-20.0, 40.0, 10.0]))
    ));
    // (Shard 1 owns the spawn point; shard 2's watcher is there already.)
    let (session_b, _) = resident(&mut c, 2, "watcher-b");
    let speed = 8.0f32;
    let start = at_border(x0, [-30.0, 40.0, 0.0]);
    let guid = c
        .shard_mut(1)
        .spawn_body(
            BodySpawn::new(moving(start, [speed, 0.0, 0.0]), ball(0.4, 10.0))
                .with_visual(space_sim::Visual(space_scene::visual::BALL)),
        )
        .expect("spawn");
    let mut a = View::default();
    let mut b = View::default();
    let mut ghost_on_b_from: Option<f64> = None;
    let mut ghost_on_a_until: Option<f64> = None;
    let mut handed_at: Option<f64> = None;
    for _ in 0..600 {
        c.tick();
        for m in c.frames_for(1, session_a) {
            a.apply(guid, &m);
        }
        for m in c.frames_for(2, session_b) {
            b.apply(guid, &m);
        }
        let sims = c.simulators(guid);
        assert!(sims.len() <= 1, "two simulators: {sims:?}");
        let x = c.state(guid).map(|s| border_x(x0, s.pos));
        if let Some(x) = x {
            if ghost_on_b_from.is_none() && c.shard(2).ghost_owner(guid) == Some(ShardId::new(1)) {
                ghost_on_b_from = Some(x);
            }
            if handed_at.is_none() && sims == vec![ShardId::new(2)] {
                handed_at = Some(x);
            }
            if handed_at.is_some()
                && ghost_on_a_until.is_none()
                && c.shard(1).ghost_owner(guid).is_none()
            {
                ghost_on_a_until = Some(x);
            }
            if x > 130.0 {
                break;
            }
        }
    }
    // The handoff: one simulator at a time, shard 2 at the end, 2 m in.
    assert_eq!(c.simulators(guid), vec![ShardId::new(2)]);
    // Offered once 2 m in (the hysteresis), taken over a lead later.
    let handed_at = handed_at.expect("handed off");
    let per_tick = f64::from(speed) * f64::from(space_sim::dt());
    let expected = border::BODY_HYSTERESIS + border::BODY_HANDOFF_LEAD as f64 * per_tick;
    assert!(
        (expected..expected + 3.0 * per_tick).contains(&handed_at),
        "taken over {handed_at:.2} m into shard 2 (offered at {} m, {} ticks of lead)",
        border::BODY_HYSTERESIS,
        border::BODY_HANDOFF_LEAD
    );
    // Ghosts: on shard 2 from the start (30 m < the 64 m + 24 m margin), on
    // shard 1 until the ball was past the margin.
    assert!(ghost_on_b_from.expect("a ghost on shard 2") < -29.0);
    let reach = border::ghost_reach(f64::from(speed));
    let left_at = ghost_on_a_until.expect("shard 1 dropped its ghost");
    assert!(
        (reach - 1.0..reach + 1.5).contains(&left_at),
        "shard 1 dropped the ghost at {left_at:.1} m, the margin is {reach:.1} m"
    );
    // What the clients saw: never removed while near, never a jump.
    for (name, view) in [("shard 1", &a), ("shard 2", &b)] {
        assert!(
            view.largest_step() < per_tick * 1.05,
            "{name}'s client saw the ball jump {:.3} m in a tick ({per_tick:.3} m of travel)",
            view.largest_step()
        );
        assert!(
            view.has(4) && view.has(5),
            "{name}'s client holds the Body and Visual"
        );
    }
    assert!(b.removed_at.is_empty(), "shard 2's client never lost it");
    assert!(
        a.removed_at.is_empty(),
        "shard 1's client never lost it either: past the ghost margin it is a far replica \
         (removed at {:?})",
        a.removed_at
    );
    // Shard 1's client saw the ball every tick while it was within 100 m of
    // the watcher (to 78 m past the border), and at the next step's rate
    // — every third tick — after that: through the handoff, and through the
    // change from ghost to far replica.
    for w in a.seen.windows(2) {
        let gap = w[1].0 - w[0].0;
        let x = border_x(x0, w[1].1);
        let most = if x < 70.0 { 1 } else { 4 };
        assert!(
            gap <= most,
            "shard 1's client saw nothing of the moving ball for {gap} ticks, at {x:.1} m"
        );
    }
    let last_x = border_x(x0, a.seen.last().expect("seen on shard 1").1);
    assert!(last_x > 120.0, "seen to the end of the run: {last_x:.1} m");
    assert_eq!(
        c.shard(1)
            .far_list()
            .iter()
            .filter(|far| far.guid == guid && far.from == ShardId::new(2))
            .count(),
        1,
        "shard 1 holds shard 2's ball as a far replica"
    );
    let origin = a
        .components
        .iter()
        .find(|(id, _)| *id == space_protocol::ORIGIN_COMPONENT)
        .and_then(|(_, bytes)| space_protocol::Origin::decode(bytes))
        .expect("the client is told what kind of copy it holds");
    assert_eq!(origin.kind, space_protocol::OriginKind::Far);
}

#[test]
fn a_body_flying_across_a_border_is_seen_without_a_gap_or_a_jump() {
    crossing(0);
}

#[test]
fn a_body_crosses_a_border_1e13_metres_from_the_origin() {
    crossing(FAR);
}

// ------------------------------------------------------------- H1..H8

/// H1 (same guid), H2 (exactly one shard steps it each tick) and H4 (the
/// state it continues with is the state at activation), checked the hard
/// way: the ball's trajectory through two shards is **bit-identical** to the
/// same ball in one world, tick for tick. A double step or a skipped tick
/// would be off by a tick of travel; a stale state by the lead.
#[test]
fn an_unowned_body_handoff_keeps_h1_to_h8() {
    for x0 in [0, FAR] {
        let mut c = side_by_side(x0);
        let start = moving(at_border(x0, [-5.0, 0.0, 0.0]), [6.0, 0.5, -0.25]);
        let mut body = ball(0.3, 4.0);
        body.angular_damping = 0.1;
        let spawn = BodySpawn::new(
            ShipState {
                ang_vel: [0.0, 1.5, 0.0],
                ..start
            },
            body,
        );
        let guid = c.shard_mut(1).spawn_body(spawn.clone()).expect("spawn");
        let mut reference = SimWorld::new(ShardId::new(1));
        reference.spawn_body(guid, spawn).expect("reference");
        let mut handoffs = 0;
        let mut last = vec![ShardId::new(1)];
        for tick in 0..240 {
            c.tick();
            reference.step();
            let sims = c.simulators(guid);
            assert!(sims.len() <= 1, "tick {tick}: two simulators {sims:?}");
            if let Some(state) = c.state(guid) {
                assert_eq!(
                    space_sim::wire::encode_ship_state(&state),
                    space_sim::wire::encode_ship_state(&reference.state_of(guid).expect("ref")),
                    "tick {tick}: the ball left the reference trajectory"
                );
                if sims != last {
                    handoffs += 1;
                    last = sims;
                }
            }
        }
        assert_eq!(handoffs, 1, "one handoff, at the border");
        assert_eq!(c.simulators(guid), vec![ShardId::new(2)]);
        // H1: the guid shard 1 minted, unchanged.
        assert_eq!(guid.shard, ShardId::new(1));
        assert_eq!(c.shard(1).stats().handed_off, 1);
        assert_eq!(c.shard(2).stats().received, 1);
        assert_eq!(c.shard(1).stats().aborted + c.shard(2).stats().aborted, 0);
    }
}

/// H6 + H8: shard 2 never answers. The offer times out, shard 1 keeps the
/// ball (and keeps simulating it), shard 2 holds nothing, and once the
/// network is back the handoff happens.
#[test]
fn an_unanswered_body_offer_times_out_and_nothing_moved() {
    let mut c = side_by_side(0);
    let guid = c
        .shard_mut(1)
        .spawn_body(BodySpawn::new(
            moving(at_border(0, [-1.0, 0.0, 0.0]), [1.0, 0.0, 0.0]),
            ball(0.3, 4.0),
        ))
        .expect("spawn");
    c.tick_n(60);
    c.cut = true;
    // Past the border and the hysteresis: shard 1 offers, nobody answers.
    let mut offered = false;
    let budget = space_server::handoff::HANDOFF_TIMEOUT_TICKS + border::GHOST_SILENCE_TICKS + 60;
    for _ in 0..budget {
        c.tick();
        offered |= c.shard(1).stats().aborted > 0;
    }
    assert_eq!(
        c.simulators(guid),
        vec![ShardId::new(1)],
        "shard 1 still has it"
    );
    assert!(offered, "the unanswered offer was given up (H8)");
    assert_eq!(c.shard(2).stats().received, 0);
    assert!(
        c.shard(2).sim().entity_of(guid).is_none(),
        "nothing staged, and the ghost of a silent neighbour is gone"
    );
    let x = border_x(0, c.state(guid).expect("simulated").pos);
    assert!(x > 3.0, "it kept flying on shard 1: {x}");
    // The network comes back: the next offer goes through.
    c.cut = false;
    c.tick_n(space_server::config::REOFFER_BACKOFF_TICKS as usize + 30);
    assert_eq!(c.simulators(guid), vec![ShardId::new(2)]);
}

/// H6 (refusal): the receiver checks that an offered body imports before it
/// accepts. Shard 1 runs the demo area (its shape catalogue has the concave
/// bowl), shard 2 runs nothing — so a bowl-shaped body cannot be built
/// there: shard 2 refuses, shard 1 keeps simulating it, unchanged, and does
/// not ask again at tick rate.
#[test]
fn a_refused_body_offer_leaves_the_body_where_it_was() {
    let mut c = side_by_side(0);
    assert!(c
        .shard_mut(1)
        .set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
    let mut bowl = Body::dynamic(ShapeRef::Asset(space_scene::BOWL));
    bowl.mass = MassSpec::Mass(50.0);
    let start = moving(at_border(0, [-1.0, 0.0, 0.0]), [1.5, 0.0, 0.0]);
    let guid = c
        .shard_mut(1)
        .spawn_body(BodySpawn::new(start, bowl.clone()))
        .expect("spawn");
    let mut reference = SimWorld::new(ShardId::new(1));
    reference.set_catalog(space_scene::catalog());
    reference
        .spawn_body(guid, BodySpawn::new(start, bowl))
        .expect("ref");
    for _ in 0..150 {
        c.tick();
        reference.step();
    }
    // Offered once it was 2 m into shard 2 (after two seconds), refused,
    // offered again after the pause, refused again.
    let refusals = c.shard(1).stats().aborted;
    assert!(refusals >= 1, "shard 2 refused the bowl");
    assert!(
        refusals <= 3,
        "a refusal pauses the re-offer ({refusals} refusals in five seconds)"
    );
    assert_eq!(c.simulators(guid), vec![ShardId::new(1)], "shard 1 kept it");
    assert_eq!(
        c.state(guid)
            .map(|s| space_sim::wire::encode_ship_state(&s)),
        reference
            .state_of(guid)
            .map(|s| space_sim::wire::encode_ship_state(&s)),
        "and kept simulating it as if nothing had happened"
    );
}

/// H7: the map changes while a body is being handed off. The transfer
/// aborts, the body stays with the shard that had it, and the guid is free
/// to be offered again under the new map.
#[test]
fn an_epoch_change_aborts_a_body_transfer() {
    let mut c = side_by_side(0);
    let guid = c
        .shard_mut(1)
        .spawn_body(BodySpawn::new(
            moving(at_border(0, [2.5, 0.0, 0.0]), [0.5, 0.0, 0.0]),
            ball(0.3, 4.0),
        ))
        .expect("spawn");
    // One tick: shard 1 offers (the ball is 2.5 m into shard 2).
    c.shards[0].tick();
    let offered = c.shards[0]
        .take_outbox()
        .into_iter()
        .any(|o| matches!(o.message, Message::HandoffOffer(ref offer) if offer.entity == guid));
    assert!(offered, "shard 1 offered the ball");
    // The map moves to epoch 2 before anything is answered.
    let map = ShardMap::new(2, c.shards[0].shard_map().regions.clone());
    c.shards[0].apply_epoch(map.clone());
    let aborted = c.shards[0]
        .take_outbox()
        .into_iter()
        .any(|o| matches!(o.message, Message::HandoffAbort(ref a) if a.entity == guid));
    assert!(aborted, "the transfer aborted with the epoch change (H7)");
    assert_eq!(c.simulators(guid), vec![ShardId::new(1)]);
    c.shards[1].apply_epoch(map);
    c.tick_n(60);
    assert_eq!(
        c.simulators(guid),
        vec![ShardId::new(2)],
        "handed off under epoch 2"
    );
}

// ------------------------------------------------------- consolidation

/// Follow a redirect the way a real client does: connect to the new shard
/// with the same token, report in, then drive the ship there.
struct Player {
    token: String,
    shard: u32,
    session: u64,
    ship: EntityGuid,
    redirected_to: Option<(u32, u64)>,
}

impl Player {
    fn join(c: &mut Cluster, shard: u32, token: &str) -> Self {
        let (session, ship) = join(c.shard_mut(shard), token);
        Self {
            token: token.into(),
            shard,
            session,
            ship,
            redirected_to: None,
        }
    }

    /// A player who is already on `shard`, which does not own the spawn
    /// point ([`resident`]): the test places its ship by hand.
    fn resident(c: &mut Cluster, shard: u32, token: &str) -> Self {
        let (session, ship) = resident(c, shard, token);
        Self {
            token: token.into(),
            shard,
            session,
            ship,
            redirected_to: None,
        }
    }

    /// Thrust along `thrust` on whichever shard simulates the ship, and
    /// follow any redirect.
    fn drive(&mut self, c: &mut Cluster, thrust: [f32; 3]) {
        for m in c.frames_for(self.shard, self.session) {
            if let Message::Redirect(r) = m {
                if r.entity == self.ship && self.redirected_to.is_none() {
                    let to = r.new_shard.0;
                    let welcome = c
                        .shard_mut(to)
                        .join(&space_net::hello(&self.token))
                        .expect("the new shard admits the player");
                    c.shard_mut(to).client_status(
                        welcome.session,
                        NeighbourStatus {
                            entity: self.ship,
                            shard: r.new_shard,
                            connected: true,
                        },
                    );
                    self.redirected_to = Some((to, welcome.session));
                }
            }
        }
        if let Some((to, session)) = self.redirected_to {
            if c.shard(to).sim().ship_state(self.ship).is_some() {
                self.shard = to;
                self.session = session;
                self.redirected_to = None;
            }
        }
        let shard = c.shard_mut(self.shard);
        let tick = shard.sim().tick() + 1;
        shard.apply_input(
            self.session,
            &InputTick {
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
            },
        );
    }
}

/// A player's ship pushes a crate along +x across the border and keeps
/// pushing it. Ship and crate are one island, so they are always simulated
/// by one shard together: the crate crosses first and stays with the ship's
/// shard; when the ship crosses, both are handed over in one group (the
/// player is redirected), and the pushing goes on on shard 2.
#[test]
fn a_player_pushes_a_crate_across_the_border_and_keeps_pushing() {
    let mut c = side_by_side(0);
    let mut player = Player::join(&mut c, 1, "pusher");
    // The ship 20 m before the border, the crate 1.5 m ahead of it.
    let ship_at = at_border(0, [-20.0, 0.0, 0.0]);
    assert!(c
        .shard_mut(1)
        .set_state(player.ship, ShipState::at_rest(ship_at)));
    let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.5; 3]));
    crate_body.mass = MassSpec::Mass(200.0);
    crate_body.friction = 0.5;
    let crate_guid = c
        .shard_mut(1)
        .spawn_body(BodySpawn::new(
            ShipState::at_rest(at_border(0, [-18.6, -0.8, 0.0])),
            crate_body,
        ))
        .expect("crate");
    let mut apart_max: f64 = 0.0;
    let mut together_on: BTreeSet<ShardId> = BTreeSet::new();
    let mut crate_crossed_on_1 = false;
    for tick in 0..900 {
        player.drive(&mut c, [1.0, 0.0, 0.0]);
        c.tick();
        let (Some(ship), Some(crate_state)) = (c.state(player.ship), c.state(crate_guid)) else {
            // Between a group's commit and its install neither is simulated
            // for a tick; both are again right after.
            continue;
        };
        let sims_ship = c.simulators(player.ship);
        let sims_crate = c.simulators(crate_guid);
        assert_eq!(
            sims_ship, sims_crate,
            "tick {tick}: ship and crate touch and must be on one shard"
        );
        together_on.extend(sims_ship.iter().copied());
        let gap = WorldPos::distance(ship.pos, crate_state.pos);
        apart_max = apart_max.max(gap);
        if border_x(0, crate_state.pos) > 3.0 && sims_crate == vec![ShardId::new(1)] {
            crate_crossed_on_1 = true;
        }
        if border_x(0, crate_state.pos) > 60.0 {
            break;
        }
    }
    let crate_x = border_x(0, c.state(crate_guid).expect("crate").pos);
    assert!(
        crate_x > 60.0,
        "the crate was pushed 60 m past the border: {crate_x:.1}"
    );
    assert!(
        crate_crossed_on_1,
        "the crate was in shard 2's region while shard 1 still simulated it (consolidated)"
    );
    assert_eq!(
        together_on,
        [ShardId::new(1), ShardId::new(2)].into_iter().collect(),
        "first on shard 1, then together on shard 2"
    );
    assert_eq!(player.shard, 2, "the player followed the redirect");
    assert!(
        apart_max < 3.0,
        "the ship kept pushing: never {apart_max:.2} m apart"
    );
    assert!(c.shard(2).ghost_owner(crate_guid).is_none());
}

/// Two balls, one from each shard, meet head-on at the border. Shard 2's
/// ball is consolidated onto shard 1 before they touch, so one solver
/// resolves the contact, and the result matches the same collision in one
/// world: momentum to 1e-6, each velocity to a millimetre per second.
fn head_on(x0: i64) {
    let mut c = side_by_side(x0);
    let a_state = moving(at_border(x0, [-12.0, 0.0, 0.0]), [4.0, 0.0, 0.0]);
    let b_state = moving(at_border(x0, [12.0, 0.05, 0.0]), [-4.0, 0.0, 0.0]);
    let a = c
        .shard_mut(1)
        .spawn_body(BodySpawn::new(a_state, ball(0.5, 10.0)))
        .expect("a");
    let b = c
        .shard_mut(2)
        .spawn_body(BodySpawn::new(b_state, ball(0.5, 30.0)))
        .expect("b");
    let mut reference = SimWorld::new(ShardId::new(9));
    reference
        .spawn_body(a, BodySpawn::new(a_state, ball(0.5, 10.0)))
        .expect("ref a");
    reference
        .spawn_body(b, BodySpawn::new(b_state, ball(0.5, 30.0)))
        .expect("ref b");
    let mut contact_on: Option<Vec<ShardId>> = None;
    let mut ref_contact_tick = None;
    for tick in 0..240u32 {
        c.tick();
        reference.step();
        if ref_contact_tick.is_none() && !reference.contacts().is_empty() {
            ref_contact_tick = Some(tick);
        }
        if contact_on.is_none() {
            if let Some(s) = c.shards.iter().find(|s| {
                s.sim()
                    .contacts()
                    .iter()
                    .any(|k| (k.a == a && k.b == Some(b)) || (k.a == b && k.b == Some(a)))
            }) {
                contact_on = Some(vec![s.shard()]);
                assert_eq!(
                    c.simulators(a),
                    vec![s.shard()],
                    "a simulated where they touch"
                );
                assert_eq!(
                    c.simulators(b),
                    vec![s.shard()],
                    "b simulated where they touch"
                );
            }
        }
    }
    assert_eq!(
        contact_on,
        Some(vec![ShardId::new(1)]),
        "consolidated on the lower guid's shard"
    );
    assert!(ref_contact_tick.is_some());
    let p = |s: ShipState, m: f64| s.vel.map(|v| f64::from(v) * m);
    let (sa, sb) = (c.state(a).expect("a"), c.state(b).expect("b"));
    let (ra, rb) = (
        reference.state_of(a).expect("ref a"),
        reference.state_of(b).expect("ref b"),
    );
    let total = |x: ShipState, y: ShipState| {
        let (px, py) = (p(x, 10.0), p(y, 30.0));
        [px[0] + py[0], px[1] + py[1], px[2] + py[2]]
    };
    let (two, one) = (total(sa, sb), total(ra, rb));
    let before = 10.0 * 4.0 - 30.0 * 4.0;
    for k in 0..3 {
        assert!(
            (two[k] - one[k]).abs() < 1e-6 * 160.0,
            "momentum {two:?} on two shards, {one:?} in one world"
        );
    }
    assert!(
        (two[0] - before).abs() < 1e-3,
        "momentum conserved: {two:?} vs {before}"
    );
    for (x, y, name) in [(sa, ra, "a"), (sb, rb, "b")] {
        for k in 0..3 {
            assert!(
                (x.vel[k] - y.vel[k]).abs() < 1e-3,
                "{name}: {:?} on two shards, {:?} in one world",
                x.vel,
                y.vel
            );
        }
    }
    assert!(
        sa.vel[0] < 0.0 && sb.vel[0] < 0.0,
        "they bounced: {:?} {:?}",
        sa.vel,
        sb.vel
    );
}

#[test]
fn bodies_colliding_head_on_at_a_border_match_one_shard() {
    head_on(0);
}

#[test]
fn bodies_colliding_head_on_1e13_metres_from_the_origin_match_one_shard() {
    head_on(FAR);
}

/// Two crates on a spring, one on each side of the border, 24 m apart (the
/// spring rests at 20 m): too far apart to be an island, so each shard pulls
/// on its own crate from the other's ghost. The oscillation has the period
/// of the same spring in one world (within 1 %), and the crates stay on
/// their shards.
#[test]
fn a_spring_across_a_border_keeps_its_period() {
    let mut c = side_by_side(0);
    let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.25; 3]));
    crate_body.mass = MassSpec::Mass(10.0);
    crate_body.linear_damping = 0.0;
    let spring_to = |guid: EntityGuid| {
        Springs(vec![Spring {
            part: 0,
            anchor: [0.0; 3],
            target: SpringTarget::Entity {
                guid,
                part: 0,
                anchor: [0.0; 3],
            },
            rest_length: 20.0,
            stiffness: 40.0,
            damping: 0.0,
            max_force: 0.0,
            mode: SpringMode::Spring,
        }])
    };
    // Shard 2's crate first, so shard 1's can name it.
    let b_state = ShipState::at_rest(at_border(0, [12.0, 0.0, 0.0]));
    let a_state = ShipState::at_rest(at_border(0, [-12.0, 0.0, 0.0]));
    let b = c
        .shard_mut(2)
        .spawn_body(BodySpawn::new(b_state, crate_body.clone()))
        .expect("b");
    let a_spawn = BodySpawn::new(a_state, crate_body.clone()).with_springs(spring_to(b));
    let a = c.shard_mut(1).spawn_body(a_spawn.clone()).expect("a");
    let mut reference = SimWorld::new(ShardId::new(9));
    reference
        .spawn_body(b, BodySpawn::new(b_state, crate_body))
        .expect("ref b");
    reference.spawn_body(a, a_spawn).expect("ref a");
    let stretch = |pa: WorldPos, pb: WorldPos| WorldPos::distance(pa, pb) - 20.0;
    let mut two = Vec::new();
    let mut one = Vec::new();
    for _ in 0..600 {
        c.tick();
        reference.step();
        two.push(stretch(
            c.state(a).expect("a").pos,
            c.state(b).expect("b").pos,
        ));
        one.push(stretch(
            reference.state_of(a).expect("a").pos,
            reference.state_of(b).expect("b").pos,
        ));
        assert_eq!(c.simulators(a), vec![ShardId::new(1)]);
        assert_eq!(c.simulators(b), vec![ShardId::new(2)]);
    }
    // Period from the downward zero crossings.
    let period = |series: &[f64]| {
        let crossings: Vec<f64> = series
            .windows(2)
            .enumerate()
            .filter(|(_, w)| w[0] > 0.0 && w[1] <= 0.0)
            .map(|(i, w)| i as f64 + w[0] / (w[0] - w[1]))
            .collect();
        assert!(crossings.len() >= 3, "it oscillates: {crossings:?}");
        (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64
    };
    let (p2, p1) = (period(&two), period(&one));
    // Analytic: reduced mass 5 kg, k 40 N/m -> 2π/√8 s = 66.6 ticks.
    let analytic = std::f64::consts::TAU / 8f64.sqrt() * f64::from(space_sim::TICK_HZ);
    assert!(
        (p2 - p1).abs() / p1 < 0.01,
        "period {p2:.2} ticks across the border, {p1:.2} in one world"
    );
    assert!(
        (p1 - analytic).abs() / analytic < 0.02,
        "{p1:.2} vs {analytic:.2}"
    );
    let amp = |s: &[f64]| s.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    assert!(
        (amp(&two) - amp(&one)).abs() < 0.05 * amp(&one),
        "amplitude {:.3} vs {:.3}",
        amp(&two),
        amp(&one)
    );
}

// --------------------------------------------------------- the scene

/// The demo area split by borders: shard 1 owns a two-cell-wide node, shard 2
/// the one cell carved out of it that the scene's spawn point lies in (the
/// node's +x+y+z cell). The scene's anchor is 6 m and 4 m inside that cell
/// and in the middle of its height (`space_scene::ANCHOR_OFFSET`), so shard 2
/// has the part of the scene with `x > −6` and `z > −4` — the spawn point,
/// the gravity room down to its bowl — and shard 1 the rest: the bowl
/// (`z < −4`), the zero-g corner (`x < −6`), the Kestrel, the planetoid. No
/// border is horizontal: the nearest one is 512 m above or below.
fn split_demo() -> Cluster {
    Cluster::new(&[
        (1, NodeRef::new(63, [0, 0, 0])),
        (2, NodeRef::new(64, [1, 1, 1])),
    ])
}

/// The scene's border planes of [`split_demo`] and of every cluster whose
/// root is more than one cell: `x = −6` and `z = −4` in scene coordinates.
const BORDER_X: f64 = -6.0;
const BORDER_Z: f64 = -4.0;

fn scene_bodies(c: &Cluster) -> Vec<(EntityGuid, ShardId)> {
    let mut out: Vec<(EntityGuid, ShardId)> = c
        .shards
        .iter()
        .flat_map(|s| {
            s.body_list()
                .into_iter()
                .filter(|b| !b.ghost && !b.ship)
                .map(move |b| (b.guid, s.shard()))
        })
        .collect();
    out.sort_unstable();
    out
}

/// The scene's spawner (the shard of the shallowest region) spawns each of
/// its bodies once for the whole cluster — a shard that is told the scene
/// later, like the child of a split, spawns nothing — and a scene change
/// replaces them all, everywhere.
#[test]
fn a_scene_change_replaces_the_bodies_cluster_wide() {
    let mut c = split_demo();
    let total = space_scene::body_defs(Scene::DemoArea).len();
    for s in &mut c.shards {
        assert!(s.set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
    }
    let first = scene_bodies(&c);
    assert_eq!(first.len(), total, "every body once: {first:?}");
    for (guid, shard) in &first {
        assert_eq!(border::scene_revision_of(*guid), Some(1));
        assert_eq!(guid.shard, ShardId::new(1), "minted under the spawner");
        assert_eq!(*shard, ShardId::new(1), "spawned by the spawner");
    }
    // A shard joining the running cluster (a split's child) is told the
    // scene and spawns nothing: its parent's bodies are coming to it.
    let mut late = Shard::new(ShardConfig::new(
        ShardId::new(3),
        addr(),
        NodeRef::new(64, [0, 1, 1]),
    ));
    let mut map = c.shards[0].shard_map().clone();
    map.regions.push(ShardRegion::new(
        NodeRef::new(64, [0, 1, 1]),
        ShardId::new(3),
        addr(),
    ));
    late.set_shard_map(map);
    assert!(late.set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
    assert_eq!(late.stats().bodies, 0, "a late shard spawns nothing");
    // A few seconds of physics across the border: still every body once,
    // each simulated by exactly one shard.
    c.tick_n(150);
    let mut guids: Vec<EntityGuid> = scene_bodies(&c).into_iter().map(|(g, _)| g).collect();
    guids.dedup();
    assert_eq!(
        guids.len(),
        total,
        "no body lost or doubled while simulated"
    );
    for guid in &guids {
        assert_eq!(c.simulators(*guid).len(), 1, "{guid} on exactly one shard");
    }
    // The materials chart has no bodies: they go, ghosts included.
    for s in &mut c.shards {
        assert!(s.set_scene(SceneState::new(Scene::Materials.id(), 2)));
    }
    c.tick_n(5);
    assert!(scene_bodies(&c).is_empty());
    assert!(c.shards.iter().all(|s| s.stats().ghosts == 0));
    // And back: a fresh set, minted for revision 3.
    for s in &mut c.shards {
        assert!(s.set_scene(SceneState::new(Scene::DemoArea.id(), 3)));
    }
    let again = scene_bodies(&c);
    assert_eq!(again.len(), total);
    assert!(again
        .iter()
        .all(|(g, _)| border::scene_revision_of(*g) == Some(3)));
}

/// The gravity room with a border through it ([`split_demo`]: the plane
/// z = −4 of the scene, shard 2 owning the side the spawn point is on): a
/// crate thrown from shard 2's part of the room falls, crosses into shard 1's
/// region, lands on the floor and slides to rest, simulated by exactly one
/// shard at every tick and handed to shard 1 on the way; its path matches the
/// same throw on one shard. A player standing in the room half a metre from
/// the border stands still, on one shard.
///
/// (Until the anchor moved off the corner where cells meet, the border
/// planes through the room were z = 0 and y = 0, and the crate fell through
/// the horizontal one as well. No horizontal border comes near a scene any
/// more; the throw is the old one moved to the lane between the room's west
/// curb and the crate stack, where the floor is clear across z = −4.)
#[test]
fn a_crate_falls_through_the_gravity_room_across_a_border() {
    let throw = |c: &mut Cluster, shard: u32| {
        let anchor = c.shards[0].installed_scene().expect("installed").1;
        let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.3; 3]));
        crate_body.mass = MassSpec::Mass(50.0);
        crate_body.friction = 0.6;
        let start = ShipState {
            vel: [0.0, 0.0, -6.0],
            // 3 m before the border, as the throw was when the border was
            // the plane z = 0.
            ..ShipState::at_rest(WorldPos::from_f64_metres_around(
                anchor,
                [10.9, 2.0, BORDER_Z + 3.0],
            ))
        };
        c.shard_mut(shard)
            .spawn_body(BodySpawn::new(start, crate_body))
            .expect("crate")
    };
    let mut c = split_demo();
    for s in &mut c.shards {
        assert!(s.set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
    }
    let anchor = c.shards[0].installed_scene().expect("installed").1;
    let floor = space_scene::demo::FLOOR;
    let crate_guid = throw(&mut c, 2);
    // (The spawn point is shard 2's: the player is on shard 1 already, and
    // put where it stands by hand.)
    let mut player = Player::resident(&mut c, 1, "stander");
    // Half a metre from the border, on shard 1's side of it.
    let stand = WorldPos::from_f64_metres_around(anchor, [16.5, floor + 1.69, BORDER_Z - 0.6]);
    assert_eq!(
        c.shards[0].shard_map().lookup(&stand),
        Some(ShardId::new(1)),
        "the player stands in shard 1's part of the room"
    );
    assert!(c
        .shard_mut(1)
        .set_state(player.ship, ShipState::at_rest(stand)));
    // The same throw on one shard that owns the whole node.
    let mut one = Cluster::new(&[(1, NodeRef::new(63, [0, 0, 0]))]);
    one.shards[0].set_scene(SceneState::new(Scene::DemoArea.id(), 1));
    let reference = throw(&mut one, 1);
    let mut on_1 = false;
    let mut player_shards = BTreeSet::new();
    for tick in 0..180 {
        player.drive(&mut c, [0.0, 0.0, 0.0]);
        c.tick();
        one.tick();
        let sims = c.simulators(crate_guid);
        assert!(sims.len() <= 1, "tick {tick}: {sims:?}");
        on_1 |= sims == vec![ShardId::new(1)];
        player_shards.extend(c.simulators(player.ship));
    }
    let crate_state = c.state(crate_guid).expect("crate");
    let rel = WorldPos::delta(anchor, crate_state.pos);
    assert!(on_1, "handed to shard 1 on the way");
    assert_eq!(c.simulators(crate_guid), vec![ShardId::new(1)]);
    // Further into shard 1's region than the body hysteresis: there by
    // distance, not only because it has rested there for a while.
    assert!(
        rel[2] < BORDER_Z - border::BODY_HYSTERESIS,
        "it ended in shard 1's region: z = {:.2}",
        rel[2]
    );
    // `cuboid` takes the full size: a 0.3 m crate rests 0.15 m up.
    assert!(
        (rel[1] - (floor + 0.15)).abs() < 0.02,
        "the crate rests on the floor: y = {:.3}",
        rel[1]
    );
    assert!(
        crate_state.vel.iter().all(|v| v.abs() < 0.05),
        "{:?}",
        crate_state.vel
    );
    let single = one.state(reference).expect("reference crate");
    let off = WorldPos::distance(crate_state.pos, single.pos);
    assert!(off < 1e-3, "{off:.4} m from the same throw on one shard");
    let feet = WorldPos::delta(anchor, c.state(player.ship).expect("ship").pos)[1] - 1.68;
    assert!(
        (feet - floor).abs() < 0.03,
        "the player stands on the floor: feet at {feet:.3}"
    );
    assert_eq!(
        player_shards.len(),
        1,
        "the player stayed on one shard: {player_shards:?}"
    );
}

/// What a live run showed, on the borders a scene has now: the demo area
/// split so that most of the gravity room is shard 2's ([`split_demo`]), and
/// a player who starts on shard 1, three metres behind the border between
/// the zero-g corner and the spawn point, and runs through the room. The
/// ship is handed to shard 2 on the way — and takes **nothing** of shard 1's
/// side with it, and nothing moves between the shards once it has gone.
///
/// (The premise of this test changed with the island rule, phase 4 of
/// `docs/plan-sharding.md`. The run used to pull the islands it passed over
/// along — bodies of the zero-g corner and of the bowl, which lie on shard
/// 1's side, more than three of them — and the test pinned that they came
/// back afterwards without being sent to and fro: at most a couple of
/// handoffs each. An island now reaches only what a body is moving
/// *towards*, so the same run pulls none of them, and what is pinned is
/// that: 0 bodies taken along, 0 handoffs afterwards. Bodies a player does
/// reach, and what becomes of them, are `tests/four_cells.rs`'.)
///
/// (The run was first seen with a *horizontal* border at eye height, the
/// floor of the room's +z half on shard 2: the ship crossed it by landing,
/// and the pendulums above it swung in and out of reach of the crates and
/// the seesaw below it. That flicker cannot be staged on the showcase any
/// more — no horizontal border comes within 512 m of a scene, and the
/// vertical ones at x = −6 and z = −4 leave the gantry and the seesaw on one
/// side — and is covered where it is built by hand:
/// [`a_crate_in_reach_of_a_swinging_pendulum_does_not_change_shard_every_swing`].
/// The run through the room is the same one, lane and all.)
#[test]
fn a_player_running_through_a_split_showcase_leaves_it_settled() {
    let mut c = split_demo();
    for s in &mut c.shards {
        assert!(s.set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
    }
    let anchor = c.shards[0].installed_scene().expect("scene").1;
    // (The spawn point is shard 2's: the runner is on shard 1 already, and
    // put on its mark by hand.)
    let mut player = Player::resident(&mut c, 1, "runner");
    let start = WorldPos::from_f64_metres_around(anchor, [BORDER_X - 3.0, 0.0, 2.8]);
    assert_eq!(
        c.shards[0].shard_map().lookup(&start),
        Some(ShardId::new(1)),
        "the run starts in shard 1's part of the scene"
    );
    assert!(c
        .shard_mut(1)
        .set_state(player.ship, ShipState::at_rest(start)));
    let handoffs = |c: &Cluster| -> usize { c.shards.iter().map(|s| s.stats().handed_off).sum() };
    // Scene bodies shard 2 simulates although shard 1 owns their position:
    // what the ship's island took along.
    let pulled_over = |c: &Cluster| -> usize {
        let map = c.shards[0].shard_map();
        c.shard(2)
            .body_list()
            .iter()
            .filter(|b| !b.ghost && !b.ship)
            .filter(|b| map.lookup(&b.state.pos) == Some(ShardId::new(1)))
            .count()
    };
    let mut most_pulled_over = 0;
    // Run +x through the room for three seconds, then let go.
    for _ in 0..90 {
        player.drive(&mut c, [1.0, 0.0, 0.0]);
        c.tick();
        most_pulled_over = most_pulled_over.max(pulled_over(&c));
    }
    for _ in 0..60 {
        player.drive(&mut c, [0.0, 0.0, 0.0]);
        c.tick();
        most_pulled_over = most_pulled_over.max(pulled_over(&c));
    }
    // The run is what it is meant to be: across the border, and not alone.
    assert_eq!(player.shard, 2, "the ship was handed to shard 2 on the way");
    let ship_x = WorldPos::delta(anchor, c.state(player.ship).expect("ship").pos)[0];
    assert!(
        ship_x > 19.0,
        "it ran through the room and out: x = {ship_x:.1}"
    );
    // It took nothing of shard 1's side along: the nearest bodies there —
    // the door and the spring pair of the zero-g corner behind its start,
    // the balls in the bowl beside its path — were never within reach of
    // where it was going. (An island used to reach as far as a body's speed
    // in every direction, behind it and beside it too: at this ship's 15 m/s
    // that was the whole corner and the bowl, more than three bodies taken
    // to shard 2 and sent back afterwards. See `space_sim::IslandQuery`.)
    assert_eq!(
        most_pulled_over, 0,
        "the ship's island took bodies of shard 1's side along"
    );
    let after_run = handoffs(&c);
    c.tick_n(600);
    let settling = handoffs(&c) - after_run;
    let total = space_scene::body_defs(Scene::DemoArea).len();
    assert_eq!(
        settling, 0,
        "handoffs in the twenty seconds after the run: bodies are ping-ponging"
    );
    let settled = handoffs(&c);
    c.tick_n(300);
    assert_eq!(
        handoffs(&c),
        settled,
        "after that, nothing changes shard any more"
    );
    // Every body once, on one shard.
    let mut guids: Vec<EntityGuid> = scene_bodies(&c).into_iter().map(|(g, _)| g).collect();
    guids.dedup();
    assert_eq!(guids.len(), total);
    for guid in &guids {
        assert_eq!(c.simulators(*guid).len(), 1);
    }
}

/// Phase 3's live run, on the borders a scene has now: the demo area with
/// shard 2 on the octant carved out beside the spawn point's cell towards -x
/// (the zero-g corner's side: x < −6, z > −4). The corner's spring pair,
/// its door and two of its floating crates spawn in it and must end up
/// there. A third floating crate spawns on shard 1's side of the z = −4
/// plane; pushed 0.6 m past that plane — inside the 2 m body hysteresis, where
/// a body once stayed on shard 1 for good — it must go over too. A settled
/// body ends up on the shard that owns its position — unless it is in an
/// island with a body of another territory: the fourth floating crate spawns
/// exactly on the border, and with a crate parked against it from shard 1's
/// side it is simulated together with that one, on the home of the island's
/// lowest guid.
///
/// (When the border planes through the scene were those of the spawn point —
/// x, y, z = 0 — the octant was x < 0, y > 0, z > 0: the spring pair spawned
/// 0.3 m above its floor, the pushed crate ended 0.6 m above it, and the
/// island across a border was a crate parked against the door, which hung
/// just below that floor. The horizontal border is gone; each of the three
/// cases is the same case at a border the scene still has.)
#[test]
fn the_zero_g_corner_settles_on_a_carved_octant() {
    let mut c = Cluster::new(&[
        (1, NodeRef::new(63, [0, 0, 0])),
        (2, NodeRef::new(64, [0, 1, 1])),
    ]);
    for s in &mut c.shards {
        assert!(s.set_scene(SceneState::new(Scene::DemoArea.id(), 1)));
    }
    let anchor = c.shards[0].installed_scene().expect("scene").1;
    let owner_of = |c: &Cluster, p: WorldPos| c.shards[0].shard_map().lookup(&p).expect("owned");
    let corner: Vec<EntityGuid> = c.shards[0]
        .body_list()
        .into_iter()
        .filter(|b| b.visual == Some(space_scene::visual::CRATE))
        .filter(|b| WorldPos::delta(anchor, b.state.pos)[0] < -9.0)
        .map(|b| b.guid)
        .collect();
    assert!(
        corner.len() >= 6,
        "the floating crates and the spring pair: {corner:?}"
    );
    let crate_at = |c: &Cluster, x: f64, z: f64| -> EntityGuid {
        *corner
            .iter()
            .find(|g| {
                let d = WorldPos::delta(anchor, c.state(**g).expect("alive").pos);
                (d[0] - x).abs() < 0.1 && (d[2] - z).abs() < 0.1
            })
            .unwrap_or_else(|| panic!("the floating crate at ({x}, {z})"))
    };
    // Most of the corner spawns in shard 2's octant (by the spawner, shard 1).
    let spawned_in_2 = corner
        .iter()
        .filter(|g| owner_of(&c, c.state(**g).expect("alive").pos) == ShardId::new(2))
        .count();
    assert!(spawned_in_2 >= 5, "{spawned_in_2} of the corner's crates");
    // Push the crate at (-15.5, 0.0, -5.5), on shard 1's side, to 0.6 m
    // inside the octant.
    let pushed = crate_at(&c, -15.5, -5.5);
    assert_eq!(
        owner_of(&c, c.state(pushed).expect("alive").pos),
        ShardId::new(1)
    );
    let to = WorldPos::from_f64_metres_around(anchor, [-15.5, 0.0, BORDER_Z + 0.6]);
    assert_eq!(owner_of(&c, to), ShardId::new(2));
    assert!(c.shard_mut(1).set_state(pushed, ShipState::at_rest(to)));
    // The crate at (-13.2, 0.6, -4.0) floats exactly on the border, which
    // makes it the octant's; another crate parked against it, 2 cm from it,
    // on shard 1's side.
    let on_border = crate_at(&c, -13.2, BORDER_Z);
    let on_border_at = WorldPos::delta(anchor, c.state(on_border).expect("alive").pos);
    assert_eq!(
        owner_of(&c, c.state(on_border).expect("alive").pos),
        ShardId::new(2),
        "a border belongs to the cell above it"
    );
    let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.4; 3]));
    crate_body.mass = MassSpec::Mass(20.0);
    let parked_at = WorldPos::from_f64_metres_around(
        anchor,
        [
            on_border_at[0],
            on_border_at[1],
            // Half the floating crate (0.7 m), the gap, half this one.
            on_border_at[2] - 0.35 - 0.02 - 0.2,
        ],
    );
    assert_eq!(owner_of(&c, parked_at), ShardId::new(1));
    let parked = c
        .shard_mut(1)
        .spawn_body(BodySpawn::new(ShipState::at_rest(parked_at), crate_body))
        .expect("parked");
    assert!(parked < on_border, "the parked crate has the lower guid");
    c.tick_n(600);
    if std::env::var_os("PHYS_DEBUG").is_some() {
        for b in c
            .shards
            .iter()
            .flat_map(|s| s.body_list())
            .filter(|b| !b.ghost)
        {
            let rel = WorldPos::delta(anchor, b.state.pos);
            if rel[0] < -5.0 {
                eprintln!(
                    "{} {:?} on {:?} at {:.2?} island home {:?} map {:?}",
                    b.guid,
                    b.visual,
                    b.simulator,
                    rel,
                    b.island_home,
                    c.shards[0].shard_map().lookup(&b.state.pos)
                );
            }
        }
    }
    assert_eq!(
        c.simulators(parked),
        c.simulators(on_border),
        "a crate against the one on the border is in its island and on its shard"
    );
    assert_eq!(
        c.simulators(on_border),
        vec![ShardId::new(1)],
        "on the home of the island's lowest guid: the parked crate's"
    );
    let mut settled_on_2 = 0;
    for guid in corner.iter().filter(|g| **g != on_border) {
        let s = c.state(*guid).expect("alive");
        let owner = owner_of(&c, s.pos);
        assert_eq!(
            c.simulators(*guid),
            vec![owner],
            "{guid} at {:.2?} is settled and must be on the shard that owns it",
            WorldPos::delta(anchor, s.pos)
        );
        settled_on_2 += usize::from(owner == ShardId::new(2));
    }
    assert!(
        settled_on_2 >= 5,
        "the corner went to the octant's shard: {settled_on_2}"
    );
    // The door too, which is not a crate.
    let door = c
        .shards
        .iter()
        .flat_map(Shard::body_list)
        .find(|b| !b.ghost && b.visual == Some(space_scene::visual::DOOR))
        .expect("the door");
    assert_eq!(door.simulator, owner_of(&c, door.state.pos));
    assert_eq!(door.simulator, ShardId::new(2));
    assert!(
        c.simulators(pushed) == vec![ShardId::new(2)],
        "the pushed crate went over"
    );
}

/// The hold: a pendulum on shard 1 swings towards the border every second,
/// its reach (velocity lead) sweeping over a crate resting 1 m inside shard
/// 2. They are an island at every swing, so the crate is consolidated onto
/// the pendulum's shard; between swings it is alone and, once settled, would
/// want to go home. It must not go back and forth at every swing: an island
/// that was in a group waits until it has wanted to go home for a while.
#[test]
fn a_crate_in_reach_of_a_swinging_pendulum_does_not_change_shard_every_swing() {
    let mut c = side_by_side(0);
    let anchor = at_border(0, [-3.0, 0.0, 0.0]);
    let mut bob = ball(0.3, 10.0);
    bob.restitution = 0.0;
    let swing = BodySpawn::new(ShipState::at_rest(at_border(0, [-1.5, 0.0, 0.0])), bob)
        .with_springs(Springs(vec![Spring {
            part: 0,
            anchor: [0.0; 3],
            target: SpringTarget::World(anchor),
            rest_length: 0.0,
            stiffness: 10.0 * std::f32::consts::PI * std::f32::consts::PI,
            damping: 0.0,
            max_force: 0.0,
            mode: SpringMode::Spring,
        }]));
    let pendulum = c.shard_mut(1).spawn_body(swing).expect("pendulum");
    let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.5; 3]));
    crate_body.mass = MassSpec::Mass(50.0);
    let crate_guid = c
        .shard_mut(2)
        .spawn_body(BodySpawn::new(
            ShipState::at_rest(at_border(0, [1.0, 0.0, 0.0])),
            crate_body,
        ))
        .expect("crate");
    let handoffs = |c: &Cluster| -> usize { c.shards.iter().map(|s| s.stats().handed_off).sum() };
    c.tick_n(150);
    let settled = handoffs(&c);
    assert_eq!(
        c.simulators(crate_guid),
        c.simulators(pendulum),
        "consolidated with the pendulum"
    );
    c.tick_n(600);
    assert_eq!(
        handoffs(&c),
        settled,
        "twenty swings, and the crate never changed shard again"
    );
    let x = border_x(0, c.state(pendulum).expect("pendulum").pos);
    assert!((-4.6..-1.4).contains(&x), "still swinging: {x}");
}

/// The hold's backoff, from a live run: something of one shard reaches a
/// body of the other once per slow swing (an 8 s pendulum here; a spring
/// pair drifting in and out of a parked player's reach there), and between
/// swings the body is alone for longer than the hold. Every swing it is
/// consolidated over, and every gap it would be sent home — unless the
/// shard that keeps getting it back holds it longer each time.
#[test]
fn a_body_that_keeps_coming_back_is_held_longer_each_time() {
    let mut c = side_by_side(0);
    let period = 8.0f32;
    let omega = std::f32::consts::TAU / period;
    let swing = BodySpawn::new(
        ShipState::at_rest(at_border(0, [-1.5, 0.0, 0.0])),
        ball(0.3, 10.0),
    )
    .with_springs(Springs(vec![Spring {
        part: 0,
        anchor: [0.0; 3],
        target: SpringTarget::World(at_border(0, [-3.0, 0.0, 0.0])),
        rest_length: 0.0,
        stiffness: 10.0 * omega * omega,
        damping: 0.0,
        max_force: 0.0,
        mode: SpringMode::Spring,
    }]));
    c.shard_mut(1).spawn_body(swing).expect("pendulum");
    let mut crate_body = Body::dynamic(ShapeRef::cuboid([0.5; 3]));
    crate_body.mass = MassSpec::Mass(50.0);
    let crate_guid = c
        .shard_mut(2)
        .spawn_body(BodySpawn::new(
            ShipState::at_rest(at_border(0, [0.5, 0.0, 0.0])),
            crate_body,
        ))
        .expect("crate");
    let moves = |c: &Cluster| -> usize { c.shards.iter().map(|s| s.stats().handed_off).sum() };
    let mut changes = 0;
    let mut last = c.simulators(crate_guid);
    for _ in 0..(120 * 30) {
        c.tick();
        let now = c.simulators(crate_guid);
        if !now.is_empty() && now != last {
            changes += 1;
            last = now;
        }
    }
    // Fifteen swings. Without the backoff the crate changes shard twice a
    // swing (about 30 times); with it, a few times until the hold outlasts
    // the gap, then never.
    assert!(
        changes <= 8,
        "the crate changed shard {changes} times in 15 swings ({} handoffs)",
        moves(&c)
    );
    let settled = moves(&c);
    c.tick_n(60 * 30);
    assert_eq!(moves(&c), settled, "and then it stays put");
}

/// A settled body — not in an island with another shard's bodies — ends up
/// on the shard that owns its position, in every layout a cluster can have:
/// sibling cells, an octant carved out of a larger node, and a node carved
/// out of a carved node; at the origin and 1e13 m out. Bodies are placed
/// (all spawned by shard 1, at rest, weightless):
///
/// * deep inside every other shard's territory;
/// * 0.6 m from a face of it but metres from the others (inside the 2 m
///   distance hysteresis: what a crate floating at spawn height is);
/// * a pair 3 m apart, 0.3 m from a face (the demo's spring pair);
/// * 3 m inside, 6 m from a body that stays on shard 1 (two bodies of one
///   shard close enough to be loosely near but in separate islands).
#[test]
fn a_settled_body_ends_on_the_shard_that_owns_its_position() {
    for base in [0i64, FAR - FAR % 4] {
        let quarter = base / 4;
        let half = base / 2;
        let layouts: [(&str, Vec<(u32, NodeRef)>); 3] = [
            (
                "siblings",
                vec![
                    (1, NodeRef::new(64, [base, 0, 0])),
                    (2, NodeRef::new(64, [base + 1, 0, 0])),
                ],
            ),
            (
                "a carved octant",
                vec![
                    (1, NodeRef::new(63, [half, 0, 0])),
                    (2, NodeRef::new(64, [base, 1, 1])),
                ],
            ),
            (
                "a carved octant of a carved octant",
                vec![
                    (1, NodeRef::new(62, [quarter, 0, 0])),
                    (2, NodeRef::new(63, [half, 1, 1])),
                    (3, NodeRef::new(64, [base + 1, 2, 2])),
                ],
            ),
        ];
        for (name, regions) in layouts {
            let mut c = Cluster::new(&regions);
            let mut placed: Vec<(EntityGuid, &str)> = Vec::new();
            let spawn = |c: &mut Cluster, at: WorldPos| {
                c.shard_mut(1)
                    .spawn_body(BodySpawn::new(ShipState::at_rest(at), ball(0.4, 5.0)))
                    .expect("spawn")
            };
            for (_, node) in regions.iter().skip(1) {
                // Offsets from the node's lower corner, `axis` being one whose
                // lower face borders another shard of the cluster (the other
                // lower faces may be the cluster's edge).
                // Siblings sit side by side along x, and the innermost octant
                // has shard 2 below it along x; a carved octant has its
                // parent below it along y.
                let side_by_side_x = node.depth == 64
                    && ((regions.len() == 2 && node.index[1] == 0) || regions.len() == 3);
                let axis = if side_by_side_x { 0 } else { 1 };
                let lo = space_world::octree::node_bounds(*node).0;
                let at = |depth: f64, along: f64, across: f64| {
                    let mut d = [across; 3];
                    d[axis] = depth;
                    d[(axis + 1) % 3] = along;
                    WorldPos::from_f64_metres_around(lo, d)
                };
                placed.push((spawn(&mut c, at(300.0, 300.0, 300.0)), "deep inside"));
                placed.push((spawn(&mut c, at(0.6, 9.0, 300.0)), "0.6 m from a face"));
                placed.push((spawn(&mut c, at(0.3, 200.0, 200.0)), "a pair, 0.3 m in"));
                placed.push((spawn(&mut c, at(0.3, 203.0, 200.0)), "a pair, 0.3 m in"));
                placed.push((
                    spawn(&mut c, at(3.0, 100.0, 100.0)),
                    "3 m in, 6 m from another",
                ));
                placed.push((
                    spawn(&mut c, at(-3.0, 100.0, 100.0)),
                    "3 m outside, 6 m from it",
                ));
            }
            c.tick_n(600);
            for (guid, what) in &placed {
                let s = c.state(*guid).expect("alive");
                let owner = c.shards[0].shard_map().lookup(&s.pos).expect("owned");
                assert_eq!(
                    c.simulators(*guid),
                    vec![owner],
                    "{name} at cell {base}: the body {what} must be on the shard that owns it"
                );
            }
        }
    }
}

/// The orchestrator's split, from the shards' side: shard 1 owns a node with
/// bodies everywhere in it; the map then gives one octant to a new shard 2.
/// Shard 1 hands that octant's bodies to it and keeps the rest.
#[test]
fn a_split_hands_the_carved_octants_bodies_to_the_child() {
    let parent = NodeRef::new(63, [0, 0, 0]);
    let child = NodeRef::new(64, [1, 1, 1]);
    let mut c = Cluster::new(&[(1, parent)]);
    let corner = WorldPos::new([1, 1, 1], [-512.0; 3]); // the octant's lower corner
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for i in 0..4 {
        let d = 20.0 + 30.0 * f64::from(i);
        let spawn = |p: [f64; 3]| {
            BodySpawn::new(
                ShipState::at_rest(WorldPos::from_f64_metres_around(corner, p)),
                ball(0.4, 5.0),
            )
        };
        inside.push(c.shards[0].spawn_body(spawn([d, d, d])).expect("in"));
        outside.push(c.shards[0].spawn_body(spawn([-d, d, d])).expect("out"));
    }
    c.tick_n(10);
    // The split: shard 2 starts on the child octant, both shards get the
    // map of epoch 2 (what the orchestrator's commit installs).
    let map = ShardMap::new(
        2,
        vec![
            ShardRegion::new(parent, ShardId::new(1), addr()),
            ShardRegion::new(child, ShardId::new(2), addr()),
        ],
    );
    let mut kid = Shard::new(ShardConfig::new(ShardId::new(2), addr(), child));
    kid.set_shard_map(map.clone());
    c.shards[0].apply_epoch(map);
    c.shards.push(kid);
    c.tick_n(60);
    for guid in &inside {
        assert_eq!(
            c.simulators(*guid),
            vec![ShardId::new(2)],
            "{guid} went to the child"
        );
    }
    for guid in &outside {
        assert_eq!(c.simulators(*guid), vec![ShardId::new(1)], "{guid} stayed");
    }
}

// ---------------------------------------------------------------- ships

/// Players joining one shard get distinct spawn points, none inside
/// anything; their ships are solid: one flown into another pushes it.
#[test]
fn joining_players_get_distinct_free_spawn_points_and_ships_are_solid() {
    let mut c = Cluster::new(&[(1, NodeRef::new(64, [0, 0, 0]))]);
    c.shards[0].set_scene(SceneState::new(Scene::DemoArea.id(), 1));
    let mut players: Vec<(u64, EntityGuid)> = Vec::new();
    for i in 0..6 {
        players.push(join(&mut c.shards[0], &format!("p{i}")));
    }
    let positions: Vec<WorldPos> = players
        .iter()
        .map(|(_, g)| c.state(*g).expect("ship").pos)
        .collect();
    for (i, p) in positions.iter().enumerate() {
        for q in &positions[i + 1..] {
            assert!(WorldPos::distance(*p, *q) >= 1.4, "{p:?} and {q:?} overlap");
        }
    }
    let anchor = c.shards[0].installed_scene().expect("scene").1;
    assert_eq!(
        positions[0], anchor,
        "the first player gets the spawn point itself"
    );
    // Two ships 1.5 m apart along -z: fly the second into the first.
    let (s0, g0) = players[0];
    let (s1, g1) = players[1];
    let before = c.state(g0).expect("ship 0").pos;
    let dir = {
        let d = WorldPos::delta(c.state(g1).expect("1").pos, before);
        let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        [(d[0] / n) as f32, (d[1] / n) as f32, (d[2] / n) as f32]
    };
    for _ in 0..60 {
        let tick = c.shards[0].sim().tick() + 1;
        c.shards[0].apply_input(
            s1,
            &InputTick {
                tick,
                seq: tick,
                inputs: vec![PlayerInput {
                    entity: g1,
                    thrust: dir,
                    torque: [0.0; 3],
                    tick,
                    fast: false,
                }],
                ack_tick: 0,
            },
        );
        c.tick();
    }
    let _ = s0;
    let after = c.state(g0).expect("ship 0").pos;
    assert!(
        WorldPos::distance(before, after) > 0.3,
        "the first ship was pushed, not flown through"
    );
    let gap = WorldPos::distance(after, c.state(g1).expect("ship 1").pos);
    assert!(
        gap > 0.55,
        "the ships do not overlap: {gap:.3} m between centres"
    );
}

/// A busy spawn point never puts a new ship across a border. The scene is cut
/// into its four shard areas — the root keeps the cell the spawn point lies
/// in, 6 m and 4 m from its borders — and a hundred players (all a shard
/// admits) join the root's shard: every one of them starts on that shard's
/// own ground, the first on the spawn point itself, no two closer than a
/// ship's width. On a shard that has no neighbours the same hundred spread
/// over both border planes, which is what the spawn grid (15 m out) does when
/// nothing holds it back.
///
/// Mutation this catches: handing out the nearest free grid point whoever
/// owns it — the 26th player already lands 0.5 m inside a neighbour and
/// spends its first second being handed over.
#[test]
fn a_busy_spawn_point_never_puts_a_ship_across_a_border() {
    let root = NodeRef::new(63, [0, 0, 0]);
    let crowd = space_server::shard::MAX_PLAYERS_PER_SHARD;
    let fill = |c: &mut Cluster| -> Vec<WorldPos> {
        (0..crowd)
            .map(|i| {
                let (_, ship) = join(&mut c.shards[0], &format!("p{i}"));
                c.shards[0].sim().ship_state(ship).expect("a ship").pos
            })
            .collect()
    };
    // The four cells that meet at the root's centre corner (cell 1 on each
    // axis): the root's own, and one each towards -z, -x and both.
    let mut four = Cluster::new(&[
        (1, root),
        (2, NodeRef::new(64, [1, 1, 0])),
        (3, NodeRef::new(64, [0, 1, 1])),
        (4, NodeRef::new(64, [0, 1, 0])),
    ]);
    let map = four.shards[0].shard_map().clone();
    let anchor = space_scene::scene_anchor(&map).expect("an anchor");
    assert_eq!(map.lookup(&anchor), Some(ShardId::new(1)));
    let at = |p: WorldPos| WorldPos::delta(anchor, p);
    // The borders are where the anchor rule puts them.
    for (d, owner) in [
        ([0.0, 0.0, 0.0], 1),
        ([BORDER_X, 0.0, BORDER_Z], 1),
        ([0.0, 0.0, BORDER_Z - 0.01], 2),
        ([BORDER_X - 0.01, 0.0, 0.0], 3),
        ([BORDER_X - 0.01, 0.0, BORDER_Z - 0.01], 4),
        ([0.0, 511.0, 0.0], 1),
        ([0.0, -511.0, 0.0], 1),
    ] {
        let p = WorldPos::from_f64_metres_around(anchor, d);
        assert_eq!(map.lookup(&p), Some(ShardId::new(owner)), "{d:?}");
    }

    let positions = fill(&mut four);
    assert_eq!(positions.len(), crowd);
    assert_eq!(
        positions[0], anchor,
        "the first player gets the spawn point"
    );
    for (i, p) in positions.iter().enumerate() {
        let d = at(*p);
        assert_eq!(
            map.lookup(p),
            Some(ShardId::new(1)),
            "player {i} starts at {d:.1?}, on another shard's ground"
        );
        assert!(d[0] >= BORDER_X && d[2] >= BORDER_Z && d[1] == 0.0, "{d:?}");
        for q in &positions[i + 1..] {
            assert!(WorldPos::distance(*p, *q) >= 1.4, "{p:?} and {q:?} overlap");
        }
    }
    // Nobody was dropped on the spawn point for want of a free one.
    assert_eq!(positions.iter().filter(|p| **p == anchor).count(), 1);

    // The same crowd on a shard that owns everything: the grid is not held
    // back, and reaches over both planes.
    let mut alone = Cluster::new(&[(1, root)]);
    let spread = fill(&mut alone);
    assert_eq!(
        space_scene::scene_anchor(alone.shards[0].shard_map()),
        Some(anchor)
    );
    assert!(spread.iter().any(|p| at(*p)[2] < BORDER_Z), "over z = -4");
    assert!(spread.iter().any(|p| at(*p)[0] < BORDER_X), "over x = -6");
    // And the players that fit on the root's ground are placed alike.
    let shared = spread
        .iter()
        .zip(&positions)
        .take_while(|(a, b)| a == b)
        .count();
    assert_eq!(
        shared, 25,
        "the first 25 points are the same; the 26th is the first across a border"
    );
    assert!(at(spread[25])[2] < BORDER_Z, "{:?}", at(spread[25]));

    // A shard that does not own the spawn point has no place for a joiner:
    // it sends it to the shard that does, and spawns nothing. (It used to put
    // the ship on the centre of its own region.)
    let ships_before = four.shard(2).stats().ships;
    let refused = four
        .shard_mut(2)
        .join(&space_net::hello("elsewhere"))
        .expect_err("shard 2 does not own the spawn point");
    let space_server::shard::JoinRefusal::Elsewhere(redirect) = refused else {
        panic!("expected a redirect to the spawn point's owner, got {refused:?}");
    };
    assert_eq!(redirect.shard, ShardId::new(1));
    assert_eq!(redirect.shard_map, map);
    assert_eq!(four.shard(2).free_spawn_point(), None);
    assert_eq!(four.shard(2).stats().ships, ships_before);
    assert_eq!(four.shard(2).stats().clients, 0);
}

/// The `Visual` (component 5) a frame entry carries, if any.
fn visual_of(e: &EntitySnapshot) -> Option<u32> {
    e.components
        .iter()
        .find(|(id, _)| *id == 5)
        .map(|(_, b)| space_sim::body::decode_visual(b).expect("a visual").0)
}

/// A player's ship carries `Visual(PLAYER)`: in its own client's frames, in
/// the ghost a neighbour shows (where its `Owner` is 0, a session of
/// another shard meaning nothing there), and still after it was handed to
/// the neighbour. A ship nobody flies has none.
#[test]
fn a_players_ship_is_drawn_as_a_player_everywhere() {
    use space_scene::visual::PLAYER;
    let mut c = side_by_side(0);
    let mut player = Player::join(&mut c, 1, "walker");
    // (Shard 1 owns the spawn point; the watcher is on shard 2 already.)
    let (watcher, _) = resident(&mut c, 2, "watcher");
    let stray = c
        .shard_mut(1)
        .spawn_ship_for(0, at_border(0, [-40.0, 10.0, 0.0]));
    assert!(c.shard_mut(1).set_state(
        player.ship,
        ShipState::at_rest(at_border(0, [-20.0, 0.0, 0.0]))
    ));
    let mut own = None;
    let mut ghost = None;
    let mut stray_seen = None;
    for _ in 0..30 {
        player.drive(&mut c, [0.0; 3]);
        c.tick();
        for m in c.frames_for(1, player.session) {
            if let Message::Snapshot(s) = &m {
                for e in &s.entities {
                    if e.guid == player.ship {
                        own = Some(visual_of(e));
                    }
                    if e.guid == stray {
                        stray_seen = Some(visual_of(e));
                    }
                }
            }
        }
        for m in c.frames_for(2, watcher) {
            let entities = match &m {
                Message::Snapshot(s) => s.entities.clone(),
                Message::Delta(d) => d.entities.clone(),
                _ => Vec::new(),
            };
            for e in entities.iter().filter(|e| e.guid == player.ship) {
                let owner = e
                    .components
                    .iter()
                    .find(|(id, _)| *id == 2)
                    .map(|(_, b)| space_protocol::codec::decode_raw::<u64>(b).expect("u64").0);
                if visual_of(e).is_some() {
                    ghost = Some((visual_of(e), owner));
                }
            }
        }
    }
    assert_eq!(own, Some(Some(PLAYER)), "the player's own snapshot");
    assert_eq!(stray_seen, Some(None), "a ship nobody flies has no visual");
    assert_eq!(
        ghost,
        Some((Some(PLAYER), Some(0))),
        "the neighbour's client sees the ghost as a player, owned by nobody it knows"
    );
    // Walk across: on shard 2 the ship is still a player's.
    for _ in 0..300 {
        player.drive(&mut c, [1.0, 0.0, 0.0]);
        c.tick();
        if player.shard == 2 {
            break;
        }
    }
    assert_eq!(player.shard, 2, "handed over");
    let entry = c
        .shard(2)
        .replicated_entity(player.ship)
        .expect("on shard 2");
    assert_eq!(visual_of(&entry), Some(PLAYER));
}

// ------------------------------------------------------------ bandwidth

/// Encoded size of a frame, bytes.
fn frame_bytes(m: &Message) -> usize {
    space_protocol::encode_frame(m).len()
}

/// A client standing at the demo area's spawn point while the showcase moves
/// (crates settling, balls rolling, the seesaw tipping, two pendulums and a
/// chain swinging): what it is sent per tick, and what the chain costs. The
/// numbers are printed (`--nocapture`); the bounds are generous ceilings.
#[test]
fn replication_bandwidth_of_the_moving_showcase() {
    let mut c = Cluster::new(&[(1, NodeRef::new(64, [0, 0, 0]))]);
    c.shards[0].set_scene(SceneState::new(Scene::DemoArea.id(), 1));
    let (session, _) = join(&mut c.shards[0], "watcher");
    let chain = c.shards[0]
        .body_list()
        .into_iter()
        .find(|b| b.visual == Some(space_scene::visual::CHAIN_LINK))
        .expect("the chain")
        .guid;
    let mut per_tick: Vec<usize> = Vec::new();
    let mut chain_bytes: Vec<usize> = Vec::new();
    let mut first_frame = 0usize;
    let started = std::time::Instant::now();
    for tick in 0..300 {
        c.tick();
        let frames = c.frames_for(1, session);
        let bytes: usize = frames.iter().map(frame_bytes).sum();
        if tick == 0 {
            first_frame = bytes;
            continue;
        }
        per_tick.push(bytes);
        let chain_entry: usize = frames
            .iter()
            .filter_map(|m| match m {
                Message::Delta(d) => d.entities.iter().find(|e| e.guid == chain),
                _ => None,
            })
            .map(EntitySnapshot::encoded_len)
            .sum();
        chain_bytes.push(chain_entry);
    }
    let elapsed = started.elapsed();
    let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;
    let first_second = mean(&per_tick[..29]);
    let last_second = mean(&per_tick[per_tick.len() - 30..]);
    let chain_mean = mean(&chain_bytes);
    let full_chain = c.shards[0]
        .replicated_entity(chain)
        .expect("chain")
        .encoded_len();
    println!(
        "showcase: join snapshot {first_frame} B; deltas {first_second:.0} B/tick in the \
         first second, {last_second:.0} B/tick after ten ({:.1} KB/s); the chain {chain_mean:.0} \
         B/tick (its full entry {full_chain} B); {} bodies; {:.2} ms per tick (shard + \
         harness, this build)",
        last_second * 30.0 / 1000.0,
        c.shards[0].stats().bodies,
        elapsed.as_secs_f64() * 1000.0 / 300.0
    );
    // The same showcase with a border through it (pendulums and chain on
    // shard 2): what the two shards send each other in ghost frames.
    let mut split = split_demo();
    for s in &mut split.shards {
        s.set_scene(SceneState::new(Scene::DemoArea.id(), 1));
    }
    split.tick_n(300);
    let ghosts_on = |c: &Cluster, id: u32| c.shard(id).stats().ghosts;
    println!(
        "split showcase: {:.0} B/tick of ghost frames between the shards (both ways, ten-second mean); ghosts held: {} on shard 1, {} on shard 2",
        split.ghost_bytes as f64 / 300.0,
        ghosts_on(&split, 1),
        ghosts_on(&split, 2)
    );
    assert!(first_second < 12_000.0, "{first_second} B/tick");
    assert!(chain_mean < 800.0, "{chain_mean} B/tick for the chain");
    assert!(
        chain_mean < full_chain as f64 * 0.75,
        "the chain's deltas carry its motion, not its whole description"
    );
}

// ----------------------------------------------------------------- QUIC

mod quic {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    use space_server::net::ShardNet;
    use tokio::sync::Mutex;
    use tokio::task::JoinHandle;

    pub struct Running {
        pub net: Arc<ShardNet>,
        pub shard: Arc<Mutex<Shard>>,
        tasks: Vec<JoinHandle<()>>,
    }

    impl Running {
        pub fn addr(&self) -> SocketAddr {
            self.net.local_addr().expect("bound")
        }

        pub fn stop(self) {
            for t in &self.tasks {
                t.abort();
            }
            self.net.close();
        }
    }

    /// A shard on loopback, served and ticked at 30 Hz like the binary.
    pub fn start(id: u32, region: NodeRef) -> Running {
        let net = Arc::new(ShardNet::bind(addr()).expect("bind"));
        let shard = Arc::new(Mutex::new(Shard::new(ShardConfig::new(
            ShardId::new(id),
            net.local_addr().expect("addr"),
            region,
        ))));
        let serve = tokio::spawn({
            let (net, shard) = (Arc::clone(&net), Arc::clone(&shard));
            async move { net.serve(shard).await }
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
}

/// The handoff through the real front end: two shards over loopback QUIC,
/// ticking on their own clocks at 30 Hz, ghost frames and handoff messages on
/// the real peer connections, and a raw client connected to shard 1 that
/// watches a ball fly into shard 2. It sees the ball in every frame, never
/// removed while near, never jumping more than a couple of ticks' travel
/// (the two shards' clocks drift, and a ghost is a tick behind its owner).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn over_quic_a_body_crosses_between_two_real_shards() {
    use std::time::Duration;
    let a = quic::start(1, NodeRef::new(64, [0, 0, 0]));
    let b = quic::start(2, NodeRef::new(64, [1, 0, 0]));
    let map = ShardMap::new(
        1,
        vec![
            ShardRegion::new(NodeRef::new(64, [0, 0, 0]), ShardId::new(1), a.addr()),
            ShardRegion::new(NodeRef::new(64, [1, 0, 0]), ShardId::new(2), b.addr()),
        ],
    );
    a.shard.lock().await.set_shard_map(map.clone());
    b.shard.lock().await.set_shard_map(map);
    let speed = 10.0f32;
    let guid = a
        .shard
        .lock()
        .await
        .spawn_body(BodySpawn::new(
            moving(at_border(0, [-15.0, 30.0, 0.0]), [speed, 0.0, 0.0]),
            ball(0.4, 10.0),
        ))
        .expect("spawn");

    let endpoint = space_net::client_endpoint_for(a.addr()).expect("endpoint");
    let (conn, _welcome) = space_net::connect(
        &endpoint,
        a.addr(),
        space_server::net::SERVER_NAME,
        &space_net::hello("watcher"),
    )
    .await
    .expect("connect");
    let view = Arc::new(std::sync::Mutex::new(View::default()));
    let reader = tokio::spawn({
        let (conn, view) = (conn.clone(), Arc::clone(&view));
        async move {
            loop {
                let next = tokio::select! {
                    m = conn.recv_datagram() => m,
                    m = conn.recv_stream() => m,
                };
                match next {
                    Ok(Some(m)) => view.lock().expect("view").apply(guid, &m),
                    _ => return,
                }
            }
        }
    });
    // Up to 12 s of real time, paced by the shards' own loops.
    let mut handed = false;
    for _ in 0..240 {
        tokio::time::sleep(Duration::from_millis(50)).await;
        let on_b = b.shard.lock().await.sim().ship_state(guid);
        if let Some(s) = on_b {
            handed = true;
            if border_x(0, s.pos) > 40.0 {
                break;
            }
        }
    }
    reader.abort();
    assert!(handed, "the ball was handed to shard 2");
    assert!(a.shard.lock().await.sim().ship_state(guid).is_none());
    assert_eq!(
        a.shard.lock().await.ghost_owner(guid),
        Some(ShardId::new(2)),
        "shard 1 shows it as shard 2's ghost"
    );
    let view = view.lock().expect("view");
    assert!(view.removed_at.is_empty(), "never removed while in sight");
    let per_tick = f64::from(speed) * f64::from(space_sim::dt());
    assert!(
        view.largest_step() < per_tick * 3.0,
        "a jump of {:.3} m per tick ({per_tick:.3} m of travel)",
        view.largest_step()
    );
    // At the rate of its distance step from the first sighting on: the
    // watcher stands on the spawn point, half a kilometre from the ball,
    // where a moving thing is sent five times a second — every sixth tick
    // (`space_protocol::far`; it was every tick while nothing was thinned).
    // Frames are datagrams and may be lost on a busy host, so allow a few.
    let seen: BTreeSet<u64> = view.seen.iter().map(|(t, _)| *t).collect();
    let span = seen.last().expect("seen") - seen.first().expect("seen") + 1;
    let every = space_protocol::far::interval(2);
    assert!(
        seen.len() as u64 * every * 10 >= span * 9,
        "seen on {} of {span} ticks, at one in {every}",
        seen.len()
    );
    drop(view);
    a.stop();
    b.stop();
}

/// A peer still on protocol 6 — a client or shard of the last release — is
/// turned away at the hello with both versions named, before any session exists.
#[tokio::test]
async fn an_old_peer_is_refused_with_both_versions() {
    assert_eq!(PROTOCOL_VERSION, 7);
    let a = quic::start(1, NodeRef::new(64, [0, 0, 0]));
    let endpoint = space_net::client_endpoint_for(a.addr()).expect("endpoint");
    for token in ["old-client", "shard:7"] {
        let err = space_net::connect(
            &endpoint,
            a.addr(),
            space_server::net::SERVER_NAME,
            &Hello {
                protocol: 6,
                player_token: token.into(),
            },
        )
        .await
        .expect_err("a v6 peer is refused");
        assert_eq!(
            err,
            space_net::NetError::VersionMismatch {
                seen: 6,
                expected: 7
            },
            "{token}"
        );
    }
    assert_eq!(a.shard.lock().await.stats().clients, 0);
    a.stop();
}
