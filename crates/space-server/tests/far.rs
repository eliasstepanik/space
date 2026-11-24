//! Far replicas (`docs/plan-sharding.md` §6): what a player sees of the
//! shards it is not on, beyond the reach of their border ghosts.
//!
//! Two shards side by side, each sixteen cells wide (the view range is 10 km),
//! stepped in one process on the same ticks and wired together by hand. A
//! client's view is rebuilt from the frames its shard sends, the way a client
//! does it: merged by guid, each sample at the tick it is the state **of**.
//!
//! Every test names the mutation it fails under.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

use space_protocol::far::{self, FarInterest, FarPlayer, FarUpdate};
use space_protocol::{
    EntityGuid, EntitySnapshot, Hello, Message, Origin, OriginKind, ShardId, ShardMap, ShardRegion,
    ORIGIN_COMPONENT, PROTOCOL_VERSION,
};
use space_server::shard::{
    Destination, Shard, FAR_COMPONENTS, FAR_RECEIVE_BYTES, FAR_REFRESH_TICKS, FAR_SILENCE_TICKS,
};
use space_server::ShardConfig;
use space_sim::body::{Body, MassSpec};
use space_sim::shape::ShapeRef;
use space_sim::{BodySpawn, ShipState};
use space_world::{NodeRef, WorldPos};

// ------------------------------------------------------------------ harness

fn listening(id: u32) -> SocketAddr {
    SocketAddr::new(
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        7_100 + u16::try_from(id).expect("a small shard id"),
    )
}

/// Shard 1 owns the sixteen cells west of the border, shard 2 the sixteen
/// east of it (16 384 m each way).
fn regions() -> Vec<(u32, NodeRef)> {
    vec![
        (1, NodeRef::new(60, [0, 0, 0])),
        (2, NodeRef::new(60, [1, 0, 0])),
    ]
}

fn map_of(regions: &[(u32, NodeRef)], epoch: u64) -> ShardMap {
    ShardMap::new(
        epoch,
        regions
            .iter()
            .map(|(id, node)| ShardRegion::new(*node, ShardId::new(*id), listening(*id)))
            .collect(),
    )
}

/// A point `d` metres from the middle of the border (x: into shard 2).
fn at_border(d: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(WorldPos::new([16, 8, 8], [-512.0, 0.0, 0.0]), d)
}

fn border_x(p: WorldPos) -> f64 {
    WorldPos::delta(at_border([0.0; 3]), p)[0]
}

fn moving(pos: WorldPos, vel: [f32; 3]) -> ShipState {
    ShipState {
        vel,
        ..ShipState::at_rest(pos)
    }
}

/// A ball with no damping (nothing pulls on it: the shards run no scene).
fn ball() -> Body {
    let mut b = Body::dynamic(ShapeRef::Ball { radius: 0.4 });
    b.mass = MassSpec::Mass(10.0);
    b.linear_damping = 0.0;
    b.angular_damping = 0.0;
    b
}

fn spawn(shard: &mut Shard, state: ShipState) -> EntityGuid {
    shard
        .spawn_body(
            BodySpawn::new(state, ball()).with_visual(space_sim::Visual(space_scene::visual::BALL)),
        )
        .expect("spawn")
}

/// What crossed between two shards.
#[derive(Default, Clone, Copy)]
struct Link {
    /// Bytes of far updates (`wire_len`).
    bytes: usize,
    /// Bytes of far interests.
    interest: usize,
    /// The largest single [`FarUpdate`].
    largest: usize,
    updates: usize,
}

struct Cluster {
    shards: Vec<Shard>,
    /// Frames for clients: `(shard, session, message)`.
    frames: Vec<(ShardId, u64, Message)>,
    /// Drop every shard-to-shard message (a dead network).
    cut: bool,
    /// Drop every [`FarUpdate`] (and nothing else): a message to a peer is
    /// lost when its queue is full or the peer is being dialled.
    lose_far_updates: bool,
    /// Deliver every [`FarUpdate`] this many ticks late (the ghost frames
    /// and everything else on time).
    far_updates_late: usize,
    /// The far updates held back, with the tick they are due at.
    held_back: Vec<(u64, ShardId, ShardId, Message)>,
    ticks: u64,
    /// Far traffic by `(from, to)`.
    links: HashMap<(u32, u32), Link>,
    /// Bytes of frames by session of shard 1 (encoded, with the header).
    client_bytes: usize,
    largest_frame: usize,
}

impl Cluster {
    fn new() -> Self {
        let map = map_of(&regions(), 1);
        let shards = regions()
            .iter()
            .map(|(id, node)| {
                let mut s = Shard::new(ShardConfig::new(ShardId::new(*id), listening(*id), *node));
                s.set_shard_map(map.clone());
                s
            })
            .collect();
        Self {
            shards,
            frames: Vec::new(),
            cut: false,
            lose_far_updates: false,
            far_updates_late: 0,
            held_back: Vec::new(),
            ticks: 0,
            links: HashMap::new(),
            client_bytes: 0,
            largest_frame: 0,
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

    fn tick(&mut self) {
        for shard in &mut self.shards {
            shard.tick();
        }
        self.route();
    }

    /// Deliver what the shards have queued.
    fn route(&mut self) {
        let mut pending: Vec<(ShardId, ShardId, Message)> = Vec::new();
        for shard in &mut self.shards {
            let from = shard.shard();
            for out in shard.take_outbox() {
                match out.dest {
                    Destination::Client(session) => {
                        if from == ShardId::new(1) {
                            let len = space_protocol::encode_frame(&out.message).len();
                            self.client_bytes += len;
                            self.largest_frame = self.largest_frame.max(len);
                        }
                        self.frames.push((from, session, out.message));
                    }
                    Destination::Shard(to) => pending.push((from, to, out.message)),
                    Destination::Orchestrator => {}
                }
            }
        }
        if self.cut {
            return;
        }
        self.ticks += 1;
        if self.lose_far_updates {
            pending.retain(|(_, _, m)| !matches!(m, Message::FarUpdate(_)));
        }
        if self.far_updates_late > 0 {
            let due = self.ticks + self.far_updates_late as u64;
            let (late, on_time): (Vec<_>, Vec<_>) = pending
                .into_iter()
                .partition(|(_, _, m)| matches!(m, Message::FarUpdate(_)));
            pending = on_time;
            self.held_back
                .extend(late.into_iter().map(|(f, t, m)| (due, f, t, m)));
        }
        let now = self.ticks;
        let (due, later): (Vec<_>, Vec<_>) = std::mem::take(&mut self.held_back)
            .into_iter()
            .partition(|(at, _, _, _)| *at <= now);
        self.held_back = later;
        pending.extend(due.into_iter().map(|(_, f, t, m)| (f, t, m)));
        for (from, to, message) in pending {
            let link = self.links.entry((from.0, to.0)).or_default();
            match &message {
                Message::FarUpdate(update) => {
                    link.bytes += update.wire_len();
                    link.largest = link.largest.max(update.wire_len());
                    link.updates += 1;
                }
                Message::FarInterest(interest) => link.interest += interest.wire_len(),
                _ => {}
            }
            if let Some(target) = self.shards.iter_mut().find(|s| s.shard() == to) {
                target.peer_message(from, message);
            }
        }
    }

    /// One tick in which shard 2 is a moment ahead: it steps, and what it
    /// says is with shard 1 before shard 1 steps the same tick (on a live
    /// stack the two step at the same instants, and which of them is first
    /// changes from tick to tick).
    fn tick_with_2_first(&mut self) {
        self.shard_mut(2).tick();
        self.route();
        self.shard_mut(1).tick();
        self.route();
    }

    fn tick_n(&mut self, n: usize) {
        for _ in 0..n {
            self.tick();
        }
    }

    fn link(&self, from: u32, to: u32) -> Link {
        self.links.get(&(from, to)).copied().unwrap_or_default()
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

/// A player on shard 1 (which owns the spawn point), put at `pos`.
fn player_on_1(c: &mut Cluster, pos: WorldPos) -> (u64, EntityGuid) {
    let (session, ship) = join(c.shard_mut(1), "player-one");
    assert!(c.shard_mut(1).set_state(ship, ShipState::at_rest(pos)));
    (session, ship)
}

/// A player who is already on shard 2 (a shard that does not own the spawn
/// point takes no fresh join: let it in under a map of its own regions).
fn player_on_2(c: &mut Cluster, pos: WorldPos) -> (u64, EntityGuid) {
    let shard = c.shard_mut(2);
    let cluster_map = shard.shard_map().clone();
    let own: Vec<ShardRegion> = cluster_map
        .regions
        .iter()
        .filter(|r| r.shard == shard.shard())
        .cloned()
        .collect();
    shard.set_shard_map(ShardMap::new(cluster_map.epoch, own));
    let (session, ship) = join(shard, "player-two");
    shard.set_shard_map(cluster_map);
    assert!(shard.set_state(ship, ShipState::at_rest(pos)));
    (session, ship)
}

/// One sample of an entity as a client got it.
#[derive(Clone, Copy, Debug)]
struct Seen {
    /// The tick the state is the state of.
    tick: u64,
    /// The tick of the frame that carried it.
    frame: u64,
    state: ShipState,
    /// `None`: the shard simulates it.
    origin: Option<OriginKind>,
}

/// What a client holds of one entity.
#[derive(Default, Debug)]
struct Held {
    components: Vec<(u16, Vec<u8>)>,
    samples: Vec<Seen>,
}

/// A client's view: every entity its frames add up to, by guid.
#[derive(Default)]
struct View {
    held: BTreeMap<EntityGuid, Held>,
    /// Every removal the client was told, with the frame's tick.
    removed: Vec<(u64, EntityGuid)>,
}

impl View {
    fn apply(&mut self, message: &Message) {
        let (tick, entities, removed): (u64, &[EntitySnapshot], &[EntityGuid]) = match message {
            Message::Snapshot(s) => (s.tick, &s.entities, &[]),
            Message::Delta(d) => (d.tick, &d.entities, &d.removed),
            _ => return,
        };
        for guid in removed {
            if !entities.iter().any(|e| e.guid == *guid) {
                self.held.remove(guid);
                self.removed.push((tick, *guid));
            }
        }
        for e in entities {
            let held = self.held.entry(e.guid).or_default();
            for (id, bytes) in &e.components {
                held.components.retain(|(i, _)| i != id);
                if !bytes.is_empty() {
                    held.components.push((*id, bytes.clone()));
                }
            }
            let state = held
                .components
                .iter()
                .find(|(id, _)| *id == 0)
                .map(|(_, b)| space_sim::wire::decode_ship_state(b).expect("a ship state"))
                .expect("every entity has a ShipState");
            let origin = held
                .components
                .iter()
                .find(|(id, _)| *id == ORIGIN_COMPONENT)
                .map(|(_, b)| Origin::decode(b).expect("an origin"));
            let at = match origin {
                Some(o) if o.kind == OriginKind::Far => o.tick,
                _ => tick,
            };
            held.samples.push(Seen {
                tick: at,
                frame: tick,
                state,
                origin: origin.map(|o| o.kind),
            });
        }
    }

    fn take(&mut self, c: &mut Cluster, shard: u32, session: u64) {
        for m in c.frames_for(shard, session) {
            self.apply(&m);
        }
    }

    fn ids(&self, guid: EntityGuid) -> Vec<u16> {
        let mut ids: Vec<u16> = self.held[&guid]
            .components
            .iter()
            .map(|(id, _)| *id)
            .collect();
        ids.sort_unstable();
        ids
    }

    fn origin(&self, guid: EntityGuid) -> Option<OriginKind> {
        self.held
            .get(&guid)
            .and_then(|h| h.samples.last())
            .and_then(|s| s.origin)
    }
}

// -------------------------------------------------------------------- tests

/// A body 300 m inside the other shard — five times the ghost margin from the
/// border — is in the view of a player near that border: once, as a far
/// replica, with what it takes to draw it and nothing else, and in nobody's
/// physics here.
///
/// Mutations this catches: not running the far pass (the body is missing, as
/// it was before this phase); putting the replica into the simulation (it
/// would be a ghost: `entity_of` finds it); sending the replicated list as it
/// is (`PosResidual` and, for a ship, `Owner` and `ShipControls` arrive).
#[test]
fn a_body_far_inside_another_shard_is_seen_as_a_far_replica() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    let body = spawn(
        c.shard_mut(2),
        ShipState::at_rest(at_border([300.0, 5.0, 0.0])),
    );
    let mut view = View::default();
    for _ in 0..40 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    assert!(c.shard(2).sim().ship_state(body).is_some());
    assert_eq!(
        c.shard(1).ghost_owner(body),
        None,
        "beyond the ghost margin"
    );
    assert!(
        c.shard(1).sim().entity_of(body).is_none(),
        "a far replica is in no world: nothing can collide with it"
    );
    let far = c.shard(1).far_list();
    assert_eq!(far.len(), 1);
    assert_eq!((far[0].guid, far[0].from), (body, ShardId::new(2)));
    assert_eq!(c.shard(1).stats().far, 1);
    assert_eq!(c.shard(2).far_sent_to(ShardId::new(1)), 1);
    // The client: there, as a far replica, drawn from its shape and model.
    assert_eq!(view.origin(body), Some(OriginKind::Far));
    assert_eq!(view.ids(body), vec![0, 4, 5, ORIGIN_COMPONENT]);
    assert!(view.ids(body)[..3]
        .iter()
        .all(|id| FAR_COMPONENTS.contains(id)));
    assert_eq!(
        view.held[&body].samples.len(),
        1,
        "a body at rest is said once: {:?}",
        view.held[&body].samples
    );
    assert!(view.removed.is_empty());
    // Nothing of shard 1's is a far replica on shard 2 (it has no players).
    assert!(c.shard(2).far_list().is_empty());
}

/// How often a far body's motion reaches a client falls with its distance in
/// the steps of the table — 10, 5 and 1 times a second at 150 m, 600 m and
/// 3 km — and beyond the 10 km of the view range it is not sent at all. Each
/// sample is the state of the tick it says, exactly: the simulating shard's
/// tick, a tick or two before the frame.
///
/// Mutations this catches: no thinning between shards or towards the client
/// (`far::interval` -> 1: thirty samples a second at every distance); no
/// range cut; stamping a sample with the tick of the frame that carries it
/// (it is then a tick or two of motion off where it says it is).
#[test]
fn far_bodies_are_sent_at_the_rate_of_their_distance_step() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    // Drifting sideways at 1 m/s: the distance stays what it is.
    let at = |d: f64| moving(at_border([d - 70.0, 0.0, 0.0]), [0.0, 0.0, 1.0]);
    let bodies = [
        (spawn(c.shard_mut(2), at(150.0)), 1u8),
        (spawn(c.shard_mut(2), at(600.0)), 2),
        (spawn(c.shard_mut(2), at(3_000.0)), 3),
    ];
    let beyond = spawn(c.shard_mut(2), at(11_500.0));
    let mut view = View::default();
    let mut truth: HashMap<(EntityGuid, u64), WorldPos> = HashMap::new();
    for _ in 0..360 {
        c.tick();
        let tick = c.shard(2).stats().tick;
        for (guid, _) in bodies {
            let state = c.shard(2).sim().ship_state(guid).expect("simulated");
            truth.insert((guid, tick), state.pos);
        }
        view.take(&mut c, 1, session);
    }
    assert!(!view.held.contains_key(&beyond), "beyond the view range");
    assert_eq!(c.shard(1).far_list().len(), 3);
    for (guid, step) in bodies {
        // (The numbers, not `far::interval`: a table that said "every tick"
        // for every step must fail here.)
        let every: u64 = [1, 3, 6, 30][usize::from(step)];
        assert_eq!(far::interval(step), every);
        let samples = &view.held[&guid].samples;
        // Over the last 300 ticks (the first sixty: the shards meet).
        let last = samples.last().expect("seen").tick;
        let counted: Vec<&Seen> = samples.iter().filter(|s| s.tick + 300 > last).collect();
        let expected = 300 / every;
        assert!(
            (counted.len() as u64).abs_diff(expected) <= 1,
            "step {step}: {} samples in 300 ticks, {expected} expected",
            counted.len()
        );
        for w in counted.windows(2) {
            assert_eq!(w[1].tick - w[0].tick, every, "step {step}: evenly spaced");
        }
        for s in counted {
            assert_eq!(s.origin, Some(OriginKind::Far));
            assert!(
                (1..=2).contains(&(s.frame - s.tick)),
                "a sample of tick {} came in the frame of tick {}",
                s.tick,
                s.frame
            );
            assert_eq!(
                Some(&s.state.pos),
                truth.get(&(guid, s.tick)),
                "step {step}: the sample of tick {} is the state of that tick",
                s.tick
            );
        }
    }
}

/// A far body at rest costs its first sample and then nothing but its name in
/// the refresh — on the link between the shards and towards the client — and
/// when it moves again its first sample is the start of the motion.
///
/// Mutations this catches: sending an entity that did not change (dropping
/// the "nothing to say" check in `far_send`, or the `touched_after` test in
/// `build_frame`); waiting out the step's spacing before the first sample of
/// a body that starts to move (it then appears a second into its motion).
#[test]
fn a_far_body_at_rest_costs_a_keep_alive_and_wakes_at_once() {
    let mut c = Cluster::new();
    let (session, ship) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    let body = spawn(
        c.shard_mut(2),
        ShipState::at_rest(at_border([3_000.0, 0.0, 0.0])),
    );
    let mut view = View::default();
    for _ in 0..60 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    assert_eq!(view.held[&body].samples.len(), 1);
    let (before, client_before) = (c.link(2, 1), c.client_bytes);
    for _ in 0..600 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    let link = c.link(2, 1);
    let refreshes = 600 / FAR_REFRESH_TICKS as usize;
    assert_eq!(
        link.updates - before.updates,
        refreshes,
        "only the refreshes went out"
    );
    // A refresh that names one entity: four empty lists and twelve bytes.
    assert_eq!(link.bytes - before.bytes, refreshes * (4 * 8 + 12));
    assert_eq!(
        view.held[&body].samples.len(),
        1,
        "the client was sent nothing"
    );
    // And the client's frames were heartbeats: 64 bytes a tick.
    let per_tick = (c.client_bytes - client_before) / 600;
    assert!(per_tick <= 64, "{per_tick} B a tick for a world at rest");
    let _ = ship;
    // It moves: seen on the very next frame, a tick's motion from where it
    // rested, although its step allows one sample a second.
    let rest = c.shard(2).sim().ship_state(body).expect("there").pos;
    assert!(c
        .shard_mut(2)
        .set_state(body, moving(rest, [0.0, 0.0, 6.0])));
    c.tick_n(2);
    view.take(&mut c, 1, session);
    let samples = &view.held[&body].samples;
    assert_eq!(samples.len(), 2, "said at once: {samples:?}");
    let moved = WorldPos::distance(rest, samples[1].state.pos);
    assert!(moved <= 0.21, "first seen {moved:.2} m into its motion");
    c.tick_n(40);
    view.take(&mut c, 1, session);
    let samples = &view.held[&body].samples;
    assert_eq!(samples.len(), 3, "then once a second");
    assert_eq!(samples[2].tick - samples[1].tick, 30);
}

/// The bound, held to numbers: sixty balls tumbling 600 m off cost a client
/// and the link between the shards what sixty samples five times a second
/// cost, not thirty times; four hundred coming into view at once never make
/// a far update or a frame larger than its bound.
///
/// Mutations this catches: the finest step for everything (six times the
/// bytes); filling a `FarUpdate` without looking at the budget (one message
/// of over 100 kB).
#[test]
fn the_cost_of_far_replicas_is_bounded() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    const N: usize = 60;
    for i in 0..N {
        let pos = at_border([530.0, 0.0, 3.0 * i as f64]);
        spawn(c.shard_mut(2), moving(pos, [0.0, 1.0, 0.0]));
    }
    let mut view = View::default();
    for _ in 0..90 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    assert_eq!(c.shard(1).far_list().len(), N);
    let (before, client_before) = (c.link(2, 1), c.client_bytes);
    c.tick_n(300);
    // One moving far body's sample towards the client: the entry (guid,
    // count), its ShipState and its origin.
    let sample = 12 + 8 + (2 + 8 + 76) + (2 + 8 + 14);
    let rate = 5; // a second, at 600 m
    let seconds = 10;
    let client = c.client_bytes - client_before;
    let bound = seconds * (N * sample * rate + 30 * 80);
    assert!(
        client <= bound,
        "{client} B to the client, the bound is {bound}"
    );
    assert!(client * 2 > bound, "and it was sent: {client} B");
    // Between the shards: the same samples without the origin, the names
    // every two seconds, the players three times a second.
    let peer = c.link(2, 1).bytes - before.bytes;
    let peer_bound = seconds * (N * (12 + 8 + 2 + 8 + 76) * rate + 30 * 32) + 5 * (32 + N * 12);
    assert!(
        c.link(2, 1).interest <= 16 * 400 / 10,
        "a shard without players says so, briefly"
    );
    assert!(
        peer <= peer_bound,
        "{peer} B between the shards, the bound is {peer_bound}"
    );
    assert!(
        c.link(1, 2).interest < 12 * 1024,
        "what shard 1 says: its players"
    );
    assert_eq!(
        c.link(1, 2).bytes,
        0,
        "shard 2 has no players to show anything"
    );
    assert!(c.largest_frame <= 64 * 1024 + 8);

    // Four hundred more, all at once.
    for i in 0..400 {
        let pos = at_border([900.0 + 2.0 * (i % 20) as f64, 0.0, 2.0 * (i / 20) as f64]);
        spawn(c.shard_mut(2), ShipState::at_rest(pos));
    }
    let before = c.link(2, 1);
    c.tick_n(60);
    let after = c.link(2, 1);
    assert_eq!(c.shard(1).far_list().len(), N + 400);
    assert!(
        after.largest <= far::MAX_FAR_UPDATE_BYTES,
        "a far update of {} B",
        after.largest
    );
    assert!(after.updates - before.updates >= 3, "split over several");
    assert!(c.largest_frame <= 64 * 1024 + 8, "{} B", c.largest_frame);
    assert_eq!(c.shard(1).stats().far_refused, 0);
}

/// A body flies from far inside the other shard, through the ghost margin and
/// across the border to the player: the client sees one entity the whole way
/// — far replica, then ghost, then its own shard's body — never told it went
/// away, and every sample lies where the one before and the velocity put it.
///
/// Mutation this catches: stamping a far sample with its frame's tick (the
/// first ghost sample is then a tick's flight — a third of a metre — off).
#[test]
fn from_far_replica_to_ghost_to_local_without_a_jump() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 20.0]));
    let speed = 10.0f32;
    let body = spawn(
        c.shard_mut(2),
        moving(at_border([260.0, 0.0, 0.0]), [-speed, 0.0, 0.0]),
    );
    let mut view = View::default();
    for _ in 0..1_200 {
        c.tick();
        view.take(&mut c, 1, session);
        if c.shard(1)
            .sim()
            .ship_state(body)
            .is_some_and(|s| border_x(s.pos) < -40.0)
        {
            break;
        }
    }
    assert!(c.shard(1).sim().ship_state(body).is_some(), "handed over");
    assert!(
        view.removed.is_empty(),
        "never told it went away: {:?}",
        view.removed
    );
    let samples = &view.held[&body].samples;
    let kinds: Vec<Option<OriginKind>> = samples.iter().map(|s| s.origin).collect();
    let mut order = kinds.clone();
    order.dedup();
    assert_eq!(
        order,
        vec![Some(OriginKind::Far), Some(OriginKind::Ghost), None],
        "far, then ghost, then local"
    );
    let per_tick = f64::from(speed) * f64::from(space_sim::dt());
    let mut worst = 0.0f64;
    for w in samples.windows(2) {
        assert!(w[1].tick > w[0].tick, "samples in order: {w:?}");
        let ticks = (w[1].tick - w[0].tick) as f64;
        let flown = border_x(w[0].state.pos) - border_x(w[1].state.pos);
        worst = worst.max((flown - ticks * per_tick).abs());
    }
    assert!(
        worst < 0.02,
        "a sample {worst:.3} m off where its motion put it ({per_tick:.3} m a tick)"
    );
    assert!(
        c.shard(1).far_list().is_empty(),
        "no far replica of what is ours"
    );
    assert_eq!(c.shard(2).far_sent_to(ShardId::new(1)), 0);
}

/// A ghost shows its owner's state **of this shard's tick**, whichever of the
/// two shards stepped first: a client's samples of a fast ghost lie on the
/// owner's track, each at the tick its frame names.
///
/// Mutation this catches: moving every ghost on one tick after every step
/// whatever its update said (dropping the `shown` test in `reckon_ghosts`):
/// in the ticks in which the owner was first the ghost is then a whole tick
/// of motion ahead — a metre at this speed — and falls back when the order
/// turns round.
#[test]
fn a_ghost_shows_its_owners_state_of_the_tick_whoever_stepped_first() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-20.0, 0.0, 30.0]));
    let speed = 20.0f32;
    let body = spawn(
        c.shard_mut(2),
        moving(at_border([30.0, 0.0, -50.0]), [0.0, 0.0, speed]),
    );
    let mut view = View::default();
    let mut truth: HashMap<u64, WorldPos> = HashMap::new();
    for i in 0..240 {
        // Ten ticks one way, ten the other.
        if (i / 10) % 2 == 0 {
            c.tick();
        } else {
            c.tick_with_2_first();
        }
        let state = c.shard(2).sim().ship_state(body).expect("simulated");
        truth.insert(c.shard(2).stats().tick, state.pos);
        view.take(&mut c, 1, session);
    }
    assert_eq!(
        c.shard(1).ghost_owner(body),
        Some(ShardId::new(2)),
        "a ghost"
    );
    let samples: Vec<&Seen> = view.held[&body]
        .samples
        .iter()
        .filter(|s| s.origin == Some(OriginKind::Ghost))
        .collect();
    assert!(samples.len() > 150, "{} ghost samples", samples.len());
    let mut worst = 0.0f64;
    for s in samples {
        let Some(at) = truth.get(&s.tick) else {
            continue;
        };
        worst = worst.max(WorldPos::distance(*at, s.state.pos));
    }
    assert!(
        worst < 0.01,
        "a ghost sample {worst:.3} m off its owner's state of that tick (a tick is {:.2} m)",
        f64::from(speed) * f64::from(space_sim::dt())
    );
}

/// The other way round: a body of the player's own shard is pushed across the
/// border and on, out of the ghost margin — the client keeps it, as a ghost
/// and then as a far replica, and on the shard it left there is one copy of
/// it at any time. Also when the far updates arrive **late** (three ticks
/// behind the ghost frame that withdraws the ghost): the withdrawn ghost
/// stays, as a far replica, until its owner's word is in.
///
/// Mutation this catches (the late run): removing a withdrawn ghost outright
/// instead of keeping it for its owner to confirm (`remove_ghost` in
/// `on_ghost_update`): the client is told it is gone and three ticks later
/// that it is there.
#[test]
fn from_local_to_ghost_to_far_replica_without_a_gap() {
    local_to_ghost_to_far(0);
    local_to_ghost_to_far(3);
}

/// A far update that was lost is made good: what the next refresh names and
/// the receiver does not hold, it asks for and is sent in full — also a body
/// at rest, which nothing else would ever say again.
///
/// Mutation this catches: not noting what a refresh names and we lack
/// (`far_refreshed`): the sender believes we hold the body, the body never
/// changes, and it stays invisible here for good.
#[test]
fn a_lost_far_update_is_asked_for_again() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    c.tick_n(15);
    c.lose_far_updates = true;
    let body = spawn(
        c.shard_mut(2),
        ShipState::at_rest(at_border([400.0, 0.0, 0.0])),
    );
    c.tick_n(20);
    assert_eq!(c.shard(2).far_sent_to(ShardId::new(1)), 1, "sent, and lost");
    assert!(c.shard(1).far_list().is_empty());
    c.lose_far_updates = false;
    let mut view = View::default();
    for _ in 0..FAR_REFRESH_TICKS + 25 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    let far: Vec<EntityGuid> = c.shard(1).far_list().iter().map(|f| f.guid).collect();
    assert_eq!(far, vec![body], "asked for, and sent again");
    assert_eq!(view.origin(body), Some(OriginKind::Far));
    assert_eq!(view.ids(body), vec![0, 4, 5, ORIGIN_COMPONENT], "in full");
}

fn local_to_ghost_to_far(far_updates_late: usize) {
    let mut c = Cluster::new();
    c.far_updates_late = far_updates_late;
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 20.0]));
    let body = spawn(
        c.shard_mut(1),
        moving(at_border([-30.0, 0.0, 0.0]), [12.0, 0.0, 0.0]),
    );
    let mut view = View::default();
    for _ in 0..900 {
        c.tick();
        view.take(&mut c, 1, session);
        let copies = usize::from(c.shard(1).sim().entity_of(body).is_some())
            + c.shard(1)
                .far_list()
                .iter()
                .filter(|f| f.guid == body)
                .count();
        assert_eq!(copies, 1, "one copy of it on shard 1");
        if c.shard(2)
            .sim()
            .ship_state(body)
            .is_some_and(|s| border_x(s.pos) > 200.0)
        {
            break;
        }
    }
    assert!(view.removed.is_empty(), "{:?}", view.removed);
    let mut order: Vec<Option<OriginKind>> =
        view.held[&body].samples.iter().map(|s| s.origin).collect();
    order.dedup();
    assert_eq!(
        order,
        vec![None, Some(OriginKind::Ghost), Some(OriginKind::Far)]
    );
    let last = view.held[&body].samples.last().expect("seen");
    assert!(border_x(last.state.pos) > 190.0, "seen to the end");
}

/// Players see each other across shards beyond ghost range: a far player is
/// a ship with the player model — and without its owner or its controls.
///
/// Mutation this catches: adding `Owner` or `ShipControls` to
/// `FAR_COMPONENTS` (a session id of another shard reaches a client, where it
/// could be mistaken for a session of its own shard).
#[test]
fn a_player_on_another_shard_is_seen_as_a_far_player() {
    let mut c = Cluster::new();
    let (session_1, ship_1) = player_on_1(&mut c, at_border([-100.0, 0.0, 0.0]));
    let (session_2, ship_2) = player_on_2(&mut c, at_border([120.0, 0.0, 0.0]));
    let (mut view_1, mut view_2) = (View::default(), View::default());
    for _ in 0..40 {
        c.tick();
        view_1.take(&mut c, 1, session_1);
        view_2.take(&mut c, 2, session_2);
    }
    for (view, theirs, mine) in [(&view_1, ship_2, ship_1), (&view_2, ship_1, ship_2)] {
        assert_eq!(view.origin(theirs), Some(OriginKind::Far));
        assert_eq!(
            view.ids(theirs),
            vec![0, 5, ORIGIN_COMPONENT],
            "state, model, origin"
        );
        let model = view.held[&theirs]
            .components
            .iter()
            .find(|(id, _)| *id == 5)
            .map(|(_, b)| space_sim::body::decode_visual(b).expect("a visual"))
            .expect("a model");
        assert_eq!(model.0, space_scene::visual::PLAYER);
        assert_eq!(view.origin(mine), None, "the own ship is the shard's own");
    }
    // It flies: the other client sees it move, ten times a second at 220 m.
    let start = c.shard(2).sim().ship_state(ship_2).expect("there").pos;
    assert!(c
        .shard_mut(2)
        .set_state(ship_2, moving(start, [0.0, 0.0, 9.0])));
    let seen_before = view_1.held[&ship_2].samples.len();
    for _ in 0..90 {
        c.tick();
        view_1.take(&mut c, 1, session_1);
    }
    let seen = view_1.held[&ship_2].samples.len() - seen_before;
    assert!((29..=31).contains(&seen), "{seen} samples in three seconds");
}

/// What a peer sends through the far messages is bounded, and cannot touch
/// what this shard simulates or holds a ghost of.
///
/// Mutations this catches, one per block: dropping the `entity_of` test in
/// `far_accept` (a peer moves our player's ship in our clients' eyes — the
/// log entry of the ship is overwritten); not checking `from` against the
/// connection; no size bound; no rate bound; letting a component outside
/// `FAR_COMPONENTS` through on receipt; no cap on the players named.
#[test]
fn what_a_peer_sends_is_bounded_and_cannot_overwrite_local_entities() {
    let mut c = Cluster::new();
    let (session, ship) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    let near = spawn(
        c.shard_mut(2),
        ShipState::at_rest(at_border([20.0, 0.0, 0.0])),
    );
    let mut view = View::default();
    for _ in 0..30 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    assert_eq!(
        c.shard(1).ghost_owner(near),
        Some(ShardId::new(2)),
        "a ghost"
    );
    let two = ShardId::new(2);
    let elsewhere =
        space_sim::wire::encode_ship_state(&ShipState::at_rest(at_border([5_000.0, 0.0, 0.0])));
    // (A tick from the future: the receiver holds it to its own.)
    let update = |new: Vec<EntitySnapshot>| FarUpdate {
        from: two,
        tick: u64::MAX,
        new,
        changed: Vec::new(),
        removed: Vec::new(),
        refresh: false,
        held: Vec::new(),
        more: false,
    };

    // Our own ship, and a ghost we hold: not through this door.
    let forged = update(vec![
        EntitySnapshot::new(ship, vec![(0, elsewhere.clone())]),
        EntitySnapshot::new(near, vec![(0, elsewhere.clone())]),
    ]);
    let before = c.shard(1).sim().state_of(near).expect("the ghost");
    c.shard_mut(1).peer_message(two, Message::FarUpdate(forged));
    assert!(
        c.shard(1).far_list().is_empty(),
        "refused at the door, not swept up later"
    );
    c.tick();
    view.take(&mut c, 1, session);
    assert!(c.shard(1).far_list().is_empty());
    assert_eq!(c.shard(1).sim().state_of(near), Some(before));
    for guid in [ship, near] {
        let last = view.held[&guid].samples.last().expect("seen");
        assert!(
            border_x(last.state.pos) < 100.0,
            "the client still sees it where it is"
        );
        assert_ne!(last.origin, Some(OriginKind::Far));
    }

    // A message that says it is another shard's.
    let stranger = EntityGuid::new(two, 4_000);
    let mut lying = update(vec![EntitySnapshot::new(
        stranger,
        vec![(0, elsewhere.clone())],
    )]);
    lying.from = ShardId::new(1);
    c.shard_mut(1).peer_message(two, Message::FarUpdate(lying));
    assert_eq!(c.shard(1).stats().far_refused, 1);
    // A shard the map does not name is not heard at all.
    let unknown = update(vec![EntitySnapshot::new(
        stranger,
        vec![(0, elsewhere.clone())],
    )]);
    c.shard_mut(1)
        .peer_message(ShardId::new(9), Message::FarUpdate(unknown));
    assert!(c.shard(1).far_list().is_empty());

    // Private and personal components do not get in.
    let dressed = update(vec![EntitySnapshot::new(
        stranger,
        vec![
            (0, elsewhere.clone()),
            (2, space_protocol::codec::encode_raw(&77u64)),
            (10, vec![1; 14]),
            (u16::MAX, b"somebody's token".to_vec()),
        ],
    )]);
    c.shard_mut(1)
        .peer_message(two, Message::FarUpdate(dressed));
    c.tick();
    view.take(&mut c, 1, session);
    assert_eq!(view.ids(stranger), vec![0, ORIGIN_COMPONENT]);

    // One update over the size bound; then more bytes than a second takes.
    let big = |serial: u64, len: usize| {
        update(vec![EntitySnapshot::new(
            EntityGuid::new(two, serial),
            vec![(0, elsewhere.clone()), (4, vec![0; len])],
        )])
    };
    // (Forty entities of a kilobyte: no one of them is too large, the
    // update is.)
    let refused = c.shard(1).stats().far_refused;
    let held = c.shard(1).far_list().len();
    let many = update(
        (0..40)
            .map(|i| {
                EntitySnapshot::new(
                    EntityGuid::new(two, 5_000 + i),
                    vec![(0, elsewhere.clone()), (4, vec![0; 1024])],
                )
            })
            .collect(),
    );
    assert!(many.wire_len() > far::MAX_FAR_UPDATE_BYTES);
    c.shard_mut(1).peer_message(two, Message::FarUpdate(many));
    assert_eq!(
        c.shard(1).stats().far_refused,
        refused + 1,
        "over the size bound"
    );
    assert_eq!(c.shard(1).far_list().len(), held, "none of it was taken");
    // One entity over its own bound, in an update that is not.
    c.shard_mut(1)
        .peer_message(two, Message::FarUpdate(big(5_500, 20 * 1024)));
    assert_eq!(c.shard(1).stats().far_refused, refused + 2, "an entity");
    assert_eq!(c.shard(1).far_list().len(), held);
    let refused = refused + 1;
    let each = 12 * 1024;
    let fit = FAR_RECEIVE_BYTES / (each + 200);
    for i in 0..fit as u64 + 20 {
        c.shard_mut(1)
            .peer_message(two, Message::FarUpdate(big(6_000 + i, each)));
    }
    let stats = c.shard(1).stats();
    assert!(
        stats.far_refused >= refused + 1 + 15,
        "over the rate: {stats:?}"
    );
    assert!(c.shard(1).far_list().len() <= fit + 2);

    // An interest that names more players than a shard has.
    let refused = c.shard(1).stats().far_refused;
    let crowd = FarInterest {
        from: two,
        tick: 0,
        players: vec![FarPlayer::at(at_border([0.0; 3])); far::MAX_FAR_PLAYERS + 1],
        missing: Vec::new(),
    };
    c.shard_mut(1)
        .peer_message(two, Message::FarInterest(crowd));
    assert_eq!(c.shard(1).stats().far_refused, refused + 1);
}

/// Far replicas are soft state: a shard that falls silent loses them here
/// within the timeout, a message that was dropped is asked for again, and a
/// shard the map no longer names is forgotten at once — and when the entity
/// is back it is there once.
///
/// Mutation this catches: no silence rule (the replicas of a dead shard stay
/// for ever). (After the cut the sender starts over by itself: it has not
/// heard where our players are for three seconds either. A single lost
/// update is `a_lost_far_update_is_asked_for_again`.)
#[test]
fn far_replicas_of_a_silent_or_departed_shard_go_and_come_back_once() {
    let mut c = Cluster::new();
    let (session, _) = player_on_1(&mut c, at_border([-70.0, 0.0, 0.0]));
    let resting = spawn(
        c.shard_mut(2),
        ShipState::at_rest(at_border([400.0, 0.0, 0.0])),
    );
    let drifting = spawn(
        c.shard_mut(2),
        moving(at_border([500.0, 0.0, 0.0]), [0.0, 0.0, 0.5]),
    );
    let mut view = View::default();
    for _ in 0..40 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    assert_eq!(c.shard(1).far_list().len(), 2);

    // The network dies.
    c.cut = true;
    for _ in 0..FAR_SILENCE_TICKS + 2 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    assert!(c.shard(1).far_list().is_empty(), "gone within the timeout");
    assert!(!view.held.contains_key(&resting) && !view.held.contains_key(&drifting));
    assert_eq!(
        view.removed.len(),
        2,
        "the client was told: {:?}",
        view.removed
    );

    // It comes back: both are there again, once — the one at rest too,
    // which its shard believes we hold.
    c.cut = false;
    for _ in 0..FAR_REFRESH_TICKS + 30 {
        c.tick();
        view.take(&mut c, 1, session);
    }
    let far: Vec<EntityGuid> = c.shard(1).far_list().iter().map(|f| f.guid).collect();
    assert_eq!(far, vec![resting, drifting]);
    assert_eq!(view.origin(resting), Some(OriginKind::Far));
    assert_eq!(view.origin(drifting), Some(OriginKind::Far));

    // The map gives shard 2's region to a shard 3 (shard 2 died and was
    // replaced): what shard 2 showed is forgotten on the next tick.
    let replaced = vec![regions()[0], (3, regions()[1].1)];
    c.shard_mut(1).apply_epoch(map_of(&replaced, 2));
    c.cut = true; // (shard 2 is dead)
    c.tick();
    view.take(&mut c, 1, session);
    assert!(c.shard(1).far_list().is_empty());
    assert!(!view.held.contains_key(&resting));

    // A scene body of another scene revision is never kept.
    let stale = space_server::border::scene_guid(ShardId::new(3), 5, 0);
    let state =
        space_sim::wire::encode_ship_state(&ShipState::at_rest(at_border([300.0, 0.0, 0.0])));
    let update = FarUpdate {
        from: ShardId::new(3),
        tick: c.shard(1).stats().tick,
        new: vec![EntitySnapshot::new(stale, vec![(0, state)])],
        changed: Vec::new(),
        removed: Vec::new(),
        refresh: false,
        held: Vec::new(),
        more: false,
    };
    c.shard_mut(1)
        .peer_message(ShardId::new(3), Message::FarUpdate(update));
    assert!(
        c.shard(1).far_list().is_empty(),
        "a body of another scene revision"
    );
    // ... and one of this revision is dropped when the scene changes.
    let current = space_server::border::scene_guid(ShardId::new(3), 0, 1);
    let state =
        space_sim::wire::encode_ship_state(&ShipState::at_rest(at_border([300.0, 0.0, 0.0])));
    let update = FarUpdate {
        from: ShardId::new(3),
        tick: c.shard(1).stats().tick,
        new: vec![EntitySnapshot::new(current, vec![(0, state)])],
        changed: Vec::new(),
        removed: Vec::new(),
        refresh: false,
        held: Vec::new(),
        more: false,
    };
    c.shard_mut(1)
        .peer_message(ShardId::new(3), Message::FarUpdate(update));
    assert_eq!(c.shard(1).far_list().len(), 1);
    assert!(c.shard_mut(1).set_scene(space_protocol::SceneState::new(
        space_protocol::Scene::Galaxy.id(),
        1
    )));
    c.tick();
    assert!(c.shard(1).far_list().is_empty(), "the scene changed");
}
