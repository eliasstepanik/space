//! Seamless crossing, on the shards' side (`docs/plan-sharding.md` §5): what
//! two shards on **one cluster clock** do with a player's ship that flies
//! from one into the other — checked tick for tick against a ship that never
//! changed shard.
//!
//! The shards are stepped by hand and their messages carried between them by
//! the test, so the order of everything is the test's to choose: a commit
//! that arrives within its tick, one that arrives ticks late, one that
//! arrives before the accepting shard has stepped that far.
//!
//! | test | what it pins |
//! |---|---|
//! | [`a_ship_handed_over_flies_as_if_it_never_changed_shard`] | H2, H4, H5 on the cluster clock: one simulator at every tick, no tick lost, no input lost or applied twice |
//! | [`a_late_commit_is_caught_up_with_the_inputs_of_the_ticks_it_missed`] | the catch-up, and the inputs a player sends to both shards |
//! | [`a_commit_that_arrives_early_waits_for_its_tick`] | no double step |
//! | [`a_standing_players_ship_is_not_handed_over_when_another_flies_past`] | a player's ship moves for where it is, not with an island |
//! | [`a_players_handoff_is_not_called_off_by_bodies_going_the_other_way`] | the crossing rule and a refused body never cost a player its handoff |
//! | [`a_refused_body_does_not_call_off_the_ship_it_travels_with`] | the bodies of a player's island travel with the ship; the ship waits for none of them |
//! | [`shards_started_seconds_apart_count_the_same_ticks`] | the cluster clock, with real tick loops |

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

use space_protocol::{
    EntityGuid, InputTick, Message, NeighbourStatus, PlayerInput, SceneClock, ShardId, ShardMap,
    ShardRegion,
};
use space_server::clock::TickClock;
use space_server::handoff::HANDOFF_LEAD;
use space_server::shard::{Destination, Shard};
use space_server::ShardConfig;
use space_sim::body::{Body, MassSpec};
use space_sim::shape::ShapeRef;
use space_sim::{BodySpawn, ShipState, SimTick};
use space_world::{NodeRef, WorldPos};

/// The cluster's tick origin in these tests.
const ORIGIN: i64 = 1_763_900_000_000_000;
/// The tick every shard is put on the clock at.
const START: SimTick = 1_000;
/// How far ahead of the newest tick a client tags its inputs
/// (`space_client::link::INPUT_LEAD`).
const INPUT_LEAD: SimTick = 3;

fn listening(n: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 7_200 + n)
}

/// `d` metres from the middle of the border between cell 0 and cell 1
/// (x = 512 m): `d[0] < 0` is shard 1's ground, `d[0] > 0` shard 2's.
fn at_border(d: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(WorldPos::new([1, 0, 0], [-512.0, 0.0, 0.0]), d)
}

fn border_x(p: WorldPos) -> f64 {
    WorldPos::delta(at_border([0.0; 3]), p)[0]
}

/// Shards stepped by hand on one cluster clock.
struct Cluster {
    shards: Vec<Shard>,
    /// Frames for clients: `(sending shard, session, message)`.
    frames: Vec<(ShardId, u64, Message)>,
    /// Shard-to-shard messages held back: `(deliver at round, from, to, message)`.
    held: Vec<(u64, ShardId, ShardId, Message)>,
    /// How many rounds a `HandoffCommit` is held back.
    commit_delay: u64,
    round: u64,
}

impl Cluster {
    /// Shard 1 owns cell 0, shard 2 cell 1; both on the cluster clock.
    fn pair() -> Self {
        Self::of(&[
            (1, NodeRef::new(64, [0, 0, 0])),
            (2, NodeRef::new(64, [1, 0, 0])),
        ])
    }

    /// One shard that owns both cells: where nothing is ever handed over.
    fn single() -> Self {
        Self::of(&[(1, NodeRef::new(63, [0, 0, 0]))])
    }

    fn of(regions: &[(u32, NodeRef)]) -> Self {
        let map = ShardMap::new(
            1,
            regions
                .iter()
                .map(|(id, node)| ShardRegion::new(*node, ShardId::new(*id), listening(*id as u16)))
                .collect(),
        );
        let shards = regions
            .iter()
            .map(|(id, node)| {
                let mut shard = Shard::new(ShardConfig::new(
                    ShardId::new(*id),
                    listening(*id as u16),
                    *node,
                ));
                shard.start_clock(SceneClock::cluster(ORIGIN), START);
                shard.set_shard_map(map.clone());
                shard
            })
            .collect();
        Self {
            shards,
            frames: Vec::new(),
            held: Vec::new(),
            commit_delay: 0,
            round: 0,
        }
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

    /// One tick of every shard, in id order, then one round of delivery.
    fn tick(&mut self) {
        for shard in &mut self.shards {
            shard.tick();
        }
        self.round += 1;
        self.deliver();
    }

    /// The shards that simulate `guid` now.
    fn simulators(&self, guid: EntityGuid) -> Vec<ShardId> {
        self.shards
            .iter()
            .filter(|s| s.sim().ship_state(guid).is_some())
            .map(Shard::shard)
            .collect()
    }

    fn state(&self, guid: EntityGuid) -> Option<ShipState> {
        self.shards.iter().find_map(|s| s.sim().ship_state(guid))
    }

    /// The frames sent to `(shard, session)`, draining them.
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
    /// The shard a redirect named, and the session there.
    next: Option<(ShardId, u64)>,
    redirects: usize,
    /// The first frame of each shard after the first, by shard: `(tick,
    /// input_ack)`.
    first_frames: HashMap<ShardId, (u64, u64)>,
}

impl Pilot {
    fn join(c: &mut Cluster, token: &str) -> Self {
        let on = ShardId::new(1);
        let welcome = c
            .shard_mut(on)
            .join(&space_net::hello(token))
            .expect("shard 1 owns the spawn point");
        let ship = c.shard(on).ship_of(welcome.session).expect("a ship");
        Self {
            token: token.into(),
            ship,
            on,
            session: welcome.session,
            next: None,
            redirects: 0,
            first_frames: HashMap::new(),
        }
    }

    /// One client turn before the shards tick: read the frames, follow a
    /// redirect, notice a switch, and send the input for the tick
    /// [`INPUT_LEAD`] ahead.
    fn act(&mut self, c: &mut Cluster, thrust: [f32; 3]) {
        for m in c.frames_for(self.on, self.session) {
            let Message::Redirect(r) = m else { continue };
            if r.entity != self.ship || self.next.is_some() {
                continue;
            }
            self.redirects += 1;
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
            // The switch: the first frame from the new shard.
            let frames = c.frames_for(to, session);
            let first = frames.iter().find_map(|m| match m {
                Message::Snapshot(s) => Some((s.tick, s.input_ack)),
                Message::Delta(d) => Some((d.tick, d.input_ack)),
                _ => None,
            });
            if let Some(first) = first {
                self.first_frames.insert(to, first);
                c.shard_mut(self.on).leave(self.session);
                self.on = to;
                self.session = session;
                self.next = None;
            }
        }
        // The cluster's tick is the same on every shard.
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
    }
}

/// The thrust a pilot holds at `tick`: ahead (+x), with a sideways and an
/// up-down part that change every few ticks — so an input applied a tick
/// early or late, twice or not at all changes where the ship ends up.
fn thrust_at(tick: SimTick) -> [f32; 3] {
    let phase = (tick / 3) % 5;
    [
        1.0,
        [0.0, 0.5, -0.5, 0.25, -1.0][phase as usize],
        [0.3, -0.3, 0.0, 1.0, -0.6][((tick / 4) % 5) as usize],
    ]
}

/// Fly the pilot's ship from 25 m before the border across it, `ticks`
/// ticks, and return its state after every tick (`None` where no shard
/// simulated it) with the shards that simulated it.
fn fly(
    c: &mut Cluster,
    pilot: &mut Pilot,
    ticks: SimTick,
    mut each: impl FnMut(&mut Cluster, &Pilot),
) -> Vec<(SimTick, Option<ShipState>, Vec<ShardId>)> {
    let mut start = ShipState::at_rest(at_border([-25.0, 0.0, 0.0]));
    start.vel = [12.0, 0.0, 0.0];
    assert!(c.shard_mut(ShardId::new(1)).set_state(pilot.ship, start));
    let mut out = Vec::new();
    for _ in 0..ticks {
        let tick = c.shards[0].sim().tick();
        pilot.act(c, thrust_at(tick + INPUT_LEAD));
        each(c, pilot);
        let tick = c.shards[0].sim().tick();
        out.push((tick, c.state(pilot.ship), c.simulators(pilot.ship)));
    }
    out
}

/// The same flight on one shard that owns both cells: where the ship is
/// after every tick when nothing is handed over.
fn reference(ticks: SimTick) -> Vec<(SimTick, Option<ShipState>, Vec<ShardId>)> {
    let mut c = Cluster::single();
    let mut pilot = Pilot::join(&mut c, "pilot");
    let flight = fly(&mut c, &mut pilot, ticks, |c, _| c.tick());
    assert_eq!(pilot.redirects, 0, "nothing to cross on one shard");
    flight
}

/// The tick a flight changed shard at: the first tick shard 2 simulated the
/// ship.
fn switch_tick(flight: &[(SimTick, Option<ShipState>, Vec<ShardId>)]) -> SimTick {
    flight
        .iter()
        .find(|(_, _, sims)| sims == &vec![ShardId::new(2)])
        .map(|(tick, ..)| *tick)
        .expect("shard 2 never simulated the ship")
}

/// A player's ship that crosses a border under changing thrust is, after
/// every tick, exactly where the same ship is that flew the same inputs on
/// one shard: the handoff costs no tick and no input, and exactly one shard
/// simulates the ship at every tick (guarantees 2, 4 and 5 on the cluster
/// clock). The new shard's first frame is for the tick after the commit's
/// state, and its `input_ack` covers every tick up to that state.
///
/// Mutation this catches: an install on the accept tick plus the lead when
/// that is a tick after the commit's state (one tick nobody simulated: the
/// ship is half a metre behind from then on); inputs sent to the old shard
/// only, or dropped by the accepting shard while the ship is staged (the
/// ticks around the commit fly on without their thrust); an input-ack floor
/// in another clock.
#[test]
fn a_ship_handed_over_flies_as_if_it_never_changed_shard() {
    let ticks = 160;
    let alone = reference(ticks);
    let mut c = Cluster::pair();
    let mut pilot = Pilot::join(&mut c, "pilot");
    let flight = fly(&mut c, &mut pilot, ticks, |c, _| c.tick());
    assert_eq!(pilot.redirects, 1, "one handoff");
    assert_eq!(pilot.on, ShardId::new(2));
    let switched = switch_tick(&flight);
    let mut in_transit = Vec::new();
    for ((tick, state, sims), (ref_tick, ref_state, _)) in flight.iter().zip(&alone) {
        assert_eq!(tick, ref_tick);
        assert!(sims.len() <= 1, "tick {tick}: simulated by {sims:?}");
        if sims.is_empty() {
            in_transit.push(*tick);
            continue;
        }
        assert_eq!(
            state, ref_state,
            "tick {tick} (the switch was at {switched}): the ship is not where it would be \
             without a handoff"
        );
    }
    // After the tick it was committed on, the ship's state is in the commit
    // message, on its way: shard 1 has let go, shard 2 installs it before its
    // next step. That one look between two ticks is the only one that finds
    // it on neither shard — and the tick after it is the next tick's state.
    assert_eq!(in_transit, vec![switched - 1], "no tick was lost");
    // It went a hysteresis past the border before it was offered, and the
    // lead after that.
    let at_switch = flight
        .iter()
        .find(|(tick, ..)| *tick == switched)
        .and_then(|(_, state, _)| *state)
        .expect("a state");
    assert!(
        border_x(at_switch.pos) > 1.0 + 12.0 * (HANDOFF_LEAD as f64 / 30.0),
        "{}",
        border_x(at_switch.pos)
    );
    // The first frame from shard 2: the tick after the committed state, with
    // every input up to that state acknowledged.
    let (first_tick, input_ack) = pilot.first_frames[&ShardId::new(2)];
    assert_eq!(first_tick, switched, "its first frame is that tick's");
    assert!(
        input_ack >= switched - 1 && input_ack <= switched,
        "input_ack {input_ack} at the switch tick {switched}"
    );
    let stats = (c.shards[0].stats(), c.shards[1].stats());
    assert_eq!((stats.0.handed_off, stats.1.received), (1, 1));
    assert_eq!(stats.1.caught_up, 0, "on time: nothing to catch up");
    assert_eq!(stats.0.aborted + stats.1.aborted, 0);
}

/// A commit that arrives late — the accepting shard has stepped one, two,
/// four ticks past the state it carries — installs the ship **brought
/// forward** by those ticks, under the inputs its player sent to both
/// shards: from the install on the ship is exactly where it would be without
/// a handoff. In between nobody simulated it (that is what "late" means), and
/// never two.
///
/// Mutation this catches: installing the commit's state as it is (the ship
/// is that many ticks of flight behind for good: 0.4 m a tick at this
/// speed), a catch-up that coasts instead of replaying the inputs (the
/// thrust of the missed ticks is lost), inputs for the ticks after the
/// install not handed to the simulation.
#[test]
fn a_late_commit_is_caught_up_with_the_inputs_of_the_ticks_it_missed() {
    let ticks = 160;
    let alone = reference(ticks);
    for late in [1u64, 2, 4] {
        let mut c = Cluster::pair();
        c.commit_delay = late;
        let mut pilot = Pilot::join(&mut c, "pilot");
        let flight = fly(&mut c, &mut pilot, ticks, |c, _| c.tick());
        assert_eq!(pilot.redirects, 1);
        let switched = switch_tick(&flight);
        let mut unsimulated = 0;
        for ((tick, state, sims), (_, ref_state, _)) in flight.iter().zip(&alone) {
            assert!(sims.len() <= 1, "tick {tick}: simulated by {sims:?}");
            if sims.is_empty() {
                unsimulated += 1;
                continue;
            }
            assert_eq!(
                state, ref_state,
                "commit {late} ticks late, tick {tick} (switch at {switched})"
            );
        }
        // The look after the commit tick (as for a commit on time), and one
        // for every tick the commit was under way.
        assert_eq!(unsimulated, late + 1, "the ticks the commit was under way");
        assert_eq!(c.shards[1].stats().caught_up, late);
        let (first_tick, input_ack) = pilot.first_frames[&ShardId::new(2)];
        assert_eq!(first_tick, switched);
        assert!(input_ack >= switched - 1, "{input_ack} at {switched}");
    }
}

/// A commit that reaches the accepting shard before that shard has stepped
/// to the tick the state describes (the two tick at the same instants, and
/// the accepting one is a little late) waits for that tick: the ship is not
/// stepped twice for one tick.
///
/// Mutation this catches: installing on arrival (the accepting shard then
/// steps the ship through the tick the committed state already is of: it is
/// a tick of flight ahead for good).
#[test]
fn a_commit_that_arrives_early_waits_for_its_tick() {
    let ticks = 160;
    let alone = reference(ticks);
    let mut c = Cluster::pair();
    let mut pilot = Pilot::join(&mut c, "pilot");
    // Shard 1 ticks, its messages are delivered, and only then does shard 2
    // tick: whatever shard 1 says after its tick `n` is in shard 2's hands
    // while shard 2 is still at `n - 1`.
    let flight = fly(&mut c, &mut pilot, ticks, |c, _| {
        c.shards[0].tick();
        c.round += 1;
        c.deliver();
        c.shards[1].tick();
        c.deliver();
    });
    assert_eq!(pilot.redirects, 1);
    let switched = switch_tick(&flight);
    let mut in_transit = 0;
    for ((tick, state, sims), (_, ref_state, _)) in flight.iter().zip(&alone) {
        assert!(sims.len() <= 1, "tick {tick}: simulated by {sims:?}");
        if sims.is_empty() {
            in_transit += 1;
            continue;
        }
        assert_eq!(state, ref_state, "tick {tick} (switch at {switched})");
    }
    assert_eq!(
        in_transit, 1,
        "the look after the commit tick, and no other"
    );
    assert_eq!(c.shards[1].stats().caught_up, 0);
}

fn crate_body() -> Body {
    let mut body = Body::dynamic(ShapeRef::cuboid([0.3; 3]));
    body.mass = MassSpec::Mass(50.0);
    body
}

/// One player stands a few metres from a border; another — who joined first,
/// so its ship has the lower guid and is the key of any island the two form
/// — flies past it, over the border and back, three times. The one who
/// stands is never offered to anybody: its ship stays on shard 1, its client
/// is sent no redirect, and no transfer is called off.
///
/// Mutation this catches: a player's ship moved with its island (the
/// stander's ship was offered to shard 2 whenever the mover's home was
/// shard 2 and the mover's second of motion reached it), an island key that
/// takes a local player's ship by its map position.
#[test]
fn a_standing_players_ship_is_not_handed_over_when_another_flies_past() {
    let mut c = Cluster::pair();
    let mut mover = Pilot::join(&mut c, "mover");
    let mut stander = Pilot::join(&mut c, "stander");
    assert!(mover.ship < stander.ship, "the mover is the island's key");
    let stands = ShipState::at_rest(at_border([-3.0, 0.0, 1.5]));
    assert!(c.shard_mut(ShardId::new(1)).set_state(stander.ship, stands));
    let mut start = ShipState::at_rest(at_border([-25.0, 0.0, 0.0]));
    start.vel = [12.0, 0.0, 0.0];
    assert!(c.shard_mut(ShardId::new(1)).set_state(mover.ship, start));
    let mut together = 0;
    let mut heading = 1.0f32;
    let mut crossings = 0;
    for _ in 0..4_000 {
        let Some(at) = c.state(mover.ship) else {
            c.tick();
            continue;
        };
        // Turn round 30 m past the border, either side.
        let x = border_x(at.pos);
        if (heading > 0.0 && x > 30.0) || (heading < 0.0 && x < -30.0) {
            heading = -heading;
            crossings += 1;
        }
        mover.act(&mut c, [heading, 0.0, 0.0]);
        stander.act(&mut c, [0.0; 3]);
        c.tick();
        assert_eq!(
            c.simulators(stander.ship),
            vec![ShardId::new(1)],
            "the one who stands was handed over"
        );
        // (The premise: the two ships do form an island when one passes.)
        let near = c.state(stander.ship).zip(c.state(mover.ship));
        if near.is_some_and(|(s, m)| WorldPos::distance(s.pos, m.pos) < 4.0) {
            together += 1;
        }
        if crossings >= 5 {
            break;
        }
    }
    assert!(crossings >= 5, "the mover crossed {crossings} times");
    assert!(
        together > 10,
        "the mover passed within reach: {together} ticks"
    );
    assert_eq!(stander.redirects, 0, "the stander was sent a redirect");
    assert!(
        mover.redirects >= 4,
        "the mover's handoffs: {}",
        mover.redirects
    );
    let moved = WorldPos::distance(c.state(stander.ship).expect("alive").pos, stands.pos);
    assert!(moved < 0.5, "the stander was pushed {moved} m");
    for shard in &c.shards {
        assert!(
            shard.recent_aborts().iter().all(|note| !note.player),
            "a player's transfer was called off: {:?}",
            shard.recent_aborts()
        );
    }
}

/// A player flies from shard 1 into shard 2 past a row of crates floating on
/// shard 2's side. As its second of motion reaches each crate, shard 2 —
/// which sees the ship as shard 1's ghost — sends that crate to shard 1 (the
/// bodies a player touches are consolidated on the player's shard): crates
/// are on their way from 2 to 1 at the very moment shard 1 offers the ship
/// to shard 2. The ship's handoff goes through at the first attempt, in the
/// lead's second; no player's transfer is called off.
///
/// This is the crossing that took three seconds on the demo area: shard 2
/// refused the ship's group as a "crossing island transfer", and the ship
/// sat out the re-offer pause.
///
/// Mutation this catches: the crossing rule applied to a player's ship, a
/// refused body calling off the ship of its group.
#[test]
fn a_players_handoff_is_not_called_off_by_bodies_going_the_other_way() {
    let mut c = Cluster::pair();
    let mut pilot = Pilot::join(&mut c, "pilot");
    // Crates beside the path, two metres apart, from 2 m to 20 m past the
    // border.
    let mut crates = Vec::new();
    for i in 0..10 {
        let at = at_border([2.0 + 2.0 * f64::from(i), 0.0, 1.6]);
        crates.push(
            c.shard_mut(ShardId::new(2))
                .spawn_body(BodySpawn::new(ShipState::at_rest(at), crate_body()))
                .expect("a crate"),
        );
    }
    let mut start = ShipState::at_rest(at_border([-25.0, 0.0, 0.0]));
    start.vel = [10.0, 0.0, 0.0];
    assert!(c.shard_mut(ShardId::new(1)).set_state(pilot.ship, start));
    let mut offered_at: Option<SimTick> = None;
    let mut switched_at: Option<SimTick> = None;
    let mut crates_went_to_1 = false;
    for _ in 0..300 {
        pilot.act(&mut c, [0.4, 0.0, 0.0]);
        c.tick();
        let tick = c.shards[0].sim().tick();
        let x = c.state(pilot.ship).map(|s| border_x(s.pos));
        if offered_at.is_none() && x.is_some_and(|x| x > 1.0) {
            offered_at = Some(tick);
        }
        crates_went_to_1 |= crates
            .iter()
            .any(|g| c.shards[0].sim().ship_state(*g).is_some());
        if switched_at.is_none() && pilot.on == ShardId::new(2) {
            switched_at = Some(tick);
        }
    }
    assert!(
        crates_went_to_1,
        "the premise: shard 2 sent crates to the player's shard as it came near"
    );
    let (offered, switched) = (
        offered_at.expect("crossed"),
        switched_at.expect("handed over"),
    );
    assert!(
        switched - offered <= HANDOFF_LEAD + 4,
        "the handoff took {} ticks from the offer",
        switched - offered
    );
    assert_eq!(pilot.redirects, 1, "one attempt");
    for shard in &c.shards {
        let aborts = shard.recent_aborts();
        assert!(
            aborts.iter().all(|note| !note.player),
            "a player's transfer was called off: {aborts:?}"
        );
    }
    // And the crates end up with the player again, on shard 2.
    for _ in 0..200 {
        pilot.act(&mut c, [0.0; 3]);
        c.tick();
    }
    let near: Vec<EntityGuid> = crates
        .iter()
        .copied()
        .filter(|g| {
            c.state(*g)
                .zip(c.state(pilot.ship))
                .is_some_and(|(b, s)| WorldPos::distance(b.pos, s.pos) < 3.0)
        })
        .collect();
    for guid in near {
        assert_eq!(
            c.simulators(guid),
            c.simulators(pilot.ship),
            "a crate beside the player is on the player's shard"
        );
    }
}

/// A player's ship crosses the border with a crate drifting along a metre
/// ahead of it: one island, the ship its key, so the crate is offered with
/// the ship, in one group. Without interference both arrive on shard 2 on
/// the same tick. When shard 2 **refuses the crate** — here: its abort for
/// it reaches shard 1 while the group is in flight — the crate stays behind
/// and the ship is handed over all the same, in the lead's second, at the
/// first attempt.
///
/// Mutation this catches: a refused member calling off its whole group, the
/// player's ship included (the ship then waits out the two seconds of the
/// re-offer pause: three seconds for a crossing); bodies that do not travel
/// with their player's ship at all (the first half: the crate would arrive
/// ticks after the ship).
#[test]
fn a_refused_body_does_not_call_off_the_ship_it_travels_with() {
    for refuse in [false, true] {
        let mut c = Cluster::pair();
        let mut pilot = Pilot::join(&mut c, "pilot");
        let mut start = ShipState::at_rest(at_border([-12.0, 0.0, 0.0]));
        start.vel = [4.0, 0.0, 0.0];
        assert!(c.shard_mut(ShardId::new(1)).set_state(pilot.ship, start));
        let mut ahead = ShipState::at_rest(at_border([-10.8, 0.0, 0.0]));
        ahead.vel = [4.0, 0.0, 0.0];
        let crate_guid = c
            .shard_mut(ShardId::new(1))
            .spawn_body(BodySpawn::new(ahead, crate_body()))
            .expect("a crate");
        let (mut offered_at, mut switched_at, mut refused) = (None, None, false);
        let mut together = true;
        for _ in 0..200 {
            pilot.act(&mut c, [0.0; 3]);
            // The ship's player has been redirected: the group is accepted
            // and in flight. Shard 2 calls the crate off.
            if refuse && !refused && pilot.next.is_some() {
                refused = true;
                c.shard_mut(ShardId::new(1)).peer_message(
                    ShardId::new(2),
                    Message::HandoffAbort(space_protocol::HandoffAbort {
                        entity: crate_guid,
                        from: ShardId::new(1),
                        to: ShardId::new(2),
                        reason: space_protocol::HandoffReason::Other("refused by the test".into()),
                    }),
                );
            }
            c.tick();
            let tick = c.shards[0].sim().tick();
            if offered_at.is_none() && c.state(pilot.ship).is_some_and(|s| border_x(s.pos) > 1.0) {
                offered_at = Some(tick);
            }
            if switched_at.is_none() && pilot.on == ShardId::new(2) {
                switched_at = Some(tick);
            }
            let (ship, body) = (c.simulators(pilot.ship), c.simulators(crate_guid));
            assert!(ship.len() <= 1 && body.len() <= 1, "{ship:?} {body:?}");
            together &= ship == body;
        }
        let (offered, switched) = (
            offered_at.expect("crossed"),
            switched_at.expect("handed over"),
        );
        assert!(
            switched - offered <= HANDOFF_LEAD + 4,
            "crate refused: {refuse}; the ship's handoff took {} ticks",
            switched - offered
        );
        assert_eq!(pilot.redirects, 1, "one attempt (crate refused: {refuse})");
        assert_eq!(c.simulators(pilot.ship), vec![ShardId::new(2)]);
        let players_called_off = c
            .shards
            .iter()
            .flat_map(Shard::recent_aborts)
            .filter(|note| note.player)
            .count();
        assert_eq!(players_called_off, 0, "crate refused: {refuse}");
        if refuse {
            assert!(refused, "the premise: the group was in flight");
            assert!(!together, "the crate stayed behind when the ship went");
        } else {
            // Ship and crate were on one shard after every tick: committed
            // on one tick, installed on one tick.
            assert!(together, "the crate travels with the ship");
            assert_eq!(c.simulators(crate_guid), vec![ShardId::new(2)]);
        }
    }
}

/// Three shards with real tick loops (`clock::run`, the binary's), the
/// second started 1.3 s after the first and the third — a replacement, say —
/// another 0.9 s later, all on one tick origin: looked at in the same
/// instant their ticks differ by **at most one** (each steps a tick when it
/// is due; one of them may just have), each is at the tick that is due by
/// the wall clock or just behind it, and none is ever ahead of it. (The
/// tolerance is for a machine that runs other tests at the same time: nine
/// looks in ten must be within one tick, every look within four — a shard
/// whose task was kept waiting is late for a moment and makes the ticks up.)
///
/// Mutation this catches: counting from the shard's own start (39 and 66
/// ticks apart), an interval that skips a late tick for good (a shard that
/// stalled once is behind for ever), a replacement placed on the timeline by
/// anything but the origin.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shards_started_seconds_apart_count_the_same_ticks() {
    let origin = space_server::config::wall_clock_us() - 77_000_000;
    let start = |id: u32| {
        let net = Arc::new(
            space_server::net::ShardNet::bind(SocketAddr::from(([127, 0, 0, 1], 0))).expect("bind"),
        );
        let shard = Arc::new(tokio::sync::Mutex::new(Shard::new(ShardConfig::new(
            ShardId::new(id),
            net.local_addr().expect("addr"),
            NodeRef::new(64, [i64::from(id), 0, 0]),
        ))));
        let clock = TickClock::new(origin);
        let task = tokio::spawn(space_server::clock::run(
            Arc::clone(&shard),
            Arc::clone(&net),
            clock,
        ));
        (shard, net, task, clock)
    };
    let a = start(1);
    tokio::time::sleep(Duration::from_millis(1_300)).await;
    let b = start(2);
    tokio::time::sleep(Duration::from_millis(900)).await;
    let c = start(3);
    tokio::time::sleep(Duration::from_millis(300)).await;
    let looks = 60;
    let (mut apart, mut behind) = (Vec::new(), Vec::new());
    for _ in 0..looks {
        // Looked at together: all three locked, then read.
        let (ga, gb, gc) = (a.0.lock().await, b.0.lock().await, c.0.lock().await);
        let now = Instant::now();
        let ticks = [ga.sim().tick(), gb.sim().tick(), gc.sim().tick()];
        let due = [a.3.tick_at(now), b.3.tick_at(now), c.3.tick_at(now)];
        drop((ga, gb, gc));
        assert_eq!(due[0], due[1]);
        assert_eq!(due[1], due[2], "one timeline: the same tick is due on all");
        let (lo, hi) = (*ticks.iter().min().unwrap(), *ticks.iter().max().unwrap());
        apart.push(hi - lo);
        behind.push(due[0].saturating_sub(lo));
        assert!(
            hi <= due[0],
            "a shard ahead of the clock: {ticks:?} at {due:?}"
        );
        tokio::time::sleep(Duration::from_millis(17)).await;
    }
    // 77 s after the origin, at 30 Hz.
    let tick = a.0.lock().await.sim().tick();
    assert!((2_300..2_700).contains(&tick), "tick {tick}");
    let within = |values: &[u64], limit: u64| values.iter().filter(|v| **v <= limit).count();
    assert!(
        within(&apart, 1) >= looks * 9 / 10 && within(&apart, 4) == looks,
        "the shards' ticks, apart: {apart:?}"
    );
    assert!(
        within(&behind, 1) >= looks * 9 / 10 && within(&behind, 4) == looks,
        "behind the clock: {behind:?}"
    );
    for (_, net, task, _) in [a, b, c] {
        task.abort();
        net.close();
    }
}
