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
//! | [`inputs_sent_before_a_late_connection_are_said_again_and_kept`] | a connection that comes up within the input lead of the commit loses no input |
//! | [`a_commit_that_arrives_early_waits_for_its_tick`] | no double step |
//! | [`a_standing_players_ship_is_not_handed_over_when_another_flies_past`] | a player's ship moves for where it is, not with an island |
//! | [`a_players_handoff_is_not_called_off_by_bodies_going_the_other_way`] | the crossing rule and a refused body never cost a player its handoff |
//! | [`a_refused_body_does_not_call_off_the_ship_it_travels_with`] | the bodies of a player's island travel with the ship; the ship waits for none of them |
//! | [`a_commit_installed_after_a_stall_is_not_brought_forward_by_the_stall`] | the catch-up is bounded when the entity is installed, not only when its commit arrives |
//! | [`two_players_pushing_each_other_across_a_border`] | two ships in contact across a border behave as on one shard: no bounce, no pass-through |
//! | [`shards_started_seconds_apart_count_the_same_ticks`] | the cluster clock, with the binary's tick loop on a clock the test sets |

mod common;

use common::*;

use space_protocol::{EntityGuid, Message, SceneClock, ShardId};
use space_server::clock::{ManualTime, Pacer, TickClock, MAX_TICK_BURST};
use space_server::handoff::{HANDOFF_LEAD, MAX_CATCH_UP};
use space_server::shard::Shard;
use space_server::ShardConfig;
use space_sim::body::{Body, MassSpec};
use space_sim::shape::ShapeRef;
use space_sim::{BodySpawn, ShipState, SimTick};
use space_world::{NodeRef, WorldPos};

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

/// A player whose connection to the accepting shard comes up late — the
/// lead's second is all but over — so that the commit follows it within the
/// ticks its inputs are tagged ahead: the inputs it had sent before, for
/// ticks after the commit, went to the old shard only. It says them again to
/// the new shard the moment it is through (before the status that opens the
/// gate), and the ship flies as if it had never changed shard, bit for bit.
///
/// Mutation this catches: a client that does not say them again (the ticks
/// right after the commit fly without their thrust: the ship is off the
/// reference from the switch on), a shard that drops what a player sends
/// for the ship staged for it before the player's status has arrived.
#[test]
fn inputs_sent_before_a_late_connection_are_said_again_and_kept() {
    let ticks = 160;
    let alone = reference(ticks);
    // Through just before the lead runs out, as it does, and a little after
    // (the gate then opens on a commit that is already due).
    let mut lost_unsaid = Vec::new();
    for takes in HANDOFF_LEAD - 2..=HANDOFF_LEAD + 3 {
        for resends in [true, false] {
            let mut c = Cluster::pair();
            let mut pilot = Pilot::join(&mut c, "pilot");
            pilot.connect_takes = takes;
            pilot.resends = resends;
            let flight = fly(&mut c, &mut pilot, ticks, |c, _| c.tick());
            assert_eq!(pilot.redirects, 1, "one handoff (connect takes {takes})");
            let switched = switch_tick(&flight);
            if !resends {
                // The premise is collected: does this delay cost an input?
                let same = flight
                    .iter()
                    .zip(&alone)
                    .filter(|((_, _, sims), _)| !sims.is_empty())
                    .all(|((_, state, _), (_, ref_state, _))| state == ref_state);
                if !same {
                    lost_unsaid.push(takes);
                }
                continue;
            }
            for ((tick, state, sims), (_, ref_state, _)) in flight.iter().zip(&alone) {
                assert!(sims.len() <= 1, "tick {tick}: simulated by {sims:?}");
                if !sims.is_empty() {
                    assert_eq!(
                        state, ref_state,
                        "connect takes {takes}: tick {tick} (the switch was at {switched})"
                    );
                }
            }
            // Every tick's input was applied once, by one of the two shards.
            let mut applied: Vec<SimTick> = c
                .shards
                .iter()
                .flat_map(|s| s.sim().applied_inputs())
                .filter(|(_, guid)| *guid == pilot.ship)
                .map(|(tick, _)| tick)
                .collect();
            applied.sort_unstable();
            let (first, last) = (applied[0], applied[applied.len() - 1]);
            assert_eq!(
                applied,
                (first..=last).collect::<Vec<_>>(),
                "connect takes {takes}: every tick once, none missing, none twice"
            );
            assert!(first <= START + INPUT_LEAD + 1 && last >= START + ticks);
        }
    }
    // The premise: among these delays are ones that cost inputs when they
    // are not said again (the commit followed the connection within the
    // input lead).
    assert!(
        !lost_unsaid.is_empty(),
        "no delay made the commit follow the connection within the input lead"
    );
    eprintln!(
        "a connection this many ticks after the redirect loses inputs unsaid: {lost_unsaid:?}"
    );
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

/// A handoff's commit is in the accepting shard's hands, a tick before its
/// install, when the host stands still for three seconds (a suspended
/// machine: both shards find ninety ticks gone and jump). The entity — a
/// player's ship, and a drifting crate — is then installed **where its
/// commit says it is**: it stood still for the stall like everything else.
///
/// Mutation this catches: bounding the catch-up only when the commit arrives
/// (`local_tick_of`: the state's tick was then within `MAX_CATCH_UP` of the
/// present) and not at the install — the ship is brought forward by the 89
/// ticks of the stall, 36 m of flight without a single contact, through
/// whatever stands there; `caught_up` counts them.
#[test]
fn a_commit_installed_after_a_stall_is_not_brought_forward_by_the_stall() {
    const STALL: SimTick = 90;
    for body in [true, false] {
        let mut c = Cluster::pair();
        let mut pilot = Pilot::join(&mut c, "pilot");
        let mut start = ShipState::at_rest(at_border([-25.0, 0.0, 0.0]));
        start.vel = [12.0, 0.0, 0.0];
        let guid = if body {
            c.shard_mut(ShardId::new(1))
                .spawn_body(BodySpawn::new(start, crate_body()))
                .expect("a crate")
        } else {
            assert!(c.shard_mut(ShardId::new(1)).set_state(pilot.ship, start));
            pilot.ship
        };
        let mut committed: Option<ShipState> = None;
        let mut installed: Option<ShipState> = None;
        for _ in 0..400 {
            if !body {
                pilot.act(&mut c, [0.0; 3]);
            }
            if committed.is_some() {
                c.tick();
                if installed.is_none() {
                    installed = c.shards[1].sim().ship_state(guid);
                }
                continue;
            }
            // Shard 1 ticks first and its messages are delivered: a commit
            // it sends is in shard 2's hands while shard 2 is a tick behind.
            let sent = c.shards[0].stats().handed_off;
            c.shards[0].tick();
            c.round += 1;
            c.deliver();
            if c.shards[0].stats().handed_off == sent {
                c.shards[1].tick();
                c.deliver();
                continue;
            }
            // The commit is there. (What it says: shard 1's ghost of it.)
            committed = c.shards[0].sim().state_of(guid);
            assert!(c.shards[1].sim().ship_state(guid).is_none());
            // Three seconds pass for nobody; then both find them gone.
            let due = c.shards[0].sim().tick() + STALL;
            assert!(c.shards[0].run_to(due).skipped > 0);
            let paced = c.shards[1].run_to(due);
            assert_eq!(paced.skipped, STALL, "shard 2 jumped");
            c.round += 1;
            c.deliver();
            installed = c.shards[1].sim().ship_state(guid);
        }
        let (committed, installed) = (
            committed.expect("the entity was committed"),
            installed.expect("and installed on shard 2"),
        );
        // One tick of flight (the step after the install), not ninety.
        let flown = WorldPos::distance(committed.pos, installed.pos);
        assert!(
            flown < 1.0,
            "body: {body}; the entity was brought forward {flown} m by a stall"
        );
        assert_eq!(
            c.shards[1].stats().caught_up,
            0,
            "body: {body}; {STALL} ticks is past the catch-up's {MAX_CATCH_UP}"
        );
        assert_eq!(c.simulators(guid), vec![ShardId::new(2)]);
        assert_eq!(c.shards[1].stats().received, 1);
    }
}

/// What two players' ships do to each other, tick by tick: `a` starts
/// `apart / 2` west of the border and `b` as far east of it, and for `ticks`
/// ticks each holds the thrust given (along x). Returns, per tick, the two
/// ships' positions along x (metres from the border) and velocities.
fn push(
    c: &mut Cluster,
    apart: f64,
    thrust: [f32; 2],
    ticks: SimTick,
) -> Vec<([f64; 2], [f32; 2])> {
    let mut a = Pilot::join(c, "a");
    let mut b = Pilot::join(c, "b");
    let at = |x: f64| ShipState::at_rest(at_border([x, 0.0, 0.0]));
    assert!(c
        .shard_mut(ShardId::new(1))
        .set_state(a.ship, at(-apart / 2.0)));
    assert!(c
        .shard_mut(ShardId::new(1))
        .set_state(b.ship, at(apart / 2.0)));
    // `b` stands on the ground east of the border: where there is a second
    // shard it is handed over to it before anything else happens.
    for _ in 0..150 {
        a.act(c, [0.0; 3]);
        b.act(c, [0.0; 3]);
        c.tick();
    }
    let east = c.shards.len().min(2) as u32;
    assert_eq!(c.simulators(b.ship), vec![ShardId::new(east)]);
    assert_eq!(c.simulators(a.ship), vec![ShardId::new(1)]);
    let mut out = Vec::new();
    for _ in 0..ticks {
        a.act(c, [thrust[0], 0.0, 0.0]);
        b.act(c, [thrust[1], 0.0, 0.0]);
        c.tick();
        // A ship that is changing shard is on neither for one look: its
        // last word then.
        let (sa, sb) = (c.state(a.ship), c.state(b.ship));
        let last = out.last().copied().unwrap_or(([0.0; 2], [0.0; 2]));
        out.push((
            [
                sa.map_or(last.0[0], |s| border_x(s.pos)),
                sb.map_or(last.0[1], |s| border_x(s.pos)),
            ],
            [
                sa.map_or(last.1[0], |s| s.vel[0]),
                sb.map_or(last.1[1], |s| s.vel[0]),
            ],
        ));
    }
    out
}

/// Two players in contact across a border, measured against the same two
/// on one shard (`docs/plan-sharding.md` §5, "Two players in contact").
/// Each shard simulates its own player's ship and holds the other as a
/// **ghost** — a kinematic copy, moved to where its owner said it was a
/// tick ago. A player's ship is never handed over for an island
/// (`border.rs`, "Players' ships"), so the two stay on two shards for as
/// long as each stays on its own ground.
///
/// * **Pushing against each other** (both thrust at the other): on one shard
///   they meet and stand, pressed together, hull to hull (0.6 m between
///   their centres). Across the border the same: they stand within a
///   centimetre or two of that, the gap moves by millimetres a tick, and
///   neither creeps.
/// * **One pushes the other** (one thrusts, the other does not): on one
///   shard the two move off together, sharing the thrust. Across the border
///   too — a little slower while they are on two shards (each hears of the
///   other a tick late) — and once the pusher is a metre across the border
///   it changes shard and the two are on one.
///
/// Mutation this catches: a ship's ghost presented to the solver at its
/// owner's velocity alone (`rigid.rs`, `meeting_velocities`: each ship is
/// then thrown back at the other's speed — measured: the two bounce apart
/// to 2.3 m and back for as long as they push, the gap changing by 0.41 m
/// a tick); ghosts of players' ships that do not collide (the ships pass
/// through each other); a pusher that cannot move what it pushes.
#[test]
fn two_players_pushing_each_other_across_a_border() {
    let ticks = 240;
    for (name, thrust) in [
        ("against each other", [1.0f32, -1.0]),
        ("one pushes", [1.0, 0.0]),
    ] {
        let mut one = Cluster::single();
        let alone = push(&mut one, 4.0, thrust, ticks);
        let mut two = Cluster::pair();
        let split = push(&mut two, 4.0, thrust, ticks);
        let gap = |run: &[([f64; 2], [f32; 2])]| -> Vec<f64> {
            run.iter().map(|(x, _)| x[1] - x[0]).collect()
        };
        let (gap_alone, gap_split) = (gap(&alone), gap(&split));
        let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        // In contact: the smallest gap on one shard is the two hulls touching.
        let touching = min(&gap_alone);
        let deepest = touching - min(&gap_split);
        // Momentum (equal masses): the sum of the two velocities.
        let sum = |run: &[([f64; 2], [f32; 2])], i: usize| f64::from(run[i].1[0] + run[i].1[1]);
        let end = ticks as usize - 1;
        // Jitter: the largest change of the gap from one tick to the next
        // over the last two seconds, when the two are in contact.
        let jitter = |g: &[f64]| {
            g[end - 60..=end]
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .fold(0.0, f64::max)
        };
        eprintln!(
            "[{name}] one shard: gap at contact {touching:.3} m, at the end {:.3} m; a at {:.2} m, \
             b at {:.2} m; velocities {:.2?}; gap jitter {:.4} m/tick",
            gap_alone[end], alone[end].0[0], alone[end].0[1], alone[end].1, jitter(&gap_alone)
        );
        eprintln!(
            "[{name}] two shards: {deepest:.3} m deeper inside each other, gap at the end {:.3} m; \
             a at {:.2} m, b at {:.2} m; velocities {:.2?}; gap jitter {:.4} m/tick; momentum \
             {:.2} against {:.2} m/s; handoffs {:?}, aborted {:?}",
            gap_split[end],
            split[end].0[0],
            split[end].0[1],
            split[end].1,
            jitter(&gap_split),
            sum(&split, end),
            sum(&alone, end),
            two.shards.iter().map(|s| s.stats().handed_off).collect::<Vec<_>>(),
            two.shards.iter().map(|s| s.stats().aborted).collect::<Vec<_>>(),
        );
        // The hulls touch at 0.6 m between the centres (one shard, at rest).
        let hull = gap_alone[end];
        assert!((hull - 0.6).abs() < 0.01, "[{name}] the premise: {hull}");
        // Never through each other: no deeper than the impact on one shard
        // goes, and a decimetre.
        assert!(
            deepest < 0.1 && min(&gap_split) > 0.3,
            "[{name}] the ships went into each other: {deepest} m deeper than on one shard"
        );
        // In contact for the last two seconds, hull to hull, without a bounce.
        for g in &gap_split[end - 60..=end] {
            assert!(
                (g - hull).abs() < 0.05,
                "[{name}] in contact the centres are {g:.3} m apart, {hull:.3} m on one shard"
            );
        }
        assert!(
            jitter(&gap_split) < 0.01,
            "[{name}] the gap jitters by {} m a tick",
            jitter(&gap_split)
        );
        assert_eq!(
            two.shards.iter().map(|s| s.stats().aborted).sum::<usize>(),
            0
        );
        if thrust[1] != 0.0 {
            // Pressed together they stand where they met, as on one shard.
            for k in 0..2 {
                assert!(
                    (split[end].0[k] - alone[end].0[k]).abs() < 0.1,
                    "[{name}] they crept: {:?} against {:?}",
                    split[end].0,
                    alone[end].0
                );
            }
            assert!(sum(&split, end).abs() < 0.05, "momentum is still none");
        } else {
            // The pushed ship is pushed: together, and at no less than four
            // fifths of the speed two ships on one shard reach.
            let (v, v_alone) = (split[end].1, alone[end].1);
            assert!((v[0] - v[1]).abs() < 0.05, "[{name}] together: {v:?}");
            assert!(
                v[1] > 0.8 * v_alone[1] && v[1] <= v_alone[1] + 0.05,
                "[{name}] pushed to {} m/s, {} on one shard",
                v[1],
                v_alone[1]
            );
        }
    }
}

/// Three shards on the binary's tick loop (`clock::Pacer`: what `clock::run`
/// does between two sleeps), the second started 1.3 s after the first and the
/// third — a replacement, say — another 0.9 s later, all on one tick origin
/// and one host (one `ManualTime`: the test decides what time it is, so
/// nothing here depends on how busy the machine is). After **every** turn of
/// the loops every shard is at exactly the tick that is due — whether the
/// turn came on time, three ticks late (made up) or three seconds late
/// (jumped) — and so at the same tick as the others.
///
/// Mutation this catches: counting from the shard's own start (39 and 66
/// ticks apart), a loop that skips a late tick for good (a shard that
/// stalled once is behind for ever), a replacement placed on the timeline by
/// anything but the origin.
#[test]
fn shards_started_seconds_apart_count_the_same_ticks() {
    let time = ManualTime::at(ORIGIN + 77_000_000);
    let cluster = SceneClock::cluster(ORIGIN);
    let mut running: Vec<(Shard, Pacer)> = Vec::new();
    let start = |id: u32, running: &mut Vec<(Shard, Pacer)>| {
        let mut shard = Shard::new(ShardConfig::new(
            ShardId::new(id),
            listening(id as u16),
            NodeRef::new(64, [i64::from(id), 0, 0]),
        ));
        let mut pacer = Pacer::new(TickClock::on(ORIGIN, time.clone()).expect("a clock"));
        pacer.start(&mut shard);
        running.push((shard, pacer));
    };
    // How long the loops sleep between turns, µs: on time, early, a few
    // ticks late — never more than the burst.
    let naps = [
        33_333i64, 7_000, 12_500, 41_000, 3_000, 29_000, 100_000, 16_667, 250_000,
    ];
    let run = |us: i64, running: &mut Vec<(Shard, Pacer)>| {
        let mut passed = 0;
        let mut turn = 0;
        while passed < us {
            let nap = naps[turn % naps.len()];
            turn += 1;
            passed += nap;
            time.advance(nap);
            let due = cluster.tick_at(ORIGIN + 77_000_000 + elapsed(&time));
            for (shard, pacer) in running.iter_mut() {
                let paced = pacer.turn(shard);
                assert_eq!(paced.skipped, 0, "late by at most the burst: made up");
                assert_eq!(
                    shard.sim().tick(),
                    due,
                    "{} after a {nap} µs nap",
                    shard.shard()
                );
                assert_eq!(shard.scene_clock(), cluster);
                assert_eq!(shard.sim().scene_time(), cluster.time_us_at(due));
            }
        }
    };
    fn elapsed(time: &ManualTime) -> i64 {
        use space_server::clock::TimeSource;
        time.mono_us()
    }
    start(1, &mut running);
    assert_eq!(running[0].0.sim().tick(), 77 * 30, "77 s after the origin");
    run(1_300_000, &mut running);
    start(2, &mut running);
    run(900_000, &mut running);
    start(3, &mut running);
    run(2_000_000, &mut running);
    let ticks: Vec<SimTick> = running.iter().map(|(s, _)| s.sim().tick()).collect();
    assert!(ticks.iter().all(|t| *t == ticks[0]), "{ticks:?}");
    assert!(ticks[0] > 77 * 30 + 120, "four seconds on: {ticks:?}");
    // Shard 2 stands still for three seconds while the others run; then it
    // jumps, and is on the cluster's tick again at once.
    for _ in 0..90 {
        time.advance(33_334);
        for (shard, pacer) in running.iter_mut().filter(|(s, _)| s.shard().0 != 2) {
            assert_eq!(pacer.turn(shard).stepped, 1);
        }
    }
    let behind = running[0].0.sim().tick() - running[1].0.sim().tick();
    assert_eq!(behind, 90);
    let paced = {
        let (shard, pacer) = &mut running[1];
        pacer.turn(shard)
    };
    assert_eq!((paced.stepped, paced.skipped), (1, 89));
    assert!(behind > MAX_TICK_BURST);
    run(500_000, &mut running);
    for (shard, _) in &running {
        assert_eq!(
            shard.stats().skipped,
            if shard.shard().0 == 2 { 89 } else { 0 }
        );
        assert_eq!(shard.stats().clock_offset_us, 0, "one host, one clock");
    }
}
