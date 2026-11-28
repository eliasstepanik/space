//! Bodies on fixed orbits of scene time: what is near one is carried along
//! (a ship lands on a moon moving thousands of kilometres a second and stays
//! landed), the scene clock decides where they are, and fast travel scales the
//! speed limit with the distance to the nearest surface.

use space_protocol::{EntityGuid, PlayerInput, SceneClock, ShardId};
use space_sim::body::{CollisionGroups, Visual};
use space_sim::gravity::BodyPlace;
use space_sim::motion::{Circle, OrbitPath};
use space_sim::physics::{fast_travel_limit, top_speed, FAST_CLEARANCE, FAST_MAX_SPEED};
use space_sim::{
    GravityField, LargeBody, ShapeRef, ShipParams, ShipState, SimWorld, StaticCollider, WorldPoint,
};
use space_world::WorldPos;

/// A moon: 1000 km in radius, 1.62 m/s² on its surface, on a circle of
/// 1e8 m around the origin once every 100 s (6.3e6 m/s), a reach of 10 radii.
const R: f64 = 1.0e6;
const REACH: f64 = 1.0e7;

fn path() -> OrbitPath {
    OrbitPath::new(vec![Circle {
        radius: 1.0e8,
        period_us: 100_000_000,
        phase: 0.3,
        tilt: 0.1,
        weight: 1.0,
    }])
}

fn moon_world(clock: SceneClock) -> SimWorld {
    let base = WorldPoint::at(WorldPos::ZERO);
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_scene_clock(clock);
    sim.set_gravity(GravityField {
        zones: vec![],
        bodies: vec![LargeBody {
            name: "moon".into(),
            gm: 1.62 * R * R,
            radius: R,
            place: BodyPlace::Orbit { base, path: path() },
            reach: REACH,
        }],
        clock: SceneClock::default(),
    });
    sim.set_statics(vec![StaticCollider {
        shape: ShapeRef::Ball { radius: R as f32 },
        place: base,
        rotation: [1.0, 0.0, 0.0, 0.0],
        friction: 0.6,
        restitution: 0.1,
        visual: Visual(0),
        groups: CollisionGroups::ALL,
        motion: Some(path()),
    }])
    .expect("the moon builds");
    sim
}

fn moon_center(sim: &SimWorld) -> WorldPoint {
    sim.gravity().bodies[0].center_at(sim.tick(), sim.scene_time())
}

fn above(sim: &SimWorld, guid: EntityGuid) -> [f64; 3] {
    moon_center(sim).delta_to(sim.ship_state(guid).expect("alive").pos)
}

fn len(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// A ship let go 50 m over the moon falls onto it and stays on it, although
/// the moon moves 210 km a tick; it stays over the same spot (the carry has
/// no fictitious force); the set clock decides where the moon is.
#[test]
fn a_ship_lands_on_a_moving_moon_and_stays_landed() {
    let clock = SceneClock::new(0, 1_763_900_000_000_000);
    let mut sim = moon_world(clock);
    let guid = EntityGuid::new(ShardId(1), 1);
    let start =
        WorldPos::from_f64_metres_around(moon_center(&sim).to_world_pos(), [0.0, R + 50.0, 0.0]);
    sim.spawn_ship(guid, 1, ShipState::at_rest(start));
    let first = above(&sim, guid);
    assert!((len(first) - (R + 50.0)).abs() < 0.01, "{first:?}");
    // The clock decides where the moon is.
    let c = moon_center(&sim).to_world_pos();
    let expect = path().offset(clock.time_us);
    let d = WorldPos::delta(WorldPos::ZERO, c);
    assert!(
        (0..3).all(|k| (d[k] - expect[k]).abs() < 1.0),
        "{d:?} {expect:?}"
    );

    for _ in 0..(20 * 30) {
        sim.step();
    }
    let rel = above(&sim, guid);
    let h = len(rel) - R;
    assert!(h > 0.0 && h < 5.0, "landed on the surface: {h} m above it");
    // Over the same spot (it only fell).
    assert!(rel[0].abs() < 1.0 && rel[2].abs() < 1.0, "{rel:?}");
    let v = sim.ship_state(guid).expect("alive").vel;
    assert!(
        v.iter().all(|c| c.abs() < 0.5),
        "at rest relative to it: {v:?}"
    );
    // The moon went a fifth of the way round meanwhile.
    assert!(sim.scene_time() - clock.time_us == 20_000_000);
}

/// Beyond the reach nothing is carried and nothing pulls: a ship there is
/// left exactly where it was.
#[test]
fn beyond_the_reach_nothing_is_carried() {
    let mut sim = moon_world(SceneClock::default());
    let guid = EntityGuid::new(ShardId(1), 2);
    let far = WorldPos::from_f64_metres_around(
        moon_center(&sim).to_world_pos(),
        [0.0, REACH + 10.0, 0.0],
    );
    sim.spawn_ship(guid, 1, ShipState::at_rest(far));
    for _ in 0..30 {
        sim.step();
    }
    assert_eq!(sim.ship_state(guid).expect("alive").pos, far);
    assert!(sim
        .gravity()
        .at(far, sim.tick())
        .iter()
        .all(|c| c.to_bits() == 0));
}

/// A re-anchored clock moves the moon at once — and the ship on it with it.
#[test]
fn a_clock_jump_carries_what_the_moon_holds() {
    let mut sim = moon_world(SceneClock::default());
    let guid = EntityGuid::new(ShardId(1), 3);
    let start =
        WorldPos::from_f64_metres_around(moon_center(&sim).to_world_pos(), [0.0, R + 0.9, 0.0]);
    sim.spawn_ship(guid, 1, ShipState::at_rest(start));
    for _ in 0..90 {
        sim.step();
    }
    let before = above(&sim, guid);
    // 40 ms ahead.
    let now = sim.tick();
    sim.set_scene_clock(SceneClock::new(
        now,
        SceneClock::default().time_us_at(now) + 40_000,
    ));
    sim.step();
    let after = above(&sim, guid);
    assert!(
        (0..3).all(|k| (after[k] - before[k]).abs() < 0.5),
        "{before:?} -> {after:?}"
    );
}

/// A world that jumps three seconds ahead (a shard that could not keep up:
/// `SimWorld::skip_to`) moves the moon by those three seconds on its next
/// step — 19 000 km — and the ship that stands on it with it; and the
/// inputs filed for ticks after the jump are still applied on their ticks.
///
/// Mutation this catches: jumping with `reset_clock` — scene time restarts at
/// the new tick, so the step after the jump carries the ship by one tick of
/// the moon's motion while the moon has moved by ninety-one (the ship is left
/// 19 000 km behind, in empty space), and every buffered input is dropped
/// (the ship does not lift off below).
#[test]
fn a_jump_ahead_carries_what_the_moon_holds_and_keeps_later_inputs() {
    let mut sim = moon_world(SceneClock::default());
    let guid = EntityGuid::new(ShardId(1), 4);
    let start =
        WorldPos::from_f64_metres_around(moon_center(&sim).to_world_pos(), [0.0, R + 0.9, 0.0]);
    sim.spawn_ship(guid, 1, ShipState::at_rest(start));
    for _ in 0..90 {
        sim.step();
    }
    let before = above(&sim, guid);
    let moon_before = moon_center(&sim).to_world_pos();
    let now = sim.tick();
    // Inputs for the next twelve ticks (full thrust up, away from the moon):
    // four of them are for ticks the jump passes, eight for later ones.
    let jump = 90;
    let inputs: Vec<PlayerInput> = (1..=12)
        .map(|i| fast(guid, now + jump - 4 + i, [0.0, 1.0, 0.0], false))
        .collect();
    // (Filed as a shard files them: within the input lead of the present.)
    sim.skip_to(now + jump - 8);
    sim.apply_inputs(sim.tick(), &inputs);
    assert_eq!(sim.buffered_input_count(), 12);
    sim.skip_to(now + jump);
    assert_eq!(sim.tick(), now + jump);
    assert_eq!(
        sim.buffered_input_count(),
        8,
        "the inputs for ticks after the jump are kept, the others dropped"
    );
    sim.step();
    let moved = WorldPos::distance(moon_before, moon_center(&sim).to_world_pos());
    assert!(moved > 1.0e7, "the premise: the moon went on, {moved} m");
    assert_eq!(
        sim.scene_time(),
        SceneClock::default().time_us_at(now + jump + 1),
        "scene time is the clock's time of the tick"
    );
    let after = above(&sim, guid);
    assert!(
        (0..3).all(|k| (after[k] - before[k]).abs() < 0.5),
        "the ship is where it stood on the moon: {before:?} -> {after:?}"
    );
    // The kept inputs act: seven more ticks of thrust up.
    for _ in 0..7 {
        sim.step();
    }
    assert_eq!(sim.buffered_input_count(), 0);
    let v = sim.ship_state(guid).expect("alive").vel;
    assert!(v[1] > 0.5, "the ship is lifting off: {v:?}");
    assert_eq!(sim.input_ack(1), now + jump + 8);
}

fn fast(guid: EntityGuid, tick: u64, thrust: [f32; 3], on: bool) -> PlayerInput {
    PlayerInput {
        entity: guid,
        thrust,
        torque: [0.0; 3],
        tick,
        fast: on,
    }
}

/// Fast travel: the speed limit is the distance to the nearest surface per
/// second, between the normal top speed and the cap; flying at a surface the
/// ship slows as it closes in and never hits it; leaving fast travel drops it
/// to the normal top speed at once.
#[test]
fn fast_travel_scales_with_the_distance_to_the_nearest_surface() {
    let p = ShipParams::PLACEHOLDER;
    let top = f64::from(top_speed(&p));
    assert_eq!(fast_travel_limit(10.0, true, &p), top);
    assert_eq!(fast_travel_limit(10.0, false, &p), top);
    assert_eq!(
        fast_travel_limit(2_050.0, false, &p),
        2_050.0,
        "leaving: no margin"
    );
    assert_eq!(
        fast_travel_limit(2_050.0, true, &p),
        top,
        "normal flight 2 km out"
    );
    assert_eq!(fast_travel_limit(5.0e6, true, &p), 5.0e6 - FAST_CLEARANCE);
    assert_eq!(fast_travel_limit(5.0e6, false, &p), 5.0e6);
    assert_eq!(fast_travel_limit(1.0e20, true, &p), FAST_MAX_SPEED);
    assert_eq!(fast_travel_limit(f64::NAN, false, &p), top);

    // A fixed moon 2e8 m ahead (down -Z, where a fresh ship looks).
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_gravity(GravityField {
        zones: vec![],
        bodies: vec![LargeBody {
            name: "moon".into(),
            gm: 1.62 * R * R,
            radius: R,
            place: BodyPlace::Fixed(WorldPoint::new(WorldPos::ZERO, [0.0, 0.0, -2.0e8])),
            reach: REACH,
        }],
        clock: SceneClock::default(),
    });
    let guid = EntityGuid::new(ShardId(1), 4);
    sim.spawn_ship(guid, 1, ShipState::at_rest(WorldPos::ZERO));
    let mut best = 0.0f64;
    for _ in 0..(40 * 30) {
        let t = sim.tick() + 1;
        // The pilot lets go three kilometres out (this moon has no collider).
        let here = sim.surface_distance(sim.ship_state(guid).expect("alive").pos);
        let push = if here > 3_000.0 { -1.0 } else { 0.0 };
        sim.apply_inputs(t, &[fast(guid, t, [0.0, 0.0, push], true)]);
        sim.step();
        let s = sim.ship_state(guid).expect("alive");
        let speed = len(s.vel.map(f64::from));
        let surface = sim.surface_distance(s.pos);
        assert!(surface > 0.0, "never into the moon");
        // The limit is taken before the tick's move (at most 1/30 of the
        // distance closer, so at most 30/29 of the limit at the new place).
        assert!(
            speed <= fast_travel_limit(surface, false, &p) * 1.04 + 1.0e-3,
            "{speed} m/s at {surface} m"
        );
        best = best.max(speed);
    }
    assert!(best > 1.0e7, "fast in open space: {best} m/s");
    let s = sim.ship_state(guid).expect("alive");
    let surface = sim.surface_distance(s.pos);
    assert!(
        surface > FAST_CLEARANCE && surface < 3_000.0,
        "arrived, and stopped short of the clearance: {surface} m"
    );
    // Off: at most the normal top speed at once.
    let t = sim.tick() + 1;
    sim.apply_inputs(t, &[fast(guid, t, [0.0, 0.0, -1.0], false)]);
    sim.step();
    let v = len(sim.ship_state(guid).expect("alive").vel.map(f64::from));
    assert!(v <= top + 1.0, "{v}");
}

/// Without the flag nothing changes: a ship in normal flight tops out at
/// 120 m/s whatever the distance to anything.
#[test]
fn normal_flight_is_unchanged() {
    let mut sim = SimWorld::new(ShardId(1));
    let guid = EntityGuid::new(ShardId(1), 5);
    sim.spawn_ship(guid, 1, ShipState::at_rest(WorldPos::ZERO));
    for _ in 0..(60 * 30) {
        let t = sim.tick() + 1;
        sim.apply_inputs(t, &[fast(guid, t, [0.0, 0.0, -1.0], false)]);
        sim.step();
    }
    let v = len(sim.ship_state(guid).expect("alive").vel.map(f64::from));
    assert!(v > 110.0 && v <= 120.0, "{v}");
}
