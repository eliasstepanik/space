//! Large-body gravity with a reach, end to end through the simulation: a
//! body in a circular orbit around a small planetoid keeps the Kepler period
//! — near the world origin and 1e13 m out alike — and the planetoid's reach
//! leaves the space beyond it weightless.

use space_protocol::{EntityGuid, ShardId};
use space_sim::gravity::BodyPlace;
use space_sim::{
    Body, BodySpawn, GravityField, LargeBody, MassSpec, ShapeRef, ShipState, SimWorld, WorldPoint,
};
use space_world::WorldPos;

/// The demo area's planetoid: 4 m in radius, 4 m/s² on its surface, a reach
/// of 16 m.
const GM: f64 = 64.0;
const RADIUS: f64 = 4.0;
const REACH: f64 = 16.0;

fn planetoid(center: WorldPos) -> GravityField {
    GravityField {
        zones: vec![],
        bodies: vec![LargeBody {
            name: "planetoid".into(),
            gm: GM,
            radius: RADIUS,
            place: BodyPlace::Fixed(WorldPoint::at(center)),
            reach: REACH,
        }],
        clock: Default::default(),
    }
}

fn ball() -> Body {
    let mut b = Body::dynamic(ShapeRef::Ball { radius: 0.25 });
    b.mass = MassSpec::Mass(5.0);
    b.linear_damping = 0.0;
    b.angular_damping = 0.0;
    b
}

/// Fly a ball in a circular orbit of radius `r` around a planetoid at
/// `center` for `ticks` ticks; returns the times (seconds) at which it
/// completed each revolution, and its distances from the centre.
fn orbit(center: WorldPos, r: f64, ticks: u64) -> (Vec<f64>, Vec<f64>, [f64; 3]) {
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_gravity(planetoid(center));
    let guid = EntityGuid::new(ShardId(1), 7);
    let v = (GM / r).sqrt();
    let mut s = ShipState::at_rest(WorldPos::from_f64_metres_around(center, [r, 0.0, 0.0]));
    s.vel = [0.0, 0.0, v as f32];
    sim.spawn_body(guid, BodySpawn::new(s, ball()))
        .expect("spawns");
    let dt = f64::from(space_sim::dt());
    let (mut laps, mut radii) = (Vec::new(), Vec::new());
    let mut last_angle = 0.0f64;
    let mut turned = 0.0f64;
    let mut last = [r, 0.0, 0.0];
    for tick in 1..=ticks {
        sim.step();
        let d = WorldPos::delta(center, sim.state_of(guid).expect("alive").pos);
        radii.push((d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt());
        let angle = d[2].atan2(d[0]);
        let mut step = angle - last_angle;
        if step < -std::f64::consts::PI {
            step += std::f64::consts::TAU;
        }
        let before = turned;
        turned += step;
        let lap = std::f64::consts::TAU * (laps.len() + 1) as f64;
        if before < lap && turned >= lap {
            // Interpolate the moment the lap closed.
            let f = (lap - before) / (turned - before);
            laps.push((tick as f64 - 1.0 + f) * dt);
        }
        last_angle = angle;
        last = d;
    }
    (laps, radii, last)
}

/// The Kepler period of a circular orbit of radius `r`.
fn kepler(r: f64) -> f64 {
    std::f64::consts::TAU * (r * r * r / GM).sqrt()
}

/// Several revolutions at 7 m (inside the reach's full-strength part): every
/// lap within 1 % of 2π√(r³/GM), the radius steady — and the same orbit 1e13 m
/// from the world origin flies the same path, to well under a millimetre.
#[test]
fn an_orbit_keeps_the_kepler_period_here_and_far_away() {
    let r = 7.0;
    let period = kepler(r);
    let ticks = (period * 4.2 / f64::from(space_sim::dt())) as u64;
    let near = WorldPos::from_f64_metres_around(WorldPos::ZERO, [3.0, 10.0, -46.0]);
    let far = WorldPos::from_f64_metres_around(WorldPos::ZERO, [1.0e13, 7.0, -3.0]);
    let mut ends = Vec::new();
    for center in [near, far] {
        let (laps, radii, end) = orbit(center, r, ticks);
        assert!(laps.len() >= 4, "{} laps", laps.len());
        let mut previous = 0.0;
        for (i, t) in laps.iter().enumerate() {
            let lap = t - previous;
            previous = *t;
            assert!(
                (lap - period).abs() / period < 0.01,
                "lap {i}: {lap:.3} s, Kepler {period:.3} s"
            );
        }
        let (lo, hi) = radii
            .iter()
            .fold((f64::MAX, f64::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
        assert!(lo > r * 0.98 && hi < r * 1.02, "radius {lo}..{hi}");
        println!(
            "orbit at {:?}: {} laps, mean {:.4} s (Kepler {period:.4} s), radius {lo:.4}..{hi:.4}",
            WorldPos::delta(WorldPos::ZERO, center),
            laps.len(),
            laps.last().unwrap() / laps.len() as f64
        );
        ends.push(end);
    }
    let gap: f64 = (0..3)
        .map(|k| (ends[0][k] - ends[1][k]).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        gap < 1.0e-3,
        "the far orbit ended {gap} m from the near one"
    );
}

/// Beyond the reach a body feels nothing: a ball at rest 20 m from the
/// planetoid stays exactly where it is, its state unchanged to the bit.
#[test]
fn beyond_the_reach_nothing_moves() {
    let center = WorldPos::from_f64_metres_around(WorldPos::ZERO, [0.0, 10.0, 0.0]);
    let mut sim = SimWorld::new(ShardId(1));
    sim.set_gravity(planetoid(center));
    let guid = EntityGuid::new(ShardId(1), 8);
    let start = ShipState::at_rest(WorldPos::from_f64_metres_around(center, [0.0, 0.0, -20.0]));
    sim.spawn_body(guid, BodySpawn::new(start, ball()))
        .expect("spawns");
    sim.step_n(300);
    assert_eq!(sim.state_of(guid).unwrap(), start);
    let g = sim.gravity().at(start.pos, 300);
    assert!(g.iter().all(|c| c.to_bits() == 0), "{g:?}");
}
