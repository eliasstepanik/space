//! The physics budget: the demo area (statics, Kestrel, showcase) plus 200
//! extra dynamic bodies and 32 ships flying among them, stepped at 30 Hz.
//!
//! Run in release to get the number that matters:
//! `cargo test --release -p space-scene --test perf -- --nocapture`.
//! In release the mean must stay under 5 ms per tick; a debug build only
//! runs a few ticks (it is ~30x slower and the number means nothing).

use std::time::{Duration, Instant};

use space_protocol::{EntityGuid, PlayerInput, Scene, ShardId};
use space_sim::body::{Body, MassSpec};
use space_sim::shape::ShapeRef;
use space_sim::{BodySpawn, ShipState, SimWorld};
use space_world::WorldPos;

#[test]
fn demo_area_with_200_bodies_and_32_ships() {
    let anchor = WorldPos::from_metres([4_000_000_000, 0, -9_000_000]);
    let at = |p: [f64; 3]| WorldPos::from_f64_metres_around(anchor, p);
    let f = space_scene::demo::FLOOR;
    let mut sim = SimWorld::new(ShardId(1));
    space_scene::install_environment(&mut sim, Scene::DemoArea, anchor).unwrap();
    let mut serial = 0u64;
    let mut mint = || {
        serial += 1;
        EntityGuid::new(ShardId(1), serial)
    };
    space_scene::spawn_bodies(&mut sim, Scene::DemoArea, anchor, &mut mint).unwrap();
    // 120 crates and balls floating in a grid over the weightless front of the
    // area, where the ships fly through them.
    let mut extra = 0;
    for i in 0..120 {
        let (gx, gy, gz) = (i % 12, (i / 12) % 2, i / 24);
        let p = [
            -17.0 + 1.3 * f64::from(gx),
            0.2 + 1.2 * f64::from(gy),
            -2.0 + 1.6 * f64::from(gz),
        ];
        let body = if i % 2 == 0 {
            Body::dynamic(ShapeRef::cuboid([0.6; 3]))
        } else {
            Body::dynamic(ShapeRef::Ball { radius: 0.3 })
        };
        sim.spawn_body(mint(), BodySpawn::new(ShipState::at_rest(at(p)), body))
            .unwrap();
        extra += 1;
    }
    // 80 crates dropped into the gravity room from above: they pile onto the
    // showcase.
    for i in 0..80 {
        let (gx, gy, gz) = (i % 4, (i / 4) % 4, i / 16);
        let p = [
            11.5 + 1.8 * f64::from(gx),
            f + 3.0 + 0.8 * f64::from(gy),
            -8.0 + 2.6 * f64::from(gz),
        ];
        let mut b = Body::dynamic(ShapeRef::cuboid([0.5; 3]));
        b.mass = MassSpec::Density(400.0);
        sim.spawn_body(mint(), BodySpawn::new(ShipState::at_rest(at(p)), b))
            .unwrap();
        extra += 1;
    }
    assert_eq!(extra, 200);
    // 32 ships at eye height around the area.
    let ships: Vec<EntityGuid> = (0..32)
        .map(|i: u32| {
            let g = mint();
            let a = f64::from(i) / 32.0 * std::f64::consts::TAU;
            sim.spawn_ship(
                g,
                100 + u64::from(i),
                ShipState::at_rest(at([12.0 * a.cos(), 0.5, -6.0 + 12.0 * a.sin()])),
            );
            g
        })
        .collect();
    let ticks: u64 = if cfg!(debug_assertions) { 20 } else { 450 };
    let mut times: Vec<Duration> = Vec::new();
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = || {
        seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    };
    for t in 1..=ticks {
        let inputs: Vec<PlayerInput> = ships
            .iter()
            .map(|g| PlayerInput {
                entity: *g,
                thrust: [rnd(), rnd() * 0.2, rnd()],
                torque: [0.0, rnd() * 0.3, 0.0],
                tick: t,
            })
            .collect();
        sim.apply_inputs(t, &inputs);
        let start = Instant::now();
        sim.step();
        times.push(start.elapsed());
    }
    let mean = times.iter().sum::<Duration>() / u32::try_from(times.len()).unwrap();
    let mut sorted = times.clone();
    sorted.sort();
    let p99 = sorted[sorted.len() * 99 / 100];
    let max = sorted[sorted.len() - 1];
    let bubbles = sim.bubbles();
    println!(
        "{} entities, {} bubbles ({} stepped): mean {:.3} ms/tick, p99 {:.3} ms, max {:.3} ms over {} ticks",
        sim.bodies().len(),
        bubbles.len(),
        bubbles.iter().filter(|b| b.stepped).count(),
        mean.as_secs_f64() * 1e3,
        p99.as_secs_f64() * 1e3,
        max.as_secs_f64() * 1e3,
        ticks
    );
    if !cfg!(debug_assertions) {
        assert!(mean < Duration::from_millis(5), "mean {mean:?} per tick");
    }
}
