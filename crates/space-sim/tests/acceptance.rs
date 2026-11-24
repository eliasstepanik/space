//! Acceptance tests for `space-sim` (Phase 1, task 1.4).
//!
//! Run with `cargo test -p space-sim` (or `cargo xtask ci`).
//!
//! | test | checks |
//!      |--------|--------|
//! | [`determinism_10k_ticks`] | two independent worlds, same seed + same 10 000-tick script → identical `state_hash()` at ticks 100/1000/10000; a third run with each tick's input Vec shuffled → same hash |
//! | [`bit_identity`] | the two worlds' full serialised state is byte-equal at tick 10000 |
//! | [`step_ship_physics`] | thrust/drag/torque/coast match the closed forms within 1 % |
//! | [`prediction_reconcile`] | prediction then authority-equals-prediction → correction ≈ 0; authority offset by 0.05 m → correction ≈ 0.05 and pending inputs survive |
//! | [`prediction_correction_bounded`] | seeded fuzz, 1000 ticks, 10 % input loss → max reconcile correction < 0.10 m (H3 budget) |
//! | [`orbits_deterministic`] | same (system, body, tick) → same `WorldPos` across two `StarSystem` instances; periodicity |
//! | [`hash_sensitivity`] | one bit of one entity's velocity changes `state_hash()` |

use space_ecs::ComponentDef;
use space_protocol::{EntityGuid, PlayerInput, ShardId};
use space_sim::params::ShipParams;
use space_sim::physics::step_ship;
use space_sim::prediction::PredictedShip;
use space_sim::tick::{dt, seconds_of, SimTick};
use space_sim::{orbits, ShipState, SimWorld};
use space_world::{StarSystem, WorldPos};

/// Seed for every seeded pseudo-random script in this file.
const SEED: u64 = 0x5EA1_2345_6789_ABCD;

/// SplitMix64 (same mixer `space-world` uses): integer-only, so the generated
/// input scripts are bit-identical on every platform.
fn splitmix64(mut x: u64) -> (u64, u64) {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (x, z ^ (z >> 31))
}

/// A deterministic pseudo-random generator over `u64`.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let (state, out) = splitmix64(self.0);
        self.0 = state;
        out
    }

    /// A float in `[-1, 1]` with 1e-6 resolution, derived from integers only.
    fn next_signed(&mut self) -> f32 {
        let u = self.next_u64() % 2_000_001;
        (u as f64 / 1_000_000.0 - 1.0) as f32
    }
}

/// The ships used by the multi-entity tests.
const SHIPS: u64 = 8;

fn guid(serial: u64) -> EntityGuid {
    EntityGuid::new(ShardId(7), serial)
}

/// Build a world with `SHIPS` ships at spread-out starting positions.
fn build_world() -> SimWorld {
    let mut sim = SimWorld::with_params(ShardId(7), ShipParams::PLACEHOLDER);
    for i in 0..SHIPS {
        // A deterministic, well-separated start so the ships' float magnitudes
        // differ (a shared origin would hide precision bugs).
        let pos = WorldPos::from_metres([i128::from(i) * 3_000, -i128::from(i) * 7_500, 1_200]);
        let mut state = ShipState::at_rest(pos);
        // ~45° about +y as a unit quaternion (cos 22.5°, sin 22.5°).
        state.quat = [0.923_879_5, 0.0, 0.382_683_4, 0.0];
        sim.spawn_ship(guid(i), 1_000 + i, state);
    }
    sim
}

/// One tick's worth of pseudo-random inputs: one per ship, with a random
/// magnitude on each axis (so the clamp path is exercised too).
fn tick_inputs(rng: &mut Rng, tick: SimTick) -> Vec<PlayerInput> {
    (0..SHIPS)
        .map(|i| PlayerInput {
            entity: guid(i),
            thrust: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
            torque: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
            tick,
            fast: false,
        })
        .collect()
}

/// Reverse a Vec (a cheap, deterministic "shuffle" that changes the order every
/// element appears in).
fn reverse_in_place(inputs: &mut [PlayerInput]) {
    inputs.reverse();
}

/// Rotate a Vec by one (a second, different order).
fn rotate_by_one(inputs: &mut Vec<PlayerInput>) {
    if !inputs.is_empty() {
        let last = inputs.pop().unwrap();
        inputs.insert(0, last);
    }
}

// ---------------------------------------------------------------------------
// determinism
// ---------------------------------------------------------------------------

/// Two independent `SimWorld`s driven by the same seeded 10 000-tick input
/// script must hash identically at ticks 100, 1000 and 10000.
///
/// A third run applies the *same* inputs but with each tick's `Vec` in a
/// different order, which only stays equal because `apply_inputs` sorts by
/// entity before applying.
#[test]
fn determinism_10k_ticks() {
    const TICKS: SimTick = 10_000;
    let checkpoints = [100_u64, 1_000, 10_000];

    let mut a = build_world();
    let mut b = build_world();
    let mut c = build_world();
    assert_eq!(
        a.state_hash(),
        b.state_hash(),
        "identical worlds must start equal"
    );

    let mut rng_a = Rng::new(SEED);
    let mut rng_b = Rng::new(SEED);
    let mut rng_c = Rng::new(SEED);

    for tick in 1..=TICKS {
        let ia = tick_inputs(&mut rng_a, tick);
        let ib = tick_inputs(&mut rng_b, tick);
        let mut ic = tick_inputs(&mut rng_c, tick);
        // Same inputs, different order within the tick's Vec.
        if tick % 2 == 0 {
            reverse_in_place(&mut ic);
        } else {
            rotate_by_one(&mut ic);
        }
        assert_eq!(ia, ib, "the two generators must agree");

        a.apply_inputs(tick, &ia);
        a.step();
        b.apply_inputs(tick, &ib);
        b.step();
        c.apply_inputs(tick, &ic);
        c.step();

        if checkpoints.contains(&tick) {
            let (ha, hb, hc) = (a.state_hash(), b.state_hash(), c.state_hash());
            assert_eq!(ha, hb, "hashes diverged at tick {tick}");
            assert_eq!(ha, hc, "input order leaked into the hash at tick {tick}");
            assert_ne!(ha, 0);
        }
    }
    assert_eq!(a.tick(), TICKS);
    assert_eq!(a.state_hash(), b.state_hash());
    assert_eq!(a.state_hash(), c.state_hash());
}

/// The full serialised state of both worlds is byte-equal at tick 10 000 — not
/// just equal under the hash, but bit-identical on the wire.
#[test]
fn bit_identity() {
    const TICKS: SimTick = 10_000;
    let mut a = build_world();
    let mut b = build_world();
    let mut rng_a = Rng::new(SEED);
    let mut rng_b = Rng::new(SEED);
    for tick in 1..=TICKS {
        let ia = tick_inputs(&mut rng_a, tick);
        let ib = tick_inputs(&mut rng_b, tick);
        a.apply_inputs(tick, &ia);
        a.step();
        b.apply_inputs(tick, &ib);
        b.step();
    }
    let sa = a.serialize_state();
    let sb = b.serialize_state();
    assert_eq!(sa.len(), sb.len());
    assert_eq!(sa, sb, "serialised state diverged at tick {TICKS}");
    // The serialisation is not degenerate: it carries every ship and grows with
    // the simulation (a 10 000-tick run of 8 ships is well over a kilobyte).
    assert!(
        sa.len() > SHIPS as usize * 60,
        "state too small: {}",
        sa.len()
    );
    assert_eq!(a.state_hash(), b.state_hash());
}

/// A single bit of one entity's velocity changes `state_hash()`.
#[test]
fn hash_sensitivity() {
    let mut a = build_world();
    let b = build_world();
    assert_eq!(a.state_hash(), b.state_hash());

    // Flip the lowest bit of the x-velocity of the third ship.
    let target = a.ships()[2];
    {
        let state = a.world_mut().get_mut::<ShipState>(target).unwrap();
        let bits = state.vel[0].to_bits() ^ 1;
        state.vel[0] = f32::from_bits(bits);
    }
    assert_ne!(
        a.state_hash(),
        b.state_hash(),
        "one bit of one velocity must change the hash"
    );

    // And a bit of the position's cell index, far from the local part.
    let mut c = build_world();
    let d = build_world();
    let target = c.ships()[0];
    {
        let state = c.world_mut().get_mut::<ShipState>(target).unwrap();
        state.pos.cell[1] ^= 1 << 40;
    }
    assert_ne!(c.state_hash(), d.state_hash());
}

// ---------------------------------------------------------------------------
// physics
// ---------------------------------------------------------------------------

/// From rest, full thrust for 1 s gives the closed-form speed of a linearly
/// damped body, within 1 %.
///
/// The exact solution of `v̇ = F/m − k v` is `v(t) = F/(m·k) · (1 − e^{−k t})`.
/// (The brief writes `thrust/mass·(1−e^{−drag·t})`, which is the same expression
/// only for `drag_lin = 1`; the `1/k` factor is required for the terminal speed
/// to come out at `thrust/(mass·drag_lin)`, so the test uses the dimensionally
/// correct form.)
#[test]
fn step_ship_physics() {
    let p = ShipParams::PLACEHOLDER;
    let rest = ShipState::IDENTITY;
    let full = PlayerInput {
        entity: EntityGuid::new(ShardId(1), 1),
        thrust: [1.0, 0.0, 0.0],
        torque: [0.0; 3],
        tick: 0,
        fast: false,
    };

    // --- linear: 1 s of full thrust from rest -------------------------------
    let mut s = rest;
    let one_second: u32 = 30;
    let mut travelled = 0.0f64;
    let start = s.pos;
    for tick in 1..=one_second {
        step_ship(&mut s, &full, &p);
        assert!(s.vel[0] > 0.0, "must accelerate on tick {tick}");
        travelled = WorldPos::distance(start, s.pos);
    }
    let t = f64::from(one_second) * f64::from(dt());
    let k = f64::from(p.drag_lin);
    let terminal = f64::from(p.thrust) / (f64::from(p.mass) * k);
    let speed = f64::from((s.vel[0] * s.vel[0] + s.vel[1] * s.vel[1] + s.vel[2] * s.vel[2]).sqrt());
    // Continuous closed form, within 1 %.
    let expected = terminal * (1.0 - (-k * t).exp());
    assert!(
        (speed - expected).abs() / expected < 0.01,
        "speed after 1 s = {speed}, closed form {expected}"
    );
    assert!(speed < terminal, "must not overshoot the terminal speed");
    // Semi-implicit Euler solves `v_{n+1} = v_n(1 − k dt) + (F/m) dt` exactly:
    // v_n = (F/(m k))·(1 − (1 − k dt)^n). That discrete form must match to
    // floating-point precision, which is the real integrator check.
    let n = f64::from(one_second);
    let r = 1.0 - k * f64::from(dt());
    let discrete_speed = terminal * (1.0 - r.powf(n));
    assert!(
        (speed - discrete_speed).abs() / discrete_speed < 1e-5,
        "speed {speed} is not the discrete solution {discrete_speed}"
    );
    // Distance = dt · Σ_{j=1..n} v_j = dt·(F/(m k))·(n − r(1 − r^n)/(1 − r)).
    let expected_dist = f64::from(dt()) * terminal * (n - r * (1.0 - r.powf(n)) / (1.0 - r));
    assert!(
        (travelled - expected_dist).abs() / expected_dist < 1e-5,
        "distance after 1 s = {travelled}, discrete closed form {expected_dist}"
    );

    // --- no input: coasts to a stop ----------------------------------------
    let mut coast = s;
    let idle = PlayerInput {
        thrust: [0.0; 3],
        torque: [0.0; 3],
        ..full
    };
    let v0 = f64::from(coast.vel[0]);
    for _ in 0..300 {
        step_ship(&mut coast, &idle, &p);
    }
    let t = f64::from(300) * f64::from(dt());
    let expected = v0 * (-f64::from(p.drag_lin) * t).exp();
    let speed = f64::from(coast.vel[0]);
    assert!(
        (speed - expected).abs() / expected < 0.01,
        "coast speed = {speed}, closed form {expected}"
    );
    // Linear damping is asymptotic, so "stop" is a threshold, not zero: after
    // 6 000 more ticks (200 s, ten time constants) the speed is e^{-10} ≈ 5e-5
    // of what it was.
    for _ in 0..6_000 {
        step_ship(&mut coast, &idle, &p);
    }
    assert!(
        f64::from(coast.vel[0]).abs() < v0 * 1e-4,
        "must coast to a stop: {} from {v0}",
        coast.vel[0]
    );

    // --- torque: spins up to the terminal rate, then decays ----------------
    let spin = PlayerInput {
        thrust: [0.0; 3],
        torque: [0.0, 1.0, 0.0],
        ..full
    };
    let mut s = rest;
    for _ in 0..30 {
        step_ship(&mut s, &spin, &p);
    }
    let w_terminal = f64::from(p.torque) / f64::from(p.drag_ang);
    assert!(
        s.ang_vel[1] > 0.0 && f64::from(s.ang_vel[1]) < w_terminal,
        "spin-up out of range: {} vs terminal {w_terminal}",
        s.ang_vel[1]
    );
    // The yaw quaternion actually turned.
    assert!(s.quat[2] > 0.0, "quat must follow the torque: {:?}", s.quat);
    let norm = (s.quat[0] * s.quat[0]
        + s.quat[1] * s.quat[1]
        + s.quat[2] * s.quat[2]
        + s.quat[3] * s.quat[3])
        .abs();
    assert!((norm - 1.0).abs() < 1e-5, "quat must stay unit: {norm}");
    // Release: angular drag decays the rate.
    let w0 = f64::from(s.ang_vel[1]);
    for _ in 0..30 {
        step_ship(&mut s, &idle, &p);
    }
    let decayed = f64::from(s.ang_vel[1]);
    assert!(
        decayed.abs() < w0 * 0.5,
        "angular drag must bleed the spin: {decayed} from {w0}"
    );
    for _ in 0..600 {
        step_ship(&mut s, &idle, &p);
    }
    assert!(f64::from(s.ang_vel[1]).abs() < 1e-6, "must stop spinning");

    // --- clamped input: a diagonal burn is not √2 stronger -----------------
    let mut diag = ShipState::IDENTITY;
    let mut straight = ShipState::IDENTITY;
    let diagonal = PlayerInput {
        thrust: [1.0, 1.0, 0.0],
        torque: [0.0; 3],
        ..full
    };
    for _ in 0..30 {
        step_ship(&mut diag, &diagonal, &p);
        step_ship(&mut straight, &full, &p);
    }
    let d = f64::from((diag.vel[0] * diag.vel[0] + diag.vel[1] * diag.vel[1]).sqrt());
    let s_ = f64::from(straight.vel[0]);
    assert!(
        (d - s_).abs() / s_ < 0.01,
        "clamped diagonal speed {d} should match straight burn {s_}"
    );
}

// ---------------------------------------------------------------------------
// prediction
// ---------------------------------------------------------------------------

/// Predict 5 ticks ahead, then feed authority equal to what the prediction
/// produced → correction ≈ 0. Feed authority offset by 0.05 m → correction
/// ≈ 0.05 and the pending inputs survive.
#[test]
fn prediction_reconcile() {
    let params = ShipParams::PLACEHOLDER;
    let mut rng = Rng::new(SEED ^ 0xC11E);
    let mut pred = PredictedShip::new(ShipState::IDENTITY, 0, params);
    for tick in 1..=8 {
        pred.apply_input(PlayerInput {
            entity: guid(1),
            thrust: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
            torque: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
            tick,
            fast: false,
        });
    }
    assert_eq!(pred.pending_ticks(), 8);

    // --- authority == prediction → no correction ---------------------------
    let predicted_5 = pred.predict(5);
    assert_ne!(
        predicted_5,
        ShipState::IDENTITY,
        "prediction must move the ship"
    );
    let correction = pred.reconcile(predicted_5, 5, 5);
    assert!(
        correction < 1e-6,
        "authority equal to the prediction must need no correction, got {correction}"
    );
    assert_eq!(pred.last_tick, 5);
    assert_eq!(
        pred.pending_ticks(),
        3,
        "the 3 unacknowledged inputs must survive"
    );
    assert_eq!(pred.pending.front().map(|i| i.tick), Some(6));
    // The surviving inputs still drive the next prediction.
    let ahead = pred.predict(8);
    assert_ne!(ahead, pred.authoritative);

    // --- authority offset by 5 cm → correction ≈ 5 cm ----------------------
    let mut offset = pred.predict(8);
    offset.pos = offset.pos.offset([0.05, 0.0, 0.0]);
    let correction = pred.reconcile(offset, 8, 8);
    assert!(
        (correction - 0.05).abs() < 1e-4,
        "correction was {correction}, expected ≈0.05"
    );
    assert_eq!(pred.authoritative, offset);
    assert_eq!(pred.last_tick, 8);
    assert_eq!(pred.pending_ticks(), 0);
}

/// Seeded fuzz: 1 000 ticks of random inputs, 10 % of them dropped on the way to
/// the server, snapshots every 3 ticks. The largest reconcile correction must
/// stay under the **0.10 m** handoff H3 budget.
#[test]
fn prediction_correction_bounded() {
    const TICKS: SimTick = 1_000;
    /// Ticks between authoritative snapshots (the client predicts this far ahead).
    const SNAPSHOT_EVERY: SimTick = 3;
    /// Percent of inputs that never reach the server.
    const LOSS_PERCENT: u64 = 10;
    /// The H3 budget.
    const BUDGET_M: f32 = 0.10;

    let params = ShipParams::PLACEHOLDER;
    let mut server = SimWorld::with_params(ShardId(7), params);
    let start = ShipState::at_rest(WorldPos::from_metres([12_345_000, -987_654_321, 4_000]));
    server.spawn_ship(guid(1), 42, start);
    let mut client = PredictedShip::new(start, 0, params);

    let mut rng = Rng::new(SEED ^ 0xB0B);
    let mut max_correction = 0.0f32;
    let mut dropped = 0usize;

    for tick in 1..=TICKS {
        let input = PlayerInput {
            entity: guid(1),
            thrust: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
            torque: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
            tick,
            fast: false,
        };
        // The client always predicts as if its input arrived.
        client.apply_input(input);
        // The server sees it with probability 1 - LOSS_PERCENT.
        let delivered = rng.next_u64() % 100 >= LOSS_PERCENT;
        let batch: Vec<PlayerInput> = if delivered { vec![input] } else { vec![] };
        if !delivered {
            dropped += 1;
        }
        server.apply_inputs(tick, &batch);
        server.step();

        if tick % SNAPSHOT_EVERY == 0 || tick == TICKS {
            let authority = server.ship_state(guid(1)).expect("ship is alive");
            let ack = server.input_ack(42);
            let correction = client.reconcile(authority, tick, ack);
            max_correction = max_correction.max(correction);
        }
    }

    assert!(dropped > TICKS as usize / 20, "the fuzz must drop inputs");
    assert!(
        max_correction < BUDGET_M,
        "max reconcile correction {max_correction} m exceeded the {BUDGET_M} m H3 budget"
    );
    // The client's view converged to the server's: after a reconcile the
    // authoritative state is exactly the server's.
    assert_eq!(
        client.authoritative,
        server.ship_state(guid(1)).unwrap(),
        "the client must end on the server's state"
    );
}

// ---------------------------------------------------------------------------
// prediction under latency + reordering + loss (B1)
// ---------------------------------------------------------------------------

/// The harder reconciliation fuzz the reviewer asked for: **N-tick one-way
/// latency**, **shuffled delivery order** and **~10 % loss**, driving a real
/// [`SimWorld`] server (which now buffers inputs by tick) against a
/// [`PredictedShip`] client (which now reconciles by `input_ack`).
///
/// The client predicts `LATENCY` ticks ahead of the server — the only way an
/// input can be in the server's buffer *before* the tick it names. Each input is
/// generated `LATENCY` ticks early, delivered to the server's arrival queue with
/// a little jitter, and the queue is **shuffled** before being handed to
/// `apply_inputs`, so the server sees them out of order. ~10 % never arrive.
///
/// Convergence: every `SNAPSHOT_EVERY` ticks the server sends its state at tick
/// `w` together with `input_ack = server.input_ack(owner)`; that snapshot
/// reaches the client `LATENCY` ticks later. `reconcile` drops every pending
/// input with `tick <= input_ack`, which is exactly the set the server has
/// already stepped in (including the ones the client is still predicting that the
/// server *lost* — the ack retires them). The client's authoritative state must
/// therefore equal the server's after each snapshot, and the largest correction
/// must stay under the **0.10 m** H3 budget.
#[test]
fn prediction_converges_under_latency_reorder_and_loss() {
    const TICKS: SimTick = 1_000;
    /// One-way latency, in ticks (the client predicts this far ahead).
    const LATENCY: SimTick = 4;
    /// Ticks between authoritative snapshots.
    const SNAPSHOT_EVERY: SimTick = 3;
    /// Percent of inputs that never reach the server.
    const LOSS_PERCENT: u64 = 10;
    /// The H3 budget.
    const BUDGET_M: f32 = 0.10;
    const OWNER: u64 = 42;

    let params = ShipParams::PLACEHOLDER;
    let mut server = SimWorld::with_params(ShardId(7), params);
    let start = ShipState::at_rest(WorldPos::from_metres([987_654_321, -12_345_000, -6_000]));
    server.spawn_ship(guid(1), OWNER, start);
    // The client's authority starts at tick 0 (the same start state).
    let mut client = PredictedShip::new(start, 0, params);

    let mut rng = Rng::new(SEED ^ 0x1a7e);
    let mut max_correction = 0.0f32;
    let mut dropped = 0usize;
    let mut reordered = 0usize;
    // The lowest input tick that was ever lost (`None` = nothing lost yet). A
    // snapshot for tick `w` with `first_lost.is_none_or(|l| l > w)` had every
    // input up to `w` delivered, so the client's replay is exactly the server's
    // step sequence and its correction must be **exactly `0.0`** — not merely
    // under the budget.
    let mut first_lost: Option<SimTick> = None;
    // How many loss-free snapshots were checked for an exactly-zero correction.
    let mut exact_zero_checks = 0usize;

    // Inputs generated early, waiting to be delivered to the server. Each entry
    // is `(arrival_wall_tick, input)`; the input's own `tick` is `arrival - jitter`
    // so it lands on the tick it names.
    let mut arrivals: Vec<(SimTick, PlayerInput)> = Vec::new();
    // Snapshots in flight: `(arrival_wall_tick, authority_state, authority_tick,
    // input_ack, no_loss_so_far)`.
    let mut in_flight: Vec<(SimTick, ShipState, SimTick, SimTick, bool)> = Vec::new();

    // The client runs `LATENCY` ticks ahead of the server, so the wall-clock loop
    // runs to `TICKS + LATENCY` to let the last snapshots land.
    for wall in 1..=(TICKS + LATENCY) {
        // --- client generates an input `LATENCY` ticks ahead (if in range) ----
        let client_tick = wall + LATENCY - 1;
        if client_tick <= TICKS {
            let input = PlayerInput {
                entity: guid(1),
                thrust: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
                torque: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
                tick: client_tick,
                fast: false,
            };
            // The client always predicts as if its input arrived.
            client.apply_input(input);
            // Delivery: ~10 % lost; the rest arrive at `wall` or `wall + 1`
            // (jitter), so a later tick's input can be delivered before an earlier
            // one that jittered — reordering the server must survive.
            if rng.next_u64() % 100 >= LOSS_PERCENT {
                let jitter = rng.next_u64() % 2;
                arrivals.push((wall + jitter, input));
            } else {
                dropped += 1;
                first_lost = Some(first_lost.map_or(client_tick, |l| l.min(client_tick)));
            }
        }

        // --- server steps (if it still has ticks to run) ---------------------
        if wall <= TICKS {
            // Collect everything due by `wall`, shuffle it, hand it over.
            let mut due: Vec<PlayerInput> = Vec::new();
            arrivals.retain(|(when, input)| {
                if *when <= wall {
                    due.push(*input);
                    false
                } else {
                    true
                }
            });
            if due.len() > 1 {
                // Deterministic shuffle (Fisher-Yates with the seeded rng).
                for i in (1..due.len()).rev() {
                    let j = (rng.next_u64() % (i as u64 + 1)) as usize;
                    due.swap(i, j);
                }
                reordered += 1;
            }
            server.apply_inputs(wall, &due);
            server.step();

            // Queue a snapshot; it reaches the client `LATENCY` ticks later.
            // Every input up to `wall` has now been generated and its fate
            // decided (inputs are generated `LATENCY - 1` ticks early), so we can
            // record whether the window was loss-free.
            if wall % SNAPSHOT_EVERY == 0 || wall == TICKS {
                let authority = server.ship_state(guid(1)).expect("ship is alive");
                let ack = server.input_ack(OWNER);
                let no_loss = first_lost.is_none_or(|l| l > wall);
                in_flight.push((wall + LATENCY, authority, wall, ack, no_loss));
            }
        }

        // --- deliver snapshots that have arrived ------------------------------
        in_flight.retain(|(when, state, tick, ack, no_loss)| {
            if *when > wall {
                return true;
            }
            let correction = client.reconcile(*state, *tick, *ack);
            max_correction = max_correction.max(correction);
            if *no_loss {
                assert_eq!(
                    correction, 0.0,
                    "snapshot at tick {tick} had every input up to it delivered, \
                     so the correction must be exactly 0.0, not {correction}"
                );
                exact_zero_checks += 1;
            }
            false
        });
    }

    assert!(dropped > TICKS as usize / 20, "the fuzz must drop inputs");
    assert!(reordered > 10, "the fuzz must reorder deliveries");
    assert!(
        exact_zero_checks > 0,
        "the fuzz must include at least one loss-free snapshot to check for an \
         exactly-zero correction (first_lost = {first_lost:?})"
    );
    assert!(
        max_correction < BUDGET_M,
        "max reconcile correction {max_correction} m exceeded the {BUDGET_M} m H3 budget"
    );
    // Convergence: the client's last authority is the server's final state.
    assert_eq!(
        client.authoritative,
        server.ship_state(guid(1)).unwrap(),
        "the client must end on the server's state"
    );
    // B1b: the client's *prediction* must be checked, not just its authority
    // (which `reconcile` copies in and so is equal by construction). Predict `k`
    // ticks past the last snapshot — a non-zero number of steps, so the
    // `steps == 0` shortcut is not taken — and compare against a SEPARATE server
    // seeded with the client's authority and stepped `k` more ticks with the
    // input the client keeps holding (its last pending input, or a coast if it
    // has none). This is the input a live client keeps sending while it waits, and
    // it is exactly what `predict` replays past the end of the pending queue.
    const K: SimTick = 3;
    let final_tick = server.tick();
    let predicted = client.predict(final_tick + K);
    let held = client
        .pending
        .back()
        .map(|i| (i.thrust, i.torque))
        .unwrap_or(([0.0; 3], [0.0; 3]));
    let mut truth = SimWorld::with_tick(ShardId(7), params, client.last_tick);
    truth.spawn_ship(guid(1), OWNER, client.authoritative);
    for step in 1..=K {
        let tick = client.last_tick + step;
        let input = PlayerInput {
            entity: guid(1),
            thrust: held.0,
            torque: held.1,
            tick,
            fast: false,
        };
        truth.apply_inputs(tick, &[input]);
        truth.step();
    }
    let truth_state = truth.ship_state(guid(1)).unwrap();
    // Both sides run the *same* step function on the *same* held input from the
    // *same* authority, so they must agree bit for bit — not just "within a
    // budget". Any drift here is a real prediction bug.
    assert_eq!(
        predicted, truth_state,
        "client prediction {K} ticks past the last snapshot must match a server \
         stepped the same {K} ticks with the same held input, bit for bit"
    );
    // The prediction really moved (it is not the trivial `steps == 0` return).
    assert_ne!(
        predicted, client.authoritative,
        "predict(final_tick + K) must actually replay steps, not return the authority"
    );
}

// ---------------------------------------------------------------------------
// orbits
// ---------------------------------------------------------------------------

/// `rail_pos` is a pure function of `(system, body, tick)`: two independently
/// constructed `StarSystem`s with the same seed return bit-identical positions,
/// and the orbit closes over one period.
#[test]
fn orbits_deterministic() {
    let seed = 0xC0FF_EE12_3456_7890;
    let star = WorldPos::from_metres([9_000_000_000_000, -4_000_000_000_000, 250_000]);
    let a = StarSystem::new(seed, star);
    let b = StarSystem::new(seed, star);
    assert_eq!(a, b, "same seed and star must generate the same system");
    assert_ne!(
        StarSystem::new(seed ^ 1, star),
        a,
        "a different seed must generate a different system"
    );

    let bodies = orbits::body_count(&a);
    assert!(bodies > 2, "a system needs a star, planets and a belt");
    // Star, planets and a spread of belt bodies.
    let probes = [
        orbits::STAR_BODY,
        1,
        2,
        a.planet_count() as u32,
        1 + a.planet_count() as u32,
        bodies / 2,
        bodies - 1,
    ];
    let ticks = [0_u64, 1, 7, 1_000, 900_000, 30 * 60 * 60 * 24 * 400];
    for body in probes {
        for tick in ticks {
            let pa = orbits::rail_pos(&a, body, tick);
            let pb = orbits::rail_pos(&b, body, tick);
            assert_eq!(pa, pb, "body {body} tick {tick} must be bit-identical");
            assert!(pa.local.iter().all(|v| v.is_finite()));
        }
    }

    // Periodicity: in seconds the orbit closes *exactly*; in ticks it closes to
    // within the tick quantisation of the period.
    for body in 1..=a.planet_count() as u32 {
        let period = orbits::period(&a, body);
        assert!(period > 0.0 && period.is_finite());
        assert_eq!(
            orbits::rail_pos_seconds(&a, body, 0.0),
            orbits::rail_pos_seconds(&a, body, period),
            "planet {body} must close its orbit exactly in seconds"
        );
        let period_ticks = orbits::period_ticks(&a, body);
        let whole = period_ticks.round() as u64;
        let drift = WorldPos::distance(
            orbits::rail_pos(&a, body, 0),
            orbits::rail_pos(&a, body, whole),
        );
        let radius = a.planets[usize::try_from(body - 1).unwrap()].semi_major_axis;
        assert!(
            drift < radius * 1e-4,
            "planet {body} drifted {drift} m over a {radius} m orbit"
        );
        // seconds_of(tick) is the time base rail_pos uses.
        assert_eq!(
            orbits::rail_pos(&a, body, whole),
            orbits::rail_pos_seconds(&a, body, seconds_of(whole))
        );
    }

    // A belt body closes too (its period is derived, not published).
    let first_belt = 1 + a.planet_count() as u32;
    let belt_period = orbits::period(&a, first_belt);
    assert!(belt_period > 0.0 && belt_period.is_finite());
    let drift = WorldPos::distance(
        orbits::rail_pos_seconds(&a, first_belt, 0.0),
        orbits::rail_pos_seconds(&a, first_belt, belt_period),
    );
    assert!(
        drift < WorldPos::distance(star, orbits::rail_pos(&a, first_belt, 0)) * 1e-6,
        "belt body drifted {drift} m"
    );
}

// ---------------------------------------------------------------------------
// registry
// ---------------------------------------------------------------------------
/// The ship and physics components are registered in the `space-ecs`
/// replication registry with the ids the protocol will carry (ids are
/// append-only: 0-3 are the ship's, 4-10 arrived with the physics core).
#[test]
fn components_are_registered() {
    use space_ecs::{ComponentInfo, MAX_COMPONENTS};
    use space_sim::components::{
        Owner, ShardAuthority, ShipControls, ShipState, COMPONENT_REGISTRY,
    };

    assert_eq!(<ShipState as ComponentDef>::ID, 0);
    assert_eq!(<ShipControls as ComponentDef>::ID, 1);
    assert_eq!(<Owner as ComponentDef>::ID, 2);
    assert_eq!(<ShardAuthority as ComponentDef>::ID, 3);

    let expected: [(u16, &str, bool); 11] = [
        (0, "ShipState", true),
        (1, "ShipControls", true),
        (2, "Owner", true),
        (3, "ShardAuthority", false),
        (4, "Body", true),
        (5, "Visual", true),
        (6, "Springs", true),
        (7, "Articulation", true),
        (8, "PartStates", true),
        (9, "Ghost", false),
        (10, "PosResidual", true),
    ];
    assert_eq!(COMPONENT_REGISTRY.len(), MAX_COMPONENTS);
    for (id, name, replicated) in expected {
        let info: ComponentInfo = COMPONENT_REGISTRY[usize::from(id)]
            .unwrap_or_else(|| panic!("component id {id} is not registered"));
        assert_eq!(info.id, id);
        assert_eq!(info.name, name);
        assert_eq!(info.replicated, replicated, "{name} replication flag");
        // `Ghost` is a marker: its presence is the information.
        assert!(info.size > 0 || name == "Ghost", "{name} has no payload");
    }
    // Nothing beyond id 10 is registered by this crate.
    assert!(COMPONENT_REGISTRY[11].is_none());
}

// ---------------------------------------------------------------------------
// change-tick replication (FIX 1c / 1d)
// ---------------------------------------------------------------------------

/// FIX 1c: `SimWorld::step` only touches ships that have work (non-zero controls
/// **or** non-zero velocity). An idle ship — at rest with no input — is never
/// written, so its change tick stays put and a `changed_since` replication pass
/// does not see it. An idle fleet costs nothing on the wire.
#[test]
fn idle_ships_are_not_replicated() {
    use space_ecs::WithEntity;

    let mut sim = SimWorld::new(ShardId(7));
    // Two idle ships (at rest, no controls) and one that will be driven, ten
    // metres apart: ships are solid against each other, so three ships at one
    // point would push each other apart and none would be idle.
    let apart = |x: f32| ShipState::at_rest(space_world::WorldPos::ZERO.offset([x, 0.0, 0.0]));
    let idle_a = sim.spawn_ship(guid(1), 1, apart(0.0));
    let idle_b = sim.spawn_ship(guid(2), 2, apart(10.0));
    let active = sim.spawn_ship(guid(3), 3, apart(20.0));

    // Settle one no-input tick: spawning stamped every row with the open tick, so
    // we let that stamp fall into the past before the driven step.
    sim.step();

    // Drive only ship 3 on the next tick.
    sim.apply_inputs(
        2,
        &[PlayerInput {
            entity: guid(3),
            thrust: [1.0, 0.0, 0.0],
            torque: [0.0; 3],
            tick: 2,
            fast: false,
        }],
    );
    sim.step();

    // A changed_since(1) pass (the settle tick) sees exactly the active ship: the
    // idle ships were never written by the driven step.
    let mut changed: Vec<space_ecs::Entity> = Vec::new();
    {
        let mut q = sim
            .world_mut()
            .query::<WithEntity<&ShipState>>()
            .changed_since(1);
        q.for_each(|(_state, entity)| changed.push(entity));
    }
    assert!(changed.contains(&active), "the driven ship must be changed");
    assert!(
        !changed.contains(&idle_a) && !changed.contains(&idle_b),
        "idle ships must not be marked changed: {changed:?}"
    );
}

/// A coasting ship (non-zero velocity, no controls) still moves and is therefore
/// replicated — the read-only filter keeps it in the active set.
#[test]
fn coasting_ship_is_replicated() {
    use space_ecs::WithEntity;

    let mut sim = SimWorld::new(ShardId(7));
    let mut moving = ShipState::IDENTITY;
    moving.vel = [10.0, 0.0, 0.0];
    let coasting = sim.spawn_ship(guid(1), 1, moving);
    let idle = sim.spawn_ship(guid(2), 2, ShipState::IDENTITY);

    // Settle one tick so the spawn stamp is in the past.
    sim.step();
    // The coasting ship moves on the second tick; the idle one does not.
    sim.step();

    let mut changed: Vec<space_ecs::Entity> = Vec::new();
    {
        let mut q = sim
            .world_mut()
            .query::<WithEntity<&ShipState>>()
            .changed_since(1);
        q.for_each(|(_state, entity)| changed.push(entity));
    }
    assert!(changed.contains(&coasting), "a coasting ship must move");
    assert!(!changed.contains(&idle), "an idle ship must not move");
}

/// FIX 1d: the ECS change tick equals the last produced [`SimTick`] plus one —
/// the "open" tick the next step will stamp its writes with.
#[test]
fn ecs_tick_tracks_sim_tick() {
    let mut sim = SimWorld::new(ShardId(7));
    // At tick 0 the open ECS tick is 1.
    assert_eq!(sim.world().current_tick(), 1);
    sim.step();
    assert_eq!(sim.tick(), 1);
    assert_eq!(sim.world().current_tick(), 2);
    sim.step_n(5);
    assert_eq!(sim.tick(), 6);
    assert_eq!(sim.world().current_tick(), 7);

    // A world that joins mid-session starts its ECS tick at start_tick + 1.
    let joined = SimWorld::with_tick(ShardId(7), space_sim::ShipParams::PLACEHOLDER, 500);
    assert_eq!(joined.tick(), 500);
    assert_eq!(joined.world().current_tick(), 501);
}

/// FIX 1d / N8: a write that lands **between** two steps is stamped with the
/// *next* step's tick (the open tick), so a replication pass that filters with
/// `changed_since(last_sent_tick)` still sees it — it can never be missed.
///
/// The ship is idle (at rest, no input) so `step()` itself never stamps it; the
/// only write is the manual `world_mut().get_mut` between the two steps. Before
/// the tick-ordering fix (ECS tick advanced at the *start* of a step) that write
/// would have been stamped with a tick a delta had already been sent for, and the
/// row would be lost forever.
#[test]
fn write_between_steps_is_replicated_by_the_next_step() {
    use space_ecs::WithEntity;

    let mut sim = SimWorld::new(ShardId(7));
    let ship = sim.spawn_ship(guid(1), 1, ShipState::IDENTITY);

    // Step to tick 1 and "send" a delta for it: last_sent_tick = 1.
    sim.step();
    let last_sent_tick = sim.tick();
    assert_eq!(last_sent_tick, 1);
    // The idle ship was not written by step() (spawn stamp is in the past).
    assert!(sim.world().row_tick(ship).unwrap() <= last_sent_tick);

    // A write that lands between step 1 and step 2 (e.g. a handoff applying an
    // incoming state, or a system this crate does not provide).
    {
        let state = sim.world_mut().get_mut::<ShipState>(ship).unwrap();
        state.vel = [0.0, 0.0, 3.0];
    }

    // The next step produces tick 2; the write was stamped with the open tick (2).
    sim.step();
    assert_eq!(sim.tick(), 2);

    // A replication pass that already sent tick 1 must still see the row: it was
    // stamped with tick 2, a tick no delta has been sent for yet.
    let mut changed: Vec<space_ecs::Entity> = Vec::new();
    {
        let mut q = sim
            .world_mut()
            .query::<WithEntity<&ShipState>>()
            .changed_since(last_sent_tick);
        q.for_each(|(_state, entity)| changed.push(entity));
    }
    assert!(
        changed.contains(&ship),
        "a write between two steps must be seen by changed_since(last_sent_tick): {changed:?}"
    );
}

// ---------------------------------------------------------------------------
// lossless ShipState wire codec (FIX 2)
// ---------------------------------------------------------------------------

/// The `ShipState` codec is lossless: every field round-trips bit-for-bit,
/// including `-0.0` and subnormals, and the encoded size is the documented 76 B.
#[test]
fn ship_state_codec_is_lossless() {
    use space_sim::wire::{decode_ship_state, encode_ship_state, SHIP_STATE_LEN};

    let cases = [
        ShipState::IDENTITY,
        ShipState::at_rest(space_world::WorldPos::from_metres([
            12_345_678_901,
            -987_654_321,
            4_000,
        ])),
        ShipState {
            pos: space_world::WorldPos::new([7, -3, 11], [1.5, -2.25, 0.0]),
            vel: [100.5, -0.0, 1e-30],
            quat: [0.923_879_5, 0.0, 0.382_683_4, -0.0],
            ang_vel: [-0.0, 0.5, -1e-38],
        },
    ];
    for state in cases {
        let bytes = encode_ship_state(&state);
        assert_eq!(bytes.len(), SHIP_STATE_LEN, "encoded size drifted");
        let back = decode_ship_state(&bytes).expect("decodes");
        // Bit-exact equality on every float field.
        assert_eq!(back.pos.cell, state.pos.cell);
        for axis in 0..3 {
            assert_eq!(
                back.pos.local[axis].to_bits(),
                state.pos.local[axis].to_bits(),
                "pos.local[{axis}]"
            );
            assert_eq!(
                back.vel[axis].to_bits(),
                state.vel[axis].to_bits(),
                "vel[{axis}]"
            );
            assert_eq!(
                back.ang_vel[axis].to_bits(),
                state.ang_vel[axis].to_bits(),
                "ang_vel[{axis}]"
            );
        }
        for axis in 0..4 {
            assert_eq!(
                back.quat[axis].to_bits(),
                state.quat[axis].to_bits(),
                "quat[{axis}]"
            );
        }
        assert_eq!(back, state);
    }
}

/// The `WorldPos` codec rejects a non-finite `local` and a `local` outside the
/// canonical `[-CELL_HALF, CELL_HALF)` range, and round-trips a valid position
/// losslessly.
#[test]
fn world_pos_codec_validates() {
    use space_protocol::pos::{decode_world_pos, encode_world_pos, WORLD_POS_LEN};
    use space_world::{WorldPos, CELL_HALF};

    // A valid position round-trips exactly.
    let p = WorldPos::new([123, -456, 789], [100.0, -200.5, 0.0]);
    let bytes = encode_world_pos(&p);
    assert_eq!(bytes.len(), WORLD_POS_LEN);
    let back = decode_world_pos(&bytes).expect("valid decodes");
    assert_eq!(back.cell, p.cell);
    for axis in 0..3 {
        assert_eq!(back.local[axis].to_bits(), p.local[axis].to_bits());
    }

    // A non-finite local is rejected: forge the first local axis (offset 24) to
    // +inf.
    let mut inf = encode_world_pos(&p);
    inf[24..28].copy_from_slice(&f32::INFINITY.to_bits().to_be_bytes());
    assert!(
        decode_world_pos(&inf).is_err(),
        "non-finite local must be rejected"
    );

    // A local at or beyond +CELL_HALF is rejected (canonical range is half-open).
    let mut oob = encode_world_pos(&p);
    oob[24..28].copy_from_slice(&CELL_HALF.to_bits().to_be_bytes());
    assert!(
        decode_world_pos(&oob).is_err(),
        "local == CELL_HALF must be rejected"
    );

    // -CELL_HALF is inside the half-open range and must be accepted.
    let mut edge = encode_world_pos(&p);
    edge[24..28].copy_from_slice(&(-CELL_HALF).to_bits().to_be_bytes());
    assert!(
        decode_world_pos(&edge).is_ok(),
        "local == -CELL_HALF is valid"
    );

    // A wrong-length buffer is rejected.
    assert!(decode_world_pos(&bytes[..bytes.len() - 1]).is_err());
}

/// B3: the `ShipState` wire codec round-trips positions at the extremes of the
/// `i64` cell range and at the boundaries of the canonical `local` range, and
/// rejects a forged non-finite `local` on every axis.
///
/// Byte layout of the 76-byte payload (big endian, fixed width):
/// `pos.cell 3×i64 (0..24) | pos.local 3×f32 (24..36) | vel 3×f32 (36..48) |
/// quat 4×f32 (48..64) | ang_vel 3×f32 (64..76)`.
#[test]
fn ship_state_codec_edge_positions() {
    use space_sim::wire::{decode_ship_state, encode_ship_state, SHIP_STATE_LEN};
    use space_world::{WorldPos, CELL_HALF};

    // ~5e20 m: a cell index near the top of the i64 range. `from_metres` takes
    // millimetres, so 5e20 m is 5e23 mm — well inside i64 (max ~9.2e18 m).
    let huge_mm: i128 = 5e23 as i128;
    let huge = ShipState::at_rest(WorldPos::from_metres([huge_mm, -huge_mm, huge_mm]));
    let bytes = encode_ship_state(&huge);
    assert_eq!(bytes.len(), SHIP_STATE_LEN);
    let back = decode_ship_state(&bytes).expect("extreme cell index round-trips");
    assert_eq!(back.pos.cell, huge.pos.cell);
    assert_eq!(back, huge);

    // `local == -CELL_HALF` is inside the half-open canonical range and must
    // round-trip.
    let neg_edge = ShipState::at_rest(WorldPos::new([3, -4, 5], [-CELL_HALF, 0.0, 0.0]));
    let back = decode_ship_state(&encode_ship_state(&neg_edge)).expect("-CELL_HALF is valid");
    assert_eq!(back.pos.local[0].to_bits(), (-CELL_HALF).to_bits());

    // The largest `f32` strictly below `+CELL_HALF` is still canonical and must
    // round-trip.
    let just_under = next_down_f32(CELL_HALF);
    assert!(just_under < CELL_HALF);
    let pos_edge = ShipState::at_rest(WorldPos::new([0, 0, 0], [0.0, just_under, 0.0]));
    let back = decode_ship_state(&encode_ship_state(&pos_edge))
        .expect("largest f32 below CELL_HALF is valid");
    assert_eq!(back.pos.local[1].to_bits(), just_under.to_bits());

    // A forged non-finite `local` is rejected on **every** axis (byte offsets
    // 24, 28, 32), for both NaN and -inf.
    let base = ShipState::at_rest(WorldPos::new([1, 2, 3], [1.0, 2.0, 3.0]));
    let poison = [f32::NAN, f32::NEG_INFINITY];
    for axis in 0..3 {
        let off = 24 + axis * 4;
        for &bad in &poison {
            let mut forged = encode_ship_state(&base);
            forged[off..off + 4].copy_from_slice(&bad.to_bits().to_be_bytes());
            assert!(
                decode_ship_state(&forged).is_err(),
                "forged local[{axis}] = {bad} must be rejected"
            );
        }
    }
}

/// B4: `decode_ship_state` rejects a non-finite `vel`, `quat` or `ang_vel`, not
/// just a non-finite `pos.local`. A decoded `ShipState` is untrusted input (a
/// handoff payload from another shard, a client delta), and a NaN would poison
/// the integrator permanently.
#[test]
fn ship_state_codec_rejects_non_finite_dynamics() {
    use space_sim::wire::{decode_ship_state, encode_ship_state};

    let base = ShipState::at_rest(space_world::WorldPos::ZERO);
    let poison = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];

    // Field byte ranges: vel 36..48, quat 48..64, ang_vel 64..76.
    let fields: [(&str, usize, usize); 3] =
        [("vel", 36, 48), ("quat", 48, 64), ("ang_vel", 64, 76)];
    for (name, start, end) in fields {
        let n = (end - start) / 4;
        for axis in 0..n {
            let off = start + axis * 4;
            for &bad in &poison {
                let mut forged = encode_ship_state(&base);
                forged[off..off + 4].copy_from_slice(&bad.to_bits().to_be_bytes());
                assert!(
                    decode_ship_state(&forged).is_err(),
                    "forged {name}[{axis}] = {bad} must be rejected"
                );
            }
        }
    }

    // A finite, plausible value in those fields is still accepted.
    let mut ok = encode_ship_state(&base);
    ok[36..40].copy_from_slice(&2.5e4f32.to_bits().to_be_bytes());
    assert!(decode_ship_state(&ok).is_ok(), "a finite vel is accepted");
}

/// Finite is not enough: `decode_ship_state` also rejects a state the integrator
/// could not have produced and could be driven to `inf`/NaN by — a quaternion
/// far from unit, an absurd linear or angular velocity — while every state the
/// simulation *does* produce still round-trips bit for bit.
///
/// The first block shows the poisoning the check exists to stop: the forged
/// state is entirely finite, and one tick of thrust later the ship is not.
#[test]
fn ship_state_codec_rejects_states_that_would_poison_the_integrator() {
    use space_sim::wire::{
        decode_ship_state, encode_ship_state, MAX_SPEED, MAX_SPIN, QUAT_NORM2_MAX, QUAT_NORM2_MIN,
    };

    let base = ShipState::at_rest(WorldPos::from_metres([5_000, -7_000, 9_000]));
    let params = ShipParams::PLACEHOLDER;
    let thrust = PlayerInput {
        entity: guid(1),
        thrust: [0.0, 0.0, 1.0],
        torque: [0.0; 3],
        tick: 1,
        fast: false,
    };
    let coast = PlayerInput {
        thrust: [0.0; 3],
        ..thrust
    };
    let is_finite = |s: &ShipState| {
        s.pos.local.iter().all(|v| v.is_finite())
            && s.vel.iter().all(|v| v.is_finite())
            && s.quat.iter().all(|v| v.is_finite())
            && s.ang_vel.iter().all(|v| v.is_finite())
    };

    // --- why: a finite but huge quaternion poisons the state in one tick -----
    let mut huge_quat = base;
    huge_quat.quat = [0.0, 2.0e19, 0.0, 0.0];
    assert!(is_finite(&huge_quat), "the forged state itself is finite");
    let mut poisoned = huge_quat;
    step_ship(&mut poisoned, &thrust, &params);
    assert!(
        !is_finite(&poisoned),
        "one tick of thrust must overflow: {poisoned:?}"
    );
    assert!(
        decode_ship_state(&encode_ship_state(&huge_quat)).is_err(),
        "a quaternion of norm 2e19 must be rejected"
    );

    // --- and a finite but absurd spin does it with no input at all ------------
    // The angular damping term overflows: -inf after one tick, NaN after two.
    let mut huge_spin = base;
    huge_spin.ang_vel = [3.0e38, 0.0, 0.0];
    let mut poisoned = huge_spin;
    step_ship(&mut poisoned, &coast, &params);
    step_ship(&mut poisoned, &coast, &params);
    assert!(
        poisoned.ang_vel[0].is_nan(),
        "two coasting ticks must turn the spin into NaN: {poisoned:?}"
    );
    assert!(decode_ship_state(&encode_ship_state(&huge_spin)).is_err());

    // --- the quaternion bound: |q|² within QUAT_NORM2_MIN..=QUAT_NORM2_MAX ----
    let with_quat = |quat: [f32; 4]| {
        let mut s = base;
        s.quat = quat;
        decode_ship_state(&encode_ship_state(&s))
    };
    for bad in [
        [0.0; 4],                // degenerate
        [0.5, 0.0, 0.0, 0.0],    // |q|² = 0.25
        [0.0, 0.0, 1.5, 0.0],    // |q|² = 2.25
        [1.0, 1.0, 1.0, 1.0],    // |q|² = 4
        [3.0e38, 0.0, 0.0, 0.0], // |q|² overflows to inf
        [0.0, 0.0, 0.0, -1.0e30],
    ] {
        assert!(with_quat(bad).is_err(), "quat {bad:?} must be rejected");
    }
    for good in [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.923_879_5, 0.0, 0.382_683_4, -0.0],
        [0.5, 0.5, 0.5, 0.5],
        // Hand-written and not quite unit: accepted, the integrator normalises.
        [0.7, 0.7, 0.0, 0.0],
        [1.1, 0.0, 0.0, 0.0],
    ] {
        let back = with_quat(good).unwrap_or_else(|e| panic!("quat {good:?}: {e}"));
        for axis in 0..4 {
            assert_eq!(back.quat[axis].to_bits(), good[axis].to_bits());
        }
    }
    // Just inside and just outside each end of the accepted |q|² range.
    for (norm2, accepted) in [
        (QUAT_NORM2_MIN * 0.99, false),
        (QUAT_NORM2_MIN * 1.01, true),
        (QUAT_NORM2_MAX * 0.99, true),
        (QUAT_NORM2_MAX * 1.01, false),
    ] {
        let w = norm2.sqrt();
        assert_eq!(
            with_quat([w, 0.0, 0.0, 0.0]).is_ok(),
            accepted,
            "|q|² = {norm2}"
        );
    }

    // --- the velocity bounds, on every axis, both signs -----------------------
    for axis in 0..3 {
        for sign in [1.0f32, -1.0] {
            let mut s = base;
            s.vel[axis] = sign * MAX_SPEED;
            assert!(
                decode_ship_state(&encode_ship_state(&s)).is_ok(),
                "vel[{axis}] at the limit is accepted"
            );
            s.vel[axis] = sign * MAX_SPEED * 1.001;
            assert!(
                decode_ship_state(&encode_ship_state(&s)).is_err(),
                "vel[{axis}] beyond the limit is rejected"
            );
            s.vel[axis] = sign * 1.0e30;
            assert!(decode_ship_state(&encode_ship_state(&s)).is_err());

            let mut s = base;
            s.ang_vel[axis] = sign * MAX_SPIN;
            assert!(
                decode_ship_state(&encode_ship_state(&s)).is_ok(),
                "ang_vel[{axis}] at the limit is accepted"
            );
            s.ang_vel[axis] = sign * MAX_SPIN * 1.001;
            assert!(
                decode_ship_state(&encode_ship_state(&s)).is_err(),
                "ang_vel[{axis}] beyond the limit is rejected"
            );
        }
    }

    // --- no state the simulation produces is ever rejected --------------------
    // A hard-driven ship for 2 000 ticks, then a long coast down to rest: every
    // state on the way must decode to exactly itself, so a valid state is the
    // same bits on both sides of the wire.
    let mut rng = Rng::new(SEED ^ 0x5AFE);
    let mut state = base;
    for tick in 1..=4_000u64 {
        let input = if tick <= 2_000 {
            PlayerInput {
                entity: guid(1),
                thrust: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
                torque: [rng.next_signed(), rng.next_signed(), rng.next_signed()],
                tick,
                fast: false,
            }
        } else {
            coast
        };
        step_ship(&mut state, &input, &params);
        let bytes = encode_ship_state(&state);
        let back = decode_ship_state(&bytes)
            .unwrap_or_else(|e| panic!("tick {tick}: a simulated state was rejected: {e}"));
        assert_eq!(
            encode_ship_state(&back),
            bytes,
            "tick {tick}: a valid state must round-trip bit for bit"
        );
    }
}

/// The largest `f32` strictly below `x` (toward negative infinity).
fn next_down_f32(x: f32) -> f32 {
    let bits = x.to_bits();
    // For a positive finite x, the next value down is the previous bit pattern.
    f32::from_bits(bits - 1)
}
