//! Ship tuning parameters.

/// One ship's flight model constants.
///
/// Everything the physics step needs beyond the state and the input, so two
/// ships can differ (a frigate does not turn like a fighter) and so a designer
/// can retune the feel without touching the integrator. All values are SI:
/// kilograms, newtons, newton-metres and their per-second drag coefficients.
///
/// The crate ships [`ShipParams::PLACEHOLDER`] because Phase 1 has no designer
/// pass yet; the values are chosen for a *nimble arcade-ish* fighter, i.e. the
/// ship a player notices as responsive rather than as a barge:
///
/// | field       | value | meaning and why this value |
/// |-------------|-------|----------------------------|
/// | `mass`      | `1.0e4` kg | 10 t. Light enough that the placeholder thrust reaches useful speed in a second or two, heavy enough that the numbers stay in a comfortable `f32` range. |
/// | `thrust`    | `6.0e4` N | 6 g of acceleration at full burn (6e4 / 1e4 = 6 m/s²). With `drag_lin` below this caps straight-line speed at `thrust / (mass · drag_lin)` ≈ 120 m/s, a speed a player can aim at a planet and reach. |
/// | `torque`    | `4.0e4` N·m | Paired with `drag_ang` this caps the pitch/yaw rate at `torque / drag_ang` ≈ 3.3 rad/s ≈ 190°/s: a fast, arcade turn. Roll is deliberately *not* modelled separately in Phase 1 (see [`crate::physics::step_ship`]). |
/// | `drag_lin`  | `0.05` /s | Linear drag coefficient, `a -= drag_lin · v`. Space has no air, but a game with pure Newtonian coasting is unplayable: this is a *damping* term standing in for "the ship's flight computer bleeds sideways speed". Small enough that a cut engine coasts for tens of seconds (time constant `1/0.05` = 20 s). |
/// | `drag_ang`  | `1.2e4` N·m·s | Angular damping, `α -= (drag_ang / mass) · ω`. With `mass` = 1e4 the angular time constant is `mass / drag_ang` ≈ 0.83 s, so a release-the-stick stop feels immediate. |
/// | `lift`      | `1.0` | The lift engines: in gravity, the part of the input that pushes *against* the local pull gets `lift` times the pull's strength of extra acceleration ([`crate::physics::lift_accel`]). At 1.0 pushing straight up in the demo area's gravity room (9.81 m/s²) climbs at the same 6 m/s² as in weightless space, while a ship that pushes nothing falls and lands like any body. In weightless space (`g = 0`) it adds nothing, so free flight is unchanged to the bit. |
///
/// The drag terms are applied as a *linear* damping (`-k·v`), which has the
/// closed form `v(t) = v₀ · e^{-k t}` the acceptance test
/// `step_ship_physics` checks against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShipParams {
    /// Mass, in kilograms.
    pub mass: f32,
    /// Full main-engine thrust, in newtons (input magnitude 1.0 along an axis).
    pub thrust: f32,
    /// Full control torque, in newton-metres.
    pub torque: f32,
    /// Linear damping coefficient, per second.
    pub drag_lin: f32,
    /// Angular damping coefficient, newton-metre-seconds.
    pub drag_ang: f32,
    /// Lift engines, as a multiple of the local gravity: the extra
    /// acceleration against the pull when the input pushes against it (see
    /// the table above and [`crate::physics::lift_accel`]). 0 = none.
    pub lift: f32,
}

impl ShipParams {
    /// The Phase-1 placeholder flight model (see the table on
    /// [`ShipParams`]).
    pub const PLACEHOLDER: Self = Self {
        mass: 1.0e4,
        thrust: 6.0e4,
        torque: 4.0e4,
        drag_lin: 0.05,
        drag_ang: 1.2e4,
        lift: 1.0,
    };
}

impl Default for ShipParams {
    fn default() -> Self {
        Self::PLACEHOLDER
    }
}

#[cfg(test)]
mod tests {
    use super::ShipParams;

    #[test]
    fn placeholder_values_are_positive_and_finite() {
        let p = ShipParams::PLACEHOLDER;
        assert_eq!(p, ShipParams::default());
        for v in [p.mass, p.thrust, p.torque, p.drag_lin, p.drag_ang, p.lift] {
            assert!(v > 0.0 && v.is_finite(), "bad placeholder value {v}");
        }
        // Documented derived numbers: linear terminal speed and turn rate.
        let v_term = p.thrust / (p.mass * p.drag_lin);
        assert!((v_term - 120.0).abs() < 1e-3, "terminal speed {v_term}");
        let w_term = p.torque / p.drag_ang;
        assert!((w_term - 3.333_333_3).abs() < 1e-3, "turn rate {w_term}");
    }
}
