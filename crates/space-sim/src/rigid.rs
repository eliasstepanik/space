//! The rigid-body world: floating-origin **physics bubbles** around
//! [rapier](https://rapier.rs) (`rapier3d-f64`).
//!
//! # Large worlds: why bubbles
//!
//! A [`WorldPos`] is an `i64` cell (1024 m) plus an `f32` offset, so the world
//! spans ~1e22 m and two ships of one shard can be an astronomical unit apart.
//! No single floating-point frame covers that: `f64` around one origin is
//! good to ~1 mm only out to ~1e13 m. So the physics never uses one frame.
//! It simulates in **bubbles** — local frames, each anchored at a
//! cell-aligned [`WorldPos`] — and every body belongs to exactly one bubble.
//!
//! * **Local coordinates are derived from the world position.** A body's
//!   exact position is its [`WorldPos`] plus its [`crate::PosResidual`] (the
//!   part finer than the `WorldPos`'s `f32` offset holds). Its local pose is
//!   `WorldPos::delta(anchor, pos) + residual` — an exact `i64` cell
//!   difference (cell-aligned anchor) plus the offsets, in `f64` — and after
//!   the step the result is written back as `anchor + local`, rounded once
//!   into the new [`WorldPos`], the remainder into the residual. Both are
//!   components: on the wire, in a handoff and in the state hash the world
//!   position is the truth. (While the solver keeps moving a body in contact,
//!   rapier's own pose of it *is* that exact position, so it is not re-derived
//!   every tick — re-deriving it would wake the body's island and stop rapier
//!   putting resting stacks to sleep; a write from outside, or free flight,
//!   re-derives it.)
//! * **Precision does not depend on where.** Inside a bubble, coordinates
//!   stay within a few kilometres of the anchor, where an `f64` resolves
//!   better than 1e-12 m (table below), and a body's exact position
//!   (`WorldPos` + residual) does not lose that when it is written back: the
//!   tests move the same scene from the origin to 1e20 m (bit-identical
//!   motion), fly 1e7 m through ~5000 re-anchors (exact) and keep two bodies
//!   one astronomical unit apart at micrometres. The one exception is a
//!   **ship**, which has no residual so that a client can predict it from its
//!   replicated `ShipState` alone: its position rounds into the `WorldPos`
//!   every tick — at most 15 µm per tick at a cell's edge, ~0.06 µm within a
//!   metre of a cell's centre — the same anywhere in the world, as before the
//!   physics core.
//! * **Free flight never touches a bubble frame.** A body that no contact and
//!   no joint touched this tick (the solver left its velocity unchanged) is
//!   integrated directly on its exact world position: the exact `f64`
//!   displacement `v · dt` added in its own cell
//!   ([`crate::physics::position_step`] for a ship). A ship in free flight is
//!   therefore stepped exactly as [`crate::step_ship`] steps it, and a body
//!   that changes bubble, or whose bubble re-anchors or merges, follows the
//!   same trajectory bit for bit. Only bodies in contact (or jointed) take
//!   their pose from the solver, in bubble-local coordinates; there a merge or
//!   re-anchor shifts the pose by the exact anchor difference (a whole number
//!   of cells), changing nothing but `f64` rounding (~1e-13 m).
//!
//! | distance from the anchor | `f64` resolution |
//! |---|---|
//! | 1 km | 1.1e-13 m |
//! | 4 km (re-anchor threshold + bubble extent) | 9.1e-13 m |
//! | 1e6 m | 1.2e-10 m |
//! | 1e10 m | 1.9e-6 m |
//! | 1e11 m | 1.5e-5 m |
//! | 1e13 m | 2.0e-3 m |
//!
//! (The last three rows are why there is no single origin. A static whose
//! *centre* is far away — a planet 6e6 m in radius whose surface is next to a
//! bubble — is placed at its exact local offset; its surface is resolved to
//! the row of its centre's distance, 1e-9 m for a planet.)
//!
//! # The bubble rules (deterministic functions of the state)
//!
//! Every tick, before the solver runs, in canonical order (entities by
//! [`EntityGuid`], bubbles by [`BubbleId`]; no hash-map iteration):
//!
//! 1. **Bounds.** Each body gets a world box in integer micrometres (`i128`,
//!    exact anywhere in the `i64` cell space): the bounding sphere of every
//!    part, grown by the **merge margin** [`MERGE_MARGIN`] `+ |v| ·`
//!    [`MERGE_LEAD`] and, separately, by the **split margin**
//!    [`SPLIT_FACTOR`] times that.
//! 2. **Split.** The members of a bubble whose split boxes fall into more than
//!    one connected group are separated: the group with the **most members**
//!    keeps it (ties: the one holding the smallest guid), every other group
//!    gets a new bubble (in order of its smallest guid). The group that keeps
//!    the bubble keeps the solver's world as it is — its contacts, what
//!    sleeps — and every other group is put into a fresh one, awake, where it
//!    finds its contacts again in the first step and re-settles within
//!    millimetres. (The bubble used to stay with its smallest guid. A
//!    player's ship has a lower guid than any body of a scene, so whenever a
//!    player left a group of resting bodies it was *they* who were moved —
//!    and, inserted asleep, they then fell centimetres into the floor before
//!    the solver had their contacts: the demo area's crate stack hopped
//!    every time somebody walked away from it.)
//! 3. **Merge.** Bodies whose merge boxes overlap (transitively) must share a
//!    bubble: their bubbles merge into the one with the most members (ties:
//!    the lowest id), new bodies join it, and a group with no bubble gets a
//!    new one. Because the split margin is twice the merge margin, a pair at a
//!    distance between the two neither merges nor splits: bubbles do not
//!    flicker.
//! 4. **Re-anchor.** A bubble whose members' centre has moved
//!    [`REANCHOR_CELLS`] cells or more from its anchor (on any axis) is
//!    re-anchored at the cell containing that centre; a new bubble is anchored
//!    there from the start. Re-anchoring re-derives every local pose in the
//!    bubble (bodies, statics, joint anchors) from world coordinates — the
//!    canonical rule: local coordinates are always a function of
//!    `(anchor, WorldPos)`, never of a previous local value.
//! 5. **Statics.** A static collider of the scene is instantiated into each
//!    bubble one of whose members it comes within the merge margin of, at its
//!    exact local pose, and removed again when it is beyond the split margin
//!    of all of them. A static far from every bubble costs nothing; a big
//!    triangle mesh is one shared `Arc` (its BVH is built once, in the
//!    [`crate::shape::ShapeCatalog`]).
//!
//! Merge-before-touch: two bodies of different bubbles are at least the merge
//! margin plus four ticks of their own motion apart, so they cannot meet
//! within the tick (a body the solver flings faster than that within one tick
//! is caught by the next tick's merge, and the solver then separates the
//! overlap like any penetration).
//!
//! A bubble is only handed to the solver when it can have a contact: an awake
//! dynamic body and a static, a second body or a joint. A lone ship in space
//! is a bubble of one that is never stepped by rapier at all.
//!
//! # One tick
//!
//! 0. **Scene time.** The tick runs from scene time `t0` (the time the last
//!    step ended at, or the clock's time of the tick after a
//!    [`crate::SimWorld::reset_clock`]) to `t1`, the
//!    [`crate::SimWorld::set_scene_clock`] mapping's time of the tick being
//!    produced. Moving statics (a planet's collider on its orbit,
//!    [`StaticCollider::motion`]) are placed where they are at `t1`, and every
//!    body (asleep or not) is first **carried** by the orbiting gravity
//!    bodies it is near ([`crate::gravity::GravityField::carry`] from `t0` to
//!    `t1`): its world position moves by that displacement, its velocity does
//!    not change, and a body rapier holds the precise pose of is moved in its
//!    bubble by the same vector, so a ship standing on the Moon and the Moon
//!    move together. The rest of the tick sees the world at `t1`.
//! 1. Read every body's state; a state written behind the physics' back
//!    (a handoff install, a test, an admin teleport) wakes the body.
//! 2. Springs ([`crate::spring`]), evaluated from the start-of-tick states;
//!    gravity ([`crate::gravity`]) at each body's [`WorldPos`], with the
//!    orbiting bodies where they are at `t1`.
//! 3. Velocities: ships by the flight model
//!    ([`crate::physics::ship_velocity_step`]) plus gravity, springs and their
//!    lift engines' answer to the gravity ([`crate::physics::lift_accel`]) —
//!    or, for a ship in fast travel, [`crate::physics::fast_velocity_step`]
//!    with the limit [`crate::physics::fast_travel_limit`] of its distance to
//!    the nearest surface ([`Physics::surface_distance`]); other
//!    dynamic bodies `v += (F/m + g·scale − k·v)·dt`,
//!    `ω += (I⁻¹·τ − k_ω·ω)·dt` (body frame, no gyroscopic term).
//! 4. Bubble maintenance (above).
//! 5. Each bubble that can have a contact: push poses (from [`WorldPos`]) and
//!    the new velocities into rapier — with **no** gravity, force or damping
//!    of rapier's own — and run one rapier step: contacts, friction,
//!    restitution, joints, CCD and position integration.
//! 6. Read back: free bodies are integrated on their [`WorldPos`] (above),
//!    constrained ones take rapier's pose, rounded into their [`WorldPos`]. A
//!    row is written (and so stamped for replication) only when its bits
//!    changed. Bodies rapier put to sleep keep their state untouched until
//!    something wakes them.
//!
//! # Determinism
//!
//! Same platform + same inputs = bit-identical state: rapier runs with
//! `enhanced-determinism` (libm transcendentals, no hash-map order), single
//! threaded, fixed step [`DT64`] with [`SOLVER_ITERATIONS`] solver
//! iterations; bodies are inserted into a bubble in guid order; bubbles are
//! formed, split, merged and re-anchored by the rules above. Bubble layout and
//! sleep flags are part of [`crate::SimWorld::state_hash`]. Spawning the same
//! bodies in a different order before a step gives the same world.
//!
//! # Dependency facts
//!
//! `rapier3d-f64` 0.36 (Apache-2.0) adds, with its features `dim3 f64 std
//! block-solver enhanced-determinism`: `parry3d-f64`, `glamx`, `glam`,
//! `nalgebra`, `nalgebra-macros`, `simba`, `wide`, `safe_arch`, `bytemuck`,
//! `matrixmultiply`, `rawpointer`, `num-complex`, `num-rational`,
//! `num-integer`, `num-traits`, `num-derive`, `approx`, `typenum`, `libm`,
//! `arrayvec`, `bitflags`, `downcast-rs`, `either`, `ena`, `foldhash`,
//! `hashbrown`, `indexmap`, `equivalent`, `ordered-float`, `rstar`,
//! `heapless`, `hash32`, `byteorder`, `stable_deref_trait`, `smallvec`,
//! `slab`, `spade`, `profiling`, `profiling-procmacros`, `static_assertions`,
//! `thiserror` — all pure Rust; none is a `-sys` crate and no build script
//! compiles C.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::sync::Arc;

use rapier3d_f64::parry::shape::SharedShape;
use rapier3d_f64::prelude::{
    ColliderBuilder, GenericJoint, GenericJointBuilder, Group, IntegrationParameters,
    InteractionGroups, InteractionTestMode, JointAxesMask, JointAxis, MassProperties, Matrix,
    PhysicsWorld, Pose, RigidBodyBuilder, RigidBodyHandle, Vector,
};
use space_ecs::{Entity, World};
use space_protocol::{EntityGuid, ShardId};
use space_world::WorldPos;

use crate::articulation::{Articulation, JointDef, JointKind, JointTarget, PartStates};
use crate::body::{Body, BodyKind, CollisionGroups, MassSpec, PosResidual, Visual};
use crate::components::{ShipControls, ShipState};
use crate::gravity::{GravityField, WorldPoint};
use crate::motion::OrbitPath;
use crate::params::ShipParams;
use crate::physics::{
    fast_travel_limit, fast_velocity_step, lift_accel, quat_step, ship_pose_step,
    ship_velocity_step, top_speed, SETTLE_EPS,
};
use crate::shape::{quat_to_rot, ShapeCatalog, ShapeError, ShapeRef};
use crate::spring::{evaluate, rotate, EndState, SpringTarget, Springs};
use crate::tick::{SimTick, DT};

/// The tick length in `f64`: the `f32` tick [`DT`] widened, so every
/// integrator in the crate agrees on the same number.
pub const DT64: f64 = DT as f64;
/// Merge margin, metres: see the [module docs](self).
pub const MERGE_MARGIN: f64 = 1.0;
/// How far ahead the merge margin looks, seconds (four ticks).
pub const MERGE_LEAD: f64 = 4.0 * DT64;
/// Split margin as a multiple of the merge margin.
pub const SPLIT_FACTOR: f64 = 2.0;
/// A bubble re-anchors when its members' centre is this many cells from its
/// anchor.
pub const REANCHOR_CELLS: i64 = 2;
/// Solver iterations (substeps) per tick.
pub const SOLVER_ITERATIONS: usize = 8;

const UM_PER_M: f64 = 1.0e6;
const UM_PER_CELL: i128 = 1_024_000_000;

/// A bubble's id: handed out in increasing order, in the canonical order the
/// rules create bubbles in, so two worlds with the same inputs agree on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BubbleId(pub u64);

/// A static collider of the scene (the floor, a wall, a bowl, a parked ship).
#[derive(Clone, Debug, PartialEq)]
pub struct StaticCollider {
    /// The shape (an [`ShapeRef::Asset`] is used as its triangle mesh).
    pub shape: ShapeRef,
    /// Where the shape's origin is, exactly.
    pub place: WorldPoint,
    /// Its orientation `(w, x, y, z)`.
    pub rotation: [f32; 4],
    /// Friction coefficient.
    pub friction: f32,
    /// Restitution.
    pub restitution: f32,
    /// What a client draws (0 = nothing; the scene's own visuals cover it).
    pub visual: Visual,
    /// Which bodies it stops ([`CollisionGroups::ALL`]: everything;
    /// [`CollisionGroups::NOT_SHIPS`]: everything but ships).
    pub groups: CollisionGroups,
    /// `None`: it stays at `place`. `Some(path)`: it is at `place` plus the
    /// path's offset at the scene time ([`crate::motion`]) — a planet's
    /// collider moving with the planet's [`crate::gravity::BodyPlace::Orbit`]
    /// (the same path, so the two agree to the bit). Teleported to its new
    /// place every tick while what is near it is carried along (see the
    /// [module docs](self), "One tick"), so nothing slides off it.
    pub motion: Option<OrbitPath>,
}

impl StaticCollider {
    /// Where its origin is at scene time `time_us`.
    #[must_use]
    pub fn place_at(&self, time_us: i64) -> WorldPoint {
        match &self.motion {
            None => self.place,
            Some(path) => {
                let o = path.offset(time_us);
                WorldPoint::new(
                    self.place.origin,
                    core::array::from_fn(|k| self.place.offset[k] + o[k]),
                )
            }
        }
    }
}

/// One bubble as [`crate::SimWorld::bubbles`] reports it.
#[derive(Clone, Debug, PartialEq)]
pub struct BubbleInfo {
    /// Its id.
    pub id: BubbleId,
    /// Its anchor (cell-aligned).
    pub anchor: WorldPos,
    /// Its members, sorted.
    pub members: Vec<EntityGuid>,
    /// The scene statics instantiated in it (indices into the static list).
    pub statics: Vec<usize>,
    /// Whether rapier stepped it in the last tick.
    pub stepped: bool,
}

/// A contact the solver resolved in the last tick (hook for sounds, damage,
/// effects).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactEvent {
    /// One body.
    pub a: EntityGuid,
    /// The other body, or `None` for a static of the scene.
    pub b: Option<EntityGuid>,
    /// Total impulse magnitude of the pair this tick, N·s.
    pub impulse: f64,
}

/// What [`crate::SimWorld::islands`] groups by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IslandQuery {
    /// Contact proximity margin, metres: two bodies whose bounding spheres
    /// are within twice this of each other are grouped.
    pub margin: f64,
    /// Velocity look-ahead, seconds: two bodies are grouped too if, each
    /// moving on in a straight line as it moves now, they come that close at
    /// some moment of the next `lead_time` seconds. (It was once "each body's
    /// box grows by `|v| · lead_time`", in every direction: a ship passing a
    /// hundred metres overhead at 120 m/s was then within reach of everything
    /// below it, in front of it and behind it.)
    pub lead_time: f64,
    /// Whether a spring between two dynamic bodies couples them.
    pub springs: bool,
}

impl Default for IslandQuery {
    /// One metre and one second of look-ahead, springs included.
    fn default() -> Self {
        Self {
            margin: 1.0,
            lead_time: 1.0,
            springs: true,
        }
    }
}

/// A group of dynamic bodies that may interact soon (contact proximity or a
/// spring): what a shard should simulate together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Island {
    /// The dynamic bodies this world simulates, sorted.
    pub bodies: Vec<EntityGuid>,
    /// Ghosts and kinematic bodies within reach of them, sorted (they do not
    /// couple bodies to each other: nothing pushes them back).
    pub touching: Vec<EntityGuid>,
    /// The bubbles the bodies are in, sorted.
    pub bubbles: Vec<BubbleId>,
}

/// Why a body could not join the physics.
#[derive(Clone, Debug, PartialEq)]
pub enum PhysicsError {
    /// A shape that cannot be built.
    Shape(ShapeError),
    /// The guid is already simulated here.
    Duplicate(EntityGuid),
    /// A component the operation needs is missing or inconsistent.
    Invalid(String),
}

impl fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(e) => write!(f, "{e}"),
            Self::Duplicate(g) => write!(f, "{g} is already in this world"),
            Self::Invalid(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for PhysicsError {}

impl From<ShapeError> for PhysicsError {
    fn from(e: ShapeError) -> Self {
        Self::Shape(e)
    }
}

// ---- per-body data ---------------------------------------------------------------

/// What the physics derives once from a part's [`Body`].
#[derive(Clone)]
struct PartPhys {
    dynamic: bool,
    shape: SharedShape,
    radius: f64,
    mprops: MassProperties,
    mass: f64,
    local_com: [f64; 3],
    inv_inertia: Matrix,
    friction: f32,
    restitution: f32,
    groups: CollisionGroups,
    ccd: bool,
    lock_rotation: bool,
    linear_damping: f64,
    angular_damping: f64,
    gravity_scale: f64,
}

struct Entry {
    entity: Entity,
    ghost: bool,
    ship: bool,
    /// A ghost of a **ship** (its owner flies it with `ShipControls`): see
    /// [`meeting_velocities`].
    ship_ghost: bool,
    parts: Vec<PartPhys>,
    joints: Vec<JointDef>,
    bubble: Option<BubbleId>,
    handles: Vec<RigidBodyHandle>,
    /// Fixed rapier bodies standing in for world-anchored joints.
    anchors: Vec<(usize, RigidBodyHandle)>,
    sleeping: bool,
    /// The part states as the physics last read or wrote them.
    last: Vec<ShipState>,
    /// Per part: rapier's pose is the precise (unrounded) pose of the part —
    /// true after the solver moved it, false after free flight or an outside
    /// write (the pose is then re-derived from the `WorldPos`).
    synced: Vec<bool>,
    /// Poses carried from the bubble the entity just left: that bubble's
    /// anchor and, per part, its precise local pose (if synced).
    carried: Option<([i64; 3], Vec<Option<Pose>>)>,
    /// Per part: the position residual (exact position − `WorldPos`) as last
    /// read or written ([`PosResidual`]; zero for a ship).
    res: Vec<[f64; 3]>,
}

impl Entry {
    fn any_dynamic(&self) -> bool {
        !self.ghost && self.parts.iter().any(|p| p.dynamic)
    }
}

struct StaticEntry {
    def: StaticCollider,
    shape: SharedShape,
    /// Where its origin is now (moves with [`StaticCollider::motion`]).
    now: WorldPoint,
    /// Its bounding sphere's centre now.
    center: WorldPoint,
    /// The bounding sphere's centre from the origin, world frame.
    mid: [f64; 3],
    radius: f64,
    /// The shape's own (tight) bounding sphere: centre from the origin,
    /// world frame, and radius — what fast travel measures the distance to
    /// the surface by (a ball's is the ball).
    tight: ([f64; 3], f64),
}

impl StaticEntry {
    fn place(&mut self, time_us: i64) {
        self.now = self.def.place_at(time_us);
        self.center = WorldPoint::new(
            self.now.origin,
            core::array::from_fn(|k| self.now.offset[k] + self.mid[k]),
        );
    }
}

struct Bubble {
    anchor: [i64; 3],
    world: PhysicsWorld,
    members: BTreeSet<EntityGuid>,
    statics: BTreeMap<usize, RigidBodyHandle>,
    stepped: bool,
}

/// Per-tick scratch for one entry.
#[derive(Default)]
struct Work {
    alive: bool,
    awake: bool,
    /// Written behind the physics' back since the last step.
    changed: bool,
    /// Position residual per part (see [`PosResidual`]).
    res: Vec<[f64; 3]>,
    states: Vec<ShipState>,
    next: Vec<ShipState>,
    force: Vec<[f64; 3]>,
    torque: Vec<[f64; 3]>,
    pushed_v: Vec<Vector>,
    pushed_w: Vec<Vector>,
    pushed_com: Vec<Vector>,
}

/// The velocity each **ship's ghost** of a bubble is presented to the solver
/// at this tick, where that is not the velocity its owner reported.
///
/// A ghost is a kinematic body: it pushes with the velocity it is given and
/// cannot be pushed. That is right for what rests on it, and wrong for two
/// **ships** that run into each other across a shard border — each shard
/// simulates its own and holds the other as a ghost, so each ship met an
/// immovable object coming at it and was thrown back at the *other's* speed:
/// the two exchanged velocities (a perfectly elastic collision) where on one
/// shard they stop against each other, and two players pushing at each other
/// across a border bounced metres apart, again and again.
///
/// So a ship's ghost that is closing in on **one** ship of this world and
/// will touch it within the tick is presented with the velocity the two
/// share after an inelastic meeting of equal masses — the mean of the two,
/// along the line between them (every ship has the same mass). The other
/// shard does the same with the roles swapped, from the same two velocities,
/// so both ships come out at that shared velocity without a word between the
/// shards: pushed against each other they stand, and one pushing the other
/// moves off with it at half the thrust's acceleration, as on one shard.
/// Nothing is changed for a ghost that is not a ship's, that is moving away,
/// or that is about to touch several ships at once.
fn meeting_velocities(
    members: &BTreeSet<EntityGuid>,
    index: &BTreeMap<EntityGuid, usize>,
    work: &[Work],
    entries: &BTreeMap<EntityGuid, Entry>,
) -> BTreeMap<EntityGuid, [f64; 3]> {
    let reach = |e: &Entry| {
        e.parts
            .first()
            .map_or(0.0, |p| p.shape.compute_local_bounding_sphere().radius())
    };
    let mut out = BTreeMap::new();
    for g in members {
        let e = &entries[g];
        let wg = &work[index[g]];
        if !e.ship_ghost || !wg.alive || wg.states.is_empty() {
            continue;
        }
        let ghost = wg.states[0];
        let v_ghost = f64s(ghost.vel);
        let mut met: Option<[f64; 3]> = None;
        let mut ships = 0;
        for o in members {
            let eo = &entries[o];
            let wo = &work[index[o]];
            if !eo.ship || !wo.alive || wo.next.is_empty() {
                continue;
            }
            // From the ghost to the ship, and how fast the gap closes.
            let d = WorldPos::delta(ghost.pos, wo.states[0].pos);
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if dist < 1e-9 {
                continue;
            }
            let n = d.map(|c| c / dist);
            let v_ship = f64s(wo.next[0].vel);
            let closing: f64 = (0..3).map(|k| (v_ghost[k] - v_ship[k]) * n[k]).sum();
            if closing <= 0.0 || dist > reach(e) + reach(eo) + closing * f64::from(DT) {
                continue;
            }
            ships += 1;
            met = Some([0, 1, 2].map(|k| v_ghost[k] - 0.5 * closing * n[k]));
        }
        if let (1, Some(shared)) = (ships, met) {
            out.insert(*g, shared);
        }
    }
    out
}

/// The rigid-body world of a [`crate::SimWorld`]. See the [module docs](self).
pub(crate) struct Physics {
    pub(crate) catalog: Arc<ShapeCatalog>,
    statics: Vec<StaticEntry>,
    pub(crate) gravity: GravityField,
    /// The scene time the last step ended at (`None` after a clock reset).
    pub(crate) last_time: Option<i64>,
    entries: BTreeMap<EntityGuid, Entry>,
    bubbles: BTreeMap<BubbleId, Bubble>,
    next_bubble: u64,
    contacts: Vec<ContactEvent>,
    params: IntegrationParameters,
}

impl fmt::Debug for Physics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Physics")
            .field("bodies", &self.entries.len())
            .field("statics", &self.statics.len())
            .field("bubbles", &self.bubbles.len())
            .field("gravity", &self.gravity)
            .finish_non_exhaustive()
    }
}

// ---- small math ------------------------------------------------------------------

fn v3(a: [f64; 3]) -> Vector {
    Vector::new(a[0], a[1], a[2])
}

fn f64s(a: [f32; 3]) -> [f64; 3] {
    [f64::from(a[0]), f64::from(a[1]), f64::from(a[2])]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

fn settle(v: [f32; 3]) -> [f32; 3] {
    if v[0] * v[0] + v[1] * v[1] + v[2] * v[2] < SETTLE_EPS * SETTLE_EPS {
        [0.0; 3]
    } else {
        v
    }
}

fn anchor_pos(anchor: [i64; 3]) -> WorldPos {
    WorldPos {
        cell: anchor,
        local: [0.0; 3],
    }
}

/// A body's pose in the bubble anchored at `anchor`.
fn local_pose(anchor: [i64; 3], pos: WorldPos, quat: [f32; 4]) -> Pose {
    Pose::from_parts(
        v3(WorldPos::delta(anchor_pos(anchor), pos)),
        quat_to_rot(quat),
    )
}

/// What to add to a local coordinate of the bubble anchored at `from` to
/// express it in the bubble anchored at `to`: a whole number of cells, exact
/// in `f64`.
fn anchor_shift(from: [i64; 3], to: [i64; 3]) -> Vector {
    let d = |k: usize| ((i128::from(from[k]) - i128::from(to[k])) * 1024) as f64;
    Vector::new(d(0), d(1), d(2))
}

/// [`local_pose`] of the exact position `pos + residual`.
fn local_pose_res(anchor: [i64; 3], pos: WorldPos, quat: [f32; 4], res: [f64; 3]) -> Pose {
    let d = WorldPos::delta(anchor_pos(anchor), pos);
    Pose::from_parts(
        Vector::new(d[0] + res[0], d[1] + res[1], d[2] + res[2]),
        quat_to_rot(quat),
    )
}

/// The residual of each part as stored, widened (zeros when absent).
fn read_residual(world: &World, entity: Entity, parts: usize) -> Vec<[f64; 3]> {
    match world.get::<PosResidual>(entity) {
        Some(r) if r.0.len() == parts => r.0.iter().map(|v| f64s(*v)).collect(),
        _ => vec![[0.0; 3]; parts],
    }
}

/// Store the residuals (only into an entity that has the component, and only
/// when the stored bits change).
fn write_residual(world: &mut World, entity: Entity, res: &[[f64; 3]]) {
    let new: Vec<[f32; 3]> = res.iter().map(|r| r.map(|x| x as f32)).collect();
    let same = world.get::<PosResidual>(entity).is_none_or(|r| {
        r.0.len() == new.len()
            && r.0
                .iter()
                .flatten()
                .zip(new.iter().flatten())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    });
    if !same {
        if let Some(r) = world.get_mut::<PosResidual>(entity) {
            r.0 = new;
        }
    }
}

/// A world point's offset from `anchor`.
fn local_point(anchor: [i64; 3], p: &WorldPoint) -> [f64; 3] {
    let d = WorldPos::delta(anchor_pos(anchor), p.origin);
    [d[0] + p.offset[0], d[1] + p.offset[1], d[2] + p.offset[2]]
}

/// Position in integer micrometres from the world origin (exact over the
/// whole `i64` cell range).
fn um(p: WorldPos) -> [i128; 3] {
    core::array::from_fn(|i| {
        i128::from(p.cell[i]) * UM_PER_CELL + (f64::from(p.local[i]) * UM_PER_M).round() as i128
    })
}

fn um_extent(m: f64) -> i128 {
    if m.is_finite() {
        (m * UM_PER_M).ceil() as i128 + 1
    } else {
        i128::MAX / 4
    }
}

type Aabb = ([i128; 3], [i128; 3]);

fn grow(b: &mut Option<Aabb>, c: [i128; 3], e: i128) {
    let lo = [c[0] - e, c[1] - e, c[2] - e];
    let hi = [c[0] + e, c[1] + e, c[2] + e];
    *b = Some(match *b {
        None => (lo, hi),
        Some((l, h)) => (
            core::array::from_fn(|i| l[i].min(lo[i])),
            core::array::from_fn(|i| h[i].max(hi[i])),
        ),
    });
}

/// Connected components of overlapping boxes (sweep and prune on x). Each
/// index maps to the smallest index of its component.
fn components(boxes: &[Option<Aabb>], extra_edges: &[(usize, usize)]) -> Vec<usize> {
    components_where(boxes, extra_edges, |_, _| true)
}

/// [`components`], where two overlapping boxes are only joined if
/// `joined(a, b)` says so too.
fn components_where(
    boxes: &[Option<Aabb>],
    extra_edges: &[(usize, usize)],
    joined: impl Fn(usize, usize) -> bool,
) -> Vec<usize> {
    let n = boxes.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
        let (ra, rb) = (find(parent, a), find(parent, b));
        if ra != rb {
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            parent[hi] = lo;
        }
    };
    let mut order: Vec<usize> = (0..n).filter(|&i| boxes[i].is_some()).collect();
    order.sort_by_key(|&i| (boxes[i].map_or(0, |b| b.0[0]), i));
    for (k, &a) in order.iter().enumerate() {
        let Some((alo, ahi)) = boxes[a] else { continue };
        for &b in &order[k + 1..] {
            let Some((blo, bhi)) = boxes[b] else { continue };
            if blo[0] > ahi[0] {
                break;
            }
            if blo[1] <= ahi[1]
                && alo[1] <= bhi[1]
                && blo[2] <= ahi[2]
                && alo[2] <= bhi[2]
                && joined(a, b)
            {
                union(&mut parent, a, b);
            }
        }
    }
    for &(a, b) in extra_edges {
        union(&mut parent, a, b);
    }
    (0..n).map(|i| find(&mut parent, i)).collect()
}

/// One part of a body as the island query sees it: where it is, how it
/// moves, and the radius of its bounding sphere.
#[derive(Clone, Copy)]
struct Reach {
    pos: WorldPos,
    vel: [f32; 3],
    radius: f64,
}

/// Whether two parts, each moving on in a straight line at the speed it has
/// now, come within `gap` metres of each other (bounding sphere to bounding
/// sphere) at some moment of the next `lead` seconds: the closest approach
/// of the relative motion over `[0, lead]`.
///
/// Plain `f64` arithmetic on exact inputs (no square root, nothing
/// transcendental): the same answer on every shard that holds the same two
/// states, and the same whichever of the two is `a`.
fn may_meet(a: &Reach, b: &Reach, gap: f64, lead: f64) -> bool {
    let d = WorldPos::delta(a.pos, b.pos);
    let v: [f64; 3] = core::array::from_fn(|k| f64::from(b.vel[k]) - f64::from(a.vel[k]));
    let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let dv = d[0] * v[0] + d[1] * v[1] + d[2] * v[2];
    let t = if vv > 0.0 && lead > 0.0 {
        (-dv / vv).clamp(0.0, lead)
    } else {
        0.0
    };
    let at: [f64; 3] = core::array::from_fn(|k| d[k] + v[k] * t);
    let reach = a.radius + b.radius + gap;
    at[0] * at[0] + at[1] * at[1] + at[2] * at[2] <= reach * reach
}

/// [`may_meet`] for two bodies: any part of one and any part of the other.
fn bodies_may_meet(a: &[Reach], b: &[Reach], gap: f64, lead: f64) -> bool {
    a.iter()
        .any(|p| b.iter().any(|q| may_meet(p, q, gap, lead)))
}

fn state_bits(s: &ShipState) -> ([i64; 3], [u32; 13]) {
    let mut f = [0u32; 13];
    for (i, v) in s
        .pos
        .local
        .iter()
        .chain(&s.vel)
        .chain(&s.quat)
        .chain(&s.ang_vel)
        .enumerate()
    {
        f[i] = v.to_bits();
    }
    (s.pos.cell, f)
}

fn states_differ(a: &[ShipState], b: &[ShipState]) -> bool {
    a.len() != b.len() || a.iter().zip(b).any(|(x, y)| state_bits(x) != state_bits(y))
}

fn entity_tag(guid: EntityGuid, part: usize) -> u128 {
    (1u128 << 126)
        | ((part as u128 & 0xff) << 96)
        | (u128::from(guid.shard.0) << 64)
        | u128::from(guid.serial)
}

fn static_tag(index: usize) -> u128 {
    (2u128 << 126) | index as u128
}

fn tag_guid(tag: u128) -> Option<EntityGuid> {
    (tag >> 126 == 1).then(|| {
        EntityGuid::new(
            ShardId(((tag >> 64) & 0xffff_ffff) as u32),
            (tag & 0xffff_ffff_ffff_ffff) as u64,
        )
    })
}

fn groups(g: CollisionGroups) -> InteractionGroups {
    InteractionGroups::new(
        Group::from_bits_truncate(g.memberships),
        Group::from_bits_truncate(g.filter),
        InteractionTestMode::And,
    )
}

fn build_part(
    body: &Body,
    catalog: &ShapeCatalog,
    ghost: bool,
    ship: Option<&ShipParams>,
) -> Result<PartPhys, PhysicsError> {
    let dynamic = !ghost && body.kind == BodyKind::Dynamic;
    let shape = body.shape.build(catalog, dynamic)?;
    let aabb = shape.compute_local_aabb();
    let radius = aabb.mins.length().max(aabb.maxs.length());
    let mprops = if let Some(p) = ship {
        let m = f64::from(p.mass);
        MassProperties::new(Vector::ZERO, m, Vector::splat(m))
    } else if dynamic {
        let mut mp = shape.mass_properties(match body.mass {
            MassSpec::Density(d) => f64::from(d),
            MassSpec::Mass(_) => 1.0,
        });
        if let MassSpec::Mass(m) = body.mass {
            mp.set_mass(f64::from(m), true);
        }
        mp
    } else {
        MassProperties::new(Vector::ZERO, 1.0, Vector::ONE)
    };
    let mass = mprops.mass();
    if dynamic && !(mass.is_finite() && mass > 0.0) {
        return Err(PhysicsError::Invalid(format!(
            "a dynamic body needs a positive finite mass, the shape gives {mass}"
        )));
    }
    let lock_rotation = body.lock_rotation;
    let inv_inertia = if lock_rotation || !dynamic {
        Matrix::ZERO
    } else {
        mprops.reconstruct_inverse_inertia_matrix()
    };
    if !inv_inertia.is_finite() {
        return Err(PhysicsError::Invalid("non-finite inertia".into()));
    }
    let c = mprops.local_com;
    Ok(PartPhys {
        dynamic,
        shape,
        radius,
        mprops,
        mass,
        local_com: [c.x, c.y, c.z],
        inv_inertia,
        friction: body.friction,
        restitution: body.restitution,
        groups: body.groups,
        ccd: body.ccd,
        lock_rotation,
        linear_damping: f64::from(body.linear_damping),
        angular_damping: f64::from(body.angular_damping),
        gravity_scale: f64::from(body.gravity_scale),
    })
}

fn joint_of(def: &JointDef) -> GenericJoint {
    let axes = match def.kind {
        JointKind::Fixed => JointAxesMask::LOCKED_FIXED_AXES,
        JointKind::Spherical => JointAxesMask::LOCKED_SPHERICAL_AXES,
        JointKind::Revolute { .. } => JointAxesMask::LOCKED_REVOLUTE_AXES,
        JointKind::Prismatic { .. } => JointAxesMask::LOCKED_PRISMATIC_AXES,
    };
    let mut b = GenericJointBuilder::new(axes)
        .local_frame1(def.frame_a.to_pose())
        .local_frame2(def.frame_b.to_pose())
        .contacts_enabled(false);
    let (axis, limits, motor) = match def.kind {
        JointKind::Revolute { limits, motor } => (JointAxis::AngX, limits, motor),
        JointKind::Prismatic { limits, motor } => (JointAxis::LinX, limits, motor),
        _ => (JointAxis::AngX, None, None),
    };
    if let Some([lo, hi]) = limits {
        b = b.limits(axis, [f64::from(lo), f64::from(hi)]);
    }
    if let Some(m) = motor {
        b = b
            .motor_velocity(axis, f64::from(m.target_velocity), 1.0)
            .motor_max_force(axis, f64::from(m.max_force));
    }
    b.build()
}

// ---- the world ---------------------------------------------------------------------

impl Physics {
    pub(crate) fn new(catalog: Arc<ShapeCatalog>) -> Self {
        let params = IntegrationParameters {
            dt: DT64,
            num_solver_iterations: SOLVER_ITERATIONS,
            // Velocities are the flight model's and the scene's business;
            // rapier's speed cap would silently change a fast ship's motion.
            normalized_max_linear_velocity: f64::MAX,
            ..IntegrationParameters::default()
        };
        Self {
            catalog,
            statics: Vec::new(),
            gravity: GravityField::none(),
            last_time: None,
            entries: BTreeMap::new(),
            bubbles: BTreeMap::new(),
            next_bubble: 1,
            contacts: Vec::new(),
            params,
        }
    }

    fn new_world(&self) -> PhysicsWorld {
        let mut w = PhysicsWorld::new();
        w.gravity = Vector::ZERO;
        w.integration_parameters = self.params;
        w
    }

    /// Replace the scene's static colliders.
    pub(crate) fn set_statics(&mut self, statics: Vec<StaticCollider>) -> Result<(), PhysicsError> {
        let mut built = Vec::with_capacity(statics.len());
        for def in statics {
            let shape = def.shape.build_static(&self.catalog)?;
            let aabb = shape.compute_local_aabb();
            let mid = (aabb.mins + aabb.maxs) * 0.5;
            let radius = (aabb.maxs - aabb.mins).length() * 0.5;
            let m = rotate(def.rotation, [mid.x, mid.y, mid.z]);
            let sphere = shape.compute_local_bounding_sphere();
            let c = sphere.center;
            let tight = (rotate(def.rotation, [c.x, c.y, c.z]), sphere.radius);
            let mut entry = StaticEntry {
                tight,
                now: def.place,
                center: def.place,
                def,
                shape,
                mid: m,
                radius,
            };
            entry.place(self.now_time());
            built.push(entry);
        }
        // Drop every instantiated static; the next tick re-instantiates.
        for b in self.bubbles.values_mut() {
            for (_, h) in std::mem::take(&mut b.statics) {
                b.world.remove_body(h);
            }
        }
        self.statics = built;
        Ok(())
    }

    pub(crate) fn statics(&self) -> impl Iterator<Item = &StaticCollider> {
        self.statics.iter().map(|s| &s.def)
    }

    /// The scene time the world is at: where the last step ended, or the
    /// clock's time of tick 0 before any.
    fn now_time(&self) -> i64 {
        self.last_time
            .unwrap_or_else(|| self.gravity.clock.time_us_at(0))
    }

    /// Distance from the point `offset` metres from `p` to the nearest
    /// surface of the scene at scene time `time_us`, metres: a large body, a
    /// gravity zone or a static's bounding sphere (see
    /// [`crate::physics::fast_travel_limit`]).
    pub(crate) fn surface_distance(
        &self,
        p: WorldPos,
        offset: [f64; 3],
        tick: SimTick,
        time_us: i64,
    ) -> f64 {
        self.nearest_surface(p, offset, tick, time_us).0
    }

    /// [`Physics::surface_distance`] and the unit vector from that surface
    /// towards the point.
    pub(crate) fn nearest_surface(
        &self,
        p: WorldPos,
        offset: [f64; 3],
        tick: SimTick,
        time_us: i64,
    ) -> (f64, [f64; 3]) {
        let mut best = self.gravity.nearest_surface(p, offset, tick, time_us);
        for s in &self.statics {
            let (mid, radius) = s.tight;
            let c = s.def.place_at(time_us);
            let c = WorldPoint::new(c.origin, core::array::from_fn(|k| c.offset[k] + mid[k]));
            let d = c.delta_to(p);
            let d: [f64; 3] = core::array::from_fn(|k| d[k] + offset[k]);
            let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let gap = (r - radius).max(0.0);
            if gap < best.0 {
                best = (gap, if r > 0.0 { d.map(|c| c / r) } else { [0.0; 3] });
            }
        }
        best
    }

    /// Step 0 of a tick (see the [module docs](self)): move the moving
    /// statics to `t1` and carry every body by the orbiting bodies from `t0`
    /// to `t1`.
    fn advance_scene_time(&mut self, world: &mut World, tick: SimTick, t0: i64, t1: i64) {
        // Moving statics, also inside the bubbles that hold them.
        let mut moved = false;
        for s in &mut self.statics {
            if s.def.motion.is_some() {
                s.place(t1);
                moved = true;
            }
        }
        if moved {
            for b in self.bubbles.values_mut() {
                let anchor = b.anchor;
                for (idx, h) in &b.statics {
                    let s = &self.statics[*idx];
                    if s.def.motion.is_none() {
                        continue;
                    }
                    if let Some(rb) = b.world.bodies.get_mut(*h) {
                        rb.set_position(
                            Pose::from_parts(
                                v3(local_point(anchor, &s.now)),
                                quat_to_rot(s.def.rotation),
                            ),
                            false,
                        );
                    }
                }
            }
        }
        if t0 == t1 || !self.gravity.has_orbits() {
            return;
        }
        let guids: Vec<EntityGuid> = self.entries.keys().copied().collect();
        for g in guids {
            let e = &self.entries[&g];
            if e.ghost {
                continue;
            }
            let Some(states) = read_states(world, e.entity, e.parts.len()) else {
                continue;
            };
            let res = read_residual(world, e.entity, e.parts.len());
            let mut out = states.clone();
            let mut res_out = res.clone();
            let mut shifts = vec![[0.0f64; 3]; e.parts.len()];
            let mut any = false;
            for (p, st) in states.iter().enumerate() {
                let r = res.get(p).copied().unwrap_or([0.0; 3]);
                let c = self.gravity.carry(st.pos, r, tick, t0, t1);
                if c == [0.0; 3] {
                    continue;
                }
                any = true;
                shifts[p] = c;
                if p == 0 && e.ship {
                    out[p].pos = WorldPos::from_f64_metres_around(st.pos, c);
                    continue;
                }
                let d: [f64; 3] = core::array::from_fn(|k| r[k] + c[k]);
                let pos = WorldPos::from_f64_metres_around(st.pos, d);
                let back = WorldPos::delta(st.pos, pos);
                out[p].pos = pos;
                if let Some(slot) = res_out.get_mut(p) {
                    *slot = core::array::from_fn(|k| d[k] - back[k]);
                }
            }
            if !any {
                continue;
            }
            let entity = e.entity;
            let was_changed = states_differ(&states, &e.last);
            write_states(world, entity, &out);
            write_residual(world, entity, &res_out);
            let e = self.entries.get_mut(&g).expect("listed");
            // A carry is not a write behind the physics' back: the body is
            // where the physics put it, only moved with the frame.
            if !was_changed {
                e.last.clone_from(&out);
            }
            e.res.clone_from(&res_out);
            if let Some(b) = e.bubble.and_then(|id| self.bubbles.get_mut(&id)) {
                for (p, h) in e.handles.iter().enumerate() {
                    if !e.synced.get(p).copied().unwrap_or(false) {
                        continue;
                    }
                    if let Some(rb) = b.world.bodies.get_mut(*h) {
                        let mut pose = *rb.position();
                        pose.translation += v3(shifts[p]);
                        rb.set_position(pose, false);
                    }
                }
            }
        }
    }

    /// Add an entity (its components are in `world`).
    pub(crate) fn register(
        &mut self,
        guid: EntityGuid,
        entity: Entity,
        world: &World,
        ghost: bool,
        params: &ShipParams,
    ) -> Result<(), PhysicsError> {
        if self.entries.contains_key(&guid) {
            return Err(PhysicsError::Duplicate(guid));
        }
        let body = world
            .get::<Body>(entity)
            .ok_or_else(|| PhysicsError::Invalid("a physics entity needs a Body".into()))?;
        let ship = world.get::<ShipControls>(entity).is_some();
        let mut parts = vec![build_part(
            body,
            &self.catalog,
            ghost,
            (ship && !ghost).then_some(params),
        )?];
        let mut joints = Vec::new();
        if let Some(a) = world.get::<Articulation>(entity) {
            for p in &a.parts {
                parts.push(build_part(&p.body, &self.catalog, ghost, None)?);
            }
            joints = a.joints.clone();
            let have = world.get::<PartStates>(entity).map_or(0, |s| s.0.len());
            if have != a.parts.len() {
                return Err(PhysicsError::Invalid(format!(
                    "an articulation of {} extra parts needs {} part states, got {have}",
                    a.parts.len(),
                    a.parts.len()
                )));
            }
        }
        let last = read_states(world, entity, parts.len())
            .ok_or_else(|| PhysicsError::Invalid("a physics entity needs a ShipState".into()))?;
        self.entries.insert(
            guid,
            Entry {
                entity,
                ghost,
                ship: ship && !ghost,
                ship_ghost: false,
                parts,
                joints,
                bubble: None,
                handles: Vec::new(),
                anchors: Vec::new(),
                sleeping: false,
                synced: vec![false; last.len()],
                res: read_residual(world, entity, last.len()),
                last,
                carried: None,
            },
        );
        Ok(())
    }

    /// Say whether the ghost `guid` is a ghost of a ship (a ghost has no
    /// `ShipControls` of its own to tell by).
    pub(crate) fn mark_ship_ghost(&mut self, guid: EntityGuid, ship_ghost: bool) {
        if let Some(e) = self.entries.get_mut(&guid) {
            e.ship_ghost = e.ghost && ship_ghost;
        }
    }

    /// Remove an entity from the physics (its ECS entity is the caller's).
    pub(crate) fn unregister(&mut self, guid: EntityGuid) -> bool {
        let Some(mut e) = self.entries.remove(&guid) else {
            return false;
        };
        if let Some(b) = e.bubble.and_then(|id| self.bubbles.get_mut(&id)) {
            remove_from(&mut b.world, &mut e);
            b.members.remove(&guid);
        }
        true
    }

    pub(crate) fn is_ghost(&self, guid: EntityGuid) -> bool {
        self.entries.get(&guid).is_some_and(|e| e.ghost)
    }

    pub(crate) fn guids(&self) -> impl Iterator<Item = (EntityGuid, Entity, bool)> + '_ {
        self.entries.iter().map(|(g, e)| (*g, e.entity, e.ghost))
    }

    pub(crate) fn is_sleeping(&self, guid: EntityGuid) -> Option<bool> {
        self.entries.get(&guid).map(|e| e.sleeping)
    }

    pub(crate) fn bubble_of(&self, guid: EntityGuid) -> Option<BubbleId> {
        self.entries.get(&guid).and_then(|e| e.bubble)
    }

    pub(crate) fn contacts(&self) -> &[ContactEvent] {
        &self.contacts
    }

    pub(crate) fn bubbles(&self) -> Vec<BubbleInfo> {
        self.bubbles
            .iter()
            .map(|(id, b)| BubbleInfo {
                id: *id,
                anchor: anchor_pos(b.anchor),
                members: b.members.iter().copied().collect(),
                statics: b.statics.keys().copied().collect(),
                stepped: b.stepped,
            })
            .collect()
    }

    /// The canonical bytes of the derived physics state the hash covers:
    /// bubbles (id, anchor, members) and the sleeping bodies.
    pub(crate) fn hash_bytes(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.bubbles.len() as u64).to_be_bytes());
        for (id, b) in &self.bubbles {
            out.extend_from_slice(&id.0.to_be_bytes());
            for c in b.anchor {
                out.extend_from_slice(&c.to_be_bytes());
            }
            out.extend_from_slice(&(b.members.len() as u64).to_be_bytes());
            for g in &b.members {
                out.extend_from_slice(&g.shard.0.to_be_bytes());
                out.extend_from_slice(&g.serial.to_be_bytes());
            }
        }
        let sleeping: Vec<&EntityGuid> = self
            .entries
            .iter()
            .filter(|(_, e)| e.sleeping)
            .map(|(g, _)| g)
            .collect();
        out.extend_from_slice(&(sleeping.len() as u64).to_be_bytes());
        for g in sleeping {
            out.extend_from_slice(&g.shard.0.to_be_bytes());
            out.extend_from_slice(&g.serial.to_be_bytes());
        }
    }

    /// Wake a body (its next step integrates it whatever rapier thought).
    pub(crate) fn wake(&mut self, guid: EntityGuid) {
        if let Some(e) = self.entries.get_mut(&guid) {
            e.sleeping = false;
            if let Some(b) = e.bubble.and_then(|id| self.bubbles.get_mut(&id)) {
                for h in &e.handles {
                    if let Some(rb) = b.world.bodies.get_mut(*h) {
                        rb.wake_up(true);
                    }
                }
            }
        }
    }

    // ---- the tick ------------------------------------------------------------

    /// Advance every body by one tick; `tick` is the tick being produced.
    pub(crate) fn step(&mut self, world: &mut World, tick: SimTick, params: &ShipParams) {
        self.contacts.clear();
        // 0. Scene time: statics on orbits move, what is near an orbiting
        //    body moves with it.
        let next = tick.wrapping_add(1);
        let t1 = self.gravity.clock.time_us_at(next);
        let t0 = self
            .last_time
            .unwrap_or_else(|| self.gravity.clock.time_us_at(tick));
        self.advance_scene_time(world, tick, t0, t1);
        self.last_time = Some(t1);
        // Entities despawned behind the physics' back leave it.
        let dead: Vec<EntityGuid> = self
            .entries
            .iter()
            .filter(|(_, e)| !world.entity_is_alive(e.entity))
            .map(|(g, _)| *g)
            .collect();
        for g in dead {
            self.unregister(g);
        }
        let guids: Vec<EntityGuid> = self.entries.keys().copied().collect();
        let index: BTreeMap<EntityGuid, usize> =
            guids.iter().enumerate().map(|(i, g)| (*g, i)).collect();

        // 1. Read states; external writes wake.
        let mut work: Vec<Work> = Vec::with_capacity(guids.len());
        for g in &guids {
            let e = &self.entries[g];
            let mut w = Work::default();
            if let Some(states) = read_states(world, e.entity, e.parts.len()) {
                w.alive = true;
                let changed = states_differ(&states, &e.last);
                let driven = e.ship
                    && world
                        .get::<ShipControls>(e.entity)
                        .is_some_and(|c| c.thrust.iter().chain(&c.torque).any(|v| *v != 0.0));
                w.awake = e.ghost || !e.sleeping || changed || driven;
                w.changed = changed;
                // A state written from outside makes the old residual meaningless.
                w.res = if changed {
                    vec![[0.0; 3]; e.parts.len()]
                } else {
                    read_residual(world, e.entity, e.parts.len())
                };
                w.next = states.clone();
                w.states = states;
                w.force = vec![[0.0; 3]; e.parts.len()];
                w.torque = vec![[0.0; 3]; e.parts.len()];
            }
            work.push(w);
        }

        // 2. Springs: evaluated from start-of-tick states, applied to the ends
        //    this world simulates.
        let end = |work: &[Work], entries: &BTreeMap<EntityGuid, Entry>, i: usize, part: usize| {
            let e = &entries[&guids[i]];
            let s = &work[i].states[part];
            let p = &e.parts[part];
            EndState {
                pos: s.pos,
                quat: s.quat,
                vel: f64s(s.vel),
                ang_vel: rotate(s.quat, f64s(s.ang_vel)),
                local_com: p.local_com,
                mass: if p.dynamic && !e.ghost {
                    p.mass
                } else {
                    f64::INFINITY
                },
                residual: work[i].res.get(part).copied().unwrap_or([0.0; 3]),
            }
        };
        let mut applied: Vec<(usize, usize, [f64; 3], [f64; 3])> = Vec::new();
        let mut wake_pairs: Vec<(usize, usize)> = Vec::new();
        for (i, g) in guids.iter().enumerate() {
            if !work[i].alive {
                continue;
            }
            let e = &self.entries[g];
            let Some(springs) = world.get::<Springs>(e.entity) else {
                continue;
            };
            for s in &springs.0 {
                let pa = usize::from(s.part);
                if pa >= e.parts.len() {
                    continue;
                }
                let a = end(&work, &self.entries, i, pa);
                let (b, other, anchor_b) = match s.target {
                    SpringTarget::World(p) => (EndState::fixed(p), None, [0.0; 3]),
                    SpringTarget::Entity { guid, part, anchor } => {
                        let Some(&j) = index.get(&guid) else { continue };
                        let pb = usize::from(part);
                        if !work[j].alive || pb >= self.entries[&guid].parts.len() {
                            continue;
                        }
                        (end(&work, &self.entries, j, pb), Some((j, pb)), anchor)
                    }
                };
                let f = evaluate(s, &a, &b, anchor_b, DT64);
                if !e.ghost {
                    applied.push((i, pa, f.force, cross(f.arm_a, f.force)));
                }
                if let Some((j, pb)) = other {
                    let neg = [-f.force[0], -f.force[1], -f.force[2]];
                    if !self.entries[&guids[j]].ghost {
                        applied.push((j, pb, neg, cross(f.arm_b, neg)));
                    }
                    wake_pairs.push((i, j));
                }
            }
        }
        // A sleeping end of a spring wakes when the other end is awake.
        for (i, j) in wake_pairs {
            let moving = |w: &Work| {
                w.awake
                    && w.states
                        .iter()
                        .any(|s| s.vel != [0.0; 3] || s.ang_vel != [0.0; 3])
            };
            let (mi, mj) = (moving(&work[i]), moving(&work[j]));
            if mj {
                work[i].awake = true;
            }
            if mi {
                work[j].awake = true;
            }
        }
        for (i, part, f, t) in applied {
            for k in 0..3 {
                work[i].force[part][k] += f[k];
                work[i].torque[part][k] += t[k];
            }
        }

        // 3. Velocities.
        for (i, g) in guids.iter().enumerate() {
            let w = &mut work[i];
            if !w.alive || !w.awake {
                continue;
            }
            let e = &self.entries[g];
            if e.ghost {
                continue;
            }
            for (p, part) in e.parts.iter().enumerate() {
                if !part.dynamic {
                    continue;
                }
                let s = w.states[p];
                let grav = if part.gravity_scale != 0.0 && !self.gravity.is_empty() {
                    let r = w.res.get(p).copied().unwrap_or([0.0; 3]);
                    self.gravity
                        .at_time(s.pos, r, tick, t1)
                        .map(|a| a * part.gravity_scale)
                } else {
                    [0.0; 3]
                };
                let inv_m = 1.0 / part.mass;
                let ext: [f64; 3] = core::array::from_fn(|k| grav[k] + w.force[p][k] * inv_m);
                let mut n = s;
                if p == 0 && e.ship {
                    let c = world
                        .get::<ShipControls>(e.entity)
                        .copied()
                        .unwrap_or_default();
                    // The ship's lift engines answer the gravity (zero in
                    // weightless space: free flight is unchanged).
                    let lift = lift_accel(s.quat, c.thrust, params, grav);
                    let ext: [f64; 3] = core::array::from_fn(|k| ext[k] + lift[k]);
                    let limit = if c.fast {
                        let (d, away) = self.nearest_surface(s.pos, [0.0; 3], tick, t1);
                        let v = f64s(s.vel);
                        let closing = v[0] * away[0] + v[1] * away[1] + v[2] * away[2] < 0.0;
                        fast_travel_limit(d, closing, params)
                    } else {
                        0.0
                    };
                    if c.fast && limit > f64::from(top_speed(params)) {
                        fast_velocity_step(
                            &mut n,
                            c.thrust,
                            c.torque,
                            params,
                            ext.map(|a| a as f32),
                            limit,
                        );
                    } else {
                        ship_velocity_step(
                            &mut n,
                            c.thrust,
                            c.torque,
                            params,
                            ext.map(|a| a as f32),
                        );
                        if c.fast {
                            crate::physics::limit_speed(&mut n, limit);
                        }
                    }
                } else {
                    let v = f64s(s.vel);
                    let nv: [f64; 3] = core::array::from_fn(|k| {
                        v[k] + (ext[k] - part.linear_damping * v[k]) * DT64
                    });
                    let wb = f64s(s.ang_vel);
                    let tb = rotate(conj(s.quat), w.torque[p]);
                    let alpha = part.inv_inertia * v3(tb);
                    let alpha = [alpha.x, alpha.y, alpha.z];
                    let nw: [f64; 3] = core::array::from_fn(|k| {
                        wb[k] + (alpha[k] - part.angular_damping * wb[k]) * DT64
                    });
                    n.vel = settle(nv.map(|x| x as f32));
                    n.ang_vel = settle(nw.map(|x| x as f32));
                }
                w.next[p] = n;
            }
        }

        // 4. Bubbles.
        self.maintain(&guids, &work);

        // 5. Push and step.
        let ids: Vec<BubbleId> = self.bubbles.keys().copied().collect();
        for id in &ids {
            let needs = {
                let b = &self.bubbles[id];
                // Something moves: an awake dynamic body, or a ghost or
                // kinematic body in motion (it may run into a sleeping one).
                let awake_dynamic = b.members.iter().any(|g| {
                    let (k, e) = (index[g], &self.entries[g]);
                    let w = &work[k];
                    w.alive
                        && ((w.awake && e.any_dynamic())
                            || ((e.ghost || !e.any_dynamic())
                                && (w.changed
                                    || w.states
                                        .iter()
                                        .any(|s| s.vel != [0.0; 3] || s.ang_vel != [0.0; 3]))))
                });
                let jointed = b.members.iter().any(|g| !self.entries[g].joints.is_empty());
                awake_dynamic && (!b.statics.is_empty() || b.members.len() > 1 || jointed)
            };
            let b = self.bubbles.get_mut(id).expect("listed above");
            b.stepped = needs;
            if !needs {
                continue;
            }
            let meeting = meeting_velocities(&b.members, &index, &work, &self.entries);
            for g in &b.members {
                let k = index[g];
                let e = &self.entries[g];
                let w = &mut work[k];
                w.pushed_v = vec![Vector::ZERO; e.parts.len()];
                w.pushed_w = vec![Vector::ZERO; e.parts.len()];
                w.pushed_com = vec![Vector::ZERO; e.parts.len()];
                if !w.alive || !w.awake {
                    continue;
                }
                for (p, part) in e.parts.iter().enumerate() {
                    let Some(rb) = e.handles.get(p).and_then(|h| b.world.bodies.get_mut(*h)) else {
                        continue;
                    };
                    let (s, n) = (w.states[p], w.next[p]);
                    let synced = e.synced.get(p).copied().unwrap_or(false);
                    if e.ghost || !part.dynamic || !synced || w.changed {
                        let r = w.res.get(p).copied().unwrap_or([0.0; 3]);
                        rb.set_position(local_pose_res(b.anchor, s.pos, s.quat, r), false);
                    }
                    let lin = v3(f64s(n.vel));
                    let ang = v3(rotate(s.quat, f64s(n.ang_vel)));
                    if e.ghost || !part.dynamic {
                        // (A ship's ghost about to meet a ship of this world
                        // is presented at the velocity the two will share.)
                        let vel = match meeting.get(g) {
                            Some(shared) if p == 0 => *shared,
                            _ => f64s(s.vel),
                        };
                        rb.set_linvel(v3(vel), false);
                        rb.set_angvel(v3(rotate(s.quat, f64s(s.ang_vel))), false);
                        continue;
                    }
                    if rb.is_sleeping() {
                        rb.wake_up(true);
                    }
                    rb.set_linvel(lin, false);
                    rb.set_angvel(
                        if part.lock_rotation {
                            Vector::ZERO
                        } else {
                            ang
                        },
                        false,
                    );
                    w.pushed_v[p] = rb.linvel();
                    w.pushed_w[p] = rb.angvel();
                    w.pushed_com[p] = rb.center_of_mass();
                }
            }
            b.world.step();
            // Contact events.
            for pair in b.world.contact_pairs() {
                let impulse = pair.total_impulse_magnitude();
                if impulse <= 0.0 {
                    continue;
                }
                let tag = |h| b.world.colliders.get(h).map_or(0, |c| c.user_data);
                let (t1, t2) = (tag(pair.collider1), tag(pair.collider2));
                match (tag_guid(t1), tag_guid(t2)) {
                    (Some(a), other) => self.contacts.push(ContactEvent {
                        a,
                        b: other,
                        impulse,
                    }),
                    (None, Some(a)) => self.contacts.push(ContactEvent {
                        a,
                        b: None,
                        impulse,
                    }),
                    (None, None) => {}
                }
            }
        }
        for c in &mut self.contacts {
            if let Some(b) = c.b {
                if b < c.a {
                    (c.a, c.b) = (b, Some(c.a));
                }
            }
        }
        self.contacts.sort_by_key(|x| (x.a, x.b));

        // 6. Read back and write.
        for (k, g) in guids.iter().enumerate() {
            let w = &work[k];
            let Some(e) = self.entries.get_mut(g) else {
                continue;
            };
            if !w.alive || e.ghost {
                if w.alive {
                    e.last.clone_from(&w.states);
                }
                continue;
            }
            let bubble = e.bubble.and_then(|id| self.bubbles.get_mut(&id));
            let (stepped, anchor) = bubble
                .as_ref()
                .map_or((false, [0; 3]), |b| (b.stepped, b.anchor));
            if !w.awake {
                // Asleep at the start of the tick: untouched, unless the solver
                // woke it (a body ran into it) — then read it back.
                let woken = stepped
                    && bubble.as_ref().is_some_and(|b| {
                        e.handles
                            .iter()
                            .any(|h| b.world.bodies.get(*h).is_some_and(|rb| !rb.is_sleeping()))
                    });
                if !woken {
                    continue;
                }
            }
            let mut out = w.states.clone();
            let mut res_out = w.res.clone();
            res_out.resize(e.parts.len(), [0.0; 3]);
            let mut asleep = stepped;
            for (p, part) in e.parts.iter().enumerate() {
                let n = w.next[p];
                if !part.dynamic {
                    // Kinematic: moves with its own velocity.
                    let (s, r) = integrate_free(w.states[p], part.local_com, false, res_out[p]);
                    out[p] = s;
                    res_out[p] = r;
                    asleep = false;
                    continue;
                }
                let rb = bubble
                    .as_ref()
                    .filter(|_| stepped)
                    .and_then(|b| e.handles.get(p).and_then(|h| b.world.bodies.get(*h)));
                let free = match rb {
                    None => true,
                    Some(rb) => {
                        asleep &= rb.is_sleeping();
                        let expected = w.pushed_com[p] + w.pushed_v[p] * DT64;
                        e.joints.is_empty()
                            && !rb.is_sleeping()
                            && rb.linvel() == w.pushed_v[p]
                            && (part.lock_rotation || rb.angvel() == w.pushed_w[p])
                            && (rb.center_of_mass() - expected).length()
                                <= 1.0e-9 * (1.0 + expected.length())
                    }
                };
                if free {
                    let (s, r) = integrate_free(n, part.local_com, p == 0 && e.ship, res_out[p]);
                    out[p] = s;
                    res_out[p] = r;
                    if let Some(s) = e.synced.get_mut(p) {
                        *s = false;
                    }
                    continue;
                }
                if let Some(s) = e.synced.get_mut(p) {
                    *s = true;
                }
                let rb = rb.expect("not free means stepped");
                let t = rb.translation();
                let pos = WorldPos::from_f64_metres_around(anchor_pos(anchor), [t.x, t.y, t.z]);
                let back = WorldPos::delta(anchor_pos(anchor), pos);
                res_out[p] = if p == 0 && e.ship {
                    [0.0; 3]
                } else {
                    [t.x - back[0], t.y - back[1], t.z - back[2]]
                };
                let v = rb.linvel();
                let mut s = n;
                if part.lock_rotation {
                    quat_step(&mut s);
                } else {
                    let r = rb.rotation();
                    s.quat = [r.w as f32, r.x as f32, r.y as f32, r.z as f32];
                    let wv = rb.angvel();
                    let wb = rotate(conj(s.quat), [wv.x, wv.y, wv.z]);
                    s.ang_vel = settle(wb.map(|x| x as f32));
                }
                s.pos = pos;
                s.vel = settle([v.x as f32, v.y as f32, v.z as f32]);
                if rb.is_sleeping() {
                    s.vel = [0.0; 3];
                    s.ang_vel = [0.0; 3];
                }
                out[p] = s;
            }
            let has_dynamic = e.parts.iter().any(|p| p.dynamic);
            e.sleeping = stepped && asleep && has_dynamic;
            if states_differ(&out, &w.states) {
                write_states(world, e.entity, &out);
            }
            write_residual(world, e.entity, &res_out);
            e.last = out;
            e.res = res_out;
        }
    }

    // ---- bubble maintenance --------------------------------------------------

    fn new_bubble_id(&mut self) -> BubbleId {
        let id = BubbleId(self.next_bubble);
        self.next_bubble += 1;
        id
    }

    fn maintain(&mut self, guids: &[EntityGuid], work: &[Work]) {
        let n = guids.len();
        // Bounds.
        let mut merge_boxes: Vec<Option<Aabb>> = vec![None; n];
        let mut split_boxes: Vec<Option<Aabb>> = vec![None; n];
        let mut centers: Vec<Option<Aabb>> = vec![None; n];
        for (i, g) in guids.iter().enumerate() {
            let w = &work[i];
            if !w.alive {
                continue;
            }
            let e = &self.entries[g];
            for (p, part) in e.parts.iter().enumerate() {
                let c = um(w.states[p].pos);
                let v = w.next[p].vel;
                let speed = f64::from((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                let m = MERGE_MARGIN + speed * MERGE_LEAD;
                grow(&mut merge_boxes[i], c, um_extent(part.radius + m));
                grow(
                    &mut split_boxes[i],
                    c,
                    um_extent(part.radius + SPLIT_FACTOR * m),
                );
                grow(&mut centers[i], c, 0);
            }
        }
        // Joints never need edges: an entity is one box.
        let c1 = components(&merge_boxes, &[]);
        let c2 = components(&split_boxes, &[]);

        let mut assign: Vec<Option<BubbleId>> =
            guids.iter().map(|g| self.entries[g].bubble).collect();
        for (i, a) in assign.iter_mut().enumerate() {
            if !work[i].alive {
                *a = None;
            }
        }
        // Split.
        let mut by_bubble: BTreeMap<BubbleId, Vec<usize>> = BTreeMap::new();
        for (i, a) in assign.iter().enumerate() {
            if let Some(b) = a {
                by_bubble.entry(*b).or_default().push(i);
            }
        }
        for (_, members) in by_bubble {
            let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
            for i in members {
                match groups.iter_mut().find(|(root, _)| *root == c2[i]) {
                    Some((_, v)) => v.push(i),
                    None => groups.push((c2[i], vec![i])),
                }
            }
            // The largest group keeps the bubble (ties: the first, which
            // holds the smallest guid); the others get new ones, in order of
            // their smallest guid. Keeping it means keeping the solver's
            // world as it is — bodies, contacts, sleep — so it is the group
            // with the most to lose that stays (see the module docs, rule 2).
            let keeps = groups
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.1.len().cmp(&b.1.len()).then(j.cmp(i)))
                .map_or(0, |(k, _)| k);
            for (k, (_, group)) in groups.into_iter().enumerate() {
                if k == keeps {
                    continue;
                }
                let id = self.new_bubble_id();
                for i in group {
                    assign[i] = Some(id);
                }
            }
        }
        // Merge.
        let mut count: BTreeMap<BubbleId, usize> = BTreeMap::new();
        for b in assign.iter().flatten() {
            *count.entry(*b).or_default() += 1;
        }
        let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for i in 0..n {
            if work[i].alive {
                comps.entry(c1[i]).or_default().push(i);
            }
        }
        for (_, members) in comps {
            let found: BTreeSet<BubbleId> = members.iter().filter_map(|&i| assign[i]).collect();
            let survivor = match found
                .iter()
                .max_by(|a, b| count[a].cmp(&count[b]).then(b.cmp(a)))
            {
                Some(s) => *s,
                None => self.new_bubble_id(),
            };
            if found.len() > 1 {
                for a in assign.iter_mut() {
                    if let Some(b) = *a {
                        if b != survivor && found.contains(&b) {
                            *a = Some(survivor);
                        }
                    }
                }
            }
            for &i in &members {
                assign[i] = Some(survivor);
            }
            count.clear();
            for b in assign.iter().flatten() {
                *count.entry(*b).or_default() += 1;
            }
        }
        // Anchors.
        let mut extent: BTreeMap<BubbleId, Aabb> = BTreeMap::new();
        for (i, a) in assign.iter().enumerate() {
            if let (Some(b), Some((lo, hi))) = (a, centers[i]) {
                let e = extent.entry(*b).or_insert((lo, hi));
                for k in 0..3 {
                    e.0[k] = e.0[k].min(lo[k]);
                    e.1[k] = e.1[k].max(hi[k]);
                }
            }
        }
        let center_cell = |(lo, hi): Aabb| -> [i64; 3] {
            core::array::from_fn(|k| {
                let c = lo[k] + (hi[k] - lo[k]) / 2;
                (c + UM_PER_CELL / 2).div_euclid(UM_PER_CELL) as i64
            })
        };

        // Apply: leave old bubbles.
        for (i, g) in guids.iter().enumerate() {
            let e = self.entries.get_mut(g).expect("listed");
            if e.bubble != assign[i] {
                if let Some(b) = e.bubble.and_then(|id| self.bubbles.get_mut(&id)) {
                    let poses = e
                        .handles
                        .iter()
                        .enumerate()
                        .map(|(p, h)| {
                            let synced = e.synced.get(p).copied().unwrap_or(false);
                            b.world
                                .bodies
                                .get(*h)
                                .filter(|_| synced)
                                .map(|rb| *rb.position())
                        })
                        .collect();
                    e.carried = Some((b.anchor, poses));
                    remove_from(&mut b.world, e);
                    b.members.remove(g);
                }
                e.bubble = None;
            }
        }
        // Drop empty bubbles, create new ones, re-anchor.
        self.bubbles.retain(|id, _| extent.contains_key(id));
        for (id, ext) in &extent {
            let want = center_cell(*ext);
            if let Some(b) = self.bubbles.get_mut(id) {
                if (0..3).any(|k| (want[k] - b.anchor[k]).abs() >= REANCHOR_CELLS) {
                    let shift = anchor_shift(b.anchor, want);
                    b.anchor = want;
                    let anchor = b.anchor;
                    for (idx, h) in &b.statics {
                        let s = &self.statics[*idx];
                        if let Some(rb) = b.world.bodies.get_mut(*h) {
                            rb.set_position(
                                Pose::from_parts(
                                    v3(local_point(anchor, &s.now)),
                                    quat_to_rot(s.def.rotation),
                                ),
                                false,
                            );
                        }
                    }
                    for g in &b.members {
                        let e = &self.entries[g];
                        for (p, h) in e.handles.iter().enumerate() {
                            if let Some(rb) = b.world.bodies.get_mut(*h) {
                                let pose = if e.synced.get(p).copied().unwrap_or(false) {
                                    let mut pose = *rb.position();
                                    pose.translation += shift;
                                    pose
                                } else {
                                    let s = &e.last[p];
                                    let r = e.res.get(p).copied().unwrap_or([0.0; 3]);
                                    local_pose_res(anchor, s.pos, s.quat, r)
                                };
                                rb.set_position(pose, false);
                            }
                        }
                        for (j, h) in &e.anchors {
                            if let (Some(rb), JointTarget::World { pos, rotation }) =
                                (b.world.bodies.get_mut(*h), e.joints[*j].b)
                            {
                                rb.set_position(local_pose(anchor, pos, rotation), false);
                            }
                        }
                    }
                }
            } else {
                let world = self.new_world();
                self.bubbles.insert(
                    *id,
                    Bubble {
                        anchor: want,
                        world,
                        members: BTreeSet::new(),
                        statics: BTreeMap::new(),
                        stepped: false,
                    },
                );
            }
        }
        // Join new bubbles, in guid order.
        for (i, g) in guids.iter().enumerate() {
            let Some(id) = assign[i] else { continue };
            let e = self.entries.get_mut(g).expect("listed");
            let b = self.bubbles.get_mut(&id).expect("created above");
            if e.bubble.is_none() {
                insert_into(&mut b.world, b.anchor, *g, e, &work[i].states);
                e.bubble = Some(id);
                e.carried = None;
            }
            b.members.insert(*g);
        }
        // Statics.
        for b in self.bubbles.values_mut() {
            for (idx, s) in self.statics.iter().enumerate() {
                let reach = |factor: f64| {
                    b.members.iter().any(|g| {
                        let e = &self.entries[g];
                        let k = guids.binary_search(g).unwrap_or(0);
                        let w = &work[k];
                        e.parts.iter().enumerate().any(|(p, part)| {
                            let Some(st) = w.states.get(p) else {
                                return false;
                            };
                            let v = w.next[p].vel;
                            let speed = f64::from((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                            let m = factor * (MERGE_MARGIN + speed * MERGE_LEAD);
                            let d = s.center.delta_to(st.pos);
                            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                            dist <= part.radius + m + s.radius
                        })
                    })
                };
                let present = b.statics.contains_key(&idx);
                if !present && reach(1.0) {
                    let pose = Pose::from_parts(
                        v3(local_point(b.anchor, &s.now)),
                        quat_to_rot(s.def.rotation),
                    );
                    let h = b.world.insert_body(RigidBodyBuilder::fixed().pose(pose));
                    let c = ColliderBuilder::new(s.shape.clone())
                        .friction(f64::from(s.def.friction))
                        .restitution(f64::from(s.def.restitution))
                        .user_data(static_tag(idx))
                        .collision_groups(groups(s.def.groups))
                        .build();
                    b.world.insert_collider(c, Some(h));
                    b.statics.insert(idx, h);
                    if s.def.motion.is_some() {
                        // A moving static is teleported every tick (and what
                        // touches it with it): rapier's contact recycling
                        // keeps a pair's contacts frozen in the bubble's
                        // frame while their relative pose barely changes,
                        // which a teleported pair defeats (the ship sinks
                        // into the Moon until the drift forces an update).
                        // Contacts are recomputed every step in such a bubble.
                        b.world
                            .integration_parameters
                            .normalized_contact_recycle_distance = 0.0;
                    }
                } else if present && !reach(SPLIT_FACTOR) {
                    if let Some(h) = b.statics.remove(&idx) {
                        b.world.remove_body(h);
                    }
                }
            }
        }
    }

    /// See [`crate::SimWorld::overlaps`].
    pub(crate) fn overlaps(
        &self,
        world: &World,
        body: &Body,
        state: &ShipState,
        clearance: f64,
    ) -> Result<bool, PhysicsError> {
        use rapier3d_f64::parry::query;
        let shape = body
            .shape
            .build(&self.catalog, body.kind == BodyKind::Dynamic)?;
        let aabb = shape.compute_local_aabb();
        let radius = aabb.mins.length().max(aabb.maxs.length());
        let here = Pose::from_parts(Vector::ZERO, quat_to_rot(state.quat));
        let interact = |g: CollisionGroups| {
            body.groups.memberships & g.filter != 0 && g.memberships & body.groups.filter != 0
        };
        // `true` when the other shape, at `offset` from `state.pos` with
        // rotation `rot`, is closer than the clearance.
        let close = |other: &SharedShape, other_radius: f64, offset: [f64; 3], rot: [f32; 4]| {
            let reach = radius + other_radius + clearance;
            if offset.iter().map(|v| v * v).sum::<f64>() > reach * reach {
                return false;
            }
            let there = Pose::from_parts(v3(offset), quat_to_rot(rot));
            match query::contact(&here, shape.as_ref(), &there, other.as_ref(), clearance) {
                Ok(Some(c)) => c.dist < clearance,
                Ok(None) => false,
                Err(_) => true,
            }
        };
        for s in &self.statics {
            if !interact(s.def.groups) {
                continue;
            }
            let d = WorldPos::delta(state.pos, s.now.origin);
            let offset = core::array::from_fn(|k| d[k] + s.now.offset[k]);
            let c = WorldPos::delta(state.pos, s.center.origin);
            let center: [f64; 3] = core::array::from_fn(|k| c[k] + s.center.offset[k]);
            let reach = radius + s.radius + clearance;
            if center.iter().map(|v| v * v).sum::<f64>() > reach * reach {
                continue;
            }
            // The static's own radius is about its centre; the contact query
            // is about its origin, so the prefilter is done above.
            if close(&s.shape, f64::INFINITY, offset, s.def.rotation) {
                return Ok(true);
            }
        }
        for e in self.entries.values() {
            let Some(states) = read_states(world, e.entity, e.parts.len()) else {
                continue;
            };
            for (part, s) in e.parts.iter().zip(&states) {
                if !interact(part.groups) {
                    continue;
                }
                if close(
                    &part.shape,
                    part.radius,
                    WorldPos::delta(state.pos, s.pos),
                    s.quat,
                ) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Islands (hook for phase 2's co-location): see [`IslandQuery`].
    pub(crate) fn islands(&self, world: &World, q: IslandQuery) -> Vec<Island> {
        let guids: Vec<EntityGuid> = self.entries.keys().copied().collect();
        // The boxes are the broad phase: each part's bounding sphere grown by
        // the margin and by everything it can cover in the lead time. Two
        // bodies whose boxes are apart cannot meet; whether two whose boxes
        // overlap do is decided by the motion itself (`may_meet`).
        let mut boxes: Vec<Option<Aabb>> = vec![None; guids.len()];
        let mut reach: Vec<Vec<Reach>> = vec![Vec::new(); guids.len()];
        let mut dynamic = vec![false; guids.len()];
        for (i, g) in guids.iter().enumerate() {
            let e = &self.entries[g];
            let Some(states) = read_states(world, e.entity, e.parts.len()) else {
                continue;
            };
            dynamic[i] = e.any_dynamic();
            for (p, part) in e.parts.iter().enumerate() {
                let v = states[p].vel;
                let speed = f64::from((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
                grow(
                    &mut boxes[i],
                    um(states[p].pos),
                    um_extent(part.radius + q.margin + speed * q.lead_time),
                );
                reach[i].push(Reach {
                    pos: states[p].pos,
                    vel: v,
                    radius: part.radius,
                });
            }
        }
        let meet =
            |a: usize, b: usize| bodies_may_meet(&reach[a], &reach[b], 2.0 * q.margin, q.lead_time);
        let mut edges = Vec::new();
        if q.springs {
            for (i, g) in guids.iter().enumerate() {
                let e = &self.entries[g];
                if let Some(s) = world.get::<Springs>(e.entity) {
                    for sp in &s.0 {
                        if let SpringTarget::Entity { guid, .. } = sp.target {
                            if let Ok(j) = guids.binary_search(&guid) {
                                if dynamic[i] && dynamic[j] {
                                    edges.push((i, j));
                                }
                            }
                        }
                    }
                }
            }
        }
        // Only dynamic bodies couple; ghosts/kinematics are attached afterwards.
        let coupling: Vec<Option<Aabb>> = boxes
            .iter()
            .zip(&dynamic)
            .map(|(b, d)| if *d { *b } else { None })
            .collect();
        let comp = components_where(&coupling, &edges, meet);
        let mut out: BTreeMap<usize, Island> = BTreeMap::new();
        for (i, g) in guids.iter().enumerate() {
            if dynamic[i] && boxes[i].is_some() {
                let island = out.entry(comp[i]).or_insert_with(|| Island {
                    bodies: Vec::new(),
                    touching: Vec::new(),
                    bubbles: Vec::new(),
                });
                island.bodies.push(*g);
                if let Some(b) = self.entries[g].bubble {
                    if !island.bubbles.contains(&b) {
                        island.bubbles.push(b);
                    }
                }
            }
        }
        for (i, g) in guids.iter().enumerate() {
            if dynamic[i] {
                continue;
            }
            let Some((lo, hi)) = boxes[i] else { continue };
            for (j, _) in guids.iter().enumerate() {
                if !dynamic[j] {
                    continue;
                }
                let Some((blo, bhi)) = boxes[j] else { continue };
                if (0..3).all(|k| blo[k] <= hi[k] && lo[k] <= bhi[k]) && meet(i, j) {
                    if let Some(island) = out.get_mut(&comp[j]) {
                        if !island.touching.contains(g) {
                            island.touching.push(*g);
                        }
                    }
                }
            }
        }
        out.into_values()
            .map(|mut island| {
                island.bubbles.sort();
                island.touching.sort();
                island
            })
            .collect()
    }
}

/// Free flight: the velocity is the pre-integrated one; orientation by the
/// small-angle rule; position by the exact displacement of the centre of
/// mass, in the body's own cell, added to the exact position `pos + res`.
/// Returns the new state and the new residual. A ship is stepped exactly as
/// [`crate::step_ship`] steps it (no residual).
fn integrate_free(
    mut s: ShipState,
    local_com: [f64; 3],
    ship: bool,
    res: [f64; 3],
) -> (ShipState, [f64; 3]) {
    if ship {
        ship_pose_step(&mut s);
        return (s, [0.0; 3]);
    }
    let before = rotate(s.quat, local_com);
    quat_step(&mut s);
    let after = rotate(s.quat, local_com);
    let d: [f64; 3] =
        core::array::from_fn(|k| res[k] + before[k] + f64::from(s.vel[k]) * DT64 - after[k]);
    let old = s.pos;
    s.pos = WorldPos::from_f64_metres_around(old, d);
    let moved = WorldPos::delta(old, s.pos);
    (s, core::array::from_fn(|k| d[k] - moved[k]))
}

/// The states of an entity's parts: its [`ShipState`] and its [`PartStates`].
pub(crate) fn read_states(world: &World, entity: Entity, parts: usize) -> Option<Vec<ShipState>> {
    let root = *world.get::<ShipState>(entity)?;
    let mut out = Vec::with_capacity(parts);
    out.push(root);
    if parts > 1 {
        let extra = world.get::<PartStates>(entity)?;
        if extra.0.len() + 1 != parts {
            return None;
        }
        out.extend_from_slice(&extra.0);
    }
    Some(out)
}

/// Write the changed parts back (each component only when its bits changed,
/// so an unchanged component is not re-stamped needlessly).
fn write_states(world: &mut World, entity: Entity, states: &[ShipState]) {
    let root_changed = world
        .get::<ShipState>(entity)
        .is_some_and(|s| state_bits(s) != state_bits(&states[0]));
    if root_changed {
        if let Some(s) = world.get_mut::<ShipState>(entity) {
            *s = states[0];
        }
    }
    if states.len() > 1 {
        let changed = world
            .get::<PartStates>(entity)
            .is_some_and(|p| states_differ(&p.0, &states[1..]));
        if changed {
            if let Some(p) = world.get_mut::<PartStates>(entity) {
                p.0.clear();
                p.0.extend_from_slice(&states[1..]);
            }
        }
    }
}

fn remove_from(world: &mut PhysicsWorld, e: &mut Entry) {
    for h in e.handles.drain(..) {
        world.remove_body(h);
    }
    for (_, h) in e.anchors.drain(..) {
        world.remove_body(h);
    }
}

fn insert_into(
    world: &mut PhysicsWorld,
    anchor: [i64; 3],
    guid: EntityGuid,
    e: &mut Entry,
    states: &[ShipState],
) {
    let states: &[ShipState] = if states.len() == e.parts.len() {
        states
    } else {
        &e.last
    };
    for (p, part) in e.parts.iter().enumerate() {
        let s = states[p];
        let carried = e
            .carried
            .as_ref()
            .and_then(|(from, poses)| poses.get(p).copied().flatten().map(|q| (*from, q)));
        let pose = match carried {
            Some((from, mut q)) => {
                q.translation += anchor_shift(from, anchor);
                q
            }
            None => {
                if let Some(s) = e.synced.get_mut(p) {
                    *s = false;
                }
                let r = e.res.get(p).copied().unwrap_or([0.0; 3]);
                local_pose_res(anchor, s.pos, s.quat, r)
            }
        };
        let builder = if e.ghost || !part.dynamic {
            RigidBodyBuilder::kinematic_velocity_based()
        } else {
            // Never inserted asleep, whatever `e.sleeping` says: the solver
            // has no contacts yet for a body it is handed asleep, and builds
            // none for some steps after something wakes it — a resting stack
            // moved into another bubble fell 6 cm (its top crate 30 cm) into
            // the floor before it was pushed out again. Inserted awake and at
            // rest it finds its contacts in the first step and is put back to
            // sleep by the solver a second later.
            let mut b = RigidBodyBuilder::dynamic()
                .additional_mass_properties(part.mprops)
                .ccd_enabled(part.ccd)
                .sleeping(false);
            if part.lock_rotation {
                b = b.lock_rotations();
            }
            b
        }
        .pose(pose)
        .linvel(v3(f64s(s.vel)))
        .angvel(v3(rotate(s.quat, f64s(s.ang_vel))))
        .gyroscopic_forces_enabled(false)
        .linear_damping(0.0)
        .angular_damping(0.0)
        .gravity_scale(0.0);
        let h = world.insert_body(builder);
        let collider = ColliderBuilder::new(part.shape.clone())
            .density(0.0)
            .friction(f64::from(part.friction))
            .restitution(f64::from(part.restitution))
            .collision_groups(groups(part.groups))
            .user_data(entity_tag(guid, p))
            .build();
        world.insert_collider(collider, Some(h));
        e.handles.push(h);
    }
    if e.ghost {
        return;
    }
    for (j, def) in e.joints.iter().enumerate() {
        let h1 = e.handles[usize::from(def.a)];
        let h2 = match def.b {
            JointTarget::Part(p) => e.handles[usize::from(p)],
            JointTarget::World { pos, rotation } => {
                let h = world
                    .insert_body(RigidBodyBuilder::fixed().pose(local_pose(anchor, pos, rotation)));
                e.anchors.push((j, h));
                h
            }
        };
        world
            .impulse_joints
            .insert(h1, h2, joint_of(def), !e.sleeping);
    }
}
