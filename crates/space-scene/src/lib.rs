//! The shared physical definition of every server scene.
//!
//! A [`space_protocol::Scene`] is what a shard simulates and what its clients
//! draw. The *drawing* lives in `space-client`; this crate is the *physics*
//! both sides need, so a shard and a client's prediction build the same world
//! from the same data:
//!
//! * **static colliders** ([`statics`]) — the floor, the stations, the
//!   Kestrel on its apron, planets — inserted into every physics world that
//!   simulates the scene (every shard, and a client's prediction world);
//! * **bodies to spawn** ([`body_defs`]) — the demo area's physics showcase;
//! * **gravity** ([`gravity_defs`]) — zones and large bodies;
//! * the **shape catalogue** ([`catalog`]) — the meshes compiled into the
//!   binaries (the Kestrel's collider, read from its GLB by [`glb`], and the
//!   showcase's bowl);
//! * the **scene anchor** ([`scene_anchor`]) — where in the world a scene
//!   is: in the middle of a cell layer of the map's root region, a few metres
//!   off the corner where its cells meet ([`ANCHOR_OFFSET`]), so that no
//!   horizontal shard border comes near it and the vertical ones cut it into
//!   four shard areas beside the spawn point, not through it.
//!
//! Everything is given **relative to the scene anchor** (scene space: the
//! anchor is the spawn point, +Y up, a fresh player looks down -Z — the same
//! space `space-client`'s scene builders use) and placed in the world with
//! [`environment`] / [`bodies`]: statics as exact `WorldPoint`s (`f64`
//! offsets from the anchor), bodies as `WorldPos`.
//!
//! # How a shard builds its world (phase 2)
//!
//! ```ignore
//! let anchor = space_scene::scene_anchor(&shard_map).unwrap();
//! let mut sim = SimWorld::new(shard);
//! space_scene::install_environment(&mut sim, scene, anchor)?;     // catalogue, statics, gravity
//! let guids = space_scene::spawn_bodies(&mut sim, scene, anchor, &mut || mint_guid())?;
//! ```
//!
//! On a scene change: `install_environment` again (statics and gravity are
//! replaced), despawn the old scene's bodies and spawn the new ones. Which
//! shard spawns a scene's bodies is phase 2's call (the shard whose region
//! holds each body).
//!
//! | scene | statics | bodies | gravity |
//! |---|---|---|---|
//! | `demo-area` | the stations (ported with the client's numbers), the Kestrel and its apron, the showcase's curbs, bowl, ramp, fulcrum, gantry, door post, the planetoid | the showcase ([`demo`]), crates resting on the planetoid and balls orbiting it | the gravity room; the planetoid (4 m/s² on its surface, a reach of 16 m) |
//! | `materials` | the chart's 21 spheres, the glass and lamp spheres, the floor slab | none | none (a chart to fly around) |
//! | `solar` | the Sun, the eight planets and the Moon as balls, where the client draws them | none | the same ten bodies as point masses (real `GM`) |
//! | `galaxy` | the spawn cube | none | none (stars are light-years apart) |

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod demo;
pub mod glb;
pub mod others;

use std::sync::{Arc, OnceLock};

use space_protocol::{EntityGuid, Scene, ShardMap};
use space_sim::articulation::{Articulation, JointTarget, PartStates};
use space_sim::body::{Body, CollisionGroups, Visual};
use space_sim::gravity::{BodyPlace, GravityField, GravityZone, LargeBody, WorldPoint, ZoneShape};
use space_sim::rigid::{PhysicsError, StaticCollider};
use space_sim::shape::{AssetId, MeshGeometry, ShapeCatalog, ShapeRef};
use space_sim::spring::{Spring, SpringTarget, Springs};
use space_sim::{BodySpawn, ShipState, SimWorld};
use space_world::{NodeRef, WorldPos};

/// Catalogue id of the Kestrel's collider (its LOD1 exterior mesh).
pub const KESTREL: AssetId = AssetId(1);
/// Catalogue id of the showcase's concave bowl.
pub const BOWL: AssetId = AssetId(2);

/// The Kestrel's low-detail exterior proxy (11,585 triangles), the source of
/// its collider. The same file `space-client` draws far away.
pub static KESTREL_LOD1_GLB: &[u8] =
    include_bytes!("../../../assets/ships/kestrel/kestrel_lod1.glb");

/// The ids a [`Visual`] carries: which model a client draws for an entity or a
/// static (its size comes from its shape). Phase 3 maps them to meshes.
pub mod visual {
    /// Nothing to draw (or: the scene's own visuals already show it).
    pub const NONE: u32 = 0;
    /// A crate (box).
    pub const CRATE: u32 = 1;
    /// A ball.
    pub const BALL: u32 = 2;
    /// The seesaw's plank.
    pub const PLANK: u32 = 3;
    /// A door.
    pub const DOOR: u32 = 4;
    /// A link of a chain (capsule).
    pub const CHAIN_LINK: u32 = 5;
    /// The bob of a spring pendulum.
    pub const PENDULUM_BOB: u32 = 6;
    /// A player (the placeholder character).
    pub const PLAYER: u32 = 7;
    /// The IC-4 Kestrel.
    pub const KESTREL: u32 = 8;
    /// The concave bowl (catalogue mesh [`crate::BOWL`]).
    pub const BOWL: u32 = 9;
    /// The ramp.
    pub const RAMP: u32 = 10;
    /// The gantry (posts and beam).
    pub const GANTRY: u32 = 11;
    /// A low curb.
    pub const CURB: u32 = 12;
    /// The seesaw's fulcrum.
    pub const FULCRUM: u32 = 13;
    /// The door's post.
    pub const DOOR_POST: u32 = 14;
    /// The demo area's planetoid (a static ball that pulls,
    /// [`crate::demo::PLANETOID_CENTER`]).
    pub const PLANETOID: u32 = 15;

    /// The one-word name of the model `id` names, for a heads-up display, a
    /// log or a script (`"crate"`, `"pendulum"`, ...); `"unknown"` for an id
    /// this build does not know.
    #[must_use]
    pub const fn name(id: u32) -> &'static str {
        match id {
            NONE => "none",
            CRATE => "crate",
            BALL => "ball",
            PLANK => "plank",
            DOOR => "door",
            CHAIN_LINK => "chain",
            PENDULUM_BOB => "pendulum",
            PLAYER => "player",
            KESTREL => "kestrel",
            BOWL => "bowl",
            RAMP => "ramp",
            GANTRY => "gantry",
            CURB => "curb",
            FULCRUM => "fulcrum",
            DOOR_POST => "door post",
            PLANETOID => "planetoid",
            _ => "unknown",
        }
    }
}

/// A static collider of a scene, anchor-relative.
#[derive(Clone, Debug, PartialEq)]
pub struct StaticDef {
    /// What it is (tests, logs).
    pub name: &'static str,
    /// Its shape.
    pub shape: ShapeRef,
    /// Its origin, metres from the anchor.
    pub offset: [f64; 3],
    /// Its orientation `(w, x, y, z)`.
    pub rotation: [f32; 4],
    /// Friction.
    pub friction: f32,
    /// Restitution.
    pub restitution: f32,
    /// What a client draws ([`visual::NONE`]: the scene's visuals cover it).
    pub visual: Visual,
    /// Which bodies it stops.
    pub groups: CollisionGroups,
}

/// Where the far end of a joint or spring of a [`BodyDef`] is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Anchor {
    /// A fixed point, metres from the scene anchor.
    World([f64; 3]),
    /// A part of the same entity (the joint's own `b`).
    Part,
    /// Another body of the scene, by [`BodyDef::name`], at this attachment
    /// in its body frame.
    Body(&'static str, [f32; 3]),
}

/// A body a scene spawns, anchor-relative.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyDef {
    /// What it is (unique within a scene when another body names it).
    pub name: &'static str,
    /// Its origin, metres from the anchor.
    pub offset: [f64; 3],
    /// Orientation `(w, x, y, z)`.
    pub quat: [f32; 4],
    /// Initial velocity, m/s.
    pub vel: [f32; 3],
    /// Initial angular velocity, rad/s, body frame.
    pub ang_vel: [f32; 3],
    /// What it is physically.
    pub body: Body,
    /// What a client draws.
    pub visual: Visual,
    /// Its springs, each with where its far end is (the spring's own target
    /// is filled in when the scene is placed).
    pub springs: Vec<(Spring, Anchor)>,
    /// Extra parts and joints: the articulation (world targets filled in
    /// when placed), one [`Anchor`] per joint, and the extra parts' initial
    /// poses (offset from the scene anchor, orientation).
    pub articulation: Option<ArticulationDef>,
}

/// The articulation of a [`BodyDef`]: the joints (world targets filled in
/// when placed), one [`Anchor`] per joint, and each extra part's initial pose
/// (offset from the scene anchor, orientation).
pub type ArticulationDef = (Articulation, Vec<Anchor>, Vec<([f64; 3], [f32; 4])>);

/// A source of gravity of a scene, anchor-relative.
#[derive(Clone, Debug, PartialEq)]
pub enum GravityDef {
    /// An axis-aligned box of constant gravity.
    Zone {
        /// Centre, metres from the anchor.
        center: [f64; 3],
        /// Half extents, metres.
        half_extents: [f64; 3],
        /// Acceleration inside, m/s².
        gravity: [f64; 3],
        /// Soft edge, metres.
        soft_edge: f64,
    },
    /// A point mass at a fixed place.
    Body {
        /// Its sphere of influence, metres from its centre: the pull fades
        /// out over the outer quarter and is exactly zero beyond
        /// ([`space_sim::LargeBody::reach`]); `f64::INFINITY` for none.
        reach: f64,
        /// Name.
        name: &'static str,
        /// `GM`, m³/s².
        gm: f64,
        /// Radius, metres.
        radius: f64,
        /// Centre, metres from the anchor.
        center: [f64; 3],
    },
}

/// Where a scene's spawn point lies from the **centre corner** of a root
/// region of more than one cell, metres: 6 m along +X, half a cell (512 m)
/// up, 4 m along +Z ([`anchor_in`]).
///
/// The centre of a node of two or more cells is the corner where its eight
/// child octants — and eight cells — meet, and a cell is the smallest shard
/// area there is. A scene placed *on* that corner (the rule until the demo
/// area was cut into four shard areas) has a possible shard border on each
/// of the three planes through its spawn point: `x = 0`, `z = 0`, and — worst
/// — `y = 0`, eye height, a horizontal border under everything that floats
/// and over everything that stands. The offset moves the scene to where
/// borders do the least harm:
///
/// * **512 m up**: the scene's height 0 is the middle of a cell layer, so no
///   horizontal border lies within 512 m of it. Nothing in a scene the size
///   of the demo area ever crosses one;
/// * **6 m and 4 m sideways**: the two vertical border planes pass through
///   the scene at `x = −6` and `z = −4` instead of through the spawn point.
///   The spawn point is strictly inside one cell (with `k` the corner's cell
///   index: `[k, k, k]`), and the four cells that meet at the line
///   `x = −6, z = −4` are that one, `[k, k, k−1]` (z < −4), `[k−1, k, k]`
///   (x < −6) and `[k−1, k, k−1]`: the four shard areas of a scene, each with
///   a piece of it.
///
/// All three are whole metres, so the anchor is exact in a position's `f32`
/// cell offset.
pub const ANCHOR_OFFSET: [f64; 3] = [6.0, 512.0, 4.0];

/// Where a scene's spawn point lies in a root region `node`.
///
/// * A root of **one cell** (depth 64 — a standalone shard, the test
///   convention): the cell's centre. Its borders are the cell's faces, 512 m
///   away on every axis; there is nothing to stay clear of.
/// * A root of **more than one cell**: its centre corner plus
///   [`ANCHOR_OFFSET`] — see there for why not the corner itself.
#[must_use]
pub fn anchor_in(node: NodeRef) -> WorldPos {
    let centre = node.center();
    if node.depth >= space_world::MAX_DEPTH {
        centre
    } else {
        WorldPos::from_f64_metres_around(centre, ANCHOR_OFFSET)
    }
}

/// Where a scene lies in the world of `map` — its **anchor**, the scene's
/// spawn point: [`anchor_in`] the map's **shallowest** region (the root),
/// ties broken by the lowest shard id, then the lowest node index. `None` for
/// a map without regions.
///
/// Shards place their scene's statics and bodies by this point and
/// `space-client` draws by it (`link::map_anchor` is this function), so what
/// a shard simulates and what a client draws are in the same place. Carves
/// only add deeper regions and merges only remove them, so the root and with
/// it this point never move while a cluster lives.
#[must_use]
pub fn scene_anchor(map: &ShardMap) -> Option<WorldPos> {
    map.regions
        .iter()
        .min_by_key(|region| (region.node.depth, region.shard.0, region.node.index))
        .map(|region| anchor_in(region.node))
}

/// The concave bowl of the showcase: the inside of a sphere cap of radius
/// [`demo::BOWL_RADIUS`], [`demo::BOWL_DEPTH`] deep, its lowest point at the
/// origin, 12 rings × 32 segments, wound so its inside faces up.
#[must_use]
pub fn bowl_mesh() -> MeshGeometry {
    let (r, depth, rings, segments) = (demo::BOWL_RADIUS, demo::BOWL_DEPTH, 12u32, 32u32);
    let max_angle = ((r - depth) / r).acos();
    let mut vertices = vec![[0.0, 0.0, 0.0]];
    for i in 1..=rings {
        let a = max_angle * f64::from(i) / f64::from(rings);
        for j in 0..segments {
            let phi = std::f64::consts::TAU * f64::from(j) / f64::from(segments);
            vertices.push([
                r * a.sin() * phi.cos(),
                r - r * a.cos(),
                r * a.sin() * phi.sin(),
            ]);
        }
    }
    let ring = |i: u32, j: u32| 1 + (i - 1) * segments + (j % segments);
    let mut indices = Vec::new();
    for j in 0..segments {
        indices.push([0, ring(1, j + 1), ring(1, j)]);
    }
    for i in 1..rings {
        for j in 0..segments {
            let (a, b, c, d) = (
                ring(i, j),
                ring(i, j + 1),
                ring(i + 1, j),
                ring(i + 1, j + 1),
            );
            indices.push([a, b, d]);
            indices.push([a, d, c]);
        }
    }
    MeshGeometry { vertices, indices }
}

/// The shape catalogue every world simulating a scene uses: [`KESTREL`]
/// (read once from [`KESTREL_LOD1_GLB`]) and [`BOWL`]. Built on first use,
/// one per process.
///
/// # Panics
/// If the compiled-in Kestrel GLB cannot be read (a broken build; the tests
/// read it).
#[must_use]
pub fn catalog() -> Arc<ShapeCatalog> {
    static CATALOG: OnceLock<Arc<ShapeCatalog>> = OnceLock::new();
    Arc::clone(CATALOG.get_or_init(|| {
        let mut c = ShapeCatalog::new();
        let kestrel = glb::read_glb(KESTREL_LOD1_GLB).expect("the compiled-in Kestrel GLB reads");
        c.insert(KESTREL, "kestrel_lod1", kestrel)
            .expect("the Kestrel mesh is valid");
        c.insert(BOWL, "bowl", bowl_mesh())
            .expect("the bowl mesh is valid");
        Arc::new(c)
    }))
}

/// A scene's static colliders, anchor-relative.
#[must_use]
pub fn statics(scene: Scene) -> Vec<StaticDef> {
    match scene {
        Scene::DemoArea => demo::statics(),
        Scene::Materials => others::materials_statics(),
        Scene::Solar => others::solar_statics(),
        Scene::Galaxy => others::galaxy_statics(),
    }
}

/// The bodies a scene spawns, anchor-relative.
#[must_use]
pub fn body_defs(scene: Scene) -> Vec<BodyDef> {
    match scene {
        Scene::DemoArea => demo::bodies(),
        Scene::Materials | Scene::Solar | Scene::Galaxy => Vec::new(),
    }
}

/// A scene's gravity, anchor-relative.
#[must_use]
pub fn gravity_defs(scene: Scene) -> Vec<GravityDef> {
    match scene {
        Scene::DemoArea => demo::gravity(),
        Scene::Solar => others::solar_gravity(),
        Scene::Materials | Scene::Galaxy => Vec::new(),
    }
}

/// What every world simulating `scene` at `anchor` needs besides its bodies.
#[derive(Clone, Debug, PartialEq)]
pub struct Environment {
    /// The static colliders, in world coordinates.
    pub statics: Vec<StaticCollider>,
    /// The gravity field.
    pub gravity: GravityField,
}

/// `scene`'s statics and gravity placed at `anchor`.
#[must_use]
pub fn environment(scene: Scene, anchor: WorldPos) -> Environment {
    let statics = statics(scene)
        .into_iter()
        .map(|s| StaticCollider {
            shape: s.shape,
            place: WorldPoint::new(anchor, s.offset),
            rotation: s.rotation,
            friction: s.friction,
            restitution: s.restitution,
            visual: s.visual,
            groups: s.groups,
        })
        .collect();
    let mut gravity = GravityField::none();
    for g in gravity_defs(scene) {
        match g {
            GravityDef::Zone {
                center,
                half_extents,
                gravity: a,
                soft_edge,
            } => gravity.zones.push(GravityZone {
                center: WorldPoint::new(anchor, center),
                rotation: [1.0, 0.0, 0.0, 0.0],
                shape: ZoneShape::Box { half_extents },
                gravity: a,
                soft_edge,
            }),
            GravityDef::Body {
                reach,
                name,
                gm,
                radius,
                center,
            } => gravity.bodies.push(LargeBody {
                name: name.into(),
                gm,
                radius,
                place: BodyPlace::Fixed(WorldPoint::new(anchor, center)),
                reach,
            }),
        }
    }
    Environment { statics, gravity }
}

/// Give `sim` the catalogue, statics and gravity of `scene` at `anchor`
/// (replacing what it had).
///
/// # Errors
/// [`PhysicsError`] if a static cannot be built (a broken scene definition;
/// the tests build every one).
pub fn install_environment(
    sim: &mut SimWorld,
    scene: Scene,
    anchor: WorldPos,
) -> Result<(), PhysicsError> {
    let env = environment(scene, anchor);
    sim.set_catalog(catalog());
    sim.set_statics(env.statics)?;
    sim.set_gravity(env.gravity);
    Ok(())
}

/// A body of a scene, placed in the world with its guid.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneBody {
    /// Its [`BodyDef::name`].
    pub name: &'static str,
    /// The guid it was given.
    pub guid: EntityGuid,
    /// What to spawn.
    pub spawn: BodySpawn,
}

/// `scene`'s bodies placed at `anchor`, each given a guid by `mint` (in
/// definition order, so springs between them can name each other).
#[must_use]
pub fn bodies(
    scene: Scene,
    anchor: WorldPos,
    mint: &mut dyn FnMut() -> EntityGuid,
) -> Vec<SceneBody> {
    let defs = body_defs(scene);
    let guids: Vec<EntityGuid> = defs.iter().map(|_| mint()).collect();
    let guid_of = |name: &str| defs.iter().position(|d| d.name == name).map(|i| guids[i]);
    let place = |offset: [f64; 3]| WorldPos::from_f64_metres_around(anchor, offset);
    defs.iter()
        .zip(&guids)
        .map(|(d, guid)| {
            let mut state = ShipState::at_rest(place(d.offset));
            state.quat = d.quat;
            state.vel = d.vel;
            state.ang_vel = d.ang_vel;
            let mut spawn = BodySpawn::new(state, d.body.clone()).with_visual(d.visual);
            if !d.springs.is_empty() {
                let springs = d
                    .springs
                    .iter()
                    .filter_map(|(s, a)| {
                        let target = match *a {
                            Anchor::World(p) => SpringTarget::World(place(p)),
                            Anchor::Body(name, anchor_b) => SpringTarget::Entity {
                                guid: guid_of(name)?,
                                part: 0,
                                anchor: anchor_b,
                            },
                            Anchor::Part => return None,
                        };
                        Some(Spring { target, ..*s })
                    })
                    .collect();
                spawn = spawn.with_springs(Springs(springs));
            }
            if let Some((art, anchors, parts)) = &d.articulation {
                let mut art = art.clone();
                for (joint, a) in art.joints.iter_mut().zip(anchors) {
                    if let (Anchor::World(p), JointTarget::World { rotation, .. }) = (*a, joint.b) {
                        joint.b = JointTarget::World {
                            pos: place(p),
                            rotation,
                        };
                    }
                }
                let states = parts
                    .iter()
                    .map(|(offset, quat)| ShipState {
                        quat: *quat,
                        ..ShipState::at_rest(place(*offset))
                    })
                    .collect();
                spawn = spawn.with_articulation(art, PartStates(states));
            }
            SceneBody {
                name: d.name,
                guid: *guid,
                spawn,
            }
        })
        .collect()
}

/// Spawn `scene`'s bodies into `sim` (see [`bodies`]); returns their guids.
///
/// # Errors
/// The first [`PhysicsError`] (a broken scene definition; the tests spawn
/// every one).
pub fn spawn_bodies(
    sim: &mut SimWorld,
    scene: Scene,
    anchor: WorldPos,
    mint: &mut dyn FnMut() -> EntityGuid,
) -> Result<Vec<EntityGuid>, PhysicsError> {
    let mut out = Vec::new();
    for b in bodies(scene, anchor, mint) {
        sim.spawn_body(b.guid, b.spawn)?;
        out.push(b.guid);
    }
    Ok(out)
}
