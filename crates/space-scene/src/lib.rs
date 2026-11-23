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
//!   is.
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
//! | `demo-area` | the stations (ported with the client's numbers), the Kestrel and its apron, the showcase's curbs, bowl, ramp, fulcrum, gantry, door post, the planetoid; Earth, the Moon, the Sun and the planets as balls moving on their orbits ([`solar`]) | the showcase ([`demo`]), crates resting on the planetoid and balls orbiting it | the gravity room; the planetoid (4 m/s² on its surface, a reach of 16 m); the solar system's ten bodies (real `GM`, each with a finite reach that ends far from the demo area) |
//! | `materials` | the chart's 21 spheres, the glass and lamp spheres, the floor slab | none | none (a chart to fly around) |
//! | `galaxy` | the spawn cube | none | none (stars are light-years apart) |
//!
//! (Scene id 2, `solar`, is retired: the solar system is part of the demo
//! area now.)

#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]

pub mod demo;
pub mod glb;
pub mod others;
pub mod solar;

use std::sync::{Arc, OnceLock};

use space_protocol::{EntityGuid, Scene, ShardMap};
use space_sim::articulation::{Articulation, JointTarget, PartStates};
use space_sim::body::{Body, CollisionGroups, Visual};
use space_sim::gravity::{BodyPlace, GravityField, GravityZone, LargeBody, WorldPoint, ZoneShape};
use space_sim::motion::OrbitPath;
use space_sim::rigid::{PhysicsError, StaticCollider};
use space_sim::shape::{AssetId, MeshGeometry, ShapeCatalog, ShapeRef};
use space_sim::spring::{Spring, SpringTarget, Springs};
use space_sim::{BodySpawn, ShipState, SimWorld};
use space_world::WorldPos;

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
    /// `Some`: it moves along this path (offset from `offset`) with scene
    /// time — a planet's collider ([`space_sim::StaticCollider::motion`]).
    pub motion: Option<OrbitPath>,
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
    /// A point mass on a fixed orbit of scene time ([`BodyPlace::Orbit`]).
    Orbit {
        /// Its reach, metres (as for [`GravityDef::Body`]).
        reach: f64,
        /// Name.
        name: &'static str,
        /// `GM`, m³/s².
        gm: f64,
        /// Radius, metres.
        radius: f64,
        /// The point the path is relative to, metres from the anchor.
        base: [f64; 3],
        /// The path.
        path: OrbitPath,
    },
}

/// Where a scene lies in the world of `map`: the centre of the map's
/// **shallowest** region, ties broken by the lowest shard id, then the lowest
/// node index. `None` for a map without regions.
///
/// This is the rule `space-client`'s `link::map_anchor` implements (splits
/// only add deeper regions and merges only remove them, so the root and with
/// it this point never move); the client calls this function from phase 3 on.
#[must_use]
pub fn scene_anchor(map: &ShardMap) -> Option<WorldPos> {
    map.regions
        .iter()
        .min_by_key(|region| (region.node.depth, region.shard.0, region.node.index))
        .map(|region| region.node.center())
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
        Scene::DemoArea => {
            let mut s = demo::statics();
            s.extend(others::sky_statics());
            s
        }
        Scene::Materials => others::materials_statics(),
        Scene::Galaxy => others::galaxy_statics(),
    }
}

/// The bodies a scene spawns, anchor-relative.
#[must_use]
pub fn body_defs(scene: Scene) -> Vec<BodyDef> {
    match scene {
        Scene::DemoArea => demo::bodies(),
        Scene::Materials | Scene::Galaxy => Vec::new(),
    }
}

/// A scene's gravity, anchor-relative.
#[must_use]
pub fn gravity_defs(scene: Scene) -> Vec<GravityDef> {
    match scene {
        Scene::DemoArea => {
            let mut g = demo::gravity();
            g.extend(others::sky_gravity());
            g
        }
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
            motion: s.motion,
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
            GravityDef::Orbit {
                reach,
                name,
                gm,
                radius,
                base,
                path,
            } => gravity.bodies.push(LargeBody {
                name: name.into(),
                gm,
                radius,
                place: BodyPlace::Orbit {
                    base: WorldPoint::new(anchor, base),
                    path,
                },
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
