//! The worlds a shard can show, as a function of [`space_protocol::Scene`].
//!
//! Which scene is drawn is the server's decision ([`crate::link`], "The
//! scene"); this module is what the client draws for each of them. A
//! [`World`] is one scene's content in **scene space**: fixed positions
//! ([`WorldPos`], so a body 4.5e12 m from the origin is placed as exactly as
//! one next to it) around a spawn point ([`World::spawn`]). The application
//! maps the server's world onto it by putting the place a ship first appeared
//! on the spawn point, so every player who joined through the same shard sees
//! the scene in the same place.
//!
//! | scene | content |
//! |---|---|
//! | `demo-area` | [`crate::demo_area`]: everything the renderer can do, around the spawn point |
//! | `materials` | a PBR material chart ([`materials_bodies`]) |
//! | `solar` | the solar system at real size ([`solar_bodies`]) |
//! | `galaxy` | the deterministic galaxy through the LOD selector |

use std::sync::{Arc, OnceLock};

use space_render::{
    CameraPose, Environment, GpuInstance, MeshData, ModelInstance, Quality, QualityPreset,
    RenderOptions, Scene, SceneLight, SHAPE_BOX, SHAPE_SPHERE,
};
use space_world::{Galaxy, LodSelector, NodeRef, WorldPos, MAX_SELECTED};

use crate::demo_area::DemoArea;
use crate::kestrel::Detail;

/// The catalogue of scenes ([`space_protocol::Scene`]), for programs that use
/// this module without depending on the protocol crate.
pub use space_protocol::Scene as SceneKind;

/// How far above the ground a player's eyes are, metres: a ship's position is
/// its pilot's eye point, and the placeholder character stands this far below
/// it. The floors of the scenes that have one are this far below the spawn
/// point.
pub const EYE_HEIGHT: f32 = 1.7;

/// The placeholder player character (one shared mesh: the renderer's model
/// cache is keyed by the `Arc`).
#[must_use]
pub fn character_mesh() -> Arc<MeshData> {
    static MESH: OnceLock<Arc<MeshData>> = OnceLock::new();
    Arc::clone(MESH.get_or_init(|| Arc::new(space_render::sample_character())))
}

/// The ship model drawn for a ship nobody flies (one shared mesh).
#[must_use]
pub fn ship_mesh() -> Arc<MeshData> {
    static MESH: OnceLock<Arc<MeshData>> = OnceLock::new();
    Arc::clone(MESH.get_or_init(|| Arc::new(space_render::sample_ship())))
}

/// The vertical field of view every scene is seen with: 50 degrees.
///
/// A rectilinear picture stretches whatever is off its axis, away from the
/// centre, by `1 / cos` of the angle — and a sphere shows it at once. At 16:9
/// the old 60 degrees put the frame's edge 46 degrees and its corner 50
/// degrees off axis (spheres 1.43x and 1.54x as long as wide); 50 degrees
/// brings that to 40 and 44 degrees (1.30x and 1.38x), about a third less
/// stretch everywhere in the picture, while the whole demo area still fits
/// into the view from its spawn point. Narrower still (45 degrees) starts to
/// cut the outer stations off and makes walking among them feel zoomed in.
pub const FOV_Y: f32 = 50.0 * std::f32::consts::PI / 180.0;

/// The faint, even light of the scenes that are in space (the solar system,
/// the galaxy): no sky is painted — space is black — but whoever stands
/// there, or flies away from every lamp, is a dim shape and not a black
/// cut-out, and polished metal has something to reflect.
pub const SPACE_AMBIENT: Environment = Environment::gradient([0.05; 3], [0.04; 3], [0.03; 3]);

/// The surroundings of the material chart: a colourless light tent, painted
/// as the background so that the mirror floor and the polished spheres
/// reflect what is actually there to see. It is bright overhead and a good
/// deal darker at the horizon (darker than [`Environment::studio`], whose
/// even grey makes the mirror floor vanish into the backdrop): the floor has
/// a horizon to end at, a polished sphere has a bright cap and a dark band to
/// show its shape with, and the chart's three lamps still model the rest.
pub const CHART_ENVIRONMENT: Environment =
    Environment::gradient([0.6; 3], [0.14; 3], [0.04; 3]).with_background(true);

/// A world position `metres` from the world origin.
#[must_use]
pub fn at(metres: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(WorldPos::ZERO, metres)
}

/// `position` as metres from the world origin (exact to well under a
/// millimetre anywhere a ship can be flown to).
#[must_use]
pub fn metres(position: WorldPos) -> [f64; 3] {
    WorldPos::delta(WorldPos::ZERO, position)
}

/// Where `position` is as seen from `camera`, metres.
#[must_use]
pub fn relative(camera: &CameraPose, position: WorldPos) -> [f32; 3] {
    WorldPos::delta(camera.position, position).map(|v| v as f32)
}

/// A mesh model at a fixed place in a scene.
#[derive(Clone, Debug)]
pub struct Prop {
    /// The model (shared: see [`character_mesh`]).
    pub mesh: Arc<MeshData>,
    /// Where the model's origin is.
    pub position: WorldPos,
    /// Orientation: the rows of a rotation matrix.
    pub rotation: [[f32; 3]; 3],
    /// Metres per model unit.
    pub scale: f32,
}

impl Prop {
    /// The render instance of this prop as seen from `camera`.
    #[must_use]
    pub fn instance(&self, camera: &CameraPose) -> ModelInstance {
        ModelInstance {
            mesh: Arc::clone(&self.mesh),
            position: relative(camera, self.position),
            rotation: self.rotation,
            scale: self.scale,
        }
    }
}

/// One sphere or box of a world-space scene.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    /// What it is (for the title bar and tests).
    pub name: &'static str,
    /// Where its centre is.
    pub position: WorldPos,
    /// Sphere radius, or a box's half-extent, in metres.
    pub radius: f32,
    /// [`SHAPE_SPHERE`] or [`SHAPE_BOX`].
    pub shape: u32,
    /// Base colour (linear rgb).
    pub albedo: [f32; 3],
    /// 0 = dielectric, 1 = metal.
    pub metallic: f32,
    /// 0 = mirror smooth, 1 = fully rough.
    pub roughness: f32,
    /// Emitted radiance (linear rgb).
    pub emissive: [f32; 3],
    /// 0 = opaque, 1 = fully transmissive (glass; needs a refraction preset).
    pub transmission: f32,
}

impl Body {
    /// An opaque, non-metallic, non-emissive sphere.
    #[must_use]
    pub fn sphere(name: &'static str, position: WorldPos, radius: f32, albedo: [f32; 3]) -> Self {
        Self {
            name,
            position,
            radius,
            shape: SHAPE_SPHERE,
            albedo,
            metallic: 0.0,
            roughness: 0.7,
            emissive: [0.0; 3],
            transmission: 0.0,
        }
    }

    /// An opaque, non-metallic, non-emissive cube, `half` metres from its
    /// centre to each face.
    #[must_use]
    pub fn cube(name: &'static str, position: WorldPos, half: f32, albedo: [f32; 3]) -> Self {
        Self {
            shape: SHAPE_BOX,
            ..Self::sphere(name, position, half, albedo)
        }
    }

    /// The render instance of this body as seen from `camera`.
    #[must_use]
    pub fn instance(&self, camera: &CameraPose) -> GpuInstance {
        let d = relative(camera, self.position);
        GpuInstance {
            center: [d[0], d[1], d[2], self.radius],
            albedo: [
                self.albedo[0],
                self.albedo[1],
                self.albedo[2],
                self.metallic,
            ],
            emissive: [self.emissive[0], self.emissive[1], self.emissive[2], 1.0],
            roughness: self.roughness,
            shape: self.shape,
            transmission: self.transmission,
            _pad1: 0.0,
        }
    }
}

/// The camera-relative render scene of `bodies` seen from `camera`.
#[must_use]
pub fn scene_of(camera: &CameraPose, bodies: &[Body], lights: Vec<SceneLight>) -> Scene {
    let instances = bodies.iter().map(|body| body.instance(camera)).collect();
    Scene::from_instances(*camera, instances).with_lights(lights)
}

/// The body whose surface is nearest to `position`, with that distance in
/// metres (negative inside the body).
#[must_use]
pub fn nearest(bodies: &[Body], position: WorldPos) -> Option<(&Body, f64)> {
    bodies
        .iter()
        .map(|body| {
            let centre = WorldPos::distance(position, body.position);
            (body, centre - f64::from(body.radius))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// A length for a heads-up line: metres, kilometres or astronomical units,
/// whichever reads best.
#[must_use]
pub fn format_distance(metres: f64) -> String {
    let m = metres.abs();
    let sign = if metres < 0.0 { "-" } else { "" };
    if m < 1.0e3 {
        format!("{sign}{m:.0} m")
    } else if m < 1.0e9 {
        format!("{sign}{:.0} km", m / 1.0e3)
    } else {
        format!("{sign}{:.3} AU", m / AU)
    }
}

// ---- the worlds -----------------------------------------------------------------

/// What a camera in a scene sees with: the scale the scene is built at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lens {
    /// Vertical field of view, radians.
    pub fov_y: f32,
    /// Near plane, metres.
    pub near: f32,
    /// Far distance, metres (the projection's far plane is infinite; this
    /// bounds what the scene builders consider).
    pub far: f32,
}

/// One scene's content. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct World {
    scene: space_protocol::Scene,
    /// What draws it: decides how soon the costly models are shown in full.
    detail: Detail,
}

impl World {
    /// The world of `scene`, as a GPU draws it.
    #[must_use]
    pub const fn new(scene: space_protocol::Scene) -> Self {
        Self {
            scene,
            detail: Detail::Hardware,
        }
    }

    /// The same world, built for a renderer of this `detail` (a front end
    /// passes [`Detail::for_device`] of its render context: a software
    /// rasteriser is shown the Kestrel's proxy for longer).
    #[must_use]
    pub const fn with_detail(mut self, detail: Detail) -> Self {
        self.detail = detail;
        self
    }

    /// What this world is built for.
    #[must_use]
    pub const fn detail(&self) -> Detail {
        self.detail
    }

    /// Which scene this is.
    #[must_use]
    pub const fn scene(&self) -> space_protocol::Scene {
        self.scene
    }

    /// Where a ship that has not moved is: the point of the scene that the
    /// place a ship first appeared is mapped to.
    #[must_use]
    pub fn spawn(&self) -> WorldPos {
        match self.scene {
            space_protocol::Scene::DemoArea | space_protocol::Scene::Materials => WorldPos::ZERO,
            space_protocol::Scene::Solar => solar_spawn(),
            space_protocol::Scene::Galaxy => at(GALAXY_SPAWN_M),
        }
    }

    /// The view a player starts with: yaw and pitch in radians. (The galaxy's
    /// spawn point sits at -Z looking back at the galaxy's centre, half a turn
    /// from yaw 0.) Always level: thrust follows the view, and a player who
    /// walks straight ahead should stay on the floor.
    #[must_use]
    pub fn initial_view(&self) -> (f32, f32) {
        match self.scene {
            space_protocol::Scene::Galaxy => (std::f32::consts::PI, 0.0),
            space_protocol::Scene::DemoArea
            | space_protocol::Scene::Materials
            | space_protocol::Scene::Solar => (0.0, 0.0),
        }
    }

    /// The camera's field of view and depth range for this scene.
    #[must_use]
    pub fn lens(&self) -> Lens {
        let (near, far) = match self.scene {
            space_protocol::Scene::DemoArea => (0.05, 1.0e5),
            space_protocol::Scene::Materials => (0.05, 1.0e3),
            space_protocol::Scene::Solar => (1.0, 1.0e13),
            space_protocol::Scene::Galaxy => (1.0, 5.0e6),
        };
        Lens {
            fov_y: FOV_Y,
            near,
            far,
        }
    }

    /// A camera at `position` looking along `forward`, with this scene's lens.
    #[must_use]
    pub fn camera(&self, position: WorldPos, forward: [f32; 3], aspect: f32) -> CameraPose {
        let lens = self.lens();
        CameraPose {
            position,
            forward,
            up: [0.0, 1.0, 0.0],
            fov_y: lens.fov_y,
            aspect: if aspect.is_finite() && aspect > 0.0 {
                aspect
            } else {
                1.0
            },
            near: lens.near,
            far: lens.far,
        }
    }

    /// The height (scene-space y, metres) of the ground around the spawn
    /// point, for the scenes that have one: the chase camera stays above it.
    #[must_use]
    pub fn floor(&self) -> Option<f64> {
        match self.scene {
            space_protocol::Scene::DemoArea | space_protocol::Scene::Materials => {
                Some(-f64::from(EYE_HEIGHT))
            }
            space_protocol::Scene::Solar | space_protocol::Scene::Galaxy => None,
        }
    }

    /// The scene's surroundings: its ambient light, what its polished metal
    /// reflects and its glass falls back to, and — for the scenes that are
    /// not in space — the sky that is painted behind everything.
    #[must_use]
    pub fn environment(&self) -> Environment {
        match self.scene {
            space_protocol::Scene::DemoArea => crate::demo_area::ENVIRONMENT,
            space_protocol::Scene::Materials => CHART_ENVIRONMENT,
            space_protocol::Scene::Solar | space_protocol::Scene::Galaxy => SPACE_AMBIENT,
        }
    }

    /// The scene's content as `camera` sees it: instances, models, lights
    /// and environment, camera-relative, with an empty overlay.
    #[must_use]
    pub fn build(&self, camera: &CameraPose) -> Scene {
        let scene = match self.scene {
            space_protocol::Scene::DemoArea => DemoArea::shared().scene_for(camera, self.detail),
            space_protocol::Scene::Materials => materials_scene(camera),
            space_protocol::Scene::Solar => solar_scene(camera),
            space_protocol::Scene::Galaxy => galaxy_scene(camera),
        };
        scene.with_environment(self.environment())
    }

    /// The render options every front end uses for this scene, so the same
    /// preset gates the same features — including the SSR/refraction post
    /// pass — whichever way the frame leaves the GPU. The scenes at
    /// astronomical scale turn the presets' distance fog off (it is tuned for
    /// metres, and would paint every star and planet solid fog colour); the
    /// demo area halves it, so its backdrop fades over a kilometre rather
    /// than half of one.
    #[must_use]
    pub fn options(&self, preset: QualityPreset) -> RenderOptions {
        let fog_scale = match self.scene {
            space_protocol::Scene::Galaxy | space_protocol::Scene::Solar => 0.0,
            space_protocol::Scene::DemoArea => 0.5,
            space_protocol::Scene::Materials => 1.0,
        };
        RenderOptions {
            gi_enabled: false,
            quality: Quality::CI,
            preset,
            fog_scale,
            ..RenderOptions::default()
        }
    }

    /// Where `position` is, in words, for the scenes that have named places:
    /// the nearest body of the solar system and how far away its surface is.
    #[must_use]
    pub fn whereabouts(&self, position: WorldPos) -> Option<String> {
        if self.scene != space_protocol::Scene::Solar {
            return None;
        }
        let bodies = solar_bodies();
        let (body, surface) = nearest(&bodies, position)?;
        Some(format!("{} from {}", format_distance(surface), body.name))
    }
}

impl Default for World {
    /// The world of the default scene.
    fn default() -> Self {
        Self::new(space_protocol::Scene::DEFAULT)
    }
}

// ---- PBR material chart --------------------------------------------------------

const CHART_COLUMNS: usize = 7;
/// Distance from the spawn point to the plane of spheres, metres.
const CHART_DISTANCE: f64 = 8.0;

/// The material chart: three rows of spheres whose roughness runs from mirror
/// smooth (left) to fully rough (right) — polished-to-brushed gold, a red
/// dielectric (plastic) and steel — over a smooth metal floor that mirrors
/// them with screen-space reflections on the `high`/`medium` presets, plus a
/// glass sphere (refraction on those presets) and an emissive one. Three
/// point lights: every sphere shows the GGX highlight widening with
/// roughness, the dielectric row keeps its diffuse colour, the metal rows
/// have none.
#[must_use]
pub fn materials_bodies() -> Vec<Body> {
    let rows: [(&str, f64, [f32; 3], f32); 3] = [
        ("gold", 1.45, [1.0, 0.78, 0.34], 1.0),
        ("plastic", 0.2, [0.8, 0.08, 0.08], 0.0),
        ("steel", -1.05, [0.56, 0.57, 0.58], 1.0),
    ];
    let mut bodies = Vec::new();
    for (name, y, albedo, metallic) in rows {
        for column in 0..CHART_COLUMNS {
            let t = column as f32 / (CHART_COLUMNS - 1) as f32;
            let x = (column as f64 - (CHART_COLUMNS - 1) as f64 * 0.5) * 1.25;
            bodies.push(Body {
                metallic,
                // 0.05 (mirror) .. 1.0 (fully rough).
                roughness: 0.05 + 0.95 * t,
                ..Body::sphere(name, at([x, y, -CHART_DISTANCE]), 0.5, albedo)
            });
        }
    }
    // A smooth metal floor under the chart, its top `EYE_HEIGHT` below the
    // spawn point.
    bodies.push(Body {
        metallic: 1.0,
        roughness: 0.08,
        ..Body::cube(
            "floor",
            at([0.0, -100.0 - f64::from(EYE_HEIGHT), -CHART_DISTANCE]),
            100.0,
            [0.75, 0.76, 0.78],
        )
    });
    // In front of the chart: a glass sphere and an emissive one.
    bodies.push(Body {
        roughness: 0.05,
        transmission: 1.0,
        ..Body::sphere("glass", at([2.2, -0.9, -4.6]), 0.75, [0.9, 0.95, 1.0])
    });
    bodies.push(Body {
        emissive: [2.6, 0.9, 0.15],
        ..Body::sphere("lamp", at([-2.4, -1.25, -4.8]), 0.4, [1.0, 0.5, 0.1])
    });
    bodies
}

/// The material chart's point lights: key, fill and back, and one behind the
/// spawn point for whatever stands there (a player's character).
#[must_use]
pub fn materials_lights() -> Vec<SceneLight> {
    vec![
        SceneLight {
            position: at([3.5, 4.5, -2.5]),
            color: [1.0, 0.98, 0.95],
            intensity: 90.0,
        },
        SceneLight {
            position: at([-5.0, 2.5, -3.0]),
            color: [0.6, 0.75, 1.0],
            intensity: 45.0,
        },
        SceneLight {
            position: at([0.0, 3.0, -13.0]),
            color: [1.0, 0.8, 0.6],
            intensity: 50.0,
        },
        SceneLight {
            position: at([-1.5, 3.0, 6.0]),
            color: [1.0, 0.95, 0.9],
            intensity: 60.0,
        },
    ]
}

/// The material chart seen from `camera`.
#[must_use]
pub fn materials_scene(camera: &CameraPose) -> Scene {
    scene_of(camera, &materials_bodies(), materials_lights())
}

// ---- real-size solar system ----------------------------------------------------

/// One astronomical unit, metres.
pub const AU: f64 = 1.495_978_707e11;

/// Mean radius (m), mean distance from the Sun (m), heliocentric longitude at
/// which the body is placed (degrees, in the XZ plane) and colour, for the
/// eight planets. Sizes and distances are the real ones; the longitudes are
/// simply a fixed, spread-out arrangement (nothing orbits yet).
const PLANETS: [(&str, f64, f64, f64, [f32; 3]); 8] = [
    ("Mercury", 2.4397e6, 5.791e10, 40.0, [0.55, 0.52, 0.50]),
    ("Venus", 6.0518e6, 1.0821e11, 130.0, [0.90, 0.78, 0.52]),
    ("Earth", 6.371e6, AU, 0.0, [0.16, 0.36, 0.75]),
    ("Mars", 3.3895e6, 2.2794e11, 310.0, [0.76, 0.33, 0.18]),
    ("Jupiter", 6.9911e7, 7.7857e11, 20.0, [0.80, 0.66, 0.50]),
    ("Saturn", 5.8232e7, 1.43353e12, 345.0, [0.88, 0.79, 0.58]),
    ("Uranus", 2.5362e7, 2.87246e12, 330.0, [0.60, 0.85, 0.90]),
    ("Neptune", 2.4622e7, 4.49506e12, 10.0, [0.25, 0.38, 0.85]),
];

const SUN_RADIUS: f64 = 6.9634e8;
const EARTH_RADIUS: f64 = 6.371e6;
const MOON_RADIUS: f64 = 1.7374e6;
const MOON_DISTANCE: f64 = 3.844e8;

/// The heliocentric position of a body `distance` metres out at `longitude`
/// degrees. Longitude 0 is the -Z axis, so that a camera on the sunward side
/// of Earth looks down -Z (the default view direction) at a fully lit planet.
fn heliocentric(distance: f64, longitude_deg: f64) -> [f64; 3] {
    let a = longitude_deg.to_radians();
    [distance * a.sin(), 0.0, -distance * a.cos()]
}

/// The Sun, the eight planets and the Moon as plain coloured spheres at their
/// real sizes and real mean distances (the Sun at the world origin).
#[must_use]
pub fn solar_bodies() -> Vec<Body> {
    let mut bodies = vec![Body {
        emissive: [24.0, 21.0, 16.0],
        ..Body::sphere("Sun", WorldPos::ZERO, SUN_RADIUS as f32, [1.0, 0.95, 0.8])
    }];
    for (name, radius, distance, longitude, color) in PLANETS {
        bodies.push(Body {
            // A faint glow of its own colour, so a planet far from the Sun (or
            // seen from its night side) is dim rather than invisible.
            emissive: color.map(|c| c * 0.04),
            ..Body::sphere(
                name,
                at(heliocentric(distance, longitude)),
                radius as f32,
                color,
            )
        });
    }
    // The Moon, a little ahead of and beside Earth as seen from the spawn.
    let earth = heliocentric(AU, 0.0);
    let moon = [
        earth[0] + MOON_DISTANCE * 0.62,
        earth[1] + MOON_DISTANCE * 0.18,
        earth[2] - MOON_DISTANCE * 0.76,
    ];
    bodies.push(Body {
        emissive: [0.02; 3],
        ..Body::sphere("Moon", at(moon), MOON_RADIUS as f32, [0.55, 0.54, 0.52])
    });
    bodies
}

/// Sunlight: one point light at the Sun, bright enough that a surface at 1 AU
/// is properly exposed. It falls off with the square of the distance like the
/// real thing, so Mercury is ~7x brighter and Neptune ~900x dimmer.
#[must_use]
pub fn solar_lights() -> Vec<SceneLight> {
    vec![SceneLight {
        position: WorldPos::ZERO,
        color: [1.0, 0.97, 0.92],
        intensity: (3.2 * AU * AU) as f32,
    }]
}

/// The solar system's spawn point: on the sunward side of Earth, four Earth
/// radii from its centre, where looking down -Z shows the lit planet.
#[must_use]
pub fn solar_spawn() -> WorldPos {
    let earth = heliocentric(AU, 0.0);
    at([earth[0], earth[1], earth[2] + 4.0 * EARTH_RADIUS])
}

/// The solar system seen from `camera`.
#[must_use]
pub fn solar_scene(camera: &CameraPose) -> Scene {
    scene_of(camera, &solar_bodies(), solar_lights())
}

// ---- the galaxy ------------------------------------------------------------------

/// Where the player spawns in the galaxy scene: pulled back along -Z from the
/// galaxy's centre at the origin, so the LOD selector refines a dense band of
/// the octree around the camera.
const GALAXY_SPAWN_M: [f64; 3] = [0.0, 0.0, -5.0e4];
/// The spawn cube sits this far in front (+Z) of the spawn point, metres.
const SPAWN_CUBE_DISTANCE_M: f64 = 12.0;
/// Distance from the spawn cube's centre to each face, metres.
const SPAWN_CUBE_HALF_M: f32 = 2.0;
/// The spawn cube's colour (linear rgb): magenta, absent from the star field.
const SPAWN_CUBE_COLOR: [f32; 3] = [1.0, 0.05, 0.8];

/// The galaxy through the LOD selector, seen from `camera`: the stars nearest
/// to the camera, a lit cube a short way in front of the spawn point and the
/// light that shows it (and whoever stands there).
#[must_use]
pub fn galaxy_scene(camera: &CameraPose) -> Scene {
    let started = std::time::Instant::now();
    let galaxy = Galaxy::new(0x5eed);
    let mut lod = LodSelector::new(30, 63, 8.0);
    // Cull against the frustum that is actually on screen (the selector's
    // default is 16:9, which would clip the sides of a wider window).
    if camera.aspect.is_finite() && camera.aspect > 0.0 {
        lod.aspect = camera.aspect;
    }
    // `Scene::build` traverses only the one depth-30 node containing the camera.
    // The galaxy is centred on the cell origin and the camera sits on its x=0/y=0
    // boundary, so that node covers a single octant and the galaxy would fill one
    // frame quadrant. Traverse the 8 nodes of that depth meeting at the origin.
    //
    // This runs every frame the camera moves, so it selects in one pass per
    // root and looks each node's distance up once: every selected node that
    // holds a star, with its distance from the camera.
    let mut stars: Vec<(f64, NodeRef)> = Vec::new();
    for octant in 0..8i64 {
        let index = [-(octant & 1), -((octant >> 1) & 1), -((octant >> 2) & 1)];
        let root = NodeRef::new(lod.min_depth, index);
        for node in lod.select_from(root, camera.position, camera.forward, camera.fov_y) {
            if galaxy.star_at(node).is_some() {
                stars.push((lod.distance_to(node, camera.position), node));
            }
        }
    }
    // Nearest to the camera first, so the instance budget goes to the stars
    // around the viewer rather than to one side (stable: ties keep root order).
    stars.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Leave room for the spawn cube.
    stars.truncate(MAX_SELECTED - 1);
    let selected: Vec<NodeRef> = stars.into_iter().map(|(_, node)| node).collect();
    let mut scene = Scene::from_selection(&galaxy, camera, &selected);
    log::trace!(
        "galaxy scene: {} stars in {:.2} ms",
        scene.instances.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    // A lit cube a short distance in front of the spawn point, in world space so
    // it stays put when the camera moves.
    let spawn = at(GALAXY_SPAWN_M);
    let cube = WorldPos::from_f64_metres_around(spawn, [0.0, 0.0, SPAWN_CUBE_DISTANCE_M]);
    scene.push_box(relative(camera, cube), SPAWN_CUBE_HALF_M, SPAWN_CUBE_COLOR);
    scene.with_lights(vec![
        SceneLight {
            position: WorldPos::from_f64_metres_around(cube, [4.0, 5.0, -6.0]),
            color: [1.0, 1.0, 1.0],
            intensity: 60.0,
        },
        // Behind the spawn point: lights the back of whoever stands there.
        SceneLight {
            position: WorldPos::from_f64_metres_around(spawn, [-2.0, 3.0, -6.0]),
            color: [0.8, 0.85, 1.0],
            intensity: 40.0,
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use space_protocol::Scene as SceneKind;

    /// The camera a player who has not moved or turned looks through, in
    /// first person.
    fn spawn_camera(world: &World, aspect: f32) -> CameraPose {
        let (yaw, pitch) = world.initial_view();
        let view = crate::input::InputState::looking(yaw, pitch);
        world.camera(world.spawn(), view.forward(), aspect)
    }

    /// Sizes and distances are the real ones, and survive the round trip
    /// through `WorldPos` at full precision: Neptune is 30 AU out to within a
    /// metre, not to within the ~500 km an `f32` position would give.
    #[test]
    fn the_solar_system_is_real_size() {
        let bodies = solar_bodies();
        let find = |name: &str| *bodies.iter().find(|b| b.name == name).expect(name);
        let sun = find("Sun");
        assert_eq!(sun.position, WorldPos::ZERO);
        assert!((f64::from(sun.radius) - 6.9634e8).abs() < 100.0);
        for (name, radius, from_sun) in [
            ("Earth", 6.371e6, AU),
            ("Jupiter", 6.9911e7, 7.7857e11),
            ("Neptune", 2.4622e7, 4.49506e12),
        ] {
            let body = find(name);
            assert!(
                (f64::from(body.radius) - radius).abs() / radius < 1e-6,
                "{name}"
            );
            let d = WorldPos::distance(sun.position, body.position);
            assert!((d - from_sun).abs() < 1.0, "{name}: {d} m from the Sun");
        }
        let moon = WorldPos::distance(find("Earth").position, find("Moon").position);
        assert!(
            (moon - MOON_DISTANCE).abs() / MOON_DISTANCE < 0.01,
            "{moon}"
        );
        assert_eq!(bodies.len(), 10);
    }

    /// The solar spawn point sits just off Earth, between it and the Sun,
    /// with Earth dead ahead — and the camera-relative instance the renderer
    /// gets is exact to well under a metre even though both are 1 AU from the
    /// origin.
    #[test]
    fn the_solar_spawn_faces_the_lit_side_of_earth() {
        let world = World::new(SceneKind::Solar);
        let camera = spawn_camera(&world, 16.0 / 9.0);
        let bodies = solar_bodies();
        let (body, surface) = nearest(&bodies, camera.position).expect("bodies");
        assert_eq!(body.name, "Earth");
        assert!((surface - 3.0 * 6.371e6).abs() < 1.0, "{surface}");
        assert_eq!(
            world.whereabouts(camera.position).as_deref(),
            Some("19113 km from Earth")
        );
        let scene = world.build(&camera);
        let earth = scene.instances[3].center;
        assert!(earth[0].abs() < 0.5 && earth[1].abs() < 0.5, "{earth:?}");
        assert!(
            (f64::from(earth[2]) + 4.0 * 6.371e6).abs() < 4.0,
            "{earth:?}"
        );
        // The Sun is behind the camera, one AU away.
        let sun = scene.instances[0].center;
        assert!((f64::from(sun[2]) - AU).abs() / AU < 1e-3, "{sun:?}");
        assert_eq!(scene.lights.len(), 1);
    }

    #[test]
    fn distances_read_naturally() {
        assert_eq!(format_distance(12.4), "12 m");
        assert_eq!(format_distance(25_484_000.0), "25484 km");
        assert_eq!(format_distance(AU), "1.000 AU");
        assert_eq!(format_distance(-500.0), "-500 m");
    }

    /// The chart is the documented 3 x 7 grid plus floor, glass and lamp,
    /// with roughness rising along each row, and its floor is where a
    /// character's feet are.
    #[test]
    fn the_material_chart_sweeps_roughness() {
        let bodies = materials_bodies();
        assert_eq!(bodies.len(), 3 * CHART_COLUMNS + 3);
        for row in bodies[..3 * CHART_COLUMNS].chunks(CHART_COLUMNS) {
            assert!((row[0].roughness - 0.05).abs() < 1e-6);
            assert!((row[CHART_COLUMNS - 1].roughness - 1.0).abs() < 1e-6);
            assert!(row.windows(2).all(|w| w[0].roughness < w[1].roughness));
            assert!(row.iter().all(|b| b.metallic == row[0].metallic));
        }
        assert!(bodies.iter().any(|b| b.transmission > 0.0));
        let floor = bodies.iter().find(|b| b.name == "floor").expect("floor");
        assert_eq!(floor.shape, SHAPE_BOX);
        let top = metres(floor.position)[1] + f64::from(floor.radius);
        let world = World::new(SceneKind::Materials);
        assert!(
            (top - world.floor().expect("a floor")).abs() < 1e-3,
            "{top}"
        );
        assert!(materials_lights().len() <= 8);
    }

    /// The galaxy scene builds through the LOD selector, stays within the
    /// `MAX_SELECTED` bound, and its last instance is the spawn cube: a box
    /// straight ahead of the spawn camera.
    #[test]
    fn the_galaxy_is_bounded_and_has_a_cube_in_front_of_the_spawn_point() {
        let world = World::new(SceneKind::Galaxy);
        let camera = spawn_camera(&world, 1.0);
        assert!(camera.forward[2] > 0.99, "{:?}", camera.forward);
        let scene = world.build(&camera);
        assert!(scene.instances.len() <= MAX_SELECTED);
        assert!(scene.instances.len() > 100, "{}", scene.instances.len());
        let cube = scene.instances.last().expect("cube instance");
        assert_eq!(cube.shape, SHAPE_BOX);
        assert!(cube.center[0].abs() < 1e-3 && cube.center[1].abs() < 1e-3);
        assert!((f64::from(cube.center[2]) - SPAWN_CUBE_DISTANCE_M).abs() < 1e-3);
        assert_eq!(cube.center[3], SPAWN_CUBE_HALF_M);
    }

    /// Every scene of the catalogue has a world: something to draw around the
    /// spawn point, lights to see it by, and options that keep the preset.
    #[test]
    fn every_scene_has_a_world() {
        for kind in SceneKind::ALL {
            let world = World::new(kind);
            assert_eq!(world.scene(), kind);
            let camera = spawn_camera(&world, 16.0 / 9.0);
            assert_eq!(camera.position, world.spawn());
            let scene = world.build(&camera);
            assert!(
                !scene.instances.is_empty() || !scene.models.is_empty(),
                "{kind}: nothing to draw"
            );
            assert!(!scene.lights.is_empty(), "{kind}: no light");
            assert!(scene.lights.len() <= 8, "{kind}: the renderer has 8 lights");
            assert!(scene.overlay.is_empty(), "{kind}: the overlay is the app's");
            let finite = scene
                .instances
                .iter()
                .all(|i| i.center.iter().all(|v| v.is_finite()));
            assert!(finite, "{kind}");
            let lens = world.lens();
            assert!(lens.near > 0.0 && lens.far > lens.near);
            assert!((lens.fov_y.to_degrees() - 50.0).abs() < 1e-3);
            // Every scene has surroundings (nothing is pitch black), and only
            // the scenes that are not in space paint them as a sky.
            let environment = world.environment();
            assert!(!environment.is_none(), "{kind}: no environment");
            assert_eq!(scene.environment, environment, "{kind}");
            for preset in QualityPreset::ALL {
                let options = world.options(preset);
                assert_eq!(options.preset, preset);
                assert!(!options.gi_enabled);
            }
            let astronomical = matches!(kind, SceneKind::Solar | SceneKind::Galaxy);
            let fog = world.options(QualityPreset::High).fog_scale;
            assert_eq!(fog == 0.0, astronomical, "{kind}");
            assert_eq!(world.floor().is_none(), astronomical, "{kind}");
            assert_eq!(environment.background, !astronomical, "{kind}");
            assert_eq!(
                world.whereabouts(world.spawn()).is_some(),
                kind == SceneKind::Solar
            );
        }
        assert_eq!(World::default().scene(), SceneKind::DemoArea);
        // A degenerate aspect (a 0-pixel-high frame) does not poison the camera.
        let camera = World::default().camera(WorldPos::ZERO, [0.0, 0.0, -1.0], f32::NAN);
        assert_eq!(camera.aspect, 1.0);
    }

    /// The scenes are different worlds: switching the scene changes what is
    /// drawn, and building the same one twice does not.
    #[test]
    fn the_scenes_differ_from_each_other() {
        let signature = |kind: SceneKind| {
            let world = World::new(kind);
            let scene = world.build(&spawn_camera(&world, 1.5));
            (
                scene.instances.len(),
                scene.models.len(),
                scene.lights.len(),
            )
        };
        let all: Vec<_> = SceneKind::ALL.into_iter().map(signature).collect();
        for (i, a) in all.iter().enumerate() {
            assert_eq!(*a, signature(SceneKind::ALL[i]), "deterministic");
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn the_shared_models_are_one_arc_each() {
        assert!(Arc::ptr_eq(&character_mesh(), &character_mesh()));
        assert!(Arc::ptr_eq(&ship_mesh(), &ship_mesh()));
        assert!(!Arc::ptr_eq(&character_mesh(), &ship_mesh()));
        let character = character_mesh();
        // Origin at the feet, 1.8 m tall: the eye point is inside the helmet.
        assert!(character.bounds_min[1].abs() < 1e-3);
        assert!(character.bounds_max[1] > EYE_HEIGHT && character.bounds_max[1] < 1.9);
    }

    #[test]
    fn positions_convert_to_metres_and_back() {
        let p = at([1500.25, -2.5, 3.0e6]);
        let m = metres(p);
        assert!((m[0] - 1500.25).abs() < 1e-3 && (m[1] + 2.5).abs() < 1e-3);
        assert!((m[2] - 3.0e6).abs() < 1e-2);
        let camera = World::default().camera(at([1.0, 2.0, 3.0]), [0.0, 0.0, -1.0], 1.0);
        let rel = relative(&camera, at([1.0, 2.0, -7.0]));
        assert!(rel[0].abs() < 1e-5 && rel[1].abs() < 1e-5 && (rel[2] + 10.0).abs() < 1e-5);
    }
}
