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
//! | `demo-area` | [`crate::demo_area`]: everything the renderer can do, around the spawn point, riding along with Earth with the real-size solar system moving around it ([`sky_bodies`]) |
//! | `materials` | a PBR material chart ([`materials_bodies`]) |
//! | `galaxy` | the deterministic galaxy through the LOD selector |
//!
//! The demo area's sky moves: where the Sun, the planets and the Moon are is
//! a function of the scene time ([`SkyTime`]: the shard's scene clock and the
//! tick a frame is drawn at), so it is not part of [`World::build`] (built
//! once per camera position) but added to every frame ([`World::sky`]).

use std::sync::{Arc, OnceLock};

use space_protocol::SceneClock;
use space_render::{
    CameraPose, Environment, GpuInstance, MeshData, ModelInstance, Quality, QualityPreset,
    RenderOptions, Scene, SceneLight, SHAPE_BOX, SHAPE_SPHERE,
};
use space_scene::solar::SkyBody;
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

/// The faint, even light of the scenes that are in space (the galaxy): no
/// sky is painted — space is black — but whoever stands
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
            space_protocol::Scene::DemoArea | space_protocol::Scene::Materials => (0.0, 0.0),
        }
    }

    /// The camera's field of view and depth range for this scene. (The
    /// projection's far plane is infinite: the demo area's near plane suits
    /// a crate at arm's length and the planets 30 AU out alike.)
    #[must_use]
    pub fn lens(&self) -> Lens {
        let (near, far) = match self.scene {
            space_protocol::Scene::DemoArea => (0.05, 1.0e13),
            space_protocol::Scene::Materials => (0.05, 1.0e3),
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
            space_protocol::Scene::Galaxy => None,
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
            space_protocol::Scene::Galaxy => SPACE_AMBIENT,
        }
    }

    /// The scene's content as `camera` sees it: instances, models, lights
    /// and environment, camera-relative, with an empty overlay.
    #[must_use]
    pub fn build(&self, camera: &CameraPose) -> Scene {
        let scene = match self.scene {
            space_protocol::Scene::DemoArea => DemoArea::shared().scene_for(camera, self.detail),
            space_protocol::Scene::Materials => materials_scene(camera),
            space_protocol::Scene::Galaxy => galaxy_scene(camera),
        };
        scene.with_environment(self.environment())
    }

    /// The render options every front end uses for this scene, so the same
    /// preset gates the same features — including the SSR/refraction post
    /// pass — whichever way the frame leaves the GPU. The scenes in space
    /// (the demo area with its planets, the galaxy) turn the presets'
    /// distance fog off: it is tuned for metres, and would paint every star
    /// and planet solid fog colour.
    #[must_use]
    pub fn options(&self, preset: QualityPreset) -> RenderOptions {
        let fog_scale = match self.scene {
            space_protocol::Scene::Galaxy | space_protocol::Scene::DemoArea => 0.0,
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

    /// Where `position` is, in words, for the scenes that have named places
    /// (the demo area's sky): the nearest body of the solar system at `time`
    /// and how far away its surface is.
    #[must_use]
    pub fn whereabouts(&self, position: WorldPos, time: &SkyTime) -> Option<String> {
        let bodies = self.sky_bodies(time);
        let (body, surface) = nearest(&bodies, position)?;
        Some(format!("{} from {}", format_distance(surface), body.name))
    }

    /// The solar system's bodies at `time`, in scene space (none for the
    /// scenes without one).
    #[must_use]
    pub fn sky_bodies(&self, time: &SkyTime) -> Vec<Body> {
        match self.scene {
            space_protocol::Scene::DemoArea => sky_bodies(time),
            space_protocol::Scene::Materials | space_protocol::Scene::Galaxy => Vec::new(),
        }
    }

    /// What moves in this scene's sky at `time` as `camera` sees it: the
    /// instances of the Sun, the planets and the Moon, and the sunlight (a
    /// point light at the Sun). Empty for the scenes without a sky.
    #[must_use]
    pub fn sky(
        &self,
        camera: &CameraPose,
        time: &SkyTime,
    ) -> (Vec<GpuInstance>, Option<SceneLight>) {
        if self.scene != space_protocol::Scene::DemoArea {
            return (Vec::new(), None);
        }
        let bodies = sky_bodies(time);
        let sun = bodies.first().map(|sun| SceneLight {
            position: sun.position,
            color: [1.0, 0.97, 0.92],
            intensity: SUN_INTENSITY,
        });
        (bodies.iter().map(|b| b.instance(camera)).collect(), sun)
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

// ---- the demo area's sky: the solar system --------------------------------------

/// One astronomical unit, metres.
pub const AU: f64 = space_scene::solar::AU;

/// Sunlight: a point light at the Sun this strong, so that a surface at 1 AU
/// is properly exposed; it falls off with the square of the distance like
/// the real thing.
pub const SUN_INTENSITY: f32 = (3.2 * AU * AU) as f32;

/// The moment the demo area's sky is drawn at: the shard's mapping from
/// ticks to scene time and the (fractional) tick of the frame.
///
/// A body is drawn **interpolated between its places at the two ticks
/// around** the frame's — the way the own ship is — so a ship carried along
/// by the Moon (and drawn between two of its ticks) stays exactly where it
/// is on the Moon in the picture too.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SkyTime {
    /// The shard's mapping.
    pub clock: SceneClock,
    /// The tick drawn at.
    pub tick: f64,
}

impl SkyTime {
    /// At `tick` by `clock`.
    #[must_use]
    pub const fn new(clock: SceneClock, tick: f64) -> Self {
        Self { clock, tick }
    }

    /// `body`'s centre, metres from the spawn point.
    #[must_use]
    pub fn center(&self, body: &SkyBody) -> [f64; 3] {
        let tick = if self.tick.is_finite() {
            self.tick.max(0.0)
        } else {
            0.0
        };
        let base = tick.floor();
        let alpha = tick - base;
        let t0 = self.clock.time_us_at(base as u64);
        let t1 = self.clock.time_us_at(base as u64 + 1);
        let (a, b) = (body.center(t0, 0.0), body.center(t1, 0.0));
        core::array::from_fn(|k| a[k] + (b[k] - a[k]) * alpha)
    }

    /// The scene time drawn at, µs (the whole microsecond).
    #[must_use]
    pub fn time_us(&self) -> i64 {
        self.clock.time_at(self.tick).0
    }
}

/// How a body of the solar system is drawn: its colour.
fn sky_color(name: &str) -> [f32; 3] {
    match name {
        "Sun" => [1.0, 0.95, 0.8],
        "Mercury" => [0.55, 0.52, 0.50],
        "Venus" => [0.90, 0.78, 0.52],
        "Earth" => [0.16, 0.36, 0.75],
        "Mars" => [0.76, 0.33, 0.18],
        "Jupiter" => [0.80, 0.66, 0.50],
        "Saturn" => [0.88, 0.79, 0.58],
        "Uranus" => [0.60, 0.85, 0.90],
        "Neptune" => [0.25, 0.38, 0.85],
        _ => [0.55, 0.54, 0.52],
    }
}

/// The Sun, the eight planets and the Moon (in `space_scene::solar`'s order:
/// the Sun first) as plain coloured spheres at their real sizes, where they
/// are at `time`, in scene space: Earth stands still ahead of the spawn
/// point, the rest move on their orbits.
#[must_use]
pub fn sky_bodies(time: &SkyTime) -> Vec<Body> {
    space_scene::solar::sky_bodies()
        .iter()
        .map(|b| {
            let color = sky_color(b.name);
            let position = at(time.center(b));
            if b.name == "Sun" {
                Body {
                    emissive: [24.0, 21.0, 16.0],
                    ..Body::sphere(b.name, position, b.radius as f32, color)
                }
            } else {
                Body {
                    // A faint glow of its own colour, so a planet far from the
                    // Sun (or seen from its night side) is dim rather than
                    // invisible.
                    emissive: color.map(|c| c * 0.03),
                    roughness: 0.9,
                    ..Body::sphere(b.name, position, b.radius as f32, color)
                }
            }
        })
        .collect()
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

    /// The demo area's sky: real sizes, Earth where `space-scene` puts it
    /// (eight Earth radii ahead of the spawn point, above the horizon), the
    /// Moon at its distance from Earth, the planets at their distances from
    /// the Sun — at any scene time, interpolated between ticks; and the
    /// whereabouts name the nearest body.
    #[test]
    fn the_demo_areas_sky_is_the_solar_system_at_real_size() {
        let world = World::new(SceneKind::DemoArea);
        let clock = SceneClock::new(1000, 1_763_900_000_000_000);
        for tick in [1000.0, 1234.5, 90_000.25] {
            let time = SkyTime::new(clock, tick);
            let bodies = world.sky_bodies(&time);
            assert_eq!(bodies.len(), 10);
            let find = |name: &str| *bodies.iter().find(|b| b.name == name).expect(name);
            let (sun, earth) = (find("Sun"), find("Earth"));
            let to_earth = WorldPos::distance(WorldPos::ZERO, earth.position);
            assert!((to_earth - 8.0 * 6.371e6).abs() < 1.0, "{to_earth}");
            assert!(metres(earth.position)[1] > 0.0, "above the horizon");
            assert!((f64::from(earth.radius) - 6.371e6).abs() < 1.0);
            let moon = WorldPos::distance(earth.position, find("Moon").position);
            assert!((moon - 3.844e8).abs() < 1.0e3, "{moon}");
            for (name, from_sun) in [
                ("Earth", AU),
                ("Jupiter", 7.7857e11),
                ("Neptune", 4.49506e12),
            ] {
                let d = WorldPos::distance(sun.position, find(name).position);
                assert!((d - from_sun).abs() / from_sun < 1e-6, "{name}: {d}");
            }
            let (instances, sun_light) = world.sky(&spawn_camera(&world, 1.5), &time);
            assert_eq!(instances.len(), 10);
            assert_eq!(sun_light.expect("sunlight").position, sun.position);
            assert_eq!(
                world.whereabouts(WorldPos::ZERO, &time).as_deref(),
                Some("44597 km from Earth")
            );
        }
        // Between two ticks a body is between its places at the two.
        let a = world.sky_bodies(&SkyTime::new(clock, 1000.0))[9].position;
        let b = world.sky_bodies(&SkyTime::new(clock, 1001.0))[9].position;
        let mid = world.sky_bodies(&SkyTime::new(clock, 1000.5))[9].position;
        let half = WorldPos::distance(a, b) / 2.0;
        assert!((WorldPos::distance(a, mid) - half).abs() < 1.0);
        // The other scenes have no sky.
        let galaxy = World::new(SceneKind::Galaxy);
        assert!(galaxy.sky_bodies(&SkyTime::default()).is_empty());
        assert_eq!(
            galaxy.whereabouts(WorldPos::ZERO, &SkyTime::default()),
            None
        );
    }

    /// The Moon goes round once in about four and a half minutes, Earth's
    /// year is a real hour: after a quarter of a lunar month the Moon is a
    /// quarter of the way round.
    #[test]
    fn the_sky_moves_one_year_per_hour() {
        let world = World::new(SceneKind::DemoArea);
        let clock = SceneClock::new(0, 0);
        let month: f64 = 27.321_661 * 86_400.0 / 8766.0 * 30.0; // ticks
        let moon_at = |tick: f64| metres(world.sky_bodies(&SkyTime::new(clock, tick))[9].position);
        let earth = metres(world.sky_bodies(&SkyTime::new(clock, 0.0))[3].position);
        let rel = |p: [f64; 3]| [p[0] - earth[0], p[1] - earth[1], p[2] - earth[2]];
        let (a, b) = (rel(moon_at(0.0)), rel(moon_at(month / 4.0)));
        let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (3.844e8 * 3.844e8);
        assert!(dot.abs() < 0.01, "a quarter turn: cos = {dot}");
        assert!((month / 30.0 - 269.3).abs() < 0.5, "{} s", month / 30.0);
        let sun_at = |tick: f64| metres(world.sky_bodies(&SkyTime::new(clock, tick))[0].position);
        let (s0, s1) = (sun_at(0.0), sun_at(3600.0 * 30.0));
        assert!(
            WorldPos::distance(at(s0), at(s1)) < 1.0e-3 * AU,
            "a year later"
        );
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
            let in_space = matches!(kind, SceneKind::DemoArea | SceneKind::Galaxy);
            let fog = world.options(QualityPreset::High).fog_scale;
            assert_eq!(fog == 0.0, in_space, "{kind}");
            assert_eq!(world.floor().is_none(), kind == SceneKind::Galaxy, "{kind}");
            assert_eq!(environment.background, !in_space, "{kind}");
            assert_eq!(
                world
                    .whereabouts(world.spawn(), &SkyTime::default())
                    .is_some(),
                kind == SceneKind::DemoArea
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
