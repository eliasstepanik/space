//! The demo area: one compact place around the spawn point that shows
//! everything the renderer can do, so a person or an agent can check it at a
//! glance.
//!
//! Scene space: the spawn point (a player's eye) is the origin, +Y is up and a
//! freshly spawned player looks down -Z. The floor's top is
//! [`EYE_HEIGHT`] below. Seen from the spawn point:
//!
//! ```text
//!          the sky: Earth (ahead, up to the right), the Moon, the Sun, stars
//!
//!   z=-30   [ the Kestrel on its apron: 24 m of ship, nose to the left,  ]
//!           [ port side to the spawn point, a floodlight on a mast       ]
//!
//!   z=-17   [ RGB shadow wall ]   [   PBR chart   ]
//!   z=-15      casters: character,   metal row                       [ ship on ]
//!              ship, pillar          plastic row                     [ its pad ]
//!   z=-11      red green blue lamps    emissive      [ texture crates, 3 x 2 ]
//!   z= -7   [ glass cube + sphere ]    beacon        [ mirror floor: ship,   ]
//!   z= -5   [ in a corner of two  ]                  [ red ball, a character ]
//!           [ textured panels     ]  two characters, a chrome ball
//!   z=  0                           spawn (looking up this page)
//! ```
//!
//! * **textures**: the checkered floor; one crate per procedural texture
//!   (checkerboard, UV grid, hull panels, brushed metal, hazard stripes, a
//!   solid colour) — the UV grid's labels read `A1` at the top left when a
//!   mapping is the right way round;
//! * **reflections**: a polished slab let into the floor (screen-space: it
//!   mirrors what is on screen — the things standing on it and the crates
//!   behind it);
//! * **refraction**: a glass cube and a glass sphere in a corner of two
//!   textured panels (UV grid and checkerboard), with a striped pillar;
//! * **materials**: gold (metal) and red plastic (dielectric) spheres from
//!   mirror smooth to fully rough;
//! * **lights and shadows**: a warm key, two warm fills, and red, green and
//!   blue lamps whose shadows of a character, a ship and a pillar fall on a
//!   white wall in the complementary colours (models shadowing a model);
//! * **environment**: space ([`ENVIRONMENT`]) — black, but with a faint,
//!   cool ambient light in every shadow (nothing that should be lit is
//!   black), what the chrome ball and the smooth end of the gold row reflect,
//!   and what the glass shows where nothing stands behind it;
//! * **sunlight**: the Sun's light from where the Sun really is at the scene
//!   time (added by the app every frame, `crate::scenes::World::sky`) — it
//!   goes round once an hour;
//! * **emissive**: the lamps around the lights, a glowing beacon cube;
//! * **models**: the ship (small over the mirror, full size on its pad) and
//!   the placeholder character, some facing the spawn point and some away;
//! * **a real asset**: the IC-4 Kestrel ([`crate::kestrel`]) landed on an
//!   apron behind the stations ([`KESTREL_BERTH`]) — a 418-thousand-triangle
//!   glTF model with a complete interior, to walk up to, around and into
//!   (nothing is solid), lit by a floodlight on the apron and, for whoever
//!   goes in, by a lamp in its crew quarters;
//! * **depth**: the solar system at real size around the area
//!   (`space_scene::solar`): Earth eight Earth radii ahead and above, the
//!   Moon, the Sun and the planets moving on their orbits, and stars all
//!   round.
//!
//! The area **rides along with Earth** (the old `solar` scene merged into
//! this one): Earth stands still in the sky, the Moon goes round it every
//! four and a half minutes and the Sun once an hour. The sky is not painted:
//! space is black and there is no distance fog. The point lights model the
//! shapes and cast the shadows: six for the stations, one for the ship, the
//! Sun as the eighth — and inside the ship the cabin lamp takes the Sun's
//! place, lit only for a camera that is in there to see it.
//!
//! The meshes are built once ([`DemoArea::shared`]) and every frame reuses
//! their `Arc`s, which is what keeps them on the GPU.
//!
//! # The physics showcase
//!
//! To the right of the spawn point (x 10..19) is the **gravity room** and to
//! its left (x -19..-10) the **zero-g corner** (`space_scene::demo`). What
//! moves there — crates, balls, the seesaw, the pendulums, the chain, the
//! door — is simulated by the shard and drawn from replication
//! ([`crate::bodies`]). What stands still is drawn here ([`DemoArea::showcase`]),
//! built from the shared scene definition itself: every static collider with
//! a visual (the curbs, the concave bowl from the shape catalogue, the ramp,
//! the seesaw's fulcrum, the gantry, the door's post) is drawn as its own
//! collision shape, so collider and picture cannot drift apart. The colliders
//! without one (visual "none") are the stations above, drawn by hand and
//! checked against their colliders by `space-scene`'s tests.
//!
//! Each of the two places says what it is: the gravity room has a blue mat
//! on the floor, glowing cyan edges along the whole volume in which gravity
//! acts (from the floor to 8 m up) and a sign over its open side facing the
//! spawn point, `GRAVITY 9.81 M/S2`; the zero-g corner a violet mat with a
//! glowing violet border and a `ZERO G` sign. Up to the left, beyond the
//! floor, hangs the planetoid (`space_scene::demo::PLANETOID_CENTER`): a rock
//! with large-body gravity (4 m/s² on its surface, none beyond 16 m), crates
//! resting on it and three balls in orbit, drawn from the same definition,
//! with a `PLANETOID 4 M/S2` sign under it.

use std::sync::{Arc, OnceLock};

use space_render::{
    rotation_y, CameraPose, Environment, MeshData, MeshMaterial, Scene, SceneLight, Texture,
    IDENTITY_ROTATION,
};

use space_scene::demo::{
    GRAVITY_ROOM_CENTER, GRAVITY_ROOM_HALF, PLANETOID_CENTER, PLANETOID_RADIUS,
};

use crate::bodies::MeshCache;
use crate::kestrel::{Berth, Detail, CABIN_LAMP, GEAR_DROP};
use crate::meshes::{glow, quat_matrix, sign_mesh};
use crate::scenes::{at, character_mesh, ship_mesh, Body, Prop, EYE_HEIGHT};
use space_world::WorldPos;

/// The height of the floor's top, metres (scene space).
pub const FLOOR: f64 = -(EYE_HEIGHT as f64);

/// The zero-g corner's footprint on the floor, scene space: (min x z, max
/// x z). Weightless like everything outside the gravity room; marked so
/// that it reads as a place, around the floating crates, the spring pair and
/// the door.
pub const ZERO_G_CORNER: ([f64; 2], [f64; 2]) = ([-19.0, -7.5], [-10.0, 6.5]);

/// A prop with no turn.
fn fixed(mesh: &Arc<MeshData>, position: [f64; 3]) -> Prop {
    Prop {
        mesh: Arc::clone(mesh),
        position: at(position),
        rotation: IDENTITY_ROTATION,
        scale: 1.0,
    }
}

/// Everything a player walks among lies within this distance of the spawn
/// point, metres; only the backdrop is further out.
pub const AREA_RADIUS: f64 = 45.0;

/// The top of the Kestrel's apron: a step above the floor.
pub const KESTREL_APRON_TOP: f64 = FLOOR + 0.12;

/// Half the apron's size along X and Z, metres.
pub const KESTREL_APRON_HALF: [f64; 2] = [15.0, 10.25];

/// Where the Kestrel stands: on its landing gear on the apron behind the
/// stations, 30 m in front of the spawn point, its nose to the left and
/// turned 15 degrees towards the newcomer — who sees its port side (the crew
/// hatch, the name) over the stations, between the shadow wall and the
/// texture crates.
///
/// This is the whole of the ship's placement (asset, position, heading), in
/// one record, for the day the area's objects are described in a scene
/// definition shared with the server.
pub const KESTREL_BERTH: Berth = Berth::landed(
    0.0,
    KESTREL_APRON_TOP,
    -30.5,
    105.0 * std::f32::consts::PI / 180.0,
);

/// Nothing at all is further from the spawn point than this, metres.
pub const BACKDROP_RADIUS: f64 = 8_000.0;

/// The demo area's surroundings: space. Nothing is painted behind
/// everything (the stars and planets are), but a faint, cool ambient light
/// stands in for the light of Earth, the stars and the scattered light of
/// the lamps: a shadow is a dim blue-grey — never black — the far corners of
/// the floor are still plainly a floor, and the red, green and blue lamps own
/// their wall. Polished metal reflects that dim gradient.
pub const ENVIRONMENT: Environment =
    Environment::gradient([0.07, 0.09, 0.15], [0.06, 0.065, 0.08], [0.03, 0.033, 0.04]);

/// The demo area's content: props (mesh models), bodies (spheres and boxes)
/// and lights at fixed places.
#[derive(Debug)]
pub struct DemoArea {
    props: Vec<Prop>,
    /// The physics showcase's statics and signs ([`DemoArea::showcase`]).
    showcase: Vec<Prop>,
    bodies: Vec<Body>,
    lights: Vec<SceneLight>,
    /// The ship on the apron: drawn in the level of detail each frame's
    /// camera calls for, so not one of the fixed `props`.
    kestrel: Berth,
    /// The lamp inside the ship: part of a frame only when its camera is in
    /// there (nothing of the deck it lights can be seen from outside).
    cabin_light: SceneLight,
}

/// The area's lamps are lights of a frame only for a camera this close to the
/// spawn point, metres (see [`DemoArea::scene_for`]).
pub const LAMP_RANGE: f64 = 20_000.0;

/// A camera this close to the ship's bounds has the cabin lamp lit, metres.
const CABIN_LIGHT_RANGE: f64 = 1.5;

/// The mirror slab: 6 m x 6 m, 20 cm thick, its centre 8 cm below the floor
/// so that its top is 2 cm proud of the floor and its underside (18 cm below
/// the floor) stays inside the 20 cm floor slab.
const MIRROR_CENTRE: [f64; 3] = [6.0, FLOOR - 0.08, -7.0];
const MIRROR_HALF: [f32; 3] = [3.0, 0.1, 3.0];

/// The mirror slab's mesh (one shared mesh, built once).
fn mirror_mesh() -> Arc<MeshData> {
    static MESH: OnceLock<Arc<MeshData>> = OnceLock::new();
    Arc::clone(MESH.get_or_init(|| {
        Arc::new(MeshData::cuboid(
            MIRROR_HALF,
            1.0,
            paint([0.9, 0.92, 0.95], 1.0, 0.04),
        ))
    }))
}

fn paint(color: [f32; 3], metallic: f32, roughness: f32) -> MeshMaterial {
    MeshMaterial {
        base_color: color,
        metallic,
        roughness,
        emissive: [0.0; 3],
        base_color_texture: None,
    }
}

/// `mesh` with `texture` on every part.
fn textured(mut mesh: MeshData, texture: Texture) -> Arc<MeshData> {
    let slot = mesh.add_texture(texture);
    for part in &mut mesh.parts {
        part.material.base_color_texture = Some(slot);
    }
    Arc::new(mesh)
}

fn light(position: [f64; 3], color: [f32; 3], intensity: f32) -> SceneLight {
    SceneLight {
        position: at(position),
        color,
        intensity,
    }
}

/// The yaw that makes a model (which faces -Z) at `from` face the point `to`.
fn facing(from: [f64; 3], to: [f64; 3]) -> f32 {
    let (dx, dz) = (to[0] - from[0], to[2] - from[2]);
    (-dx).atan2(-dz) as f32
}

/// A small deterministic generator for the backdrop's arrangement.
struct Dice(u64);

impl Dice {
    /// The next value in `0..1`.
    fn roll(&mut self) -> f64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 11) as f64 / (1u64 << 53) as f64
    }

    fn between(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.roll()
    }
}

impl DemoArea {
    /// The one demo area of this process: its meshes are built on first use
    /// and shared by every frame (and every render context) after that.
    #[must_use]
    pub fn shared() -> &'static Self {
        static AREA: OnceLock<DemoArea> = OnceLock::new();
        AREA.get_or_init(Self::new)
    }

    /// Build the demo area. Prefer [`DemoArea::shared`]: a second instance
    /// has its own meshes, which a renderer uploads a second time.
    #[must_use]
    pub fn new() -> Self {
        let mut area = Self {
            props: Vec::new(),
            showcase: Vec::new(),
            bodies: Vec::new(),
            lights: Vec::new(),
            kestrel: KESTREL_BERTH,
            cabin_light: light(KESTREL_BERTH.to_scene(CABIN_LAMP), [1.0, 0.9, 0.74], 6.0),
        };
        area.floor();
        area.people();
        area.glass_corner();
        area.mirror_floor();
        area.texture_crates();
        area.material_chart();
        area.shadow_theatre();
        area.beacon_and_pad();
        area.lighting();
        area.kestrel_apron();
        area.backdrop();
        area.physics_showcase();
        area
    }

    /// The physics showcase's statics from the shared scene definition, and
    /// the marks that say where gravity acts and where it does not (see the
    /// module docs, "The physics showcase").
    fn physics_showcase(&mut self) {
        let mut meshes = MeshCache::new();
        for s in space_scene::statics(space_protocol::Scene::DemoArea) {
            if s.visual.0 == space_scene::visual::NONE {
                continue;
            }
            let Some(mesh) = meshes.mesh(s.visual.0, &s.shape) else {
                continue;
            };
            self.showcase.push(Prop {
                mesh,
                position: at(s.offset),
                rotation: quat_matrix(s.rotation),
                scale: 1.0,
            });
        }
        let [cx, cy, cz] = GRAVITY_ROOM_CENTER;
        let [hx, hy, hz] = GRAVITY_ROOM_HALF;
        // The zone reaches a metre under the floor; what shows is the floor
        // up to its top.
        let top = cy + hy;
        let room = ([cx - hx, cz - hz], [cx + hx, cz + hz]);
        self.zone_marks(
            room,
            top,
            [0.42, 0.56, 0.86],
            Texture::checkerboard(256, 8, [176, 196, 236], [140, 160, 206]),
            [0.25, 1.6, 2.4],
        );
        self.sign(
            "GRAVITY 9.81 M/S2",
            [cx - hx + 0.06, FLOOR + 5.2, cz],
            -std::f32::consts::FRAC_PI_2,
            [0.25, 1.6, 2.4],
        );
        let (lo, hi) = ZERO_G_CORNER;
        self.zone_marks(
            (lo, hi),
            FLOOR,
            [0.7, 0.56, 0.96],
            Texture::checkerboard(256, 8, [206, 188, 240], [170, 150, 214]),
            [1.4, 0.5, 2.4],
        );
        self.sign(
            "ZERO G",
            [hi[0] - 0.06, FLOOR + 4.2, 0.5 * (lo[1] + hi[1])],
            std::f32::consts::FRAC_PI_2,
            [1.4, 0.5, 2.4],
        );
        // Under the planetoid, turned to the spawn point.
        let [px, py, pz] = PLANETOID_CENTER;
        let to_spawn = (-px).atan2(-pz) as f32;
        let along = [-px, -pz];
        let l = (along[0] * along[0] + along[1] * along[1]).sqrt();
        self.sign(
            "PLANETOID 4 M/S2",
            [
                px + along[0] / l * 1.5,
                py - PLANETOID_RADIUS - 1.6,
                pz + along[1] / l * 1.5,
            ],
            to_spawn,
            [2.4, 1.3, 0.5],
        );
    }

    /// A floor mat over the rectangle `(min x z, max x z)` and glowing edges
    /// round it: along the floor, and — when `top` is above the floor — up
    /// the four corners and round the top as well.
    fn zone_marks(
        &mut self,
        (lo, hi): ([f64; 2], [f64; 2]),
        top: f64,
        tint: [f32; 3],
        texture: Texture,
        radiance: [f32; 3],
    ) {
        let half = [(hi[0] - lo[0]) * 0.5, (hi[1] - lo[1]) * 0.5];
        let centre = [(hi[0] + lo[0]) * 0.5, (hi[1] + lo[1]) * 0.5];
        // A mat a few millimetres thick on the floor.
        const MAT: f32 = 0.003;
        let mat = textured(
            MeshData::cuboid(
                [half[0] as f32, MAT, half[1] as f32],
                2.0,
                paint(tint, 0.0, 0.8),
            ),
            texture,
        );
        self.showcase
            .push(fixed(&mat, [centre[0], FLOOR + f64::from(MAT), centre[1]]));
        let bar = |half: [f32; 3]| Arc::new(MeshData::cuboid(half, 1.0, glow(radiance)));
        const W: f32 = 0.025;
        // Along the floor, just inside the edge.
        let y = FLOOR + 2.0 * f64::from(MAT) + f64::from(W) * 0.5;
        let along_x = bar([half[0] as f32, W * 0.5, W]);
        let along_z = bar([W, W * 0.5, half[1] as f32]);
        for z in [lo[1] + f64::from(W), hi[1] - f64::from(W)] {
            self.showcase.push(fixed(&along_x, [centre[0], y, z]));
        }
        for x in [lo[0] + f64::from(W), hi[0] - f64::from(W)] {
            self.showcase.push(fixed(&along_z, [x, y, centre[1]]));
        }
        if top <= FLOOR + 0.5 {
            return;
        }
        // Up the corners and round the top: the volume gravity acts in.
        let rise = ((top - FLOOR) * 0.5) as f32;
        let post = bar([W, rise, W]);
        let top_x = bar([half[0] as f32, W, W]);
        let top_z = bar([W, W, half[1] as f32]);
        for x in [lo[0], hi[0]] {
            for z in [lo[1], hi[1]] {
                self.showcase
                    .push(fixed(&post, [x, FLOOR + f64::from(rise), z]));
            }
        }
        for z in [lo[1], hi[1]] {
            self.showcase.push(fixed(&top_x, [centre[0], top, z]));
        }
        for x in [lo[0], hi[0]] {
            self.showcase.push(fixed(&top_z, [x, top, centre[1]]));
        }
    }

    /// A glowing sign at `position`, turned `yaw` (its text faces +Z before
    /// the turn).
    fn sign(&mut self, text: &str, position: [f64; 3], yaw: f32, radiance: [f32; 3]) {
        if let Some(mesh) = sign_mesh(text, 0.06, 0.05, radiance) {
            self.showcase.push(Prop {
                mesh: Arc::new(mesh),
                position: at(position),
                rotation: rotation_y(yaw),
                scale: 1.0,
            });
        }
    }

    fn prop(&mut self, mesh: &Arc<MeshData>, position: [f64; 3], yaw: f32, scale: f32) {
        self.props.push(Prop {
            mesh: Arc::clone(mesh),
            position: at(position),
            rotation: if yaw == 0.0 {
                IDENTITY_ROTATION
            } else {
                rotation_y(yaw)
            },
            scale,
        });
    }

    /// A character standing on the floor at (`x`, `z`), facing `towards`.
    fn character(&mut self, x: f64, z: f64, lift: f64, towards: [f64; 3]) {
        let position = [x, FLOOR + lift, z];
        self.prop(&character_mesh(), position, facing(position, towards), 1.0);
    }

    fn floor(&mut self) {
        // 38 m x 38 m, 2 m squares, reaching from 9 m behind the spawn point
        // to 29 m in front of it. A slab, not a plane: it also blocks light.
        let floor = textured(
            MeshData::cuboid([19.0, 0.1, 19.0], 4.0, paint([1.0; 3], 0.0, 0.85)),
            Texture::checkerboard(256, 2, [150, 154, 162], [62, 66, 76]),
        );
        self.prop(&floor, [0.0, FLOOR - 0.1, -10.0], 0.0, 1.0);
    }

    fn people(&mut self) {
        let spawn = [0.0, 0.0, 0.0];
        // One greeting the newcomer, one walking off towards the chart.
        self.character(-1.9, -3.6, 0.0, spawn);
        self.character(1.4, -6.6, 0.0, [2.2, 0.0, -30.0]);
        // One beside the material chart, for scale.
        self.character(5.6, -15.0, 0.0, spawn);
    }

    fn glass_corner(&mut self) {
        // Glass shows what is behind it, bent — so it stands in a corner of
        // two panels (UV grid at the back, checkerboard at the side) with a
        // striped pillar beside it: from the spawn point and from most of the
        // area there is something to see through it.
        // The renderer's refraction is screen-space: it bends the view ray at
        // the glass surface and looks the scene up two metres along it. Seen
        // steeply from above that lands far from what is really behind, so
        // the glass stands on pedestals at eye height, close in front of
        // panels much larger than itself (3.2 m square each).
        let back = textured(
            MeshData::cuboid([1.6, 1.6, 0.08], 3.2, paint([1.0; 3], 0.0, 0.7)),
            Texture::uv_grid(512),
        );
        let side = textured(
            MeshData::cuboid([0.08, 1.6, 1.6], 1.6, paint([1.0; 3], 0.0, 0.7)),
            Texture::checkerboard(256, 4, [236, 238, 242], [36, 150, 96]),
        );
        let pillar = textured(
            MeshData::cuboid([0.22, 1.3, 0.22], 0.44, paint([1.0; 3], 0.0, 0.6)),
            Texture::hazard_stripes(128, 2, [240, 196, 20], [24, 24, 26]),
        );
        let pedestal = Arc::new(MeshData::cuboid(
            [0.4, 0.5, 0.4],
            1.0,
            paint([0.12, 0.13, 0.15], 0.0, 0.7),
        ));
        self.prop(&back, [-6.4, FLOOR + 1.6, -7.9], 0.0, 1.0);
        self.prop(&side, [-8.08, FLOOR + 1.6, -6.3], 0.0, 1.0);
        self.prop(&pillar, [-4.4, FLOOR + 1.3, -7.9], 0.0, 1.0);
        let (cube, sphere) = ([-5.8, -6.6], [-7.0, -5.2]);
        for [x, z] in [cube, sphere] {
            self.prop(&pedestal, [x, FLOOR + 0.5, z], 0.0, 1.0);
        }
        self.bodies.push(Body {
            roughness: 0.05,
            transmission: 1.0,
            ..Body::cube(
                "glass cube",
                at([cube[0], FLOOR + 1.0 + 0.5, cube[1]]),
                0.5,
                [0.9, 0.95, 1.0],
            )
        });
        self.bodies.push(Body {
            roughness: 0.05,
            transmission: 1.0,
            ..Body::sphere(
                "glass sphere",
                at([sphere[0], FLOOR + 1.0 + 0.55, sphere[1]]),
                0.55,
                [0.92, 1.0, 0.95],
            )
        });
    }

    fn mirror_floor(&mut self) {
        // A polished slab let into the floor, its top 2 cm proud of it and its
        // underside inside the floor slab. A thin mesh, not a box instance: an
        // instance is a cube, and the 6 m cube this used to be hung below the
        // floor as a mirrored block that reflected the sky from underneath.
        self.prop(&mirror_mesh(), MIRROR_CENTRE, 0.0, 1.0);
        // Standing on it: the ship (hovering), a red plastic ball and a
        // character.
        self.prop(&ship_mesh(), [6.9, FLOOR + 1.0, -7.4], 0.7, 0.55);
        self.bodies.push(Body {
            roughness: 0.3,
            ..Body::sphere(
                "red ball",
                at([4.3, FLOOR + 0.42, -5.2]),
                0.4,
                [0.8, 0.08, 0.08],
            )
        });
        self.character(4.4, -8.6, 0.02, [0.0, 0.0, 0.0]);
        // A chrome ball beside it, on the checkered floor (which it reflects).
        self.bodies.push(Body {
            metallic: 1.0,
            roughness: 0.06,
            ..Body::sphere(
                "chrome ball",
                at([2.3, FLOOR + 0.6, -4.3]),
                0.6,
                [0.95, 0.95, 0.97],
            )
        });
    }

    fn texture_crates(&mut self) {
        // One 1.8 m crate per procedural texture, one repeat per face, stacked
        // three wide and two high right behind the mirror, which reflects them.
        let crate_of = |texture: Texture, metallic: f32, roughness: f32| {
            textured(
                MeshData::cuboid([0.9; 3], 1.8, paint([1.0; 3], metallic, roughness)),
                texture,
            )
        };
        let crates = [
            crate_of(Texture::brushed_metal(256, [214, 216, 222]), 0.6, 0.35),
            crate_of(
                Texture::hazard_stripes(256, 4, [240, 196, 20], [24, 24, 26]),
                0.0,
                0.6,
            ),
            crate_of(Texture::solid([40, 170, 160]), 0.0, 0.5),
            crate_of(
                Texture::checkerboard(256, 4, [230, 60, 50], [40, 60, 200]),
                0.0,
                0.7,
            ),
            crate_of(Texture::uv_grid(512), 0.0, 0.7),
            crate_of(Texture::hull_panels(256, [236, 238, 242]), 0.1, 0.5),
        ];
        for (i, mesh) in crates.iter().enumerate() {
            let (column, row) = ((i % 3) as f64, (i / 3) as f64);
            self.prop(
                mesh,
                [5.2 + 1.9 * column, FLOOR + 0.9 + 1.8 * row, -11.4],
                0.0,
                1.0,
            );
        }
    }

    fn material_chart(&mut self) {
        // Two steps, the metal row on the lower one and the dielectric row on
        // the upper one behind it; roughness runs from 0.05 (left) to 1.
        const COLUMNS: usize = 7;
        const RADIUS: f32 = 0.45;
        let step = |half_height: f32| {
            Arc::new(MeshData::cuboid(
                [4.5, half_height, 0.6],
                1.0,
                paint([0.16, 0.17, 0.2], 0.0, 0.8),
            ))
        };
        let rows: [(&'static str, f64, f64, [f32; 3], f32); 2] = [
            ("gold", 0.25, -15.2, [1.0, 0.78, 0.34], 1.0),
            ("plastic", 0.8, -16.4, [0.8, 0.08, 0.08], 0.0),
        ];
        for (name, half_height, z, albedo, metallic) in rows {
            self.prop(
                &step(half_height as f32),
                [0.0, FLOOR + half_height, z],
                0.0,
                1.0,
            );
            for column in 0..COLUMNS {
                let t = column as f32 / (COLUMNS - 1) as f32;
                let x = (column as f64 - (COLUMNS - 1) as f64 * 0.5) * 1.25;
                self.bodies.push(Body {
                    metallic,
                    roughness: 0.05 + 0.95 * t,
                    ..Body::sphere(
                        name,
                        at([x, FLOOR + 2.0 * half_height + f64::from(RADIUS), z]),
                        RADIUS,
                        albedo,
                    )
                });
            }
        }
    }

    fn shadow_theatre(&mut self) {
        // A white wall, three coloured lamps in front of it and things in
        // between: each caster throws a cyan, a magenta and a yellow shadow.
        let white = paint([0.55, 0.55, 0.55], 0.0, 0.9);
        let wall = Arc::new(MeshData::cuboid([4.4, 2.2, 0.15], 1.0, white));
        let pillar = Arc::new(MeshData::cuboid([0.18, 1.1, 0.18], 1.0, white));
        // Turned a little towards the spawn point.
        self.prop(&wall, [-9.4, FLOOR + 2.2, -17.4], 0.35, 1.0);
        self.character(-9.6, -14.8, 0.0, [0.0, 0.0, 0.0]);
        self.prop(&ship_mesh(), [-7.0, FLOOR + 1.9, -15.2], -0.5, 0.4);
        self.prop(&pillar, [-11.9, FLOOR + 1.1, -15.4], 0.0, 1.0);
        for (name, x, z, color) in [
            ("red lamp", -11.4, -11.4, [1.0, 0.12, 0.08]),
            ("green lamp", -9.2, -10.8, [0.1, 1.0, 0.15]),
            ("blue lamp", -7.0, -11.2, [0.12, 0.2, 1.0]),
        ] {
            let position = [x, FLOOR + 0.55, z];
            self.lights.push(light(position, color, 85.0));
            // The lamp around the light (a sphere that holds a light casts no
            // shadow from it).
            self.bodies.push(Body {
                emissive: color.map(|c: f32| c * 2.4 + 0.02),
                ..Body::sphere(name, at(position), 0.12, [0.0; 3])
            });
        }
    }

    fn beacon_and_pad(&mut self) {
        // An emissive cube on a dark pedestal, left of the middle.
        let pedestal = Arc::new(MeshData::cuboid(
            [0.3, 0.45, 0.3],
            1.0,
            paint([0.12, 0.13, 0.15], 0.0, 0.7),
        ));
        self.prop(&pedestal, [-2.6, FLOOR + 0.45, -10.2], 0.0, 1.0);
        self.bodies.push(Body {
            emissive: [3.4, 1.3, 0.2],
            ..Body::cube(
                "beacon",
                at([-2.6, FLOOR + 1.25, -10.2]),
                0.35,
                [1.0, 0.5, 0.1],
            )
        });
        // The full-size ship on a striped pad, out to the right.
        let pad = textured(
            MeshData::cuboid([3.2, 0.12, 3.2], 1.6, paint([1.0; 3], 0.0, 0.7)),
            Texture::hazard_stripes(256, 4, [240, 196, 20], [24, 24, 26]),
        );
        self.prop(&pad, [14.0, FLOOR + 0.12, -15.5], 0.0, 1.0);
        self.prop(&ship_mesh(), [14.0, FLOOR + 1.5, -15.5], 0.9, 1.0);
    }

    fn lighting(&mut self) {
        // Key: warm, above the middle of the area, inside its lamp.
        let key = [0.8, 4.4, -6.8];
        self.lights.push(light(key, [1.0, 0.95, 0.86], 130.0));
        self.bodies.push(Body {
            emissive: [6.0, 5.4, 4.2],
            ..Body::sphere("key lamp", at(key), 0.14, [1.0; 3])
        });
        // The sky ([`ENVIRONMENT`]) is the fill: every shadow has its cool
        // light in it. Two warm lamps are left to answer it — one behind the
        // spawn point, on the faces a newcomer sees, and one over the mirror,
        // the crates and the pad. (The cool fills that used to stand at the
        // front left and behind the chart are what the sky now does.)
        self.lights
            .push(light([2.5, 3.8, 8.5], [1.0, 0.96, 0.92], 70.0));
        self.lights
            .push(light([10.0, 4.8, -6.5], [1.0, 0.9, 0.78], 110.0));
    }

    fn kestrel_apron(&mut self) {
        // The apron: a dark slab under the ship with room to walk round it,
        // its top a step above the floor, edged with hazard stripes. It
        // reaches out behind the floor, which ends under the ship.
        let [half_x, half_z] = KESTREL_APRON_HALF;
        let (x, z) = (self.kestrel.position[0], self.kestrel.position[2]);
        let apron = textured(
            MeshData::cuboid(
                [half_x as f32, 0.15, half_z as f32],
                8.0,
                paint([1.0; 3], 0.0, 0.8),
            ),
            Texture::checkerboard(256, 2, [84, 88, 96], [68, 72, 80]),
        );
        self.prop(&apron, [x, KESTREL_APRON_TOP - 0.15, z], 0.0, 1.0);
        let stripes = || Texture::hazard_stripes(128, 2, [240, 196, 20], [24, 24, 26]);
        let edge = paint([1.0; 3], 0.0, 0.7);
        let (width, inset) = (0.3f32, 0.3f64);
        let long = textured(
            MeshData::cuboid([half_x as f32, 0.02, width], 0.6, edge),
            stripes(),
        );
        let short = textured(
            MeshData::cuboid([width, 0.02, half_z as f32 - 2.0 * width], 0.6, edge),
            stripes(),
        );
        for side in [-1.0, 1.0] {
            let along = z + side * (half_z - inset);
            self.prop(&long, [x, KESTREL_APRON_TOP, along], 0.0, 1.0);
            let across = x + side * (half_x - inset);
            self.prop(&short, [across, KESTREL_APRON_TOP, z], 0.0, 1.0);
        }
        // The renderer has eight lights and the stations use six. The seventh
        // is a floodlight on a mast on the apron, abeam of the ship on the
        // side a newcomer sees, beyond the port radiator's tip: near enough
        // to light the flank, the belly and the gear from bow to stern, and
        // weak again by the time it reaches the stations. (The eighth is the
        // cabin lamp: see `scene_for`.)
        const MAST: f32 = 4.2;
        let mast = Arc::new(MeshData::cuboid(
            [0.06, MAST / 2.0, 0.06],
            1.0,
            paint([0.08, 0.09, 0.1], 0.0, 0.6),
        ));
        let lamp = self.kestrel.to_scene([-10.0, MAST - GEAR_DROP as f32, 0.5]);
        self.lights.push(light(lamp, [1.0, 0.96, 0.9], 230.0));
        self.bodies.push(Body {
            emissive: [6.0, 5.6, 5.0],
            ..Body::sphere("flood lamp", at(lamp), 0.16, [1.0; 3])
        });
        // The mast ends just under the lamp.
        let foot = [lamp[0], lamp[1] - f64::from(MAST) / 2.0 - 0.16, lamp[2]];
        self.prop(&mast, foot, 0.0, 1.0);
    }

    fn backdrop(&mut self) {
        let mut dice = Dice(0x5eed_d0e5);
        // Stars all round, pale dots a few kilometres out (the planets, the
        // Moon and the Sun are the app's: they move, `crate::scenes::World::sky`).
        const STARS: usize = 200;
        for _ in 0..STARS {
            // Uniform on the sphere.
            let y: f64 = dice.between(-1.0, 1.0);
            let around = dice.between(0.0, std::f64::consts::TAU);
            let flat = (1.0 - y * y).sqrt();
            let distance = dice.between(4_500.0, 7_000.0);
            let size = dice.between(7.0, 18.0) as f32;
            let glow = dice.between(0.4, 1.6) as f32;
            self.bodies.push(Body {
                emissive: [glow, glow, glow * 1.1],
                ..Body::sphere(
                    "star",
                    at([
                        distance * flat * around.cos(),
                        distance * y,
                        distance * flat * around.sin(),
                    ]),
                    size,
                    [0.0; 3],
                )
            });
        }
    }

    /// The props, in draw order.
    #[must_use]
    pub fn props(&self) -> &[Prop] {
        &self.props
    }

    /// The physics showcase's statics (from the shared scene definition) and
    /// marks, in draw order (see the module docs, "The physics showcase").
    #[must_use]
    pub fn showcase(&self) -> &[Prop] {
        &self.showcase
    }

    /// The spheres and boxes.
    #[must_use]
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// The point lights that are always on.
    #[must_use]
    pub fn lights(&self) -> &[SceneLight] {
        &self.lights
    }

    /// The lamp inside the Kestrel ([`CABIN_LAMP`]): lit for a camera in the
    /// ship.
    #[must_use]
    pub fn cabin_light(&self) -> &SceneLight {
        &self.cabin_light
    }

    /// Where the Kestrel stands ([`KESTREL_BERTH`]).
    #[must_use]
    pub fn kestrel(&self) -> &Berth {
        &self.kestrel
    }

    /// The demo area as `camera` sees it on a GPU ([`DemoArea::scene_for`]
    /// with [`Detail::Hardware`]).
    #[must_use]
    pub fn scene(&self, camera: &CameraPose) -> Scene {
        self.scene_for(camera, Detail::Hardware)
    }

    /// The demo area as `camera` sees it, under its sky ([`ENVIRONMENT`]), on
    /// a renderer of this `detail`.
    #[must_use]
    pub fn scene_for(&self, camera: &CameraPose, detail: Detail) -> Scene {
        let instances = self.bodies.iter().map(|b| b.instance(camera)).collect();
        // The lamps light the area; from further away than this nothing they
        // light can be told apart, and a shadow ray from a planet's surface
        // to a lamp half a million kilometres away would wade through the
        // whole area's geometry at the end (a frame on the Moon took seconds
        // on the High preset): out there the Sun is the only light.
        let in_range = WorldPos::distance(WorldPos::ZERO, camera.position) < LAMP_RANGE;
        let mut lights = if in_range {
            self.lights.clone()
        } else {
            Vec::new()
        };
        // The cabin lamp lights a deck that cannot be seen from outside the
        // hull (the canopy is opaque): it is a light of the frame — a shadow
        // ray for every pixel — only for a camera that is in the ship.
        let to_the_ship = self
            .kestrel
            .distance(crate::scenes::metres(camera.position));
        if to_the_ship < CABIN_LIGHT_RANGE {
            lights.push(self.cabin_light);
        }
        let mut scene = Scene::from_instances(*camera, instances)
            .with_lights(lights)
            .with_environment(ENVIRONMENT);
        // The showcase first: its floor mats are then in the depth buffer
        // before the floor they lie on, which is not shaded twice under them.
        scene.models = self
            .showcase
            .iter()
            .chain(&self.props)
            .map(|p| p.instance(camera))
            .collect();
        // After the showcase and the props: the ship, in the detail this
        // camera calls for.
        scene.models.push(self.kestrel.instance(camera, detail));
        scene
    }
}

impl Default for DemoArea {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::{metres, World};
    use space_render::{SHAPE_BOX, SHAPE_SPHERE};
    use space_world::WorldPos;

    fn distance(position: WorldPos) -> f64 {
        WorldPos::distance(WorldPos::ZERO, position)
    }

    fn spawn_camera() -> CameraPose {
        World::default().camera(WorldPos::ZERO, [0.0, 0.0, -1.0], 16.0 / 9.0)
    }

    /// Every number in the area is finite and inside its bounds: the things a
    /// player walks among within `AREA_RADIUS`, the backdrop within
    /// `BACKDROP_RADIUS`, and everything within the renderer's budgets.
    #[test]
    fn the_area_is_finite_and_bounded() {
        let area = DemoArea::shared();
        assert!(area.lights().len() <= 8, "the renderer has 8 lights");
        assert!(area.bodies().len() <= space_world::MAX_SELECTED);
        for body in area.bodies() {
            let far = distance(body.position);
            assert!(far.is_finite() && far < BACKDROP_RADIUS, "{}", body.name);
            assert!(
                body.radius.is_finite() && body.radius > 0.0,
                "{}",
                body.name
            );
            assert!(
                body.shape == SHAPE_SPHERE || body.shape == SHAPE_BOX,
                "{}",
                body.name
            );
            let material = [body.metallic, body.roughness, body.transmission];
            assert!(
                material.iter().all(|v| (0.0..=1.0).contains(v)),
                "{}",
                body.name
            );
            assert!(
                body.albedo
                    .iter()
                    .chain(&body.emissive)
                    .all(|v| v.is_finite() && *v >= 0.0),
                "{}",
                body.name
            );
            let backdrop = body.name == "star";
            assert_eq!(far > AREA_RADIUS, backdrop, "{} at {far} m", body.name);
        }
        for prop in area.props() {
            assert!(distance(prop.position) < 1_000.0);
            assert!(prop.scale.is_finite() && prop.scale > 0.0);
            assert!(prop.rotation.iter().flatten().all(|v| v.is_finite()));
            prop.mesh.validate().expect("a valid mesh");
        }
        for light in area.lights() {
            assert!(distance(light.position) < AREA_RADIUS);
            assert!(light.intensity.is_finite() && light.intensity > 0.0);
            // Above the floor: a light under it would be shadowed everywhere.
            assert!(metres(light.position)[1] > FLOOR + 0.3);
        }
    }

    /// The area shows what it is there to show.
    #[test]
    fn the_area_has_every_feature_on_display() {
        let area = DemoArea::shared();
        let bodies = area.bodies();
        let named = |name: &str| bodies.iter().filter(|b| b.name == name).count();
        // Reflection: a smooth metal slab whose top is just above the floor
        // and whose underside stays inside the floor slab (seen from below the
        // floor there is no mirrored block hanging down).
        let mirror_mesh = mirror_mesh();
        let mirror = area
            .props()
            .iter()
            .find(|p| Arc::ptr_eq(&p.mesh, &mirror_mesh))
            .expect("mirror");
        let material = mirror.mesh.parts[0].material;
        assert!(material.metallic == 1.0 && material.roughness < 0.1);
        let centre = metres(mirror.position)[1];
        let top = centre + f64::from(mirror.mesh.bounds_max[1]);
        let bottom = centre + f64::from(mirror.mesh.bounds_min[1]);
        assert!(top > FLOOR && top < FLOOR + 0.05, "no z-fighting: {top}");
        let floor = &area.props()[0];
        let floor_bottom = metres(floor.position)[1] + f64::from(floor.mesh.bounds_min[1]);
        assert!(
            bottom > floor_bottom,
            "the mirror's underside {bottom} pokes out of the floor's {floor_bottom}"
        );
        // Refraction: glass, a cube and a sphere.
        let glass: Vec<_> = bodies.iter().filter(|b| b.transmission > 0.0).collect();
        assert_eq!(glass.len(), 2);
        assert!(glass.iter().any(|b| b.shape == SHAPE_BOX));
        assert!(glass.iter().any(|b| b.shape == SHAPE_SPHERE));
        // Materials: a metal and a dielectric row, smooth to rough.
        for (name, metallic) in [("gold", 1.0), ("plastic", 0.0)] {
            let row: Vec<_> = bodies.iter().filter(|b| b.name == name).collect();
            assert_eq!(row.len(), 7, "{name}");
            assert!(row.iter().all(|b| b.metallic == metallic));
            assert!(row.windows(2).all(|w| w[0].roughness < w[1].roughness));
            assert!(row[0].roughness < 0.1 && row[6].roughness > 0.95);
        }
        // Lights: white ones and a red, a green and a blue one, each in a lamp.
        let lights = area.lights();
        assert_eq!(
            lights.len(),
            7,
            "a key, two fills, the three lamps and the ship's floodlight"
        );
        // The rest of the light is space's: nothing painted, a faint cool
        // ambient light, brighter overhead than below, and much dimmer on the
        // wall of the shadow theatre than one of its lamps is (or the shadows
        // wash out). (The Sun's light is the app's: it moves.)
        assert!(!ENVIRONMENT.background && !ENVIRONMENT.is_none());
        let [zenith, horizon, ground] = ENVIRONMENT.colors();
        for c in 0..3 {
            assert!(zenith[c] >= horizon[c] && horizon[c] > ground[c]);
        }
        assert!(zenith[2] > zenith[0], "cool");
        let on_the_wall = ENVIRONMENT.ambient([0.0, 0.0, 1.0]);
        let wall = area
            .props()
            .iter()
            .find(|p| p.mesh.bounds_max[0] > 4.0 && p.mesh.bounds_max[1] > 2.0)
            .expect("the white wall");
        let lamp = lights
            .iter()
            .find(|l| l.color[0] > 0.9 && l.color[1] < 0.5)
            .expect("the red lamp");
        let reach = WorldPos::distance(lamp.position, wall.position) as f32;
        let from_the_lamp = lamp.intensity / (reach * reach) / std::f32::consts::PI;
        assert!(
            on_the_wall
                .iter()
                .all(|c| *c > 0.02 && *c * 2.0 < from_the_lamp),
            "sky {on_the_wall:?} against a lamp's {from_the_lamp}"
        );
        for channel in 0..3 {
            let pure = lights
                .iter()
                .any(|l| (0..3).all(|c| (l.color[c] > 0.9) == (c == channel)));
            assert!(pure, "no light of pure channel {channel}");
        }
        for name in [
            "red lamp",
            "green lamp",
            "blue lamp",
            "key lamp",
            "flood lamp",
        ] {
            let lamp = bodies.iter().find(|b| b.name == name).expect(name);
            let inside = lights
                .iter()
                .any(|l| WorldPos::distance(l.position, lamp.position) < f64::from(lamp.radius));
            assert!(inside, "{name} holds no light");
        }
        // Emissive: the beacon cube.
        let beacon = bodies.iter().find(|b| b.name == "beacon").expect("beacon");
        assert!(beacon.emissive.iter().sum::<f32>() > 1.0);
        // Backdrop: stars; no towers, and no painted moon (the real Moon is
        // the app's, in the sky that moves).
        assert!(named("star") >= 100);
        assert_eq!(named("tower beacon") + named("moon") + named("moon sea"), 0);

        // Models: the character several times, the ship more than once, and
        // textures on the floor, the crates and the panels.
        let props = area.props();
        let count =
            |mesh: &Arc<MeshData>| props.iter().filter(|p| Arc::ptr_eq(&p.mesh, mesh)).count();
        assert!(count(&character_mesh()) >= 4);
        assert!(count(&ship_mesh()) >= 2);
        let textures: usize = {
            let mut seen: Vec<*const MeshData> = Vec::new();
            let mut total = 0;
            for prop in props {
                let ptr = Arc::as_ptr(&prop.mesh);
                if !seen.contains(&ptr) {
                    seen.push(ptr);
                    total += prop.mesh.textures.len();
                }
            }
            total
        };
        assert!(textures >= 10, "{textures} textures");
        // Some characters face the spawn point and some face away.
        let character = character_mesh();
        let mut towards = 0;
        let mut away = 0;
        for prop in props.iter().filter(|p| Arc::ptr_eq(&p.mesh, &character)) {
            // The model faces -Z; row 2 of the rotation gives where -Z goes.
            let forward = [-prop.rotation[0][2], -prop.rotation[2][2]];
            let p = metres(prop.position);
            let to_spawn = [-p[0], -p[2]];
            let dot = f64::from(forward[0]) * to_spawn[0] + f64::from(forward[1]) * to_spawn[1];
            if dot > 0.0 {
                towards += 1;
            } else {
                away += 1;
            }
        }
        assert!(towards >= 2 && away >= 1, "{towards} towards, {away} away");
    }

    /// The meshes are built once: every frame's models are the same `Arc`s
    /// (the renderer's cache is keyed by them), wherever the camera is.
    #[test]
    fn the_static_meshes_are_reused_from_frame_to_frame() {
        let area = DemoArea::shared();
        assert!(std::ptr::eq(area, DemoArea::shared()));
        let first = area.scene(&spawn_camera());
        let moved = World::default().camera(at([3.0, 1.0, -5.0]), [0.6, 0.0, -0.8], 1.0);
        let second = World::default().build(&moved);
        // The props, and the Kestrel after them.
        assert_eq!(
            first.models.len(),
            area.props().len() + area.showcase().len() + 1
        );
        assert_eq!(first.models.len(), second.models.len());
        assert_eq!(first.instances.len(), area.bodies().len());
        for (a, b) in first.models.iter().zip(&second.models) {
            assert!(Arc::ptr_eq(&a.mesh, &b.mesh));
        }
        // On a GPU the ship is the full model from anywhere in the area; a
        // software rasteriser is given the proxy until the camera has walked
        // up to it. Each is the same `Arc` every frame, and the props are
        // untouched by the switch.
        let (proxy, full) = (crate::kestrel::proxy_mesh(), crate::kestrel::full_mesh());
        assert!(!Arc::ptr_eq(&full, &proxy));
        let ship = |scene: &Scene| Arc::clone(&scene.models.last().expect("the ship").mesh);
        assert!(Arc::ptr_eq(&ship(&first), &full) && Arc::ptr_eq(&ship(&second), &full));
        let slow = |camera: &CameraPose| area.scene_for(camera, Detail::Software);
        let near = |x: f64| World::default().camera(at([x, 0.5, -21.0]), [0.0, 0.0, -1.0], 1.0);
        let from_spawn = slow(&spawn_camera());
        assert!(Arc::ptr_eq(&ship(&from_spawn), &proxy));
        assert!(Arc::ptr_eq(&ship(&slow(&moved)), &proxy));
        assert!(Arc::ptr_eq(&ship(&slow(&near(-2.0))), &full));
        assert!(Arc::ptr_eq(&ship(&slow(&near(3.0))), &full));
        assert_eq!(from_spawn.models.len(), first.models.len());
        for (a, b) in first
            .models
            .iter()
            .zip(&from_spawn.models)
            .take(area.props().len())
        {
            assert!(Arc::ptr_eq(&a.mesh, &b.mesh));
        }
        // The world passes its renderer's detail on.
        let software = World::default().with_detail(Detail::Software);
        assert!(Arc::ptr_eq(&ship(&software.build(&spawn_camera())), &proxy));
        assert!(Arc::ptr_eq(
            &ship(&World::default().build(&spawn_camera())),
            &full
        ));
        // Fewer meshes than props: the crates, characters and ships share
        // theirs.
        let props = area.props().len();
        let mut meshes: Vec<*const MeshData> = first
            .models
            .iter()
            .skip(area.showcase().len())
            .take(props)
            .map(|m| Arc::as_ptr(&m.mesh))
            .collect();
        meshes.sort_unstable();
        meshes.dedup();
        assert!(meshes.len() < props, "{} of {props}", meshes.len());
        // The scene moves with the camera, by exactly the camera's move.
        let (a, b) = (first.models[0].position, second.models[0].position);
        assert!((a[0] - b[0] - 3.0).abs() < 1e-4 && (a[2] - b[2] + 5.0).abs() < 1e-4);
    }

    /// From the spawn point the floor is under the player's feet and the
    /// stations are in front: nothing stands on the spawn point itself.
    #[test]
    fn the_kestrel_stands_on_its_apron_clear_of_the_stations() {
        use crate::kestrel::{BOUNDS_MAX, BOUNDS_MIN};
        let area = DemoArea::shared();
        let berth = area.kestrel();
        assert_eq!(*berth, KESTREL_BERTH);
        assert_eq!(berth.asset, "ships/kestrel");

        // The apron: the one big flat prop under the ship, a step above the
        // floor.
        let apron = area
            .props()
            .iter()
            .find(|p| (f64::from(p.mesh.bounds_max[0]) - KESTREL_APRON_HALF[0]).abs() < 1e-3)
            .expect("the apron");
        let centre = metres(apron.position);
        let top = centre[1] + f64::from(apron.mesh.bounds_max[1]);
        assert!((top - KESTREL_APRON_TOP).abs() < 1e-4, "{top}");
        assert!(top > FLOOR && top < FLOOR + 0.3, "a step, not a wall");
        let over_the_apron = |point: [f64; 3], margin: f64| {
            (point[0] - centre[0]).abs() < KESTREL_APRON_HALF[0] - margin
                && (point[2] - centre[2]).abs() < KESTREL_APRON_HALF[1] - margin
        };

        // On its gear, on the apron: the lowest point of the hull's bounds is
        // the apron's top, and so are the three wheels' contact points.
        let (min, max) = berth.bounds();
        assert!(
            (min[1] - top).abs() < 1e-4,
            "{} above the apron",
            min[1] - top
        );
        assert!(
            (max[1] - top - 5.6436).abs() < 1e-3,
            "5.64 m tall on its gear"
        );
        for wheel in [[0.0, -3.3, -7.3], [-1.925, -3.3, 3.3], [1.925, -3.3, 3.3]] {
            let contact = berth.to_scene(wheel);
            assert!((contact[1] - top).abs() < 1e-4, "{contact:?}");
            assert!(over_the_apron(contact, 2.0), "{contact:?}");
        }
        // All of it over the apron, inside the hazard edge, and all of it —
        // apron included — inside the area.
        for corner in 0..4 {
            let x = if corner & 1 == 0 { min[0] } else { max[0] };
            let z = if corner & 2 == 0 { min[2] } else { max[2] };
            assert!(over_the_apron([x, 0.0, z], 0.6), "({x}, {z})");
            let far = (x * x + z * z).sqrt();
            assert!(far < AREA_RADIUS, "the ship reaches {far} m out");
            let rim = [
                centre[0] + (x - centre[0]).signum() * KESTREL_APRON_HALF[0],
                centre[2] + (z - centre[2]).signum() * KESTREL_APRON_HALF[1],
            ];
            let far = (rim[0] * rim[0] + rim[1] * rim[1]).sqrt();
            assert!(far < AREA_RADIUS, "the apron reaches {far} m out");
        }
        // The nose is to the left, turned towards the newcomer, who looks at
        // the port side.
        let nose = berth.to_scene([0.0, 0.0, BOUNDS_MIN[2]]);
        let tail = berth.to_scene([0.0, 0.0, BOUNDS_MAX[2]]);
        assert!(nose[0] < -10.0 && tail[0] > 10.0 && nose[2] > tail[2] + 3.0);
        let port = berth.to_scene([BOUNDS_MIN[0], 0.0, 0.0]);
        assert!(port[2] > berth.position[2] + 5.0, "port is the near side");

        // Clear of everything else: no sphere or box of the area reaches into
        // the hull's bounds, and no prop has its middle or a corner in them.
        for body in area.bodies() {
            let gap = berth.distance(metres(body.position)) - f64::from(body.radius);
            assert!(gap > 0.5, "{} is {gap} m from the ship", body.name);
        }
        for (i, prop) in area.props().iter().enumerate() {
            let origin = metres(prop.position);
            let (low, high) = (prop.mesh.bounds_min, prop.mesh.bounds_max);
            let mut samples = vec![[0.0f32; 3]];
            for corner in 0..8 {
                samples.push([0, 1, 2].map(|k| {
                    if (corner >> k) & 1 == 0 {
                        low[k]
                    } else {
                        high[k]
                    }
                }));
            }
            for sample in samples {
                let point = [0, 1, 2].map(|row| {
                    let turned: f32 = (0..3).map(|k| prop.rotation[row][k] * sample[k]).sum();
                    origin[row] + f64::from(turned * prop.scale)
                });
                let gap = berth.distance(point);
                assert!(gap > 0.05, "prop {i} reaches into the ship at {point:?}");
            }
        }
        // The stations are in front of it: nothing of theirs is behind the
        // apron's near edge.
        let near_edge = centre[2] + KESTREL_APRON_HALF[1];
        for body in area.bodies() {
            let p = metres(body.position);
            let station = distance(body.position) < AREA_RADIUS && body.name != "flood lamp";
            assert!(
                !station || p[2] - f64::from(body.radius) > near_edge + 1.0,
                "{}",
                body.name
            );
        }

        // The spawn point, and every station (they end 6 m in front of the
        // apron), are outside the range in which a software rasteriser is
        // given the full model, and inside a GPU's: the ordinary views of the
        // area cost the one a proxy and show the other the real ship.
        let from_spawn = berth.distance([0.0; 3]);
        assert!(
            from_spawn > 20.0 && from_spawn < Detail::Hardware.full_detail_range() - 20.0,
            "the spawn point is {from_spawn} m from the hull"
        );
        for (station, place) in [
            ("behind the chart", [0.0, 0.0, -16.0]),
            ("in the shadow theatre", [-9.6, 0.0, -14.8]),
            ("on the small ship's pad", [14.0, 0.0, -15.5]),
        ] {
            let away = berth.distance(place);
            assert!(
                away > Detail::Software.full_detail_range() + 1.0,
                "{station}: {away} m"
            );
        }

        // Its lights. The floodlight is in its lamp, abeam on the port side
        // and clear of the hull; none of the lights that are always on is
        // inside the ship.
        let flood = area
            .bodies()
            .iter()
            .find(|b| b.name == "flood lamp")
            .expect("the flood lamp");
        let local = berth.to_model(metres(flood.position));
        assert!(local[0] < f64::from(BOUNDS_MIN[0]) - 2.0, "{local:?}");
        assert!(local[2].abs() < 3.0 && local[1] > -1.0, "{local:?}");
        for light in area.lights() {
            assert!(berth.distance(metres(light.position)) > 2.0);
        }
        // The cabin lamp hangs inside, under the ceiling of the main deck. It
        // is the eighth light of a frame whose camera is in the ship, and of
        // no other.
        let cabin = area.cabin_light();
        assert_eq!(berth.distance(metres(cabin.position)), 0.0);
        let hangs = berth.to_model(metres(cabin.position));
        assert!(
            hangs[1] > 0.5 && hangs[1] < 1.45,
            "under the ceiling: {hangs:?}"
        );
        assert!(cabin.intensity < 20.0, "a cabin lamp, not a floodlight");
        let lights_at = |point: [f64; 3]| {
            let camera = World::default().camera(at(point), [0.0, 0.0, -1.0], 1.0);
            area.scene(&camera).lights.len()
        };
        assert_eq!(lights_at([0.0; 3]), 7, "from the spawn point");
        assert_eq!(lights_at([0.0, 0.5, -20.0]), 7, "on the apron");
        let in_the_crew_quarters = berth.to_scene([0.0, 0.9, 0.55]);
        assert_eq!(lights_at(in_the_crew_quarters), 8);
        assert_eq!(lights_at(berth.to_scene(crate::kestrel::PILOT_EYE)), 8);
        assert!(
            lights_at(in_the_crew_quarters) <= 8,
            "the renderer has 8 lights"
        );
    }

    #[test]
    fn the_spawn_point_is_clear_and_on_the_floor() {
        let area = DemoArea::shared();
        for body in area.bodies() {
            let clear = distance(body.position) - f64::from(body.radius);
            assert!(clear > 2.0, "{} at {clear} m", body.name);
        }
        let floor = &area.props()[0];
        let p = metres(floor.position);
        let top = p[1] + f64::from(floor.mesh.bounds_max[1]);
        assert!((top - FLOOR).abs() < 1e-4, "{top}");
        assert!(floor.mesh.bounds_min[0] + (p[0] as f32) < -10.0);
        assert!(
            floor.mesh.bounds_max[2] + (p[2] as f32) > 5.0,
            "floor behind the spawn"
        );
        assert_eq!(World::default().floor(), Some(FLOOR));
        // The same build twice is the same area (the arrangement is not random).
        let again = DemoArea::new();
        assert_eq!(again.bodies().len(), area.bodies().len());
        for (a, b) in again.bodies().iter().zip(area.bodies()) {
            assert_eq!(a.position, b.position);
        }
    }
}
