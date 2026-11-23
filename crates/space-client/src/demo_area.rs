//! The demo area: one compact place around the spawn point that shows
//! everything the renderer can do, so a person or an agent can check it at a
//! glance.
//!
//! Scene space: the spawn point (a player's eye) is the origin, +Y is up and a
//! freshly spawned player looks down -Z. The floor's top is
//! [`EYE_HEIGHT`] below. Seen from the spawn point:
//!
//! ```text
//!                         far backdrop: beacon towers out to ~1 km, a moon, stars
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
//! * **environment**: a dusk sky ([`ENVIRONMENT`]) — the ambient light in
//!   every shadow, what the chrome ball and the smooth end of the gold row
//!   reflect, and what the glass shows where nothing stands behind it;
//! * **emissive**: the lamps around the lights, a glowing beacon cube, the
//!   beacons on the towers;
//! * **models**: the ship (small over the mirror, full size on its pad) and
//!   the placeholder character, some facing the spawn point and some away;
//! * **depth, fog, tone mapping**: self-lit towers receding to ~1 km into
//!   the haze of the horizon, a moon and stars above.
//!
//! The sky is painted ([`Environment::background`]): its horizon is the
//! renderer's fog colour, so what the fog takes fades *into the sky* instead
//! of glowing against black, and the polished metal reflects the same sky
//! that is seen behind it. The ambient light comes from that sky as well;
//! the point lights only model the shapes and cast the shadows, which is why
//! there are six of them and not eight.
//!
//! The meshes are built once ([`DemoArea::shared`]) and every frame reuses
//! their `Arc`s, which is what keeps them on the GPU.

use std::sync::{Arc, OnceLock};

use space_render::quality::FOG_COLOR;
use space_render::{
    rotation_y, CameraPose, Environment, MeshData, MeshMaterial, Scene, SceneLight, Texture,
    IDENTITY_ROTATION,
};

use crate::scenes::{at, character_mesh, ship_mesh, Body, Prop, EYE_HEIGHT};

/// The height of the floor's top, metres (scene space).
pub const FLOOR: f64 = -(EYE_HEIGHT as f64);

/// Everything a player walks among lies within this distance of the spawn
/// point, metres; only the backdrop is further out.
pub const AREA_RADIUS: f64 = 40.0;

/// Nothing at all is further from the spawn point than this, metres.
pub const BACKDROP_RADIUS: f64 = 8_000.0;

/// The demo area's surroundings: a clear sky at dusk, painted behind
/// everything.
///
/// * The **horizon** has the hue of the renderer's fog, so what the fog
///   takes fades towards the sky behind it: haze, not a glow against black.
/// * The **zenith** is a much darker blue: the picture keeps a gradient, and
///   the moon and the stars still stand out overhead.
/// * The **ground** is dark: seen from the platform's edge, the haze thickens
///   into night below.
/// * The **intensity** is one half. At full strength the sky is a bright
///   day: its ambient light on a wall is then as strong as one of the
///   coloured lamps, and their shadows wash out to pastel. At half, a shadow
///   is a dim blue-grey — never black — the far corners of the floor are
///   still plainly a floor, and the red, green and blue lamps own their wall.
pub const ENVIRONMENT: Environment =
    Environment::gradient([0.045, 0.085, 0.22], FOG_COLOR, [0.05, 0.06, 0.08])
        .with_intensity(0.5)
        .with_background(true);

/// The demo area's content: props (mesh models), bodies (spheres and boxes)
/// and lights at fixed places.
#[derive(Debug)]
pub struct DemoArea {
    props: Vec<Prop>,
    bodies: Vec<Body>,
    lights: Vec<SceneLight>,
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
            bodies: Vec::new(),
            lights: Vec::new(),
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
        area.backdrop();
        area
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
        // A polished slab let into the floor, its top 2 cm proud of it (a box
        // instance is a cube: the rest of it is below the floor).
        self.bodies.push(Body {
            metallic: 1.0,
            roughness: 0.04,
            ..Body::cube(
                "mirror",
                at([6.0, FLOOR + 0.02 - 3.0, -7.0]),
                3.0,
                [0.9, 0.92, 0.95],
            )
        });
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

    fn backdrop(&mut self) {
        let mut dice = Dice(0x5eed_d0e5);
        // Towers: dark, faintly self-lit slabs with a beacon on top, in a wide
        // arc ahead and to the sides, from 140 m to 1 km. Nothing lights them
        // out there but themselves; the fog takes them one after another.
        let glow = MeshMaterial {
            base_color: [0.2, 0.22, 0.28],
            metallic: 0.0,
            roughness: 0.9,
            emissive: [0.014, 0.02, 0.038],
            base_color_texture: None,
        };
        let towers = [
            Arc::new(MeshData::cuboid([1.2, 16.0, 1.2], 4.0, glow)),
            Arc::new(MeshData::cuboid([0.9, 24.0, 0.9], 4.0, glow)),
            Arc::new(MeshData::cuboid([2.0, 10.0, 1.6], 4.0, glow)),
        ];
        const TOWERS: usize = 24;
        for i in 0..TOWERS {
            let t = i as f64 / (TOWERS - 1) as f64;
            // Bearing from straight ahead, radians: about -110..110 degrees.
            let bearing = (t * 2.0 - 1.0) * 1.9 + dice.between(-0.04, 0.04);
            let distance = 140.0 * 7.0f64.powf(dice.roll());
            let mesh = &towers[i % towers.len()];
            let scale = dice.between(0.8, 1.5) * (distance / 300.0).clamp(1.0, 2.4);
            let (x, z) = (distance * bearing.sin(), -distance * bearing.cos());
            let half_height = f64::from(mesh.bounds_max[1]) * scale;
            // Standing on the plane of the floor (well outside the floor).
            let base = FLOOR - 1.0;
            self.prop(
                mesh,
                [x, base + half_height, z],
                dice.between(0.0, 1.5) as f32,
                scale as f32,
            );
            let warm = dice.roll() < 0.6;
            self.bodies.push(Body {
                emissive: if warm {
                    [5.0, 1.0, 0.3]
                } else {
                    [0.6, 2.2, 5.0]
                },
                ..Body::sphere(
                    "tower beacon",
                    at([x, base + 2.0 * half_height + 0.5 * scale, z]),
                    (0.5 * scale) as f32,
                    [0.0; 3],
                )
            });
        }
        // A moon, high ahead to the right: self-lit (no light reaches it), with
        // darker seas — smaller spheres let into its near side — and far
        // enough away to be tinted by the fog.
        let moon = [225.0, 150.0, -540.0];
        const MOON_RADIUS: f64 = 34.0;
        self.bodies.push(Body {
            emissive: [0.5, 0.48, 0.44],
            roughness: 1.0,
            ..Body::sphere("moon", at(moon), MOON_RADIUS as f32, [0.0; 3])
        });
        let range = moon.iter().map(|c| c * c).sum::<f64>().sqrt();
        let towards_spawn = moon.map(|c| -c / range);
        for (across, up, radius) in [
            (-0.38, 0.30, 12.0),
            (0.30, 0.42, 8.0),
            (0.12, -0.36, 10.0),
            (-0.48, -0.30, 5.0),
            (0.55, -0.02, 6.0),
        ] {
            // A direction on the near side: `towards_spawn` tilted across
            // (about the vertical) and up.
            let (sin, cos) = f64::sin_cos(across);
            let flat = [
                towards_spawn[0] * cos - towards_spawn[2] * sin,
                towards_spawn[1] + up,
                towards_spawn[0] * sin + towards_spawn[2] * cos,
            ];
            let length = flat.iter().map(|c| c * c).sum::<f64>().sqrt();
            // Let in so far that only a shallow cap shows.
            let depth = MOON_RADIUS - 0.82 * radius;
            let centre = [0, 1, 2].map(|i| moon[i] + flat[i] / length * depth);
            self.bodies.push(Body {
                emissive: [0.3, 0.29, 0.275],
                roughness: 1.0,
                ..Body::sphere("moon sea", at(centre), radius as f32, [0.0; 3])
            });
        }
        // Stars overhead, further than the fog reaches: pale dots. None in
        // the haze of the horizon or below it — against the painted sky a
        // star down there reads as a speck of dust in front of the picture —
        // and the lower ones are the fainter ones.
        const STARS: usize = 200;
        for _ in 0..STARS {
            // Uniform on the cap of the sphere above 9 degrees of elevation.
            let y: f64 = dice.between(0.16, 1.0);
            let around = dice.between(0.0, std::f64::consts::TAU);
            let flat = (1.0 - y * y).sqrt();
            let distance = dice.between(4_500.0, 7_000.0);
            let size = dice.between(7.0, 18.0) as f32;
            let glow = (dice.between(0.4, 1.6) * (0.3 + 0.7 * y)) as f32;
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

    /// The spheres and boxes.
    #[must_use]
    pub fn bodies(&self) -> &[Body] {
        &self.bodies
    }

    /// The point lights.
    #[must_use]
    pub fn lights(&self) -> &[SceneLight] {
        &self.lights
    }

    /// The demo area as `camera` sees it, under its sky ([`ENVIRONMENT`]).
    #[must_use]
    pub fn scene(&self, camera: &CameraPose) -> Scene {
        let instances = self.bodies.iter().map(|b| b.instance(camera)).collect();
        let mut scene = Scene::from_instances(*camera, instances)
            .with_lights(self.lights.clone())
            .with_environment(ENVIRONMENT);
        scene.models = self.props.iter().map(|p| p.instance(camera)).collect();
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
            let backdrop = matches!(body.name, "tower beacon" | "moon" | "moon sea" | "star");
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
        // Reflection: a smooth metal slab whose top is just above the floor.
        let mirror = bodies.iter().find(|b| b.name == "mirror").expect("mirror");
        assert!(mirror.metallic == 1.0 && mirror.roughness < 0.1);
        let top = metres(mirror.position)[1] + f64::from(mirror.radius);
        assert!(top > FLOOR && top < FLOOR + 0.05, "no z-fighting: {top}");
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
        assert_eq!(lights.len(), 6, "a key, two fills and the three lamps");
        // The rest of the light is the sky: painted, its horizon the hue of
        // the fog, darker overhead, and much dimmer on the wall of the shadow
        // theatre than one of its lamps is (or the shadows wash out).
        assert!(ENVIRONMENT.background && !ENVIRONMENT.is_none());
        let [zenith, horizon, _ground] = ENVIRONMENT.colors();
        for c in 0..3 {
            let same_hue = horizon[c] / FOG_COLOR[c] - horizon[0] / FOG_COLOR[0];
            assert!(same_hue.abs() < 1e-6);
            assert!(zenith[c] < horizon[c]);
        }
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
        for name in ["red lamp", "green lamp", "blue lamp", "key lamp"] {
            let lamp = bodies.iter().find(|b| b.name == name).expect(name);
            let inside = lights
                .iter()
                .any(|l| WorldPos::distance(l.position, lamp.position) < f64::from(lamp.radius));
            assert!(inside, "{name} holds no light");
        }
        // Emissive: the beacon cube.
        let beacon = bodies.iter().find(|b| b.name == "beacon").expect("beacon");
        assert!(beacon.emissive.iter().sum::<f32>() > 1.0);
        // Backdrop.
        assert!(named("tower beacon") >= 10 && named("star") >= 100 && named("moon") == 1);

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
        let moved = World::default().camera(at([3.0, 1.0, -8.0]), [0.6, 0.0, -0.8], 1.0);
        let second = World::default().build(&moved);
        assert_eq!(first.models.len(), area.props().len());
        assert_eq!(first.models.len(), second.models.len());
        assert_eq!(first.instances.len(), area.bodies().len());
        for (a, b) in first.models.iter().zip(&second.models) {
            assert!(Arc::ptr_eq(&a.mesh, &b.mesh));
        }
        // Far fewer meshes than props: the crates, characters, ships and
        // towers share theirs.
        let mut meshes: Vec<*const MeshData> =
            first.models.iter().map(|m| Arc::as_ptr(&m.mesh)).collect();
        meshes.sort_unstable();
        meshes.dedup();
        assert!(meshes.len() * 2 < first.models.len(), "{}", meshes.len());
        // The scene moves with the camera, by exactly the camera's move.
        let (a, b) = (first.models[0].position, second.models[0].position);
        assert!((a[0] - b[0] - 3.0).abs() < 1e-4 && (a[2] - b[2] + 8.0).abs() < 1e-4);
    }

    /// From the spawn point the floor is under the player's feet and the
    /// stations are in front: nothing stands on the spawn point itself.
    #[test]
    fn the_spawn_point_is_clear_and_on_the_floor() {
        let area = DemoArea::shared();
        for body in area.bodies() {
            let clear = distance(body.position) - f64::from(body.radius);
            assert!(
                clear > 2.0 || body.name == "mirror",
                "{} at {clear} m",
                body.name
            );
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
