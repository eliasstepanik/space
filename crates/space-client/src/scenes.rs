//! Built-in test scenes that live in world space: a PBR material chart and a
//! real-size solar system.
//!
//! Both are lists of [`Body`]s — a sphere or box with a world position, a size
//! and a metallic-roughness material — turned into a camera-relative
//! [`Scene`] for whatever camera looks at them. Positions are [`WorldPos`]
//! (`i64` cell + `f32` offset), so a body 4.5e12 m from the origin is placed as
//! exactly as one next to it; only the *camera-relative* offset the renderer
//! draws with is `f32`.

use space_render::{CameraPose, GpuInstance, Scene, SceneLight, SHAPE_BOX, SHAPE_SPHERE};
use space_world::WorldPos;

/// One object of a world-space scene.
#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub name: &'static str,
    pub position: WorldPos,
    /// Sphere radius, or a box's half-extent, in metres.
    pub radius: f32,
    pub shape: u32,
    /// Base colour (linear rgb).
    pub albedo: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    /// Emitted radiance (linear rgb).
    pub emissive: [f32; 3],
    /// 0 = opaque, 1 = fully transmissive (glass; needs a refraction preset).
    pub transmission: f32,
}

impl Body {
    /// An opaque, non-metallic, non-emissive sphere.
    fn sphere(name: &'static str, position: WorldPos, radius: f32, albedo: [f32; 3]) -> Self {
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
}

/// A world position `metres` from the world origin.
fn at(metres: [f64; 3]) -> WorldPos {
    WorldPos::from_f64_metres_around(WorldPos::ZERO, metres)
}

/// The camera-relative render scene of `bodies` seen from `camera`.
pub fn scene_of(camera: &CameraPose, bodies: &[Body], lights: Vec<SceneLight>) -> Scene {
    let instances = bodies
        .iter()
        .map(|body| {
            let d = WorldPos::delta(camera.position, body.position);
            GpuInstance {
                center: [d[0] as f32, d[1] as f32, d[2] as f32, body.radius],
                albedo: [
                    body.albedo[0],
                    body.albedo[1],
                    body.albedo[2],
                    body.metallic,
                ],
                emissive: [body.emissive[0], body.emissive[1], body.emissive[2], 1.0],
                roughness: body.roughness,
                shape: body.shape,
                transmission: body.transmission,
                _pad1: 0.0,
            }
        })
        .collect();
    Scene::from_instances(*camera, instances).with_lights(lights)
}

/// The body whose surface is nearest to `position`, with that distance in
/// metres (negative inside the body).
pub fn nearest(bodies: &[Body], position: WorldPos) -> Option<(&Body, f64)> {
    bodies
        .iter()
        .map(|body| {
            let d = WorldPos::delta(position, body.position);
            let centre = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            (body, centre - f64::from(body.radius))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
}

/// A length for a heads-up line: metres, kilometres or astronomical units,
/// whichever reads best.
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

// ---- PBR material chart --------------------------------------------------------

const CHART_COLUMNS: usize = 7;
/// Distance from the chart camera to the plane of spheres, metres.
const CHART_DISTANCE: f64 = 8.0;

/// The base camera of the material chart: at the origin, looking down -Z.
pub fn materials_camera(width: u32, height: u32) -> CameraPose {
    CameraPose {
        position: WorldPos::ZERO,
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 0.05,
        far: 1.0e3,
    }
}

/// The material chart: three rows of spheres whose roughness runs from mirror
/// smooth (left) to fully rough (right) — polished-to-brushed gold, a red
/// dielectric (plastic) and steel — over a smooth metal floor that mirrors
/// them with screen-space reflections on the `high`/`medium` presets, plus a
/// glass sphere (refraction on those presets) and an emissive one. Three
/// point lights: every sphere shows the GGX highlight widening with
/// roughness, the dielectric row keeps its diffuse colour, the metal rows
/// have none.
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
    // A smooth metal floor under the chart.
    bodies.push(Body {
        shape: SHAPE_BOX,
        metallic: 1.0,
        roughness: 0.08,
        ..Body::sphere(
            "floor",
            at([0.0, -101.7, -CHART_DISTANCE]),
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

/// The material chart's three point lights: key, fill and back.
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
    ]
}

/// The material chart seen from `camera`.
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
pub fn solar_lights() -> Vec<SceneLight> {
    vec![SceneLight {
        position: WorldPos::ZERO,
        color: [1.0, 0.97, 0.92],
        intensity: (3.2 * AU * AU) as f32,
    }]
}

/// The solar system's spawn camera: on the sunward side of Earth, four Earth
/// radii from its centre, looking away from the Sun at the lit planet.
pub fn solar_camera(width: u32, height: u32) -> CameraPose {
    let earth = heliocentric(AU, 0.0);
    CameraPose {
        position: at([earth[0], earth[1], earth[2] + 4.0 * 6.371e6]),
        forward: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y: std::f32::consts::FRAC_PI_3,
        aspect: width as f32 / height as f32,
        near: 1.0,
        far: 1.0e13,
    }
}

/// The solar system seen from `camera`.
pub fn solar_scene(camera: &CameraPose) -> Scene {
    scene_of(camera, &solar_bodies(), solar_lights())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distance(a: WorldPos, b: WorldPos) -> f64 {
        let d = WorldPos::delta(a, b);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
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
            let d = distance(sun.position, body.position);
            assert!((d - from_sun).abs() < 1.0, "{name}: {d} m from the Sun");
        }
        let moon = distance(find("Earth").position, find("Moon").position);
        assert!(
            (moon - MOON_DISTANCE).abs() / MOON_DISTANCE < 0.01,
            "{moon}"
        );
        assert_eq!(bodies.len(), 10);
    }

    /// The spawn camera sits just off Earth, between it and the Sun, with
    /// Earth dead ahead — and the camera-relative instance the renderer gets
    /// is exact to well under a metre even though both are 1 AU from the
    /// origin.
    #[test]
    fn the_spawn_camera_faces_the_lit_side_of_earth() {
        let camera = solar_camera(16, 9);
        let bodies = solar_bodies();
        let (body, surface) = nearest(&bodies, camera.position).expect("bodies");
        assert_eq!(body.name, "Earth");
        assert!((surface - 3.0 * 6.371e6).abs() < 1.0, "{surface}");
        let scene = solar_scene(&camera);
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
    /// with roughness rising along each row.
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
        assert!(bodies.iter().any(|b| b.shape == SHAPE_BOX));
        assert_eq!(materials_lights().len(), 3);
    }
}
