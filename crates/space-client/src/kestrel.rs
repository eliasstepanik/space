//! The IC-4 "Kestrel": the 23 m patrol interceptor of `assets/ships/kestrel/`
//! as something a scene can park somewhere.
//!
//! # Where the model comes from
//!
//! Both files are **in the binary** ([`FULL_GLB`], [`PROXY_GLB`]): a released
//! client is one executable with no assets folder next to it, and the Docker
//! image has none either. The full model costs the executable 14.9 MB and the
//! proxy 1.7 MB; nothing is read from disk at run time.
//!
//! Each is parsed the first time a frame draws it ([`proxy_mesh`],
//! [`full_mesh`]: 3 ms and about 100 ms in a release build) and is one shared
//! `Arc` for the life of the process after that, which is what keeps it on
//! the GPU (the renderer's model cache is keyed by the `Arc`).
//!
//! # Two levels of detail
//!
//! | | triangles | parts | what it has |
//! |-|-----------|-------|-------------|
//! | full  | 417,546 | 16 | exterior with its decals, the complete interior |
//! | proxy | 11,585  | 9  | the exterior, one flat colour per face |
//!
//! [`mesh_for`] picks by the camera's distance from the hull and by what
//! draws the picture ([`Detail`]): on a GPU the full model from as far as it
//! can be told from the proxy (60 m — the whole of the demo area), on a
//! software rasteriser only for whoever stands next to the ship or in it
//! (5 m). The proxy is the same shape to within a few centimetres, so the
//! switch changes the detail and not the silhouette.
//!
//! Measured in the demo area at the `high` preset, milliseconds a frame
//! (in brackets: the same view of the area before the ship was in it):
//!
//! | view | RTX 4080 SUPER, 1280x720 | lavapipe, 32 threads, 640x480 |
//! |------|--------------------------|-------------------------------|
//! | from the spawn point, 23 m        | 1.5, full (1.4) | 83, proxy (62)  |
//! | in front of the chart, 13 m       | 1.6, full (1.4) | 80, proxy (60)  |
//! | behind the chart, 6 m             | 1.9, full (1.2) | 111, proxy (59) |
//! | on the apron, 3 m                 | 1.8, full (1.1) | 189, full (55)  |
//! | inside, in the crew quarters      | 3.7, full (1.1) | 409, full (24)  |
//!
//! A GPU does not notice the full model. On lavapipe it costs 161 ms a frame
//! from 15 m where the proxy costs 93, and 0.2 to 0.4 s a frame up close: so
//! a software rasteriser is given the proxy for every view of the area that
//! is not a visit to the ship. What the proxy still costs it there — a third
//! of a frame from the spawn point — is mostly the pixels of a large lit
//! object (14 ms of the 21), then the floodlight's shadow rays (5 ms).
//!
//! # What of the model the renderer shows
//!
//! Geometry, base colours, metalness and roughness, the decal atlas and the
//! emissive strength of the lamps and screens. Not shown, because the renderer
//! has no such feature yet: the normal and occlusion/roughness/metalness
//! textures (the factors are used alone), the screens' emissive texture (they
//! glow evenly, in its average colour), and transparency — the canopy glass
//! is an opaque dark pane.
//!
//! # Placement
//!
//! A [`Berth`] says where a ship stands: an asset, a position and a heading.
//! It is plain data on purpose — the scene definition that will be shared
//! with the server (and the physics that comes with it) can take it as it is.

use std::sync::{Arc, OnceLock};

use space_render::{rotation_y, CameraPose, MeshData, ModelInstance};

use crate::scenes::{at, metres, relative};

/// `assets/ships/kestrel/kestrel.glb`: the full model.
pub static FULL_GLB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ships/kestrel/kestrel.glb"
));

/// `assets/ships/kestrel/kestrel_lod1.glb`: the low-detail exterior proxy.
pub static PROXY_GLB: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/ships/kestrel/kestrel_lod1.glb"
));

/// The asset's id, as in `assets/ships/<id>/manifest.json`.
pub const ASSET: &str = "ships/kestrel";

/// How far below the model's origin its landing gear touches the ground,
/// metres (the three `SOCKET_Gear_Contact_*` sockets; the gear is deployed
/// in the model).
pub const GEAR_DROP: f64 = 3.3;

/// The model's bounds, metres, model space (+Y up, nose towards -Z): the
/// manifest's, which both levels of detail keep to within a few centimetres.
pub const BOUNDS_MIN: [f32; 3] = [-6.32, -3.3, -11.84];
/// See [`BOUNDS_MIN`].
pub const BOUNDS_MAX: [f32; 3] = [6.32, 2.3436, 11.955];

/// Where the pilot's eyes are, model space (`SOCKET_Camera_PilotEye`).
pub const PILOT_EYE: [f32; 3] = [0.0, 1.85, -6.85];

/// The crew hatch on the port side, model space (`SOCKET_Hatch_Crew`).
pub const CREW_HATCH: [f32; 3] = [-1.84, 0.25, -1.5];

/// Where the one lamp that lights the inside hangs, model space: under the
/// ceiling of the crew quarters, in the middle of the row of rooms on the
/// main deck, whose doorways let its light down the whole deck.
pub const CABIN_LAMP: [f32; 3] = [0.0, 1.15, 1.3];

/// What draws the picture, as far as the ship's level of detail goes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Detail {
    /// A GPU: the full model costs it a fraction of a millisecond.
    #[default]
    Hardware,
    /// A software rasteriser (Mesa's lavapipe in a container or on a CI
    /// runner): every triangle and every ray is paid for on the CPU.
    Software,
}

impl Detail {
    /// The detail for the device a render context runs on, by its name
    /// (`RenderContext::device_name`): `llvmpipe (LLVM 20.1.2, 256 bits)` is
    /// software, `NVIDIA GeForce RTX 4080 SUPER` is not.
    #[must_use]
    pub fn for_device(name: &str) -> Self {
        let name = name.to_ascii_lowercase();
        let software = ["llvmpipe", "lavapipe", "swiftshader", "softpipe"];
        if software.iter().any(|word| name.contains(word)) {
            Self::Software
        } else {
            Self::Hardware
        }
    }

    /// A camera closer to the hull's bounds than this is shown the full
    /// model, metres; one further away the proxy.
    ///
    /// * Hardware, 60 m: about where a decal or a panel line of the full
    ///   model shrinks under a pixel. The whole demo area is inside it — its
    ///   spawn point is 23 m from the hull — so there the ship is always the
    ///   real one.
    /// * Software, 5 m: "is standing next to it". The spawn point and the
    ///   stations of the demo area are outside, so every ordinary view draws
    ///   11.6 thousand triangles for the ship instead of 418 thousand; only
    ///   a visit to the ship pays for it.
    #[must_use]
    pub const fn full_detail_range(self) -> f64 {
        match self {
            Self::Hardware => 60.0,
            Self::Software => 5.0,
        }
    }
}

fn load(bytes: &[u8], what: &str) -> Result<MeshData, String> {
    let started = std::time::Instant::now();
    let (mesh, warnings) = MeshData::from_glb_with_warnings(bytes).map_err(|e| e.to_string())?;
    // What the renderer leaves out of this model is known (the module docs
    // say it): not worth a warning in every log.
    for warning in &warnings {
        log::debug!("space-client: Kestrel {what}: {warning}");
    }
    log::info!(
        "space-client: Kestrel {what}: {} triangles in {} parts, loaded in {} ms",
        mesh.triangle_count(),
        mesh.parts.len(),
        started.elapsed().as_millis()
    );
    Ok(mesh)
}

/// The low-detail proxy (one shared mesh).
///
/// # Panics
///
/// If the embedded file does not load, which the tests rule out.
#[must_use]
pub fn proxy_mesh() -> Arc<MeshData> {
    static MESH: OnceLock<Arc<MeshData>> = OnceLock::new();
    Arc::clone(MESH.get_or_init(|| {
        Arc::new(load(PROXY_GLB, "proxy").expect("the embedded Kestrel proxy loads"))
    }))
}

/// The full model (one shared mesh), parsed the first time it is asked for.
/// Should the embedded file ever fail to load, this is the proxy.
#[must_use]
pub fn full_mesh() -> Arc<MeshData> {
    static MESH: OnceLock<Arc<MeshData>> = OnceLock::new();
    Arc::clone(MESH.get_or_init(|| match load(FULL_GLB, "full model") {
        Ok(mesh) => Arc::new(mesh),
        Err(e) => {
            log::error!(
                "space-client: the Kestrel's full model does not load ({e}); using the proxy"
            );
            proxy_mesh()
        }
    }))
}

/// The model to draw for a camera `distance` metres from the hull's bounds,
/// on a renderer of this `detail`.
#[must_use]
pub fn mesh_for(distance: f64, detail: Detail) -> Arc<MeshData> {
    if distance < detail.full_detail_range() {
        full_mesh()
    } else {
        proxy_mesh()
    }
}

/// Where a ship stands in a scene: which asset, where its origin is and which
/// way it points. Scene space is metres from the scene's spawn point, +Y up.
///
/// Plain data: this is the record a shared scene definition would hold for
/// the ship (and a rigid body would be created from).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Berth {
    /// The asset's id ([`ASSET`]).
    pub asset: &'static str,
    /// Where the model's origin is, scene space, metres.
    pub position: [f64; 3],
    /// The heading: a rotation about +Y, radians. At 0 the nose points down
    /// -Z; positive turns it towards -X (to the left, seen from above with
    /// -Z ahead).
    pub yaw: f32,
}

impl Berth {
    /// A Kestrel standing on its landing gear on ground whose top is at
    /// height `ground`, its origin over (`x`, `z`), heading `yaw`.
    #[must_use]
    pub const fn landed(x: f64, ground: f64, z: f64, yaw: f32) -> Self {
        Self {
            asset: ASSET,
            position: [x, ground + GEAR_DROP, z],
            yaw,
        }
    }

    /// The orientation as the rows of a rotation matrix (model to scene).
    #[must_use]
    pub fn rotation(&self) -> [[f32; 3]; 3] {
        rotation_y(self.yaw)
    }

    /// A point of the model (model space, metres) in scene space.
    #[must_use]
    pub fn to_scene(&self, point: [f32; 3]) -> [f64; 3] {
        let rotation = self.rotation();
        [0, 1, 2].map(|row| {
            let turned: f32 = (0..3).map(|k| rotation[row][k] * point[k]).sum();
            self.position[row] + f64::from(turned)
        })
    }

    /// A point of the scene (metres) in model space.
    #[must_use]
    pub fn to_model(&self, point: [f64; 3]) -> [f64; 3] {
        let rotation = self.rotation();
        let offset = [0, 1, 2].map(|k| point[k] - self.position[k]);
        // The inverse of a rotation is its transpose.
        [0, 1, 2].map(|column| {
            (0..3)
                .map(|k| f64::from(rotation[k][column]) * offset[k])
                .sum()
        })
    }

    /// The scene-space box around the ship: the axis-aligned bounds of its
    /// (turned) model bounds, as (minimum, maximum).
    #[must_use]
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let (mut min, mut max) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for corner in 0..8 {
            let point = [0, 1, 2].map(|axis| {
                if (corner >> axis) & 1 == 0 {
                    BOUNDS_MIN[axis]
                } else {
                    BOUNDS_MAX[axis]
                }
            });
            let point = self.to_scene(point);
            for axis in 0..3 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        (min, max)
    }

    /// How far `point` (scene space) is from the hull's bounds, metres: 0
    /// inside them.
    #[must_use]
    pub fn distance(&self, point: [f64; 3]) -> f64 {
        let local = self.to_model(point);
        (0..3)
            .map(|axis| {
                let below = f64::from(BOUNDS_MIN[axis]) - local[axis];
                let above = local[axis] - f64::from(BOUNDS_MAX[axis]);
                below.max(above).max(0.0).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }

    /// The ship as `camera` sees it: the level of detail its distance calls
    /// for on a renderer of this `detail` ([`mesh_for`]), at this berth.
    #[must_use]
    pub fn instance(&self, camera: &CameraPose, detail: Detail) -> ModelInstance {
        ModelInstance {
            mesh: mesh_for(self.distance(metres(camera.position)), detail),
            position: relative(camera, at(self.position)),
            rotation: self.rotation(),
            scale: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], tolerance: f32) -> bool {
        (0..3).all(|k| (a[k] - b[k]).abs() <= tolerance)
    }

    /// Both embedded files load, hold what the manifest says, and come out of
    /// the loader as a handful of parts: one per material of the full model,
    /// one per face colour of the proxy.
    #[test]
    fn both_levels_of_detail_load_from_the_binary() {
        assert_eq!(FULL_GLB.len(), 14_862_392, "manifest.json: kestrel.glb");
        assert_eq!(
            PROXY_GLB.len(),
            1_712_052,
            "manifest.json: kestrel_lod1.glb"
        );

        let (proxy, warnings) =
            MeshData::from_glb_with_warnings(PROXY_GLB).expect("the proxy loads");
        proxy.validate().expect("a valid mesh");
        assert_eq!(proxy.triangle_count(), 11_585);
        // One primitive with a vertex colour per face: nine paints.
        assert_eq!(proxy.parts.len(), 9, "{:?}", proxy.parts);
        let paints: Vec<[f32; 3]> = proxy.parts.iter().map(|p| p.material.base_color).collect();
        for expected in [
            [0.66, 0.67, 0.68],    // white hull paint
            [0.56, 0.57, 0.58],    // bare metal
            [0.86, 0.56, 0.02],    // safety yellow
            [0.05, 0.09, 0.12],    // canopy glass
            [0.032, 0.032, 0.035], // thermal black
        ] {
            assert!(
                paints.iter().any(|paint| close(*paint, expected, 1e-4)),
                "no face is painted {expected:?}: {paints:?}"
            );
        }
        for part in &proxy.parts {
            let material = part.material;
            assert_eq!((material.metallic, material.roughness), (0.2, 0.45));
            assert_eq!(material.base_color_texture, None);
        }
        assert!(proxy.textures.is_empty(), "no base-colour texture");
        // What is left out is said, and is what the module docs say.
        assert_eq!(warnings.len(), 3, "{warnings:?}");
        for slot in [
            "metallicRoughnessTexture",
            "normalTexture",
            "occlusionTexture",
        ] {
            assert!(warnings.iter().any(|w| w.contains(slot)), "{slot}");
        }

        let (full, warnings) = MeshData::from_glb_with_warnings(FULL_GLB).expect("it loads");
        full.validate().expect("a valid mesh");
        assert_eq!(full.triangle_count(), 417_546);
        // 1,621 primitives on 1,010 nodes, 16 materials: 16 parts.
        assert_eq!(full.parts.len(), 16);
        // The decal atlas is the one base-colour texture.
        assert_eq!(full.textures.len(), 1);
        assert_eq!(
            (full.textures[0].width(), full.textures[0].height()),
            (2048, 2048)
        );
        let decals: Vec<_> = full
            .parts
            .iter()
            .filter(|p| p.material.base_color_texture.is_some())
            .collect();
        assert_eq!(decals.len(), 1);
        assert_eq!(decals[0].index_count, 366 * 3);
        // The lamps and the screens glow with their emissive strength.
        let glowing: Vec<[f32; 3]> = full
            .parts
            .iter()
            .map(|p| p.material.emissive)
            .filter(|e| e.iter().sum::<f32>() > 0.0)
            .collect();
        assert_eq!(glowing.len(), 2, "{glowing:?}");
        assert!(glowing.iter().any(|e| close(*e, [12.0, 11.04, 9.6], 1e-3)));
        // The screens' factor is (1.65, 2.55, 3.0); their picture cannot be
        // drawn, so they glow in its average: a dim blue-green.
        let screens = glowing.iter().find(|e| e[0] < 5.0).expect("the screens");
        assert!(
            screens[0] > 0.02 && screens[0] < screens[1] && screens[1] < screens[2],
            "{screens:?}"
        );
        assert!(screens[2] > 0.15 && screens[2] < 1.2, "{screens:?}");
        // Bare metal is the bulk of it.
        let metal = full
            .parts
            .iter()
            .max_by_key(|p| p.index_count)
            .expect("parts");
        assert_eq!(metal.index_count, 199_024 * 3);
        assert_eq!(
            (metal.material.metallic, metal.material.roughness),
            (1.0, 0.32)
        );
        for word in [
            "normalTexture",
            "emissiveTexture",
            "average",
            "BLEND",
            "MASK",
        ] {
            assert!(
                warnings.iter().any(|w| w.contains(word)),
                "{word}: {warnings:?}"
            );
        }
        assert_eq!(warnings.len(), 6, "{warnings:?}");

        // Both keep to the manifest's bounds, and so to each other.
        for (name, mesh) in [("proxy", &proxy), ("full", &full)] {
            assert!(
                close(mesh.bounds_min, BOUNDS_MIN, 0.02)
                    && close(mesh.bounds_max, BOUNDS_MAX, 0.02),
                "{name}: {:?} .. {:?}",
                mesh.bounds_min,
                mesh.bounds_max
            );
            // The gear reaches down to the contact plane.
            assert!(
                (f64::from(mesh.bounds_min[1]) + GEAR_DROP).abs() < 0.01,
                "{name}"
            );
        }
        // The sockets this module quotes are inside the hull.
        for socket in [PILOT_EYE, CREW_HATCH, CABIN_LAMP] {
            assert!((0..3).all(|k| socket[k] > BOUNDS_MIN[k] && socket[k] < BOUNDS_MAX[k]));
        }
    }

    /// The shared meshes are one `Arc` each, and the level of detail follows
    /// the distance.
    #[test]
    fn the_detail_follows_the_distance_and_the_meshes_are_shared() {
        let (proxy, full) = (proxy_mesh(), full_mesh());
        assert!(Arc::ptr_eq(&proxy, &proxy_mesh()));
        assert!(Arc::ptr_eq(&full, &full_mesh()));
        assert!(!Arc::ptr_eq(&proxy, &full));
        assert!(full.triangle_count() > 30 * proxy.triangle_count());
        for detail in [Detail::Hardware, Detail::Software] {
            let range = detail.full_detail_range();
            assert!(Arc::ptr_eq(&mesh_for(0.0, detail), &full));
            assert!(Arc::ptr_eq(&mesh_for(range - 0.01, detail), &full));
            assert!(Arc::ptr_eq(&mesh_for(range, detail), &proxy));
            assert!(Arc::ptr_eq(&mesh_for(1.0e6, detail), &proxy));
        }
        // A software rasteriser is given the full model much later.
        assert_eq!(Detail::default(), Detail::Hardware);
        assert!(Detail::Software.full_detail_range() * 4.0 < Detail::Hardware.full_detail_range());
        // Told apart by the device's name.
        for (name, detail) in [
            ("llvmpipe (LLVM 20.1.2, 256 bits)", Detail::Software),
            ("SwiftShader Device (LLVM 10.0.0)", Detail::Software),
            ("NVIDIA GeForce RTX 4080 SUPER", Detail::Hardware),
            ("Intel(R) UHD Graphics 770", Detail::Hardware),
            ("", Detail::Hardware),
        ] {
            assert_eq!(Detail::for_device(name), detail, "{name}");
        }
    }

    /// A berth turns and moves the model: points go there and back, the
    /// landed ship's gear is on the ground, and the distance is measured from
    /// the hull, not from its origin.
    #[test]
    fn a_berth_places_the_model() {
        let ground = -1.7;
        // Heading a quarter turn: the nose (-Z) towards -X.
        let berth = Berth::landed(10.0, ground, -30.0, std::f32::consts::FRAC_PI_2);
        assert_eq!(berth.asset, "ships/kestrel");
        assert_eq!(berth.position, [10.0, ground + 3.3, -30.0]);
        let nose = berth.to_scene([0.0, 0.0, BOUNDS_MIN[2]]);
        assert!((nose[0] - (10.0 - 11.84)).abs() < 1e-4 && (nose[2] + 30.0).abs() < 1e-4);
        // Port (-X of the model) is towards +Z of the scene.
        let port = berth.to_scene([-1.0, 0.0, 0.0]);
        assert!((port[2] + 29.0).abs() < 1e-5, "{port:?}");
        for point in [[1.5f32, -2.0, 7.0], PILOT_EYE, BOUNDS_MIN] {
            let back = berth.to_model(berth.to_scene(point));
            assert!((0..3).all(|k| (back[k] - f64::from(point[k])).abs() < 1e-4));
        }
        // On its gear: the lowest point of the bounds is the ground.
        let (min, max) = berth.bounds();
        assert!((min[1] - ground).abs() < 1e-5);
        assert!((max[1] - (ground + 5.6436)).abs() < 1e-4);
        // Turned a quarter: 23.8 m along X, 12.6 m along Z.
        assert!((max[0] - min[0] - 23.795).abs() < 1e-3);
        assert!((max[2] - min[2] - 12.64).abs() < 1e-3);
        // Distances: inside, beside the hull, off the nose.
        assert_eq!(berth.distance(berth.to_scene(PILOT_EYE)), 0.0);
        let beside = berth.to_scene([BOUNDS_MIN[0] - 5.0, 0.0, 0.0]);
        assert!((berth.distance(beside) - 5.0).abs() < 1e-4);
        let ahead = berth.to_scene([0.0, 0.0, BOUNDS_MIN[2] - 20.0]);
        assert!((berth.distance(ahead) - 20.0).abs() < 1e-4);
    }
}
