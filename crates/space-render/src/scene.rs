//! CPU-side scene construction: camera maths, placeholder meshes, and the
//! LOD → render-instance bridge over `space-world`'s deterministic galaxy.
//!
//! Everything here is pure CPU and unit-testable without a GPU: the LOD
//! selector's output (≤ [`space_world::MAX_SELECTED`] nodes) is turned into
//! one render instance per star (sphere or box per LOD level), positioned
//! camera-relative so the vertex shader works in f32 around the camera.

use space_world::{Galaxy, LodSelector, NodeRef, WorldPos, MAX_SELECTED};

use crate::params::GpuInstance;

/// Shape selector for an LOD depth: coarse nodes render as boxes, fine nodes
/// as spheres (the "sphere/box per LOD level" placeholder rule).
pub const SHAPE_SPHERE: u32 = 0;
pub const SHAPE_BOX: u32 = 1;

/// Depth at/below which an LOD node is drawn as a box instead of a sphere.
pub const BOX_LOD_DEPTH: u8 = 55;

/// A camera pose in world coordinates (camera-relative maths happens in
/// [`Scene::build`]).
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    /// Eye position in the world.
    pub position: WorldPos,
    /// Unit view direction.
    pub forward: [f32; 3],
    /// Unit up hint.
    pub up: [f32; 3],
    /// Full vertical field of view, radians.
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for CameraPose {
    fn default() -> Self {
        Self {
            position: WorldPos::ZERO,
            forward: [0.0, 0.0, -1.0],
            up: [0.0, 1.0, 0.0],
            fov_y: std::f32::consts::FRAC_PI_3,
            aspect: 1.0,
            near: 0.1,
            far: 1.0e7,
        }
    }
}

/// Column-major 4x4 matrix (`[[column; 4]; 4]`), matching WGSL `mat4x4<f32>`.
pub type Mat4 = [[f32; 4]; 4];

/// Column-major multiply `a * b`.
#[must_use]
pub fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0f32; 4]; 4];
    for (c, col) in out.iter_mut().enumerate() {
        for (r, v) in col.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

/// Right-handed perspective projection (Vulkan clip space, y flipped).
#[must_use]
pub fn mat4_perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

/// Look-at view matrix (camera at `eye` looking at `target`).
#[must_use]
pub fn mat4_look_at(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> Mat4 {
    let f = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let s = normalize(cross(f, up));
    let u = cross(s, f);
    [
        [s[0], u[0], -f[0], 0.0],
        [s[1], u[1], -f[1], 0.0],
        [s[2], u[2], -f[2], 0.0],
        [-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0],
    ]
}

/// General 4x4 inverse (used for `inv_view_proj`). Returns `None` when the
/// matrix is singular.
#[must_use]
pub fn mat4_inverse(m: &Mat4) -> Option<Mat4> {
    // Standard cofactor expansion; `a[col][row]` indexing.
    let a = m;
    let mut inv = [[0.0f32; 4]; 4];
    inv[0][0] =
        a[1][1] * a[2][2] * a[3][3] - a[1][1] * a[3][2] * a[2][3] - a[2][1] * a[1][2] * a[3][3]
            + a[2][1] * a[3][2] * a[1][3]
            + a[3][1] * a[1][2] * a[2][3]
            - a[3][1] * a[2][2] * a[1][3];
    inv[1][0] =
        -(a[1][0] * a[2][2] * a[3][3] - a[1][0] * a[3][2] * a[2][3] - a[2][0] * a[1][2] * a[3][3]
            + a[2][0] * a[3][2] * a[1][3]
            + a[3][0] * a[1][2] * a[2][3]
            - a[3][0] * a[2][2] * a[1][3]);
    inv[2][0] =
        a[1][0] * a[2][1] * a[3][3] - a[1][0] * a[3][1] * a[2][3] - a[2][0] * a[1][1] * a[3][3]
            + a[2][0] * a[3][1] * a[1][3]
            + a[3][0] * a[1][1] * a[2][3]
            - a[3][0] * a[2][1] * a[1][3];
    inv[3][0] =
        -(a[1][0] * a[2][1] * a[3][2] - a[1][0] * a[3][1] * a[2][2] - a[2][0] * a[1][1] * a[3][2]
            + a[2][0] * a[3][1] * a[1][2]
            + a[3][0] * a[1][1] * a[2][2]
            - a[3][0] * a[2][1] * a[1][2]);
    inv[0][1] =
        -(a[0][1] * a[2][2] * a[3][3] - a[0][1] * a[3][2] * a[2][3] - a[2][1] * a[0][2] * a[3][3]
            + a[2][1] * a[3][2] * a[0][3]
            + a[3][1] * a[0][2] * a[2][3]
            - a[3][1] * a[2][2] * a[0][3]);
    inv[1][1] =
        a[0][0] * a[2][2] * a[3][3] - a[0][0] * a[3][2] * a[2][3] - a[2][0] * a[1][2] * a[3][3]
            + a[2][0] * a[3][2] * a[1][3]
            + a[3][0] * a[1][2] * a[2][3]
            - a[3][0] * a[2][2] * a[1][3];
    inv[2][1] =
        -(a[0][0] * a[2][1] * a[3][3] - a[0][0] * a[3][1] * a[2][3] - a[2][0] * a[1][1] * a[3][3]
            + a[2][0] * a[3][1] * a[1][3]
            + a[3][0] * a[1][1] * a[2][3]
            - a[3][0] * a[2][1] * a[1][3]);
    inv[3][1] =
        a[0][0] * a[2][1] * a[3][2] - a[0][0] * a[3][1] * a[2][2] - a[2][0] * a[1][1] * a[3][2]
            + a[2][0] * a[3][1] * a[1][2]
            + a[3][0] * a[1][1] * a[2][2]
            - a[3][0] * a[2][1] * a[1][2];
    inv[0][2] =
        a[0][1] * a[1][2] * a[3][3] - a[0][1] * a[3][2] * a[1][3] - a[1][1] * a[0][2] * a[3][3]
            + a[1][1] * a[3][2] * a[0][3]
            + a[3][1] * a[0][2] * a[1][3]
            - a[3][1] * a[1][2] * a[0][3];
    inv[1][2] =
        -(a[0][0] * a[1][2] * a[3][3] - a[0][0] * a[3][2] * a[1][3] - a[1][0] * a[0][2] * a[3][3]
            + a[1][0] * a[3][2] * a[0][3]
            + a[3][0] * a[0][2] * a[1][3]
            - a[3][0] * a[1][2] * a[0][3]);
    inv[2][2] =
        a[0][0] * a[1][1] * a[3][3] - a[0][0] * a[3][1] * a[1][3] - a[1][0] * a[0][1] * a[3][3]
            + a[1][0] * a[3][1] * a[0][3]
            + a[3][0] * a[0][1] * a[1][3]
            - a[3][0] * a[1][1] * a[0][3];
    inv[3][2] =
        -(a[0][0] * a[1][1] * a[3][2] - a[0][0] * a[3][1] * a[1][2] - a[1][0] * a[0][1] * a[3][2]
            + a[1][0] * a[3][1] * a[0][2]
            + a[3][0] * a[0][1] * a[1][2]
            - a[3][0] * a[1][1] * a[0][2]);
    inv[0][3] =
        -(a[0][1] * a[1][2] * a[2][3] - a[0][1] * a[2][2] * a[1][3] - a[1][1] * a[0][2] * a[2][3]
            + a[1][1] * a[2][2] * a[0][3]
            + a[2][1] * a[0][2] * a[1][3]
            - a[2][1] * a[1][2] * a[0][3]);
    inv[1][3] =
        a[0][0] * a[1][2] * a[2][3] - a[0][0] * a[2][2] * a[1][3] - a[1][0] * a[0][2] * a[2][3]
            + a[1][0] * a[2][2] * a[0][3]
            + a[2][0] * a[0][2] * a[1][3]
            - a[2][0] * a[1][2] * a[0][3];
    inv[2][3] =
        -(a[0][0] * a[1][1] * a[2][3] - a[0][0] * a[2][1] * a[1][3] - a[1][0] * a[0][1] * a[2][3]
            + a[1][0] * a[2][1] * a[0][3]
            + a[2][0] * a[0][1] * a[1][3]
            - a[2][0] * a[1][1] * a[0][3]);
    inv[3][3] =
        a[0][0] * a[1][1] * a[2][2] - a[0][0] * a[2][1] * a[1][2] - a[1][0] * a[0][1] * a[2][2]
            + a[1][0] * a[2][1] * a[0][2]
            + a[2][0] * a[0][1] * a[1][2]
            - a[2][0] * a[1][1] * a[0][2];
    let det = a[0][0] * inv[0][0] + a[0][1] * inv[1][0] + a[0][2] * inv[2][0] + a[0][3] * inv[3][0];
    if det == 0.0 {
        return None;
    }
    let inv_det = 1.0 / det;
    for col in inv.iter_mut() {
        for v in col.iter_mut() {
            *v *= inv_det;
        }
    }
    Some(inv)
}

const fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

const fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt().max(1e-9);
    [v[0] / l, v[1] / l, v[2] / l]
}

/// One analytic point light (world-space position; converted to
/// camera-relative when the frame's uniform is built).
#[derive(Clone, Copy, Debug)]
pub struct SceneLight {
    pub position: WorldPos,
    /// Linear radiance (rgb).
    pub color: [f32; 3],
    pub intensity: f32,
}

/// A fully CPU-built frame: the render instances (camera-relative) and the
/// triangle soup for the acceleration structures, both derived from the LOD
/// selection over the deterministic galaxy.
pub struct Scene {
    /// Render instances, camera-relative (≤ `MAX_SELECTED`).
    pub instances: Vec<GpuInstance>,
    /// World-space triangle vertices (camera-relative), one mesh per instance.
    /// `meshes[i]` is the mesh of `instances[i]` (icosphere or cube).
    pub meshes: Vec<Vec<[f32; 3]>>,
    /// The camera pose the scene was built for.
    pub camera: CameraPose,
    /// World-space camera position, kept for light conversion.
    pub lights: Vec<SceneLight>,
    /// The LOD-selected nodes behind `instances` (for tests/diagnostics).
    pub selected: Vec<NodeRef>,
}

/// Placeholder radius of a star instance, in metres (stars are drawn far
/// larger than physical so they are visible at LOD distances).
pub const STAR_RADIUS_M: f32 = 90.0;

impl Scene {
    /// Build the render scene for `camera`: run the LOD selector over the
    /// octree band around the camera, look up a deterministic star per node,
    /// and emit one camera-relative instance per star.
    #[must_use]
    pub fn build(galaxy: &Galaxy, lod: &LodSelector, camera: &CameraPose) -> Self {
        let candidates = lod.candidates(camera.position, camera.forward, camera.fov_y);
        let mut selected = lod.select(&candidates, camera.position, camera.fov_y);
        // `select` stops refining once the budget is hit but still emits the
        // nodes already on its stack, so it can overshoot `MAX_SELECTED` by the
        // stack depth. The selection is sorted shallowest-first, so truncating
        // keeps the coarsest (largest) regions.
        selected.truncate(MAX_SELECTED);
        Self::from_selection(galaxy, camera, &selected)
    }

    /// Same as [`Scene::build`] but from an explicit selection (lets tests
    /// drive exact node sets).
    #[must_use]
    pub fn from_selection(galaxy: &Galaxy, camera: &CameraPose, selected: &[NodeRef]) -> Self {
        assert!(
            selected.len() <= MAX_SELECTED,
            "LOD selection exceeded MAX_SELECTED ({} > {MAX_SELECTED})",
            selected.len()
        );
        let mut instances = Vec::new();
        let mut meshes = Vec::new();
        let mut used = Vec::new();
        for &node in selected {
            let Some(star) = galaxy.star_at(node) else {
                continue;
            };
            // Star world position = node centre + deterministic offset.
            let world = WorldPos::from_f64_metres_around(
                node.center(),
                [
                    f64::from(star.pos_in_node[0]),
                    f64::from(star.pos_in_node[1]),
                    f64::from(star.pos_in_node[2]),
                ],
            );
            // Camera-relative f32 position (precision-safe near the camera).
            let d = WorldPos::delta(camera.position, world);
            let rel = [d[0] as f32, d[1] as f32, d[2] as f32];
            // Skip instances far outside the near frustum envelope to keep the
            // instance buffer bounded (the LOD band already bounds this).
            if !rel.iter().all(|v| v.is_finite()) {
                continue;
            }
            let tint = [
                f32::from(star.spectral_tint[0]) / 255.0,
                f32::from(star.spectral_tint[1]) / 255.0,
                f32::from(star.spectral_tint[2]) / 255.0,
            ];
            let lum = star.luminosity;
            let shape = if node.depth <= BOX_LOD_DEPTH {
                SHAPE_BOX
            } else {
                SHAPE_SPHERE
            };
            instances.push(GpuInstance {
                center: [rel[0], rel[1], rel[2], STAR_RADIUS_M],
                albedo: [tint[0], tint[1], tint[2], 0.0],
                emissive: [
                    tint[0] * lum * 4.0,
                    tint[1] * lum * 4.0,
                    tint[2] * lum * 4.0,
                    1.0,
                ],
                roughness: 0.6,
                shape,
                transmission: 0.0,
                _pad1: 0.0,
            });
            meshes.push(match shape {
                SHAPE_BOX => cube_mesh(),
                _ => icosphere_mesh(),
            });
            used.push(node);
        }
        Self {
            instances,
            meshes,
            camera: *camera,
            lights: Vec::new(),
            selected: used,
        }
    }

    /// The flat camera-relative triangle soup for the BLAS: `(vertices,
    /// indices)` with `indices` implicitly `0..vertices.len()` grouped in
    /// threes (meshes are stored as flat triangle lists).
    #[must_use]
    pub fn triangle_soup(&self) -> (Vec<[f32; 3]>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for mesh in &self.meshes {
            let base = vertices.len() as u32;
            vertices.extend_from_slice(mesh);
            for i in 0..(mesh.len() / 3) {
                indices.extend_from_slice(&[
                    base + (i * 3) as u32,
                    base + (i * 3 + 1) as u32,
                    base + (i * 3 + 2) as u32,
                ]);
            }
        }
        (vertices, indices)
    }
}

/// Unit icosphere mesh (60 vertices, positions == normals), matching the
/// `sphere_vertex(sphere_index(..))` sequence in `shaders/forward.wgsl`.
#[must_use]
pub fn icosphere_mesh() -> Vec<[f32; 3]> {
    // The golden ratio; written as a literal because `std::f32::consts` has no
    // `PHI` and the clippy approximation lint would otherwise fire on the value.
    #[allow(clippy::approx_constant)]
    let phi = 1.618_034_f32;
    let inv = 1.0 / (1.0 + phi * phi).sqrt();
    let v: [[f32; 3]; 12] = [
        [-inv, phi * inv, 0.0],
        [inv, phi * inv, 0.0],
        [-inv, -phi * inv, 0.0],
        [inv, -phi * inv, 0.0],
        [0.0, -inv, inv],
        [0.0, inv, inv],
        [0.0, -inv, -inv],
        [0.0, inv, -inv],
        [phi * inv, 0.0, -inv],
        [phi * inv, 0.0, inv],
        [-phi * inv, 0.0, -inv],
        [-phi * inv, 0.0, inv],
    ];
    let f: [u32; 60] = [
        0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7,
        1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 5, 8, 1, 2, 4, 11, 6, 2, 7, 8,
        6, 10,
    ];
    f.iter().map(|&i| v[i as usize]).collect()
}

/// Unit cube mesh as 36 vertices (6 per face, flat normals implicit in the
/// face), matching `box_vertex`/`box_normal` in `shaders/forward.wgsl`.
#[must_use]
pub fn cube_mesh() -> Vec<[f32; 3]> {
    let normals: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];
    let mut out = Vec::with_capacity(36);
    for &n in &normals {
        let mut helper = [1.0f32, 0.0, 0.0];
        if n[0].abs() > 0.9 {
            helper = [0.0, 1.0, 0.0];
        }
        let t = normalize(cross(n, helper));
        let b = cross(n, t);
        let corners = [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ];
        for c in corners {
            out.push([
                n[0] + t[0] * c[0] + b[0] * c[1],
                n[1] + t[1] * c[0] + b[1] * c[1],
                n[2] + t[2] * c[0] + b[2] * c[1],
            ]);
        }
    }
    out
}

impl Scene {
    /// An explicit scene (tests): instances + their meshes, with the camera
    /// they are rendered with. `instances` are already camera-relative.
    #[must_use]
    pub fn from_instances(camera: CameraPose, instances: Vec<GpuInstance>) -> Self {
        let meshes = instances
            .iter()
            .map(|i| match i.shape {
                SHAPE_BOX => cube_mesh(),
                _ => icosphere_mesh(),
            })
            .collect();
        Self {
            instances,
            meshes,
            camera,
            lights: Vec::new(),
            selected: Vec::new(),
        }
    }

    /// Append a solid, lit box (`half_extent` metres from its centre to each
    /// face) at the camera-relative `center`, in `color` (linear rgb). The
    /// box is albedo-lit plus a mild emissive term so it stands out from stars.
    pub fn push_box(&mut self, center: [f32; 3], half_extent: f32, color: [f32; 3]) {
        self.instances.push(GpuInstance {
            center: [center[0], center[1], center[2], half_extent],
            albedo: [color[0], color[1], color[2], 0.0],
            emissive: [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, 1.0],
            roughness: 0.5,
            shape: SHAPE_BOX,
            transmission: 0.0,
            _pad1: 0.0,
        });
        self.meshes.push(cube_mesh());
    }

    /// Attach analytic lights (world-space positions).
    #[must_use]
    pub fn with_lights(mut self, lights: Vec<SceneLight>) -> Self {
        self.lights = lights;
        self
    }

    /// The view-projection matrix for the scene's camera pose.
    #[must_use]
    pub fn view_proj(&self) -> Mat4 {
        let eye = [0.0f32; 3]; // instances are camera-relative
        let view = mat4_look_at(
            eye,
            [
                self.camera.forward[0],
                self.camera.forward[1],
                self.camera.forward[2],
            ],
            self.camera.up,
        );
        let proj = mat4_perspective(
            self.camera.fov_y,
            self.camera.aspect,
            self.camera.near,
            self.camera.far,
        );
        mat4_mul(&proj, &view)
    }

    /// `view_proj` inverted; `None` for a degenerate camera.
    #[must_use]
    pub fn inv_view_proj(&self) -> Option<Mat4> {
        mat4_inverse(&self.view_proj())
    }
}
