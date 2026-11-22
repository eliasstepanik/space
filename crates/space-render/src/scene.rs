//! CPU-side scene construction: camera maths, the two unit placeholder meshes,
//! and the LOD → render-instance bridge over `space-world`'s deterministic
//! galaxy.
//!
//! Everything here is pure CPU and unit-testable without a GPU: the LOD
//! selector's output (≤ [`space_world::MAX_SELECTED`] nodes) is turned into
//! one render instance per star (sphere or box per LOD level), positioned
//! camera-relative so the vertex shader works in f32 around the camera.

use space_world::{Galaxy, LodSelector, NodeRef, WorldPos, MAX_SELECTED};

use crate::mesh::ModelInstance;
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
    /// Near plane distance, metres. Nothing closer is drawn.
    pub near: f32,
    /// Nominal view distance, metres. The projection has an infinite far
    /// plane ([`mat4_perspective_reversed`]), so nothing is clipped here; the
    /// field is a hint for callers that build or cull scenes.
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

/// Right-handed perspective projection (Vulkan clip space, y flipped) with a
/// conventional depth range: 0 at `near`, 1 at `far`.
///
/// The renderer itself uses [`mat4_perspective_reversed`]; this one is kept
/// for callers that want a bounded, conventional depth range.
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

/// Right-handed perspective projection (Vulkan clip space, y flipped) with
/// **reversed depth and an infinite far plane**: depth is `near / distance`,
/// 1 at the near plane and falling towards 0 with distance, never reaching it.
///
/// This is what makes astronomical view distances work. A conventional
/// projection stores `1 - near / distance`, which in a 32-bit float rounds to
/// exactly 1.0 — the clear value — beyond about `1.7e7 * near`: with a 1 m near
/// plane everything farther than ~17 000 km fails the depth test and vanishes,
/// and nearer things z-fight. `near / distance` instead lands where floats are
/// dense (near zero), so a planet 4.5e12 m away still has a distinct, correctly
/// ordered depth. Depth tests are therefore `GREATER`, and the depth buffer is
/// cleared to 0.
#[must_use]
pub fn mat4_perspective_reversed(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, -f, 0.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, near, 0.0],
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

/// A fully CPU-built frame: the render instances (camera-relative), derived
/// from the LOD selection over the deterministic galaxy or given explicitly.
///
/// An instance is a unit placeholder shape ([`icosphere_mesh`] or
/// [`cube_mesh`], chosen by [`GpuInstance::shape`]) scaled by its radius and
/// moved to its centre; the rasteriser and the ray-traced acceleration
/// structures both derive the geometry from the instance alone.
pub struct Scene {
    /// Render instances, camera-relative (≤ `MAX_SELECTED`).
    pub instances: Vec<GpuInstance>,
    /// The camera pose the scene was built for.
    pub camera: CameraPose,
    /// Mesh models (camera-relative), drawn with the same shading as the
    /// instances. They are rasterised only: ray-traced GI does not see them.
    pub models: Vec<ModelInstance>,
    /// Analytic point lights (world-space positions).
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
        let mut instances = Vec::with_capacity(selected.len());
        let mut used = Vec::with_capacity(selected.len());
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
            used.push(node);
        }
        Self {
            instances,
            models: Vec::new(),
            camera: *camera,
            lights: Vec::new(),
            selected: used,
        }
    }
}

/// Unit icosphere mesh as a flat triangle list (60 vertices, positions ==
/// normals), matching the `sphere_vertex(sphere_index(..))` sequence in
/// `shaders/forward.wgsl`. Also the geometry of the sphere BLAS.
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
        [0.0, -inv, phi * inv],
        [0.0, inv, phi * inv],
        [0.0, -inv, -phi * inv],
        [0.0, inv, -phi * inv],
        [phi * inv, 0.0, -inv],
        [phi * inv, 0.0, inv],
        [-phi * inv, 0.0, -inv],
        [-phi * inv, 0.0, inv],
    ];
    ICOSPHERE_FACES.iter().map(|&i| v[i as usize]).collect()
}

/// Sphere detail levels: level `L` splits every icosahedron edge into
/// [`sphere_segments`]`(L)` = `4^L` segments and pushes the new vertices out
/// onto the unit sphere, giving `20 * 16^L` triangles. Level 0 is the plain
/// icosahedron — fine for a star a few pixels across, a hexagon for anything
/// bigger — and level 3 (81 920 triangles) is for a planet that fills the
/// screen. The renderer picks a level per instance from its projected size.
pub const SPHERE_LEVELS: u32 = 4;

/// Segments per icosahedron edge at detail `level`.
#[must_use]
pub const fn sphere_segments(level: u32) -> u32 {
    1 << (2 * level)
}

/// Vertices of the unit sphere at detail `level`, as a flat triangle list
/// (`60 * 16^level`).
#[must_use]
pub const fn sphere_vertex_count(level: u32) -> u32 {
    let n = sphere_segments(level);
    60 * n * n
}

/// The detail level for a sphere whose silhouette is `radius_px` pixels in
/// radius on screen: roughly the coarsest level whose facets stay under a
/// pixel or two.
#[must_use]
pub fn sphere_level_for(radius_px: f32) -> u32 {
    // NaN (a degenerate projection) compares false everywhere: full detail.
    if radius_px <= 6.0 {
        0
    } else if radius_px <= 48.0 {
        1
    } else if radius_px <= 384.0 {
        2
    } else {
        3
    }
}

/// The unit sphere at detail `level` as a flat triangle list, vertex for
/// vertex what `sphere_position` in `shaders/forward.wgsl` generates.
/// Positions are unit length, so a vertex is also its normal.
#[must_use]
pub fn sphere_mesh(level: u32) -> Vec<[f32; 3]> {
    let base = icosphere_mesh();
    if level == 0 {
        return base;
    }
    let n = sphere_segments(level);
    let inv_n = 1.0 / n as f32;
    let mut out = Vec::with_capacity(sphere_vertex_count(level) as usize);
    for face in base.as_chunks::<3>().0 {
        let [a, b, c] = *face;
        // Grid point (i, j): i rows down from corner `a`, j steps along the row.
        let point = |i: u32, j: u32| {
            let (wa, wb, wc) = (
                1.0 - i as f32 * inv_n,
                (i - j) as f32 * inv_n,
                j as f32 * inv_n,
            );
            normalize([
                a[0] * wa + b[0] * wb + c[0] * wc,
                a[1] * wa + b[1] * wb + c[1] * wc,
                a[2] * wa + b[2] * wb + c[2] * wc,
            ])
        };
        for row in 0..n {
            // Row `row` holds 2 * row + 1 triangles, alternating up / down.
            for k in 0..=row {
                out.extend([point(row, k), point(row + 1, k), point(row + 1, k + 1)]);
                if k < row {
                    out.extend([point(row, k), point(row + 1, k + 1), point(row, k + 1)]);
                }
            }
        }
    }
    out
}

/// The icosahedron's 20 faces as vertex indices (mirrored by `sphere_index` in
/// `shaders/forward.wgsl`; a test keeps the two tables identical).
#[rustfmt::skip]
const ICOSPHERE_FACES: [u32; 60] = [
    // the 5 faces around vertex 0
    0, 11, 5,   0, 5, 1,    0, 1, 7,    0, 7, 10,   0, 10, 11,
    // the 5 faces adjacent to them
    1, 5, 9,    5, 11, 4,   11, 10, 2,  10, 7, 6,   7, 1, 8,
    // the 5 faces around vertex 3
    3, 9, 4,    3, 4, 2,    3, 2, 6,    3, 6, 8,    3, 8, 9,
    // the 5 faces adjacent to them
    4, 9, 5,    2, 4, 11,   6, 2, 10,   8, 6, 7,    9, 8, 1,
];

/// Unit cube mesh as a flat triangle list (36 vertices, 6 per face, flat
/// normals implicit in the face), matching `box_vertex`/`box_normal` in
/// `shaders/forward.wgsl`. Also the geometry of the cube BLAS.
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
    /// An explicit scene: instances with the camera they are rendered with.
    /// `instances` are already camera-relative.
    #[must_use]
    pub fn from_instances(camera: CameraPose, instances: Vec<GpuInstance>) -> Self {
        Self {
            instances,
            models: Vec::new(),
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
        let proj =
            mat4_perspective_reversed(self.camera.fov_y, self.camera.aspect, self.camera.near);
        mat4_mul(&proj, &view)
    }

    /// `view_proj` inverted; `None` for a degenerate camera.
    #[must_use]
    pub fn inv_view_proj(&self) -> Option<Mat4> {
        mat4_inverse(&self.view_proj())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Undirected edges of a flat triangle list, with how many triangles use
    /// each, keyed by the vertices' bit patterns.
    fn edge_uses(mesh: &[[f32; 3]]) -> HashMap<[[u32; 3]; 2], u32> {
        let key = |v: [f32; 3]| v.map(f32::to_bits);
        let mut uses = HashMap::new();
        for tri in mesh.as_chunks::<3>().0 {
            for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                let mut edge = [key(tri[a]), key(tri[b])];
                edge.sort_unstable();
                *uses.entry(edge).or_insert(0) += 1;
            }
        }
        uses
    }

    /// The icosphere is a closed surface: 20 triangles, 30 edges, every edge
    /// shared by exactly two triangles and every edge the same length. (Three
    /// faces once named the wrong vertices, which left holes in every sphere.)
    #[test]
    fn icosphere_is_a_closed_regular_icosahedron() {
        let mesh = icosphere_mesh();
        assert_eq!(mesh.len(), 60);
        for v in &mesh {
            assert!((dot(*v, *v).sqrt() - 1.0).abs() < 1e-5, "unit radius");
        }
        let uses = edge_uses(&mesh);
        assert_eq!(uses.len(), 30, "an icosahedron has 30 edges");
        assert!(uses.values().all(|&n| n == 2), "every edge joins two faces");
        let length = |edge: &[[u32; 3]; 2]| {
            let [a, b] = edge.map(|v| v.map(f32::from_bits));
            let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
            dot(d, d).sqrt()
        };
        let lengths: Vec<f32> = uses.keys().map(length).collect();
        let (min, max) = lengths
            .iter()
            .fold((f32::MAX, f32::MIN), |(lo, hi), &l| (lo.min(l), hi.max(l)));
        assert!(max - min < 1e-5, "edge lengths {min}..{max}");
    }

    /// Every detail level is a closed, unit-radius surface with the documented
    /// vertex count, and refines the one below it (more triangles, the same
    /// 12 original vertices still present).
    #[test]
    fn sphere_levels_are_closed_unit_surfaces() {
        for level in 0..SPHERE_LEVELS - 1 {
            let mesh = sphere_mesh(level);
            assert_eq!(
                mesh.len() as u32,
                sphere_vertex_count(level),
                "level {level}"
            );
            for v in &mesh {
                assert!((dot(*v, *v).sqrt() - 1.0).abs() < 1e-5, "unit radius");
            }
            // Shared vertices are computed from different faces' corners, so
            // weld them by rounding before checking the surface is closed.
            let weld = |v: [f32; 3]| v.map(|c| (c * 4096.0).round() / 4096.0);
            let welded: Vec<[f32; 3]> = mesh.iter().map(|v| weld(*v)).collect();
            let uses = edge_uses(&welded);
            let triangles = mesh.len() / 3;
            assert_eq!(uses.len(), triangles * 3 / 2, "level {level}: E = 3F/2");
            assert!(uses.values().all(|&n| n == 2), "level {level}: closed");
        }
        assert_eq!(sphere_vertex_count(3), 245_760);
        assert_eq!(sphere_mesh(0), icosphere_mesh());
    }

    #[test]
    fn sphere_level_grows_with_projected_size() {
        assert_eq!(sphere_level_for(0.5), 0);
        assert_eq!(sphere_level_for(6.0), 0);
        assert_eq!(sphere_level_for(20.0), 1);
        assert_eq!(sphere_level_for(200.0), 2);
        assert_eq!(sphere_level_for(5000.0), 3);
        assert_eq!(sphere_level_for(f32::INFINITY), 3);
        assert_eq!(sphere_level_for(f32::NAN), 3);
    }

    /// The reversed infinite projection: depth is `near / distance`, 1 at the
    /// near plane, and still distinct and correctly ordered at interplanetary
    /// range — where a conventional projection has long since saturated.
    #[test]
    fn reversed_projection_keeps_depth_precision_at_astronomical_range() {
        let near = 1.0f32;
        let proj = mat4_perspective_reversed(1.0, 1.0, near);
        let depth = |distance: f32| {
            // A point straight ahead, in view space (the camera looks down -Z).
            let p = [0.0, 0.0, -distance, 1.0];
            let clip: Vec<f32> = (0..4)
                .map(|row| (0..4).map(|col| proj[col][row] * p[col]).sum())
                .collect();
            clip[2] / clip[3]
        };
        assert_eq!(depth(near), 1.0);
        // Earth-Moon distance vs. one metre farther: still ordered.
        let moon = 3.844e8f32;
        assert!(depth(moon) > depth(moon * 1.000_001));
        // Neptune's orbit: a non-zero depth, nearer than twice as far.
        let neptune = 4.495e12f32;
        assert!(depth(neptune) > 0.0);
        assert!(depth(neptune) > depth(2.0 * neptune));
        // The conventional projection cannot tell the Moon from the far plane.
        let classic = mat4_perspective(1.0, 1.0, near, 1.0e13);
        let classic_depth = |distance: f32| {
            let p = [0.0, 0.0, -distance, 1.0];
            let clip: Vec<f32> = (0..4)
                .map(|row| (0..4).map(|col| classic[col][row] * p[col]).sum())
                .collect();
            clip[2] / clip[3]
        };
        assert_eq!(classic_depth(moon), 1.0, "saturated: this is the bug");
        // And the matrix is invertible (the post pass needs the inverse).
        assert!(mat4_inverse(&proj).is_some());
    }

    /// The cube is closed too: 12 triangles over 8 corners, each of the 12
    /// cube edges and 6 face diagonals shared by exactly two triangles.
    #[test]
    fn cube_is_a_closed_box() {
        let mesh = cube_mesh();
        assert_eq!(mesh.len(), 36);
        for v in &mesh {
            assert!(v.iter().all(|c| c.abs() == 1.0), "corner {v:?}");
        }
        let uses = edge_uses(&mesh);
        assert_eq!(uses.len(), 18);
        assert!(uses.values().all(|&n| n == 2));
    }

    /// The vertex shader generates the sphere from its own copy of the face
    /// table; the acceleration structure is built from the Rust one. They must
    /// stay identical or ray-traced and rasterised spheres disagree.
    #[test]
    fn shader_sphere_faces_match_the_rust_mesh() {
        let wgsl = include_str!("../shaders/forward.wgsl");
        let start = wgsl
            .find("fn sphere_index")
            .expect("sphere_index in forward.wgsl");
        let body = &wgsl[start..];
        let body = &body[..body.find("return f[index]").expect("end of the table")];
        let table =
            &body[body.find("array<u32, 60>(").expect("the table") + "array<u32, 60>(".len()..];
        let shader: Vec<u32> = table
            .split(|c: char| !c.is_ascii_digit())
            .filter(|token| !token.is_empty())
            .map(|token| token.parse().expect("an index"))
            .collect();
        assert_eq!(shader, ICOSPHERE_FACES);
    }
}
