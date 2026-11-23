//! Meshes built from collision shapes, and the few procedural pieces the
//! physics showcase needs (signs, outlines, the bowl).
//!
//! What the client draws for a physics body or for a static collider of the
//! shared scene definition (`space-scene`) is made from the **same shape the
//! physics uses** ([`shape_mesh`]): a crate's box, a ball's sphere, a chain
//! link's capsule, the bowl's triangles from the shape catalogue. The picture
//! and the collider therefore cannot drift apart.
//!
//! Every mesh here is built once per shape and size and then shared (the
//! renderer keys its GPU copy by the `Arc`): see `crate::bodies::MeshCache`.
//! The solids are closed and wound counter-clockwise seen from outside, so
//! ray-traced shadows treat them as solids.

use space_render::{MeshData, MeshMaterial, MeshPart, MeshVertex, Texture};
use space_sim::shape::{LocalPose, MeshGeometry, ShapeRef};

/// A material from a colour, metalness and roughness.
#[must_use]
pub fn paint(color: [f32; 3], metallic: f32, roughness: f32) -> MeshMaterial {
    MeshMaterial {
        base_color: color,
        metallic,
        roughness,
        emissive: [0.0; 3],
        base_color_texture: None,
    }
}

/// A material that glows with `radiance` (linear rgb) over a dark base.
#[must_use]
pub fn glow(radiance: [f32; 3]) -> MeshMaterial {
    MeshMaterial {
        emissive: radiance,
        ..paint([0.05; 3], 0.0, 0.8)
    }
}

/// Assembles a mesh part by part.
#[derive(Debug, Default)]
pub struct Builder {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    parts: Vec<MeshPart>,
    textures: Vec<Texture>,
    open: Option<(u32, MeshMaterial)>,
}

impl Builder {
    /// An empty mesh.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a texture; its index is for [`MeshMaterial::base_color_texture`].
    pub fn texture(&mut self, texture: Texture) -> u32 {
        self.textures.push(texture);
        (self.textures.len() - 1) as u32
    }

    /// Start a part drawn with `material` (ends the one before).
    pub fn part(&mut self, material: MeshMaterial) {
        self.close();
        self.open = Some((self.indices.len() as u32, material));
    }

    fn close(&mut self) {
        if let Some((first, material)) = self.open.take() {
            let count = self.indices.len() as u32 - first;
            if count > 0 {
                self.parts.push(MeshPart {
                    first_index: first,
                    index_count: count,
                    material,
                });
            }
        }
    }

    /// Add a vertex; returns its index.
    pub fn vertex(&mut self, position: [f32; 3], normal: [f32; 3], uv: [f32; 2]) -> u32 {
        self.vertices.push(MeshVertex {
            position,
            normal,
            uv,
        });
        (self.vertices.len() - 1) as u32
    }

    /// Add a triangle (counter-clockwise seen from the side it faces).
    pub fn triangle(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    /// Add a box centred on `center`, `half` from it to each face, with
    /// flat normals and the texture repeating every `tile` units.
    pub fn cuboid(&mut self, center: [f32; 3], half: [f32; 3], tile: f32) {
        // (normal axis, sign): each face as two triangles.
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let (u_axis, v_axis) = match axis {
                    0 => (2, 1),
                    1 => (0, 2),
                    _ => (0, 1),
                };
                let mut normal = [0.0; 3];
                normal[axis] = sign;
                let corner = |su: f32, sv: f32| {
                    let mut p = center;
                    p[axis] += sign * half[axis];
                    p[u_axis] += su * half[u_axis];
                    p[v_axis] += sv * half[v_axis];
                    let uv = [
                        (su + 1.0) * half[u_axis] / tile,
                        (1.0 - sv) * half[v_axis] / tile,
                    ];
                    (p, uv)
                };
                let quad = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)];
                let base = self.vertices.len() as u32;
                for (su, sv) in quad {
                    let (p, uv) = corner(su, sv);
                    self.vertex(p, normal, uv);
                }
                // u × v points along +normal for (axis, +) when the axes
                // follow x→y→z; flip the winding where it points the other way.
                let mut u = [0.0f32; 3];
                u[u_axis] = 1.0;
                let mut v = [0.0f32; 3];
                v[v_axis] = 1.0;
                let uxv = cross(u, v);
                if dot(uxv, normal) > 0.0 {
                    self.indices
                        .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                } else {
                    self.indices
                        .extend([base, base + 2, base + 1, base, base + 3, base + 2]);
                }
            }
        }
    }

    /// Add a surface of revolution about +Y through `center`: the profile is
    /// a list of `(radius, height, normal_radial, normal_y)` from the bottom
    /// pole to the top pole, `segments` around. The texture's u runs around
    /// (`u_repeat` times), v down the profile.
    pub fn revolve(
        &mut self,
        center: [f32; 3],
        profile: &[[f32; 4]],
        segments: u32,
        u_repeat: f32,
    ) {
        let rows = profile.len();
        let base = self.vertices.len() as u32;
        let around = segments + 1;
        for (r, p) in profile.iter().enumerate() {
            for j in 0..=segments {
                // The seam column repeats the first one's position exactly,
                // so the surface is closed.
                let phi = std::f32::consts::TAU * (j % segments) as f32 / segments as f32;
                let (s, c) = phi.sin_cos();
                let position = [center[0] + p[0] * c, center[1] + p[1], center[2] + p[0] * s];
                let normal = normalize([p[2] * c, p[3], p[2] * s]);
                let uv = [
                    u_repeat * j as f32 / segments as f32,
                    1.0 - r as f32 / (rows - 1) as f32,
                ];
                self.vertex(position, normal, uv);
            }
        }
        for r in 0..rows as u32 - 1 {
            for j in 0..segments {
                let a = base + r * around + j;
                let b = a + around;
                // Bottom to top, counter-clockwise seen from outside.
                self.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
    }

    /// Add the triangles of `geometry` (positions in metres, scaled by
    /// `scale`), smooth-shaded, with the texture projected from above every
    /// `tile` metres; `flip` reverses their winding.
    pub fn triangles(&mut self, geometry: &MeshGeometry, scale: f32, tile: f32, flip: bool) {
        let base = self.vertices.len() as u32;
        let mut normals = vec![[0.0f32; 3]; geometry.vertices.len()];
        let position = |i: u32| geometry.vertices[i as usize].map(|c| c as f32 * scale);
        for t in &geometry.indices {
            let [a, b, c] = t.map(position);
            let n = cross(sub(b, a), sub(c, a));
            let n = if flip { n.map(|v| -v) } else { n };
            for i in t {
                for k in 0..3 {
                    normals[*i as usize][k] += n[k];
                }
            }
        }
        for (i, v) in geometry.vertices.iter().enumerate() {
            let p = v.map(|c| c as f32 * scale);
            self.vertex(p, normalize(normals[i]), [p[0] / tile, p[2] / tile]);
        }
        for t in &geometry.indices {
            let [a, b, c] = t.map(|i| base + i);
            if flip {
                self.triangle(a, c, b);
            } else {
                self.triangle(a, b, c);
            }
        }
    }

    /// The mesh, with its bounds. `None` for one without a triangle.
    #[must_use]
    pub fn finish(mut self) -> Option<MeshData> {
        self.close();
        if self.indices.is_empty() {
            return None;
        }
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for v in &self.vertices {
            for k in 0..3 {
                lo[k] = lo[k].min(v.position[k]);
                hi[k] = hi[k].max(v.position[k]);
            }
        }
        Some(MeshData {
            vertices: self.vertices,
            indices: self.indices,
            parts: self.parts,
            textures: self.textures,
            bounds_min: lo,
            bounds_max: hi,
        })
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = dot(v, v).sqrt();
    if l > 0.0 && l.is_finite() {
        v.map(|c| c / l)
    } else {
        [0.0, 1.0, 0.0]
    }
}

/// The profile of a sphere of `radius` (see [`Builder::revolve`]), `rings`
/// bands from pole to pole, its centre `lift` above the revolve centre.
fn sphere_profile(radius: f32, rings: u32, lift: f32) -> Vec<[f32; 4]> {
    (0..=rings)
        .map(|i| {
            let a = std::f32::consts::PI * i as f32 / rings as f32;
            let (s, c) = a.sin_cos();
            // From the bottom pole (a = 0 → y = -radius) up.
            let (r, y) = if i == 0 || i == rings {
                (0.0, -c * radius)
            } else {
                (s * radius, -c * radius)
            };
            [r, y + lift, s, -c]
        })
        .collect()
}

/// The profile of a capsule along +Y: two half spheres joined by a cylinder.
fn capsule_profile(half_height: f32, radius: f32, rings: u32) -> Vec<[f32; 4]> {
    let half = rings / 2;
    let mut out = Vec::new();
    for i in 0..=half {
        let a = std::f32::consts::FRAC_PI_2 * i as f32 / half as f32;
        let (s, c) = a.sin_cos();
        let r = if i == 0 { 0.0 } else { s * radius };
        out.push([r, -half_height - c * radius, s, -c]);
    }
    for i in 0..=half {
        let a = std::f32::consts::FRAC_PI_2 + std::f32::consts::FRAC_PI_2 * i as f32 / half as f32;
        let (s, c) = a.sin_cos();
        let r = if i == half { 0.0 } else { s * radius };
        out.push([r, half_height - c * radius, s, -c]);
    }
    out
}

/// A solid of `shape` (as `space-sim` builds its collider) in `material`,
/// one part. The texture (if the material names one, the caller adds it)
/// repeats every `tile` metres. `None` for a shape with nothing to draw here
/// (an asset: see [`bowl_mesh`] and the Kestrel's own model).
#[must_use]
pub fn shape_mesh(
    shape: &ShapeRef,
    material: MeshMaterial,
    texture: Option<Texture>,
    tile: f32,
) -> Option<MeshData> {
    let mut b = Builder::new();
    let material = match texture {
        Some(t) => MeshMaterial {
            base_color_texture: Some(b.texture(t)),
            ..material
        },
        None => material,
    };
    b.part(material);
    add_shape(&mut b, shape, LocalPose::IDENTITY, tile)?;
    b.finish()
}

/// Add `shape` at `pose` to `b`; `None` if it holds nothing drawable.
fn add_shape(b: &mut Builder, shape: &ShapeRef, pose: LocalPose, tile: f32) -> Option<()> {
    let start = b.vertices.len();
    match shape {
        ShapeRef::Ball { radius } => {
            let rings = sphere_rings(*radius);
            b.revolve(
                [0.0; 3],
                &sphere_profile(*radius, rings, 0.0),
                rings * 2,
                2.0,
            );
        }
        ShapeRef::Cuboid { half_extents } => {
            b.cuboid([0.0; 3], *half_extents, tile);
        }
        ShapeRef::Capsule {
            half_height,
            radius,
        } => {
            let rings = sphere_rings(*radius).max(8);
            b.revolve(
                [0.0; 3],
                &capsule_profile(*half_height, *radius, rings),
                rings * 2,
                1.0,
            );
        }
        ShapeRef::Cylinder {
            half_height,
            radius,
        } => {
            let (h, r) = (*half_height, *radius);
            let profile = [
                [0.0, -h, 0.0, -1.0],
                [r, -h, 0.0, -1.0],
                [r, -h, 1.0, 0.0],
                [r, h, 1.0, 0.0],
                [r, h, 0.0, 1.0],
                [0.0, h, 0.0, 1.0],
            ];
            b.revolve([0.0; 3], &profile, 24, 1.0);
        }
        ShapeRef::Cone {
            half_height,
            radius,
        } => {
            let (h, r) = (*half_height, *radius);
            let slant = normalize([2.0 * h, r, 0.0]);
            let profile = [
                [0.0, -h, 0.0, -1.0],
                [r, -h, 0.0, -1.0],
                [r, -h, slant[0], slant[1]],
                [0.0, h, slant[0], slant[1]],
            ];
            b.revolve([0.0; 3], &profile, 24, 1.0);
        }
        ShapeRef::ConvexHull { points } => {
            // Drawn as the box around its points (no hull of the scene
            // needs better yet).
            let mut lo = [f32::MAX; 3];
            let mut hi = [f32::MIN; 3];
            for p in points {
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
            }
            let center = [0, 1, 2].map(|k| 0.5 * (lo[k] + hi[k]));
            let half = [0, 1, 2].map(|k| (0.5 * (hi[k] - lo[k])).max(1e-3));
            b.cuboid(center, half, tile);
        }
        ShapeRef::Compound { parts } => {
            for (child_pose, child) in parts {
                add_shape(b, child, *child_pose, tile);
            }
            return (b.vertices.len() > start).then_some(());
        }
        ShapeRef::Asset(_) => return None,
    }
    if pose != LocalPose::IDENTITY {
        let m = quat_matrix(pose.rotation);
        for v in &mut b.vertices[start..] {
            v.position = add(mul(&m, v.position), pose.translation);
            v.normal = mul(&m, v.normal);
        }
    }
    Some(())
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// Bands from pole to pole for a sphere of `radius`: smooth up close.
fn sphere_rings(radius: f32) -> u32 {
    if radius < 0.3 {
        12
    } else if radius < 2.0 {
        16
    } else {
        24
    }
}

/// The rows of the rotation matrix of a unit quaternion `(w, x, y, z)` (the
/// identity for one that is not finite or has no length).
#[must_use]
pub fn quat_matrix(quat: [f32; 4]) -> [[f32; 3]; 3] {
    let length = quat.iter().map(|v| v * v).sum::<f32>().sqrt();
    if !length.is_finite() || length < 1e-6 {
        return space_render::IDENTITY_ROTATION;
    }
    let [w, x, y, z] = quat.map(|v| v / length);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// The rotation (rows) that turns +Y onto the unit vector `axis`.
#[must_use]
pub fn y_onto(axis: [f32; 3]) -> [[f32; 3]; 3] {
    let a = normalize(axis);
    // Rodrigues: rotate +Y onto `a` about +Y × a.
    let c = a[1];
    if c > 1.0 - 1e-6 {
        return space_render::IDENTITY_ROTATION;
    }
    if c < -1.0 + 1e-6 {
        return [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];
    }
    let k = [a[2], 0.0, -a[0]]; // (0,1,0) × a
    let s = (k[0] * k[0] + k[2] * k[2]).sqrt();
    let k = k.map(|v| v / s);
    let t = 1.0 - c;
    let [kx, ky, kz] = k;
    [
        [t * kx * kx + c, t * kx * ky - s * kz, t * kx * kz + s * ky],
        [t * kx * ky + s * kz, t * ky * ky + c, t * ky * kz - s * kx],
        [t * kx * kz - s * ky, t * ky * kz + s * kx, t * kz * kz + c],
    ]
}

/// The showcase's concave bowl as a solid: the catalogue's inner surface
/// (exactly the collider), an outer shell `thickness` below it and the rim
/// that joins them. `material` with a checker texture projected from above,
/// so its curve reads.
#[must_use]
pub fn bowl_mesh(
    geometry: &MeshGeometry,
    thickness: f32,
    material: MeshMaterial,
    texture: Texture,
) -> Option<MeshData> {
    let mut b = Builder::new();
    let slot = b.texture(texture);
    b.part(MeshMaterial {
        base_color_texture: Some(slot),
        ..material
    });
    // Inside: the collider's triangles as they are (wound to face up/in).
    b.triangles(geometry, 1.0, 0.5, false);
    // Outside: the same surface pushed down by `thickness`, facing out.
    let shell = MeshGeometry {
        vertices: geometry
            .vertices
            .iter()
            .map(|v| [v[0], v[1] - f64::from(thickness), v[2]])
            .collect(),
        indices: geometry.indices.clone(),
    };
    b.triangles(&shell, 1.0, 0.5, true);
    // The rim: the outermost ring of the inside joined to the shell's.
    let (_, hi) = geometry.bounds();
    let top = hi[1];
    let rim: Vec<(usize, [f64; 3])> = {
        let mut ring: Vec<(usize, [f64; 3])> = geometry
            .vertices
            .iter()
            .enumerate()
            .filter(|(_, v)| (v[1] - top).abs() < 1e-9)
            .map(|(i, v)| (i, *v))
            .collect();
        ring.sort_by(|a, b| a.1[2].atan2(a.1[0]).total_cmp(&b.1[2].atan2(b.1[0])));
        ring
    };
    if rim.len() >= 3 {
        let n = rim.len();
        for k in 0..n {
            let (_, p) = rim[k];
            let (_, q) = rim[(k + 1) % n];
            let out = normalize([p[0] as f32, 0.0, p[2] as f32]);
            // Exactly the inside's and the shell's rim vertices (the shell's
            // heights are taken in f64, as the shell's are).
            let top = |v: [f64; 3]| v.map(|c| c as f32);
            let low = |v: [f64; 3]| {
                [
                    v[0] as f32,
                    (v[1] - f64::from(thickness)) as f32,
                    v[2] as f32,
                ]
            };
            let (p_top, q_top, p_low, q_low) = (top(p), top(q), low(p), low(q));
            let a = b.vertex(p_low, out, [0.0, 1.0]);
            let c = b.vertex(p_top, out, [0.0, 0.0]);
            let d = b.vertex(q_top, out, [1.0, 0.0]);
            let e = b.vertex(q_low, out, [1.0, 1.0]);
            // Counter-clockwise seen from outside, and against the
            // direction the inside's and the shell's boundary edges run.
            b.triangle(a, c, d);
            b.triangle(a, d, e);
        }
    }
    b.finish()
}

/// A sign: `text` in glowing capitals (the overlay's 5x7 font, one dot
/// `dot` metres) on a dark panel `depth` thick, facing +Z, centred on the
/// origin. `radiance` is the letters' glow.
#[must_use]
pub fn sign_mesh(text: &str, dot: f32, depth: f32, radiance: [f32; 3]) -> Option<MeshData> {
    let chars: Vec<char> = text.chars().collect();
    let advance = 6.0 * dot;
    let width = chars.len() as f32 * advance - dot;
    let height = 7.0 * dot;
    let margin = 2.0 * dot;
    let mut b = Builder::new();
    b.part(paint([0.04, 0.045, 0.06], 0.3, 0.6));
    b.cuboid(
        [0.0, 0.0, -depth * 0.5],
        [width * 0.5 + margin, height * 0.5 + margin, depth * 0.5],
        1.0,
    );
    b.part(glow(radiance));
    let left = -width * 0.5;
    let top = height * 0.5;
    for (i, ch) in chars.iter().enumerate() {
        let rows = space_render::overlay::glyph(ch.to_ascii_uppercase());
        for (y, row) in rows.iter().enumerate() {
            // Runs of lit dots in a row become one bar.
            let mut x = 0;
            while x < 5 {
                if row >> (4 - x) & 1 == 0 {
                    x += 1;
                    continue;
                }
                let start = x;
                while x < 5 && row >> (4 - x) & 1 == 1 {
                    x += 1;
                }
                let run = (x - start) as f32;
                let cx = left + i as f32 * advance + (start as f32 + run * 0.5) * dot;
                let cy = top - (y as f32 + 0.5) * dot;
                b.cuboid(
                    [cx, cy, dot * 0.15],
                    [run * dot * 0.5, dot * 0.45, dot * 0.15],
                    1.0,
                );
            }
        }
    }
    b.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn closed_and_outward(mesh: &MeshData) {
        mesh.validate().expect("valid");
        assert!(mesh.is_closed(), "closed");
        assert!(mesh.signed_volume() > 0.0, "wound outward");
    }

    /// Every primitive becomes a closed, outward-wound solid whose bounds are
    /// the shape's: what is drawn is the collider.
    #[test]
    fn shapes_become_closed_solids_of_their_size() {
        let m = paint([1.0; 3], 0.0, 0.5);
        let cases: [(ShapeRef, [f32; 3]); 5] = [
            (ShapeRef::Ball { radius: 0.25 }, [0.25; 3]),
            (ShapeRef::cuboid([1.0, 0.4, 0.6]), [0.5, 0.2, 0.3]),
            (
                ShapeRef::Capsule {
                    half_height: 0.15,
                    radius: 0.05,
                },
                [0.05, 0.2, 0.05],
            ),
            (
                ShapeRef::Cylinder {
                    half_height: 0.3,
                    radius: 0.1,
                },
                [0.1, 0.3, 0.1],
            ),
            (
                ShapeRef::Cone {
                    half_height: 0.3,
                    radius: 0.1,
                },
                [0.1, 0.3, 0.1],
            ),
        ];
        for (shape, half) in cases {
            let mesh = shape_mesh(&shape, m, None, 1.0).expect("a mesh");
            closed_and_outward(&mesh);
            for k in 0..3 {
                assert!(
                    (mesh.bounds_max[k] - half[k]).abs() < 1e-3,
                    "{shape:?} {mesh:?}"
                );
                assert!((mesh.bounds_min[k] + half[k]).abs() < 1e-3, "{shape:?}");
            }
        }
        // A compound is its children, placed.
        let compound = ShapeRef::Compound {
            parts: vec![
                (LocalPose::at([1.0, 0.0, 0.0]), ShapeRef::cuboid([0.2; 3])),
                (
                    LocalPose::at([-1.0, 0.0, 0.0]),
                    ShapeRef::Ball { radius: 0.1 },
                ),
            ],
        };
        let mesh = shape_mesh(&compound, m, None, 1.0).expect("a mesh");
        assert!((mesh.bounds_max[0] - 1.1).abs() < 1e-4 && (mesh.bounds_min[0] + 1.1).abs() < 1e-4);
        assert!(shape_mesh(&ShapeRef::Asset(space_sim::AssetId(2)), m, None, 1.0).is_none());
    }

    /// The bowl is the catalogue's surface plus a shell and a rim: a closed
    /// solid whose inside is exactly the collider's triangles.
    #[test]
    fn the_bowl_is_its_collider_made_solid() {
        let geometry = space_scene::bowl_mesh();
        let mesh = bowl_mesh(
            &geometry,
            0.06,
            paint([1.0; 3], 0.0, 0.6),
            Texture::checkerboard(64, 4, [200; 3], [120; 3]),
        )
        .expect("a mesh");
        closed_and_outward(&mesh);
        for (v, g) in mesh.vertices.iter().zip(&geometry.vertices) {
            for k in 0..3 {
                assert!((f64::from(v.position[k]) - g[k]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn signs_read_and_rotations_turn() {
        let sign = sign_mesh("GRAVITY", 0.05, 0.04, [0.2, 1.0, 2.0]).expect("a sign");
        sign.validate().expect("valid");
        assert_eq!(sign.parts.len(), 2);
        assert!(sign.parts[1].material.emissive[2] > 1.0);
        // 7 letters of 6 dots, minus the last gap, plus the margins.
        assert!((sign.bounds_max[0] - (0.05 * 41.0 * 0.5 + 0.1)).abs() < 1e-4);
        let r = y_onto([1.0, 0.0, 0.0]);
        let up = mul(&r, [0.0, 1.0, 0.0]);
        assert!((up[0] - 1.0).abs() < 1e-6 && up[1].abs() < 1e-6, "{up:?}");
        let down = mul(&y_onto([0.0, -1.0, 0.0]), [0.0, 1.0, 0.0]);
        assert!((down[1] + 1.0).abs() < 1e-6);
        let q = quat_matrix([0.5f32.cos(), 0.0, 0.5f32.sin(), 0.0]);
        let want = space_render::rotation_y(1.0);
        for r in 0..3 {
            for c in 0..3 {
                assert!((q[r][c] - want[r][c]).abs() < 1e-6);
            }
        }
    }
}
