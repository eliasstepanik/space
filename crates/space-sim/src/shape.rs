//! Collision shapes: [`ShapeRef`] (what a [`crate::Body`] carries on the wire)
//! and the [`ShapeCatalog`] of big meshes compiled into the binaries.
//!
//! # What a shape can be
//!
//! | [`ShapeRef`] | parry shape | notes |
//! |---|---|---|
//! | `Ball`, `Cuboid`, `Capsule`, `Cylinder`, `Cone` | the primitive | capsule, cylinder and cone are along local **+Y** |
//! | `ConvexHull` | convex hull of ≤ [`MAX_HULL_POINTS`] points | the points must span a volume |
//! | `Compound` | compound of ≤ [`MAX_COMPOUND_PARTS`] children | children are primitives or hulls, never nested |
//! | `Asset(id)` | a [`ShapeCatalog`] entry | **static/kinematic/ghost**: the triangle mesh (concave); **dynamic**: its convex decomposition |
//!
//! Concave geometry is a triangle mesh, which a rigid-body solver can only
//! use for bodies that do not respond to contacts (the floor, a bowl, a
//! parked ship). A *dynamic* concave body is approximated by the VHACD convex
//! decomposition of its mesh ([`CatalogAsset::convex_decomposition`]),
//! computed once per asset on first use and cached; the parameters
//! ([`DECOMPOSITION`]) bound the work.
//!
//! # Bounds (untrusted input)
//!
//! A `ShapeRef` arrives over the network (phase 2), so its decoder
//! ([`decode_shape`]) is total and every size is capped: dimensions in
//! [`MIN_DIM`]`..=`[`MAX_DIM`] metres, offsets and hull points within
//! [`MAX_OFFSET`], at most [`MAX_HULL_POINTS`] points and
//! [`MAX_COMPOUND_PARTS`] children, no nesting. A hull whose points do not span
//! a volume is rejected at decode time, so building the parry shape cannot
//! fail later. Big meshes never travel: they are named by [`AssetId`] and
//! looked up in the catalogue both ends were built with.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

use rapier3d_f64::parry::shape::SharedShape;
use rapier3d_f64::parry::transformation::vhacd::VHACDParameters;
use rapier3d_f64::prelude::{Pose, Rotation, Vector};
use space_protocol::CodecError;

use crate::codec::{check_quat, invalid, Reader, Writer};

/// Smallest radius / half-extent / half-height a primitive may have, metres.
pub const MIN_DIM: f32 = 1.0e-3;
/// Largest radius / half-extent / half-height a primitive may have, metres.
pub const MAX_DIM: f32 = 1.0e4;
/// Largest coordinate of a hull point or compound offset, metres.
pub const MAX_OFFSET: f32 = 1.0e4;
/// Most points a [`ShapeRef::ConvexHull`] may have.
pub const MAX_HULL_POINTS: usize = 64;
/// Most children a [`ShapeRef::Compound`] may have.
pub const MAX_COMPOUND_PARTS: usize = 16;
/// Most triangles a catalogue mesh may have.
pub const MAX_CATALOG_TRIANGLES: usize = 1 << 20;
/// Largest dimension or offset of a **static** collider's shape, metres. A
/// static is compiled into the scene definition and never decoded from the
/// network, so it may be as big as a star (a planet is a static ball); the
/// wire bounds ([`MAX_DIM`]) apply to bodies only.
pub const MAX_STATIC_DIM: f64 = 1.0e10;

/// The id of a [`ShapeCatalog`] entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AssetId(pub u32);

/// A rigid transform relative to a body's frame (or, for a static collider,
/// relative to its own placement).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalPose {
    /// Offset, metres.
    pub translation: [f32; 3],
    /// Rotation as a unit quaternion `(w, x, y, z)`.
    pub rotation: [f32; 4],
}

impl LocalPose {
    /// No offset, no rotation.
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        rotation: [1.0, 0.0, 0.0, 0.0],
    };

    /// A pure offset.
    #[must_use]
    pub const fn at(translation: [f32; 3]) -> Self {
        Self {
            translation,
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// An offset and a rotation of `radians` about local +Y (the yaw
    /// convention of the client's `rotation_y`).
    #[must_use]
    pub fn yawed(translation: [f32; 3], radians: f32) -> Self {
        let (s, c) = (radians * 0.5).sin_cos();
        Self {
            translation,
            rotation: [c, 0.0, s, 0.0],
        }
    }

    /// The parry/rapier pose.
    #[must_use]
    pub fn to_pose(self) -> Pose {
        Pose::from_parts(
            Vector::new(
                f64::from(self.translation[0]),
                f64::from(self.translation[1]),
                f64::from(self.translation[2]),
            ),
            quat_to_rot(self.rotation),
        )
    }
}

impl Default for LocalPose {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// `(w, x, y, z)` `f32` quaternion to a normalised `f64` rotation.
#[must_use]
pub fn quat_to_rot(q: [f32; 4]) -> Rotation {
    Rotation::from_xyzw(
        f64::from(q[1]),
        f64::from(q[2]),
        f64::from(q[3]),
        f64::from(q[0]),
    )
    .normalize()
}

/// A collision shape, as a [`crate::Body`] or a static collider names it.
/// See the [module docs](self).
#[derive(Clone, Debug, PartialEq)]
pub enum ShapeRef {
    /// A sphere.
    Ball {
        /// Radius, metres.
        radius: f32,
    },
    /// A box.
    Cuboid {
        /// Half the edge lengths, metres.
        half_extents: [f32; 3],
    },
    /// A capsule along local +Y.
    Capsule {
        /// Half the length of the core segment, metres.
        half_height: f32,
        /// Radius, metres.
        radius: f32,
    },
    /// A cylinder along local +Y.
    Cylinder {
        /// Half the height, metres.
        half_height: f32,
        /// Radius, metres.
        radius: f32,
    },
    /// A cone along local +Y (apex up).
    Cone {
        /// Half the height, metres.
        half_height: f32,
        /// Base radius, metres.
        radius: f32,
    },
    /// The convex hull of a bounded point list.
    ConvexHull {
        /// Hull points in the body frame, metres (4..=[`MAX_HULL_POINTS`]).
        points: Vec<[f32; 3]>,
    },
    /// Several children, each a primitive or a hull, at a local pose.
    Compound {
        /// The children (1..=[`MAX_COMPOUND_PARTS`]).
        parts: Vec<(LocalPose, ShapeRef)>,
    },
    /// A shape from the [`ShapeCatalog`].
    Asset(AssetId),
}

/// Why a shape could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShapeError {
    /// The [`ShapeRef`] is outside the bounds of the [module docs](self).
    Invalid(String),
    /// An [`AssetId`] the catalogue does not have.
    UnknownAsset(AssetId),
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "invalid shape: {why}"),
            Self::UnknownAsset(id) => write!(f, "unknown shape asset {}", id.0),
        }
    }
}

impl std::error::Error for ShapeError {}

impl From<CodecError> for ShapeError {
    fn from(e: CodecError) -> Self {
        Self::Invalid(e.to_string())
    }
}

// Wire tags of the shape variants (append-only).
const TAG_BALL: u8 = 1;
const TAG_CUBOID: u8 = 2;
const TAG_CAPSULE: u8 = 3;
const TAG_CYLINDER: u8 = 4;
const TAG_CONE: u8 = 5;
const TAG_HULL: u8 = 6;
const TAG_COMPOUND: u8 = 7;
const TAG_ASSET: u8 = 8;

impl ShapeRef {
    /// A box from its full edge lengths.
    #[must_use]
    pub fn cuboid(size: [f32; 3]) -> Self {
        Self::Cuboid {
            half_extents: [size[0] * 0.5, size[1] * 0.5, size[2] * 0.5],
        }
    }

    /// Build the shape of a **static** collider: the same shapes, but the
    /// size bound is [`MAX_STATIC_DIM`] instead of the wire's [`MAX_DIM`]
    /// (counts and the no-nesting rule still hold), and an asset is used as its
    /// triangle mesh.
    ///
    /// # Errors
    /// As [`ShapeRef::build`].
    pub fn build_static(&self, catalog: &ShapeCatalog) -> Result<SharedShape, ShapeError> {
        self.check_static(0)?;
        self.build_unchecked(catalog, false)
    }

    fn check_static(&self, depth: u8) -> Result<(), ShapeError> {
        let dim = |v: f32| {
            let v = f64::from(v);
            if v.is_finite() && v >= f64::from(MIN_DIM) && v <= MAX_STATIC_DIM {
                Ok(())
            } else {
                Err(ShapeError::Invalid(format!(
                    "static dimension {v} out of range"
                )))
            }
        };
        let off = |v: f32| {
            if v.is_finite() && f64::from(v).abs() <= MAX_STATIC_DIM {
                Ok(())
            } else {
                Err(ShapeError::Invalid(format!(
                    "static offset {v} out of range"
                )))
            }
        };
        match self {
            Self::Ball { radius } => dim(*radius),
            Self::Cuboid { half_extents } => half_extents.iter().try_for_each(|v| dim(*v)),
            Self::Capsule {
                half_height,
                radius,
            }
            | Self::Cylinder {
                half_height,
                radius,
            }
            | Self::Cone {
                half_height,
                radius,
            } => dim(*half_height).and(dim(*radius)),
            Self::ConvexHull { points } => {
                if !(4..=MAX_HULL_POINTS).contains(&points.len()) {
                    return Err(ShapeError::Invalid("hull point count".into()));
                }
                points.iter().flatten().try_for_each(|v| off(*v))?;
                if spans_volume(points) {
                    Ok(())
                } else {
                    Err(ShapeError::Invalid("degenerate convex hull".into()))
                }
            }
            Self::Compound { parts } => {
                if depth > 0 || parts.is_empty() || parts.len() > MAX_COMPOUND_PARTS {
                    return Err(ShapeError::Invalid("compound shape out of bounds".into()));
                }
                for (pose, child) in parts {
                    pose.translation.iter().try_for_each(|v| off(*v))?;
                    let n2 = pose.rotation.iter().map(|c| c * c).sum::<f32>();
                    if !(0.5..=2.0).contains(&n2) {
                        return Err(ShapeError::Invalid("compound rotation is not unit".into()));
                    }
                    if matches!(child, Self::Asset(_)) {
                        return Err(ShapeError::Invalid(
                            "a compound child cannot be an asset".into(),
                        ));
                    }
                    child.check_static(1)?;
                }
                Ok(())
            }
            Self::Asset(_) => Ok(()),
        }
    }

    /// Check the bounds of the [module docs](self) by encoding and decoding
    /// (the decoder is the one validator).
    ///
    /// # Errors
    /// [`ShapeError::Invalid`] naming the first violated bound.
    pub fn validate(&self) -> Result<(), ShapeError> {
        let mut w = Writer::new();
        encode_shape(&mut w, self);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes, "ShapeRef");
        decode_shape(&mut r)?;
        r.finish()?;
        Ok(())
    }

    /// The parry shape for a body of this shape. `dynamic` selects the convex
    /// decomposition of a catalogue mesh instead of the mesh itself.
    ///
    /// # Errors
    /// [`ShapeError::UnknownAsset`] for an asset the catalogue lacks,
    /// [`ShapeError::Invalid`] for a shape outside the bounds.
    pub fn build(&self, catalog: &ShapeCatalog, dynamic: bool) -> Result<SharedShape, ShapeError> {
        self.validate()?;
        self.build_unchecked(catalog, dynamic)
    }

    fn build_unchecked(
        &self,
        catalog: &ShapeCatalog,
        dynamic: bool,
    ) -> Result<SharedShape, ShapeError> {
        let d = f64::from;
        Ok(match self {
            Self::Ball { radius } => SharedShape::ball(d(*radius)),
            Self::Cuboid { half_extents: h } => SharedShape::cuboid(d(h[0]), d(h[1]), d(h[2])),
            Self::Capsule {
                half_height,
                radius,
            } => SharedShape::capsule_y(d(*half_height), d(*radius)),
            Self::Cylinder {
                half_height,
                radius,
            } => SharedShape::cylinder(d(*half_height), d(*radius)),
            Self::Cone {
                half_height,
                radius,
            } => SharedShape::cone(d(*half_height), d(*radius)),
            Self::ConvexHull { points } => {
                let pts: Vec<Vector> = points
                    .iter()
                    .map(|p| Vector::new(d(p[0]), d(p[1]), d(p[2])))
                    .collect();
                SharedShape::convex_hull(&pts)
                    .ok_or_else(|| ShapeError::Invalid("degenerate convex hull".into()))?
            }
            Self::Compound { parts } => {
                let mut children = Vec::with_capacity(parts.len());
                for (pose, child) in parts {
                    children.push((pose.to_pose(), child.build_unchecked(catalog, dynamic)?));
                }
                SharedShape::compound(children)
            }
            Self::Asset(id) => {
                let asset = catalog.get(*id).ok_or(ShapeError::UnknownAsset(*id))?;
                if dynamic {
                    asset.convex_decomposition()
                } else {
                    asset.trimesh()
                }
            }
        })
    }
}

/// Append the wire form of `shape` (see [`crate::body`] for the layout).
pub(crate) fn encode_shape(w: &mut Writer, shape: &ShapeRef) {
    match shape {
        ShapeRef::Ball { radius } => {
            w.u8(TAG_BALL);
            w.f32(*radius);
        }
        ShapeRef::Cuboid { half_extents } => {
            w.u8(TAG_CUBOID);
            w.f32s(half_extents);
        }
        ShapeRef::Capsule {
            half_height,
            radius,
        } => {
            w.u8(TAG_CAPSULE);
            w.f32(*half_height);
            w.f32(*radius);
        }
        ShapeRef::Cylinder {
            half_height,
            radius,
        } => {
            w.u8(TAG_CYLINDER);
            w.f32(*half_height);
            w.f32(*radius);
        }
        ShapeRef::Cone {
            half_height,
            radius,
        } => {
            w.u8(TAG_CONE);
            w.f32(*half_height);
            w.f32(*radius);
        }
        ShapeRef::ConvexHull { points } => {
            w.u8(TAG_HULL);
            w.u8(u8::try_from(points.len()).unwrap_or(u8::MAX));
            for p in points {
                w.f32s(p);
            }
        }
        ShapeRef::Compound { parts } => {
            w.u8(TAG_COMPOUND);
            w.u8(u8::try_from(parts.len()).unwrap_or(u8::MAX));
            for (pose, child) in parts {
                encode_pose(w, pose);
                encode_shape(w, child);
            }
        }
        ShapeRef::Asset(id) => {
            w.u8(TAG_ASSET);
            w.u32(id.0);
        }
    }
}

pub(crate) fn encode_pose(w: &mut Writer, pose: &LocalPose) {
    w.f32s(&pose.translation);
    w.f32s(&pose.rotation);
}

pub(crate) fn decode_pose(r: &mut Reader<'_>) -> Result<LocalPose, CodecError> {
    let translation = r.f32s_in::<3>("pose.translation", -MAX_OFFSET, MAX_OFFSET)?;
    let rotation = r.f32s_in::<4>("pose.rotation", -2.0, 2.0)?;
    check_quat(r.what(), rotation)?;
    Ok(LocalPose {
        translation,
        rotation,
    })
}

/// Decode one shape (the total, bounded decoder of the [module docs](self)).
pub(crate) fn decode_shape(r: &mut Reader<'_>) -> Result<ShapeRef, CodecError> {
    decode_shape_depth(r, 0)
}

fn decode_shape_depth(r: &mut Reader<'_>, depth: u8) -> Result<ShapeRef, CodecError> {
    let dim = |r: &mut Reader<'_>, f: &str| r.f32_in(f, MIN_DIM, MAX_DIM);
    let tag = r.u8()?;
    Ok(match tag {
        TAG_BALL => ShapeRef::Ball {
            radius: dim(r, "radius")?,
        },
        TAG_CUBOID => ShapeRef::Cuboid {
            half_extents: r.f32s_in::<3>("half_extents", MIN_DIM, MAX_DIM)?,
        },
        TAG_CAPSULE | TAG_CYLINDER | TAG_CONE => {
            let half_height = dim(r, "half_height")?;
            let radius = dim(r, "radius")?;
            match tag {
                TAG_CAPSULE => ShapeRef::Capsule {
                    half_height,
                    radius,
                },
                TAG_CYLINDER => ShapeRef::Cylinder {
                    half_height,
                    radius,
                },
                _ => ShapeRef::Cone {
                    half_height,
                    radius,
                },
            }
        }
        TAG_HULL => {
            let n = r.count_u8("hull points", MAX_HULL_POINTS)?;
            if n < 4 {
                return Err(invalid(format!(
                    "{}: a hull needs 4 points, got {n}",
                    r.what()
                )));
            }
            let mut points = Vec::with_capacity(n);
            for _ in 0..n {
                points.push(r.f32s_in::<3>("hull point", -MAX_OFFSET, MAX_OFFSET)?);
            }
            if !spans_volume(&points) {
                return Err(invalid(format!(
                    "{}: the hull points do not span a volume",
                    r.what()
                )));
            }
            ShapeRef::ConvexHull { points }
        }
        TAG_COMPOUND if depth == 0 => {
            let n = r.count_u8("compound parts", MAX_COMPOUND_PARTS)?;
            if n == 0 {
                return Err(invalid(format!("{}: an empty compound", r.what())));
            }
            let mut parts = Vec::with_capacity(n);
            for _ in 0..n {
                let pose = decode_pose(r)?;
                let child = decode_shape_depth(r, 1)?;
                if matches!(child, ShapeRef::Asset(_)) {
                    return Err(invalid(format!(
                        "{}: a compound child cannot be an asset",
                        r.what()
                    )));
                }
                parts.push((pose, child));
            }
            ShapeRef::Compound { parts }
        }
        TAG_COMPOUND => {
            return Err(invalid(format!("{}: nested compound", r.what())));
        }
        TAG_ASSET => ShapeRef::Asset(AssetId(r.u32()?)),
        other => return Err(invalid(format!("{}: unknown shape tag {other}", r.what()))),
    })
}

/// Do `points` contain four points spanning a tetrahedron of non-trivial
/// volume (relative to their extent)? Exact enough to keep parry's hull
/// builder away from degenerate (flat, collinear, coincident) input.
fn spans_volume(points: &[[f32; 3]]) -> bool {
    let p: Vec<[f64; 3]> = points
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])])
        .collect();
    let sub = |a: [f64; 3], b: [f64; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let len2 = |a: [f64; 3]| dot(a, a);
    // Extent of the cloud, for a scale-relative tolerance.
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for q in &p {
        for i in 0..3 {
            lo[i] = lo[i].min(q[i]);
            hi[i] = hi[i].max(q[i]);
        }
    }
    let extent = len2(sub(hi, lo)).sqrt();
    if extent < f64::from(MIN_DIM) {
        return false;
    }
    let eps = extent * 1.0e-4;
    let a = p[0];
    // Farthest point from `a`.
    let Some(b) = p
        .iter()
        .copied()
        .max_by(|x, y| len2(sub(*x, a)).total_cmp(&len2(sub(*y, a))))
    else {
        return false;
    };
    let ab = sub(b, a);
    if len2(ab).sqrt() < eps {
        return false;
    }
    // Farthest from the line ab.
    let Some(c) = p
        .iter()
        .copied()
        .max_by(|x, y| len2(cross(ab, sub(*x, a))).total_cmp(&len2(cross(ab, sub(*y, a)))))
    else {
        return false;
    };
    let n = cross(ab, sub(c, a));
    let n_len = len2(n).sqrt();
    if n_len < eps * len2(ab).sqrt() {
        return false;
    }
    // Farthest from the plane abc.
    let height = p
        .iter()
        .map(|x| (dot(n, sub(*x, a)) / n_len).abs())
        .fold(0.0f64, f64::max);
    height >= eps
}

/// Triangle-mesh geometry of a catalogue entry.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshGeometry {
    /// Vertex positions, metres, in the asset's own frame.
    pub vertices: Vec<[f64; 3]>,
    /// Triangles as vertex indices.
    pub indices: Vec<[u32; 3]>,
}

impl MeshGeometry {
    /// The axis-aligned bounds of the vertices: `(min, max)`.
    #[must_use]
    pub fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::MAX; 3];
        let mut hi = [f64::MIN; 3];
        for v in &self.vertices {
            for i in 0..3 {
                lo[i] = lo[i].min(v[i]);
                hi[i] = hi[i].max(v[i]);
            }
        }
        (lo, hi)
    }
}

/// The VHACD parameters a dynamic catalogue body is decomposed with: a
/// voxel resolution of 48 and at most 24 hulls keep one decomposition of an
/// ~12k-triangle mesh to a few seconds on one core, once per asset.
pub const DECOMPOSITION: VHACDParameters = VHACDParameters {
    concavity: 0.01,
    alpha: 0.05,
    beta: 0.05,
    resolution: 48,
    plane_downsampling: 4,
    convex_hull_downsampling: 4,
    fill_mode: rapier3d_f64::parry::transformation::voxelization::FillMode::FloodFill {
        detect_cavities: false,
    },
    convex_hull_approximation: true,
    max_convex_hulls: 24,
};

/// One catalogue entry: a triangle mesh and what is derived from it, each
/// built once on first use.
#[derive(Debug)]
pub struct CatalogAsset {
    name: String,
    mesh: Arc<MeshGeometry>,
    trimesh: OnceLock<SharedShape>,
    decomposition: OnceLock<SharedShape>,
}

impl CatalogAsset {
    /// The asset's name (for logs and tests).
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The source mesh.
    #[must_use]
    pub fn mesh(&self) -> &MeshGeometry {
        &self.mesh
    }

    /// The concave triangle mesh, for static, kinematic and ghost bodies.
    /// Built on first use; every world shares the one `Arc`.
    #[must_use]
    pub fn trimesh(&self) -> SharedShape {
        self.trimesh
            .get_or_init(|| {
                let (v, i) = self.parry_mesh();
                // Validated by `ShapeCatalog::insert`, so this cannot fail.
                SharedShape::trimesh(v, i).unwrap_or_else(|_| SharedShape::ball(0.01))
            })
            .clone()
    }

    /// The convex decomposition ([`DECOMPOSITION`]), for dynamic bodies.
    /// Computed on first use and cached.
    #[must_use]
    pub fn convex_decomposition(&self) -> SharedShape {
        self.decomposition
            .get_or_init(|| {
                let (v, i) = self.parry_mesh();
                SharedShape::convex_decomposition_with_params(&v, &i, &DECOMPOSITION)
            })
            .clone()
    }

    fn parry_mesh(&self) -> (Vec<Vector>, Vec<[u32; 3]>) {
        (
            self.mesh
                .vertices
                .iter()
                .map(|v| Vector::new(v[0], v[1], v[2]))
                .collect(),
            self.mesh.indices.clone(),
        )
    }
}

/// The shapes compiled into the binaries, by [`AssetId`]. Built once per
/// process (by `space-scene`) and shared by every world as an `Arc`.
#[derive(Debug, Default)]
pub struct ShapeCatalog {
    assets: BTreeMap<AssetId, CatalogAsset>,
}

impl ShapeCatalog {
    /// An empty catalogue.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mesh under `id`.
    ///
    /// # Errors
    /// [`ShapeError::Invalid`] for a duplicate id, an empty or oversized mesh,
    /// a non-finite vertex or an index past the vertex list.
    pub fn insert(
        &mut self,
        id: AssetId,
        name: impl Into<String>,
        mesh: MeshGeometry,
    ) -> Result<(), ShapeError> {
        if self.assets.contains_key(&id) {
            return Err(ShapeError::Invalid(format!("duplicate asset id {}", id.0)));
        }
        if mesh.indices.is_empty() || mesh.indices.len() > MAX_CATALOG_TRIANGLES {
            return Err(ShapeError::Invalid(format!(
                "asset {}: {} triangles (1..={MAX_CATALOG_TRIANGLES})",
                id.0,
                mesh.indices.len()
            )));
        }
        if !mesh.vertices.iter().flatten().all(|c| c.is_finite()) {
            return Err(ShapeError::Invalid(format!(
                "asset {}: non-finite vertex",
                id.0
            )));
        }
        let n = mesh.vertices.len();
        if mesh
            .indices
            .iter()
            .flatten()
            .any(|&i| usize::try_from(i).map_or(true, |i| i >= n))
        {
            return Err(ShapeError::Invalid(format!(
                "asset {}: index past the vertex list",
                id.0
            )));
        }
        let parry: Vec<Vector> = mesh
            .vertices
            .iter()
            .map(|v| Vector::new(v[0], v[1], v[2]))
            .collect();
        let trimesh = SharedShape::trimesh(parry, mesh.indices.clone())
            .map_err(|e| ShapeError::Invalid(format!("asset {}: {e:?}", id.0)))?;
        let entry = CatalogAsset {
            name: name.into(),
            mesh: Arc::new(mesh),
            trimesh: OnceLock::new(),
            decomposition: OnceLock::new(),
        };
        let _ = entry.trimesh.set(trimesh);
        self.assets.insert(id, entry);
        Ok(())
    }

    /// The entry for `id`.
    #[must_use]
    pub fn get(&self, id: AssetId) -> Option<&CatalogAsset> {
        self.assets.get(&id)
    }

    /// The ids in the catalogue, ascending.
    pub fn ids(&self) -> impl Iterator<Item = AssetId> + '_ {
        self.assets.keys().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        decode_shape, encode_shape, AssetId, LocalPose, MeshGeometry, ShapeCatalog, ShapeError,
        ShapeRef, MAX_COMPOUND_PARTS, MAX_HULL_POINTS,
    };
    use crate::codec::{Reader, Writer};

    fn round_trip(s: &ShapeRef) -> ShapeRef {
        let mut w = Writer::new();
        encode_shape(&mut w, s);
        let bytes = w.into_bytes();
        let mut r = Reader::new(&bytes, "ShapeRef");
        let back = decode_shape(&mut r).unwrap();
        r.finish().unwrap();
        back
    }

    fn cube_hull() -> Vec<[f32; 3]> {
        let mut pts = Vec::new();
        for x in [-0.5f32, 0.5] {
            for y in [-0.5f32, 0.5] {
                for z in [-0.5f32, 0.5] {
                    pts.push([x, y, z]);
                }
            }
        }
        pts
    }

    #[test]
    fn every_shape_round_trips_and_builds() {
        let catalog = ShapeCatalog::new();
        let shapes = [
            ShapeRef::Ball { radius: 0.5 },
            ShapeRef::cuboid([1.0, 2.0, 3.0]),
            ShapeRef::Capsule {
                half_height: 0.6,
                radius: 0.3,
            },
            ShapeRef::Cylinder {
                half_height: 0.5,
                radius: 0.2,
            },
            ShapeRef::Cone {
                half_height: 0.5,
                radius: 0.2,
            },
            ShapeRef::ConvexHull {
                points: cube_hull(),
            },
            ShapeRef::Compound {
                parts: vec![
                    (
                        LocalPose::at([0.0, 1.0, 0.0]),
                        ShapeRef::Ball { radius: 0.2 },
                    ),
                    (
                        LocalPose::yawed([1.0, 0.0, 0.0], 0.4),
                        ShapeRef::ConvexHull {
                            points: cube_hull(),
                        },
                    ),
                ],
            },
        ];
        for s in &shapes {
            assert_eq!(&round_trip(s), s);
            s.build(&catalog, true).expect("builds");
        }
        assert_eq!(
            ShapeRef::Asset(AssetId(9)).build(&catalog, false).err(),
            Some(ShapeError::UnknownAsset(AssetId(9)))
        );
    }

    #[test]
    fn hostile_shapes_are_rejected() {
        let bad = [
            ShapeRef::Ball { radius: 0.0 },
            ShapeRef::Ball { radius: f32::NAN },
            ShapeRef::Ball { radius: 1.0e9 },
            ShapeRef::Cuboid {
                half_extents: [1.0, -1.0, 1.0],
            },
            // Flat: four coplanar points.
            ShapeRef::ConvexHull {
                points: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            },
            // Too few points.
            ShapeRef::ConvexHull {
                points: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            },
            ShapeRef::ConvexHull {
                points: vec![[0.5; 3]; MAX_HULL_POINTS + 1],
            },
            ShapeRef::Compound { parts: vec![] },
            ShapeRef::Compound {
                parts: vec![
                    (LocalPose::IDENTITY, ShapeRef::Ball { radius: 1.0 });
                    MAX_COMPOUND_PARTS + 1
                ],
            },
            ShapeRef::Compound {
                parts: vec![(
                    LocalPose::IDENTITY,
                    ShapeRef::Compound {
                        parts: vec![(LocalPose::IDENTITY, ShapeRef::Ball { radius: 1.0 })],
                    },
                )],
            },
            ShapeRef::Compound {
                parts: vec![(LocalPose::IDENTITY, ShapeRef::Asset(AssetId(1)))],
            },
            ShapeRef::Compound {
                parts: vec![(
                    LocalPose {
                        translation: [0.0; 3],
                        rotation: [0.0; 4],
                    },
                    ShapeRef::Ball { radius: 1.0 },
                )],
            },
        ];
        for s in &bad {
            assert!(s.validate().is_err(), "{s:?} must be rejected");
        }
        // Arbitrary bytes never panic.
        let mut seed = 0x1234_5678u64;
        for len in 0..200usize {
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                    (seed >> 56) as u8
                })
                .collect();
            let mut r = Reader::new(&bytes, "fuzz");
            let _ = decode_shape(&mut r);
        }
    }

    #[test]
    fn catalogue_validates_meshes() {
        let mut c = ShapeCatalog::new();
        let tri = MeshGeometry {
            vertices: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            indices: vec![[0, 1, 2]],
        };
        c.insert(AssetId(1), "tri", tri.clone()).unwrap();
        assert!(c.insert(AssetId(1), "again", tri.clone()).is_err());
        let mut broken = tri.clone();
        broken.indices = vec![[0, 1, 3]];
        assert!(c.insert(AssetId(2), "broken", broken).is_err());
        let mut nan = tri;
        nan.vertices[0][0] = f64::NAN;
        assert!(c.insert(AssetId(3), "nan", nan).is_err());
        assert_eq!(c.get(AssetId(1)).unwrap().name(), "tri");
        assert!(ShapeRef::Asset(AssetId(1)).build(&c, false).is_ok());
    }
}
