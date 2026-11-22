//! Triangle meshes and the GLB (binary glTF 2.0) loader.
//!
//! [`MeshData`] is a model ready to draw: one vertex buffer (position +
//! normal), one index buffer, and a list of [`MeshPart`]s — index ranges that
//! each carry a metallic-roughness material. A [`ModelInstance`] places a
//! shared `MeshData` in a [`Scene`](crate::Scene).
//!
//! # What the GLB loader supports
//!
//! [`MeshData::from_glb`] reads a self-contained `.glb` file: the default
//! scene's node hierarchy (matrix or translation/rotation/scale, baked into the
//! vertices), every triangle primitive (lists, strips and fans, indexed or
//! not), `POSITION` and `NORMAL` (normals are generated when a primitive has
//! none), and each material's `baseColorFactor`, `metallicFactor`,
//! `roughnessFactor` and `emissiveFactor` (with
//! `KHR_materials_emissive_strength`).
//!
//! Not supported, by design for now: textures (a textured material renders
//! with its factors only), vertex colours, skins, animations, morph targets,
//! cameras and lights (ignored); sparse accessors, quantised or compressed
//! geometry (`KHR_mesh_quantization`, Draco, meshopt) and buffers outside the
//! file (rejected with an error rather than drawn wrongly).
//!
//! The loader treats the file as untrusted input: every offset, count and
//! index is bounds-checked, sizes are capped, and a malformed file is an
//! [`MeshError`], never a panic.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

/// Why a model could not be loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshError(pub String);

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MeshError {}

fn err<T>(message: impl Into<String>) -> Result<T, MeshError> {
    Err(MeshError(message.into()))
}

/// One mesh vertex: 24 bytes, position then normal (the mesh pipeline's
/// vertex layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

/// A metallic-roughness material (factors only, no textures).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshMaterial {
    /// Base colour, linear rgb.
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    /// Emitted radiance, linear rgb.
    pub emissive: [f32; 3],
}

impl Default for MeshMaterial {
    /// glTF's default material: white, fully metallic, fully rough.
    fn default() -> Self {
        Self {
            base_color: [1.0; 3],
            metallic: 1.0,
            roughness: 1.0,
            emissive: [0.0; 3],
        }
    }
}

/// A run of indices drawn with one material.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshPart {
    pub first_index: u32,
    pub index_count: u32,
    pub material: MeshMaterial,
}

/// A model: shared vertices, a triangle-list index buffer and its parts.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<MeshVertex>,
    /// Triangle list; every index is `< vertices.len()`.
    pub indices: Vec<u32>,
    pub parts: Vec<MeshPart>,
    /// Axis-aligned bounds of the vertices, in model space.
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// A [`MeshData`] placed in a scene.
#[derive(Clone, Debug)]
pub struct ModelInstance {
    pub mesh: Arc<MeshData>,
    /// Where the model's origin is, camera-relative, metres.
    pub position: [f32; 3],
    /// Orientation: the rows of an orthonormal 3x3 rotation matrix.
    pub rotation: [[f32; 3]; 3],
    /// Uniform scale from model units to metres.
    pub scale: f32,
}

/// The 3x3 identity rotation.
pub const IDENTITY_ROTATION: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// A rotation of `radians` about +Y (positive turns +Z towards +X).
#[must_use]
pub fn rotation_y(radians: f32) -> [[f32; 3]; 3] {
    let (s, c) = radians.sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Largest file [`MeshData::from_glb`] accepts, and the cap on what it will
/// allocate for vertices and indices. Far above any sensible model, far below
/// "exhaust memory because a header said so".
pub const MAX_GLB_BYTES: usize = 1 << 30;
const MAX_ELEMENTS: usize = 64_000_000;
/// Deepest node hierarchy followed (a cycle guard as much as a limit).
const MAX_NODE_DEPTH: usize = 128;

const GLB_MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A; // "JSON"
const CHUNK_BIN: u32 = 0x004E_4942; // "BIN\0"

const COMPONENT_U8: u64 = 5121;
const COMPONENT_U16: u64 = 5123;
const COMPONENT_U32: u64 = 5125;
const COMPONENT_F32: u64 = 5126;

const MODE_TRIANGLES: u64 = 4;
const MODE_TRIANGLE_STRIP: u64 = 5;
const MODE_TRIANGLE_FAN: u64 = 6;

/// Extensions a file may *require* and still load correctly.
const SUPPORTED_REQUIRED_EXTENSIONS: &[&str] = &["KHR_materials_emissive_strength"];

impl MeshData {
    /// Read a `.glb` file from disk.
    pub fn load(path: &Path) -> Result<Self, MeshError> {
        let size = std::fs::metadata(path)
            .map_err(|e| MeshError(format!("{}: {e}", path.display())))?
            .len();
        if size > MAX_GLB_BYTES as u64 {
            return err(format!(
                "{}: {size} bytes is larger than the {MAX_GLB_BYTES}-byte limit",
                path.display()
            ));
        }
        let bytes =
            std::fs::read(path).map_err(|e| MeshError(format!("{}: {e}", path.display())))?;
        Self::from_glb(&bytes).map_err(|e| MeshError(format!("{}: {e}", path.display())))
    }

    /// Parse the bytes of a `.glb` file.
    pub fn from_glb(bytes: &[u8]) -> Result<Self, MeshError> {
        if bytes.len() > MAX_GLB_BYTES {
            return err("file is larger than the size limit");
        }
        let (json, bin) = split_chunks(bytes)?;
        let doc: Value = serde_json::from_slice(json)
            .map_err(|e| MeshError(format!("the JSON chunk is not valid JSON: {e}")))?;
        for required in array(&doc["extensionsRequired"]) {
            let name = required.as_str().unwrap_or("?");
            if !SUPPORTED_REQUIRED_EXTENSIONS.contains(&name) {
                return err(format!(
                    "the file requires the unsupported extension {name}"
                ));
            }
        }
        if let Some(uri) = doc["buffers"][0]["uri"].as_str() {
            let shown: String = uri.chars().take(40).collect();
            return err(format!(
                "buffer 0 refers to external data ({shown}); only self-contained .glb is supported"
            ));
        }

        let mut out = Builder::default();
        let nodes = array(&doc["nodes"]);
        let roots = scene_roots(&doc, nodes.len());
        let mut visited = vec![false; nodes.len()];
        for root in roots {
            visit_node(&doc, bin, root, &IDENTITY4, 0, &mut visited, &mut out)?;
        }
        out.finish()
    }

    /// Centre of the bounding box, model space.
    #[must_use]
    pub fn center(&self) -> [f32; 3] {
        [0, 1, 2].map(|a| 0.5 * (self.bounds_min[a] + self.bounds_max[a]))
    }

    /// Radius of the sphere about [`MeshData::center`] that contains the
    /// bounding box.
    #[must_use]
    pub fn radius(&self) -> f32 {
        let half = [0, 1, 2].map(|a| 0.5 * (self.bounds_max[a] - self.bounds_min[a]));
        (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt()
    }

    /// Number of triangles.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Serialise as a minimal, valid `.glb`: one node, one mesh, one primitive
    /// per part. The inverse of [`MeshData::from_glb`] for anything it can
    /// load (used for tests and for writing sample assets).
    #[must_use]
    pub fn to_glb(&self) -> Vec<u8> {
        let vertex_bytes = self.vertices.len() * 24;
        let index_bytes = self.indices.len() * 4;
        let mut bin = Vec::with_capacity(vertex_bytes + index_bytes);
        for v in &self.vertices {
            for c in v.position.iter().chain(&v.normal) {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in &self.indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }

        let mut accessors = vec![
            serde_json::json!({
                "bufferView": 0, "byteOffset": 0, "componentType": COMPONENT_F32,
                "count": self.vertices.len(), "type": "VEC3",
                "min": self.bounds_min, "max": self.bounds_max,
            }),
            serde_json::json!({
                "bufferView": 0, "byteOffset": 12, "componentType": COMPONENT_F32,
                "count": self.vertices.len(), "type": "VEC3",
            }),
        ];
        let mut materials = Vec::new();
        let mut primitives = Vec::new();
        for part in &self.parts {
            let m = part.material;
            // glTF's emissiveFactor is capped at 1; anything brighter is
            // carried by the strength extension (factor * strength = radiance).
            let strength = m.emissive.iter().fold(1.0f32, |a, &c| a.max(c));
            materials.push(serde_json::json!({
                "pbrMetallicRoughness": {
                    "baseColorFactor": [m.base_color[0], m.base_color[1], m.base_color[2], 1.0],
                    "metallicFactor": m.metallic,
                    "roughnessFactor": m.roughness,
                },
                "emissiveFactor": m.emissive.map(|c| c / strength),
                "extensions": { "KHR_materials_emissive_strength": {
                    "emissiveStrength": strength,
                }},
            }));
            primitives.push(serde_json::json!({
                "attributes": { "POSITION": 0, "NORMAL": 1 },
                "indices": accessors.len(),
                "material": materials.len() - 1,
                "mode": MODE_TRIANGLES,
            }));
            accessors.push(serde_json::json!({
                "bufferView": 1, "byteOffset": u64::from(part.first_index) * 4,
                "componentType": COMPONENT_U32, "count": part.index_count, "type": "SCALAR",
            }));
        }
        let doc = serde_json::json!({
            "asset": { "version": "2.0", "generator": "space-render" },
            "extensionsUsed": ["KHR_materials_emissive_strength"],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": primitives }],
            "materials": materials,
            "accessors": accessors,
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": vertex_bytes,
                  "byteStride": 24, "target": 34962 },
                { "buffer": 0, "byteOffset": vertex_bytes, "byteLength": index_bytes,
                  "target": 34963 },
            ],
            "buffers": [{ "byteLength": bin.len() }],
        });
        let mut json = doc.to_string().into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }
}

// ---- container -----------------------------------------------------------------

fn u32_at(bytes: &[u8], offset: usize) -> Option<u32> {
    let b: [u8; 4] = bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?;
    Some(u32::from_le_bytes(b))
}

/// Split a GLB container into its JSON chunk and (possibly empty) BIN chunk.
fn split_chunks(bytes: &[u8]) -> Result<(&[u8], &[u8]), MeshError> {
    let truncated = || MeshError("the file is truncated".into());
    if u32_at(bytes, 0).ok_or_else(truncated)? != GLB_MAGIC {
        return err("not a GLB file (bad magic; a text .gltf is not supported)");
    }
    let version = u32_at(bytes, 4).ok_or_else(truncated)?;
    if version != 2 {
        return err(format!(
            "GLB container version {version} is not supported (need 2)"
        ));
    }
    let declared = u32_at(bytes, 8).ok_or_else(truncated)? as usize;
    if declared > bytes.len() {
        return err("the file is shorter than its header says");
    }
    let bytes = &bytes[..declared];
    let (mut json, mut bin): (Option<&[u8]>, &[u8]) = (None, &[]);
    let mut offset = 12;
    while offset < bytes.len() {
        let length = u32_at(bytes, offset).ok_or_else(truncated)? as usize;
        let kind = u32_at(bytes, offset + 4).ok_or_else(truncated)?;
        let start = offset + 8;
        let end = start.checked_add(length).ok_or_else(truncated)?;
        let data = bytes.get(start..end).ok_or_else(truncated)?;
        match kind {
            CHUNK_JSON if json.is_none() => json = Some(data),
            CHUNK_BIN if bin.is_empty() => bin = data,
            // Unknown chunk types are skipped, as the format requires.
            _ => {}
        }
        // Chunks are padded to 4 bytes.
        offset = end.checked_add(3).ok_or_else(truncated)? & !3;
    }
    Ok((
        json.ok_or_else(|| MeshError("the file has no JSON chunk".into()))?,
        bin,
    ))
}

// ---- JSON helpers --------------------------------------------------------------

fn array(value: &Value) -> &[Value] {
    value.as_array().map_or(&[], Vec::as_slice)
}

fn index(value: &Value) -> Option<usize> {
    value.as_u64().and_then(|v| usize::try_from(v).ok())
}

/// `N` finite numbers from a JSON array, or `None` if it is anything else.
fn floats<const N: usize>(value: &Value) -> Option<[f64; N]> {
    let items = value.as_array()?;
    if items.len() != N {
        return None;
    }
    let mut out = [0.0; N];
    for (slot, item) in out.iter_mut().zip(items) {
        *slot = item.as_f64().filter(|v| v.is_finite())?;
    }
    Some(out)
}

/// The root nodes to draw: the default scene's, else the first scene's, else
/// (a file with no scenes) every node that is nobody's child.
fn scene_roots(doc: &Value, node_count: usize) -> Vec<usize> {
    let scenes = array(&doc["scenes"]);
    let scene = index(&doc["scene"])
        .and_then(|i| scenes.get(i))
        .or(scenes.first());
    if let Some(scene) = scene {
        return array(&scene["nodes"]).iter().filter_map(index).collect();
    }
    let mut is_child = vec![false; node_count];
    for node in array(&doc["nodes"]) {
        for child in array(&node["children"]).iter().filter_map(index) {
            if let Some(flag) = is_child.get_mut(child) {
                *flag = true;
            }
        }
    }
    (0..node_count).filter(|&i| !is_child[i]).collect()
}

// ---- transforms ----------------------------------------------------------------

/// Column-major 4x4, `f64` so a deep hierarchy does not accumulate error.
type Mat4d = [[f64; 4]; 4];

const IDENTITY4: Mat4d = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn mul4(a: &Mat4d, b: &Mat4d) -> Mat4d {
    let mut out = [[0.0; 4]; 4];
    for (c, column) in out.iter_mut().enumerate() {
        for (r, value) in column.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][r] * b[c][k]).sum();
        }
    }
    out
}

/// A node's local transform: its `matrix`, or translation * rotation * scale.
fn local_transform(node: &Value) -> Mat4d {
    if let Some(m) = floats::<16>(&node["matrix"]) {
        let mut out = [[0.0; 4]; 4];
        for (i, value) in m.iter().enumerate() {
            out[i / 4][i % 4] = *value;
        }
        return out;
    }
    let t = floats::<3>(&node["translation"]).unwrap_or([0.0; 3]);
    let s = floats::<3>(&node["scale"]).unwrap_or([1.0; 3]);
    let [x, y, z, w] = floats::<4>(&node["rotation"]).unwrap_or([0.0, 0.0, 0.0, 1.0]);
    // Normalise: a slightly denormal quaternion would otherwise shear.
    let len = (x * x + y * y + z * z + w * w).sqrt();
    let (x, y, z, w) = if len > 1e-12 {
        (x / len, y / len, z / len, w / len)
    } else {
        (0.0, 0.0, 0.0, 1.0)
    };
    let rot = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut out = IDENTITY4;
    for c in 0..3 {
        for r in 0..3 {
            out[c][r] = rot[c][r] * s[c];
        }
    }
    out[3] = [t[0], t[1], t[2], 1.0];
    out
}

fn transform_point(m: &Mat4d, p: [f32; 3]) -> [f32; 3] {
    let p = p.map(f64::from);
    [0, 1, 2].map(|r| (m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r]) as f32)
}

/// Transform a normal by the inverse transpose of `m`'s upper 3x3 (its
/// cofactor matrix, up to a scale that normalising removes), so normals stay
/// perpendicular under non-uniform scale.
fn transform_normal(m: &Mat4d, n: [f32; 3]) -> [f32; 3] {
    let a = |c: usize, r: usize| m[c][r];
    let cof = [
        [
            a(1, 1) * a(2, 2) - a(2, 1) * a(1, 2),
            a(2, 0) * a(1, 2) - a(1, 0) * a(2, 2),
            a(1, 0) * a(2, 1) - a(2, 0) * a(1, 1),
        ],
        [
            a(2, 1) * a(0, 2) - a(0, 1) * a(2, 2),
            a(0, 0) * a(2, 2) - a(2, 0) * a(0, 2),
            a(2, 0) * a(0, 1) - a(0, 0) * a(2, 1),
        ],
        [
            a(0, 1) * a(1, 2) - a(1, 1) * a(0, 2),
            a(1, 0) * a(0, 2) - a(0, 0) * a(1, 2),
            a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1),
        ],
    ];
    let n = n.map(f64::from);
    // `cof[c][r]` is the cofactor of the element in row r, column c. The
    // cofactor matrix is det * inverse-transpose, so a mirroring transform
    // (negative determinant) would flip the normal inwards: undo that.
    let det = a(0, 0) * cof[0][0] + a(1, 0) * cof[1][0] + a(2, 0) * cof[2][0];
    let sign = if det < 0.0 { -1.0 } else { 1.0 };
    let out = [0, 1, 2].map(|r| sign * (cof[0][r] * n[0] + cof[1][r] * n[1] + cof[2][r] * n[2]));
    normalised(out.map(|v| v as f32))
}

fn normalised(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1e-20 && len.is_finite() {
        v.map(|c| c / len)
    } else {
        [0.0, 1.0, 0.0]
    }
}

// ---- accessors -----------------------------------------------------------------

/// A bounds-checked view of one accessor's elements inside the BIN chunk.
struct Accessor<'a> {
    data: &'a [u8],
    component: u64,
    count: usize,
    stride: usize,
    /// Bytes of one element.
    size: usize,
}

fn component_size(component: u64) -> Option<usize> {
    Some(match component {
        5120 | COMPONENT_U8 => 1,
        5122 | COMPONENT_U16 => 2,
        COMPONENT_U32 | COMPONENT_F32 => 4,
        _ => return None,
    })
}

fn accessor<'a>(
    doc: &Value,
    bin: &'a [u8],
    accessor_index: usize,
    kind: &str,
    components: usize,
) -> Result<Accessor<'a>, MeshError> {
    let bad = |what: &str| MeshError(format!("accessor {accessor_index}: {what}"));
    let acc = array(&doc["accessors"])
        .get(accessor_index)
        .ok_or_else(|| bad("does not exist"))?;
    if !acc["sparse"].is_null() {
        return Err(bad("sparse accessors are not supported"));
    }
    if acc["type"].as_str() != Some(kind) {
        return Err(bad(&format!("expected type {kind}")));
    }
    let component = acc["componentType"].as_u64().unwrap_or(0);
    let size = component_size(component).ok_or_else(|| bad("unknown component type"))? * components;
    let count = index(&acc["count"]).ok_or_else(|| bad("has no count"))?;
    if count > MAX_ELEMENTS {
        return Err(bad("has more elements than the loader accepts"));
    }
    let view_index = index(&acc["bufferView"])
        .ok_or_else(|| bad("has no bufferView (implicit-zero accessors are not supported)"))?;
    let view = array(&doc["bufferViews"])
        .get(view_index)
        .ok_or_else(|| bad("names a bufferView that does not exist"))?;
    if index(&view["buffer"]).unwrap_or(0) != 0 {
        return Err(bad("uses a buffer other than the file's BIN chunk"));
    }
    let view_offset = index(&view["byteOffset"]).unwrap_or(0);
    let view_length = index(&view["byteLength"]).ok_or_else(|| bad("bufferView has no length"))?;
    let stride = match index(&view["byteStride"]) {
        Some(stride) if stride >= size => stride,
        Some(_) => return Err(bad("bufferView stride is smaller than one element")),
        None => size,
    };
    let offset = index(&acc["byteOffset"]).unwrap_or(0);
    let view_bytes = view_offset
        .checked_add(view_length)
        .and_then(|end| bin.get(view_offset..end))
        .ok_or_else(|| bad("bufferView runs past the end of the BIN chunk"))?;
    // The last element starts at (count - 1) * stride and is `size` long.
    let needed = match count {
        0 => 0,
        _ => (count - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(size))
            .ok_or_else(|| bad("size overflows"))?,
    };
    let data = offset
        .checked_add(needed)
        .and_then(|end| view_bytes.get(offset..end))
        .ok_or_else(|| bad("runs past the end of its bufferView"))?;
    Ok(Accessor {
        data,
        component,
        count,
        stride,
        size,
    })
}

impl Accessor<'_> {
    /// The bytes of element `i` (`i < count`).
    fn element(&self, i: usize) -> &[u8] {
        &self.data[i * self.stride..i * self.stride + self.size]
    }

    fn vec3(&self, i: usize) -> [f32; 3] {
        let e = self.element(i);
        [0, 4, 8].map(|o| f32::from_le_bytes([e[o], e[o + 1], e[o + 2], e[o + 3]]))
    }

    fn index(&self, i: usize) -> u32 {
        let e = self.element(i);
        match self.component {
            COMPONENT_U8 => u32::from(e[0]),
            COMPONENT_U16 => u32::from(u16::from_le_bytes([e[0], e[1]])),
            _ => u32::from_le_bytes([e[0], e[1], e[2], e[3]]),
        }
    }
}

// ---- building the mesh ---------------------------------------------------------

#[derive(Default)]
struct Builder {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    parts: Vec<MeshPart>,
    /// Vertex ranges already loaded, by `(node, POSITION accessor, NORMAL
    /// accessor)`: primitives of one mesh often differ only in material and
    /// index range, and must not each get their own copy of the vertices.
    shared: std::collections::HashMap<(usize, usize, usize), (usize, usize)>,
}

impl Builder {
    fn finish(self) -> Result<MeshData, MeshError> {
        if self.indices.is_empty() {
            return err("the file contains no triangles");
        }
        let (mut min, mut max) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for v in &self.vertices {
            for a in 0..3 {
                min[a] = min[a].min(v.position[a]);
                max[a] = max[a].max(v.position[a]);
            }
        }
        Ok(MeshData {
            vertices: self.vertices,
            indices: self.indices,
            parts: self.parts,
            bounds_min: min,
            bounds_max: max,
        })
    }
}

fn visit_node(
    doc: &Value,
    bin: &[u8],
    node_index: usize,
    parent: &Mat4d,
    depth: usize,
    visited: &mut [bool],
    out: &mut Builder,
) -> Result<(), MeshError> {
    let Some(node) = array(&doc["nodes"]).get(node_index) else {
        return err(format!("node {node_index} does not exist"));
    };
    // A node may appear once in a hierarchy; a repeat means a cycle or a DAG.
    if depth > MAX_NODE_DEPTH || std::mem::replace(&mut visited[node_index], true) {
        return err(format!(
            "node {node_index} is reached twice or nested too deeply"
        ));
    }
    let world = mul4(parent, &local_transform(node));
    if let Some(mesh_index) = index(&node["mesh"]) {
        let mesh = array(&doc["meshes"])
            .get(mesh_index)
            .ok_or_else(|| MeshError(format!("mesh {mesh_index} does not exist")))?;
        for primitive in array(&mesh["primitives"]) {
            add_primitive(doc, bin, primitive, node_index, &world, out)?;
        }
    }
    for child in array(&node["children"]).iter().filter_map(index) {
        visit_node(doc, bin, child, &world, depth + 1, visited, out)?;
    }
    Ok(())
}

fn material_of(doc: &Value, primitive: &Value) -> MeshMaterial {
    let Some(material) =
        index(&primitive["material"]).and_then(|i| array(&doc["materials"]).get(i))
    else {
        return MeshMaterial::default();
    };
    let pbr = &material["pbrMetallicRoughness"];
    let unit = |value: &Value, default: f64| {
        value
            .as_f64()
            .filter(|v| v.is_finite())
            .unwrap_or(default)
            .clamp(0.0, 1.0) as f32
    };
    let base = floats::<4>(&pbr["baseColorFactor"]).unwrap_or([1.0; 4]);
    let strength = material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(1.0);
    let emissive = floats::<3>(&material["emissiveFactor"]).unwrap_or([0.0; 3]);
    MeshMaterial {
        base_color: [0, 1, 2].map(|i| base[i].clamp(0.0, 1.0) as f32),
        metallic: unit(&pbr["metallicFactor"], 1.0),
        roughness: unit(&pbr["roughnessFactor"], 1.0),
        emissive: emissive.map(|c| (c.clamp(0.0, 1.0) * strength).min(1.0e6) as f32),
    }
}

fn add_primitive(
    doc: &Value,
    bin: &[u8],
    primitive: &Value,
    node_index: usize,
    world: &Mat4d,
    out: &mut Builder,
) -> Result<(), MeshError> {
    let mode = primitive["mode"].as_u64().unwrap_or(MODE_TRIANGLES);
    if !matches!(
        mode,
        MODE_TRIANGLES | MODE_TRIANGLE_STRIP | MODE_TRIANGLE_FAN
    ) {
        // Points and lines: nothing to draw as a surface.
        return Ok(());
    }
    let Some(position_index) = index(&primitive["attributes"]["POSITION"]) else {
        return Ok(());
    };
    let positions = accessor(doc, bin, position_index, "VEC3", 3)?;
    if positions.component != COMPONENT_F32 {
        return err("POSITION is not 32-bit float (quantised geometry is not supported)");
    }
    let normal_index = index(&primitive["attributes"]["NORMAL"]);
    let normals = match normal_index {
        Some(i) => {
            let normals = accessor(doc, bin, i, "VEC3", 3)?;
            if normals.component != COMPONENT_F32 {
                return err("NORMAL is not 32-bit float (quantised geometry is not supported)");
            }
            if normals.count != positions.count {
                return err("NORMAL and POSITION have different counts");
            }
            Some(normals)
        }
        None => None,
    };

    // Vertices this node already loaded from the same accessors are reused.
    // (Only with normals from the file: generated normals depend on the
    // primitive's own triangles, so those vertices stay private to it.)
    let share_key = normal_index.map(|n| (node_index, position_index, n));
    let reused = share_key.and_then(|key| out.shared.get(&key).copied());
    let base = match reused {
        Some((base, count)) if count == positions.count => base,
        _ => {
            let base = out.vertices.len();
            if base + positions.count > MAX_ELEMENTS || base + positions.count > u32::MAX as usize {
                return err("the model has more vertices than the loader accepts");
            }
            for i in 0..positions.count {
                let p = positions.vec3(i);
                if !p.iter().all(|c| c.is_finite()) {
                    return err("a vertex position is not finite");
                }
                out.vertices.push(MeshVertex {
                    position: transform_point(world, p),
                    normal: normals
                        .as_ref()
                        .map_or([0.0; 3], |n| transform_normal(world, n.vec3(i))),
                });
            }
            if let Some(key) = share_key {
                out.shared.insert(key, (base, positions.count));
            }
            base
        }
    };

    // The primitive's own index sequence (or 0..count), then triangulated.
    let sequence: Vec<u32> = match index(&primitive["indices"]) {
        Some(i) => {
            let acc = accessor(doc, bin, i, "SCALAR", 1)?;
            if !matches!(acc.component, COMPONENT_U8 | COMPONENT_U16 | COMPONENT_U32) {
                return err("indices are not an unsigned integer type");
            }
            (0..acc.count).map(|k| acc.index(k)).collect()
        }
        None => (0..positions.count as u32).collect(),
    };
    if sequence.iter().any(|&i| i as usize >= positions.count) {
        return err("an index points past the end of the vertices");
    }
    let first_index = out.indices.len();
    let base = base as u32;
    match mode {
        MODE_TRIANGLE_STRIP => {
            for (k, w) in sequence.windows(3).enumerate() {
                // Every other triangle of a strip has reversed winding.
                let tri = if k % 2 == 0 {
                    [w[0], w[1], w[2]]
                } else {
                    [w[1], w[0], w[2]]
                };
                out.indices.extend(tri.map(|i| base + i));
            }
        }
        MODE_TRIANGLE_FAN => {
            for w in sequence.get(1..).unwrap_or(&[]).windows(2) {
                out.indices
                    .extend([sequence[0], w[0], w[1]].map(|i| base + i));
            }
        }
        _ => {
            let whole = sequence.len() - sequence.len() % 3;
            out.indices
                .extend(sequence[..whole].iter().map(|i| base + i));
        }
    }
    if out.indices.len() > MAX_ELEMENTS {
        return err("the model has more indices than the loader accepts");
    }
    let added = out.indices.len() - first_index;
    if added == 0 {
        return Ok(());
    }
    if normals.is_none() {
        // Without NORMAL the vertices are this primitive's own, at the end.
        generate_normals(
            &mut out.vertices[base as usize..],
            &out.indices[first_index..],
            base,
        );
    }
    out.parts.push(MeshPart {
        first_index: first_index as u32,
        index_count: added as u32,
        material: material_of(doc, primitive),
    });
    Ok(())
}

/// Area-weighted vertex normals for a primitive that came without any.
/// `vertices` is the primitive's own vertex range; `indices` index the whole
/// mesh, so `base` is subtracted.
fn generate_normals(vertices: &mut [MeshVertex], indices: &[u32], base: u32) {
    for tri in indices.as_chunks::<3>().0 {
        let [a, b, c] = tri.map(|i| (i - base) as usize);
        let (pa, pb, pc) = (
            vertices[a].position,
            vertices[b].position,
            vertices[c].position,
        );
        let e1 = [0, 1, 2].map(|k| pb[k] - pa[k]);
        let e2 = [0, 1, 2].map(|k| pc[k] - pa[k]);
        // The cross product's length is twice the triangle's area.
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for v in [a, b, c] {
            for k in 0..3 {
                vertices[v].normal[k] += n[k];
            }
        }
    }
    for v in vertices {
        v.normal = normalised(v.normal);
    }
}

/// A small multi-part sample model — a stylised ship: a painted hull, a
/// polished steel nose, gold wings, a red fin, a glossy cockpit and a glowing
/// engine, about 4 units long with its
/// nose along -Z. Deterministic; used by the tests and by the
/// `make_test_glb` example to produce a `.glb` without any external asset.
#[must_use]
pub fn sample_ship() -> MeshData {
    let mut out = Builder::default();
    let mut add_box = |centre: [f32; 3], half: [f32; 3], material: MeshMaterial| {
        let first_index = out.indices.len() as u32;
        // Six faces, each its own four vertices so the normals are flat.
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let base = out.vertices.len() as u32;
                let mut normal = [0.0; 3];
                normal[axis] = sign;
                for (du, dv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let mut p = centre;
                    p[axis] += sign * half[axis];
                    p[u] += du * half[u];
                    p[v] += dv * half[v];
                    out.vertices.push(MeshVertex {
                        position: p,
                        normal,
                    });
                }
                // Counter-clockwise seen from outside.
                let quad = if sign > 0.0 {
                    [0, 1, 2, 0, 2, 3]
                } else {
                    [0, 2, 1, 0, 3, 2]
                };
                out.indices.extend(quad.map(|i| base + i));
            }
        }
        out.parts.push(MeshPart {
            first_index,
            index_count: out.indices.len() as u32 - first_index,
            material,
        });
    };
    let paint = |base_color: [f32; 3], metallic: f32, roughness: f32| MeshMaterial {
        base_color,
        metallic,
        roughness,
        emissive: [0.0; 3],
    };
    // Hull, nose, wings, tail fin, cockpit, engine.
    // (Mostly dielectric paint: with point lights only and no environment to
    // reflect, a fully metallic hull would be black away from its highlight.)
    add_box(
        [0.0, 0.0, 0.0],
        [0.5, 0.35, 1.6],
        paint([0.78, 0.79, 0.82], 0.1, 0.5),
    );
    add_box(
        [0.0, -0.05, -1.9],
        [0.3, 0.2, 0.35],
        paint([0.56, 0.57, 0.58], 1.0, 0.2),
    );
    add_box(
        [0.0, -0.1, 0.5],
        [1.9, 0.06, 0.6],
        paint([1.0, 0.72, 0.25], 0.4, 0.4),
    );
    add_box(
        [0.0, 0.6, 1.2],
        [0.05, 0.4, 0.4],
        paint([0.8, 0.08, 0.08], 0.0, 0.5),
    );
    add_box(
        [0.0, 0.4, -0.7],
        [0.3, 0.12, 0.45],
        paint([0.05, 0.2, 0.6], 0.0, 0.1),
    );
    add_box(
        [0.0, 0.0, 1.7],
        [0.3, 0.2, 0.12],
        MeshMaterial {
            base_color: [0.1, 0.1, 0.1],
            metallic: 0.0,
            roughness: 0.8,
            emissive: [0.6, 2.2, 6.0],
        },
    );
    // Infallible: the builder above always produces triangles.
    out.finish().unwrap_or_else(|_| MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
        parts: Vec::new(),
        bounds_min: [0.0; 3],
        bounds_max: [0.0; 3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Wrap a JSON document and a BIN payload into a GLB container.
    fn glb(doc: &Value, bin: &[u8]) -> Vec<u8> {
        let mut json = doc.to_string().into_bytes();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bin = bin.to_vec();
        while !bin.len().is_multiple_of(4) {
            bin.push(0);
        }
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::new();
        out.extend_from_slice(&GLB_MAGIC.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_JSON.to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&CHUNK_BIN.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    /// One triangle in the XY plane, positions only, u16 indices, under a
    /// parent node that scales by 2 and a child that moves +10 in X.
    fn triangle_file() -> (Value, Vec<u8>) {
        let mut bin = Vec::new();
        for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for c in p {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let doc = serde_json::json!({
            "asset": { "version": "2.0" },
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [
                { "scale": [2.0, 2.0, 2.0], "children": [1] },
                { "translation": [10.0, 0.0, 0.0], "mesh": 0 },
            ],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0 }, "indices": 1, "material": 0,
            }]}],
            "materials": [{
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.2, 0.4, 0.6, 1.0],
                    "metallicFactor": 0.25, "roughnessFactor": 0.75,
                },
                "emissiveFactor": [1.0, 0.5, 0.0],
                "extensions": { "KHR_materials_emissive_strength": { "emissiveStrength": 4.0 } },
            }],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR" },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
                { "buffer": 0, "byteOffset": 36, "byteLength": 6 },
            ],
            "buffers": [{ "byteLength": 42 }],
        });
        (doc, bin)
    }

    #[test]
    fn a_triangle_loads_with_its_hierarchy_material_and_generated_normal() {
        let (doc, bin) = triangle_file();
        let mesh = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(mesh.indices, [0, 1, 2]);
        // Parent scale 2 applied after the child's +10 translation.
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        assert_eq!(
            positions,
            [[20.0, 0.0, 0.0], [22.0, 0.0, 0.0], [20.0, 2.0, 0.0]]
        );
        // No NORMAL in the file: generated, facing +Z for a CCW triangle in XY.
        for v in &mesh.vertices {
            assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        }
        assert_eq!(mesh.parts.len(), 1);
        let m = mesh.parts[0].material;
        assert_eq!(m.base_color, [0.2, 0.4, 0.6]);
        assert_eq!((m.metallic, m.roughness), (0.25, 0.75));
        assert_eq!(m.emissive, [4.0, 2.0, 0.0]);
        assert_eq!(
            (mesh.bounds_min, mesh.bounds_max),
            ([20.0, 0.0, 0.0], [22.0, 2.0, 0.0])
        );
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn the_sample_ship_survives_a_round_trip_through_glb() {
        let ship = sample_ship();
        assert_eq!(ship.parts.len(), 6);
        assert_eq!(ship.triangle_count(), 6 * 12);
        let loaded = MeshData::from_glb(&ship.to_glb()).expect("our own writer's output loads");
        assert_eq!(loaded.vertices.len(), ship.vertices.len());
        for (i, (a, b)) in loaded.vertices.iter().zip(&ship.vertices).enumerate() {
            assert_eq!(a, b, "vertex {i}");
        }
        assert_eq!(loaded.indices, ship.indices);
        assert_eq!(loaded.parts.len(), ship.parts.len());
        for (a, b) in loaded.parts.iter().zip(&ship.parts) {
            assert_eq!(
                (a.first_index, a.index_count),
                (b.first_index, b.index_count)
            );
            assert_eq!(a.material.base_color, b.material.base_color);
            assert_eq!(
                (a.material.metallic, a.material.roughness),
                (b.material.metallic, b.material.roughness)
            );
            for k in 0..3 {
                assert!((a.material.emissive[k] - b.material.emissive[k]).abs() < 1e-4);
            }
        }
        // Normals are unit length and the nose points down -Z.
        assert!(ship.vertices.iter().all(|v| {
            let n = v.normal;
            ((n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt() - 1.0).abs() < 1e-6
        }));
        assert!(ship.bounds_min[2] < -2.0 && ship.radius() > 2.0);
    }

    #[test]
    fn strips_fans_and_unindexed_primitives_are_triangulated() {
        let (mut doc, mut bin) = triangle_file();
        // A fourth vertex, so a strip/fan of 4 makes two triangles.
        bin.truncate(36);
        for c in [1.0f32, 1.0, 0.0] {
            bin.extend_from_slice(&c.to_le_bytes());
        }
        doc["accessors"][0]["count"] = 4.into();
        doc["bufferViews"][0]["byteLength"] = 48.into();
        doc["meshes"][0]["primitives"][0]
            .as_object_mut()
            .unwrap()
            .remove("indices");
        for (mode, expected) in [
            (MODE_TRIANGLE_STRIP, vec![0, 1, 2, 2, 1, 3]),
            (MODE_TRIANGLE_FAN, vec![0, 1, 2, 0, 2, 3]),
            (MODE_TRIANGLES, vec![0, 1, 2]),
        ] {
            doc["meshes"][0]["primitives"][0]["mode"] = mode.into();
            let mesh = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
            assert_eq!(mesh.indices, expected, "mode {mode}");
        }
    }

    /// A malformed file is an error, never a panic or a wild read.
    #[test]
    fn malformed_files_are_rejected() {
        let (doc, bin) = triangle_file();
        let good = glb(&doc, &bin);
        assert!(MeshData::from_glb(&good).is_ok());

        let rejects = |bytes: &[u8], why: &str| {
            assert!(MeshData::from_glb(bytes).is_err(), "{why} must be rejected");
        };
        rejects(&[], "an empty file");
        rejects(b"glTF", "a bare magic");
        rejects(&good[..good.len() - 9], "a truncated file");
        let mut bad_magic = good.clone();
        bad_magic[0] = b'x';
        rejects(&bad_magic, "a wrong magic");
        let mut bad_version = good.clone();
        bad_version[4] = 1;
        rejects(&bad_version, "container version 1");
        // Every prefix of a valid file: must return, not panic.
        for cut in 0..good.len() {
            let _ = MeshData::from_glb(&good[..cut]);
        }

        let with = |edit: &dyn Fn(&mut Value)| {
            let mut doc = doc.clone();
            edit(&mut doc);
            glb(&doc, &bin)
        };
        rejects(
            &with(&|d| d["accessors"][0]["count"] = 4.into()),
            "a count past the view",
        );
        rejects(
            &with(&|d| d["accessors"][1]["count"] = 1000.into()),
            "indices past the view",
        );
        rejects(
            &with(&|d| d["bufferViews"][0]["byteLength"] = 99_999.into()),
            "a view past BIN",
        );
        rejects(
            &with(&|d| d["accessors"][0]["byteOffset"] = 30.into()),
            "an offset past the view",
        );
        rejects(
            &with(&|d| d["accessors"][0]["componentType"] = 5123.into()),
            "quantised positions",
        );
        rejects(
            &with(&|d| d["accessors"][0]["sparse"] = serde_json::json!({})),
            "sparse",
        );
        rejects(
            &with(&|d| d["meshes"][0]["primitives"][0]["indices"] = 7.into()),
            "a bad accessor",
        );
        rejects(
            &with(&|d| d["nodes"][1]["mesh"] = 3.into()),
            "a missing mesh",
        );
        rejects(
            &with(&|d| d["nodes"][1]["children"] = serde_json::json!([0])),
            "a node cycle",
        );
        rejects(
            &with(&|d| d["extensionsRequired"] = serde_json::json!(["KHR_draco_mesh_compression"])),
            "a required unsupported extension",
        );
        rejects(
            &with(&|d| d["buffers"][0]["uri"] = "data.bin".into()),
            "an external buffer",
        );
        rejects(
            &with(&|d| d["scenes"][0]["nodes"] = serde_json::json!([])),
            "an empty scene",
        );

        // An index past the vertices.
        let mut bad_index = bin.clone();
        bad_index[36] = 9;
        rejects(&glb(&doc, &bad_index), "an out-of-range index");
        // A non-finite position.
        let mut nan = bin.clone();
        nan[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        rejects(&glb(&doc, &nan), "a NaN position");
    }

    #[test]
    fn rotation_helpers_are_orthonormal() {
        let r = rotation_y(0.7);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f32 = (0..3).map(|k| r[i][k] * r[j][k]).sum();
                assert!((dot - f32::from(u8::from(i == j))).abs() < 1e-6);
            }
        }
        assert_eq!(rotation_y(0.0), IDENTITY_ROTATION);
    }
}
