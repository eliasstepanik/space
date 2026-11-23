//! Triangle meshes and the GLB (binary glTF 2.0) loader.
//!
//! [`MeshData`] is a model ready to draw: one vertex buffer (position, normal,
//! texture coordinate), one index buffer, a list of [`MeshPart`]s — index
//! ranges that each carry a metallic-roughness material — and the base-colour
//! [`Texture`]s those materials refer to by index. A [`ModelInstance`] places
//! a shared `MeshData` in a [`Scene`](crate::Scene).
//!
//! # Conventions
//!
//! Triangles are counter-clockwise seen from outside, +Y is up, and the built-in
//! models ([`sample_ship`], [`sample_character`]) face **-Z** (the default
//! camera's view direction, so a model in front of the camera is seen from
//! behind). Texture coordinate (0, 0) is the top-left corner of the image, as
//! in glTF.
//!
//! # What the GLB loader supports
//!
//! [`MeshData::from_glb`] reads a self-contained `.glb` file: the default
//! scene's node hierarchy (matrix or translation/rotation/scale, baked into the
//! vertices), every triangle primitive (lists, strips and fans, indexed or
//! not), `POSITION`, `NORMAL` (normals are generated when a primitive has
//! none) and `TEXCOORD_0` (float or normalised integer), and each material's
//! `baseColorFactor`, `metallicFactor`, `roughnessFactor`, `emissiveFactor`
//! (with `KHR_materials_emissive_strength`) and `baseColorTexture` — a PNG or
//! JPEG image embedded in the file (see [`crate::texture`]).
//!
//! **Vertex colours** (`COLOR_0`: RGB or RGBA, float or normalised integer;
//! alpha is not used) are multiplied into the base colour *per triangle*. The
//! renderer's vertices carry no colour, so a primitive is split into one part
//! per face colour instead: exact for a model that uses vertex colour as paint
//! per face — what merging many materials into one leaves behind, as in an
//! exported low-detail proxy — and the triangle's average, with a warning, for
//! a model that shades smoothly with it.
//!
//! **Parts are merged by material**: every triangle of the file that is drawn
//! with the same material (after the vertex colour) ends up in one
//! [`MeshPart`], in the order the materials first appear, whatever the number
//! of nodes and primitives it came from. A part is a draw call and a geometry
//! of the ray-traced scene, so a model exported as a thousand objects sharing
//! a dozen materials costs a dozen of each, not a thousand.
//!
//! Features the renderer does not have are **ignored with a logged warning**
//! and the model still loads: the metallic-roughness, normal and occlusion
//! textures (the material's factors are used alone), the emissive texture
//! (the material glows evenly, in the texture's average colour — the factor
//! alone would make a mostly dark screen as bright as its brightest pixel),
//! `KHR_texture_transform`, a second UV set, wrap modes other than repeat,
//! images outside the file or in another format, images that fail to decode
//! or exceed the size caps (such a material is drawn with its factors only),
//! and the alpha modes: base-colour alpha is not used, so a `BLEND` material
//! is drawn opaque and a `MASK` material without its cut-outs.
//!
//! Not supported, by design for now: skins, animations, morph targets, cameras
//! and lights (ignored); sparse accessors, quantised or compressed geometry
//! (`KHR_mesh_quantization`, Draco, meshopt) and buffers outside the file
//! (rejected with an error rather than drawn wrongly).
//!
//! The loader treats the file as untrusted input: every offset, count and
//! index is bounds-checked, sizes are capped (decoded images at
//! [`MAX_TEXTURE_DIMENSION`](crate::texture::MAX_TEXTURE_DIMENSION) per side
//! and [`MAX_GLB_TEXTURE_BYTES`] per file), and a malformed file is a
//! [`MeshError`], never a panic.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use crate::texture::{decode_image, encode_png, Texture};

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

/// One mesh vertex: 32 bytes — position, normal, texture coordinate (the mesh
/// pipeline's vertex layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Texture coordinate; (0, 0) is the top-left corner of the image and
    /// values outside 0..1 repeat it.
    pub uv: [f32; 2],
}

/// A metallic-roughness material: factors plus an optional base-colour texture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshMaterial {
    /// Base colour, linear rgb. Multiplied with the texture, if there is one.
    pub base_color: [f32; 3],
    pub metallic: f32,
    pub roughness: f32,
    /// Emitted radiance, linear rgb.
    pub emissive: [f32; 3],
    /// The base-colour texture: an index into [`MeshData::textures`]. `None`
    /// (or an index past the end of that list) is an untextured material.
    pub base_color_texture: Option<u32>,
}

impl Default for MeshMaterial {
    /// glTF's default material: white, fully metallic, fully rough.
    fn default() -> Self {
        Self {
            base_color: [1.0; 3],
            metallic: 1.0,
            roughness: 1.0,
            emissive: [0.0; 3],
            base_color_texture: None,
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
    /// The base-colour textures the parts' materials name by index
    /// ([`MeshMaterial::base_color_texture`]). Uploaded with the mesh.
    pub textures: Vec<Texture>,
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
/// The most memory the decoded (RGBA8) textures of one file may take. An image
/// that would go past it is skipped with a warning.
pub const MAX_GLB_TEXTURE_BYTES: usize = 1 << 29;
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

/// Extensions a file may require that are not implemented but safe to leave
/// out: the model loads, with a warning, drawn without the feature.
const IGNORED_REQUIRED_EXTENSIONS: &[&str] = &["KHR_texture_transform"];

const WRAP_REPEAT: u64 = 10497;

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

    /// Parse the bytes of a `.glb` file. Anything the loader leaves out on
    /// purpose (see the module docs) is logged as a warning.
    pub fn from_glb(bytes: &[u8]) -> Result<Self, MeshError> {
        let (mesh, warnings) = Self::from_glb_with_warnings(bytes)?;
        for warning in &warnings {
            log::warn!("glb: {warning}");
        }
        Ok(mesh)
    }

    /// [`MeshData::from_glb`], returning the warnings instead of logging them
    /// (each distinct warning once).
    pub fn from_glb_with_warnings(bytes: &[u8]) -> Result<(Self, Vec<String>), MeshError> {
        if bytes.len() > MAX_GLB_BYTES {
            return err("file is larger than the size limit");
        }
        let (json, bin) = split_chunks(bytes)?;
        let doc: Value = serde_json::from_slice(json)
            .map_err(|e| MeshError(format!("the JSON chunk is not valid JSON: {e}")))?;
        let mut out = Builder::default();
        for required in array(&doc["extensionsRequired"]) {
            let name = required.as_str().unwrap_or("?");
            if IGNORED_REQUIRED_EXTENSIONS.contains(&name) {
                out.warn(format!("the required extension {name} is ignored"));
            } else if !SUPPORTED_REQUIRED_EXTENSIONS.contains(&name) {
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

        let nodes = array(&doc["nodes"]);
        let roots = scene_roots(&doc, nodes.len());
        let mut visited = vec![false; nodes.len()];
        for root in roots {
            visit_node(&doc, bin, root, &IDENTITY4, 0, &mut visited, &mut out)?;
        }
        let warnings = std::mem::take(&mut out.warnings);
        Ok((out.finish()?, warnings))
    }

    /// Add `texture` to the mesh and return its index, for
    /// [`MeshMaterial::base_color_texture`].
    pub fn add_texture(&mut self, texture: Texture) -> u32 {
        self.textures.push(texture);
        (self.textures.len() - 1) as u32
    }

    /// The texture a material samples, if it names one that exists.
    #[must_use]
    pub fn texture_of(&self, material: &MeshMaterial) -> Option<&Texture> {
        self.textures.get(material.base_color_texture? as usize)
    }

    /// Check what the renderer relies on before it hands the mesh to the GPU:
    /// every index names a vertex, every part is a whole number of triangles
    /// inside the index buffer, and every position is finite. (The loader and
    /// the built-in models always produce valid meshes; this is for meshes
    /// assembled by hand.)
    pub fn validate(&self) -> Result<(), MeshError> {
        if self.vertices.len() > u32::MAX as usize || self.indices.len() > u32::MAX as usize {
            return err("the mesh has more than u32::MAX vertices or indices");
        }
        let count = self.vertices.len() as u32;
        if self.indices.iter().any(|&i| i >= count) {
            return err("an index points past the end of the vertices");
        }
        for (i, part) in self.parts.iter().enumerate() {
            let end = u64::from(part.first_index) + u64::from(part.index_count);
            if end > self.indices.len() as u64 || !part.index_count.is_multiple_of(3) {
                return err(format!(
                    "part {i} is not a whole number of triangles inside the index buffer"
                ));
            }
        }
        if !self
            .vertices
            .iter()
            .all(|v| v.position.iter().all(|c| c.is_finite()))
        {
            return err("a vertex position is not finite");
        }
        Ok(())
    }

    /// True when the triangles form closed, consistently wound surfaces: after
    /// welding vertices that share a position, every edge is used by exactly
    /// two triangles, in opposite directions. Such a mesh has an inside and an
    /// outside, which lets ray-traced shadows ignore the faces a ray leaves
    /// through (see `passes.rs`). Degenerate triangles are ignored.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        // -0.0 and 0.0 are the same place.
        let place = |i: u32| {
            self.vertices
                .get(i as usize)
                .map(|v| v.position.map(|c| (c + 0.0).to_bits()))
        };
        let mut edges = std::collections::HashSet::new();
        for tri in self.indices.as_chunks::<3>().0 {
            let (Some(a), Some(b), Some(c)) = (place(tri[0]), place(tri[1]), place(tri[2])) else {
                return false;
            };
            if a == b || b == c || c == a {
                continue;
            }
            for edge in [(a, b), (b, c), (c, a)] {
                // The same directed edge twice: a fold, or flipped winding.
                if !edges.insert(edge) {
                    return false;
                }
            }
        }
        !edges.is_empty() && edges.iter().all(|&(a, b)| edges.contains(&(b, a)))
    }

    /// The volume the triangles enclose (model units cubed), by the divergence
    /// theorem: positive for a closed mesh wound counter-clockwise seen from
    /// outside, negative for one that is inside out. Only meaningful when
    /// [`MeshData::is_closed`].
    #[must_use]
    pub fn signed_volume(&self) -> f64 {
        let position = |i: u32| {
            self.vertices
                .get(i as usize)
                .map_or([0.0; 3], |v| v.position.map(f64::from))
        };
        let mut six_volumes = 0.0;
        for tri in self.indices.as_chunks::<3>().0 {
            let [a, b, c] = tri.map(position);
            six_volumes += a[0] * (b[1] * c[2] - b[2] * c[1])
                + a[1] * (b[2] * c[0] - b[0] * c[2])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        six_volumes / 6.0
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
    /// per part, and every texture as an embedded PNG. The inverse of
    /// [`MeshData::from_glb`] for anything it can load (used for tests and for
    /// writing sample assets).
    #[must_use]
    pub fn to_glb(&self) -> Vec<u8> {
        const VERTEX_BYTES: usize = std::mem::size_of::<MeshVertex>();
        let vertex_bytes = self.vertices.len() * VERTEX_BYTES;
        let index_bytes = self.indices.len() * 4;
        let mut bin = Vec::with_capacity(vertex_bytes + index_bytes);
        for v in &self.vertices {
            for c in v.position.iter().chain(&v.normal).chain(&v.uv) {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        for i in &self.indices {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        let mut views = vec![
            serde_json::json!({
                "buffer": 0, "byteOffset": 0, "byteLength": vertex_bytes,
                "byteStride": VERTEX_BYTES, "target": 34962,
            }),
            serde_json::json!({
                "buffer": 0, "byteOffset": vertex_bytes, "byteLength": index_bytes,
                "target": 34963,
            }),
        ];
        // Each texture is a PNG in its own (4-byte aligned) buffer view.
        let (mut images, mut textures) = (Vec::new(), Vec::new());
        for texture in &self.textures {
            while !bin.len().is_multiple_of(4) {
                bin.push(0);
            }
            let png = encode_png(texture);
            images.push(serde_json::json!({ "bufferView": views.len(), "mimeType": "image/png" }));
            textures.push(serde_json::json!({ "source": textures.len(), "sampler": 0 }));
            views.push(serde_json::json!({
                "buffer": 0, "byteOffset": bin.len(), "byteLength": png.len(),
            }));
            bin.extend_from_slice(&png);
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
            serde_json::json!({
                "bufferView": 0, "byteOffset": 24, "componentType": COMPONENT_F32,
                "count": self.vertices.len(), "type": "VEC2",
            }),
        ];
        let mut materials = Vec::new();
        let mut primitives = Vec::new();
        for part in &self.parts {
            let m = part.material;
            // glTF's emissiveFactor is capped at 1; anything brighter is
            // carried by the strength extension (factor * strength = radiance).
            let strength = m.emissive.iter().fold(1.0f32, |a, &c| a.max(c));
            let mut material = serde_json::json!({
                "pbrMetallicRoughness": {
                    "baseColorFactor": [m.base_color[0], m.base_color[1], m.base_color[2], 1.0],
                    "metallicFactor": m.metallic,
                    "roughnessFactor": m.roughness,
                },
                "emissiveFactor": m.emissive.map(|c| c / strength),
                "extensions": { "KHR_materials_emissive_strength": {
                    "emissiveStrength": strength,
                }},
            });
            if let Some(texture) = m
                .base_color_texture
                .filter(|&t| (t as usize) < self.textures.len())
            {
                material["pbrMetallicRoughness"]["baseColorTexture"] =
                    serde_json::json!({ "index": texture });
            }
            materials.push(material);
            primitives.push(serde_json::json!({
                "attributes": { "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 },
                "indices": accessors.len(),
                "material": materials.len() - 1,
                "mode": MODE_TRIANGLES,
            }));
            accessors.push(serde_json::json!({
                "bufferView": 1, "byteOffset": u64::from(part.first_index) * 4,
                "componentType": COMPONENT_U32, "count": part.index_count, "type": "SCALAR",
            }));
        }
        let mut doc = serde_json::json!({
            "asset": { "version": "2.0", "generator": "space-render" },
            "extensionsUsed": ["KHR_materials_emissive_strength"],
            "scene": 0,
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": primitives }],
            "materials": materials,
            "accessors": accessors,
            "bufferViews": views,
            "buffers": [{ "byteLength": bin.len() }],
        });
        if !images.is_empty() {
            // (glTF does not allow empty arrays, hence only when there are any.)
            doc["images"] = images.into();
            doc["textures"] = textures.into();
            // Trilinear filtering, repeating in both directions.
            doc["samplers"] = serde_json::json!([{
                "magFilter": 9729, "minFilter": 9987,
                "wrapS": WRAP_REPEAT, "wrapT": WRAP_REPEAT,
            }]);
        }
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

/// Determinant of `m`'s upper 3x3: negative for a transform that mirrors.
fn determinant3(m: &Mat4d) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
        - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
        + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])
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

    /// A texture coordinate: float, or an unsigned integer normalised to
    /// 0..=1. A non-finite coordinate becomes 0.
    fn uv(&self, i: usize) -> [f32; 2] {
        let e = self.element(i);
        let uv = match self.component {
            COMPONENT_U8 => [f32::from(e[0]) / 255.0, f32::from(e[1]) / 255.0],
            COMPONENT_U16 => {
                [0, 2].map(|o| f32::from(u16::from_le_bytes([e[o], e[o + 1]])) / 65535.0)
            }
            _ => [0, 4].map(|o| f32::from_le_bytes([e[o], e[o + 1], e[o + 2], e[o + 3]])),
        };
        uv.map(|c| if c.is_finite() { c } else { 0.0 })
    }

    /// The rgb of a vertex colour (`COLOR_0`, linear): float, or an unsigned
    /// integer normalised to 0..=1; a fourth component (alpha) is not read.
    /// Clamped to 0..=1, and white where it is not a number.
    fn color(&self, i: usize) -> [f32; 3] {
        let e = self.element(i);
        let rgb = match self.component {
            COMPONENT_U8 => [0, 1, 2].map(|o| f32::from(e[o]) / 255.0),
            COMPONENT_U16 => {
                [0, 2, 4].map(|o| f32::from(u16::from_le_bytes([e[o], e[o + 1]])) / 65535.0)
            }
            _ => [0, 4, 8].map(|o| f32::from_le_bytes([e[o], e[o + 1], e[o + 2], e[o + 3]])),
        };
        rgb.map(|c| {
            if c.is_finite() {
                c.clamp(0.0, 1.0)
            } else {
                1.0
            }
        })
    }
}

// ---- building the mesh ---------------------------------------------------------

#[derive(Default)]
struct Builder {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
    parts: Vec<MeshPart>,
    textures: Vec<Texture>,
    /// The texture slot of every glTF image looked at so far (`None`: it could
    /// not be used), so an image shared by several materials is decoded once.
    images: std::collections::HashMap<usize, Option<u32>>,
    /// Bytes of decoded textures so far, against [`MAX_GLB_TEXTURE_BYTES`].
    texture_bytes: usize,
    /// The average colour (linear rgb) of every glTF image asked for it so
    /// far (`None`: it could not be read).
    averages: std::collections::HashMap<usize, Option<[f32; 3]>>,
    /// What was left out on purpose; each distinct message once.
    warnings: Vec<String>,
    /// Vertex ranges (first vertex, count) already loaded, by where they came
    /// from: primitives of one mesh often differ only in material and index
    /// range, and must not each get their own copy of the vertices.
    shared: std::collections::HashMap<VertexSource, (usize, usize)>,
}

/// Where a primitive's vertices come from: its node and its `POSITION`,
/// `NORMAL` and (if any) `TEXCOORD_0` accessors.
type VertexSource = (usize, usize, usize, Option<usize>);

impl Builder {
    fn warn(&mut self, message: String) {
        if !self.warnings.contains(&message) {
            self.warnings.push(message);
        }
    }

    /// The texture slot for glTF image `image_index`, decoding it on first use.
    /// An image that cannot be used is a warning and `None`.
    fn image(&mut self, doc: &Value, bin: &[u8], image_index: usize) -> Option<u32> {
        if let Some(slot) = self.images.get(&image_index) {
            return *slot;
        }
        let budget = MAX_GLB_TEXTURE_BYTES.saturating_sub(self.texture_bytes);
        let slot = match load_image(doc, bin, image_index, budget) {
            Ok(texture) => {
                self.texture_bytes += texture.pixels().len();
                self.textures.push(texture);
                Some((self.textures.len() - 1) as u32)
            }
            Err(e) => {
                self.warn(format!(
                    "image {image_index}: {e}; its material is drawn untextured"
                ));
                None
            }
        };
        self.images.insert(image_index, slot);
        slot
    }

    /// The average colour of glTF image `image_index`, in linear light,
    /// decoding it on first use (the pixels are not kept).
    fn image_average(&mut self, doc: &Value, bin: &[u8], image_index: usize) -> Option<[f32; 3]> {
        if let Some(average) = self.averages.get(&image_index) {
            return *average;
        }
        let budget = MAX_GLB_TEXTURE_BYTES.saturating_sub(self.texture_bytes);
        let average = load_image(doc, bin, image_index, budget).ok().map(|image| {
            // sRGB byte -> linear light.
            let linear: Vec<f64> = (0..256)
                .map(|byte| {
                    let c = f64::from(byte) / 255.0;
                    if c <= 0.04045 {
                        c / 12.92
                    } else {
                        ((c + 0.055) / 1.055).powf(2.4)
                    }
                })
                .collect();
            let mut sum = [0.0f64; 3];
            let texels = image.pixels().as_chunks::<4>().0;
            for texel in texels {
                for k in 0..3 {
                    sum[k] += linear[usize::from(texel[k])];
                }
            }
            sum.map(|s| (s / texels.len().max(1) as f64) as f32)
        });
        self.averages.insert(image_index, average);
        average
    }

    fn finish(mut self) -> Result<MeshData, MeshError> {
        if self.indices.is_empty() {
            return err("the file contains no triangles");
        }
        merge_parts_by_material(&mut self.indices, &mut self.parts);
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
            textures: self.textures,
            bounds_min: min,
            bounds_max: max,
        })
    }
}

/// A material as a map key: its numbers bit for bit, and its texture.
fn material_key(material: &MeshMaterial) -> ([u32; 8], Option<u32>) {
    let [r, g, b] = material.base_color;
    let [er, eg, eb] = material.emissive;
    (
        [r, g, b, material.metallic, material.roughness, er, eg, eb].map(f32::to_bits),
        material.base_color_texture,
    )
}

/// Make one part of all the parts that share a material, keeping the
/// materials in the order they first appear and the triangles of each in file
/// order. `parts` must cover `indices` without overlap (the loader's do); a
/// list in which every material is already unique is left exactly as it is.
fn merge_parts_by_material(indices: &mut Vec<u32>, parts: &mut Vec<MeshPart>) {
    let mut groups = std::collections::HashMap::new();
    // Per group: its material and index count; per part: its group.
    let mut merged: Vec<(MeshMaterial, usize)> = Vec::new();
    let mut group_of = Vec::with_capacity(parts.len());
    for part in parts.iter() {
        let group = *groups
            .entry(material_key(&part.material))
            .or_insert(merged.len());
        if group == merged.len() {
            merged.push((part.material, 0));
        }
        merged[group].1 += part.index_count as usize;
        group_of.push(group);
    }
    let covered: usize = merged.iter().map(|(_, count)| count).sum();
    if merged.len() == parts.len() || covered != indices.len() {
        return;
    }
    // Where each group's run starts, then copy part after part behind it.
    let mut cursor = Vec::with_capacity(merged.len());
    let mut start = 0;
    for (_, count) in &merged {
        cursor.push(start);
        start += count;
    }
    let starts = cursor.clone();
    let mut sorted = vec![0u32; indices.len()];
    for (part, &group) in parts.iter().zip(&group_of) {
        let (from, count) = (part.first_index as usize, part.index_count as usize);
        let Some(source) = indices.get(from..from + count) else {
            return;
        };
        sorted[cursor[group]..cursor[group] + count].copy_from_slice(source);
        cursor[group] += count;
    }
    *indices = sorted;
    *parts = merged
        .iter()
        .zip(starts)
        .map(|(&(material, count), first)| MeshPart {
            first_index: first as u32,
            index_count: count as u32,
            material,
        })
        .collect();
}

/// The most face colours one primitive is split into; a primitive with more
/// (a smooth gradient) has its colours rounded until they fit.
const MAX_FACE_COLORS: usize = 256;

/// Vertex colours two of which differ by more than this within one triangle
/// make it a gradient (which is flattened to the triangle's average).
const FACE_COLOR_TOLERANCE: f32 = 1.0 / 512.0;

/// The parts of a primitive that has vertex colours: its triangles
/// (`indices[first_index..]`, mesh-wide vertex indices starting at `base`)
/// are reordered into runs of one face colour, and each run becomes a part
/// whose base colour is `material`'s times that colour. See the module docs.
fn split_by_face_color(
    out: &mut Builder,
    first_index: usize,
    base: u32,
    colors: &Accessor<'_>,
    material: MeshMaterial,
) {
    let triangles: Vec<[u32; 3]> = out.indices[first_index..].as_chunks::<3>().0.to_vec();
    let mut gradient = false;
    let faces: Vec<[f32; 3]> = triangles
        .iter()
        .map(|triangle| {
            let [a, b, c] = triangle.map(|i| colors.color((i - base) as usize));
            if a == b && a == c {
                return a;
            }
            gradient |=
                (0..3).any(|k| (a[k] - b[k]).abs().max((a[k] - c[k]).abs()) > FACE_COLOR_TOLERANCE);
            [0, 1, 2].map(|k| (a[k] + b[k] + c[k]) / 3.0)
        })
        .collect();
    if gradient {
        out.warn(
            "COLOR_0 varies within triangles: each is drawn in its average colour \
             (vertex colours are applied per face)"
                .into(),
        );
    }
    // Group the triangles by colour, to 10 bits a channel — or to fewer, if
    // that is what it takes to stay within the limit.
    let mut bits = 10;
    let (group_of, groups) = loop {
        let scale = ((1u32 << bits) - 1) as f32;
        let mut seen = std::collections::HashMap::new();
        let mut groups: Vec<[f32; 3]> = Vec::new();
        let group_of: Vec<usize> = faces
            .iter()
            .map(|face| {
                let key = face.map(|c| (c * scale).round() as u32);
                *seen.entry(key).or_insert_with(|| {
                    // The group's colour: its first triangle's, as given.
                    groups.push(*face);
                    groups.len() - 1
                })
            })
            .collect();
        if groups.len() <= MAX_FACE_COLORS || bits == 1 {
            break (group_of, groups);
        }
        bits -= 1;
    };
    if bits < 10 {
        out.warn(format!(
            "COLOR_0 has more than {MAX_FACE_COLORS} face colours in one primitive: \
             rounded to {bits} bits a channel"
        ));
    }
    let mut order: Vec<usize> = (0..triangles.len()).collect();
    order.sort_by_key(|&triangle| group_of[triangle]);
    out.indices.truncate(first_index);
    let mut run_start = first_index;
    for (position, &triangle) in order.iter().enumerate() {
        out.indices.extend(triangles[triangle]);
        let group = group_of[triangle];
        if order.get(position + 1).map(|&next| group_of[next]) != Some(group) {
            let color = groups[group];
            out.parts.push(MeshPart {
                first_index: run_start as u32,
                index_count: (out.indices.len() - run_start) as u32,
                material: MeshMaterial {
                    base_color: [0, 1, 2].map(|k| material.base_color[k] * color[k]),
                    ..material
                },
            });
            run_start = out.indices.len();
        }
    }
}

/// Decode glTF image `image_index` from its buffer view in the BIN chunk.
fn load_image(
    doc: &Value,
    bin: &[u8],
    image_index: usize,
    budget: usize,
) -> Result<Texture, MeshError> {
    let image = array(&doc["images"])
        .get(image_index)
        .ok_or_else(|| MeshError("does not exist".into()))?;
    if image["uri"].is_string() {
        return err("is not embedded in the file (uri)");
    }
    let view = index(&image["bufferView"])
        .and_then(|i| array(&doc["bufferViews"]).get(i))
        .ok_or_else(|| MeshError("has no bufferView".into()))?;
    if index(&view["buffer"]).unwrap_or(0) != 0 {
        return err("uses a buffer other than the file's BIN chunk");
    }
    let offset = index(&view["byteOffset"]).unwrap_or(0);
    let bytes = index(&view["byteLength"])
        .and_then(|length| offset.checked_add(length))
        .and_then(|end| bin.get(offset..end))
        .ok_or_else(|| MeshError("runs past the end of the BIN chunk".into()))?;
    decode_image(bytes, budget)
}

/// The texture slot of `material`'s `baseColorTexture`, if it has one the
/// renderer can use. Everything it cannot honour is a warning, not an error.
fn base_color_texture(
    doc: &Value,
    bin: &[u8],
    material: &Value,
    has_uv: bool,
    out: &mut Builder,
) -> Option<u32> {
    let pbr = &material["pbrMetallicRoughness"];
    for (slot, present) in [
        (
            "metallicRoughnessTexture",
            !pbr["metallicRoughnessTexture"].is_null(),
        ),
        ("normalTexture", !material["normalTexture"].is_null()),
        ("occlusionTexture", !material["occlusionTexture"].is_null()),
    ] {
        if present {
            out.warn(format!(
                "{slot} is not supported: the material's factors are used alone"
            ));
        }
    }
    let info = &pbr["baseColorTexture"];
    if info.is_null() {
        return None;
    }
    if !has_uv {
        out.warn("a textured primitive has no TEXCOORD_0: drawn untextured".into());
        return None;
    }
    if info["texCoord"].as_u64().unwrap_or(0) != 0 {
        out.warn("a baseColorTexture uses a second UV set: drawn untextured".into());
        return None;
    }
    if !info["extensions"]["KHR_texture_transform"].is_null() {
        out.warn("KHR_texture_transform is ignored: the texture is mapped untransformed".into());
    }
    let Some(texture) = index(&info["index"]).and_then(|i| array(&doc["textures"]).get(i)) else {
        out.warn("a baseColorTexture names a texture that does not exist".into());
        return None;
    };
    if let Some(sampler) = index(&texture["sampler"]).and_then(|i| array(&doc["samplers"]).get(i)) {
        let wrap = |key: &str| sampler[key].as_u64().unwrap_or(WRAP_REPEAT);
        if wrap("wrapS") != WRAP_REPEAT || wrap("wrapT") != WRAP_REPEAT {
            out.warn("texture wrap modes other than REPEAT are ignored".into());
        }
    }
    let Some(image_index) = index(&texture["source"]) else {
        out.warn(
            "a texture has no PNG/JPEG source (other image formats are not supported): \
             drawn untextured"
                .into(),
        );
        return None;
    };
    out.image(doc, bin, image_index)
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

fn material_of(
    doc: &Value,
    bin: &[u8],
    primitive: &Value,
    has_uv: bool,
    out: &mut Builder,
) -> MeshMaterial {
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
    // Alpha is not used: say what that does to a material that relies on it.
    match material["alphaMode"].as_str() {
        Some("BLEND") => {
            out.warn("alphaMode BLEND is not supported: the material is drawn opaque".into());
        }
        Some("MASK") => out
            .warn("alphaMode MASK is not supported: the material is drawn without cut-outs".into()),
        _ => {}
    }
    let strength = material["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"]
        .as_f64()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(1.0);
    let emissive = floats::<3>(&material["emissiveFactor"]).unwrap_or([0.0; 3]);
    let mut emissive = emissive.map(|c| (c.clamp(0.0, 1.0) * strength).min(1.0e6) as f32);
    // glTF: what a surface emits is this factor times its emissive texture.
    // The renderer cannot sample that texture, so the whole surface is given
    // the texture's average: the right amount of light, without the picture.
    if !material["emissiveTexture"].is_null() {
        let average = index(&material["emissiveTexture"]["index"])
            .and_then(|i| array(&doc["textures"]).get(i))
            .and_then(|texture| index(&texture["source"]))
            .and_then(|image| out.image_average(doc, bin, image));
        match average {
            Some(average) => {
                emissive = [0, 1, 2].map(|k| emissive[k] * average[k]);
                out.warn(
                    "emissiveTexture is not supported: the material glows evenly, in the \
                     texture's average colour"
                        .into(),
                );
            }
            None => out.warn(
                "emissiveTexture is not supported and could not be read: the emissive \
                 factor is used alone"
                    .into(),
            ),
        }
    }
    MeshMaterial {
        base_color: [0, 1, 2].map(|i| base[i].clamp(0.0, 1.0) as f32),
        metallic: unit(&pbr["metallicFactor"], 1.0),
        roughness: unit(&pbr["roughnessFactor"], 1.0),
        emissive,
        base_color_texture: base_color_texture(doc, bin, material, has_uv, out),
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
    let uv_index = index(&primitive["attributes"]["TEXCOORD_0"]);
    let uvs = match uv_index {
        Some(i) => {
            let uvs = accessor(doc, bin, i, "VEC2", 2)?;
            if !matches!(uvs.component, COMPONENT_F32 | COMPONENT_U8 | COMPONENT_U16) {
                return err("TEXCOORD_0 is neither float nor a normalised unsigned integer");
            }
            if uvs.count != positions.count {
                return err("TEXCOORD_0 and POSITION have different counts");
            }
            Some(uvs)
        }
        None => None,
    };
    let colors = match index(&primitive["attributes"]["COLOR_0"]) {
        Some(i) => {
            let rgba = array(&doc["accessors"])
                .get(i)
                .is_some_and(|acc| acc["type"].as_str() == Some("VEC4"));
            let colors = if rgba {
                accessor(doc, bin, i, "VEC4", 4)?
            } else {
                accessor(doc, bin, i, "VEC3", 3)?
            };
            if !matches!(
                colors.component,
                COMPONENT_F32 | COMPONENT_U8 | COMPONENT_U16
            ) {
                return err("COLOR_0 is neither float nor a normalised unsigned integer");
            }
            if colors.count != positions.count {
                return err("COLOR_0 and POSITION have different counts");
            }
            Some(colors)
        }
        None => None,
    };

    // Vertices this node already loaded from the same accessors are reused.
    // (Only with normals from the file: generated normals depend on the
    // primitive's own triangles, so those vertices stay private to it.)
    let share_key = normal_index.map(|n| (node_index, position_index, n, uv_index));
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
                    uv: uvs.as_ref().map_or([0.0; 2], |t| t.uv(i)),
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
    if determinant3(world) < 0.0 {
        // A mirroring transform turns counter-clockwise triangles clockwise
        // (glTF: a node with a negative determinant has reversed winding), so
        // put them back: front faces stay counter-clockwise in model space.
        for tri in out.indices[first_index..].as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
    }
    if normals.is_none() {
        // Without NORMAL the vertices are this primitive's own, at the end.
        generate_normals(
            &mut out.vertices[base as usize..],
            &out.indices[first_index..],
            base,
        );
    }
    let material = material_of(doc, bin, primitive, uvs.is_some(), out);
    match colors {
        // Vertex colours: one part per face colour.
        Some(colors) => split_by_face_color(out, first_index, base, &colors, material),
        None => out.parts.push(MeshPart {
            first_index: first_index as u32,
            index_count: added as u32,
            material,
        }),
    }
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

// ---- procedural models ---------------------------------------------------------

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// One horizontal cross-section of a [`Shape::loft`]: an ellipse at height
/// `y`, centred on (`centre[0]`, `centre[1]`) in x and z.
#[derive(Clone, Copy)]
struct Ring {
    y: f32,
    centre: [f32; 2],
    /// Radii along x and z.
    radii: [f32; 2],
}

impl Ring {
    /// Point `i` of `segments` around the ring (`i == segments` is point 0
    /// again, bit for bit, so the seam welds).
    fn point(&self, i: usize, segments: usize) -> [f32; 3] {
        let (sin, cos) = Self::angle(i, segments).sin_cos();
        [
            self.centre[0] + self.radii[0] * cos,
            self.y,
            self.centre[1] + self.radii[1] * sin,
        ]
    }

    fn angle(i: usize, segments: usize) -> f32 {
        (i % segments) as f32 / segments as f32 * std::f32::consts::TAU
    }

    /// Circumference (Ramanujan's approximation for an ellipse).
    fn girth(&self) -> f32 {
        let [a, b] = self.radii;
        std::f32::consts::PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt())
    }
}

/// How many times a texture repeats along `length` at one repeat per `tile`:
/// a whole number (at least one), so the pattern meets itself at the seam.
fn whole_tiles(length: f32, tile: f32) -> f32 {
    (length / tile).round().max(1.0)
}

/// The geometry of one material part under construction. Every primitive it
/// adds is a closed surface wound counter-clockwise seen from outside, with
/// texture coordinates at roughly one repeat per `tile` model units, laid out
/// so that a texture is seen the right way round from outside (never
/// mirrored) and, on every side face, the right way up.
#[derive(Default)]
struct Shape {
    vertices: Vec<MeshVertex>,
    indices: Vec<u32>,
}

impl Shape {
    /// An axis-aligned box: six faces, each its own four vertices so the
    /// normals are flat. The texture's top is +Y on the four side faces and
    /// -Z on the top face (+Z on the bottom one), and its origin the face's
    /// top-left corner seen from outside.
    fn cuboid(&mut self, centre: [f32; 3], half: [f32; 3], tile: f32) {
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                let base = self.vertices.len() as u32;
                let mut normal = [0.0; 3];
                normal[axis] = sign;
                // The image's "up" on this face, and its "right" seen from
                // outside: right x up = the outward normal.
                let up = if axis == 1 {
                    [0.0, 0.0, -sign]
                } else {
                    [0.0, 1.0, 0.0]
                };
                let right = cross3(up, normal);
                let extent =
                    |dir: [f32; 3]| -> f32 { (0..3).map(|k| dir[k].abs() * half[k]).sum() };
                for (du, dv) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let mut offset = [0.0; 3];
                    offset[axis] = sign * half[axis];
                    offset[u] = du * half[u];
                    offset[v] = dv * half[v];
                    let along = |dir: [f32; 3]| -> f32 { (0..3).map(|k| dir[k] * offset[k]).sum() };
                    let mut p = centre;
                    p[axis] += sign * half[axis];
                    p[u] += du * half[u];
                    p[v] += dv * half[v];
                    self.vertices.push(MeshVertex {
                        position: p,
                        normal,
                        uv: [
                            (extent(right) + along(right)) / tile,
                            (extent(up) - along(up)) / tile,
                        ],
                    });
                }
                // Counter-clockwise seen from outside.
                let quad = if sign > 0.0 {
                    [0, 1, 2, 0, 2, 3]
                } else {
                    [0, 2, 1, 0, 3, 2]
                };
                self.indices.extend(quad.map(|i| base + i));
            }
        }
    }

    /// A tube through `rings` (given bottom to top), smooth-shaded around and
    /// along, with a flat cap at each end.
    fn loft(&mut self, rings: &[Ring], segments: usize, tile: f32) {
        let (Some(first), Some(last)) = (rings.first(), rings.last()) else {
            return;
        };
        let base = self.vertices.len() as u32;
        let widest = rings.iter().map(Ring::girth).fold(0.0, f32::max);
        let u_tiles = whole_tiles(widest, tile);
        // `v` runs down the tube from 0 at its top, so the texture is upright.
        let step = |a: &Ring, b: &Ring| {
            let (dx, dz) = (b.centre[0] - a.centre[0], b.centre[1] - a.centre[1]);
            (dx * dx + (b.y - a.y).powi(2) + dz * dz).sqrt() / tile
        };
        let mut down: f32 = rings.windows(2).map(|w| step(&w[0], &w[1])).sum();
        for (j, ring) in rings.iter().enumerate() {
            let below = &rings[j.saturating_sub(1)];
            let above = &rings[(j + 1).min(rings.len() - 1)];
            if j > 0 {
                down = (down - step(below, ring)).max(0.0);
            }
            for i in 0..=segments {
                // The surface's two tangents: up the tube and around it.
                let (lower, upper) = (below.point(i, segments), above.point(i, segments));
                let up = [0, 1, 2].map(|k| upper[k] - lower[k]);
                let (sin, cos) = Ring::angle(i, segments).sin_cos();
                let around = [-ring.radii[0] * sin, 0.0, ring.radii[1] * cos];
                self.vertices.push(MeshVertex {
                    position: ring.point(i, segments),
                    normal: normalised(cross3(up, around)),
                    // (The ring runs anticlockwise seen from above: `u` has to
                    // run the other way for the texture not to be mirrored.)
                    uv: [u_tiles * (1.0 - i as f32 / segments as f32), down],
                });
            }
        }
        let row = segments as u32 + 1;
        for j in 0..rings.len() as u32 - 1 {
            for i in 0..segments as u32 {
                let a = base + j * row + i;
                let b = a + row;
                self.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        self.cap(first, segments, false, tile);
        self.cap(last, segments, true, tile);
    }

    /// The flat end of a loft: a fan over `ring`, facing up or down.
    fn cap(&mut self, ring: &Ring, segments: usize, up: bool, tile: f32) {
        let base = self.vertices.len() as u32;
        let normal = [0.0, if up { 1.0 } else { -1.0 }, 0.0];
        self.vertices.push(MeshVertex {
            position: [ring.centre[0], ring.y, ring.centre[1]],
            normal,
            uv: [0.5, 0.5],
        });
        // Seen from outside, +X is to the right on the top cap and to the
        // left on the bottom one.
        let side = if up { 1.0 } else { -1.0 };
        for i in 0..segments {
            let p = ring.point(i, segments);
            self.vertices.push(MeshVertex {
                position: p,
                normal,
                uv: [
                    0.5 + side * (p[0] - ring.centre[0]) / tile,
                    0.5 + (p[2] - ring.centre[1]) / tile,
                ],
            });
        }
        let n = segments as u32;
        for i in 0..n {
            let (p, q) = (base + 1 + i, base + 1 + (i + 1) % n);
            self.indices
                .extend(if up { [base, q, p] } else { [base, p, q] });
        }
    }

    /// An ellipsoid: `segments` around, `bands` from pole to pole.
    fn ellipsoid(
        &mut self,
        centre: [f32; 3],
        radii: [f32; 3],
        segments: usize,
        bands: usize,
        tile: f32,
    ) {
        let base = self.vertices.len() as u32;
        let equator = Ring {
            y: 0.0,
            centre: [0.0; 2],
            radii: [radii[0], radii[2]],
        };
        let meridian = Ring {
            y: 0.0,
            centre: [0.0; 2],
            radii: [radii[0].max(radii[2]), radii[1]],
        };
        let u_tiles = whole_tiles(equator.girth(), tile);
        let v_tiles = whole_tiles(0.5 * meridian.girth(), tile);
        for j in 0..=bands {
            // The poles are exact points, so every triangle around them meets.
            let (sin_lat, cos_lat) = if j == 0 {
                (-1.0, 0.0)
            } else if j == bands {
                (1.0, 0.0)
            } else {
                let latitude = (j as f32 / bands as f32 - 0.5) * std::f32::consts::PI;
                latitude.sin_cos()
            };
            for i in 0..=segments {
                let (sin, cos) = Ring::angle(i, segments).sin_cos();
                let unit = [cos_lat * cos, sin_lat, cos_lat * sin];
                self.vertices.push(MeshVertex {
                    position: [0, 1, 2].map(|k| centre[k] + radii[k] * unit[k]),
                    // The gradient of (x/a)^2 + (y/b)^2 + (z/c)^2.
                    normal: normalised([0, 1, 2].map(|k| unit[k] / radii[k])),
                    uv: [
                        u_tiles * (1.0 - i as f32 / segments as f32),
                        v_tiles * (1.0 - j as f32 / bands as f32),
                    ],
                });
            }
        }
        let row = segments as u32 + 1;
        for j in 0..bands as u32 {
            for i in 0..segments as u32 {
                let a = base + j * row + i;
                let b = a + row;
                // A band next to a pole is a row of triangles, not quads.
                if j > 0 {
                    self.indices.extend([a, b, a + 1]);
                }
                if j + 1 < bands as u32 {
                    self.indices.extend([a + 1, b, b + 1]);
                }
            }
        }
    }
}

/// One mesh out of shapes: a part per (non-empty) shape, in order.
fn assemble(shapes: Vec<(Shape, MeshMaterial)>, textures: Vec<Texture>) -> MeshData {
    let mut out = Builder {
        textures,
        ..Builder::default()
    };
    for (shape, material) in shapes {
        if shape.indices.is_empty() {
            continue;
        }
        let base = out.vertices.len() as u32;
        let first_index = out.indices.len() as u32;
        out.vertices.extend(shape.vertices);
        out.indices.extend(shape.indices.iter().map(|i| base + i));
        out.parts.push(MeshPart {
            first_index,
            index_count: shape.indices.len() as u32,
            material,
        });
    }
    // `finish` only fails for a mesh without triangles.
    out.finish().unwrap_or_else(|_| MeshData {
        vertices: Vec::new(),
        indices: Vec::new(),
        parts: Vec::new(),
        textures: Vec::new(),
        bounds_min: [0.0; 3],
        bounds_max: [0.0; 3],
    })
}

/// A positive tile size (a non-positive or NaN one becomes 1).
fn tile_or_one(tile: f32) -> f32 {
    if tile > 0.0 {
        tile
    } else {
        1.0
    }
}

impl MeshData {
    /// A box centred on the origin, `half` model units from the centre to each
    /// face: one material part, closed, flat-shaded. The texture repeats once
    /// per `tile` model units across every face. (The textured counterpart of
    /// a box instance — a floor, a wall, a crate.)
    #[must_use]
    pub fn cuboid(half: [f32; 3], tile: f32, material: MeshMaterial) -> Self {
        let mut shape = Shape::default();
        shape.cuboid([0.0; 3], half, tile_or_one(tile));
        assemble(vec![(shape, material)], Vec::new())
    }

    /// A rectangle in the XZ plane at y = 0, facing +Y, `half[0]` by `half[1]`
    /// model units from its centre to its edges in x and z. The texture's
    /// origin is at the (-x, -z) corner, `u` runs along +X and `v` along +Z,
    /// repeating `tiles` times across: seen from above with -Z at the top, the
    /// image is upright. One-sided and open (it has no thickness); use
    /// [`MeshData::cuboid`] for a floor that should also block light from
    /// below.
    #[must_use]
    pub fn plane(half: [f32; 2], tiles: [f32; 2], material: MeshMaterial) -> Self {
        let corner = |x: f32, z: f32| MeshVertex {
            position: [x * half[0], 0.0, z * half[1]],
            normal: [0.0, 1.0, 0.0],
            uv: [(x + 1.0) * 0.5 * tiles[0], (z + 1.0) * 0.5 * tiles[1]],
        };
        let shape = Shape {
            vertices: vec![
                corner(-1.0, -1.0),
                corner(1.0, -1.0),
                corner(1.0, 1.0),
                corner(-1.0, 1.0),
            ],
            indices: vec![0, 3, 2, 0, 2, 1],
        };
        assemble(vec![(shape, material)], Vec::new())
    }
}

/// A small multi-part sample model — a stylised ship: a painted, panelled
/// hull, a polished steel nose, brushed gold wings, a red fin, a glossy
/// cockpit and a glowing engine, about 4 units long, centred on its origin.
///
/// **Orientation:** the nose points along -Z, +Y is up (the fin), +X is the
/// starboard wing — the same convention as [`sample_character`].
///
/// Deterministic; used by the tests and by the `make_test_glb` example to
/// produce a `.glb` without any external asset. Two procedural textures: hull
/// plating (hull) and brushed metal (nose and wings).
#[must_use]
pub fn sample_ship() -> MeshData {
    /// Model units per texture repeat.
    const TILE: f32 = 1.6;
    let paint =
        |base_color: [f32; 3], metallic: f32, roughness: f32, texture: Option<u32>| MeshMaterial {
            base_color,
            metallic,
            roughness,
            emissive: [0.0; 3],
            base_color_texture: texture,
        };
    let boxed = |centre: [f32; 3], half: [f32; 3], material: MeshMaterial| {
        let mut shape = Shape::default();
        shape.cuboid(centre, half, TILE);
        (shape, material)
    };
    let (plating, brushed) = (Some(0), Some(1));
    // Hull, nose, wings, tail fin, cockpit, engine.
    // (Mostly dielectric paint: with point lights only and no environment to
    // reflect, a fully metallic hull would be black away from its highlight.)
    let shapes = vec![
        boxed(
            [0.0, 0.0, 0.0],
            [0.5, 0.35, 1.6],
            paint([0.78, 0.79, 0.82], 0.1, 0.5, plating),
        ),
        boxed(
            [0.0, -0.05, -1.9],
            [0.3, 0.2, 0.35],
            paint([0.56, 0.57, 0.58], 1.0, 0.2, brushed),
        ),
        boxed(
            [0.0, -0.1, 0.5],
            [1.9, 0.06, 0.6],
            paint([1.0, 0.72, 0.25], 0.4, 0.4, brushed),
        ),
        boxed(
            [0.0, 0.6, 1.2],
            [0.05, 0.4, 0.4],
            paint([0.8, 0.08, 0.08], 0.0, 0.5, None),
        ),
        boxed(
            [0.0, 0.4, -0.7],
            [0.3, 0.12, 0.45],
            paint([0.05, 0.2, 0.6], 0.0, 0.1, None),
        ),
        boxed(
            [0.0, 0.0, 1.7],
            [0.3, 0.2, 0.12],
            MeshMaterial {
                base_color: [0.1, 0.1, 0.1],
                metallic: 0.0,
                roughness: 0.8,
                emissive: [0.6, 2.2, 6.0],
                base_color_texture: None,
            },
        ),
    ];
    let textures = vec![
        Texture::hull_panels(256, [236, 238, 242]),
        Texture::brushed_metal(128, [240, 240, 240]),
    ];
    assemble(shapes, textures)
}

/// The placeholder player character: a low-poly astronaut, 1.8 units (metres)
/// tall — helmet with a glowing visor, torso, arms hanging slightly away from
/// the body, legs, boots, gloves, belt and a backpack.
///
/// **Orientation and origin:** the origin is on the ground between the feet;
/// +Y is up; the character faces **-Z** (visor, chest light and toes point
/// along -Z, the backpack along +Z) and +X is its right-hand side — the same
/// convention as [`sample_ship`], whose nose points along -Z. To face a
/// direction `d` in the XZ plane, rotate by [`rotation_y`]`(atan2(-d.x, -d.z))`.
///
/// About 1 700 triangles in four material parts — suit (the hull-plating
/// procedural texture), trim, visor (emissive) and pack — every piece a closed
/// surface with outward normals, and texture coordinates that tile at one
/// repeat per 0.8 m (all within 0..=4).
#[must_use]
pub fn sample_character() -> MeshData {
    /// Metres per texture repeat.
    const TILE: f32 = 0.8;
    const ROUND: usize = 16;
    const LIMB: usize = 12;
    let ring = |y: f32, x: f32, z: f32, rx: f32, rz: f32| Ring {
        y,
        centre: [x, z],
        radii: [rx, rz],
    };
    let (mut suit, mut trim, mut visor, mut pack) = (
        Shape::default(),
        Shape::default(),
        Shape::default(),
        Shape::default(),
    );

    // Torso: hips, waist, chest, shoulders.
    suit.loft(
        &[
            ring(0.86, 0.0, 0.0, 0.175, 0.125),
            ring(0.95, 0.0, 0.0, 0.190, 0.130),
            ring(1.08, 0.0, 0.0, 0.165, 0.115),
            ring(1.26, 0.0, 0.0, 0.205, 0.135),
            ring(1.40, 0.0, 0.0, 0.220, 0.130),
            ring(1.47, 0.0, 0.0, 0.170, 0.110),
        ],
        ROUND,
        TILE,
    );
    // Collar, helmet, and the visor bulging out of the helmet's front.
    trim.loft(
        &[
            ring(1.45, 0.0, 0.0, 0.085, 0.085),
            ring(1.53, 0.0, 0.0, 0.080, 0.080),
        ],
        LIMB,
        TILE,
    );
    suit.ellipsoid([0.0, 1.65, 0.0], [0.150, 0.150, 0.155], ROUND, 10, TILE);
    visor.ellipsoid([0.0, 1.66, -0.07], [0.125, 0.07, 0.10], ROUND, 8, TILE);
    // A light on the chest: the front is recognisable from the side too.
    visor.cuboid([0.08, 1.33, -0.135], [0.03, 0.018, 0.012], TILE);
    // Belt and backpack.
    trim.loft(
        &[
            ring(1.015, 0.0, 0.0, 0.190, 0.135),
            ring(1.075, 0.0, 0.0, 0.182, 0.128),
        ],
        ROUND,
        TILE,
    );
    pack.cuboid([0.0, 1.25, 0.16], [0.14, 0.2, 0.07], TILE);

    for side in [-1.0f32, 1.0] {
        // Leg (ankle, calf, knee, thigh, hip) and boot, toes forward.
        suit.loft(
            &[
                ring(0.08, side * 0.110, 0.000, 0.065, 0.072),
                ring(0.30, side * 0.110, 0.008, 0.078, 0.086),
                ring(0.50, side * 0.108, -0.004, 0.070, 0.076),
                ring(0.76, side * 0.105, 0.000, 0.095, 0.105),
                ring(0.93, side * 0.100, 0.000, 0.100, 0.110),
            ],
            LIMB,
            TILE,
        );
        trim.cuboid([side * 0.11, 0.05, -0.05], [0.07, 0.05, 0.14], TILE);
        // Arm (wrist, elbow, upper arm, shoulder), shoulder pad and glove.
        suit.loft(
            &[
                ring(0.86, side * 0.370, -0.030, 0.045, 0.048),
                ring(1.12, side * 0.330, 0.000, 0.054, 0.058),
                ring(1.36, side * 0.280, 0.000, 0.064, 0.068),
                ring(1.44, side * 0.265, 0.000, 0.056, 0.058),
            ],
            LIMB,
            TILE,
        );
        trim.ellipsoid(
            [side * 0.275, 1.43, 0.0],
            [0.085, 0.055, 0.085],
            LIMB,
            6,
            TILE,
        );
        trim.ellipsoid(
            [side * 0.378, 0.80, -0.03],
            [0.05, 0.075, 0.058],
            LIMB,
            6,
            TILE,
        );
    }

    let plain =
        |base_color: [f32; 3], metallic: f32, roughness: f32, texture: Option<u32>| MeshMaterial {
            base_color,
            metallic,
            roughness,
            emissive: [0.0; 3],
            base_color_texture: texture,
        };
    assemble(
        vec![
            (suit, plain([0.90, 0.91, 0.94], 0.0, 0.55, Some(0))),
            (trim, plain([0.90, 0.36, 0.06], 0.0, 0.5, None)),
            (
                visor,
                MeshMaterial {
                    base_color: [0.03, 0.06, 0.08],
                    metallic: 0.0,
                    roughness: 0.15,
                    emissive: [0.5, 2.2, 3.0],
                    base_color_texture: None,
                },
            ),
            (pack, plain([0.42, 0.45, 0.50], 0.3, 0.45, Some(0))),
        ],
        vec![Texture::hull_panels(256, [236, 238, 242])],
    )
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
            assert_eq!(a.material.base_color_texture, b.material.base_color_texture);
        }
        // Both textures come back pixel for pixel (PNG is lossless).
        assert_eq!(ship.textures.len(), 2);
        assert_eq!(loaded.textures, ship.textures);
        assert!(ship.is_closed() && ship.signed_volume() > 0.0);
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

    fn plain(texture: Option<u32>) -> MeshMaterial {
        MeshMaterial {
            base_color: [0.9, 0.8, 0.7],
            metallic: 0.25,
            roughness: 0.5,
            emissive: [0.0; 3],
            base_color_texture: texture,
        }
    }

    /// A mesh with texture coordinates and two textures survives the GLB round
    /// trip exactly: vertices (coordinates included), indices, parts with
    /// their texture slots, and the textures pixel for pixel.
    #[test]
    fn uvs_and_textures_survive_a_round_trip_through_glb() {
        let mut mesh = MeshData::plane([2.0, 1.0], [4.0, 2.0], plain(None));
        // The quad's two triangles as two parts, each with its own texture.
        mesh.parts = vec![
            MeshPart {
                first_index: 0,
                index_count: 3,
                material: plain(Some(0)),
            },
            MeshPart {
                first_index: 3,
                index_count: 3,
                material: plain(Some(1)),
            },
        ];
        assert_eq!(mesh.add_texture(Texture::uv_grid(64)), 0);
        let checker = Texture::checkerboard(8, 4, [255, 0, 0], [0, 0, 255]);
        assert_eq!(mesh.add_texture(checker), 1);
        let uvs: Vec<[f32; 2]> = mesh.vertices.iter().map(|v| v.uv).collect();
        assert_eq!(uvs, [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]]);

        let bytes = mesh.to_glb();
        let (loaded, warnings) =
            MeshData::from_glb_with_warnings(&bytes).expect("our own writer's output loads");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(loaded, mesh);
        assert_eq!(
            loaded.texture_of(&loaded.parts[1].material),
            Some(&mesh.textures[1])
        );
        // A second trip changes nothing, byte for byte.
        assert_eq!(loaded.to_glb(), bytes);

        // The character too: 1 texture shared by two of its four parts.
        let character = sample_character();
        let loaded = MeshData::from_glb(&character.to_glb()).expect("the character loads");
        assert_eq!(loaded.vertices.len(), character.vertices.len());
        for (a, b) in loaded.vertices.iter().zip(&character.vertices) {
            assert_eq!((a.position, a.uv), (b.position, b.uv));
            // (The loader renormalises normals: equal to the last bit or so.)
            for k in 0..3 {
                assert!((a.normal[k] - b.normal[k]).abs() < 1e-6);
            }
        }
        assert_eq!(loaded.indices, character.indices);
        assert_eq!(loaded.textures, character.textures);
        let slots = |m: &MeshData| -> Vec<Option<u32>> {
            m.parts
                .iter()
                .map(|p| p.material.base_color_texture)
                .collect()
        };
        assert_eq!(slots(&loaded), slots(&character));

        // A slot past the texture list is written (and read back) as untextured.
        let mut dangling = MeshData::plane([1.0, 1.0], [1.0, 1.0], plain(Some(7)));
        assert!(dangling.texture_of(&dangling.parts[0].material).is_none());
        let loaded = MeshData::from_glb(&dangling.to_glb()).expect("loads");
        assert_eq!(loaded.parts[0].material.base_color_texture, None);
        dangling.add_texture(Texture::solid([1, 2, 3]));
        assert!(dangling.texture_of(&dangling.parts[0].material).is_none());
    }

    /// `triangle_file` plus texture coordinates (normalised `u16`), an
    /// embedded PNG and a material that uses it.
    fn textured_triangle_file() -> (Value, Vec<u8>) {
        let (mut doc, mut bin) = triangle_file();
        bin.extend_from_slice(&[0, 0]); // align the next view to 4 bytes
        let uv_offset = bin.len();
        for uv in [[0u16, 0], [65535, 0], [0, 65535]] {
            for c in uv {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let png = encode_png(&Texture::checkerboard(4, 2, [255, 0, 0], [0, 0, 255]));
        let png_offset = bin.len();
        bin.extend_from_slice(&png);
        doc["accessors"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "bufferView": 2, "componentType": 5123, "normalized": true,
                "count": 3, "type": "VEC2",
            }));
        doc["bufferViews"].as_array_mut().unwrap().extend([
            serde_json::json!({ "buffer": 0, "byteOffset": uv_offset, "byteLength": 12 }),
            serde_json::json!({ "buffer": 0, "byteOffset": png_offset, "byteLength": png.len() }),
        ]);
        doc["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 2.into();
        doc["images"] = serde_json::json!([{ "bufferView": 3, "mimeType": "image/png" }]);
        doc["samplers"] = serde_json::json!([{ "wrapS": 10497, "wrapT": 10497 }]);
        doc["textures"] = serde_json::json!([{ "source": 0, "sampler": 0 }]);
        doc["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] =
            serde_json::json!({ "index": 0 });
        doc["buffers"][0]["byteLength"] = bin.len().into();
        (doc, bin)
    }

    #[test]
    fn texture_coordinates_and_an_embedded_png_load() {
        let (doc, bin) = textured_triangle_file();
        let (mesh, warnings) =
            MeshData::from_glb_with_warnings(&glb(&doc, &bin)).expect("valid file");
        assert!(warnings.is_empty(), "{warnings:?}");
        let uvs =
            |mesh: &MeshData| -> Vec<[f32; 2]> { mesh.vertices.iter().map(|v| v.uv).collect() };
        assert_eq!(uvs(&mesh), [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        assert_eq!(mesh.parts[0].material.base_color_texture, Some(0));
        let checker = Texture::checkerboard(4, 2, [255, 0, 0], [0, 0, 255]);
        assert_eq!(mesh.textures, [checker]);
        // The factors are still there, to be multiplied with the texture.
        assert_eq!(mesh.parts[0].material.base_color, [0.2, 0.4, 0.6]);

        // The other two encodings: normalised bytes and plain floats (which
        // may leave 0..1, and whose NaN becomes 0).
        let with_uvs = |component: u64, data: &[u8]| {
            let (mut doc, mut bin) = (doc.clone(), bin.clone());
            doc["bufferViews"][2] = serde_json::json!({
                "buffer": 0, "byteOffset": bin.len(), "byteLength": data.len(),
            });
            doc["accessors"][2]["componentType"] = component.into();
            bin.extend_from_slice(data);
            doc["buffers"][0]["byteLength"] = bin.len().into();
            MeshData::from_glb(&glb(&doc, &bin)).expect("valid file")
        };
        assert_eq!(
            uvs(&with_uvs(5121, &[0, 255, 255, 0, 51, 0])),
            [[0.0, 1.0], [1.0, 0.0], [0.2, 0.0]]
        );
        let floats: Vec<u8> = [0.25f32, -1.5, 3.0, f32::NAN, f32::INFINITY, 0.5]
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect();
        assert_eq!(
            uvs(&with_uvs(5126, &floats)),
            [[0.25, -1.5], [3.0, 0.0], [0.0, 0.5]]
        );

        // A JPEG in place of the PNG (still declared `image/png`: the bytes
        // decide, not the label).
        let (mut jpeg_doc, mut jpeg_bin) = (doc.clone(), bin.clone());
        jpeg_doc["bufferViews"][3] = serde_json::json!({
            "buffer": 0, "byteOffset": jpeg_bin.len(),
            "byteLength": crate::texture::TINY_JPEG.len(),
        });
        jpeg_bin.extend_from_slice(&crate::texture::TINY_JPEG);
        jpeg_doc["buffers"][0]["byteLength"] = jpeg_bin.len().into();
        let (mesh, warnings) =
            MeshData::from_glb_with_warnings(&glb(&jpeg_doc, &jpeg_bin)).expect("valid file");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(mesh.parts[0].material.base_color_texture, Some(0));
        assert_eq!(
            (mesh.textures[0].width(), mesh.textures[0].height()),
            (16, 8)
        );

        // Two materials sharing one image: decoded once, one texture.
        let mut shared = doc.clone();
        let second = shared["materials"][0].clone();
        shared["materials"].as_array_mut().unwrap().push(second);
        let mut primitive = shared["meshes"][0]["primitives"][0].clone();
        primitive["material"] = 1.into();
        shared["meshes"][0]["primitives"]
            .as_array_mut()
            .unwrap()
            .push(primitive);
        shared["textures"] = serde_json::json!([{ "source": 0 }, { "source": 0 }]);
        shared["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]["index"] = 1.into();
        // (A different roughness: two materials that draw exactly alike would
        // be merged into one part.)
        shared["materials"][1]["pbrMetallicRoughness"]["roughnessFactor"] = 0.5.into();
        let mesh = MeshData::from_glb(&glb(&shared, &bin)).expect("valid file");
        assert_eq!((mesh.parts.len(), mesh.textures.len()), (2, 1));
        assert!(mesh
            .parts
            .iter()
            .all(|p| p.material.base_color_texture == Some(0)));
    }

    /// Texture features the renderer lacks never fail the load: each is a
    /// warning, and the material falls back to something drawable.
    #[test]
    fn unsupported_texture_features_are_warnings_not_errors() {
        let (doc, bin) = textured_triangle_file();
        let load = |edit: &dyn Fn(&mut Value), bin: &[u8]| {
            let mut doc = doc.clone();
            edit(&mut doc);
            MeshData::from_glb_with_warnings(&glb(&doc, bin)).expect("the model still loads")
        };
        let textured = |mesh: &MeshData| mesh.parts[0].material.base_color_texture.is_some();
        fn info(d: &mut Value) -> &mut Value {
            &mut d["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
        }

        // The other texture slots: named in a warning each, base colour kept.
        let slots = ["normalTexture", "occlusionTexture", "emissiveTexture"];
        let (mesh, warnings) = load(
            &|d| {
                for slot in slots {
                    d["materials"][0][slot] = serde_json::json!({ "index": 0 });
                }
                d["materials"][0]["pbrMetallicRoughness"]["metallicRoughnessTexture"] =
                    serde_json::json!({ "index": 0 });
            },
            &bin,
        );
        assert!(textured(&mesh));
        assert_eq!(warnings.len(), 4, "{warnings:?}");
        for slot in slots.iter().chain(&["metallicRoughnessTexture"]) {
            assert!(warnings.iter().any(|w| w.contains(slot)), "{slot}");
        }

        // KHR_texture_transform, even when required: mapped untransformed.
        let (mesh, warnings) = load(
            &|d| {
                d["extensionsRequired"] = serde_json::json!(["KHR_texture_transform"]);
                info(d)["extensions"] =
                    serde_json::json!({ "KHR_texture_transform": { "scale": [2.0, 2.0] } });
            },
            &bin,
        );
        assert!(textured(&mesh));
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings.iter().all(|w| w.contains("KHR_texture_transform")));

        // A wrap mode other than repeat: the texture is used, repeating.
        let (mesh, warnings) = load(&|d| d["samplers"][0]["wrapS"] = 33071.into(), &bin);
        assert!(textured(&mesh));
        assert_eq!(warnings.len(), 1, "{warnings:?}");

        // Everything below leaves the material untextured, with one warning
        // and no texture decoded for nothing.
        type Edit<'a> = (&'a str, &'a dyn Fn(&mut Value));
        let untextured: [Edit<'_>; 8] = [
            ("a second UV set", &|d| info(d)["texCoord"] = 1.into()),
            ("no texture coordinates", &|d| {
                d["meshes"][0]["primitives"][0]["attributes"]
                    .as_object_mut()
                    .unwrap()
                    .remove("TEXCOORD_0");
            }),
            ("a missing texture", &|d| info(d)["index"] = 5.into()),
            ("a missing image", &|d| {
                d["textures"][0]["source"] = 9.into()
            }),
            ("an image outside the file", &|d| {
                d["images"][0] = serde_json::json!({ "uri": "albedo.png" });
            }),
            ("a texture without a PNG/JPEG source", &|d| {
                d["textures"][0] = serde_json::json!({
                    "extensions": { "KHR_texture_basisu": { "source": 0 } },
                });
            }),
            ("an image view past the BIN chunk", &|d| {
                d["bufferViews"][3]["byteLength"] = 1_000_000.into();
            }),
            ("an image in a second buffer", &|d| {
                d["bufferViews"][3]["buffer"] = 1.into();
            }),
        ];
        for (what, edit) in untextured {
            let (mesh, warnings) = load(edit, &bin);
            assert!(!textured(&mesh), "{what}");
            assert!(mesh.textures.is_empty(), "{what}");
            assert_eq!(warnings.len(), 1, "{what}: {warnings:?}");
        }

        // A corrupt image (here: its header's CRC no longer matches), and one
        // that is not an image at all.
        let png_offset = doc["bufferViews"][3]["byteOffset"].as_u64().unwrap() as usize;
        let mut corrupt = bin.clone();
        corrupt[png_offset + 20] ^= 0xff;
        let mut garbage = bin.clone();
        garbage[png_offset] = b'x';
        for bad in [&corrupt, &garbage] {
            let (mesh, warnings) = load(&|_| {}, bad);
            assert!(!textured(&mesh) && mesh.textures.is_empty());
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert_eq!(mesh.triangle_count(), 1, "the geometry is unaffected");
        }
        // Every prefix of the textured file: must return, not panic.
        let good = glb(&doc, &bin);
        for cut in 0..good.len() {
            let _ = MeshData::from_glb_with_warnings(&good[..cut]);
        }

        // Still errors, not warnings: broken texture *coordinates* (they are
        // geometry) and extensions that change the geometry.
        let fails = |edit: &dyn Fn(&mut Value)| {
            let mut doc = doc.clone();
            edit(&mut doc);
            MeshData::from_glb(&glb(&doc, &bin)).is_err()
        };
        assert!(fails(&|d| d["accessors"][2]["count"] = 2.into()));
        assert!(fails(&|d| d["accessors"][2]["componentType"] = 5125.into()));
        assert!(fails(&|d| d["accessors"][2]["type"] = "VEC3".into()));
        assert!(fails(&|d| {
            d["extensionsRequired"] = serde_json::json!(["KHR_texture_basisu"]);
        }));
    }

    /// Three triangles side by side (nine vertices of their own), unindexed,
    /// with a `COLOR_0` per vertex: `colors` as float RGB.
    fn colored_file(colors: &[[f32; 3]; 9]) -> (Value, Vec<u8>) {
        let mut bin = Vec::new();
        for triangle in 0..3 {
            let x = triangle as f32 * 2.0;
            for p in [[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x, 1.0, 0.0]] {
                for c in p {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        for color in colors {
            for c in color {
                bin.extend_from_slice(&c.to_le_bytes());
            }
        }
        let doc = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{
                "attributes": { "POSITION": 0, "COLOR_0": 1 }, "material": 0,
            }]}],
            "materials": [{ "pbrMetallicRoughness": {
                "baseColorFactor": [0.5, 0.5, 1.0, 1.0],
                "metallicFactor": 0.2, "roughnessFactor": 0.45,
            }}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": 9, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": 9, "type": "VEC3" },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": 108 },
                { "buffer": 0, "byteOffset": 108, "byteLength": 108 },
            ],
            "buffers": [{ "byteLength": 216 }],
        });
        (doc, bin)
    }

    /// Vertex colours paint whole faces: the primitive becomes one part per
    /// face colour (base colour factor times the colour), its triangles
    /// regrouped; the vertices are untouched. Float or normalised integer,
    /// with or without alpha.
    #[test]
    fn vertex_colours_become_one_part_per_face_colour() {
        let (red, green) = ([0.8f32, 0.1, 0.1], [0.1, 0.6, 0.2]);
        // Red, green, red: the two red triangles end up in one part.
        let paint = [red, red, red, green, green, green, red, red, red];
        let (doc, bin) = colored_file(&paint);
        let (mesh, warnings) =
            MeshData::from_glb_with_warnings(&glb(&doc, &bin)).expect("valid file");
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(mesh.vertices.len(), 9);
        assert_eq!(mesh.indices, [0, 1, 2, 6, 7, 8, 3, 4, 5]);
        assert_eq!(mesh.parts.len(), 2);
        let (first, second) = (mesh.parts[0], mesh.parts[1]);
        assert_eq!((first.first_index, first.index_count), (0, 6));
        assert_eq!((second.first_index, second.index_count), (6, 3));
        // The factor (0.5, 0.5, 1.0) times the face colour; the rest as is.
        assert_eq!(first.material.base_color, [0.4, 0.05, 0.1]);
        assert_eq!(second.material.base_color, [0.05, 0.3, 0.2]);
        for part in &mesh.parts {
            assert_eq!(
                (part.material.metallic, part.material.roughness),
                (0.2, 0.45)
            );
        }
        mesh.validate().expect("a valid mesh");

        // The same as normalised bytes with an alpha channel (not used).
        let (mut doc, mut bin) = colored_file(&paint);
        bin.truncate(108);
        for color in paint {
            bin.extend(color.map(|c| (c * 255.0).round() as u8));
            bin.push(77);
        }
        doc["accessors"][1] = serde_json::json!({
            "bufferView": 1, "componentType": 5121, "normalized": true,
            "count": 9, "type": "VEC4",
        });
        doc["bufferViews"][1]["byteLength"] = 36.into();
        let bytes = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(bytes.indices, mesh.indices);
        assert_eq!(bytes.parts.len(), 2);
        for (a, b) in bytes.parts.iter().zip(&mesh.parts) {
            for k in 0..3 {
                let (x, y) = (a.material.base_color[k], b.material.base_color[k]);
                assert!((x - y).abs() < 0.003, "{x} / {y}");
            }
        }

        // All one colour: one part, as without vertex colours — but painted.
        let (doc, bin) = colored_file(&[green; 9]);
        let plain = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(plain.parts.len(), 1);
        assert_eq!(plain.indices, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(plain.parts[0].material.base_color, [0.05, 0.3, 0.2]);

        // A gradient inside a triangle is flattened to its average, and said.
        let mut shaded = [green; 9];
        shaded[0] = [0.4, 0.6, 0.2];
        shaded[1] = [0.7, 0.6, 0.2];
        let (doc, bin) = colored_file(&shaded);
        let (mesh, warnings) =
            MeshData::from_glb_with_warnings(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("COLOR_0") && warnings[0].contains("average"));
        assert_eq!(mesh.parts.len(), 2);
        let average = mesh.parts[0].material.base_color;
        assert!((average[0] - 0.5 * 0.4).abs() < 1e-6, "{average:?}");

        // Broken colours are an error like any other broken attribute.
        let fails = |edit: &dyn Fn(&mut Value)| {
            let (mut doc, bin) = colored_file(&paint);
            edit(&mut doc);
            MeshData::from_glb(&glb(&doc, &bin)).is_err()
        };
        assert!(fails(&|d| d["accessors"][1]["count"] = 8.into()));
        assert!(fails(&|d| d["accessors"][1]["componentType"] = 5125.into()));
        assert!(fails(&|d| d["accessors"][1]["type"] = "VEC2".into()));
        assert!(fails(&|d| d["bufferViews"][1]["byteLength"] = 100.into()));
    }

    /// A smooth vertex-colour gradient over many triangles stays within the
    /// part limit: the colours are rounded, with a warning.
    #[test]
    fn a_vertex_colour_gradient_is_rounded_into_a_bounded_number_of_parts() {
        // 600 triangles, each its own shade.
        let count = 600usize;
        let mut bin = Vec::new();
        for triangle in 0..count {
            let x = triangle as f32;
            for p in [[x, 0.0, 0.0], [x + 0.5, 0.0, 0.0], [x, 1.0, 0.0]] {
                for c in p {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        for triangle in 0..count {
            let shade = triangle as f32 / (count - 1) as f32;
            for _ in 0..3 {
                for c in [shade, 1.0 - shade, 0.5] {
                    bin.extend_from_slice(&c.to_le_bytes());
                }
            }
        }
        let bytes = count * 36;
        let doc = serde_json::json!({
            "asset": { "version": "2.0" },
            "scenes": [{ "nodes": [0] }],
            "nodes": [{ "mesh": 0 }],
            "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0, "COLOR_0": 1 } }]}],
            "accessors": [
                { "bufferView": 0, "componentType": 5126, "count": count * 3, "type": "VEC3" },
                { "bufferView": 1, "componentType": 5126, "count": count * 3, "type": "VEC3" },
            ],
            "bufferViews": [
                { "buffer": 0, "byteOffset": 0, "byteLength": bytes },
                { "buffer": 0, "byteOffset": bytes, "byteLength": bytes },
            ],
            "buffers": [{ "byteLength": 2 * bytes }],
        });
        let (mesh, warnings) =
            MeshData::from_glb_with_warnings(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(mesh.triangle_count(), count);
        assert!(
            mesh.parts.len() <= MAX_FACE_COLORS && mesh.parts.len() > 100,
            "{} parts",
            mesh.parts.len()
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("rounded"), "{warnings:?}");
        // Still a gradient: dark red first, bright red last.
        let reds: Vec<f32> = mesh
            .parts
            .iter()
            .map(|p| p.material.base_color[0])
            .collect();
        assert!(reds.windows(2).all(|w| w[0] < w[1]), "{reds:?}");
        mesh.validate().expect("a valid mesh");
    }

    /// Everything drawn with one material is one part, however many nodes and
    /// primitives it came from; the materials keep the order they first
    /// appear in, and a file whose materials all differ is left as it was.
    #[test]
    fn parts_that_share_a_material_are_merged() {
        let (mut doc, bin) = triangle_file();
        // The mesh three times (the hierarchy's node, and two more beside
        // it), and a second primitive with another material in between.
        doc["scenes"][0]["nodes"] = serde_json::json!([0, 2, 3]);
        doc["nodes"].as_array_mut().unwrap().extend([
            serde_json::json!({ "translation": [0.0, 5.0, 0.0], "mesh": 1 }),
            serde_json::json!({ "translation": [0.0, 9.0, 0.0], "mesh": 0 }),
        ]);
        doc["meshes"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "primitives": [
                    { "attributes": { "POSITION": 0 }, "indices": 1, "material": 1 },
                    { "attributes": { "POSITION": 0 }, "indices": 1, "material": 0 },
                ],
            }));
        doc["materials"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({
                "pbrMetallicRoughness": { "baseColorFactor": [0.9, 0.9, 0.1, 1.0] },
            }));
        let mesh = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(mesh.triangle_count(), 4);
        // File order: A (node 1), B then A (node 2), A (node 3) — merged into
        // A (three triangles), B (one).
        assert_eq!(mesh.parts.len(), 2);
        let (a, b) = (mesh.parts[0], mesh.parts[1]);
        assert_eq!((a.first_index, a.index_count), (0, 9));
        assert_eq!((b.first_index, b.index_count), (9, 3));
        assert_eq!(a.material.base_color, [0.2, 0.4, 0.6]);
        assert_eq!(b.material.base_color, [0.9, 0.9, 0.1]);
        // A's triangles are in file order, B's after them: the y of each
        // triangle's first corner says which node it came from.
        let heights: Vec<f32> = mesh
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| mesh.vertices[t[0] as usize].position[1])
            .collect();
        assert_eq!(heights, [0.0, 5.0, 9.0, 5.0]);
        mesh.validate().expect("a valid mesh");
    }

    /// An emissive texture cannot be sampled, so the material glows evenly:
    /// its factor (and strength) times the texture's average colour, taken in
    /// linear light — not the factor alone, which is the texture's white.
    #[test]
    fn an_emissive_texture_dims_the_glow_to_its_average() {
        // The file's image is half red, half blue; its material emits
        // (1, 0.5, 0) at strength 4.
        let (mut doc, bin) = textured_triangle_file();
        let plain = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
        assert_eq!(plain.parts[0].material.emissive, [4.0, 2.0, 0.0]);
        doc["materials"][0]["emissiveTexture"] = serde_json::json!({ "index": 0 });
        let (mesh, warnings) =
            MeshData::from_glb_with_warnings(&glb(&doc, &bin)).expect("valid file");
        let glow = mesh.parts[0].material.emissive;
        assert!(
            (glow[0] - 2.0).abs() < 1e-4 && glow[1] == 0.0 && glow[2] == 0.0,
            "{glow:?}"
        );
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("emissiveTexture") && warnings[0].contains("average"));
        // The image is decoded for its average only: one texture, the base
        // colour's, is kept.
        assert_eq!(mesh.textures.len(), 1);

        // A texture that cannot be read: the factor alone, and a warning.
        type Edit<'a> = &'a dyn Fn(&mut Value);
        let no_such_texture: Edit<'_> =
            &|d| d["materials"][0]["emissiveTexture"]["index"] = 7.into();
        let no_such_image: Edit<'_> = &|d| {
            d["textures"]
                .as_array_mut()
                .unwrap()
                .push(serde_json::json!({ "source": 9 }));
            d["materials"][0]["emissiveTexture"]["index"] = 1.into();
        };
        for edit in [no_such_texture, no_such_image] {
            let mut broken = doc.clone();
            edit(&mut broken);
            let (mesh, warnings) =
                MeshData::from_glb_with_warnings(&glb(&broken, &bin)).expect("still loads");
            assert_eq!(mesh.parts[0].material.emissive, [4.0, 2.0, 0.0]);
            assert_eq!(warnings.len(), 1, "{warnings:?}");
            assert!(warnings[0].contains("could not be read"), "{warnings:?}");
        }
    }

    /// The alpha modes are not drawn as such, and the load says so instead of
    /// keeping quiet: `BLEND` is opaque, `MASK` has no cut-outs.
    #[test]
    fn alpha_modes_are_warnings() {
        for (mode, word) in [("BLEND", "opaque"), ("MASK", "cut-outs")] {
            let (mut doc, bin) = triangle_file();
            doc["materials"][0]["alphaMode"] = mode.into();
            doc["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] =
                serde_json::json!([0.05, 0.09, 0.12, 0.28]);
            let (mesh, warnings) =
                MeshData::from_glb_with_warnings(&glb(&doc, &bin)).expect("valid file");
            assert_eq!(warnings.len(), 1, "{mode}: {warnings:?}");
            assert!(warnings[0].contains(mode) && warnings[0].contains(word));
            assert_eq!(mesh.parts[0].material.base_color, [0.05, 0.09, 0.12]);
        }
        let (mut doc, bin) = triangle_file();
        doc["materials"][0]["alphaMode"] = "OPAQUE".into();
        let (_, warnings) = MeshData::from_glb_with_warnings(&glb(&doc, &bin)).unwrap();
        assert!(warnings.is_empty(), "{warnings:?}");
    }

    /// A node that mirrors (negative determinant) reverses the winding of its
    /// triangles; the loader puts it back, so front faces stay counter-clockwise
    /// and a generated normal still points out of the front.
    #[test]
    fn a_mirroring_node_keeps_triangles_counter_clockwise() {
        let (mut doc, bin) = triangle_file();
        doc["nodes"][0]["scale"] = serde_json::json!([-2.0, 2.0, 2.0]);
        let mesh = MeshData::from_glb(&glb(&doc, &bin)).expect("valid file");
        let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
        assert_eq!(
            positions,
            [[-20.0, 0.0, 0.0], [-22.0, 0.0, 0.0], [-20.0, 2.0, 0.0]]
        );
        assert_eq!(mesh.indices, [0, 2, 1]);
        for v in &mesh.vertices {
            assert_eq!(v.normal, [0.0, 0.0, 1.0]);
        }
    }

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Every triangle's geometric normal agrees with the normals of its three
    /// vertices: the mesh is wound counter-clockwise seen from where its
    /// normals point, nowhere inside out.
    fn assert_winding_matches_normals(mesh: &MeshData, name: &str) {
        for (t, tri) in mesh.indices.as_chunks::<3>().0.iter().enumerate() {
            let [a, b, c] = tri.map(|i| mesh.vertices[i as usize]);
            let face = cross3(sub(b.position, a.position), sub(c.position, a.position));
            let area = dot(face, face).sqrt();
            assert!(area > 1e-9, "{name}: triangle {t} is degenerate");
            for v in [a, b, c] {
                assert!(
                    dot(face, v.normal) / area > 0.2,
                    "{name}: triangle {t} faces away from its vertex normal"
                );
            }
        }
    }

    /// Seen from outside, every triangle shows its texture the right way
    /// round: with `u` to the right and `v` downwards, the surface faces the
    /// viewer. (Otherwise lettering in a texture would read backwards.)
    fn assert_textures_are_not_mirrored(mesh: &MeshData, name: &str) {
        for (t, tri) in mesh.indices.as_chunks::<3>().0.iter().enumerate() {
            let [a, b, c] = tri.map(|i| mesh.vertices[i as usize]);
            let (e1, e2) = (sub(b.position, a.position), sub(c.position, a.position));
            let (u1, v1) = (b.uv[0] - a.uv[0], b.uv[1] - a.uv[1]);
            let (u2, v2) = (c.uv[0] - a.uv[0], c.uv[1] - a.uv[1]);
            // Triangles are counter-clockwise seen from outside; so must their
            // texture coordinates be, with `v` flipped to point up.
            let area = u1 * -v2 - u2 * -v1;
            let face = cross3(e1, e2);
            if dot(face, face) > 1e-12 {
                assert!(area > 0.0, "{name}: triangle {t} maps its texture mirrored");
            }
        }
    }

    #[test]
    fn the_sample_character_is_a_closed_upright_textured_person() {
        let character = sample_character();
        character.validate().expect("a valid mesh");
        assert_eq!(character, sample_character(), "deterministic");

        // 1.8 m tall, standing on the origin, symmetric left to right, and
        // person-shaped: much taller than wide, wider than deep.
        let (min, max) = (character.bounds_min, character.bounds_max);
        assert!(min[1].abs() < 1e-6, "feet on the ground: {}", min[1]);
        assert!((max[1] - 1.8).abs() < 0.005, "height {}", max[1]);
        assert!((min[0] + max[0]).abs() < 1e-5, "symmetric: {min:?} {max:?}");
        let (width, depth) = (max[0] - min[0], max[2] - min[2]);
        assert!((0.6..0.9).contains(&width), "width {width}");
        assert!((0.3..0.5).contains(&depth), "depth {depth}");

        let triangles = character.triangle_count();
        assert!((1000..5000).contains(&triangles), "{triangles} triangles");
        eprintln!(
            "character: {triangles} triangles, {} vertices",
            character.vertices.len()
        );

        // Unit normals; closed surfaces wound outwards; about a person's
        // volume (the pieces overlap, so somewhat more than the 0.07 m^3 of
        // a real one).
        for v in &character.vertices {
            assert!((dot(v.normal, v.normal).sqrt() - 1.0).abs() < 1e-5);
        }
        assert!(character.is_closed(), "every piece is a closed surface");
        let volume = character.signed_volume();
        assert!((0.06..0.2).contains(&volume), "volume {volume} m^3");
        assert_winding_matches_normals(&character, "character");

        // Texture coordinates tile within 0..=4 and actually vary.
        let (mut uv_min, mut uv_max) = ([f32::MAX; 2], [f32::MIN; 2]);
        for v in &character.vertices {
            for k in 0..2 {
                assert!(v.uv[k].is_finite());
                uv_min[k] = uv_min[k].min(v.uv[k]);
                uv_max[k] = uv_max[k].max(v.uv[k]);
            }
        }
        for k in 0..2 {
            assert!(
                uv_min[k] >= 0.0 && uv_max[k] <= 4.0,
                "{uv_min:?}..{uv_max:?}"
            );
            assert!(uv_max[k] - uv_min[k] > 0.9, "{uv_min:?}..{uv_max:?}");
        }
        assert_textures_are_not_mirrored(&character, "character");

        // Four parts: suit (textured), trim, visor (emissive), pack (textured).
        assert_eq!(character.parts.len(), 4);
        let [suit, trim, visor, pack] = [0, 1, 2, 3].map(|i| character.parts[i]);
        assert_eq!(character.textures.len(), 1);
        assert_eq!(suit.material.base_color_texture, Some(0));
        assert_eq!(pack.material.base_color_texture, Some(0));
        assert_eq!(trim.material.base_color_texture, None);
        assert!(visor.material.emissive.iter().all(|&c| c > 0.4));
        assert!(suit.material.emissive == [0.0; 3] && trim.material.emissive == [0.0; 3]);
        assert!(suit.index_count > trim.index_count, "mostly suit");

        // Facing -Z: the visor and the toes are in front, the pack behind.
        let mean_z = |part: MeshPart| {
            let range = part.first_index as usize..(part.first_index + part.index_count) as usize;
            let z: f32 = character.indices[range.clone()]
                .iter()
                .map(|&i| character.vertices[i as usize].position[2])
                .sum();
            z / range.len() as f32
        };
        assert!(mean_z(visor) < -0.07, "visor at z = {}", mean_z(visor));
        assert!(mean_z(pack) > 0.15, "pack at z = {}", mean_z(pack));
        let toes = character
            .vertices
            .iter()
            .filter(|v| v.position[1] < 0.1)
            .map(|v| v.position[2])
            .fold(f32::MAX, f32::min);
        assert!(toes < -0.15 && toes == min[2], "toes at z = {toes}");
        // The head is the highest thing, centred over the feet.
        let top = character
            .vertices
            .iter()
            .max_by(|a, b| a.position[1].total_cmp(&b.position[1]))
            .unwrap();
        assert!(top.position[0].abs() < 1e-6 && top.position[2].abs() < 1e-6);
    }

    #[test]
    fn closed_and_open_meshes_are_told_apart() {
        let cube = MeshData::cuboid([1.0, 2.0, 3.0], 1.0, plain(None));
        cube.validate().expect("valid");
        assert_eq!(cube.triangle_count(), 12);
        assert!(cube.is_closed());
        assert!((cube.signed_volume() - 48.0).abs() < 1e-4);
        assert_winding_matches_normals(&cube, "cuboid");
        assert_eq!(
            (cube.bounds_min, cube.bounds_max),
            ([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0])
        );
        // The first face is the -X one, 6 units (z) by 4 (y): at one repeat
        // per unit its coordinates run 0..6 across and 0..4 down.
        let uv_max = cube.vertices[..4]
            .iter()
            .fold([0.0f32; 2], |m, v| [m[0].max(v.uv[0]), m[1].max(v.uv[1])]);
        assert_eq!(uv_max, [6.0, 4.0]);
        assert_textures_are_not_mirrored(&cube, "cuboid");
        // The texture's top edge (v = 0) is the top of every side face, and
        // the far (-Z) edge of the top face.
        for v in &cube.vertices {
            if v.normal[1] == 0.0 {
                assert_eq!(v.uv[1] == 0.0, v.position[1] == 2.0, "{v:?}");
            } else if v.normal[1] > 0.0 {
                assert_eq!(v.uv[1] == 0.0, v.position[2] == -3.0, "{v:?}");
            }
        }

        // Inside out: still closed, negative volume.
        let mut inverted = cube.clone();
        for tri in inverted.indices.as_chunks_mut::<3>().0 {
            tri.swap(1, 2);
        }
        assert!(inverted.is_closed());
        assert!((inverted.signed_volume() + 48.0).abs() < 1e-4);
        // One triangle flipped: an edge is used twice in the same direction.
        let mut folded = cube.clone();
        folded.indices.swap(1, 2);
        assert!(!folded.is_closed());
        // A face missing: open.
        let mut open = cube.clone();
        open.indices.truncate(30);
        assert!(!open.is_closed());

        let plane = MeshData::plane([3.0, 2.0], [1.0, 1.0], plain(None));
        plane.validate().expect("valid");
        assert!(!plane.is_closed());
        assert_winding_matches_normals(&plane, "plane");
        assert_eq!(plane.triangle_count(), 2);
        assert_eq!(
            (plane.bounds_min, plane.bounds_max),
            ([-3.0, 0.0, -2.0], [3.0, 0.0, 2.0])
        );
        // The texture's origin is the (-x, -z) corner.
        let origin = plane.vertices.iter().find(|v| v.uv == [0.0, 0.0]).unwrap();
        assert_eq!(origin.position, [-3.0, 0.0, -2.0]);

        // The ship's six boxes and the built-in shapes are valid too.
        sample_ship().validate().expect("valid");
        assert_winding_matches_normals(&sample_ship(), "ship");
        assert_textures_are_not_mirrored(&sample_ship(), "ship");
    }

    #[test]
    fn hand_built_meshes_are_validated() {
        let good = MeshData::cuboid([1.0; 3], 1.0, plain(None));
        let broken = |edit: &dyn Fn(&mut MeshData)| {
            let mut mesh = good.clone();
            edit(&mut mesh);
            mesh.validate().is_err()
        };
        assert!(broken(&|m| m.indices[5] = 24));
        assert!(broken(&|m| m.parts[0].index_count = 37));
        assert!(broken(&|m| m.parts[0].index_count = 35));
        assert!(broken(&|m| m.parts[0].first_index = 3));
        assert!(broken(&|m| m.vertices[3].position[1] = f32::NAN));
        assert!(broken(&|m| m.vertices[0].position[0] = f32::INFINITY));
        // Out-of-range indices make a mesh "not closed" rather than a panic.
        let mut wild = good.clone();
        wild.indices[0] = 9999;
        assert!(!wild.is_closed());
        assert!(wild.signed_volume().is_finite());
    }
}
