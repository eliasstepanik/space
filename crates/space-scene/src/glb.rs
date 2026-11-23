//! A small, bounds-checked GLB (binary glTF 2.0) reader for collision
//! geometry: vertex positions and triangle indices, nothing else.
//!
//! The server cannot depend on the renderer's GLB loader (`space-render`,
//! Vulkan), and a collider needs none of what that loader does (materials,
//! textures, normals, UVs). This reader walks the default scene's node tree
//! (node transforms applied: translation/rotation/scale or a matrix), takes
//! every `TRIANGLES` primitive's `POSITION` accessor and its index accessor
//! (`u8`, `u16` or `u32`; a primitive without indices is a plain triangle
//! list) and returns one merged [`MeshGeometry`].
//!
//! Every offset, length, stride and count is checked against the buffer
//! before it is read, node recursion is bounded ([`MAX_DEPTH`]) and the totals
//! are capped ([`MAX_VERTICES`], [`MAX_TRIANGLES`]), so a malformed file is an
//! error, never a panic or an unbounded allocation. The input is a file
//! compiled into the binary, but the reader does not rely on that.

use serde_json::Value;
use space_sim::shape::MeshGeometry;

/// Deepest node nesting followed.
pub const MAX_DEPTH: usize = 32;
/// Most vertices read from one file.
pub const MAX_VERTICES: usize = 1 << 22;
/// Most triangles read from one file.
pub const MAX_TRIANGLES: usize = 1 << 22;

const MAGIC: u32 = 0x4654_6C67; // "glTF"
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

fn u32_at(b: &[u8], at: usize) -> Result<u32, String> {
    let s = b
        .get(at..at.checked_add(4).ok_or("offset overflow")?)
        .ok_or_else(|| format!("truncated at {at}"))?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn usize_of(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or_else(|| format!("{what} is not an index"))
}

fn f64s(v: &Value, n: usize, what: &str) -> Result<Vec<f64>, String> {
    let a = v
        .as_array()
        .ok_or_else(|| format!("{what} is not an array"))?;
    if a.len() != n {
        return Err(format!("{what} has {} elements, not {n}", a.len()));
    }
    a.iter()
        .map(|x| {
            x.as_f64()
                .filter(|f| f.is_finite())
                .ok_or_else(|| format!("{what} has a non-number"))
        })
        .collect()
}

/// A column-major 4×4 matrix.
type Mat = [f64; 16];

const IDENTITY: Mat = [
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
];

fn mul(a: &Mat, b: &Mat) -> Mat {
    let mut out = [0.0; 16];
    for c in 0..4 {
        for r in 0..4 {
            out[c * 4 + r] = (0..4).map(|k| a[k * 4 + r] * b[c * 4 + k]).sum();
        }
    }
    out
}

fn node_matrix(node: &Value) -> Result<Mat, String> {
    if let Some(m) = node.get("matrix") {
        let v = f64s(m, 16, "node.matrix")?;
        let mut out = [0.0; 16];
        out.copy_from_slice(&v);
        return Ok(out);
    }
    let t = match node.get("translation") {
        Some(v) => f64s(v, 3, "node.translation")?,
        None => vec![0.0; 3],
    };
    let q = match node.get("rotation") {
        Some(v) => f64s(v, 4, "node.rotation")?,
        None => vec![0.0, 0.0, 0.0, 1.0],
    };
    let s = match node.get("scale") {
        Some(v) => f64s(v, 3, "node.scale")?,
        None => vec![1.0; 3],
    };
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let r = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    let mut m = IDENTITY;
    for c in 0..3 {
        for row in 0..3 {
            m[c * 4 + row] = r[row][c] * s[c];
        }
    }
    m[12] = t[0];
    m[13] = t[1];
    m[14] = t[2];
    Ok(m)
}

struct Doc<'a> {
    json: Value,
    bin: &'a [u8],
}

impl Doc<'_> {
    fn get(&self, kind: &str, index: usize) -> Result<&Value, String> {
        self.json
            .get(kind)
            .and_then(|a| a.get(index))
            .ok_or_else(|| format!("{kind}[{index}] does not exist"))
    }

    /// The bytes of accessor `index` as `(data, stride, count, component
    /// type, components per element)`, every range checked.
    fn accessor(&self, index: usize) -> Result<(&[u8], usize, usize, u64, usize), String> {
        let a = self.get("accessors", index)?;
        let view = usize_of(
            a.get("bufferView").ok_or("sparse/empty accessor")?,
            "bufferView",
        )?;
        let count = usize_of(a.get("count").ok_or("accessor.count")?, "accessor.count")?;
        let ctype = a
            .get("componentType")
            .and_then(Value::as_u64)
            .ok_or("accessor.componentType")?;
        let comps = match a.get("type").and_then(Value::as_str) {
            Some("SCALAR") => 1,
            Some("VEC3") => 3,
            other => return Err(format!("unsupported accessor type {other:?}")),
        };
        let size = match ctype {
            5121 => 1,
            5123 => 2,
            5125 | 5126 => 4,
            other => return Err(format!("unsupported component type {other}")),
        };
        let a_off = match a.get("byteOffset") {
            Some(v) => usize_of(v, "accessor.byteOffset")?,
            None => 0,
        };
        let v = self.get("bufferViews", view)?;
        if v.get("buffer").and_then(Value::as_u64) != Some(0) {
            return Err("a buffer view outside the GLB's own buffer".into());
        }
        let v_off = match v.get("byteOffset") {
            Some(x) => usize_of(x, "bufferView.byteOffset")?,
            None => 0,
        };
        let v_len = usize_of(
            v.get("byteLength").ok_or("bufferView.byteLength")?,
            "byteLength",
        )?;
        let elem = size * comps;
        let stride = match v.get("byteStride") {
            Some(x) => usize_of(x, "byteStride")?,
            None => elem,
        };
        if stride < elem {
            return Err(format!("stride {stride} below the element size {elem}"));
        }
        let view_bytes = self
            .bin
            .get(v_off..v_off.checked_add(v_len).ok_or("view overflow")?)
            .ok_or("buffer view past the binary chunk")?;
        let needed = match count {
            0 => 0,
            n => (n - 1)
                .checked_mul(stride)
                .and_then(|x| x.checked_add(elem))
                .and_then(|x| x.checked_add(a_off))
                .ok_or("accessor overflow")?,
        };
        if needed > view_bytes.len() {
            return Err(format!("accessor {index} reads past its view"));
        }
        Ok((&view_bytes[a_off..], stride, count, ctype, comps))
    }
}

/// Read the triangle geometry of a GLB file. See the [module docs](self).
///
/// # Errors
/// A description of the first thing that is malformed or unsupported.
pub fn read_glb(bytes: &[u8]) -> Result<MeshGeometry, String> {
    if u32_at(bytes, 0)? != MAGIC {
        return Err("not a GLB file".into());
    }
    if u32_at(bytes, 4)? != 2 {
        return Err("not glTF 2.0".into());
    }
    let total = usize::try_from(u32_at(bytes, 8)?).map_err(|e| e.to_string())?;
    let bytes = bytes.get(..total).ok_or("shorter than its header says")?;
    let (mut json, mut bin) = (None, None);
    let mut at = 12usize;
    while at < bytes.len() {
        let len = usize::try_from(u32_at(bytes, at)?).map_err(|e| e.to_string())?;
        let kind = u32_at(bytes, at + 4)?;
        let start = at + 8;
        let chunk = bytes
            .get(start..start.checked_add(len).ok_or("chunk overflow")?)
            .ok_or("chunk past the end")?;
        match kind {
            CHUNK_JSON if json.is_none() => json = Some(chunk),
            CHUNK_BIN if bin.is_none() => bin = Some(chunk),
            _ => {}
        }
        at = start + len;
    }
    let json: Value =
        serde_json::from_slice(json.ok_or("no JSON chunk")?).map_err(|e| e.to_string())?;
    let doc = Doc {
        json,
        bin: bin.ok_or("no binary chunk")?,
    };
    let scene = match doc.json.get("scene") {
        Some(v) => usize_of(v, "scene")?,
        None => 0,
    };
    let roots: Vec<usize> = doc
        .get("scenes", scene)?
        .get("nodes")
        .and_then(Value::as_array)
        .ok_or("scene without nodes")?
        .iter()
        .map(|v| usize_of(v, "scene node"))
        .collect::<Result<_, _>>()?;
    let mut mesh = MeshGeometry {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for root in roots {
        walk(&doc, root, &IDENTITY, 0, &mut mesh)?;
    }
    if mesh.indices.is_empty() {
        return Err("no triangles".into());
    }
    Ok(mesh)
}

fn walk(
    doc: &Doc<'_>,
    index: usize,
    parent: &Mat,
    depth: usize,
    out: &mut MeshGeometry,
) -> Result<(), String> {
    if depth > MAX_DEPTH {
        return Err("node tree too deep (or a cycle)".into());
    }
    let node = doc.get("nodes", index)?;
    let m = mul(parent, &node_matrix(node)?);
    if let Some(mesh) = node.get("mesh") {
        let mesh = doc.get("meshes", usize_of(mesh, "node.mesh")?)?;
        let prims = mesh
            .get("primitives")
            .and_then(Value::as_array)
            .ok_or("mesh without primitives")?;
        for prim in prims {
            if prim.get("mode").and_then(Value::as_u64).unwrap_or(4) != 4 {
                continue;
            }
            add_primitive(doc, prim, &m, out)?;
        }
    }
    if let Some(children) = node.get("children").and_then(Value::as_array) {
        for c in children {
            walk(doc, usize_of(c, "child")?, &m, depth + 1, out)?;
        }
    }
    Ok(())
}

fn add_primitive(
    doc: &Doc<'_>,
    prim: &Value,
    m: &Mat,
    out: &mut MeshGeometry,
) -> Result<(), String> {
    let pos = usize_of(
        prim.get("attributes")
            .and_then(|a| a.get("POSITION"))
            .ok_or("primitive without POSITION")?,
        "POSITION",
    )?;
    let (data, stride, count, ctype, comps) = doc.accessor(pos)?;
    if ctype != 5126 || comps != 3 {
        return Err("POSITION is not VEC3 float".into());
    }
    let base = out.vertices.len();
    if base + count > MAX_VERTICES {
        return Err("too many vertices".into());
    }
    let base_u32 = u32::try_from(base).map_err(|e| e.to_string())?;
    for i in 0..count {
        let at = i * stride;
        let f = |k: usize| -> Result<f64, String> {
            let v = f64::from(f32::from_bits(u32_at(data, at + 4 * k)?));
            if v.is_finite() {
                Ok(v)
            } else {
                Err("non-finite vertex".into())
            }
        };
        let (x, y, z) = (f(0)?, f(1)?, f(2)?);
        out.vertices.push([
            m[0] * x + m[4] * y + m[8] * z + m[12],
            m[1] * x + m[5] * y + m[9] * z + m[13],
            m[2] * x + m[6] * y + m[10] * z + m[14],
        ]);
    }
    let indices: Vec<u32> = match prim.get("indices") {
        None => (0..u32::try_from(count).map_err(|e| e.to_string())?).collect(),
        Some(i) => {
            let (data, stride, n, ctype, comps) = doc.accessor(usize_of(i, "indices")?)?;
            if comps != 1 {
                return Err("indices are not SCALAR".into());
            }
            (0..n)
                .map(|k| {
                    let at = k * stride;
                    match ctype {
                        5121 => data
                            .get(at)
                            .map(|b| u32::from(*b))
                            .ok_or("index past the view".into()),
                        5123 => data
                            .get(at..at + 2)
                            .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
                            .ok_or_else(|| "index past the view".to_string()),
                        5125 => u32_at(data, at),
                        _ => Err("indices are not unsigned integers".to_string()),
                    }
                })
                .collect::<Result<_, _>>()?
        }
    };
    if !indices.len().is_multiple_of(3) {
        return Err("index count is not a multiple of 3".into());
    }
    if out.indices.len() + indices.len() / 3 > MAX_TRIANGLES {
        return Err("too many triangles".into());
    }
    for t in indices.as_chunks::<3>().0 {
        if t.iter()
            .any(|&i| usize::try_from(i).map_or(true, |i| i >= count))
        {
            return Err("index past the vertices".into());
        }
        out.indices
            .push([base_u32 + t[0], base_u32 + t[1], base_u32 + t[2]]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::read_glb;

    /// A hand-built GLB: one triangle, a node translated by (1, 2, 3).
    fn tiny() -> Vec<u8> {
        let json = br#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0,"translation":[1,2,3]}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"},{"bufferView":1,"componentType":5123,"count":3,"type":"SCALAR"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36},{"buffer":0,"byteOffset":36,"byteLength":6}],"buffers":[{"byteLength":44}]}"#;
        let mut json = json.to_vec();
        while !json.len().is_multiple_of(4) {
            json.push(b' ');
        }
        let mut bin = Vec::new();
        for v in [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0] {
            bin.extend_from_slice(&v.to_le_bytes());
        }
        for i in [0u16, 1, 2] {
            bin.extend_from_slice(&i.to_le_bytes());
        }
        bin.extend_from_slice(&[0, 0]);
        let total = 12 + 8 + json.len() + 8 + bin.len();
        let mut out = Vec::new();
        out.extend_from_slice(&0x4654_6C67u32.to_le_bytes());
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&(total as u32).to_le_bytes());
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
        out.extend_from_slice(&0x004E_4942u32.to_le_bytes());
        out.extend_from_slice(&bin);
        out
    }

    #[test]
    fn reads_a_tiny_glb_and_applies_the_node_transform() {
        let mesh = read_glb(&tiny()).unwrap();
        assert_eq!(mesh.indices, vec![[0, 1, 2]]);
        assert_eq!(mesh.vertices[1], [2.0, 2.0, 3.0]);
        assert_eq!(mesh.vertices[2], [1.0, 2.0, 4.0]);
    }

    #[test]
    fn malformed_input_is_an_error_not_a_panic() {
        let good = tiny();
        for len in 0..good.len() {
            assert!(read_glb(&good[..len]).is_err(), "truncated to {len}");
        }
        // Flip every byte in turn: never a panic.
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0xA5;
            let _ = read_glb(&b);
        }
        assert!(read_glb(b"not a glb at all").is_err());
    }
}
