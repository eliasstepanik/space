//! Structural validation of the committed 3D assets under `assets/ships/`.
//!
//! Reads `assets/ships/kestrel/manifest.json`, verifies size + SHA-256 of every
//! file it lists, parses the GLB container of both LODs and cross-checks the glTF
//! JSON (primitive modes, triangle counts, no external URIs, no required
//! extensions) against the manifest. Every check names itself in its failure
//! message. No Blender or GPU is needed.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Asset budgets, mirrored in `assets/README.md`.
const MAX_FILE_BYTES: u64 = 45 * 1024 * 1024;
const MAX_TRIS_PER_MESH: u64 = 60_000;
const MAX_MATERIALS: u64 = 16;
/// glTF primitive mode for TRIANGLES.
const MODE_TRIANGLES: u64 = 4;
/// `JSON` / `BIN\0` chunk types, little endian.
const CHUNK_JSON: u32 = 0x4E4F_534A;
const CHUNK_BIN: u32 = 0x004E_4942;

fn kestrel_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent (workspace root)")
        .join("assets/ships/kestrel")
}

fn sha256_file(path: &Path) -> String {
    let mut file =
        std::fs::File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut buf)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn manifest() -> Value {
    let path = kestrel_dir().join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("[manifest] cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("[manifest] invalid JSON: {e}"))
}

fn u32_at(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().expect("4 bytes"))
}

/// Parse and check the GLB container; returns the glTF JSON document.
fn parse_glb(name: &str, data: &[u8]) -> Value {
    assert!(
        data.len() >= 20,
        "[{name}: header] file is only {} bytes, too short for a GLB",
        data.len()
    );
    assert_eq!(&data[0..4], b"glTF", "[{name}: magic] expected `glTF`");
    assert_eq!(u32_at(data, 4), 2, "[{name}: version] expected glTF 2");
    assert_eq!(
        u32_at(data, 8) as usize,
        data.len(),
        "[{name}: total length] header length differs from file size"
    );
    let json_len = u32_at(data, 12) as usize;
    assert_eq!(
        u32_at(data, 16),
        CHUNK_JSON,
        "[{name}: chunk 0] first chunk is not JSON"
    );
    assert_eq!(json_len % 4, 0, "[{name}: JSON chunk] length not 4-aligned");
    let json_end = 20 + json_len;
    assert!(
        json_end <= data.len(),
        "[{name}: JSON chunk length] {json_len} overruns the file ({} bytes)",
        data.len()
    );
    // The BIN chunk must fill the rest of the file exactly.
    assert!(
        json_end + 8 <= data.len(),
        "[{name}: BIN chunk] missing after the JSON chunk"
    );
    let bin_len = u32_at(data, json_end) as usize;
    assert_eq!(
        u32_at(data, json_end + 4),
        CHUNK_BIN,
        "[{name}: chunk 1] second chunk is not BIN"
    );
    assert_eq!(
        json_end + 8 + bin_len,
        data.len(),
        "[{name}: BIN chunk length] chunks do not add up to the file size"
    );
    serde_json::from_slice(&data[20..json_end])
        .unwrap_or_else(|e| panic!("[{name}: JSON chunk] invalid JSON: {e}"))
}

fn has_uri_key(v: &Value) -> bool {
    match v {
        Value::Object(map) => map.contains_key("uri") || map.values().any(has_uri_key),
        Value::Array(items) => items.iter().any(has_uri_key),
        _ => false,
    }
}

fn array<'a>(gltf: &'a Value, key: &str) -> &'a [Value] {
    gltf.get(key)
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice)
}

/// Triangles of one primitive: `indices.count / 3`, or `POSITION.count / 3`
/// for a non-indexed primitive.
fn primitive_tris(name: &str, gltf: &Value, prim: &Value) -> u64 {
    let accessors = array(gltf, "accessors");
    let accessor = prim
        .get("indices")
        .or_else(|| prim.get("attributes").and_then(|a| a.get("POSITION")))
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("[{name}: accessors] primitive has no indices or POSITION"));
    let count = accessors
        .get(accessor as usize)
        .and_then(|a| a.get("count"))
        .and_then(Value::as_u64)
        .unwrap_or_else(|| panic!("[{name}: accessors] accessor {accessor} has no count"));
    assert_eq!(
        count % 3,
        0,
        "[{name}: accessors] accessor {accessor} count {count} is not a multiple of 3"
    );
    count / 3
}

struct GlbStats {
    meshes: usize,
    primitives: usize,
    nodes: usize,
    sockets: usize,
    materials: usize,
    images: usize,
    /// Triangles summed over every mesh *node* (instanced meshes count per use).
    instanced_tris: u64,
}

fn check_gltf(name: &str, gltf: &Value) -> GlbStats {
    assert!(
        !has_uri_key(gltf),
        "[{name}: no uri] a `uri` key exists in the glTF JSON (buffers/images must be embedded)"
    );
    let required = array(gltf, "extensionsRequired");
    assert!(
        required.is_empty(),
        "[{name}: extensionsRequired] must be absent or empty, got {required:?}"
    );
    assert_eq!(
        array(gltf, "buffers").len(),
        1,
        "[{name}: buffers] expected exactly one embedded buffer"
    );

    let meshes = array(gltf, "meshes");
    let mut primitives = 0;
    let mut mesh_tris = Vec::with_capacity(meshes.len());
    for (mi, mesh) in meshes.iter().enumerate() {
        let prims = array(mesh, "primitives");
        let mut tris = 0;
        for (pi, prim) in prims.iter().enumerate() {
            // `mode` defaults to 4 (TRIANGLES) when absent.
            let mode = prim.get("mode").and_then(Value::as_u64).unwrap_or(4);
            assert_eq!(
                mode, MODE_TRIANGLES,
                "[{name}: primitive mode] mesh {mi} primitive {pi} has mode {mode}, expected TRIANGLES (4)"
            );
            tris += primitive_tris(name, gltf, prim);
            primitives += 1;
        }
        assert!(
            tris <= MAX_TRIS_PER_MESH,
            "[{name}: budget] mesh {mi} has {tris} triangles, budget is {MAX_TRIS_PER_MESH}"
        );
        mesh_tris.push(tris);
    }

    let nodes = array(gltf, "nodes");
    let mut instanced_tris = 0;
    let mut sockets = 0;
    for node in nodes {
        if let Some(mi) = node.get("mesh").and_then(Value::as_u64) {
            instanced_tris += mesh_tris
                .get(mi as usize)
                .unwrap_or_else(|| panic!("[{name}: nodes] node references missing mesh {mi}"));
        }
        if node
            .get("name")
            .and_then(Value::as_str)
            .is_some_and(|n| n.starts_with("SOCKET_"))
        {
            sockets += 1;
        }
    }

    let materials = array(gltf, "materials").len();
    assert!(
        materials as u64 <= MAX_MATERIALS,
        "[{name}: budget] {materials} materials, budget is {MAX_MATERIALS}"
    );
    GlbStats {
        meshes: meshes.len(),
        primitives,
        nodes: nodes.len(),
        sockets,
        materials,
        images: array(gltf, "images").len(),
        instanced_tris,
    }
}

fn m_u64(v: &Value, path: &[&str]) -> u64 {
    let mut cur = v;
    for key in path {
        cur = cur
            .get(key)
            .unwrap_or_else(|| panic!("[manifest] missing key `{}`", path.join(".")));
    }
    cur.as_u64()
        .unwrap_or_else(|| panic!("[manifest] `{}` is not an unsigned integer", path.join(".")))
}

#[test]
fn manifest_files_exist_with_matching_size_and_sha256() {
    let dir = kestrel_dir();
    let manifest = manifest();
    assert_eq!(manifest["id"], "kestrel", "[manifest] id");
    let files = manifest["files"]
        .as_object()
        .expect("[manifest] `files` must be an object");
    for required in ["kestrel.glb", "kestrel_lod1.glb", "source/kestrel.blend"] {
        assert!(
            files.contains_key(required),
            "[manifest] `files` lacks an entry for {required}"
        );
    }
    for (rel, entry) in files {
        let path = dir.join(rel);
        // The Docker build context drops `source/` and `previews/`; entries
        // flagged optional may be absent there but are verified when present.
        let optional = entry["optional_in_build_context"].as_bool() == Some(true);
        if optional && !path.exists() {
            eprintln!("assets: {rel} absent (optional in build context), skipped");
            continue;
        }
        let meta = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("[{rel}: exists] {}: {e}", path.display()));
        let want_bytes = entry["bytes"]
            .as_u64()
            .unwrap_or_else(|| panic!("[{rel}: manifest] `bytes` missing"));
        assert_eq!(
            meta.len(),
            want_bytes,
            "[{rel}: size] bytes differ from manifest"
        );
        assert!(
            meta.len() <= MAX_FILE_BYTES,
            "[{rel}: budget] {} bytes exceeds the {MAX_FILE_BYTES} byte limit",
            meta.len()
        );
        let want_sha = entry["sha256"]
            .as_str()
            .unwrap_or_else(|| panic!("[{rel}: manifest] `sha256` missing"));
        assert_eq!(
            sha256_file(&path),
            want_sha.to_ascii_lowercase(),
            "[{rel}: sha256] hash differs from manifest"
        );
    }
}

#[test]
fn glbs_are_valid_and_match_manifest() {
    let dir = kestrel_dir();
    let manifest = manifest();

    let full_data = std::fs::read(dir.join("kestrel.glb")).expect("[kestrel.glb] read");
    let full_json = parse_glb("kestrel.glb", &full_data);
    let full = check_gltf("kestrel.glb", &full_json);
    drop(full_data);

    let lod_data = std::fs::read(dir.join("kestrel_lod1.glb")).expect("[kestrel_lod1.glb] read");
    let lod_json = parse_glb("kestrel_lod1.glb", &lod_data);
    let lod1 = check_gltf("kestrel_lod1.glb", &lod_json);

    // Full model.
    assert!(full.meshes >= 1, "[kestrel.glb: meshes] no meshes");
    assert_eq!(
        full.instanced_tris,
        m_u64(&manifest, &["triangles", "full"]),
        "[kestrel.glb: triangles] accessor total differs from manifest triangles.full"
    );
    assert_eq!(
        full.meshes as u64,
        m_u64(&manifest, &["gltf", "full", "meshes"])
    );
    assert_eq!(
        full.primitives as u64,
        m_u64(&manifest, &["gltf", "full", "primitives"])
    );
    assert_eq!(
        full.nodes as u64,
        m_u64(&manifest, &["gltf", "full", "nodes"])
    );
    assert_eq!(
        full.sockets as u64,
        m_u64(&manifest, &["sockets"]),
        "[kestrel.glb: sockets] SOCKET_* node count differs from manifest"
    );
    assert_eq!(
        full.materials as u64,
        m_u64(&manifest, &["materials"]),
        "[kestrel.glb: materials] count differs from manifest"
    );
    assert_eq!(
        full.images as u64,
        m_u64(&manifest, &["images"]),
        "[kestrel.glb: images] count differs from manifest"
    );

    // Proxy: exactly one mesh, one primitive.
    assert_eq!(
        lod1.meshes, 1,
        "[kestrel_lod1.glb: meshes] expected exactly 1"
    );
    assert_eq!(
        lod1.primitives, 1,
        "[kestrel_lod1.glb: primitives] expected exactly 1"
    );
    assert_eq!(lod1.meshes as u64, m_u64(&manifest, &["lod1", "meshes"]));
    assert_eq!(
        lod1.primitives as u64,
        m_u64(&manifest, &["lod1", "primitives"])
    );
    assert_eq!(
        lod1.instanced_tris,
        m_u64(&manifest, &["triangles", "lod1"]),
        "[kestrel_lod1.glb: triangles] accessor total differs from manifest triangles.lod1"
    );
}

#[test]
fn manifest_lists_the_fourteen_room_ids() {
    let manifest = manifest();
    let rooms = manifest["rooms"]
        .as_array()
        .expect("[manifest] `rooms` must be an array");
    let ids: BTreeSet<&str> = rooms.iter().filter_map(|r| r["id"].as_str()).collect();
    assert_eq!(
        rooms.len(),
        14,
        "[manifest: rooms] expected 14 room entries"
    );
    assert_eq!(ids.len(), 14, "[manifest: rooms] room IDs must be unique");
    for room in rooms {
        assert!(
            room["name"].as_str().is_some_and(|n| !n.is_empty()),
            "[manifest: rooms] room {} has no name",
            room["id"]
        );
    }
}
