"""IC-4 "Kestrel" — 51 GLB validator (pure struct/json/numpy, no bpy needed).

    blender -b --factory-startup --python 51_validate_glb.py -- --glb file.glb [--lod1] [--report out.json]

Parses the GLB container and JSON chunk and checks the delivery rules: magic/version/
length, one embedded buffer, no `uri` anywhere, images via bufferViews,
extensionsRequired absent/empty, extensionsUsed only KHR_materials_emissive_strength,
no Draco/meshopt/skins/animations, every primitive TRIANGLES with NORMAL + TEXCOORD_0,
accessor/bufferView ranges inside the BIN chunk.  Computes the exact world bbox from
the POSITION data and node transforms (glTF Y-up, nose -Z) and the socket node world
positions.  --lod1 additionally requires ONE mesh node, ONE mesh and ONE primitive.
"""
import json
import struct
import sys

import numpy as np

ALLOWED_EXT = {"KHR_materials_emissive_strength"}
CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out = {"lod1": "--lod1" in a}
    for k in ("--glb", "--report"):
        if k in a:
            out[k[2:]] = a[a.index(k) + 1]
    return out


def find_key(o, key, path="$"):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key:
                hits.append(f"{path}.{k}")
            hits += find_key(v, key, f"{path}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            hits += find_key(v, key, f"{path}[{i}]")
    return hits


def node_matrix(n):
    if "matrix" in n:
        return np.array(n["matrix"], dtype=float).reshape(4, 4).T
    t = n.get("translation", [0, 0, 0])
    x, y, z, w = n.get("rotation", [0, 0, 0, 1])
    s = n.get("scale", [1, 1, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.array(s)[None, :]
    M[:3, 3] = t
    return M


def validate(path, lod1=False):
    raw = open(path, "rb").read()
    checks = {}
    magic, version, length = struct.unpack_from("<4sII", raw, 0)
    checks["magic_glTF"] = magic == b"glTF"
    checks["version_2"] = version == 2
    checks["length_matches_file"] = length == len(raw)
    off, chunks = 12, []
    while off < len(raw):
        clen, ctype = struct.unpack_from("<II", raw, off)
        chunks.append((ctype, raw[off + 8: off + 8 + clen]))
        off += 8 + clen
    checks["chunk0_JSON"] = chunks[0][0] == 0x4E4F534A
    checks["chunk1_BIN"] = len(chunks) == 2 and chunks[1][0] == 0x004E4942
    g = json.loads(chunks[0][1].decode("utf-8"))
    binc = chunks[1][1]
    bufs = g.get("buffers", [])
    checks["one_embedded_buffer"] = len(bufs) == 1 and "uri" not in bufs[0] and bufs[0]["byteLength"] <= len(binc)
    uri_hits = find_key(g, "uri")
    checks["no_uri_anywhere"] = not uri_hits
    imgs = g.get("images", [])
    checks["images_via_bufferView"] = all("bufferView" in i and "mimeType" in i for i in imgs)
    checks["extensionsRequired_absent_or_empty"] = not g.get("extensionsRequired")
    used = set(g.get("extensionsUsed", []))
    checks["extensionsUsed_only_emissive_strength"] = used <= ALLOWED_EXT
    checks["no_draco_or_meshopt"] = not ({"KHR_draco_mesh_compression", "EXT_meshopt_compression",
                                          "KHR_meshopt_compression"} & used) and not find_key(g, "KHR_draco_mesh_compression")
    checks["no_skins"] = not g.get("skins")
    checks["no_animations"] = not g.get("animations")
    prims = [p for m in g.get("meshes", []) for p in m["primitives"]]
    checks["all_primitives_TRIANGLES"] = all(p.get("mode", 4) == 4 for p in prims)
    checks["all_primitives_NORMAL"] = all("NORMAL" in p["attributes"] for p in prims)
    checks["all_primitives_TEXCOORD_0"] = all("TEXCOORD_0" in p["attributes"] for p in prims)
    bvs, accs = g.get("bufferViews", []), g.get("accessors", [])
    checks["bufferViews_in_range"] = all(bv.get("byteOffset", 0) + bv["byteLength"] <= bufs[0]["byteLength"] for bv in bvs)
    checks["no_cameras_lights"] = not g.get("cameras") and "KHR_lights_punctual" not in used

    def read(ai):
        a = accs[ai]
        bv = bvs[a["bufferView"]]
        dt = np.dtype(CT[a["componentType"]])
        n = NC[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", dt.itemsize * n)
        if stride == dt.itemsize * n:
            return np.frombuffer(binc, dt, a["count"] * n, start).reshape(a["count"], n)
        rows = [np.frombuffer(binc, dt, n, start + i * stride) for i in range(a["count"])]
        return np.array(rows)

    # world transforms
    nodes = g.get("nodes", [])
    world = [None] * len(nodes)
    scene = g["scenes"][g.get("scene", 0)]

    def walk(i, P):
        world[i] = P @ node_matrix(nodes[i])
        for c in nodes[i].get("children", []):
            walk(c, world[i])
    for r in scene["nodes"]:
        walk(r, np.eye(4))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    tris = 0
    mesh_cache = {}
    for i, n in enumerate(nodes):
        if "mesh" not in n or world[i] is None:
            continue
        mi = n["mesh"]
        if mi not in mesh_cache:
            pts, t = [], 0
            for p in g["meshes"][mi]["primitives"]:
                pts.append(read(p["attributes"]["POSITION"]).astype(float))
                t += (accs[p["indices"]]["count"] if "indices" in p else accs[p["attributes"]["POSITION"]]["count"]) // 3
            mesh_cache[mi] = (np.vstack(pts), t)
        P, t = mesh_cache[mi]
        W = P @ world[i][:3, :3].T + world[i][:3, 3]
        lo = np.minimum(lo, W.min(0)); hi = np.maximum(hi, W.max(0))
        tris += t
    dims = (hi - lo)
    sockets = {}
    for i, n in enumerate(nodes):
        if n.get("name", "").startswith("SOCKET_") and world[i] is not None:
            sockets[n["name"]] = {"pos": [round(float(v), 4) for v in world[i][:3, 3]],
                                  "rotation": n.get("rotation", [0, 0, 0, 1]),
                                  "extras": n.get("extras", {})}
    mesh_nodes = [n for n in nodes if "mesh" in n]
    rep = {
        "file": path, "bytes": len(raw), "generator": g.get("asset", {}).get("generator"),
        "extensionsUsed": sorted(used), "extensionsRequired": g.get("extensionsRequired", []),
        "uri_hits": uri_hits[:10],
        "counts": {"nodes": len(nodes), "mesh_nodes": len(mesh_nodes), "meshes": len(g.get("meshes", [])),
                   "primitives": len(prims), "materials": len(g.get("materials", [])),
                   "images": len(imgs), "textures": len(g.get("textures", [])), "accessors": len(accs),
                   "bufferViews": len(bvs), "triangles_instanced": int(tris)},
        "materials": [m.get("name") for m in g.get("materials", [])],
        "images": [{"name": i.get("name"), "mimeType": i.get("mimeType"),
                    "bytes": bvs[i["bufferView"]]["byteLength"]} for i in imgs if "bufferView" in i],
        "bbox_gltf": {"min": [round(float(v), 4) for v in lo], "max": [round(float(v), 4) for v in hi],
                      "dims_xyz": [round(float(v), 4) for v in dims]},
        "node_names": [n.get("name") for n in nodes],
        "sockets": sockets,
    }
    checks["longest_axis_Z"] = bool(dims[2] >= dims[0] and dims[2] >= dims[1])
    if lod1:
        checks["single_mesh_node"] = len(mesh_nodes) == 1
        checks["single_mesh"] = len(g.get("meshes", [])) == 1
        checks["single_primitive"] = len(prims) == 1
        checks["lod1_tris_8k_15k"] = 8000 <= tris <= 15000
    checks["size_lt_45MB"] = len(raw) < 45 * 1024 * 1024
    rep["checks"] = {k: bool(v) for k, v in checks.items()}
    rep["all_pass"] = all(rep["checks"].values())
    return rep


if __name__ == "__main__":
    a = args()
    rep = validate(a["glb"], a["lod1"])
    if a.get("report"):
        with open(a["report"], "w") as f:
            json.dump(rep, f, indent=1)
    short = {k: v for k, v in rep.items() if k not in ("node_names", "sockets", "images")}
    print("KESTREL_GLB", json.dumps(short))
