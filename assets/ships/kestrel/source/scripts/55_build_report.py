"""IC-4 "Kestrel" — 55 build report assembler (does NOT modify kestrel.blend).

    blender -b --factory-startup ../kestrel.blend --python 55_build_report.py -- --work DIR [--out build_report.json]

Collects the stage reports written by the chain (DIR/rep10..rep40.json), the export /
validation / re-import / enclosure / hero reports (DIR/rep50_*.json, val_*.json,
ri_*.json, enclosure.json, hero_report.json), measures every preview JPEG, checks the
socket orientation in the GLB against the source and writes build_report.json with
file sizes + sha256 of the deliverables and the git commit of the scripts.
"""
import hashlib
import importlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")
V = importlib.import_module("51_validate_glb")

BG_TOP = np.array([0.46, 0.49, 0.54])      # studio backdrop of the 03-23 preview scripts
BG_BOT = np.array([0.24, 0.25, 0.28])
HERO_TOP = np.array([0.20, 0.215, 0.24])
HERO_BOT = np.array([0.075, 0.08, 0.09])


# Analyst notes (task 7 follow-up methodology, see 53_enclosure.py docstring):
# pressurised rooms strict from the centre; unpressurised bays from the max-clearance
# free point with hull-skin back-face accounting; gear wells = closed (0 escapes).
ENCLOSURE_NOTES = {
    "R03a": "PASS (strict) 500/500 front-facing after the dock-trunk fix in 20_exterior.py: the flat dorsal "
            "skin under EXT_DOCK_Collar is opened along its seams (X -0.9..0, Y 1.05..2.05) with a rim "
            "(EXT_HULL_DockOpeningRim); rays up the open ceiling hatch now end on the closed EXT_DOCK_Hatch "
            "(front faces). Before the fix: 494/500 (6 hits on the skin underside).",
    "B01": "Unpressurised sensor bay, nose. All back-face hits are EXT_HULL_Fuselage (hull skin); rays come "
           "from every side (+/-X sides, +Y nose cone, -Z belly) with open space behind = single-layer outer "
           "skin, not a hole. The bay has no aft lining, so some rays travel aft inside the hull and end on "
           "the skin interior around the cockpit shell (hit Y down to 3.6) - still inside the skin.",
    "B02": "Nose gear well: closed in the model (500/500 hits from centre and free point). Centre lies inside "
           "the strut; free-point back faces are the belly skin (-Z) around the well = hull-skin interior.",
    "B03": "Free point 99.8 % front-facing (1 back-face hit on the belly skin).",
    "B04_L": "Main gear well L: closed (500/500). Back faces = EXT_HULL_Sponson_L interior, mostly -Z (sponson belly).",
    "B04_R": "Main gear well R: closed (500/500). Back faces = EXT_HULL_Sponson_R interior, mostly -Z.",
    "B05": "Tank bay: back faces all EXT_HULL_Fuselage interior, +/-X sides and -Z belly, open space behind.",
    "B06": "Reactor bay: centre inside INT_B06_ReactorVessel; from the free point all back faces are the "
           "fuselage skin interior (+/-X, -Z, -Y aft bulkhead skin).",
    "B07": "Engine bay: back faces all EXT_HULL_Fuselage interior, mostly +Z (dorsal skin over the engine "
           "bay) and -Z belly; open space behind.",
    "exterior": "PASS: 2,084 of 4,000 inbound rays hit the ship, 100 % front-facing (unchanged by the dock fix; "
                "the opening lies entirely under the collar skirt).",
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fileinfo(path):
    return {"path": path, "bytes": os.path.getsize(path),
            "MB": round(os.path.getsize(path) / 1048576, 3), "sha256": sha(path)}


def load(p):
    with open(p) as f:
        return json.load(f)


def preview_coverage(path):
    img = bpy.data.images.load(path)
    img.reload()
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    q = px.reshape(h, w, 4)[..., :3]
    name = os.path.basename(path)
    t = np.linspace(0, 1, h).reshape(h, 1, 1)
    if name[:2] in ("24", "25"):
        method = "diff>0.03 vs known hero backdrop (vignetted, conservative)"
        yy, xx = np.mgrid[0:h, 0:w]
        vign = 1.0 - 0.35 * (((xx / w - 0.5) ** 2 + (yy / h - 0.45) ** 2) * 2.2).reshape(h, w, 1)
        bg = (HERO_BOT * (1 - t) + HERO_TOP * t) * np.clip(vign, 0.55, 1.0)
    elif name[:2] in ("01", "02"):
        method = "diff>0.04 vs backdrop estimated from the 4 corner patches"
        top = (q[-6:, :6].reshape(-1, 3).mean(0) + q[-6:, -6:].reshape(-1, 3).mean(0)) / 2
        bot = (q[:6, :6].reshape(-1, 3).mean(0) + q[:6, -6:].reshape(-1, 3).mean(0)) / 2
        bg = bot * (1 - t) + top * t
    else:
        method = "diff>0.03 vs known studio backdrop gradient"
        bg = BG_BOT * (1 - t) + BG_TOP * t
    thr = 0.04 if name[:2] in ("01", "02") else 0.03
    cov = float((np.abs(q - bg).max(axis=2) > thr).mean())
    bpy.data.images.remove(img)
    return {"file": name, "size": [w, h], "bytes": os.path.getsize(path),
            "non_background_fraction": round(cov, 4), "method": method}


def socket_checks(glb_rep):
    """Blender local -Z of every socket (its 'look / thrust' axis) must match glTF local -Y
    rotated by the exported node rotation, after the Z-up -> Y-up conversion."""
    conv = lambda v: np.array([v[0], v[2], -v[1]])
    out, ok_all = {}, True
    for o in bpy.data.collections[C.COL_SOCK].all_objects:
        s = glb_rep["sockets"].get(o.name)
        if s is None:
            out[o.name] = {"present_in_glb": False}
            ok_all = False
            continue
        mw = o.matrix_world
        b_axis = conv((mw.to_3x3() @ Vector((0, 0, -1))).normalized())
        b_pos = conv(mw.translation)
        M = V.node_matrix({"rotation": s["rotation"]})[:3, :3]
        g_axis = M @ np.array([0.0, -1.0, 0.0])
        pos_err = float(np.abs(np.array(s["pos"]) - b_pos).max())
        ang = float(np.degrees(np.arccos(np.clip(np.dot(b_axis, g_axis), -1, 1))))
        ok = pos_err < 2e-3 and ang < 0.1
        ok_all &= ok
        out[o.name] = {"pos_gltf": s["pos"], "pos_err_m": round(pos_err, 5),
                       "axis_neg_local_z_gltf": [round(float(v), 4) for v in g_axis],
                       "axis_angle_err_deg": round(ang, 4), "rotation_xyzw": s["rotation"],
                       "extras": s["extras"], "ok": bool(ok)}
    eye = glb_rep["sockets"]["SOCKET_Camera_PilotEye"]["pos"]
    eng = {k: v["pos"] for k, v in glb_rep["sockets"].items() if k.startswith("SOCKET_Engine_Main_")}
    named = {
        "SOCKET_Camera_PilotEye_z_lt_0": {"z": eye[2], "pass": eye[2] < 0},
        "SOCKET_Engine_Main_z_gt_0": {k: {"z": v[2], "pass": v[2] > 0} for k, v in eng.items()},
        "PilotEye_looks_nose_minus_Z": out["SOCKET_Camera_PilotEye"]["axis_neg_local_z_gltf"],
        "Engine_exhaust_axis_plus_Z": {k: out[k]["axis_neg_local_z_gltf"] for k in eng},
    }
    named["all_pass"] = bool(eye[2] < 0 and all(v[2] > 0 for v in eng.values()) and ok_all
                             and out["SOCKET_Camera_PilotEye"]["axis_neg_local_z_gltf"][2] < -0.99
                             and all(out[k]["axis_neg_local_z_gltf"][2] > 0.99 for k in eng))
    return {"summary": named, "per_socket_all_ok": bool(ok_all), "sockets": out}


def reimport_compare(src, imp):
    s, i = src["per_object"], imp["per_object"]
    only_s, only_i = sorted(set(s) - set(i)), sorted(set(i) - set(s))
    mism = []
    for n in sorted(set(s) & set(i)):
        for k in ("type", "parent", "tris", "mesh"):
            if s[n].get(k) != i[n].get(k):
                mism.append({"node": n, "field": k, "source": s[n].get(k), "reimport": i[n].get(k)})
        if max(abs(a - b) for a, b in zip(s[n]["loc"], i[n]["loc"])) > 1e-3:
            mism.append({"node": n, "field": "loc", "source": s[n]["loc"], "reimport": i[n]["loc"]})
    bb_err = max(abs(a - b) for a, b in zip(src["bbox_min"] + src["bbox_max"], imp["bbox_min"] + imp["bbox_max"]))
    return {"source": {k: src[k] for k in ("objects", "mesh_objects", "meshes", "tris", "bbox_min", "bbox_max")},
            "reimport": {k: imp[k] for k in ("objects", "mesh_objects", "meshes", "tris", "bbox_min", "bbox_max")},
            "tris_match": src["tris"] == imp["tris"], "object_count_match": src["objects"] == imp["objects"],
            "bbox_max_abs_err_m": round(bb_err, 5), "names_only_in_source": only_s,
            "names_only_in_reimport": only_i, "renames": [], "mismatches": mism,
            "materials_match": src["materials"] == imp["materials"],
            "images_match": src["images"] == imp["images"],
            "pass": not only_s and not only_i and not mism and bb_err < 1e-3}


def main():
    a = C.script_args()
    wd = a["work"]
    ship = C.SHIP_DIR
    full, lod = os.path.join(ship, "kestrel.glb"), os.path.join(ship, "kestrel_lod1.glb")
    vfull, vlod = V.validate(full), V.validate(lod, lod1=True)
    stages = {s: load(os.path.join(wd, f"rep{s}.json")) for s in ("10", "20", "30", "31", "40")}
    r40 = stages["40"]
    env = C.SPEC
    dims = vfull["bbox_gltf"]["dims_xyz"]
    env_chk = {
        "longest_axis": "Z" if vfull["checks"]["longest_axis_Z"] else "not Z",
        "nose_direction": "-Z (SOCKET_Camera_PilotEye z<0, nose tip at min z)",
        "x_span_m": {"measured": dims[0], "envelope": env["span"], "dev_pct": round(100 * (dims[0] / env["span"] - 1), 2)},
        "y_height_on_gear_m": {"measured": dims[1], "envelope": env["height_on_gear"],
                               "dev_pct": round(100 * (dims[1] / env["height_on_gear"] - 1), 2)},
        "z_length_m": {"measured": dims[2], "envelope": env["length"], "dev_pct": round(100 * (dims[2] / env["length"] - 1), 2)},
    }
    env_chk["within_10pct"] = all(abs(env_chk[k]["dev_pct"]) <= 10 for k in ("x_span_m", "y_height_on_gear_m", "z_length_m"))
    sock = socket_checks(vfull)
    ri = reimport_compare(load(os.path.join(wd, "ri_src.json")), load(os.path.join(wd, "ri_imp.json")))
    enc = load(os.path.join(wd, "enclosure.json"))
    enc["notes"] = ENCLOSURE_NOTES
    enc["summary"] = {}
    for k, v in enc["rooms"].items():
        fp = v.get("free_point")
        enc["summary"][k] = {
            "class": v["class"], "criterion": v["criterion"], "pass": v["pass"],
            "pass_basis": v.get("pass_basis", "strict"),
            "centre": {"hits": v["hits"], "rays": v["rays"], "front_fraction": v["front_fraction"]},
            "free_point": None if fp is None else {
                "origin": fp["origin"], "clearance_m": fp["clearance_m"], "hits": fp["hits"],
                "front_fraction": fp["front_fraction"],
                "backface_objects": {o: {"hits": e["hits"], "hull_skin": e["hull_skin"], "mean_dir": e["mean_dir"],
                                         "dir_buckets": e["dir_buckets"], "open_space_behind": e["open_space_behind"]}
                                     for o, e in fp["backface_account"]["by_object"].items()},
                "all_backfaces_on_hull_skin": fp["backface_account"]["all_backfaces_on_hull_skin"]}}
    enc["all_pass"] = all(v["pass"] for v in enc["rooms"].values()) and enc["exterior"]["pass"]
    pdir = C.PREVIEW_DIR
    previews = [preview_coverage(os.path.join(pdir, f)) for f in sorted(os.listdir(pdir)) if f.endswith(".jpg")]
    hero = load(os.path.join(wd, "hero_report.json")) if os.path.exists(os.path.join(wd, "hero_report.json")) else []
    try:
        commit = subprocess.check_output(["git", "-C", C.SCRIPTS_DIR, "log", "-1", "--format=%H", "--", "."], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", C.SCRIPTS_DIR, "status", "--porcelain", "--", "."], text=True).strip()
    except Exception as e:  # noqa
        commit, dirty = f"unavailable: {e}", ""
    scripts = {f: sha(os.path.join(C.SCRIPTS_DIR, f)) for f in sorted(os.listdir(C.SCRIPTS_DIR)) if f.endswith(".py")}
    rep = {
        "asset": "IC-4 Kestrel",
        "blender": bpy.app.version_string,
        "frame": "source Blender Z-up, nose +Y, metres; GLB Y-up, nose -Z (export_yup=True)",
        "scripts": {"git_commit": commit, "uncommitted_changes": dirty.splitlines(), "sha256": scripts,
                    "chain": ["10_blockout.py", "20_exterior.py", "30_interior.py", "31_bays.py", "40_materials.py"],
                    "post": ["50_export.py --what full", "50_export.py --what lod1", "51_validate_glb.py",
                             "52_reimport_check.py", "53_enclosure.py", "54_hero_previews.py", "55_build_report.py"]},
        "stages": {
            "10_blockout": {"tris_total": stages["10"]["triangles_total"], "tris_exterior": stages["10"]["triangles_exterior"],
                            "tris_interior": stages["10"]["triangles_interior"]},
            "20_exterior": {"tris_exterior": stages["20"]["exterior_tris"], "tris_total": stages["20"]["triangles_total_scene"]},
            "30_interior": {"tris_interior_rooms": stages["30"]["interior_fitted_tris"], "tris_total": stages["30"]["triangles_total_scene"],
                            "objects": stages["30"]["objects_total"]},
            "31_bays": {"tris_bays": stages["31"]["bays_tris"], "tris_total": stages["31"]["triangles_total_scene"],
                        "objects": stages["31"]["objects_total"]},
            "40_materials": {"tris_total": r40["triangles"], "objects": r40["objects"], "mesh_objects": r40["mesh_objects"],
                             "meshes": r40["meshes"]},
        },
        "final_totals": {"objects": r40["objects"], "mesh_objects": r40["mesh_objects"], "meshes": r40["meshes"],
                         "triangles": r40["triangles"], "triangles_exterior": stages["31"]["exterior_tris"],
                         "triangles_interior_incl_bays": stages["31"]["interior_plus_bays_tris"],
                         "previous_release": {"objects": 1071, "triangles": 417512,
                                              "note": "before the task-7 dock-trunk fix"},
                         "delta_vs_previous": {"objects": r40["objects"] - 1071, "triangles": r40["triangles"] - 417512}},
        "dock_trunk_fix": load(os.path.join(wd, "dock_fix.json")),
        "chain_reproducibility": load(os.path.join(wd, "repro.json")),
        "materials": {"count": len(r40["materials"]), "list": r40["materials"],
                      "object_count": r40["material_object_count"], "mesh_count": r40["material_mesh_count"]},
        "textures": {"list": r40["textures"], "total_packed_bytes": r40["textures_total_bytes"],
                     "total_MB": r40["textures_total_MB"]},
        "glb_full": {"checks": vfull["checks"], "all_pass": vfull["all_pass"], "counts": vfull["counts"],
                     "extensionsUsed": vfull["extensionsUsed"], "extensionsRequired": vfull["extensionsRequired"],
                     "materials": vfull["materials"], "images": vfull["images"], "bbox_gltf": vfull["bbox_gltf"],
                     "envelope_check": env_chk, "export": load(os.path.join(wd, "rep50_full.json"))},
        "glb_lod1": {"checks": vlod["checks"], "all_pass": vlod["all_pass"], "counts": vlod["counts"],
                     "materials": vlod["materials"], "images": vlod["images"], "bbox_gltf": vlod["bbox_gltf"],
                     "build": load(os.path.join(wd, "rep50_lod1.json"))},
        "socket_orientation": sock,
        "reimport_comparison": ri,
        "enclosure": enc,
        "previews": {"count": len(previews), "required_min": 21, "list": previews, "hero_renders": hero,
                     "dock_fix_impact": load(os.path.join(wd, "preview_impact.json"))},
        "files": {"kestrel.glb": fileinfo(full), "kestrel_lod1.glb": fileinfo(lod),
                  "kestrel.blend": fileinfo(C.BLEND_PATH)},
    }
    rep["files"]["kestrel.blend"]["compressed_zstd"] = open(C.BLEND_PATH, "rb").read(4) == b"\x28\xb5\x2f\xfd"
    rep["files"]["all_lt_45MB"] = all(v["bytes"] < 45 * 1048576 for k, v in rep["files"].items() if isinstance(v, dict) and "bytes" in v)
    rep["glb_full"]["export"].pop("nodes", None)
    out = a.get("out") or os.path.join(ship, "build_report.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print("KESTREL_REPORT", out, json.dumps({"glb": vfull["all_pass"], "lod1": vlod["all_pass"], "sockets": sock["summary"]["all_pass"],
                                              "reimport": ri["pass"], "enclosure": enc["all_pass"],
                                              "previews": len(previews)}))


main()
