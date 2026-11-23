"""IC-4 "Kestrel" — 53 enclosure ray checks (does NOT modify kestrel.blend).

    blender -b --factory-startup ../kestrel.blend --python 53_enclosure.py -- --out enclosure.json
        [--rays 500] [--ext-rays 2000] [--seed 7]

Volumes: every clear volume of the blockout table (R03a and both B04 wells are their
own volumes).  `--rays` rays are cast in uniformly random directions against the whole
ship (scene.ray_cast on the evaluated depsgraph, Preview_Rig hidden).  A hit is
front-facing when dot(hit normal, ray dir) < 0.  Criteria by class (task 7 follow-up):

  pressurised rooms (R01-R06, R03a, C01) -- STRICT, from the volume centre:
      100 % hits and >= 99 % front-facing.
  unpressurised bays (B01, B03, B05, B06, B07) -- FREE POINT: rays from the best free
      point (largest clearance on a 7x7x7 grid of the volume; clearance = nearest hit
      of 64 probe rays; the centre can sit inside machinery).  Pass = 100 % hits AND
      (>= 99 % front-facing  OR  every back-face hit lies on a hull-skin object
      (custom prop kestrel_skin: the bay wall IS the inside of the hull skin)).
      Back-face hits are accounted per bay: object, dominant direction bucket,
      mean direction and hit-point bounds; for each, the ray is continued past the
      back face -- open space there means it was the single-layer outer skin, which
      is a hull-skin interior and not a hole (the ray was stopped, nothing escaped).
  gear wells (B02, B04_L, B04_R) -- CLOSED WELL: the wells are closed in the model,
      pass = 100 % hits from the centre AND from the free point (no ray escapes);
      front-facing fraction and back-face accounting are reported, not gated.

The centre result is always reported for every volume.
Exterior: `--ext-rays` rays start on a sphere enclosing the ship and aim at random
points inside the ship bbox; pass = >= 99.5 % of the hits front-facing.
"""
import importlib
import json
import math
import os
import random
import sys

sys.dont_write_bytecode = True

import bpy
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")


def load_interior_table():
    src = open(os.path.join(C.SCRIPTS_DIR, "10_blockout.py")).read()
    start = src.index("INTERIOR = [")
    end = src.index("]\nROOM_IDS")
    ns = {}
    exec(src[start:end + 1], ns)
    return ns["INTERIOR"]


def rand_dir(rng):
    while True:
        v = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        if 1e-4 < v.length <= 1.0:
            return v.normalized()


def cast(scn, dg, o, d):
    hit, loc, nrm, idx, ob, mw = scn.ray_cast(dg, o + d * 1e-4, d)
    if not hit:
        return None
    r = {"front": nrm.dot(d) < 0.0, "obj": ob.name, "dist": (loc - o).length}
    if not r["front"]:
        # what lies behind a back face: more ship (sealed) or open space (ray leaves the ship)
        h2 = scn.ray_cast(dg, loc + d * 1e-4, d)
        r["behind"] = h2[4].name if h2[0] else None
    return r


def room_test(scn, dg, centre, n, rng):
    hits = misses = front = 0
    back_objs, miss_dirs = {}, []
    back_exit = back_sealed = 0
    for _ in range(n):
        d = rand_dir(rng)
        h = cast(scn, dg, centre, d)
        if h is None:
            misses += 1
            if len(miss_dirs) < 5:
                miss_dirs.append([round(v, 3) for v in d])
            continue
        hits += 1
        if h["front"]:
            front += 1
        else:
            back_objs[h["obj"]] = back_objs.get(h["obj"], 0) + 1
            if h["behind"] is None:
                back_exit += 1
            else:
                back_sealed += 1
    ff = front / hits if hits else 0.0
    return {"rays": n, "hits": hits, "misses": misses, "front_hits": front,
            "front_fraction": round(ff, 4), "all_hit": misses == 0,
            "pass": misses == 0 and ff >= 0.99,
            "backface_then_open_space": back_exit, "backface_then_more_ship": back_sealed,
            "backface_objects": dict(sorted(back_objs.items(), key=lambda kv: -kv[1])[:6]),
            "miss_dir_examples": miss_dirs}


PRESSURISED = {"R01", "R02", "R03", "R03a", "R04", "R05", "R06", "C01"}
GEAR_WELLS = {"B02", "B04"}
BUCKETS = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")


def bucket(d):
    ax = max(range(3), key=lambda i: abs(d[i]))
    return ("+" if d[ax] >= 0 else "-") + "XYZ"[ax]


def is_skin(name):
    o = bpy.data.objects.get(name)
    return bool(o is not None and o.get("kestrel_skin"))


def clearance(scn, dg, p, rng, k=64):
    best = 1e9
    for _ in range(k):
        h = scn.ray_cast(dg, p, rand_dir(rng))
        best = min(best, (h[1] - p).length if h[0] else 1e9)
    return best


def best_free_point(scn, dg, box):
    x0, x1, y0, y1, z0, z1 = box
    fr = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9)
    best = None
    for fx in fr:
        for fy in fr:
            for fz in fr:
                p = Vector((x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy, z0 + (z1 - z0) * fz))
                pr = room_test(scn, dg, p, 32, random.Random(3))
                if pr["front_fraction"] < 0.5 or pr["misses"]:
                    continue                    # inside a solid (sees its back faces) / leaks
                c = clearance(scn, dg, p, random.Random(5))
                if best is None or c > best[0] + 1e-6:
                    best = (c, p)
    return best


def backface_account(scn, dg, o, n, rng):
    """Same ray set as room_test(o, n, rng) but with per-back-face-hit detail."""
    acc = {"by_object": {}, "by_direction": {b: 0 for b in BUCKETS}, "skin_hits": 0,
           "nonskin_hits": 0, "then_open_space": 0, "then_more_ship": 0}
    per = {}
    for _ in range(n):
        d = rand_dir(rng)
        h = scn.ray_cast(dg, o + d * 1e-4, d)
        if not h[0] or h[2].dot(d) < 0.0:
            continue
        name = h[4].name
        h2 = scn.ray_cast(dg, h[1] + d * 1e-4, d)
        e = per.setdefault(name, {"hits": 0, "dsum": Vector(), "lo": Vector((1e9,) * 3),
                                  "hi": Vector((-1e9,) * 3), "dirs": {}, "open_behind": 0,
                                  "skin": is_skin(name)})
        e["hits"] += 1
        e["dsum"] += d
        e["lo"] = Vector(map(min, e["lo"], h[1])); e["hi"] = Vector(map(max, e["hi"], h[1]))
        b = bucket(d)
        e["dirs"][b] = e["dirs"].get(b, 0) + 1
        acc["by_direction"][b] += 1
        if not h2[0]:
            e["open_behind"] += 1
            acc["then_open_space"] += 1
        else:
            acc["then_more_ship"] += 1
        acc["skin_hits" if e["skin"] else "nonskin_hits"] += 1
    for name, e in sorted(per.items(), key=lambda kv: -kv[1]["hits"]):
        m = e["dsum"].normalized() if e["dsum"].length > 0 else e["dsum"]
        acc["by_object"][name] = {
            "hits": e["hits"], "hull_skin": e["skin"],
            "role": "hull-skin interior (bay wall = inside of the hull skin)" if e["skin"] else "fitting / structure",
            "mean_dir": [round(v, 3) for v in m], "dir_buckets": e["dirs"],
            "hit_bounds": [[round(v, 3) for v in e["lo"]], [round(v, 3) for v in e["hi"]]],
            "open_space_behind": e["open_behind"]}
    acc["all_backfaces_on_hull_skin"] = acc["nonskin_hits"] == 0
    return acc


def main():
    a = C.script_args()
    n = int(a.get("rays", 500))
    ne = int(a.get("ext-rays", 2000))
    seed = int(a.get("seed", 7))
    scn = bpy.context.scene
    if C.COL_RIG in bpy.data.collections:
        for name in [o.name for o in bpy.data.collections[C.COL_RIG].objects]:
            bpy.data.objects[name].hide_viewport = True   # cameras / sun: never hit anyway
    dg = bpy.context.evaluated_depsgraph_get()
    out = {"method": "pressurised rooms: strict from centre (all hit, >=99% front-facing); unpressurised bays: from best free point (max clearance), all hit and (>=99% front-facing or all back-face hits on hull skin); gear wells: closed (all hit from centre and free point); exterior: rays from enclosing sphere aimed at random points in ship bbox, pass >=99.5% front-facing", "seed": seed, "rooms": {}}
    for rid, name, x0, x1, y0, y1, z0, z1, _purpose in load_interior_table():
        key = rid if rid != "B04" else f"B04_{name[-1]}"
        rng = random.Random(f"{seed}-{key}")
        c = Vector(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
        r = room_test(scn, dg, c, n, rng)
        r.update({"id": rid, "name": name, "centre": [round(v, 3) for v in c]})
        if rid in PRESSURISED:
            r["class"] = "pressurised"
            r["criterion"] = "strict: centre, 100% hits and >=99% front-facing"
            r["pass"] = r["all_hit"] and r["front_fraction"] >= 0.99
            if r["backface_objects"]:
                r["backface_account"] = backface_account(scn, dg, c, n, random.Random(f"{seed}-{key}"))
        else:
            cl, fp = best_free_point(scn, dg, (x0, x1, y0, y1, z0, z1))
            frng = f"{seed}-{key}-free"
            fr_ = room_test(scn, dg, fp, n, random.Random(frng))
            fr_["origin"] = [round(v, 3) for v in fp]
            fr_["clearance_m"] = round(cl, 3)
            fr_["backface_account"] = backface_account(scn, dg, fp, n, random.Random(frng))
            r["free_point"] = fr_
            r["centre_pass_strict"] = r.pop("pass")
            ba = fr_["backface_account"]
            if rid in GEAR_WELLS:
                r["class"] = "gear well (closed)"
                r["criterion"] = "closed well: 100% hits from centre and from free point (no ray escapes)"
                r["pass"] = r["all_hit"] and fr_["all_hit"]
                r["pass_basis"] = "closed (0 escapes)" if r["pass"] else "ray escaped"
            else:
                r["class"] = "unpressurised bay"
                r["criterion"] = ("free point: 100% hits and (>=99% front-facing or all back-face hits "
                                  "on hull skin)")
                ok_ff = fr_["front_fraction"] >= 0.99
                ok_skin = ba["all_backfaces_on_hull_skin"]
                r["pass"] = fr_["all_hit"] and (ok_ff or ok_skin)
                r["pass_basis"] = ("front-facing >= 99%" if ok_ff else
                                   "back faces are hull-skin interiors only" if ok_skin else
                                   "non-skin back faces present")
        out["rooms"][key] = r
        print("KESTREL_ENCL", key, r["class"], r["hits"], r["misses"], r["front_fraction"], r["pass"],
              ("free " + str(r["free_point"]["front_fraction"]) + " " + r["pass_basis"]) if "free_point" in r else "")
    # exterior
    objs = [o for o in bpy.data.collections[C.ROOT].all_objects if o.type == 'MESH']
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    for o in objs:
        for corner in o.bound_box:
            w = o.matrix_world @ Vector(corner)
            lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
    ctr = (lo + hi) / 2
    R = (hi - lo).length / 2 * 1.05
    rng = random.Random(f"{seed}-ext")
    hits = front = 0
    back_exit = back_sealed = 0
    back_objs = {}
    for _ in range(ne):
        p = ctr + rand_dir(rng) * R
        tgt = Vector((rng.uniform(lo.x, hi.x), rng.uniform(lo.y, hi.y), rng.uniform(lo.z, hi.z)))
        d = (tgt - p).normalized()
        h = cast(scn, dg, p, d)
        if h is None:
            continue
        hits += 1
        if h["front"]:
            front += 1
        else:
            back_objs[h["obj"]] = back_objs.get(h["obj"], 0) + 1
    ff = front / hits if hits else 0
    out["exterior"] = {"rays": ne, "hits": hits, "misses": ne - hits, "front_hits": front,
                       "front_fraction": round(ff, 5), "pass": ff >= 0.995,
                       "backface_then_open_space": back_exit, "backface_then_more_ship": back_sealed,
                       "sphere_centre": [round(v, 3) for v in ctr], "sphere_radius": round(R, 3),
            "backface_objects": dict(sorted(back_objs.items(), key=lambda kv: -kv[1])[:10])}
    print("KESTREL_ENCL exterior", hits, front, ff)
    with open(a["out"], "w") as f:
        json.dump(out, f, indent=1)


main()
