"""IC-4 "Kestrel" — interior room previews (does NOT modify kestrel.blend).

    blender -b --factory-startup --python 31_interior_previews.py -- [--blend path] [--samples 96]
        [--only 09,10] [--outdir dir] [--pct 100] [--engine CYCLES|BLENDER_WORKBENCH]

Writes previews/09..16_room_*.jpg (Cycles, 1280x720).  One camera per room, inside
the room (placed from the Light_<ID> socket / room box); R01 is a cutaway (Ceiling
node + canopy hidden, camera above the cockpit opening).  Temporary lights: one
point light per Light_<ID> socket (R03a: chamber centre) and a soft fill near the
camera.  Film is transparent so the alpha coverage measures leaks: an enclosed room
must cover ~100 %; the result is composited over a flat backdrop before saving.
"""
import importlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")

# key: (file, room, cam loc, target, lens mm, hide prefixes)
SHOTS = {
    "09": ("09_room_R01.jpg", "R01", (1.55, 3.55, 3.55), (0.0, 6.0, 0.95), 24,
           ("INT_R01_Ceiling", "INT_R01_WindowReveals", "INT_R01_OverheadBreakers", "EXT_CANOPY_")),
    "10": ("10_room_R02.jpg", "R02", (0.62, 2.72, 1.30), (-0.45, 4.15, 0.20), 13, ()),
    "11": ("11_room_R03.jpg", "R03", (0.02, 2.44, 1.42), (0.80, 0.80, -0.05), 11, ()),
    "12": ("12_room_R03a.jpg", "R03a", (-0.30, 2.42, 1.00), (-1.00, 0.70, 0.45), 11, ()),
    "13": ("13_room_R04.jpg", "R04", (-0.22, 0.18, 1.30), (0.45, -1.30, 0.05), 12, ()),
    "14": ("14_room_R05.jpg", "R05", (0.0, -1.74, 1.05), (0.0, -2.60, 0.55), 10, ()),
    "15": ("15_room_R06.jpg", "R06", (-0.55, -2.72, 1.32), (0.45, -4.55, 0.25), 13, ()),
    "16": ("16_room_C01.jpg", "C01", (-0.12, -5.02, 1.20), (0.05, -9.6, 1.0), 14, ()),
}
BG_TOP = np.array([0.46, 0.49, 0.54])
BG_BOT = np.array([0.24, 0.25, 0.28])
LIGHT_W = {"R01": 30, "R02": 60, "R03": 45, "R04": 70, "R05": 40, "R06": 70, "C01": 35}


def add_light(name, loc, watts, radius=0.08, kind='POINT', color=(1.0, 0.93, 0.84)):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = watts
    ld.color = color
    if kind == 'POINT':
        ld.shadow_soft_size = radius
    o = bpy.data.objects.new(name, ld)
    o.location = loc
    bpy.context.scene.collection.objects.link(o)
    return o


def lights():
    out = []
    for o in bpy.data.collections[C.COL_SOCK].all_objects:
        if o.name.startswith("SOCKET_Light_"):
            rid = o.name.split("_")[-1]
            if rid in LIGHT_W:
                p = o.matrix_world.translation.copy()
                p.z -= 0.08
                out.append(add_light("TMP_L_" + rid, p, LIGHT_W[rid]))
    out.append(add_light("TMP_L_R03a", (-0.62, 1.55, 1.40), 40))
    for y in (-5.7, -8.7):                                   # C01 ceiling strip lights
        out.append(add_light(f"TMP_L_C01_{y}", (0.0, y, 1.47), 20))
    return out


def camera(key):
    fname, rid, loc, target, lens, _h = SHOTS[key]
    cd = bpy.data.cameras.new("TMP_" + key)
    cd.lens = lens
    cd.clip_start = 0.02
    cd.clip_end = 200
    o = bpy.data.objects.new("TMP_CAM_" + key, cd)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.collection.objects.link(o)
    return o


def set_hidden(prefixes, state):
    for o in bpy.context.scene.objects:
        if any(o.name.startswith(p) for p in prefixes):
            o.hide_render = state


def render(key, outdir, samples, pct, engine):
    scn = bpy.context.scene
    fname, rid, loc, target, lens, hide = SHOTS[key]
    set_hidden(hide, True)
    cam = camera(key)
    fill = add_light("TMP_Fill_" + key, Vector(loc) + Vector((0, 0, -0.15)), 6 if rid != "R01" else 0, 0.3)
    scn.camera = cam
    scn.render.engine = engine
    if engine == 'CYCLES':
        scn.cycles.samples = samples
        scn.cycles.use_denoising = True
        scn.cycles.max_bounces = 8
    scn.render.film_transparent = True
    scn.render.resolution_x, scn.render.resolution_y = 1280, 720
    scn.render.resolution_percentage = pct
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'
    tmp = os.path.join("/tmp", f"kestrel_int_{key}.png")
    scn.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(tmp)
    img.reload()
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    a = px[..., 3:4]
    t = np.linspace(0, 1, h).reshape(h, 1, 1)
    bg = BG_BOT * (1 - t) + BG_TOP * t
    rgb = px[..., :3] * a + bg * (1 - a)
    out = np.concatenate([rgb, np.ones_like(a)], axis=2)
    res = bpy.data.images.new("TMP_out_" + key, w, h, alpha=False)
    res.pixels.foreach_set(out.astype(np.float32).ravel())
    path = os.path.join(outdir, fname)
    res.filepath_raw = path
    res.file_format = 'JPEG'
    res.save(filepath=path, quality=92)
    chk = bpy.data.images.load(path)
    chk.reload()
    q = np.empty(chk.size[0] * chk.size[1] * 4, dtype=np.float32)
    chk.pixels.foreach_get(q)
    q = q.reshape(-1, 4)[:, :3]
    luma = q.mean(1)
    stats = {
        "file": path, "size": list(chk.size), "room": rid,
        "alpha_coverage_gt_0.9": round(float((a > 0.9).mean()), 4),
        "mean_rgb": [round(float(v), 3) for v in q.mean(0)],
        "std_luma": round(float(luma.std()), 3),
        "dark_px_lt_0.03": round(float((luma < 0.03).mean()), 3),
        "clipped_px_gt_0.98": round(float((luma > 0.98).mean()), 3),
    }
    print("PREVIEW " + json.dumps(stats), flush=True)
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.objects.remove(fill, do_unlink=True)
    set_hidden(hide, False)
    return stats


if __name__ == "__main__":
    args = C.script_args()
    C.open_blend(args.get("blend", C.BLEND_PATH))
    samples = int(args.get("samples", 96))
    pct = int(args.get("pct", 100))
    engine = args.get("engine", "CYCLES")
    outdir = args.get("outdir", C.PREVIEW_DIR)
    os.makedirs(outdir, exist_ok=True)
    keys = args["only"].split(",") if isinstance(args.get("only"), str) else list(SHOTS)
    scn = bpy.context.scene
    scn.view_settings.view_transform = 'AgX'
    lights()
    for k in keys:
        render(k, outdir, samples, pct, engine)
