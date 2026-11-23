"""IC-4 "Kestrel" — exterior detail previews (does NOT modify kestrel.blend).

    blender -b --factory-startup --python 21_exterior_previews.py -- [--blend path] [--samples 64]
        [--only 03,05] [--outdir dir] [--pct 100]

Writes previews/03..08_exterior_*.jpg with Cycles at 1280x720.  Cameras, the
fill light and the shadow-catcher ground are created in memory only.  The ship
is rendered over a transparent film, its alpha coverage is measured, and the
frame is composited over a flat studio backdrop before saving the JPEG.
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

SHOTS = {
    # key: (file, cam loc, target, lens or ortho, ortho?, ground?)
    "03": ("03_exterior_starboard_fg.jpg", (17.0, 21.0, 6.5), (0.0, 0.3, -0.6), 38, False, True),
    "04": ("04_exterior_port_aft.jpg", (-17.0, -21.0, 7.5), (0.0, -2.0, -0.6), 38, False, True),
    "05": ("05_exterior_top.jpg", (0.0, 0.0, 40.0), None, 25.5, True, False),
    "06": ("06_exterior_bottom.jpg", (0.0, 0.0, -40.0), None, 25.5, True, False),
    "07": ("07_exterior_nose_close.jpg", (4.6, 15.6, 2.4), (0.0, 9.9, -0.2), 45, False, False),
    "08": ("08_exterior_engine_close.jpg", (-5.2, -17.6, 2.2), (-0.6, -9.6, -0.6), 38, False, False),
}
BG_TOP = np.array([0.46, 0.49, 0.54])   # display-space backdrop
BG_BOT = np.array([0.24, 0.25, 0.28])


def camera(key):
    fname, loc, target, lens, ortho, _g = SHOTS[key]
    cd = bpy.data.cameras.new("TMP_" + key)
    cd.clip_end = 500
    if ortho:
        cd.type = 'ORTHO'
        cd.ortho_scale = lens
    else:
        cd.lens = lens
    o = bpy.data.objects.new("TMP_CAM_" + key, cd)
    o.location = loc
    if key == "05":
        o.rotation_euler = (0.0, 0.0, math.pi / 2)          # nose to image right
    elif key == "06":
        o.rotation_euler = (math.pi, 0.0, math.pi / 2)      # looking up, nose to image right
    else:
        d = Vector(target) - Vector(loc)
        o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.collection.objects.link(o)
    return o


def ground():
    me = bpy.data.meshes.new("TMP_Ground")
    s = 80
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new("TMP_Ground", me)
    o.location.z = C.Z_GROUND
    o.is_shadow_catcher = True
    bpy.context.scene.collection.objects.link(o)
    return o


def lights():
    # under-fill so the belly and the shadow side read; key sun comes from the saved rig
    ld = bpy.data.lights.new("TMP_Fill", 'SUN')
    ld.energy = 1.2
    ld.angle = math.radians(10)
    o = bpy.data.objects.new("TMP_Fill", ld)
    o.rotation_euler = (math.radians(150), math.radians(-20), math.radians(-140))
    bpy.context.scene.collection.objects.link(o)
    rim_d = bpy.data.lights.new("TMP_Rim", 'SUN')
    rim_d.energy = 1.5
    rim = bpy.data.objects.new("TMP_Rim", rim_d)
    rim.rotation_euler = (math.radians(70), 0.0, math.radians(200))
    bpy.context.scene.collection.objects.link(rim)


def render(key, outdir, samples, pct):
    scn = bpy.context.scene
    fname, *_rest, use_ground = SHOTS[key]
    g = ground() if use_ground else None
    cam = camera(key)
    scn.camera = cam
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    scn.render.film_transparent = True
    scn.render.resolution_x, scn.render.resolution_y = 1280, 720
    scn.render.resolution_percentage = pct
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'
    tmp = os.path.join("/tmp", f"kestrel_prev_{key}.png")
    scn.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(tmp)
    img.reload()
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    a = px[..., 3:4]
    t = np.linspace(0, 1, h).reshape(h, 1, 1)           # row 0 = bottom
    bg = BG_BOT * (1 - t) + BG_TOP * t
    rgb = px[..., :3] * a + bg * (1 - a)                 # straight alpha, display space
    out = np.concatenate([rgb, np.ones_like(a)], axis=2)
    res = bpy.data.images.new("TMP_out_" + key, w, h, alpha=False)
    res.pixels.foreach_set(out.astype(np.float32).ravel())
    path = os.path.join(outdir, fname)
    res.filepath_raw = path
    res.file_format = 'JPEG'
    res.save(filepath=path, quality=92)
    # verify the written JPEG
    chk = bpy.data.images.load(path)
    chk.reload()
    q = np.empty(chk.size[0] * chk.size[1] * 4, dtype=np.float32)
    chk.pixels.foreach_get(q)
    q = q.reshape(-1, 4)[:, :3]
    stats = {
        "file": path, "size": list(chk.size),
        "ship_coverage_alpha_gt_0.9": round(float((a > 0.9).mean()), 4),
        "any_alpha": round(float((a > 0.02).mean()), 4),
        "mean_rgb": [round(float(v), 3) for v in q.mean(0)],
        "std_luma": round(float(q.mean(1).std()), 3),
    }
    print("PREVIEW " + json.dumps(stats), flush=True)
    for o in (cam, g):
        if o is not None:
            bpy.data.objects.remove(o, do_unlink=True)
    return stats


if __name__ == "__main__":
    args = C.script_args()
    C.open_blend(args.get("blend", C.BLEND_PATH))
    samples = int(args.get("samples", 64))
    pct = int(args.get("pct", 100))
    outdir = args.get("outdir", C.PREVIEW_DIR)
    os.makedirs(outdir, exist_ok=True)
    keys = args["only"].split(",") if isinstance(args.get("only"), str) else list(SHOTS)
    scn = bpy.context.scene
    scn.view_settings.view_transform = 'AgX'
    lights()
    # exterior previews: hide the remaining BLOCKOUT volumes (bays B01-B07, still boxes until task 5);
    # fitted rooms (INT_R*/INT_C*) stay visible so the cockpit reads through the canopy.
    for o in list(bpy.data.objects):
        if o is None or not o.name.startswith("INT_"):
            continue
        if str(o.get("room_id", "")).startswith("B") or o.name.startswith("INT_B"):
            o.hide_render = True
    for k in keys:
        render(k, outdir, samples, pct)
