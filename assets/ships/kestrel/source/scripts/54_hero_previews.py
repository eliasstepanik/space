"""IC-4 "Kestrel" — 54 final hero renders (does NOT modify kestrel.blend).

    blender -b --factory-startup --python 54_hero_previews.py -- [--blend path] [--samples 256]
        [--only 24,25] [--outdir dir] [--pct 100]

Writes previews/24_hero_port_bow.jpg and previews/25_hero_full_deployed.jpg (Cycles,
1920x1080, AgX).  Camera, lights and the shadow-catcher pad are created in memory
only.  The ship renders over a transparent film and is composited over a dark studio
gradient; a 1-sample pass with the pad hidden gives the clean ship alpha used for the
coverage figure (fraction of pixels with alpha > 0.5).  The saved JPEG is reloaded and
its pixels are read back with foreach_get.
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

# key: (file, cam loc, target, lens, key-light (elev deg, azimuth deg))
SHOTS = {
    "24": ("24_hero_port_bow.jpg", (-15.5, 21.0, 1.0), (-0.2, 0.9, -0.7), 38, (30, 125)),
    "25": ("25_hero_full_deployed.jpg", (19.0, -17.0, 13.0), (0.0, -1.0, -0.9), 39, (40, 150)),
}
BG_TOP = np.array([0.20, 0.215, 0.24])     # display space
BG_BOT = np.array([0.075, 0.08, 0.09])


def add(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def sun(name, energy, elev, azim, angle=3.0, color=(1, 1, 1)):
    ld = bpy.data.lights.new(name, 'SUN')
    ld.energy = energy
    ld.angle = math.radians(angle)
    ld.color = color
    o = bpy.data.objects.new(name, ld)
    # light travels along local -Z; point it from (elev, azim) towards the origin
    d = -Vector((math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
                 math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
                 math.sin(math.radians(elev))))
    o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
    return add(o)


def setup(key):
    scn = bpy.context.scene
    fname, loc, target, lens, (kel, kaz) = SHOTS[key]
    tmp = []
    cd = bpy.data.cameras.new("TMP_HERO")
    cd.lens = lens
    cd.clip_end = 500
    cam = add(bpy.data.objects.new("TMP_HERO", cd))
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    scn.camera = cam
    tmp.append(cam)
    tmp.append(sun("TMP_Key", 3.3, kel, kaz, 2.0, (1.0, 0.95, 0.88)))
    tmp.append(sun("TMP_Rim", 3.0, 18, kaz + 165, 4.0, (0.70, 0.82, 1.0)))
    tmp.append(sun("TMP_Fill", 0.6, -25, kaz - 70, 15.0, (0.80, 0.86, 1.0)))
    me = bpy.data.meshes.new("TMP_Pad")
    s = 120
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    pad = add(bpy.data.objects.new("TMP_Pad", me))
    pad.location.z = C.Z_GROUND
    pad.is_shadow_catcher = True
    tmp.append(pad)
    w = bpy.data.worlds.new("TMP_World")
    w.use_nodes = True
    bgn = w.node_tree.nodes["Background"]
    bgn.inputs["Color"].default_value = (0.045, 0.05, 0.062, 1.0)
    bgn.inputs["Strength"].default_value = 1.0
    scn.world = w
    return tmp, pad


def render_one(key, outdir, samples, pct):
    scn = bpy.context.scene
    fname = SHOTS[key][0]
    tmp, pad = setup(key)
    scn.render.engine = 'CYCLES'
    scn.cycles.use_denoising = True
    scn.render.film_transparent = True
    scn.render.resolution_x, scn.render.resolution_y = 1920, 1080
    scn.render.resolution_percentage = pct
    scn.render.image_settings.file_format = 'PNG'
    scn.render.image_settings.color_mode = 'RGBA'
    scn.view_settings.view_transform = 'AgX'
    scn.view_settings.look = 'AgX - Medium High Contrast'
    # pass 1: clean ship alpha (pad hidden, 1 sample)
    pad.hide_render = True
    scn.cycles.samples = 1
    scn.cycles.use_denoising = False
    mpath = f"/tmp/kestrel_hero_mask_{key}.png"
    scn.render.filepath = mpath
    bpy.ops.render.render(write_still=True)
    m = bpy.data.images.load(mpath); m.reload()
    mw, mh = m.size
    mp = np.empty(mw * mh * 4, dtype=np.float32); m.pixels.foreach_get(mp)
    coverage = float((mp.reshape(-1, 4)[:, 3] > 0.5).mean())
    # pass 2: beauty with shadow catcher
    pad.hide_render = False
    scn.cycles.samples = samples
    scn.cycles.use_denoising = True
    tpath = f"/tmp/kestrel_hero_{key}.png"
    scn.render.filepath = tpath
    import time
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    secs = time.time() - t0
    img = bpy.data.images.load(tpath); img.reload()
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32); img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    a = px[..., 3:4]
    t = np.linspace(0, 1, h).reshape(h, 1, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    vign = 1.0 - 0.35 * (((xx / w - 0.5) ** 2 + (yy / h - 0.45) ** 2) * 2.2).reshape(h, w, 1)
    bg = (BG_BOT * (1 - t) + BG_TOP * t) * np.clip(vign, 0.55, 1.0)
    rgb = px[..., :3] * a + bg * (1 - a)
    out = np.concatenate([np.clip(rgb, 0, 1), np.ones_like(a)], axis=2).astype(np.float32)
    res = bpy.data.images.new("TMP_out_" + key, w, h, alpha=False)
    res.pixels.foreach_set(out.ravel())
    path = os.path.join(outdir, fname)
    res.filepath_raw = path
    res.file_format = 'JPEG'
    res.save(filepath=path, quality=93)
    chk = bpy.data.images.load(path); chk.reload()
    q = np.empty(chk.size[0] * chk.size[1] * 4, dtype=np.float32)
    chk.pixels.foreach_get(q)
    q = q.reshape(chk.size[1], chk.size[0], 4)[..., :3]
    # non-background pixels in the saved JPEG: differ from the backdrop by > 0.03
    diff = np.abs(q - bg[::1, :, :]).max(axis=2)
    stats = {"file": path, "size": list(chk.size), "samples": samples, "render_s": round(secs, 1),
             "ship_coverage_alpha_gt_0.5": round(coverage, 4),
             "jpeg_non_background_fraction": round(float((diff > 0.03).mean()), 4),
             "mean_rgb": [round(float(v), 3) for v in q.reshape(-1, 3).mean(0)],
             "std_luma": round(float(q.mean(2).std()), 3)}
    print("HERO " + json.dumps(stats), flush=True)
    for o in tmp:
        bpy.data.objects.remove(o, do_unlink=True)
    return stats


if __name__ == "__main__":
    args = C.script_args()
    C.open_blend(args.get("blend", C.BLEND_PATH))
    for o in list(bpy.data.collections[C.COL_RIG].objects):
        o.hide_render = True
    samples = int(args.get("samples", 256))
    pct = int(args.get("pct", 100))
    outdir = args.get("outdir", C.PREVIEW_DIR)
    os.makedirs(outdir, exist_ok=True)
    keys = args["only"].split(",") if isinstance(args.get("only"), str) else list(SHOTS)
    allst = [render_one(k, outdir, samples, pct) for k in keys]
    if args.get("report"):
        with open(args["report"], "w") as f:
            json.dump(allst, f, indent=1)
