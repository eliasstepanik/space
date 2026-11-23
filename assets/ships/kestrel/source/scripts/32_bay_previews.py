"""IC-4 "Kestrel" — technical bay previews (does NOT modify kestrel.blend).

    blender -b --factory-startup --python 32_bay_previews.py -- [--blend path] [--samples 128]
        [--only 17,18] [--outdir dir] [--pct 100] [--engine CYCLES|BLENDER_WORKBENCH]

Writes previews/17..23_bay_*.jpg (Cycles, 1280x720).  Each shot is either an INSIDE view
(bay enclosed, everything rendered) or a CUTAWAY: only the objects matching the shot's
`show` prefixes are rendered (the bay + the exterior parts it connects to), everything
else is hidden for that render.  Temporary lights: a point light at the bay's
Light_<ID> socket, a fill near the camera and (cutaways) the Preview_Rig sun.  Film is
transparent; alpha coverage is reported and the image is composited over a flat
backdrop before saving.
"""
import importlib
import json
import os
import sys

sys.dont_write_bytecode = True

import bpy
import bmesh
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")

# key: (file, bay, cam loc, target, lens, show prefixes (None = inside view, render all), light W)
SHOTS = {
    "17": ("17_bay_B01.jpg", "B01", (1.85, 11.95, 1.55), (0.0, 10.0, -0.35), 24,
           ("INT_B01_", "EXT_SENSORS_RadarFairing"), 25),
    "18": ("18_bay_B02.jpg", "B02", (1.95, 8.65, -1.95), (0.0, 7.15, -1.0), 18,
           ("INT_B02_", "EXT_GEAR_Nose"), 25),
    "19": ("19_bay_B03.jpg", "B03", (0.62, 2.52, -0.72), (-0.15, -1.2, -1.45), 12, None, 30),
    "20": ("20_bay_B04.jpg", "B04", (3.65, -2.0, -1.75), (1.9, -3.35, -0.85), 18,
           ("INT_B04_", "EXT_GEAR_Main"), 25),
    "21": ("21_bay_B05.jpg", "B05", (2.55, -4.15, 1.75), (0.0, -6.05, -0.55), 22,
           ("INT_B05_", "INT_B06_ShieldPlate", "INT_B06_Conduits", "INT_C01_Floor"), 60),
    "22": ("22_bay_B06.jpg", "B06", (2.25, -9.55, 1.35), (0.0, -7.95, -0.55), 22,
           ("INT_B06_", "INT_C01_Floor"), 60),
    "23": ("23_bay_B07.jpg", "B07", (2.45, -8.35, 1.35), (0.0, -10.15, -0.45), 20,
           ("INT_B07_", "EXT_ENGINES_Bell", "EXT_ENGINES_Gimbal", "EXT_ENGINES_Actuator",
            "EXT_ENGINES_FeedLines", "EXT_ENGINES_ThrustStructure", "INT_C01_Floor"), 60),
}
# preview-only section cuts (Boolean DIFFERENCE, removed after the render; nothing is saved):
# key: [(object, box min, box max)] - the near side wall of each gear well is cut away.
CUTTERS = {
    "18": [("INT_B02_WellStructure", (0.35, 6.15, -1.85), (0.65, 8.35, -0.47))],
    "20": [("INT_B04_WellStructure_R", (2.14, -4.55, -1.60), (2.45, -2.05, -0.27))],
}
BG_TOP = np.array([0.46, 0.49, 0.54])
BG_BOT = np.array([0.24, 0.25, 0.28])


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


def camera(key):
    fname, rid, loc, target, lens, _s, _w = SHOTS[key]
    cd = bpy.data.cameras.new("TMP_" + key)
    cd.lens = lens
    cd.clip_start = 0.02
    cd.clip_end = 200
    o = bpy.data.objects.new("TMP_CAM_" + key, cd)
    o.location = loc
    o.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
    bpy.context.scene.collection.objects.link(o)
    return o


def render(key, outdir, samples, pct, engine):
    scn = bpy.context.scene
    fname, rid, loc, target, lens, show, watts = SHOTS[key]
    hidden = []
    sun = bpy.data.objects.get("LGT_Preview_Sun")
    if show is not None:
        for o in scn.objects:
            port_well = rid == "B04" and o.name.endswith("_L")     # B04 shot shows the starboard well only
            if o.type in ('MESH', 'CURVE') and (not any(o.name.startswith(p) for p in show) or o.get("cutaway")
                                                or port_well) and not o.hide_render:
                o.hide_render = True
                hidden.append(o)
        if rid == "B07":                                  # catwalk side of B06 would hide nothing; ok
            pass
    if sun is not None:
        sun.hide_render = show is None
    cut_tmp = []
    for oname, bmin, bmax in CUTTERS.get(key, []):
        tgt = bpy.data.objects.get(oname)
        if tgt is None:
            continue
        # temporary copy of the mesh without the faces whose world centre lies in the cut box
        mw = tgt.matrix_world
        bm = bmesh.new()
        bm.from_mesh(tgt.data)
        kill = [f for f in bm.faces
                if all(lo <= c <= hi for lo, c, hi in zip(bmin, mw @ f.calc_center_median(), bmax))]
        bmesh.ops.delete(bm, geom=kill, context='FACES')
        me = bpy.data.meshes.new("TMP_CUT")
        bm.to_mesh(me)
        bm.free()
        for m in tgt.data.materials:
            me.materials.append(m)
        cut_tmp.append((tgt, tgt.data, me))
        tgt.data = me
    cam = camera(key)
    tmp_l = []
    for o in scn.objects:
        if o.name.startswith(f"SOCKET_Light_{rid}"):
            p = o.matrix_world.translation.copy()
            p.z -= 0.15
            tmp_l.append(add_light("TMP_L_" + rid, p, watts))
    if rid == "B04":
        tmp_l.append(add_light("TMP_L_B04R", (1.925, -3.3, -0.45), watts))
    tmp_l.append(add_light("TMP_Fill_" + key, Vector(loc) + Vector((0, 0, -0.15)), 8 if show is None else 25, 0.3))
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
    tmp = os.path.join("/tmp", f"kestrel_bay_{key}.png")
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
        "file": path, "size": list(chk.size), "bay": rid, "mode": "inside" if show is None else "cutaway",
        "section_cut": bool(cut_tmp),
        "alpha_coverage_gt_0.9": round(float((a > 0.9).mean()), 4),
        "mean_rgb": [round(float(v), 3) for v in q.mean(0)],
        "std_luma": round(float(luma.std()), 3),
        "dark_px_lt_0.03": round(float((luma < 0.03).mean()), 3),
        "clipped_px_gt_0.98": round(float((luma > 0.98).mean()), 3),
    }
    print("PREVIEW " + json.dumps(stats), flush=True)
    for o in hidden:
        o.hide_render = False
    for o in tmp_l + [cam]:
        bpy.data.objects.remove(o, do_unlink=True)
    for tgt, orig, me in cut_tmp:
        tgt.data = orig
        bpy.data.meshes.remove(me)
    return stats


if __name__ == "__main__":
    args = C.script_args()
    C.open_blend(args.get("blend", C.BLEND_PATH))
    samples = int(args.get("samples", 128))
    pct = int(args.get("pct", 100))
    engine = args.get("engine", "CYCLES")
    outdir = args.get("outdir", C.PREVIEW_DIR)
    os.makedirs(outdir, exist_ok=True)
    keys = args["only"].split(",") if isinstance(args.get("only"), str) else list(SHOTS)
    bpy.context.scene.view_settings.view_transform = 'AgX'
    for k in keys:
        render(k, outdir, samples, pct, engine)
