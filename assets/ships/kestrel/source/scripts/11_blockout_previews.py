"""IC-4 "Kestrel" — blockout preview renders (does NOT modify kestrel.blend).

    blender -b --factory-startup --python 11_blockout_previews.py -- [--blend path] [--samples 64] [--only ext|top]

Writes previews/01_blockout_exterior.jpg (CAM_Preview_34) and
previews/02_blockout_top_cutaway.jpg (CAM_Preview_Top; hull bisected at
Z = -0.55 so the lower shell stays, upper hull / canopy / dorsal parts hidden,
main-deck volumes semi-transparent so the ventral bays read through, ID labels).
All changes are made in memory and the file is never saved.
"""
import importlib
import math
import os
import sys

sys.dont_write_bytecode = True  # keep scripts/ free of __pycache__

import bpy
import bmesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")


def render(path, camera, samples):
    scn = bpy.context.scene
    scn.camera = bpy.data.objects[camera]
    scn.cycles.samples = samples
    scn.render.image_settings.file_format = 'JPEG'
    scn.render.image_settings.quality = 92
    scn.render.filepath = path
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(path)
    img.reload()
    px = [0.0] * (img.size[0] * img.size[1] * 4)
    img.pixels.foreach_get(px)
    mean = sum(px[0::4]) / (len(px) / 4)
    print(f"PREVIEW {path} {img.size[0]}x{img.size[1]} mean_r={mean:.3f}")


def ground():
    me = bpy.data.meshes.new("TMP_Ground")
    s = 60
    me.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    o = bpy.data.objects.new("TMP_Ground", me)
    o.location.z = C.Z_GROUND
    o.data.materials.append(C.placeholder_material("TMP_Ground", (0.30, 0.30, 0.31), roughness=0.9))
    bpy.context.scene.collection.objects.link(o)
    return o


def cutaway(z_cut=-0.55):
    keep_whole = ("EXT_RADIATORS_", "EXT_GEAR_", "EXT_WEAPONS_")
    bisect = ("EXT_HULL_", "EXT_ENGINES_")
    for o in bpy.data.objects:
        if not o.name.startswith("EXT_") or o.name.startswith(keep_whole):
            continue
        if o.name.startswith(bisect):
            bm = bmesh.new()
            bm.from_mesh(o.data)
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=(0, 0, z_cut - o.location.z),
                                   plane_no=(0, 0, 1), clear_outer=True)
            bm.to_mesh(o.data)
            bm.free()
        else:
            o.hide_render = True
    # translucent main-deck rooms + crawlway so ventral bays read through
    for o in bpy.data.objects:
        if o.name.startswith(("INT_R", "INT_C")):
            src = o.data.materials[0]
            m = src.copy()
            m.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.55
            o.data.materials[0] = m
    # labels (text reads along +Y = image right)
    label_pos = {
        "B01": (0.0, 9.8), "B02": (0.3, 7.3), "R01": (-0.35, 5.4), "R02": (0.0, 3.45),
        "R03": (0.55, 1.55), "R03a": (-0.62, 1.55), "R04": (-0.6, -0.5), "B03": (0.45, 0.1),
        "R05": (0.0, -2.1), "R06": (0.0, -3.65), "B04": None, "C01": (0.0, -7.2),
        "B05": (-0.72, -5.95), "B06": (-0.72, -8.05), "B07": (-0.72, -10.15),
    }
    mtxt = C.placeholder_material("TMP_Label", (0.0, 0.0, 0.0), roughness=1.0)
    for o in bpy.data.objects:
        if not o.name.startswith("INT_"):
            continue
        rid = o["room_id"]
        if rid == "R03" and "Chamber" in o.name:
            rid = "R03a"
        pos = label_pos.get(rid)
        if rid == "B04":
            pos = (o.location.x, o.location.y)
        cu = bpy.data.curves.new(f"TMP_Label_{o.name}", 'FONT')
        cu.body = rid
        cu.size = 0.34
        cu.align_x, cu.align_y = 'CENTER', 'CENTER'
        t = bpy.data.objects.new(cu.name, cu)
        t.location = (pos[0], pos[1], 3.0)
        t.rotation_euler = (0, 0, math.pi / 2)
        t.data.materials.append(mtxt)
        t.visible_shadow = False
        bpy.context.scene.collection.objects.link(t)


if __name__ == "__main__":
    args = C.script_args()
    blend = args.get("blend", C.BLEND_PATH)
    samples = int(args.get("samples", 64))
    only = args.get("only")
    os.makedirs(C.PREVIEW_DIR, exist_ok=True)
    C.open_blend(blend)
    if only in (None, "ext"):
        g = ground()
        render(os.path.join(C.PREVIEW_DIR, "01_blockout_exterior.jpg"), "CAM_Preview_34", samples)
        bpy.data.objects.remove(g)
    if only in (None, "top"):
        cutaway()
        render(os.path.join(C.PREVIEW_DIR, "02_blockout_top_cutaway.jpg"), "CAM_Preview_Top", samples)
