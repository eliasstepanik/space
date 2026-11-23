"""IC-4 "Kestrel" — 52 re-import verification (does NOT modify kestrel.blend).

    blender -b --factory-startup ../kestrel.blend --python 52_reimport_check.py -- --mode source --out src.json
    blender -b --factory-startup --python 52_reimport_check.py -- --mode import --glb ../../kestrel.glb --out imp.json

Dumps, per object, type / parent / triangle count / world location and the world
bbox of the whole ship (Blender Z-up frame), either from the source file (objects
under the `Kestrel` collection only) or from a fresh factory scene into which the GLB
was imported.  The comparison is done by the caller (build report).
"""
import json
import os
import sys

sys.dont_write_bytecode = True

import bpy
from mathutils import Vector


def args():
    a = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    return {a[i][2:]: a[i + 1] for i in range(0, len(a) - 1, 2) if a[i].startswith("--")}


def dump(objs):
    lo = Vector((1e9,) * 3); hi = Vector((-1e9,) * 3)
    per = {}
    tris = 0
    for o in objs:
        d = {"type": o.type, "parent": o.parent.name if o.parent else None,
             "loc": [round(v, 4) for v in o.matrix_world.translation]}
        if o.type == 'MESH':
            t = sum(len(p.vertices) - 2 for p in o.data.polygons)
            d["tris"] = t
            d["mesh"] = o.data.name
            tris += t
            mw = o.matrix_world
            for v in o.data.vertices:
                w = mw @ v.co
                lo = Vector(map(min, lo, w)); hi = Vector(map(max, hi, w))
        per[o.name] = d
    return {"objects": len(objs), "mesh_objects": sum(1 for o in objs if o.type == 'MESH'),
            "meshes": len({o.data.name for o in objs if o.type == 'MESH'}),
            "tris": tris, "bbox_min": [round(v, 4) for v in lo], "bbox_max": [round(v, 4) for v in hi],
            "per_object": per, "materials": sorted(m.name for m in bpy.data.materials if m.users),
            "images": sorted(i.name for i in bpy.data.images)}


a = args()
if a.get("mode") == "import":
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(a["glb"]))
    objs = list(bpy.context.scene.objects)
else:
    objs = list(bpy.data.collections["Kestrel"].all_objects)
rep = dump(objs)
with open(a["out"], "w") as f:
    json.dump(rep, f)
print("KESTREL_REIMPORT", json.dumps({k: v for k, v in rep.items() if k != "per_object"}))
