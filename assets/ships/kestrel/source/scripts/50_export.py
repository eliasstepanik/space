"""IC-4 "Kestrel" — 50 final GLB exports (does NOT modify kestrel.blend).

    blender -b --factory-startup ../kestrel.blend --python 50_export.py -- --what full|lod1
        [--out path.glb] [--report report.json] [--target 12000]

full : exports the `Kestrel` hierarchy (root empty + Kestrel_Exterior / _Interior /
       _Sockets) as one GLB.  Preview_Rig (cameras, sun) is excluded by exporting the
       selection only.  Object transforms are NOT applied: the glTF exporter writes
       node TRS (sockets keep their rotations, linked duplicates stay instanced).
lod1 : proxy.  Every exterior mesh is copied (single-user data, world transform baked
       into the copy) into a temporary collection of the unsaved file, joined into ONE mesh, merged by
       distance and decimated (Collapse) to --target triangles (accepted 8k-15k).
       The 16 materials are merged into one material M_Kestrel_LOD1: per-face base
       colour of the source material is written to a corner colour attribute (glTF
       COLOR_0) and the hull ORM + normal textures are reused (UVMap is preserved), so
       the proxy is ONE mesh with ONE primitive and embedded textures.
The source file is never saved.
"""
import importlib
import json
import os
import sys

sys.dont_write_bytecode = True

import bpy
import bmesh
from mathutils import Matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")

GLTF_KW = dict(export_format='GLB', export_yup=True, export_apply=True,
               export_image_format='AUTO', export_materials='EXPORT', export_texcoords=True,
               export_normals=True, export_draco_mesh_compression_enable=False,
               export_meshopt_compression_enable=False, export_use_gltfpack=False,
               export_cameras=False, export_lights=False, use_selection=True,
               export_extras=True, export_animations=False)


def tri_count(me):
    return sum(len(p.vertices) - 2 for p in me.polygons)


def select_only(objs):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]


def export_full(out):
    root = bpy.data.collections[C.ROOT]
    objs = list(root.all_objects)
    rig = set(bpy.data.collections[C.COL_RIG].all_objects) if C.COL_RIG in bpy.data.collections else set()
    assert not (set(objs) & rig)
    select_only([bpy.data.objects[C.ROOT]] + [o for o in objs if o.name != C.ROOT])
    bpy.ops.export_scene.gltf(filepath=out, **GLTF_KW)
    tris = sum(tri_count(o.data) for o in objs if o.type == 'MESH')
    return {"exported_objects": len(objs), "excluded": sorted(o.name for o in rig),
            "source_tris": tris, "nodes": sorted(o.name for o in objs)}


def lod1_material():
    mat = bpy.data.materials.new("M_Kestrel_LOD1")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    L = nt.links.new
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    L(bsdf.outputs["BSDF"], out.inputs["Surface"])
    col = nt.nodes.new("ShaderNodeVertexColor")
    col.layer_name = "Col"
    L(col.outputs["Color"], bsdf.inputs["Base Color"])
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = "UVMap"
    to = nt.nodes.new("ShaderNodeTexImage")
    to.image = bpy.data.images["T_Hull_ORM"]
    L(uv.outputs["UV"], to.inputs["Vector"])
    sep = nt.nodes.new("ShaderNodeSeparateColor")
    L(to.outputs["Color"], sep.inputs["Color"])
    mr = nt.nodes.new("ShaderNodeMath"); mr.operation = "MULTIPLY"
    L(sep.outputs["Green"], mr.inputs[0]); mr.inputs[1].default_value = 0.45
    L(mr.outputs["Value"], bsdf.inputs["Roughness"])
    mm = nt.nodes.new("ShaderNodeMath"); mm.operation = "MULTIPLY"
    L(sep.outputs["Blue"], mm.inputs[0]); mm.inputs[1].default_value = 0.2
    L(mm.outputs["Value"], bsdf.inputs["Metallic"])
    tn = nt.nodes.new("ShaderNodeTexImage")
    tn.image = bpy.data.images["T_Hull_Normal"]
    L(uv.outputs["UV"], tn.inputs["Vector"])
    nm = nt.nodes.new("ShaderNodeNormalMap"); nm.uv_map = "UVMap"
    L(tn.outputs["Color"], nm.inputs["Color"])
    L(nm.outputs["Normal"], bsdf.inputs["Normal"])
    g = bpy.data.node_groups.get("glTF Material Output")
    if g is not None:
        gn = nt.nodes.new("ShaderNodeGroup"); gn.node_tree = g
        L(sep.outputs["Red"], gn.inputs["Occlusion"])
    return mat


def base_color(mat):
    if mat is None:
        return (0.5, 0.5, 0.5, 1.0)
    p = next((n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    c = list(p.inputs["Base Color"].default_value) if p else list(mat.diffuse_color)
    if mat.name == "M_Decals":
        c = [0.66, 0.67, 0.68, 1.0]          # decals sit on white paint
    if mat.name in ("M_Screens_Emissive", "M_Lights_Emissive"):
        c = [0.9, 0.9, 0.85, 1.0]
    return (c[0], c[1], c[2], 1.0)


def export_lod1(out, target):
    src = [o for o in bpy.data.collections[C.COL_EXT].all_objects if o.type == 'MESH']
    scn = bpy.context.scene            # copies live in the open (never saved) file
    tmp = bpy.data.collections.new("LOD1_Tmp")
    scn.collection.children.link(tmp)
    copies = []
    for o in src:
        me = o.data.copy()
        me.transform(o.matrix_world)
        if o.matrix_world.determinant() < 0:
            me.flip_normals()
        # per-face colour of the source material, stored on corners
        ca = me.color_attributes.get("Col") or me.color_attributes.new("Col", 'FLOAT_COLOR', 'CORNER')
        cols = [base_color(me.materials[i] if i < len(me.materials) else None) for i in range(max(1, len(me.materials)))]
        data = ca.data
        for p in me.polygons:
            c = cols[min(p.material_index, len(cols) - 1)]
            for li in p.loop_indices:
                data[li].color = c
        ob = bpy.data.objects.new("LOD1_" + o.name, me)
        tmp.objects.link(ob)
        copies.append(ob)
    src_tris = sum(tri_count(o.data) for o in copies)
    bpy.context.view_layer.update()
    vl = bpy.context.view_layer
    for o in vl.objects:
        o.select_set(o in copies)
    vl.objects.active = copies[0]
    with bpy.context.temp_override(scene=scn, view_layer=vl, active_object=copies[0],
                                   selected_editable_objects=copies, selected_objects=copies):
        bpy.ops.object.join()
    ob = copies[0]
    ob.name = "Kestrel_LOD1"
    ob.data.name = "Kestrel_LOD1"
    me = ob.data
    joined_tris = tri_count(me)
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)
    bm.to_mesh(me); bm.free()
    welded_tris = tri_count(me)
    me.materials.clear()
    me.materials.append(lod1_material())
    for p in me.polygons:
        p.material_index = 0
    ratio = target / welded_tris
    tries = []
    dg = bpy.context.evaluated_depsgraph_get()
    md = ob.modifiers.new("Decimate", 'DECIMATE')
    md.decimate_type = 'COLLAPSE'
    md.use_collapse_triangulate = True
    best = None
    for it in range(10):
        md.ratio = ratio
        dg.update()
        ev = ob.evaluated_get(dg)
        cand = bpy.data.meshes.new_from_object(ev)
        raw_t = tri_count(cand)
        # drop zero-area faces / zero-length edges the collapse leaves behind (the glTF
        # exporter would silently skip them, so Blender and GLB counts would differ)
        bm = bmesh.new(); bm.from_mesh(cand)
        bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-5)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bad = [f for f in bm.faces if f.calc_area() < 1e-9]
        if bad:
            bmesh.ops.delete(bm, geom=bad, context='FACES')
        bm.to_mesh(cand); bm.free()
        cand.validate()          # removes the coincident duplicate faces the collapse leaves
        t = tri_count(cand)
        tries.append({"ratio": round(ratio, 5), "decimated_tris": raw_t, "clean_tris": t})
        if best is not None:
            bpy.data.meshes.remove(best)
        best = cand
        if abs(t - target) <= 0.08 * target:
            break
        ratio *= target / max(t, 1)
    ob.modifiers.remove(md)
    old = ob.data
    ob.data = best
    bpy.data.meshes.remove(old)
    best.name = "Kestrel_LOD1"
    final_tris = tri_count(ob.data)
    for p in ob.data.polygons:
        p.use_smooth = False
    # glTF: root node name
    vl.objects.active = ob
    for o in vl.objects:
        o.select_set(o == ob)
    kw = dict(GLTF_KW)
    bpy.ops.export_scene.gltf(filepath=out, **kw)
    return {"source_exterior_objects": len(src), "source_exterior_tris": src_tris,
            "joined_tris": joined_tris, "welded_tris": welded_tris, "decimate_tries": tries,
            "final_tris": final_tris, "material": "M_Kestrel_LOD1",
            "material_note": "16 source materials merged: base colour per face -> COLOR_0, "
                             "T_Hull_ORM (rough x0.45, metal x0.2, occlusion) + T_Hull_Normal"}


def main():
    a = C.script_args()
    what = a.get("what", "full")
    out = a.get("out") or os.path.join(C.SHIP_DIR, "kestrel.glb" if what == "full" else "kestrel_lod1.glb")
    rep = export_full(out) if what == "full" else export_lod1(out, int(a.get("target", 12000)))
    rep["out"] = out
    rep["bytes"] = os.path.getsize(out)
    if a.get("report"):
        with open(a["report"], "w") as f:
            json.dump(rep, f, indent=1)
    print("KESTREL_EXPORT", json.dumps({k: v for k, v in rep.items() if k != "nodes"}))


main()
