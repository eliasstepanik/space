"""IC-4 "Kestrel" build — shared helpers.

Conventions (all build scripts):
  * Blender Z-up, nose +Y, starboard +X, 1 unit = 1 m.
  * glTF export with export_yup=True maps the nose to glTF -Z.
  * Object names: EXT_<Group>_<Part>[_L|_R], INT_<ID>_<Name>, SOCKET_<Kind>_<Name>,
    CAM_<Name>.  Root collection `Kestrel` with children Kestrel_Exterior,
    Kestrel_Interior, Kestrel_Sockets.  Preview cameras + sun live in the
    scene-level collection Preview_Rig (outside Kestrel).

Import from another build script with:
    import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib; common = importlib.import_module("00_common")
"""
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector

# ---------------------------------------------------------------- paths
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.dirname(SCRIPTS_DIR)
SHIP_DIR = os.path.dirname(SOURCE_DIR)
BLEND_PATH = os.path.join(SOURCE_DIR, "kestrel.blend")
PREVIEW_DIR = os.path.join(SHIP_DIR, "previews")
EXPORT_DIR = os.path.join(SHIP_DIR, "export")

# ---------------------------------------------------------------- spec / units
M = 1.0                      # 1 Blender unit = 1 metre
SPEC = {
    "length": 23.0,          # overall, Y
    "beam": 4.8,             # hull beam incl. sponsons, X (radiators stowed)
    "span": 12.5,            # radiators deployed, X
    "height": 4.2,           # hull height, Z (gear excluded)
    "height_on_gear": 5.6,
    "tolerance": 0.10,
}

# Key reference heights (Z, metres)
Z_HULL_BOTTOM = -1.9
Z_HULL_TOP = 1.9             # fuselage octagon top
Z_TOP_MAX = 2.3              # canopy / dish top  -> hull height 4.2
Z_GROUND = Z_TOP_MAX - SPEC["height_on_gear"]   # -3.3
Z_DECK = -0.5                # main deck floor
Z_DECK_CEIL = 1.5            # main deck ceiling (2.0 m headroom)

ROOT = "Kestrel"
COL_EXT = "Kestrel_Exterior"
COL_INT = "Kestrel_Interior"
COL_SOCK = "Kestrel_Sockets"
COL_RIG = "Preview_Rig"      # scene-level, NOT under Kestrel (never exported as ship)

EXT_GROUPS = {"HULL", "ENGINES", "RADIATORS", "CANOPY", "GEAR", "WEAPONS",
              "SENSORS", "DOCK", "HATCH", "RCS", "GREEBLES"}


# ---------------------------------------------------------------- argv
def script_args():
    """Arguments after '--' on the blender command line, as a dict of --key value."""
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    out, i = {}, 0
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out[key] = argv[i + 1]
                i += 2
                continue
            out[key] = True
        i += 1
    return out


# ---------------------------------------------------------------- scene io
def reset_scene():
    """Empty the current file completely (works inside a running session)."""
    for coll in (bpy.data.objects, bpy.data.meshes, bpy.data.curves,
                 bpy.data.cameras, bpy.data.lights, bpy.data.materials,
                 bpy.data.collections, bpy.data.images, bpy.data.actions,
                 bpy.data.node_groups):
        for block in list(coll):
            coll.remove(block)
    for w in list(bpy.data.worlds):
        bpy.data.worlds.remove(w)
    scn = bpy.context.scene
    scn.name = "Kestrel"
    scn.unit_settings.system = 'METRIC'
    scn.unit_settings.scale_length = 1.0
    scn.unit_settings.length_unit = 'METERS'
    return scn


def open_blend(path=BLEND_PATH):
    bpy.ops.wm.open_mainfile(filepath=path)
    return bpy.context.scene


def save_blend(path=BLEND_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    # remove Blender's .blend1 backup so the source dir stays clean
    if os.path.exists(path + "1"):
        os.remove(path + "1")
    return path


# ---------------------------------------------------------------- collections
def ensure_collection(name, parent=None):
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    parent_coll = parent or bpy.context.scene.collection
    if coll.name not in parent_coll.children:
        parent_coll.children.link(coll)
    return coll


def setup_collections():
    root = ensure_collection(ROOT)
    return {
        "root": root,
        "ext": ensure_collection(COL_EXT, root),
        "int": ensure_collection(COL_INT, root),
        "sock": ensure_collection(COL_SOCK, root),
    }


def link(obj, coll, parent=None):
    coll.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    return obj


# ---------------------------------------------------------------- naming
def ext_name(group, part, side=None):
    assert group in EXT_GROUPS, group
    return f"EXT_{group}_{part}" + (f"_{side}" if side else "")


def int_name(room_id, name):
    return f"INT_{room_id}_{name}"


def socket_name(kind, name):
    return f"SOCKET_{kind}_{name}"


# ---------------------------------------------------------------- materials
def placeholder_material(name, rgba, alpha=1.0, roughness=0.6):
    """Temporary blockout material (flat colour)."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgba[:3], 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Alpha"].default_value = alpha
    mat.diffuse_color = (*rgba[:3], alpha)
    mat["kestrel_placeholder"] = True
    return mat


# ---------------------------------------------------------------- mesh helpers
def mesh_object(name, verts, faces, coll, mat=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in verts], [], faces)
    me.validate()
    me.update()
    obj = bpy.data.objects.new(name, me)
    coll.objects.link(obj)
    if mat:
        me.materials.append(mat)
    return obj


def recalc_normals(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def box_object(name, xmin, xmax, ymin, ymax, zmin, zmax, coll, mat=None):
    """Axis-aligned box from bounds; origin at box centre."""
    cx, cy, cz = (xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2
    hx, hy, hz = (xmax - xmin) / 2, (ymax - ymin) / 2, (zmax - zmin) / 2
    v = [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
         (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    obj = mesh_object(name, v, f, coll, mat)
    obj.location = (cx, cy, cz)
    return obj


def octagon_ring(hw, hh, chamfer, cx=0.0, cz=0.0):
    """8 (x, z) points of a chamfered rectangle (half-width hw, half-height hh),
    counter-clockwise starting at the starboard side, bottom."""
    c = chamfer
    return [(cx + hw, cz - hh + c), (cx + hw, cz + hh - c), (cx + hw - c, cz + hh),
            (cx - hw + c, cz + hh), (cx - hw, cz + hh - c), (cx - hw, cz - hh + c),
            (cx - hw + c, cz - hh), (cx + hw - c, cz - hh)]


def loft_object(name, sections, coll, mat=None, axis='Y', cap=True):
    """Loft equal-count rings along an axis.
    sections: list of (t, [(u, v), ...]) — for axis 'Y' (u, v) = (x, z);
    for axis 'X' (u, v) = (y, z).  Origin at world 0 (bounds are absolute)."""
    n = len(sections[0][1])
    verts, faces = [], []
    for t, ring in sections:
        for u, v in ring:
            verts.append((u, t, v) if axis == 'Y' else (t, u, v))
    for s in range(len(sections) - 1):
        a, b = s * n, (s + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces.append((a + i, a + j, b + j, b + i))
    if cap:
        faces.append(tuple(range(n)))
        last = (len(sections) - 1) * n
        faces.append(tuple(range(last, last + n)))
    obj = mesh_object(name, verts, faces, coll, mat)
    recalc_normals(obj)
    return obj


def cylinder_object(name, radius, depth, segments, coll, mat=None, axis='Z'):
    """N-gon prism centred at origin along axis."""
    ring = [(radius * math.cos(2 * math.pi * i / segments + math.pi / segments),
             radius * math.sin(2 * math.pi * i / segments + math.pi / segments))
            for i in range(segments)]
    h = depth / 2
    verts = []
    for z in (-h, h):
        for a, b in ring:
            verts.append({'Z': (a, b, z), 'Y': (a, z, b), 'X': (z, a, b)}[axis])
    n = segments
    faces = [(i, (i + 1) % n, n + (i + 1) % n, n + i) for i in range(n)]
    faces += [tuple(range(n)), tuple(range(n, 2 * n))]
    obj = mesh_object(name, verts, faces, coll, mat)
    recalc_normals(obj)
    return obj


# ---------------------------------------------------------------- empties
def empty_object(name, location, coll, display='PLAIN_AXES', size=0.25,
                 rotation=(0.0, 0.0, 0.0), props=None):
    obj = bpy.data.objects.new(name, None)
    obj.empty_display_type = display
    obj.empty_display_size = size
    obj.location = location
    obj.rotation_euler = rotation
    coll.objects.link(obj)
    for k, v in (props or {}).items():
        obj[k] = v
    return obj


# ---------------------------------------------------------------- stats
def triangle_count(objects=None):
    dg = bpy.context.evaluated_depsgraph_get()
    total = 0
    for obj in (bpy.context.scene.objects if objects is None else objects):
        if obj.type != 'MESH':
            continue
        me = obj.evaluated_get(dg).to_mesh()
        me.calc_loop_triangles()
        total += len(me.loop_triangles)
        obj.evaluated_get(dg).to_mesh_clear()
    return total


def world_bbox(objects):
    lo = Vector((1e9, 1e9, 1e9))
    hi = Vector((-1e9, -1e9, -1e9))
    for obj in objects:
        if obj.type != 'MESH':
            continue
        for v in obj.data.vertices:
            w = obj.matrix_world @ v.co
            lo = Vector(map(min, lo, w))
            hi = Vector(map(max, hi, w))
    return lo, hi
