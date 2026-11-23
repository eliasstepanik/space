"""IC-4 "Kestrel" — 30 interior fit-out (pressurised volumes R01-R06 + C01).

Runs ON TOP of 20_exterior.py output: opens the .blend, replaces the blockout
boxes of R01, R02, R03 (+R03a), R04, R05, R06 and C01 with fitted rooms (shell
+ contents) and saves after every room.  Bays B01-B07 are left untouched, as
are Kestrel_Exterior and Kestrel_Sockets (sockets are only READ for placement).

    blender -b --factory-startup --python 30_interior.py -- [--blend in.blend] [--out out.blend]
                                                          [--rooms R01,R02,...] [--report report.json]

Defaults: --blend/--out ../kestrel.blend.  Re-runnable: each room step first
deletes everything named INT_<ID>_* of that room (R03 step also R03a).

Hierarchy:  Kestrel > INT_<ID>_<Name> (empty, room_id props, origin at floor
centre) > INT_<ID>_<Part>.  INT_<ID>_Ceiling is its own node (cutaway views);
ceiling-mounted kit is parented to it.  Pressure doors are separate nodes with
their origin ON THE HINGE (local Z = hinge axis, leaf along local +X), modelled
open; custom props door_* describe them.
Room shells sit OUTSIDE the approved clear volumes (inner faces = blockout box
faces, 0.05 m thick: 0.02 backing + 0.03 bevelled inset panels), so adjacent
rooms share the 0.10 m gaps of the blockout exactly.
Repeated items are linked duplicates of INTM_* library meshes.

Connections: R01-R02 door, R02-R03 inner airlock hatch, R03-R03a chamber hatch,
R03a outer + dock hatches, R03-R04 door, R04-R05 open doorway (no leaf: stores
alcove of the habitat), R05-R06 pressure door, R06-C01 crawl hatch (ladder in R06),
C01 floor hatches to B05/B06/B07 (closed).  The report (--report) lists every door
node, per-room triangle counts and the minimum vertex clearance to the hull skin.
"""
import importlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True

import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")
B = importlib.import_module("10_blockout")
E = importlib.import_module("20_exterior")

PI = math.pi
Tm, R = E.Tm, E.R
add_box, add_cyl, add_revolve, add_torus, add_tube, add_prism, fillet = (
    E.add_box, E.add_cyl, E.add_revolve, E.add_torus, E.add_tube, E.add_prism, E.fillet)

ROOM_ORDER = ["R01", "R02", "R03", "R04", "R05", "R06", "C01"]
SH = 0.05                      # shell thickness
BK = 0.02                      # backing part of the shell (panels = SH - BK)
INT = None
ROOT = None
MAT = {}
BUILDERS = {}
ROOMS = {r[0]: r for r in B.INTERIOR}        # id -> (id, name, x0, x1, y0, y1, z0, z1, purpose)
NODES = {}                                   # id -> room empty


# =================================================================== setup
def setup():
    global INT, ROOT
    cols = C.setup_collections()
    INT = cols["int"]
    ROOT = bpy.data.objects[C.ROOT]
    P = C.placeholder_material
    MAT.update(
        panel=P("MAT_Placeholder_IntPanel", (0.62, 0.63, 0.61), roughness=0.55),
        floor=P("MAT_Placeholder_Floor", (0.15, 0.155, 0.16), roughness=0.75),
        fabric=P("MAT_Placeholder_Fabric", (0.19, 0.26, 0.35), roughness=0.9),
        equip=P("MAT_Placeholder_Equipment", (0.30, 0.32, 0.34), roughness=0.45),
        screen=P("MAT_Placeholder_Screen", (0.02, 0.03, 0.035), roughness=0.15),
        light=P("MAT_Placeholder_IntLight", (0.95, 0.93, 0.88), roughness=0.5),
        metal=P("MAT_Placeholder_Metal", (0.42, 0.42, 0.44), roughness=0.35),
        dark=P("MAT_Placeholder_Dark", (0.12, 0.12, 0.13), roughness=0.5),
        rail=P("MAT_Placeholder_Handrail", (0.95, 0.72, 0.04), roughness=0.5),
        red=P("MAT_Placeholder_Airlock", (0.85, 0.20, 0.25)),
        foil=P("MAT_Placeholder_Foil", (0.80, 0.58, 0.20), roughness=0.3),
        glass=P("MAT_Placeholder_Canopy", (0.05, 0.09, 0.12), roughness=0.1),
    )
    for key, col, strength in (("screen", (0.20, 0.62, 0.80), 3.0), ("light", (1.0, 0.92, 0.80), 12.0)):
        bsdf = MAT[key].node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Emission Color"].default_value = (*col, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    roles = {"panel": "int_panel", "floor": "int_floor", "fabric": "fabric", "equip": "equipment",
             "screen": "screen_emissive", "light": "light_emissive"}
    for k, role in roles.items():
        MAT[k]["kestrel_role"] = role
    E.MAT.update(rail=MAT["rail"])           # E.handrail_mesh() reads E.MAT["rail"]


def room_ids(rid):
    return ["R03", "R03a"] if rid == "R03" else [rid]


def wipe_room(rid):
    ids = set(room_ids(rid))
    for o in list(INT.all_objects):
        parts = o.name.split("_")
        if len(parts) > 2 and parts[0] == "INT" and parts[1] in ids:
            bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)


def room_node(rid):
    _id, name, x0, x1, y0, y1, z0, z1, purpose = ROOMS[rid]
    o = bpy.data.objects.new(C.int_name(rid, name), None)
    INT.objects.link(o)
    o.parent = ROOT
    o.location = ((x0 + x1) / 2, (y0 + y1) / 2, z0)
    o.empty_display_type = 'CUBE'
    o.empty_display_size = 0.25
    o["room_id"] = "R03" if rid == "R03a" else rid
    o["room_name"] = name
    o["purpose"] = purpose
    o["dims_m"] = [round(x1 - x0, 3), round(y1 - y0, 3), round(z1 - z0, 3)]
    o["clear_volume"] = [[x0, x1], [y0, y1], [z0, z1]]
    o["fitted"] = True
    NODES[rid] = o
    return o


# =================================================================== object helpers
def wmat(o):
    m = Matrix.Identity(4)
    while o is not None:
        m = o.matrix_basis @ m
        o = o.parent
    return m


def _mat4(rot):
    if rot is None:
        return Matrix.Identity(4)
    if isinstance(rot, Matrix):
        return rot.to_4x4() if len(rot) == 3 else rot.copy()
    from mathutils import Euler
    return Euler(rot).to_matrix().to_4x4()


def place(name, data, parent, loc=(0, 0, 0), rot=None, world=True):
    assert name not in bpy.data.objects, name
    o = bpy.data.objects.new(name, data)
    INT.objects.link(o)
    o.parent = parent
    M = Tm(loc) @ _mat4(rot)
    if world and parent is not None:
        M = wmat(parent).inverted() @ M
    o.matrix_basis = M
    return o


def empty(name, parent, loc=(0, 0, 0), rot=None, world=True, size=0.2):
    o = place(name, None, parent, loc, rot, world)
    o.empty_display_type = 'PLAIN_AXES'
    o.empty_display_size = size
    return o


def mk_mesh(name, mats, build, smooth=30.0):
    bm = bmesh.new()
    build(bm)
    return E.finish_mesh(name, bm, [MAT[m] for m in mats], smooth)


def part(name, mats, build, parent, loc=(0, 0, 0), rot=None, smooth=30.0, world=True):
    return place(name, mk_mesh(name, mats, build, smooth), parent, loc, rot, world)


LIB_FRESH = set()


def lib(key, mats, build, smooth=30.0):
    """Shared library mesh INTM_<key>: built once per run and reused by linked duplicates.
    A mesh left by a previous run (possibly still used by rooms not rebuilt now) is rebuilt
    once and remapped, so every user picks up the current geometry."""
    name = "INTM_" + key
    old = bpy.data.meshes.get(name)
    if old is not None and name in LIB_FRESH:
        return old
    me = mk_mesh(name + "__new", mats, build, smooth)
    if old is not None:
        old.user_remap(me)
        bpy.data.meshes.remove(old)
    me.name = name
    LIB_FRESH.add(name)
    return me


def inst(name, me, parent, loc=(0, 0, 0), rot=None, world=True):
    return place(name, me, parent, loc, rot, world)


def sock(name):
    return bpy.data.objects[name].matrix_world.translation.copy()


# =================================================================== bmesh helpers
def bx(bm, x0, x1, y0, y1, z0, z1, mi=0, bevel=0.0, segs=1, M=None):
    """Axis-aligned box from bounds (in the frame M)."""
    T = Tm(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    s = min(x1 - x0, y1 - y0, z1 - z0)
    return add_box(bm, x1 - x0, y1 - y0, z1 - z0, (M @ T) if M is not None else T,
                   min(bevel, 0.45 * s), segs, mi)


def align_z(d):
    """4x4 rotation taking local +Z onto direction d."""
    return Vector(d).normalized().to_track_quat('Z', 'Y').to_matrix().to_4x4()


def cylp(bm, p0, p1, r, segs, mi=0, M=None):
    """Capped cylinder from p0 to p1."""
    p0, p1 = Vector(p0), Vector(p1)
    T = Tm(p0) @ align_z(p1 - p0)
    return add_cyl(bm, r, (p1 - p0).length, segs, (M @ T) if M is not None else T, mi, z0=0.0)


def strap(bm, pts, w=0.045, t=0.006, side=(1, 0, 0), mi=0):
    """Flat strap (webbing) along a polyline; width along `side`."""
    for p0, p1 in zip(pts, pts[1:]):
        p0, p1 = Vector(p0), Vector(p1)
        d = p1 - p0
        L = d.length
        if L < 1e-5:
            continue
        y = d / L
        x = Vector(side)
        x = x - y * x.dot(y)
        if x.length < 1e-5:
            x = y.orthogonal()
        x.normalize()
        z = x.cross(y)
        M = Matrix((x, y, z)).transposed().to_4x4()
        M.translation = (p0 + p1) / 2
        add_box(bm, w, L + t, t, M, mi=mi)


def chamfer_rect(w, h, c):
    """Chamfered rectangle (x 0..w, y 0..h) CCW."""
    return [(c, 0), (w - c, 0), (w, c), (w, h - c), (w - c, h), (c, h), (0, h - c), (0, c)]


XZ = Matrix(((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))   # (x, y, z) -> (x, z, y)


# =================================================================== room shells
# A surface is a slab on one face of the clear box.  Axes: u, v in the plane, w normal.
#   side: 'F' floor, 'C' ceiling, 'N' (+Y), 'S' (-Y), 'E' (+X), 'W' (-X)
def _cuts(a, b, edges, pitch):
    n = max(1, round((b - a) / pitch))
    reg = [a + (b - a) * i / n for i in range(n + 1)]
    extra = [e for e in edges if a + 1e-6 < e < b - 1e-6]
    keep = [c for c in reg if c in (reg[0], reg[-1]) or all(abs(c - e) > 0.12 for e in extra)]
    return sorted(set(round(c, 5) for c in keep + extra))


def surface(bm, side, box, holes=(), panel_holes=(), pitch=(0.55, 0.5), mi_panel=0, mi_back=0,
            bevel=0.010, ext=(0.0, 0.0, 0.0, 0.0)):
    """Shell slab for one side of box=(x0,x1,y0,y1,z0,z1).  holes/panel_holes: (u0,u1,v0,v1)
    in world units (panel_holes only omit the bevelled panels, the backing stays).
    ext extends the slab (u0-, u1+, v0-, v1+) so walls overlap at corners."""
    x0, x1, y0, y1, z0, z1 = box
    if side in "FC":
        u0, u1, v0, v1 = x0, x1, y0, y1
        w_in = z0 if side == "F" else z1
        sgn = -1 if side == "F" else 1
        axes = (0, 1, 2)
    elif side in "NS":
        u0, u1, v0, v1 = x0, x1, z0, z1
        w_in = y1 if side == "N" else y0
        sgn = 1 if side == "N" else -1
        axes = (0, 2, 1)
    else:
        u0, u1, v0, v1 = y0, y1, z0, z1
        w_in = x1 if side == "E" else x0
        sgn = 1 if side == "E" else -1
        axes = (1, 2, 0)
    u0, u1, v0, v1 = u0 - ext[0], u1 + ext[1], v0 - ext[2], v1 + ext[3]
    us = _cuts(u0, u1, [h[0] for h in holes] + [h[1] for h in holes], pitch[0])
    vs = _cuts(v0, v1, [h[2] for h in holes] + [h[3] for h in holes], pitch[1])

    def in_any(hs, cu, cv):
        return any(h[0] < cu < h[1] and h[2] < cv < h[3] for h in hs)

    def emit(a0, a1, b0, b1, w0, w1, mi, bev):
        lo, hi = [0, 0, 0], [0, 0, 0]
        for (ax, p, q) in ((axes[0], a0, a1), (axes[1], b0, b1), (axes[2], min(w0, w1), max(w0, w1))):
            lo[ax], hi[ax] = p, q
        bx(bm, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], mi, bev)
    n = 0
    for i in range(len(us) - 1):
        for j in range(len(vs) - 1):
            a0, a1, b0, b1 = us[i], us[i + 1], vs[j], vs[j + 1]
            cu, cv = (a0 + a1) / 2, (b0 + b1) / 2
            if in_any(holes, cu, cv):
                continue
            emit(a0, a1, b0, b1, w_in + sgn * (SH - BK), w_in + sgn * SH, mi_back, 0.0)
            if in_any(panel_holes, cu, cv):
                continue
            g = 0.003
            emit(a0 + g, a1 - g, b0 + g, b1 - g, w_in, w_in + sgn * (SH - BK), mi_panel, bevel)
            n += 1
    return n


def room_shell(rid, holes=None, panel_holes=None, skip=(), floor_mat="floor", ceiling_extra=None,
               pitch_wall=(0.55, 0.5), light=True):
    """Floor + walls (one mesh each) and the Ceiling node of a box room."""
    _id, name, x0, x1, y0, y1, z0, z1, _p = ROOMS[rid]
    box = (x0, x1, y0, y1, z0, z1)
    holes = holes or {}
    panel_holes = panel_holes or {}
    node = NODES[rid]
    if "F" not in skip:
        part(C.int_name(rid, "Floor"), [floor_mat, "dark"], lambda bm: surface(
            bm, "F", box, holes.get("F", ()), panel_holes.get("F", ()), (0.55, 0.55), 0, 1, 0.006),
            node, smooth=None)

    def walls(bm):
        for s in "NSEW":
            if s in skip:
                continue
            ext = (SH, SH, SH, SH) if s in "EW" else (0.0, 0.0, SH, SH)
            surface(bm, s, box, holes.get(s, ()), panel_holes.get(s, ()), pitch_wall, 0, 1, 0.010, ext)
    part(C.int_name(rid, "Walls"), ["panel", "equip"], walls, node, smooth=None)
    ceil = None
    if "C" not in skip:
        def ceiling(bm):
            surface(bm, "C", box, holes.get("C", ()), panel_holes.get("C", ()), (0.55, 0.55), 0, 1, 0.010)
            if light:                                     # recessed light panel at Light_<ID>
                sn = f"SOCKET_Light_{rid}"
                p = sock(sn) if sn in bpy.data.objects else Vector(((x0 + x1) / 2, (y0 + y1) / 2, z1))
                if True:
                    bx(bm, p.x - 0.28, p.x + 0.28, p.y - 0.11, p.y + 0.11, z1 - 0.022, z1 - 0.002, 2, 0.006)
                    bx(bm, p.x - 0.25, p.x + 0.25, p.y - 0.08, p.y + 0.08, z1 - 0.026, z1 - 0.02, 3)
            if ceiling_extra:
                ceiling_extra(bm)
        ceil = part(C.int_name(rid, "Ceiling"), ["panel", "equip", "metal", "light"], ceiling, node, smooth=None)
        ceil["cutaway"] = "hide this node (and its children) to look into the room from above"
    return ceil


def doorway(name, parent, axis, w0, w1, u0, u1, v0, v1, trim=0.045):
    """Jamb lining through a wall gap (axis 'Y': opening in an XZ wall spanning w0..w1 in Y;
    axis 'X': opening in a YZ wall spanning w0..w1 in X) + trim lips on both faces."""
    def b(bm):
        t = 0.018
        if axis == "Y":
            M = Matrix.Identity(4)
        else:   # map (u=y, w=x) -> swap x/y
            M = Matrix(((0, 1, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)))
        lo, hi = min(w0, w1), max(w0, w1)
        # lining: 4 sides of the opening (u = x, w = y, v = z in the local frame)
        bx(bm, u0, u0 + t, lo, hi, v0, v1, 0, M=M)
        bx(bm, u1 - t, u1, lo, hi, v0, v1, 0, M=M)
        bx(bm, u0, u1, lo, hi, v1 - t, v1, 0, M=M)
        bx(bm, u0, u1, lo, hi, v0, v0 + t, 1, M=M)             # sill
        for wf, sg in ((lo, -1), (hi, 1)):                      # trim lips
            wa, wb = sorted((wf, wf + sg * 0.012))
            bx(bm, u0 - trim, u0 + t, wa, wb, v0 - (trim if v0 > -0.45 else 0), v1 + trim, 0, 0.004, M=M)
            bx(bm, u1 - t, u1 + trim, wa, wb, v0 - (trim if v0 > -0.45 else 0), v1 + trim, 0, 0.004, M=M)
            bx(bm, u0 - trim, u1 + trim, wa, wb, v1 - t, v1 + trim, 0, 0.004, M=M)
            if v0 > -0.45:
                bx(bm, u0 - trim, u1 + trim, wa, wb, v0 - trim, v0 + t, 0, 0.004, M=M)
            for zz in (v0 + 0.25, v1 - 0.25):                  # door-seal latch strikers
                bx(bm, u1 - t - 0.012, u1 - t, (lo + hi) / 2 - 0.03, (lo + hi) / 2 + 0.03, zz - 0.04, zz + 0.04, 1, M=M)
    return part(name, ["metal", "dark"], b, parent, smooth=None)


# =================================================================== doors (origin on the hinge)
def door_leaf_mesh(key, w, h, t=0.06, wheel=True, chamfer=0.08, window=False):
    """Leaf in local frame: hinge axis = local Z (0..h), leaf along +X (0..w), normal = local Y."""
    def b(bm):
        prof = chamfer_rect(w - 0.02, h, chamfer)
        M = Tm((0.02, 0, 0)) @ R(PI / 2, 'X')                   # polygon XY -> XZ, extruded along Y
        add_prism(bm, prof, t, M, mi=0)
        for sg in (-1, 1):                                       # raised rim on both faces
            inner = chamfer_rect(w - 0.14, h - 0.12, chamfer * 0.7)
            add_prism(bm, inner, 0.012, Tm((0.08, sg * (t / 2 + 0.006), 0.06)) @ R(PI / 2, 'X'), mi=0)
        for zz in (0.18, h - 0.18):                              # hinge knuckles + pins
            add_cyl(bm, 0.03, 0.16, 24, Tm((0.0, 0.0, zz)), mi=1)
        for zz in (0.3, h / 2, h - 0.3):                         # dogs on the latch edge
            bx(bm, w - 0.02, w + 0.02, -0.02, 0.02, zz - 0.05, zz + 0.05, 1, 0.006)
        if window:
            bx(bm, w / 2 - 0.09, w / 2 + 0.09, -t / 2 - 0.016, t / 2 + 0.016, h * 0.68, h * 0.68 + 0.22, 1, 0.01)
            bx(bm, w / 2 - 0.07, w / 2 + 0.07, -t / 2 - 0.018, t / 2 + 0.018, h * 0.68 + 0.02, h * 0.68 + 0.2, 2)
        if wheel:
            cz = h * 0.5
            for sg in (-1, 1):
                cy = sg * (t / 2 + 0.07)
                Mw = Tm((w / 2, cy, cz)) @ R(PI / 2, 'X')
                add_torus(bm, 0.13, 0.013, 24, 8, Mw, mi=3)
                add_cyl(bm, 0.035, 0.05, 24, Mw, mi=1)
                for k in range(3):
                    a = k * 2 * PI / 3 + PI / 2
                    p1 = Vector((w / 2 + 0.13 * math.cos(a), cy, cz + 0.13 * math.sin(a)))
                    cylp(bm, (w / 2, cy, cz), p1, 0.009, 8, mi=1)
                cylp(bm, (w / 2, sg * t / 2, cz), (w / 2, cy, cz), 0.018, 8, mi=1)
        else:
            for sg in (-1, 1):                                   # lever latch
                cylp(bm, (w - 0.12, sg * t / 2, h * 0.5), (w - 0.12, sg * (t / 2 + 0.05), h * 0.5), 0.014, 8, 1)
                bx(bm, w - 0.30, w - 0.10, sg * (t / 2 + 0.04), sg * (t / 2 + 0.065), h * 0.5 - 0.015,
                   h * 0.5 + 0.015, 1, 0.006)
    return lib(key, ["panel", "metal", "glass", "rail"], b, 35)


def door(name, parent, me, hinge, axis, leaf_dir, swing_dir, open_deg, connects, note=""):
    """Place a door leaf mesh so local Z = axis (hinge line from `hinge`), local X = leaf_dir
    when closed, then rotate open by open_deg toward swing_dir."""
    Z = Vector(axis).normalized()
    X = Vector(leaf_dir).normalized()
    Y = Z.cross(X)
    if Y.dot(Vector(swing_dir)) < 0:          # make positive rotation swing toward swing_dir
        raise ValueError(f"{name}: choose axis so that axis x leaf_dir points to the swing side")
    basis = Matrix((X, Y, Z)).transposed().to_4x4()
    rot = basis @ R(math.radians(open_deg), 'Z')
    o = place(name, me, parent, hinge, rot)
    o["door"] = True
    o["door_hinge_world"] = [round(v, 3) for v in hinge]
    o["door_hinge_axis"] = [round(v, 3) for v in Z]
    o["door_open_deg"] = open_deg
    o["door_connects"] = connects
    o["door_state"] = "open" if open_deg > 1 else "closed"
    if note:
        o["note"] = note
    return o


# =================================================================== shared library meshes
def grate_tile(w, l):
    """Floor grating tile, top at z=0, x/y centred (frame + 0.05 m bearing bars + cross bars)."""
    def b(bm):
        f, h = 0.022, 0.03
        bx(bm, -w / 2, w / 2, -l / 2, -l / 2 + f, -h, 0, 0)
        bx(bm, -w / 2, w / 2, l / 2 - f, l / 2, -h, 0, 0)
        bx(bm, -w / 2, -w / 2 + f, -l / 2 + f, l / 2 - f, -h, 0, 0)
        bx(bm, w / 2 - f, w / 2, -l / 2 + f, l / 2 - f, -h, 0, 0)
        n = max(2, round((w - 2 * f) / 0.05))
        for i in range(1, n):
            x = -w / 2 + f + (w - 2 * f) * i / n
            bx(bm, x - 0.003, x + 0.003, -l / 2 + f, l / 2 - f, -h + 0.004, -0.001, 0)
        m = max(2, round((l - 2 * f) / 0.1))
        for j in range(1, m):
            y = -l / 2 + f + (l - 2 * f) * j / m
            bx(bm, -w / 2 + f, w / 2 - f, y - 0.004, y + 0.004, -h + 0.008, -0.004, 0)
    return lib(f"Grate_{int(w * 1000)}x{int(l * 1000)}", ["metal"], b, None)


def lay_grating(rid, parent, x0, x1, y0, y1, ztop, nx, ny):
    me = grate_tile((x1 - x0) / nx, (y1 - y0) / ny)
    k = 0
    for i in range(nx):
        for j in range(ny):
            k += 1
            inst(C.int_name(rid, f"FloorGrate{k:02d}"), me, parent,
                 (x0 + (i + 0.5) * (x1 - x0) / nx, y0 + (j + 0.5) * (y1 - y0) / ny, ztop))


def handhold(length=0.4):
    """Yellow EVA/IVA handhold along local Y, standing off local +Z (feet at z=0)."""
    key = f"INTM_Handhold{int(length * 100):02d}"
    me = bpy.data.meshes.get(key)
    return me if me is not None else E.handrail_mesh(key, length, stand=0.07, r=0.014)


def wall_rot(normal, along=(0, 0, 1)):
    return E.frame(normal, along)


def extinguisher():
    def b(bm):
        add_revolve(bm, [(0.0, 0.0), (0.055, 0.0), (0.06, 0.012), (0.06, 0.34), (0.052, 0.38), (0.03, 0.40),
                         (0.022, 0.415)], 24, cap0=False, cap1=True, mi=0)
        add_cyl(bm, 0.02, 0.04, 8, Tm((0, 0, 0.43)), mi=1)                  # valve
        bx(bm, -0.02, 0.09, -0.012, 0.012, 0.445, 0.46, 1, 0.004)            # handle lever
        bx(bm, -0.02, 0.07, -0.01, 0.01, 0.415, 0.425, 1, 0.003)
        add_cyl(bm, 0.014, 0.012, 8, Tm((0, -0.024, 0.43)) @ R(PI / 2, 'X'), mi=2)   # gauge
        add_tube(bm, fillet([(-0.02, 0, 0.43), (-0.07, 0, 0.42), (-0.075, 0, 0.25), (-0.07, 0, 0.12)], 0.03, 2),
                 0.009, 8, mi=2)
        add_cyl(bm, 0.013, 0.05, 8, Tm((-0.07, 0, 0.1)), mi=2)              # nozzle
        for z in (0.1, 0.3):                                                 # bracket bands
            add_revolve(bm, [(0.061, z - 0.012), (0.066, z - 0.012), (0.066, z + 0.012), (0.061, z + 0.012)],
                        24, closed=True, mi=1)
        bx(bm, -0.05, 0.05, 0.062, 0.075, 0.05, 0.36, 1, 0.004)            # wall plate (+Y = wall)
    return lib("Extinguisher", ["red", "metal", "dark"], b, 40)


def mfd(w=0.25, h=0.23):
    """Display: bezel + emissive screen + bezel keys; faces local +Z, up = local +Y."""
    def b(bm):
        bx(bm, -w / 2, w / 2, -h / 2, h / 2, -0.035, 0, 0, 0.006)
        bx(bm, -w / 2 + 0.03, w / 2 - 0.03, -h / 2 + 0.03, h / 2 - 0.03, 0, 0.002, 1)
        n = 5
        for i in range(n):
            t = (i + 0.5) / n
            y = -h / 2 + 0.03 + (h - 0.06) * t
            for x in (-w / 2 + 0.015, w / 2 - 0.015):
                bx(bm, x - 0.008, x + 0.008, y - 0.01, y + 0.01, 0, 0.008, 2)
        for i in range(4):
            x = -w / 2 + 0.04 + (w - 0.08) * (i + 0.5) / 4
            for y in (-h / 2 + 0.015, h / 2 - 0.015):
                bx(bm, x - 0.012, x + 0.012, y - 0.007, y + 0.007, 0, 0.008, 2)
    return lib(f"MFD_{int(w * 100)}x{int(h * 100)}", ["dark", "screen", "metal"], b, None)


def button_pad(cols=6, rows=4):
    def b(bm):
        w, h = cols * 0.032 + 0.02, rows * 0.03 + 0.02
        bx(bm, -w / 2, w / 2, -h / 2, h / 2, -0.012, 0, 0, 0.003)
        for i in range(cols):
            for j in range(rows):
                x = -w / 2 + 0.01 + 0.032 * (i + 0.5)
                y = -h / 2 + 0.01 + 0.03 * (j + 0.5)
                bx(bm, x - 0.011, x + 0.011, y - 0.009, y + 0.009, 0, 0.007, 1 if (i + j) % 5 else 2)
    return lib(f"ButtonPad{cols}x{rows}", ["dark", "metal", "screen"], b, None)


def switch_strip(n=8):
    def b(bm):
        w = n * 0.03 + 0.02
        bx(bm, -w / 2, w / 2, -0.025, 0.025, -0.008, 0, 0, 0.003)
        for i in range(n):
            x = -w / 2 + 0.01 + 0.03 * (i + 0.5)
            add_cyl(bm, 0.008, 0.008, 6, Tm((x, -0.004, 0.004)), mi=1)
            cylp(bm, (x, -0.004, 0.006), (x, 0.004, 0.026), 0.0035, 6, mi=1)
            bx(bm, x - 0.008, x + 0.008, 0.012, 0.02, 0, 0.002, 2)          # label / LED
    return lib(f"SwitchStrip{n}", ["dark", "metal", "screen"], b, None)


def breaker_row(n=10):
    """Circuit-breaker row: plate + n pull-type breakers; faces local +Z, row along X."""
    def b(bm):
        w = n * 0.028 + 0.02
        bx(bm, -w / 2, w / 2, -0.024, 0.024, -0.006, 0, 0, 0.002)
        for i in range(n):
            x = -w / 2 + 0.01 + 0.028 * (i + 0.5)
            add_cyl(bm, 0.0065, 0.014, 6, Tm((x, 0.002, 0.007)), mi=2 if i % 4 else 3)
        bx(bm, -w / 2 + 0.01, w / 2 - 0.01, -0.021, -0.013, 0, 0.0015, 1)     # label strip
    return lib(f"BreakerRow{n}", ["dark", "metal", "dark", "rail"], b, None)


def ladder(height, width=0.4, standoff=0.12, pitch=0.28, key=None):
    """Ladder against a wall: rails along local Z (0..height), rungs along X, standing off +Y."""
    def b(bm):
        for x in (-width / 2, width / 2):
            bx(bm, x - 0.02, x + 0.02, standoff - 0.02, standoff + 0.02, 0, height, 0, 0.006)
            for z in (0.1, height - 0.1):
                bx(bm, x - 0.015, x + 0.015, 0, standoff - 0.02, z - 0.02, z + 0.02, 0, 0.004)
        n = int((height - 0.15) / pitch)
        for i in range(n):
            z = 0.2 + i * pitch
            cylp(bm, (-width / 2, standoff, z), (width / 2, standoff, z), 0.015, 8, mi=1)
    return lib(key or f"Ladder{int(height * 100)}", ["metal", "rail"], b, 35)


def cable_tray(length, width=0.2, depth=0.06):
    """U channel along local Y (0..length), opening +Z, with cable bundle."""
    def b(bm):
        bx(bm, -width / 2, width / 2, 0, length, 0, 0.006, 0)
        for x in (-width / 2, width / 2 - 0.006):
            bx(bm, x, x + 0.006, 0, length, 0, depth, 0)
        n = max(2, int(length / 0.4))
        for i in range(n + 1):
            y = length * i / n
            bx(bm, -width / 2 - 0.01, width / 2 + 0.01, y - 0.012, y + 0.012, -0.006, 0.0, 0)
        for k, (x, z, r) in enumerate(((-0.05, 0.022, 0.016), (-0.015, 0.02, 0.014), (0.02, 0.022, 0.017),
                                       (0.055, 0.018, 0.012), (0.0, 0.045, 0.014), (-0.035, 0.048, 0.012))):
            if abs(x) + r > width / 2:
                continue
            add_tube(bm, [(x, 0.02, z), (x, length - 0.02, z)], r, 6, mi=1 + k % 2)
    return lib(f"CableTray{int(length * 100)}", ["metal", "dark", "red"], b, 40)


def locker(w, h, d, doors=1, key=None):
    """Locker carcass: x -w/2..w/2, y 0..d (front at y=d), z 0..h; vented doors on the front."""
    def b(bm):
        bx(bm, -w / 2, w / 2, 0, d - 0.02, 0, h, 0, 0.008)
        dw = w / doors
        for i in range(doors):
            x0 = -w / 2 + i * dw + 0.006
            x1 = x0 + dw - 0.012
            bx(bm, x0, x1, d - 0.02, d, 0.02, h - 0.02, 1, 0.006)
            for k in range(5):                                      # vent slots
                z = h - 0.15 - k * 0.03
                bx(bm, x0 + 0.05, x1 - 0.05, d, d + 0.003, z - 0.006, z + 0.006, 2)
            hx = x1 - 0.04 if i % 2 == 0 else x0 + 0.04
            bx(bm, hx - 0.012, hx + 0.012, d, d + 0.03, h * 0.5 - 0.08, h * 0.5 + 0.08, 3, 0.006)
            bx(bm, (x0 + x1) / 2 - 0.05, (x0 + x1) / 2 + 0.05, d, d + 0.002, h - 0.09, h - 0.06, 4)   # name plate
    return lib(key or f"Locker{int(w * 100)}x{int(h * 100)}x{int(d * 100)}_{doors}",
               ["panel", "equip", "dark", "metal", "rail"], b, None)


def crate(w, l, h, key):
    """Cargo container: ribbed bevelled box, handles, latches; bottom at z=0."""
    def b(bm):
        bx(bm, -w / 2, w / 2, -l / 2, l / 2, 0, h, 0, 0.015, 2)
        for x in (-w / 2 + 0.05, w / 2 - 0.05):                     # ribs
            bx(bm, x - 0.012, x + 0.012, -l / 2 - 0.006, l / 2 + 0.006, 0.01, h - 0.01, 1, 0.004)
        for sy in (-1, 1):                                          # latches on the long sides
            for x in (-w / 4, w / 4):
                bx(bm, x - 0.025, x + 0.025, sy * (l / 2) - 0.012, sy * (l / 2) + 0.012, h - 0.05, h - 0.01, 1, 0.003)
        for sx in (-1, 1):                                          # end handles
            bx(bm, sx * w / 2 - 0.012, sx * w / 2 + 0.012, -0.07, 0.07, h * 0.6, h * 0.6 + 0.03, 1, 0.006)
        bx(bm, -0.06, 0.06, -l / 2 - 0.002, -l / 2, h * 0.3, h * 0.3 + 0.05, 2)        # label
    return lib(key, ["equip", "metal", "rail"], b, None)


# =================================================================== R01 cockpit
SEAT_RECLINE = math.radians(12)


def seat_mesh():
    """Crew seat, origin on the floor socket, facing +Y."""
    def b(bm):
        for x in (-0.17, 0.17):                                                  # floor rails
            bx(bm, x - 0.025, x + 0.025, -0.30, 0.28, 0, 0.04, 2, 0.006)
        bx(bm, -0.14, 0.14, -0.20, 0.15, 0.04, 0.34, 0, 0.01)                   # pedestal
        bx(bm, -0.26, 0.26, -0.24, 0.26, 0.34, 0.40, 0, 0.012, 2)               # pan frame
        bx(bm, -0.23, 0.23, -0.20, 0.25, 0.40, 0.47, 1, 0.025, 2)               # pan cushion
        Mb = Tm((0, -0.22, 0.40)) @ R(SEAT_RECLINE, 'X')
        bx(bm, -0.26, 0.26, -0.03, 0.03, 0.0, 0.80, 0, 0.012, 2, M=Mb)        # back frame
        bx(bm, -0.21, 0.21, 0.03, 0.10, 0.06, 0.64, 1, 0.025, 2, M=Mb)        # back cushion
        bx(bm, -0.15, 0.15, 0.0, 0.12, 0.66, 0.86, 1, 0.03, 2, M=Mb)          # headrest
        for sx in (-1, 1):
            bx(bm, sx * 0.20 - 0.03, sx * 0.20 + 0.03, 0.03, 0.13, 0.10, 0.58, 1, 0.02, 2, M=Mb)   # bolsters
            bx(bm, sx * 0.295 - 0.035, sx * 0.295 + 0.035, -0.10, 0.28, 0.62, 0.67, 0, 0.012, 2)  # armrest
            cylp(bm, (sx * 0.295, 0.0, 0.40), (sx * 0.295, 0.0, 0.62), 0.016, 8, 2)
            cylp(bm, (sx * 0.295, 0.22, 0.40), (sx * 0.295, 0.22, 0.62), 0.012, 8, 2)
        add_tube(bm, fillet([(-0.07, 0.25, 0.40), (-0.07, 0.31, 0.36), (0.07, 0.31, 0.36), (0.07, 0.25, 0.40)],
                            0.02, 2), 0.01, 8, mi=3)                               # ejection / egress D-ring
    return lib("Seat", ["equip", "fabric", "metal", "rail"], b, 35)


def harness_mesh():
    """5-point harness lying slack on the empty seat (same local frame as seat_mesh)."""
    s, c = math.sin(SEAT_RECLINE), math.cos(SEAT_RECLINE)

    def back(x, h, d=0.105):
        return Vector((x, -0.22 - h * s + d * c, 0.40 + h * c + d * s))

    def b(bm):
        for sx in (-1, 1):
            strap(bm, [back(sx * 0.09, 0.66, 0.06), back(sx * 0.09, 0.62), back(sx * 0.085, 0.30),
                       back(sx * 0.08, 0.10), (sx * 0.07, -0.05, 0.476), (sx * 0.06, 0.10, 0.476)], side=(1, 0, 0))
            strap(bm, [(sx * 0.235, -0.18, 0.45), (sx * 0.22, -0.10, 0.476), (sx * 0.08, 0.12, 0.476)],
                  side=(1, 0, 0))
            bx(bm, sx * 0.06 - 0.025, sx * 0.06 + 0.025, 0.09, 0.13, 0.474, 0.484, 1, 0.003)    # tongues
            bx(bm, sx * 0.09 - 0.02, sx * 0.09 + 0.02, -0.02, 0.0, 0.47, 0.50, 1, 0.003)       # adjusters
        strap(bm, [(0, 0.27, 0.43), (0, 0.24, 0.476), (0, 0.15, 0.476)], side=(1, 0, 0))
        add_cyl(bm, 0.045, 0.014, 24, Tm((0, 0.14, 0.483)), mi=1)                               # rotary buckle
        add_cyl(bm, 0.02, 0.008, 8, Tm((0, 0.14, 0.494)), mi=1)
    return lib("Harness", ["dark", "metal"], b, 40)


def sidestick_mesh():
    def b(bm):
        add_cyl(bm, 0.035, 0.03, 8, Tm((0, 0, 0.015)), mi=1)                        # boot
        cylp(bm, (0, 0, 0.03), (0, 0.01, 0.10), 0.012, 8, 0)
        Mg = Tm((0, 0.015, 0.10)) @ R(math.radians(-12), 'X')
        bx(bm, -0.022, 0.022, -0.028, 0.028, 0.0, 0.12, 1, 0.012, 2, M=Mg)       # grip
        bx(bm, -0.008, 0.008, 0.026, 0.036, 0.06, 0.09, 2, 0.003, M=Mg)            # trigger
        add_cyl(bm, 0.008, 0.012, 8, Mg @ Tm((0, 0.0, 0.125)), mi=2)               # hat switch
        bx(bm, 0.018, 0.026, -0.01, 0.01, 0.09, 0.105, 3, M=Mg)                    # pickle button
    return lib("SideStick", ["metal", "dark", "equip", "red"], b, 35)


def throttle_mesh():
    def b(bm):
        bx(bm, -0.03, 0.03, -0.09, 0.09, 0.0, 0.03, 0, 0.006)                      # quadrant
        bx(bm, -0.004, 0.004, -0.07, 0.07, 0.03, 0.032, 2)                          # slot
        Ml = Tm((0, 0.02, 0.02)) @ R(math.radians(15), 'X')
        cylp(bm, (0, 0, 0), (0, 0, 0.08), 0.008, 8, 0, M=Ml)
        bx(bm, -0.03, 0.03, -0.022, 0.022, 0.07, 0.115, 1, 0.012, 2, M=Ml)         # handle
        for x in (-0.012, 0.012):
            add_cyl(bm, 0.006, 0.01, 8, Ml @ Tm((x, -0.022, 0.1)) @ R(PI / 2, 'X'), mi=2)
    return lib("Throttle", ["metal", "dark", "equip"], b, 35)


def pedal_pair_mesh():
    def b(bm):
        for x in (-0.2, 0.2):
            Mp = Tm((x, 0.0, 0.06)) @ R(math.radians(-20), 'X')
            bx(bm, -0.045, 0.045, -0.015, 0.015, 0.0, 0.17, 0, 0.008, M=Mp)
            for z in (0.04, 0.09, 0.14):                                            # grip ribs
                bx(bm, -0.04, 0.04, -0.021, -0.015, z - 0.005, z + 0.005, 1, M=Mp)
            cylp(bm, (x, 0.05, 0.13), (x, 0.12, 0.33), 0.012, 8, 1)
            bx(bm, x - 0.03, x + 0.03, 0.10, 0.16, 0.30, 0.36, 1, 0.006)             # pivot bracket
            add_cyl(bm, 0.014, 0.08, 8, Tm((x, 0.13, 0.33)) @ R(PI / 2, 'Y'), mi=1)
    return lib("PedalPair", ["equip", "metal"], b, 35)


def side_console_mesh(length, width):
    """Side console: x -w/2..w/2, y 0..length, floor at z=0, deck at z ~0.5."""
    def b(bm):
        w = width
        bx(bm, -w / 2 + 0.02, w / 2, 0, length, 0, 0.47, 0, 0.01)
        bx(bm, -w / 2, w / 2, -0.01, length + 0.01, 0.47, 0.51, 1, 0.008, 2)
        for i in range(int(length / 0.25)):                                        # DZUS-style deck panels
            y = 0.04 + i * 0.25
            bx(bm, -w / 2 + 0.015, w / 2 - 0.015, y, y + 0.235, 0.51, 0.514, 2, 0.002)
    return lib(f"SideConsole{int(length * 100)}x{int(width * 100)}", ["equip", "dark", "dark"], b, None)


_TREES = []


def signed_clearance(p):
    """Signed distance of p to the pressure-hull skin (see skin_volume_trees); + = inside."""
    if not _TREES:
        _TREES.append(skin_volume_trees())
    dist, inside = _TREES[0]
    d = min(tr.find_nearest(p)[3] for tr in dist)
    return d if inside(p) else -d


def window_reveals(bm):
    """Interior frame reveals following the 5 canopy panes (one closed loop per pane),
    0.08 m inboard of the glazing and 0.045 m inset from each pane edge."""
    rings = E.canopy_rings()
    (y0, r0), (y1, r1), (y2, r2), _ = rings
    yf = 8.9                               # windscreen reveal stops at Y 8.9 (nose top below)
    rf = [(r1[i][0] + (r0[i][0] - r1[i][0]) * ((yf - y1) / (y0 - y1)),
           r1[i][1] + (r0[i][1] - r1[i][1]) * ((yf - y1) / (y0 - y1))) for i in range(7)]

    def pane(ya, ra, yb, rb, i0, i1):
        pts = [Vector((ra[i][0], ya, ra[i][1])) for i in range(i0, i1 + 1)]
        pts += [Vector((rb[i][0], yb, rb[i][1])) for i in range(i1, i0 - 1, -1)]
        return pts
    panes = [pane(yf, rf, y1, r1, 0, 3), pane(yf, rf, y1, r1, 3, 6), pane(y1, r1, y2, r2, 0, 2),
             pane(y1, r1, y2, r2, 4, 6), pane(y1, r1, y2, r2, 2, 4)]
    for P0 in panes:
        c = sum(P0, Vector()) / len(P0)
        P = []
        for a, b_ in zip(P0, P0[1:] + P0[:1]):          # subdivide long edges (~0.15 m)
            n = max(1, int((b_ - a).length / 0.15))
            P += [a.lerp(b_, k / n) for k in range(n)]
        loop = []
        for p in P:
            inward = Vector((0, p.y, 1.45)) - p
            inward.y = 0
            q = p + inward.normalized() * 0.08
            q = q + (c - q).normalized() * 0.045
            for _ in range(40):                  # walk toward the pane centre until 0.075 m clear
                if signed_clearance(q) >= 0.075:
                    break
                q = q + (c - q).normalized() * 0.01
            else:
                continue                         # cannot clear here (glass/skin wedge): skip point
            if not loop or (q - loop[-1]).length > 0.03:
                loop.append(q)
        add_tube(bm, loop, 0.022, 4, closed=True, mi=0)


def build_R01():
    rid = "R01"
    node = room_node(rid)
    x0, x1, y0, y1, z0 = -0.75, 0.75, 4.40, 7.90, 0.55
    DOOR = (-0.75, -0.27, 0.55, 1.45)            # aft wall opening to R02 (x, z)

    # ---- shell: floor backing + grating, walls to the canopy sill, aft wall to the liner
    part(C.int_name(rid, "Floor"), ["floor", "dark"], lambda bm: surface(
        bm, "F", (x0, x1, y0, y1, z0, 2.15), (), [(-9, 9, -99, 99)], (0.55, 0.55), 0, 1), node, smooth=None)
    lay_grating(rid, node, x0, x1, y0, y1, z0, 3, 7)

    def walls(bm):
        surface(bm, "E", (x0, x1, y0, y1, z0, 1.57), (), (), (0.5, 0.5), 0, 1, 0.01, (SH, SH, SH, 0))
        surface(bm, "W", (x0, x1, y0, y1, z0, 1.57), (), (), (0.5, 0.5), 0, 1, 0.01, (SH, SH, SH, 0))
        surface(bm, "N", (x0, x1, y0, y1, z0, 1.70), (), (), (0.5, 0.5), 0, 1, 0.01, (0, 0, SH, 0))
        surface(bm, "S", (x0, x1, y0, y1, z0, 1.80), [DOOR], (), (0.5, 0.5), 0, 1, 0.01, (0, 0, SH, 0))
        for sx in (-1, 1):                                       # canopy sill liners (Y 4.40 .. 7.60)
            poly = [(0.75, 1.57), (1.0, 1.57), (1.0, 1.70), (0.885, 1.815), (0.75, 1.60)]
            poly = [(sx * px, pz) for px, pz in poly]
            if sx < 0:
                poly.reverse()
            add_prism(bm, poly, 3.2, Tm((0, 6.0, 0)) @ R(PI / 2, 'X'), mi=0)
    part(C.int_name(rid, "Walls"), ["panel", "equip"], walls, node, smooth=None)
    doorway(C.int_name(rid, "DoorwayR02"), node, "Y", 4.30, 4.40, *DOOR)
    me = door_leaf_mesh("Door_048x088", 0.47, 0.88, 0.05, wheel=False, chamfer=0.06)
    door(C.int_name(rid, "DoorR02"), node, me, (-0.725, 4.43, 0.56), (0, 0, 1), (1, 0, 0), (0, 1, 0), 90,
         "R01<->R02", "cockpit aft pressure door; swings forward against the port wall")

    # ---- ceiling node: aft overhead liner under the hull top (Y 4.40-4.62) with the CB panel,
    #      plus the window reveals (children, hidden together for cutaways)
    def ceiling(bm):
        bx(bm, -0.80, 0.80, 4.40, 4.62, 1.80, 1.84, 0, 0.006)
        for sx in (-1, 1):
            bx(bm, sx * 0.20, sx * 0.64, 4.41, 4.61, 1.785, 1.80, 1, 0.004) if sx > 0 else \
                bx(bm, -0.64, -0.20, 4.41, 4.61, 1.785, 1.80, 1, 0.004)
        bx(bm, -0.12, 0.12, 4.44, 4.58, 1.788, 1.80, 3, 0.004)              # dome light
    ceil = part(C.int_name(rid, "Ceiling"), ["panel", "equip", "metal", "light"], ceiling, node, smooth=None)
    ceil["cutaway"] = "hide this node (and its children) + EXT_CANOPY_* to look into the cockpit from above"
    rot_dn = wall_rot((0, 0, -1), (0, 1, 0))
    k = 0
    for sx in (-1, 1):
        for yy in (4.465, 4.555):
            k += 1
            inst(C.int_name(rid, f"OverheadBreakers{k}"), breaker_row(10), ceil, (sx * 0.42, yy, 1.785), rot_dn)
    part(C.int_name(rid, "WindowReveals"), ["equip"], window_reveals, ceil, smooth=None)

    # ---- seats, harnesses, side-sticks, throttles
    sm, hm, st, th = seat_mesh(), harness_mesh(), sidestick_mesh(), throttle_mesh()
    for role in ("Pilot", "Engineer"):
        p = sock(f"SOCKET_Seat_{role}")
        s = inst(C.int_name(rid, f"Seat{role}"), sm, node, p)
        s["socket"] = f"SOCKET_Seat_{role}"
        inst(C.int_name(rid, f"Harness{role}"), hm, s, (0, 0, 0), world=False)
        inst(C.int_name(rid, f"SideStick{role}"), st, s, (0.295, 0.20, 0.67), world=False)
        inst(C.int_name(rid, f"Throttle{role}"), th, s, (-0.295, 0.12, 0.67), world=False)

    # ---- pilot main instrument panel + glare shield + HUD
    def main_panel(bm):
        poly = [(7.95, 0.55), (7.95, 1.55), (7.66, 1.55), (7.55, 0.95), (7.75, 0.95), (7.75, 0.55)]
        add_prism(bm, [(y, z) for y, z in poly], 1.36, Tm((0, 0, 0)) @ Matrix(
            ((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))), mi=0)
        gs = [(7.40, 1.55), (7.95, 1.55), (7.95, 1.70), (7.40, 1.60)]
        add_prism(bm, gs, 1.44, Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))), mi=1)
        bx(bm, -0.66, 0.66, 7.38, 7.42, 1.53, 1.60, 1, 0.01)                   # glare-shield lip
        bx(bm, -0.5, 0.5, 7.42, 7.46, 1.535, 1.55, 3)                           # flood strip
        for x in (-0.48, -0.36, 0.36, 0.48):                                    # standby dials
            add_cyl(bm, 0.045, 0.02, 24, Tm((x, 7.66 - 0.01, 1.45)) @ R(-PI / 2 - 0.18, 'X'), mi=1)
            add_cyl(bm, 0.036, 0.004, 24, Tm((x, 7.645, 1.45)) @ R(-PI / 2 - 0.18, 'X'), mi=2)
        for x in (-0.07, 0.07):                                                 # HUD combiner posts
            cylp(bm, (x, 7.50, 1.60), (x, 7.47, 1.74), 0.008, 8, 1)
        bx(bm, -0.09, 0.09, 7.44, 7.47, 1.66, 1.75, 4, 0.004)                   # combiner glass
    mp = part(C.int_name(rid, "MainPanel"), ["equip", "dark", "screen", "light", "glass"], main_panel, node,
              smooth=None)
    face_n = Vector((0, -0.98, 0.18)).normalized()
    face_up = Vector((0, 0.18, 0.98)).normalized()
    rot_face = wall_rot(face_n, face_up)
    for i, x in enumerate((-0.34, 0.0, 0.34), 1):
        inst(C.int_name(rid, f"PilotMFD{i}"), mfd(), mp, (x, 7.603, 1.25), rot_face)
    inst(C.int_name(rid, "PilotSwitchesL"), switch_strip(8), mp, (-0.34, 7.567, 1.05), rot_face)
    inst(C.int_name(rid, "PilotSwitchesR"), switch_strip(8), mp, (0.34, 7.567, 1.05), rot_face)
    inst(C.int_name(rid, "PilotKeypad"), button_pad(6, 4), mp, (0.0, 7.567, 1.05), rot_face)

    # ---- centre pedestal console + rudder pedals
    def pedestal(bm):
        add_prism(bm, [(7.30, 0.55), (7.75, 0.55), (7.75, 0.95), (7.30, 0.86)], 0.2,
                  Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))), mi=0)
    cp = part(C.int_name(rid, "CenterConsole"), ["equip"], pedestal, node, smooth=None)
    top_n = Vector((0, -0.196, 0.98)).normalized()
    inst(C.int_name(rid, "CenterConsoleKeys"), button_pad(5, 3), cp, (0, 7.52, 0.907),
         wall_rot(top_n, (0, 0.98, 0.196)))
    inst(C.int_name(rid, "CenterConsoleSwitches"), switch_strip(6), cp, (0, 7.30, 0.72),
         wall_rot((0, -1, 0), (0, 0, 1)))
    pp = pedal_pair_mesh()
    inst(C.int_name(rid, "PedalsPilot"), pp, node, (0, 7.60, z0))
    inst(C.int_name(rid, "PedalsEngineer"), pp, node, (0, 5.86, z0))

    # ---- side consoles (pilot full length, engineer short), with keys/switches/small displays
    flat = wall_rot((0, 0, 1), (0, 1, 0))
    up_in = {1: flat, -1: flat}
    for tag, (ya, L, w) in (("Pilot", (6.15, 1.35, 0.30)), ("Eng", (5.25, 0.60, 0.22))):
        me = side_console_mesh(L, w)
        for sx, side in ((-1, "L"), (1, "R")):
            xc = sx * (0.75 - w / 2)
            con = inst(C.int_name(rid, f"SideConsole{tag}_{side}"), me, node, (xc, ya, z0))
            zt = z0 + 0.514
            items = [("Keys", button_pad(4, 3), 0.17), ("Sw1", switch_strip(6), 0.42)]
            if L > 1.0:
                items += [("Sw2", switch_strip(6), 0.62), ("MFD", mfd(0.16, 0.14), 0.92), ("Sw3", switch_strip(6), 1.2)]
            for nm, mesh_, dy in items:
                inst(C.int_name(rid, f"SideConsole{tag}_{side}_{nm}"), mesh_, con, (xc, ya + dy, zt), up_in[sx])

    # ---- engineer instrument panel on the back of the pilot station
    def eng_panel(bm):
        add_prism(bm, [(6.20, 0.55), (6.20, 1.30), (6.05, 1.30), (5.95, 0.80), (6.05, 0.55)], 0.84,
                  Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))), mi=0)
        bx(bm, -0.44, 0.44, 6.03, 6.22, 1.30, 1.33, 1, 0.006)
    ep = part(C.int_name(rid, "EngineerPanel"), ["equip", "dark"], eng_panel, node, smooth=None)
    en = Vector((0, -0.98, 0.196)).normalized()
    rot_e = wall_rot(en, (0, 0.196, 0.98))
    inst(C.int_name(rid, "EngMFD1"), mfd(), ep, (-0.15, 6.004, 1.08), rot_e)
    inst(C.int_name(rid, "EngMFD2"), mfd(), ep, (0.15, 6.004, 1.08), rot_e)
    inst(C.int_name(rid, "EngKeypad"), button_pad(6, 4), ep, (0.0, 5.963, 0.88), rot_e)

    # ---- safety kit
    ext = inst(C.int_name(rid, "FireExtinguisher"), extinguisher(), node, (0.58, 4.475, 0.60), (0, 0, PI))
    ext["kind"] = "fire_extinguisher"
    hh = handhold(0.4)
    inst(C.int_name(rid, "HandholdAft"), hh, node, (-0.17, 4.40, 1.15), wall_rot((0, 1, 0), (0, 0, 1)))
    inst(C.int_name(rid, "HandholdAftR"), hh, node, (0.40, 4.40, 1.25), wall_rot((0, 1, 0), (0, 0, 1)))
    return node


BUILDERS["R01"] = build_R01


# =================================================================== R02 avionics / life support
def rack_unit(variant):
    """19-inch style rack: x -0.275..0.275, back at y=0, front at y=0.48 (+Y), z 0..1.5."""
    W, D, H = 0.55, 0.48, 1.5

    def drawer(bm, z0, h, kind):
        x0, x1 = -W / 2 + 0.035, W / 2 - 0.035
        bx(bm, x0, x1, D - 0.025, D - 0.005, z0 + 0.004, z0 + h - 0.004, 1, 0.004)
        for x in (x0 + 0.03, x1 - 0.03):                                     # handles
            bx(bm, x - 0.008, x + 0.008, D - 0.005, D + 0.025, z0 + h * 0.25, z0 + h * 0.75, 2, 0.003)
        if kind == "vent":
            for k in range(int((h - 0.04) / 0.022)):
                zz = z0 + 0.03 + k * 0.022
                bx(bm, -0.15, 0.15, D - 0.005, D - 0.002, zz, zz + 0.01, 3)
        else:
            bx(bm, -0.18, 0.02, D - 0.005, D - 0.001, z0 + h - 0.05, z0 + h - 0.03, 4)      # status LEDs
            bx(bm, 0.05, 0.17, D - 0.005, D - 0.002, z0 + 0.03, z0 + h - 0.03, 3)            # label/fan grille

    def b(bm):
        for x in (-W / 2, W / 2 - 0.03):
            for y in (0.0, D - 0.03):
                bx(bm, x, x + 0.03, y, y + 0.03, 0, H, 2, 0.004)                # posts
            bx(bm, x + 0.005, x + 0.025, 0.03, D - 0.03, 0.02, H - 0.02, 0)      # side skins
        bx(bm, -W / 2, W / 2, 0, D, 0, 0.04, 0, 0.006)                          # plinth
        bx(bm, -W / 2, W / 2, 0, D, H - 0.03, H, 0, 0.006)                      # top
        bx(bm, -W / 2 + 0.03, W / 2 - 0.03, 0, 0.012, 0.04, H - 0.03, 0)        # back
        z = 0.06
        plan = ([(0.13, "led"), (0.13, "vent"), (0.22, "led"), (0.13, "led"), (0.18, "vent"), (0.13, "led"),
                 (0.13, "led"), (0.22, "vent")] if variant == "A"
                else [(0.13, "led"), (0.22, "vent"), (0.22, None), (0.22, None), (0.22, None), (0.13, "vent")])
        for h, kind in plan:
            if kind is None:                                                    # open bay with slide rails
                for x in (-W / 2 + 0.03, W / 2 - 0.05):
                    bx(bm, x, x + 0.02, 0.03, D - 0.03, z, z + 0.015, 2)
            else:
                drawer(bm, z, h, kind)
            z += h + 0.006
    return lib(f"Rack{variant}", ["equip", "dark", "metal", "dark", "screen"], b, None)


def flight_computer():
    """LRU box, front (+Y) connector panel; x +-0.23, y 0..0.40, z 0..0.19."""
    def b(bm):
        bx(bm, -0.23, 0.23, 0, 0.38, 0, 0.19, 0, 0.008)
        bx(bm, -0.235, 0.235, 0.38, 0.40, -0.005, 0.195, 1, 0.006)
        for k in range(6):
            x = -0.17 + k * 0.055
            Mc = Tm((x, 0.40, 0.095)) @ R(-PI / 2, 'X')
            add_cyl(bm, 0.019, 0.03, 8, Mc @ Tm((0, 0, 0.015)), mi=2)
            add_cyl(bm, 0.012, 0.006, 8, Mc @ Tm((0, 0, 0.033)), mi=1)
        for x in (-0.215, 0.215):
            cylp(bm, (x, 0.40, 0.03), (x, 0.43, 0.03), 0.006, 6, 2)
            cylp(bm, (x, 0.40, 0.16), (x, 0.43, 0.16), 0.006, 6, 2)
            cylp(bm, (x, 0.43, 0.03), (x, 0.43, 0.16), 0.006, 6, 2)
        bx(bm, -0.17, 0.10, 0.40, 0.403, 0.155, 0.175, 3)
    return lib("FlightComputer", ["equip", "dark", "metal", "screen"], b, None)


def gas_tank(length=0.42, r=0.19, key="GasTank"):
    """Horizontal tank along local +Y (valve manifold at +Y end), centre at origin."""
    L = length / 2
    A = R(-PI / 2, 'X')                                    # build along Z, lay along Y

    def b(bm):
        prof = [(0.0, -L - 0.02), (0.11, -L - 0.01), (r * 0.9, -L + 0.03), (r, -L + 0.08),
                (r, L - 0.08), (r * 0.9, L - 0.03), (0.11, L + 0.01), (0.0, L + 0.02)]
        add_revolve(bm, prof, 24, A, mi=0)
        for z in (-L + 0.10, L - 0.10):                                             # bands
            add_revolve(bm, [(r + 0.001, z - 0.018), (r + 0.008, z - 0.018), (r + 0.008, z + 0.018),
                             (r + 0.001, z + 0.018)], 24, A, closed=True, mi=1)
            bx(bm, -0.13, 0.13, z - 0.03, z + 0.03, -r - 0.07, -r + 0.03, 1, 0.006)          # saddle
        add_cyl(bm, 0.035, 0.05, 24, A @ Tm((0, 0, L + 0.04)), mi=1)                      # neck
        bx(bm, -0.04, 0.04, L + 0.06, L + 0.11, -0.03, 0.05, 1, 0.006)                    # valve block
        add_torus(bm, 0.045, 0.008, 16, 6, Tm((0, L + 0.085, 0.10)), mi=2)                # handwheel
        cylp(bm, (0, L + 0.085, 0.05), (0, L + 0.085, 0.10), 0.006, 6, 1)
        add_tube(bm, fillet([(0.04, L + 0.085, 0.0), (0.09, L + 0.085, 0.0), (0.09, L + 0.085, r + 0.06)],
                            0.03, 2), 0.012, 8, mi=1)                                    # outlet
        add_cyl(bm, 0.022, 0.012, 8, Tm((-0.06, L + 0.11, 0.02)) @ R(-PI / 2, 'X'), mi=3)    # gauge
        bx(bm, -0.08, 0.08, -0.05, 0.05, r - 0.001, r + 0.004, 4)                          # label patch
    return lib(key, ["equip", "metal", "red", "dark", "panel"], b, 35)


def scrubber():
    """Vertical CO2 scrubber canister on a skid; origin floor centre."""
    def b(bm):
        bx(bm, -0.13, 0.13, -0.13, 0.13, 0, 0.06, 1, 0.006)
        add_revolve(bm, [(0, 0.06), (0.10, 0.06), (0.12, 0.09), (0.12, 0.92), (0.10, 0.96), (0.05, 0.98),
                         (0, 0.98)], 24, mi=0)
        for z in (0.25, 0.55, 0.85):
            add_revolve(bm, [(0.121, z - 0.015), (0.128, z - 0.015), (0.128, z + 0.015), (0.121, z + 0.015)],
                        24, closed=True, mi=1)
        add_cyl(bm, 0.07, 0.10, 24, Tm((0, 0, 1.03)), mi=1)                         # blower
        add_tube(bm, [(0, 0, 1.08), (0, 0, 1.40)], 0.035, 8, mi=1)                    # outlet duct
        add_tube(bm, fillet([(0.0, -0.12, 0.3), (0.0, -0.16, 0.3), (0.0, -0.16, 1.2), (0.0, -0.05, 1.4)], 0.05, 2),
                 0.022, 8, mi=2)                                                       # inlet hose
        bx(bm, -0.05, 0.05, -0.125, -0.115, 0.55, 0.65, 3)                            # status window
    return lib("Scrubber", ["panel", "metal", "dark", "screen"], b, 35)


def build_R02():
    rid = "R02"
    node = room_node(rid)
    holes = {"N": [(-0.75, -0.27, 0.55, 1.45)], "S": [(0.22, 0.88, -0.35, 1.25)]}

    def ceiling_extra(bm):
        for x, y0_, y1_ in ((-0.42, 2.65, 3.85), (0.42, 2.65, 4.25)):           # tray hangers
            for y in (y0_ + 0.1, y1_ - 0.1):
                for dx in (-0.11, 0.11):
                    cylp(bm, (x + dx, y, 1.40), (x + dx, y, 1.5), 0.006, 6, 2)
    ceil = room_shell(rid, holes, {"F": [(-0.6, 0.6, 2.6, 4.3)]}, ceiling_extra=ceiling_extra)
    lay_grating(rid, node, -0.6, 0.6, 2.6, 4.3, -0.5, 2, 3)
    inst(C.int_name(rid, "CableTrayPort"), cable_tray(1.2), ceil, (-0.42, 2.65, 1.40))
    inst(C.int_name(rid, "CableTrayStbd"), cable_tray(1.6), ceil, (0.42, 2.65, 1.40))

    # racks: port bank (A, B) facing +X, starboard (A) facing -X
    ra, rb = rack_unit("A"), rack_unit("B")
    rot_p, rot_s = (0, 0, -PI / 2), (0, 0, PI / 2)
    inst(C.int_name(rid, "RackPort1"), ra, node, (-1.1, 3.275, -0.5), rot_p)
    rk = inst(C.int_name(rid, "RackPort2"), rb, node, (-1.1, 3.825, -0.5), rot_p)
    inst(C.int_name(rid, "RackStbd1"), ra, node, (1.1, 3.575, -0.5), rot_s)
    fc = flight_computer()
    z = 0.06 + 0.13 + 0.006 + 0.22 + 0.006 + 0.015
    for i in range(3):
        inst(C.int_name(rid, f"FlightComputer{i + 1}"), fc, rk, (0, 0.05, z + i * 0.226), world=False)

    def cables(bm):
        for y in (3.275, 3.825):
            for k, dy in enumerate((-0.06, 0.0, 0.06)):
                add_tube(bm, fillet([(-0.70, y + dy, 0.99), (-0.60, y + dy, 1.02), (-0.50, y + dy, 1.34),
                                     (-0.42 + dy, min(y + dy, 3.8), 1.43)], 0.06, 3), 0.014, 6, mi=k % 2)
        for k, dy in enumerate((-0.05, 0.05)):
            add_tube(bm, fillet([(0.70, 3.575 + dy, 0.99), (0.60, 3.575 + dy, 1.02), (0.50, 3.575 + dy, 1.34),
                                 (0.42 + dy, 3.575 + dy, 1.43)], 0.06, 3), 0.016, 6, mi=k % 2)
        for k, dx in enumerate((-0.04, 0.0, 0.04)):                                 # breaker panel feeds
            add_tube(bm, fillet([(-0.5 + dx, 2.64, 1.30), (-0.5 + dx, 2.64, 1.43), (-0.42 + dx, 2.75, 1.43)],
                                0.05, 2), 0.012, 6, mi=k % 2)
    part(C.int_name(rid, "CableBundles"), ["dark", "red"], cables, node, smooth=40)

    # tanks above the racks, scrubbers forward-starboard
    gt = gas_tank()
    for nm, loc, rz in (("TankO2", (-0.85, 3.24, 1.24), 0.0), ("TankN2", (-0.85, 3.80, 1.24), 0.0),
                        ("TankWater", (0.85, 3.575, 1.24), PI)):
        t = inst(C.int_name(rid, nm), gt, node, loc, (0, 0, rz))
        t["contents"] = nm[4:]
    sc = scrubber()
    inst(C.int_name(rid, "Scrubber1"), sc, node, (0.70, 4.10, -0.5))
    inst(C.int_name(rid, "Scrubber2"), sc, node, (0.96, 4.10, -0.5))

    # breaker panel on the aft bulkhead (port of the airlock hatch)
    def bp(bm):
        bx(bm, -1.05, -0.10, 2.60, 2.66, 0.0, 1.30, 0, 0.01)
        bx(bm, -1.02, -0.13, 2.66, 2.665, 1.20, 1.27, 1)
        bx(bm, -0.20, -0.14, 2.66, 2.72, 0.10, 0.30, 2, 0.006)                  # main isolator
        cylp(bm, (-0.17, 2.72, 0.20), (-0.17, 2.80, 0.20), 0.015, 8, 3)
    bpo = part(C.int_name(rid, "BreakerPanel"), ["equip", "panel", "dark", "red"], bp, node, smooth=None)
    rot_b = wall_rot((0, 1, 0), (0, 0, 1))
    k = 0
    for x in (-0.82, -0.46):
        for j in range(6):
            k += 1
            inst(C.int_name(rid, f"Breakers{k:02d}"), breaker_row(10), bpo, (x, 2.66, 0.35 + j * 0.14), rot_b)

    # ladder up into the cockpit door, ECLSS control panel, handholds
    inst(C.int_name(rid, "LadderR01"), ladder(1.10, 0.36, 0.12), node, (-0.51, 4.30, -0.5), (0, 0, PI))

    def eclss(bm):
        bx(bm, 0.10, 0.50, 4.24, 4.30, 0.55, 1.20, 0, 0.008)
    ep = part(C.int_name(rid, "ECLSSPanel"), ["equip"], eclss, node, smooth=None)
    # task-5 audit: fire extinguisher on the forward wall between the R01 ladder handhold and the ECLSS panel
    ex = inst(C.int_name(rid, "FireExtinguisher"), extinguisher(), node, (-0.01, 4.225, -0.30))
    ex["kind"] = "fire_extinguisher"
    rot_f = wall_rot((0, -1, 0), (0, 0, 1))
    inst(C.int_name(rid, "ECLSSDisplay"), mfd(), ep, (0.30, 4.24, 1.0), rot_f)
    inst(C.int_name(rid, "ECLSSSwitches"), switch_strip(8), ep, (0.30, 4.24, 0.68), rot_f)
    hh = handhold(0.4)
    inst(C.int_name(rid, "HandholdAft"), hh, node, (0.05, 2.60, 0.9), wall_rot((0, 1, 0), (0, 0, 1)))
    inst(C.int_name(rid, "HandholdFwd"), hh, node, (-0.15, 4.30, 0.95), wall_rot((0, -1, 0), (0, 0, 1)))
    return node


BUILDERS["R02"] = build_R02


# =================================================================== R03 vestibule + R03a airlock chamber
def add_arc(bm, prof, a0, a1, segs, M=None, mi=0):
    """Partial surface of revolution about local Z (angles a0..a1, radians)."""
    M = M or Matrix.Identity(4)
    rings = []
    for k in range(segs + 1):
        a = a0 + (a1 - a0) * k / segs
        rings.append([bm.verts.new(M @ Vector((r * math.cos(a), r * math.sin(a), z))) for r, z in prof])
    for k in range(segs):
        for i in range(len(prof) - 1):
            f = bm.faces.new((rings[k][i], rings[k + 1][i], rings[k + 1][i + 1], rings[k][i + 1]))
            f.material_index = mi


def eva_suit():
    """Simplified EVA suit on its stand: origin = floor under the stand, facing +Y.
    Local Y: stand post -0.31, backpack -0.27..-0.09, torso -0.11..0.13, chest unit to 0.17."""
    def b(bm):
        bx(bm, -0.22, 0.22, -0.34, 0.18, 0, 0.03, 4, 0.008)                     # stand base
        bx(bm, -0.025, 0.025, -0.335, -0.285, 0.03, 1.55, 4, 0.006)               # post
        bx(bm, -0.06, 0.06, -0.31, -0.25, 1.30, 1.36, 4, 0.006)                   # PLSS bracket
        for sx in (-1, 1):
            x = sx * 0.11
            bx(bm, x - 0.06, x + 0.06, -0.10, 0.18, 0.03, 0.15, 0, 0.03, 2)       # boots
            add_tube(bm, [(x, 0.02, 0.13), (x, 0.04, 0.52)], 0.068, 24, mi=0)     # shin
            add_tube(bm, [(x, 0.04, 0.50), (x * 1.05, 0.0, 0.92)], 0.085, 24, mi=0)   # thigh
            bx(bm, x - 0.05, x + 0.05, 0.08, 0.13, 0.44, 0.58, 1, 0.015)          # knee pad
            xs = sx * 0.25
            add_revolve(bm, [(0.075, -0.03), (0.085, -0.03), (0.085, 0.03), (0.075, 0.03)], 24,
                        Tm((xs, 0.01, 1.40)) @ R(PI / 2, 'Y'), closed=True, mi=2)          # shoulder bearing
            add_tube(bm, [(xs + sx * 0.02, 0.01, 1.40), (sx * 0.30, 0.05, 1.14)], 0.062, 24, mi=0)
            add_tube(bm, [(sx * 0.30, 0.05, 1.15), (sx * 0.27, 0.17, 0.97)], 0.055, 24, mi=0)
            add_revolve(bm, [(0.05, -0.012), (0.062, -0.012), (0.062, 0.012), (0.05, 0.012)], 24,
                        Tm((sx * 0.268, 0.18, 0.955)) @ align_z((0, 0.7, -1)), closed=True, mi=2)  # wrist ring
            bx(bm, -0.035, 0.035, -0.05, 0.05, -0.12, 0.0, 1, 0.025, 2,
               M=Tm((sx * 0.265, 0.20, 0.94)) @ R(0.5, 'X'))                       # glove
        bx(bm, -0.19, 0.19, -0.11, 0.13, 0.86, 1.06, 0, 0.05, 2)                  # brief / hips
        add_torus(bm, 0.175, 0.02, 24, 6, Tm((0, 0.01, 1.075)), mi=2)             # waist ring
        bx(bm, -0.22, 0.22, -0.11, 0.13, 1.08, 1.47, 0, 0.08, 2)                  # hard upper torso
        bx(bm, -0.11, 0.11, 0.12, 0.17, 1.16, 1.28, 1, 0.012)                     # chest control unit
        for k, x in enumerate((-0.06, 0.0, 0.06)):
            add_cyl(bm, 0.012, 0.015, 8, Tm((x, 0.175, 1.25)) @ R(-PI / 2, 'X'), mi=3 if k == 1 else 2)
        bx(bm, -0.20, 0.20, -0.27, -0.09, 1.00, 1.55, 0, 0.03, 2)                 # PLSS backpack
        bx(bm, -0.16, 0.16, -0.275, -0.265, 1.08, 1.45, 1, 0.004)                 # PLSS access panel
        add_revolve(bm, [(0.13, 1.47), (0.15, 1.47), (0.15, 1.52), (0.13, 1.52)], 24, closed=True, mi=2)  # neck ring
        add_revolve(bm, [(0.145, 1.52), (0.158, 1.58), (0.155, 1.67), (0.135, 1.75), (0.085, 1.80), (0.0, 1.815)],
                    24, mi=0)                                                     # helmet shell
        add_arc(bm, [(0.163, 1.565), (0.168, 1.62), (0.163, 1.69), (0.14, 1.75)],
                math.radians(30), math.radians(150), 12, mi=5)                    # gold sun visor
        for sx in (-1, 1):                                                        # helmet lights
            bx(bm, sx * 0.13 - 0.025, sx * 0.13 + 0.025, 0.02, 0.10, 1.72, 1.76, 1, 0.006)
    return lib("EVASuit", ["panel", "equip", "metal", "red", "dark", "foil"], b, 35)


def octagon_trunk(bm, hw, hh, ch, t, a0, a1, axis, center, mi=0, flange=None):
    """Thin-walled chamfered-rectangle tube along `axis` ('X' or 'Z') from a0 to a1.
    center = the two cross-section coordinates ((y, z) for X, (x, y) for Z)."""
    def pt(a, u, v):
        if axis == 'X':
            return Vector((a, center[0] + u, center[1] + v))
        return Vector((center[0] + u, center[1] + v, a))
    outer = [(u, v) for u, v in C.octagon_ring(hw + t, hh + t, ch + t * 0.4)]
    inner = [(u, v) for u, v in C.octagon_ring(hw, hh, ch)]
    rings = [[bm.verts.new(pt(a, u, v)) for u, v in ring] for a in (a0, a1) for ring in (outer, inner)]
    oa, ia, ob, ib = rings
    n = 8
    for i in range(n):
        j = (i + 1) % n
        for f in (bm.faces.new((oa[i], oa[j], ob[j], ob[i])), bm.faces.new((ia[j], ia[i], ib[i], ib[j])),
                  bm.faces.new((oa[j], oa[i], ia[i], ia[j])), bm.faces.new((ob[i], ob[j], ib[j], ib[i]))):
            f.material_index = mi
    if flange is not None:                                    # trim plate (a, thickness, margin)
        fa, ft, fm = flange
        fo = [(u, v) for u, v in C.octagon_ring(hw + fm, hh + fm, ch + fm * 0.4)]
        r = [[bm.verts.new(pt(a, u, v)) for u, v in ring] for a in (fa, fa + ft) for ring in (fo, inner)]
        fa_o, fa_i, fb_o, fb_i = r
        for i in range(n):
            j = (i + 1) % n
            for f in (bm.faces.new((fa_o[i], fa_o[j], fb_o[j], fb_o[i])), bm.faces.new((fa_i[j], fa_i[i], fb_i[i], fb_i[j])),
                      bm.faces.new((fa_o[j], fa_o[i], fa_i[i], fa_i[j])), bm.faces.new((fb_o[i], fb_o[j], fb_i[j], fb_i[i]))):
                f.material_index = mi


def pressure_panel():
    """Hatch-side pressure equalisation panel: gauge, valve wheel, red lamp; faces +Z, up +Y."""
    def b(bm):
        bx(bm, -0.16, 0.16, -0.065, 0.065, -0.02, 0, 0, 0.006)
        add_cyl(bm, 0.045, 0.02, 24, Tm((-0.08, 0, 0.01)), mi=1)
        add_cyl(bm, 0.038, 0.004, 24, Tm((-0.08, 0, 0.021)), mi=2)
        bx(bm, -0.082, -0.078, -0.002, 0.03, 0.022, 0.025, 1)                       # needle
        add_torus(bm, 0.04, 0.007, 16, 6, Tm((0.05, 0, 0.05)), mi=3)
        add_cyl(bm, 0.012, 0.05, 8, Tm((0.05, 0, 0.025)), mi=1)
        bx(bm, 0.115, 0.145, -0.02, 0.02, 0, 0.02, 4, 0.004)                       # warning lamp
    return lib("PressurePanel", ["equip", "dark", "panel", "rail", "red"], b, None)


def tool_rack(bm, x, y0, y1, z0, z1):
    """Pegboard on a port-side wall face at x (facing +X) with hung tools."""
    bx(bm, x, x + 0.02, y0, y1, z0, z1, 0, 0.006)
    for k in range(int((y1 - y0) / 0.08)):                                       # peg rows
        for zz in (z0 + 0.25, z0 + 0.62):
            cylp(bm, (x + 0.02, y0 + 0.05 + k * 0.08, zz), (x + 0.07, y0 + 0.05 + k * 0.08, zz + 0.01), 0.004, 6, 2)
    yc = (y0 + y1) / 2
    X = x + 0.04
    bx(bm, X - 0.008, X + 0.008, y0 + 0.06, y0 + 0.08, z0 + 0.36, z0 + 0.60, 2, 0.003)       # wrench handle
    add_torus(bm, 0.022, 0.007, 8, 4, Tm((X, y0 + 0.07, z0 + 0.62)) @ R(PI / 2, 'Y'), mi=2)
    cylp(bm, (X, y0 + 0.16, z0 + 0.38), (X, y0 + 0.16, z0 + 0.50), 0.014, 8, 3)                # screwdriver
    cylp(bm, (X, y0 + 0.16, z0 + 0.50), (X, y0 + 0.16, z0 + 0.62), 0.004, 6, 2)
    for dy in (-0.012, 0.012):                                                                  # pliers
        cylp(bm, (X, yc + dy * 2, z0 + 0.36), (X, yc + dy * 0.4, z0 + 0.54), 0.006, 6, 3)
    bx(bm, X - 0.008, X + 0.008, yc - 0.012, yc + 0.012, z0 + 0.54, z0 + 0.6, 2, 0.003)
    cylp(bm, (X, y1 - 0.12, z0 + 0.30), (X, y1 - 0.12, z0 + 0.58), 0.012, 8, 1)                # hammer
    bx(bm, X - 0.02, X + 0.02, y1 - 0.18, y1 - 0.06, z0 + 0.58, z0 + 0.62, 2, 0.005)
    bx(bm, X - 0.025, X + 0.025, y0 + 0.08, y0 + 0.22, z0 + 0.72, z0 + 0.82, 4, 0.01)          # EVA power tool
    bx(bm, X - 0.02, X + 0.02, y0 + 0.09, y0 + 0.13, z0 + 0.64, z0 + 0.74, 4, 0.008)
    cylp(bm, (X, y0 + 0.22, z0 + 0.77), (X, y0 + 0.27, z0 + 0.77), 0.008, 8, 2)
    cylp(bm, (X, yc + 0.05, z0 + 0.66), (X, yc + 0.05, z0 + 0.88), 0.009, 8, 2)                # torque wrench
    bx(bm, X - 0.012, X + 0.012, yc + 0.03, yc + 0.07, z0 + 0.86, z0 + 0.9, 3, 0.004)
    add_torus(bm, 0.05, 0.01, 12, 6, Tm((X + 0.01, y1 - 0.12, z0 + 0.78)) @ R(PI / 2, 'Y'), mi=4)  # tether coil


def build_R03():
    # ---------------- R03 vestibule
    rid = "R03"
    node = room_node(rid)
    room_shell(rid, {"N": [(0.22, 0.88, -0.35, 1.25)], "S": [(0.22, 0.88, -0.45, 1.30)],
                     "W": [(1.25, 1.91, -0.35, 1.25)]})
    doorway(C.int_name(rid, "DoorwayR02"), node, "Y", 2.50, 2.60, 0.22, 0.88, -0.35, 1.25)
    hatch = door_leaf_mesh("Hatch_066x160", 0.66, 1.58, 0.06, wheel=True, chamfer=0.10)
    door(C.int_name(rid, "InnerHatch"), node, hatch, (0.87, 2.63, 1.24), (0, 0, -1), (-1, 0, 0), (0, 1, 0), 90,
         "R03<->R02", "airlock inner hatch; opens into R02 against the starboard rack end")
    sm = eva_suit()
    for i, y in enumerate((1.0, 2.0), 1):
        s = inst(C.int_name(rid, f"EVASuit{i}"), sm, node, (0.765, y, -0.5), (0, 0, PI / 2))
        s["kind"] = "eva_suit_on_stand"
    part(C.int_name(rid, "ToolRack"), ["panel", "dark", "metal", "red", "rail"],
         lambda bm: tool_rack(bm, -0.05, 1.97, 2.45, 0.30, 1.25), node, smooth=None)
    inst(C.int_name(rid, "SuitLockers"), locker(0.5, 1.95, 0.25, 2), node, (-0.05, 0.93, -0.5), (0, 0, -PI / 2))

    def service(bm):                                    # suit umbilical service panel + hose coils
        bx(bm, 1.04, 1.10, 1.36, 1.64, 0.35, 0.85, 0, 0.008)
        for k, y in enumerate((1.43, 1.57)):
            cylp(bm, (1.04, y, 0.75), (1.0, y, 0.75), 0.02, 8, 1)
            add_torus(bm, 0.08, 0.012, 16, 6, Tm((1.02, y, 0.52)) @ R(PI / 2, 'Y'), mi=2)
            cylp(bm, (1.0, y, 0.75), (1.02, y, 0.60), 0.012, 8, 2)
        bx(bm, 1.035, 1.04, 1.40, 1.60, 0.40, 0.48, 3)
    part(C.int_name(rid, "SuitServicePanel"), ["equip", "metal", "dark", "screen"], service, node, smooth=40)
    hh = handhold(0.4)
    for nm, loc, n in (("HandholdFwd", (0.12, 2.5, 0.75), (0, -1, 0)), ("HandholdAft", (0.12, 0.6, 0.75), (0, 1, 0)),
                       ("HandholdPort", (-0.05, 1.08 + 0.53, 1.38), (1, 0, 0))):
        inst(C.int_name(rid, nm), hh, node, loc, wall_rot(n, (0, 0, 1) if n[0] == 0 else (0, 1, 0)))
    pp = pressure_panel()
    inst(C.int_name(rid, "PressurePanelR03"), pp, node, (0.30, 2.5, 1.38),
         wall_rot((0, -1, 0), (0, 0, 1)))
    # task-5 audit: fire extinguisher on the port wall between the suit lockers and the tool rack
    ex = inst(C.int_name(rid, "FireExtinguisher"), extinguisher(), node, (0.025, 1.575, -0.30), (0, 0, PI / 2))
    ex["kind"] = "fire_extinguisher"

    # ---------------- R03a airlock chamber
    rid = "R03a"
    na = room_node(rid)
    room_shell(rid, {"E": [(1.25, 1.91, -0.35, 1.25)], "W": [(1.10, 1.90, -0.25, 0.75)],
                     "C": [(-0.77, -0.17, 1.25, 1.85)]})
    doorway(C.int_name(rid, "DoorwayR03"), na, "X", -0.15, -0.05, 1.25, 1.91, -0.35, 1.25)
    door(C.int_name(rid, "ChamberHatch"), na, hatch, (-0.02, 1.26, 1.24), (0, 0, -1), (0, 1, 0), (1, 0, 0), 75,
         "R03a<->R03", "airlock chamber hatch; opens into the vestibule")
    part(C.int_name(rid, "OuterTrunk"), ["metal"], lambda bm: octagon_trunk(
        bm, 0.40, 0.50, 0.15, 0.03, -1.73, -1.10, 'X', (1.50, 0.25), 0, (-1.10, 0.012, 0.09)), na, smooth=None)
    oh = door_leaf_mesh("Hatch_080x100", 0.80, 1.0, 0.06, wheel=True, chamfer=0.15)
    door(C.int_name(rid, "OuterHatch"), na, oh, (-1.075, 1.11, 0.75), (0, 0, -1), (0, 1, 0), (1, 0, 0), 90,
         "R03a<->outside (crew hatch trunk, SOCKET_Hatch_Crew)",
         "outer pressure hatch at the inboard end of the crew-hatch trunk; skin door = EXT_HATCH_Crew")
    part(C.int_name(rid, "DockTrunk"), ["metal"], lambda bm: octagon_trunk(
        bm, 0.30, 0.30, 0.10, 0.03, 1.55, 1.835, 'Z', (-0.47, 1.55), 0, (1.485, 0.015, 0.07)), na, smooth=None)
    ch = door_leaf_mesh("Hatch_060x060", 0.60, 0.60, 0.05, wheel=True, chamfer=0.10)
    door(C.int_name(rid, "CeilingHatch"), na, ch, (-0.17, 1.27, 1.47), (-1, 0, 0), (0, 1, 0), (0, 0, -1), 95,
         "R03a<->dorsal dock (SOCKET_Dock_Dorsal)", "ceiling hatch up the dock trunk; hangs open into the chamber")
    inst(C.int_name(rid, "LadderDock"), ladder(1.95, 0.40, 0.15), na, (-0.47, 2.5, -0.5), (0, 0, PI))

    def drain(bm):
        add_revolve(bm, [(0.08, -0.5), (0.11, -0.5), (0.11, -0.494), (0.08, -0.494)], 24,
                    Tm((-0.62, 1.55, 0)), closed=True, mi=0)
        for k in range(5):
            y = 1.55 - 0.06 + k * 0.03
            bx(bm, -0.70, -0.54, y - 0.005, y + 0.005, -0.5, -0.496, 1)
        add_cyl(bm, 0.08, 0.004, 24, Tm((-0.62, 1.55, -0.502)), mi=1)
    part(C.int_name(rid, "FloorDrain"), ["metal", "dark"], drain, na, smooth=None)
    hh = handhold(0.4)
    for nm, loc, n, al in (("HandholdPortFwd", (-1.1, 2.15, 0.25), (1, 0, 0), (0, 0, 1)),
                           ("HandholdPortAft", (-1.1, 0.85, 0.25), (1, 0, 0), (0, 0, 1)),
                           ("HandholdOverhead", (-0.62, 0.9, 1.5), (0, 0, -1), (1, 0, 0)),
                           ("HandholdStbd", (-0.15, 2.2, 0.6), (-1, 0, 0), (0, 0, 1))):
        inst(C.int_name(rid, nm), hh, na, loc, wall_rot(n, al))
    inst(C.int_name(rid, "PressurePanel"), pp, na, (-0.15, 2.18, 1.1), wall_rot((-1, 0, 0), (0, 0, 1)))
    inst(C.int_name(rid, "EVAEquipLocker"), locker(0.8, 0.6, 0.25, 2), na, (-0.62, 0.6, 0.55))

    def lamp(bm):
        bx(bm, -1.10, -1.07, 1.42, 1.58, 0.92, 0.98, 0, 0.006)
        bx(bm, -1.07, -1.06, 1.44, 1.56, 0.93, 0.97, 1)
    part(C.int_name(rid, "HatchWarningLamp"), ["equip", "red"], lamp, na, smooth=None)
    return node


BUILDERS["R03"] = build_R03


# =================================================================== R04 crew quarters
YZX = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))     # prism: polygon (y, z), extruded along X


def curtain_bunch(bm, x, y0, y1, z0, z1, depth=0.05, folds=5, mi=0, axis="Y"):
    """Bunched-up curtain: zig-zag pleats spanning y0..y1 (or x0..x1 when axis='X'),
    centred on x (resp. y), hanging z0..z1."""
    n = folds * 2
    pts = []
    for k in range(n + 1):
        t = k / n
        pts.append((y0 + (y1 - y0) * t, depth / 2 * (1 if k % 2 else -1)))
    poly = pts + [(p[0], p[1] + 0.012) for p in reversed(pts)]
    poly = [(u, v) for (u, v) in poly]
    if axis == "Y":   # polygon (y, x) -> world
        M = Matrix(((0, 1, 0, x), (1, 0, 0, 0), (0, 0, 1, z0), (0, 0, 0, 1)))
    else:             # polygon (x, y)
        M = Matrix(((1, 0, 0, 0), (0, 1, 0, x), (0, 0, 1, z0), (0, 0, 0, 1)))
    add_prism(bm, poly, z1 - z0, M, mi=mi, z0=0.0)


def berth():
    """One sleeping berth: wall side at x=0, inboard edge x=0.70, y -0.975..0.975 (head +Y),
    pan bottom z=0.  Mattress, sleeping bag with restraint straps, pillow, lee rail,
    reading light, shelf, curtain rail with the curtain bunched at the foot."""
    def b(bm):
        bx(bm, 0.0, 0.70, -0.975, 0.975, 0.0, 0.04, 0, 0.006)                    # pan
        bx(bm, 0.0, 0.70, -0.975, -0.955, 0.04, 0.20, 0, 0.004)                  # foot board
        bx(bm, 0.0, 0.70, 0.955, 0.975, 0.04, 0.24, 0, 0.004)                    # head board
        bx(bm, 0.02, 0.68, -0.95, 0.95, 0.04, 0.13, 4, 0.035, 2)                 # mattress
        bx(bm, 0.12, 0.58, 0.64, 0.92, 0.13, 0.20, 4, 0.03, 2)                   # pillow
        bx(bm, 0.07, 0.64, -0.92, 0.56, 0.13, 0.21, 1, 0.04, 2)                  # sleeping bag
        bx(bm, 0.07, 0.64, 0.50, 0.62, 0.13, 0.225, 1, 0.045, 2)                 # bag collar
        for y in (-0.55, -0.05, 0.40):                                           # restraint straps
            strap(bm, [(0.03, y, 0.13), (0.08, y, 0.218), (0.63, y, 0.218), (0.68, y, 0.13)],
                  w=0.04, t=0.005, side=(0, 1, 0), mi=3)
            bx(bm, 0.33, 0.39, y - 0.03, y + 0.03, 0.214, 0.226, 2, 0.003)       # buckle
        for y in (-0.93, 0.93):                                                   # lee rail posts + rail
            cylp(bm, (0.68, y, 0.04), (0.68, y, 0.26), 0.012, 8, 2)
        cylp(bm, (0.68, -0.93, 0.26), (0.68, 0.93, 0.26), 0.012, 8, 2)
        bx(bm, 0.0, 0.025, 0.68, 0.84, 0.42, 0.50, 0, 0.006)                     # reading light housing
        bx(bm, 0.025, 0.03, 0.70, 0.82, 0.43, 0.49, 5)
        bx(bm, 0.0, 0.14, -0.60, -0.10, 0.46, 0.48, 0, 0.004)                    # personal shelf
        bx(bm, 0.12, 0.14, -0.60, -0.10, 0.48, 0.52, 0, 0.003)
        bx(bm, 0.03, 0.10, -0.55, -0.40, 0.48, 0.58, 3, 0.01)                    # stowed kit on the shelf
        bx(bm, 0.03, 0.11, -0.34, -0.22, 0.48, 0.53, 1, 0.01)
        cylp(bm, (0.72, -0.975, 0.70), (0.72, 0.975, 0.70), 0.008, 6, 2)         # curtain rail
        curtain_bunch(bm, 0.72, -0.95, -0.70, 0.12, 0.69, 0.05, 5, mi=1)
    return lib("Berth", ["equip", "fabric", "metal", "dark", "panel", "light"], b, 35)


GALLEY_DY = -0.295


def galley(bm):
    """Galley module on the starboard wall, built at y -0.55..0.11 and shifted by GALLEY_DY
    (final y -0.845..-0.185, clear of the open forward door)."""
    xa, xw, ya, yb = 0.72, 1.10, -0.55, 0.11
    bx(bm, xa + 0.03, xw, ya, yb, -0.50, 0.38, 0, 0.008)                         # base cabinet
    bx(bm, xa + 0.04, xw, ya + 0.01, yb - 0.01, -0.50, -0.44, 3)                 # toe kick
    for k, (za, zb) in enumerate(((-0.42, -0.12), (-0.10, 0.12), (0.14, 0.36))):    # drawers
        for (y0_, y1_) in ((ya + 0.01, (ya + yb) / 2 - 0.004), ((ya + yb) / 2 + 0.004, yb - 0.01)):
            bx(bm, xa + 0.015, xa + 0.03, y0_, y1_, za, zb, 1, 0.005)
            yc = (y0_ + y1_) / 2
            bx(bm, xa, xa + 0.015, yc - 0.07, yc + 0.07, zb - 0.05, zb - 0.03, 2, 0.004)   # pull handle
    bx(bm, xa - 0.01, xw, ya - 0.01, yb + 0.01, 0.38, 0.42, 1, 0.006)            # worktop
    bx(bm, xa - 0.012, xa + 0.005, ya, yb, 0.42, 0.45, 2, 0.003)                 # fiddle rail
    # food warmer (oven) with window and keypad, wall mounted above the worktop
    bx(bm, 0.80, xw, ya + 0.02, -0.15, 0.55, 0.86, 1, 0.012)
    bx(bm, 0.795, 0.80, ya + 0.06, -0.30, 0.60, 0.81, 4)                        # window
    bx(bm, 0.795, 0.80, -0.27, -0.18, 0.72, 0.82, 5)                            # display
    for k in range(3):
        bx(bm, 0.792, 0.80, -0.27 + k * 0.03, -0.25 + k * 0.03, 0.62, 0.65, 2)
    bx(bm, 0.785, 0.80, ya + 0.08, -0.32, 0.61, 0.625, 2, 0.004)                 # handle
    # water dispenser: hot/cold nozzles over a drip tray
    bx(bm, 0.86, xw, -0.10, 0.10, 0.50, 0.95, 0, 0.01)
    bx(bm, 0.855, 0.86, -0.08, 0.08, 0.82, 0.92, 4)
    for k, y in enumerate((-0.04, 0.04)):
        cylp(bm, (0.86, y, 0.72), (0.80, y, 0.72), 0.012, 8, 2)
        cylp(bm, (0.80, y, 0.72), (0.80, y, 0.66), 0.009, 8, 2)
        bx(bm, 0.845, 0.86, y - 0.015, y + 0.015, 0.76, 0.79, 6 if k else 7, 0.003)
    bx(bm, 0.76, 0.86, -0.08, 0.08, 0.42, 0.44, 3, 0.004)                        # drip tray
    # drink / food pouches held under an elastic strap on the worktop
    for k in range(5):
        y = ya + 0.10 + k * 0.05
        bx(bm, 0.93, 1.06, y - 0.018, y + 0.018, 0.42, 0.58, 8 if k % 2 else 9, 0.012)
    strap(bm, [(0.92, ya + 0.05, 0.42), (0.92, ya + 0.06, 0.50), (0.92, ya + 0.34, 0.50), (0.92, ya + 0.35, 0.42)],
          w=0.02, t=0.004, side=(0, 0, 1), mi=3)
    # overhead cabinet
    bx(bm, 0.78, xw, ya, yb, 1.06, 1.46, 0, 0.008)
    for (y0_, y1_) in ((ya + 0.01, (ya + yb) / 2 - 0.004), ((ya + yb) / 2 + 0.004, yb - 0.01)):
        bx(bm, 0.765, 0.78, y0_, y1_, 1.07, 1.45, 1, 0.005)
        yc = (y0_ + y1_) / 2
        bx(bm, 0.75, 0.765, yc - 0.06, yc + 0.06, 1.09, 1.11, 2, 0.004)
    bx(bm, 0.78, 0.80, ya + 0.02, yb - 0.02, 1.04, 1.06, 5)                      # under-cabinet light strip
    # wash / rehydration sink let into the worktop (between the pouches and the drip tray, under the
    # oven's front edge): raised rim, dark basin plate, drain, short spout + mixer lever from the back
    sx0, sx1, sy0, sy1 = 0.745, 0.915, -0.245, -0.115
    for (a, b_, c, d) in ((sx0, sx1, sy0, sy0 + 0.015), (sx0, sx1, sy1 - 0.015, sy1),
                          (sx0, sx0 + 0.015, sy0, sy1), (sx1 - 0.015, sx1, sy0, sy1)):
        bx(bm, a, b_, c, d, 0.42, 0.432, 2, 0.003)                               # rim
    bx(bm, sx0 + 0.015, sx1 - 0.015, sy0 + 0.015, sy1 - 0.015, 0.42, 0.4215, 3)  # basin
    add_cyl(bm, 0.014, 0.003, 12, Tm(((sx0 + sx1) / 2, (sy0 + sy1) / 2, 0.4225)), mi=2)   # drain
    add_tube(bm, fillet([(0.945, -0.18, 0.42), (0.945, -0.18, 0.51), (0.875, -0.18, 0.51),
                         (0.875, -0.18, 0.47)], 0.025, 3), 0.008, 8, mi=2)       # spout (below the oven)
    cylp(bm, (0.945, -0.15, 0.42), (0.945, -0.15, 0.47), 0.012, 8, 2)            # mixer
    bx(bm, 0.94, 0.95, -0.15, -0.125, 0.465, 0.475, 2, 0.002)
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0.0, GALLEY_DY, 0.0))


def hygiene(bm):
    """Hygiene cubicle in the starboard-aft corner (x 0.35..1.10, y -1.60..-0.85): panel walls with
    an open doorway (x 0.35 side, y -1.47..-0.98), zero-g toilet, hand-wash unit, mirror, towel."""
    xa, ya, yb = 0.35, -1.60, -0.85
    t = 0.035
    bx(bm, xa, 1.10, yb - t, yb, -0.50, 1.50, 0, 0.006)                           # forward partition
    bx(bm, xa, xa + t, ya, -1.47, -0.50, 1.50, 0, 0.006)                          # inboard partition (aft of door)
    bx(bm, xa, xa + t, -0.98, yb - t, -0.50, 1.50, 0, 0.006)                      # (fwd of door)
    bx(bm, xa, xa + t, -1.47, -0.98, 1.30, 1.50, 0, 0.006)                        # header
    for y0_ in (-1.47, -0.98):                                                    # jamb trims
        bx(bm, xa - 0.012, xa + t + 0.012, y0_ - 0.02, y0_ + 0.02, -0.50, 1.32, 1, 0.005)
    bx(bm, xa - 0.012, xa + t + 0.012, -1.49, -0.96, 1.30, 1.34, 1, 0.005)
    cylp(bm, (xa - 0.03, -1.50, 1.27), (xa - 0.03, -0.95, 1.27), 0.008, 6, 2)     # curtain rail
    curtain_bunch(bm, xa - 0.03, -1.12, -0.97, -0.40, 1.26, 0.05, 4, mi=3)
    # zero-g toilet against the aft wall
    x, y = 0.80, -1.38
    bx(bm, x - 0.17, x + 0.17, y - 0.20, y + 0.12, -0.50, -0.12, 4, 0.03, 2)      # body
    add_revolve(bm, [(0.10, -0.12), (0.16, -0.12), (0.17, -0.09), (0.15, -0.07), (0.10, -0.075)], 24,
                Tm((x, y, 0)), mi=1)                                              # seat ring
    for sx in (-1, 1):                                                            # thigh bars
        cylp(bm, (x + sx * 0.20, y - 0.12, -0.12), (x + sx * 0.20, y - 0.12, 0.02), 0.012, 8, 2)
        cylp(bm, (x + sx * 0.20, y - 0.12, 0.02), (x + sx * 0.20, y + 0.12, 0.02), 0.012, 8, 2)
    add_tube(bm, fillet([(x + 0.17, y - 0.10, -0.25), (0.99, y - 0.10, -0.20), (1.04, y - 0.10, 0.30),
                         (1.04, y + 0.10, 0.45)], 0.05, 2), 0.022, 8, mi=5)        # urine hose
    add_revolve(bm, [(0.0, 0.0), (0.02, 0.0), (0.045, 0.09), (0.04, 0.10)], 16,
                Tm((1.04, y + 0.10, 0.45)) @ R(0.3, 'X'), mi=1)                  # funnel
    for k, xx in enumerate((x - 0.10, x + 0.10)):                                 # foot restraints
        bx(bm, xx - 0.05, xx + 0.05, -1.00, -0.92, -0.50, -0.46, 3, 0.01)
    # hand-wash unit + mirror on the forward partition (facing aft)
    yw = yb - t
    bx(bm, 0.60, 1.05, yw - 0.18, yw, 0.30, 0.55, 4, 0.02)
    add_revolve(bm, [(0.0, 0.52), (0.10, 0.52), (0.12, 0.56), (0.13, 0.56)], 24, Tm((0.83, yw - 0.09, 0)), mi=1)
    cylp(bm, (0.83, yw, 0.72), (0.83, yw - 0.07, 0.72), 0.01, 8, 2)                # tap
    bx(bm, 0.66, 1.00, yw - 0.006, yw, 0.78, 1.20, 6)                              # mirror
    bx(bm, 0.65, 1.01, yw - 0.008, yw - 0.002, 0.77, 0.79, 2)
    bx(bm, 0.43, 0.57, yw - 0.03, yw, 0.60, 1.00, 3, 0.02)                         # towel
    cylp(bm, (0.42, yw - 0.035, 1.0), (0.58, yw - 0.035, 1.0), 0.008, 6, 2)
    bx(bm, 0.45, 0.55, -1.60, -1.585, 1.15, 1.32, 7, 0.006)                        # cubicle fan grille
    bx(bm, 0.98, 1.10, -1.25, -1.05, 1.42, 1.48, 7, 0.006)                         # cubicle light
    bx(bm, 0.985, 1.095, -1.24, -1.06, 1.415, 1.42, 8)


def build_R04():
    rid = "R04"
    node = room_node(rid)
    DOOR_N = (0.22, 0.88, -0.45, 1.30)                     # = R03 aft wall opening
    DOOR_S = (-0.33, 0.33, -0.35, 1.25)                    # to R05
    def vent(bm):                                                    # ECLSS supply diffuser
        bx(bm, -0.15, 0.15, -1.30, -1.10, 1.47, 1.50, 1, 0.006)
        for k in range(6):
            bx(bm, -0.13, 0.13, -1.28 + k * 0.03, -1.27 + k * 0.03, 1.465, 1.47, 2)
    room_shell(rid, {"N": [DOOR_N], "S": [DOOR_S]}, ceiling_extra=vent)
    doorway(C.int_name(rid, "DoorwayR03"), node, "Y", 0.50, 0.60, *DOOR_N)
    dm = door_leaf_mesh("Door_064x173", 0.64, 1.73, 0.05, wheel=False, chamfer=0.08, window=True)
    door(C.int_name(rid, "DoorR03"), node, dm, (0.865, 0.465, -0.44), (0, 0, 1), (-1, 0, 0), (0, -1, 0), 90,
         "R04<->R03", "quarters forward pressure door (lever latch, view port); swings aft into R04 "
         "against the starboard wall, forward of the galley")
    # bunks: two berths stacked on the port wall, frame posts on the inboard corners
    bm_ = berth()
    for k, z in enumerate((-0.12, 0.72), 1):
        b_ = inst(C.int_name(rid, f"Berth{k}"), bm_, node, (-1.10, -0.55, z))
        b_["kind"] = "bunk"

    def frame(bm):
        for y in (-1.575, 0.475):
            bx(bm, -0.43, -0.39, y - 0.02, y + 0.02, -0.50, 1.50, 0, 0.006)
            for z in (-0.12, 0.72):
                bx(bm, -1.10, -0.39, y - 0.015, y + 0.015, z - 0.04, z, 0, 0.004)
        bx(bm, -1.09, -0.42, -1.52, 0.42, -0.50, -0.165, 0, 0.008)
        for k, (y0_, y1_) in enumerate(((-1.52, -0.88), (-0.86, -0.22), (-0.20, 0.42))):
            bx(bm, -0.42, -0.405, y0_ + 0.01, y1_ - 0.01, -0.46, -0.20, 1, 0.006)
            yc = (y0_ + y1_) / 2
            bx(bm, -0.405, -0.39, yc - 0.08, yc + 0.08, -0.25, -0.23, 2, 0.004)
        for z in (0.05, 0.35):                                                   # footholds to the upper berth
            bx(bm, -0.44, -0.38, 0.40, 0.455, z, z + 0.025, 2, 0.006)
    part(C.int_name(rid, "BerthFrame"), ["equip", "panel", "metal"], frame, node, smooth=None)
    part(C.int_name(rid, "Galley"), ["panel", "equip", "metal", "dark", "glass", "screen", "red", "rail",
                                     "fabric", "foil"], galley, node, smooth=None)
    part(C.int_name(rid, "Hygiene"), ["panel", "metal", "equip", "fabric", "dark", "foil", "glass", "light", "light"],
         hygiene, node, smooth=40)
    ex = inst(C.int_name(rid, "FireExtinguisher"), extinguisher(), node, (-0.12, 0.425, -0.30))
    ex["kind"] = "fire_extinguisher"
    # bunk ladder: on the inboard edge of the berths, aft of mid-length (clear of the bunched foot curtain)
    lad = inst(C.int_name(rid, "BunkLadder"), ladder(1.45, 0.30, 0.10, key="LadderBunk"), node,
               (-0.39, -1.05, -0.5), (0, 0, -PI / 2))
    lad["kind"] = "ladder to the upper berth"
    # fold-down mess table on the forward wall (port of the door) + 2 stowable stools
    def table(bm):
        x0, x1, y0, y1, zt = -0.38, 0.15, 0.10, 0.50, 0.25
        bx(bm, x0, x1, 0.47, 0.50, zt - 0.06, zt - 0.01, 1, 0.004)                 # wall hinge rail
        bx(bm, x0, x1, y0, 0.47, zt - 0.03, zt, 0, 0.012, 2)                       # leaf
        bx(bm, x0 + 0.01, x1 - 0.01, y0 - 0.006, y0 + 0.004, zt, zt + 0.018, 1, 0.003)   # fiddle lip
        for x in (x0 + 0.04, x1 - 0.04):                                           # folding braces
            bx(bm, x - 0.015, x + 0.015, 0.47, 0.50, zt - 0.36, zt - 0.30, 1, 0.004)
            cylp(bm, (x, 0.48, zt - 0.33), (x, y0 + 0.08, zt - 0.03), 0.011, 8, 1)
        for x in (x0 + 0.12, x1 - 0.12):                                           # cup / tray clips
            add_torus(bm, 0.035, 0.006, 16, 4, Tm((x, y0 + 0.12, zt + 0.004)), mi=2)
        strap(bm, [(x0 + 0.02, 0.30, zt + 0.002), (x1 - 0.02, 0.30, zt + 0.002)], w=0.03, t=0.004,
              side=(0, 1, 0), mi=3)                                                # velcro strip
    tb = part(C.int_name(rid, "FoldTable"), ["equip", "metal", "rail", "fabric"], table, node, smooth=None)
    tb["kind"] = "fold-down mess table (deployed)"

    def stool(bm):
        add_cyl(bm, 0.10, 0.012, 16, Tm((0, 0, 0.006)), mi=1)                     # floor flange
        add_cyl(bm, 0.028, 0.40, 16, z0=0.012, mi=1)                              # post
        add_torus(bm, 0.12, 0.009, 24, 6, Tm((0, 0, 0.17)), mi=1)                 # foot ring
        for k in range(3):
            a = k * 2 * PI / 3
            cylp(bm, (0.028 * math.cos(a), 0.028 * math.sin(a), 0.17), (0.12 * math.cos(a), 0.12 * math.sin(a), 0.17),
                 0.007, 6, 1)
        add_revolve(bm, [(0.0, 0.40), (0.15, 0.40), (0.165, 0.415), (0.165, 0.445), (0.15, 0.46), (0.0, 0.46)],
                    24, mi=0)                                                     # cushion
        bx(bm, -0.02, 0.02, -0.17, -0.155, 0.42, 0.44, 2, 0.003)                  # thigh-strap anchor
    st = lib("Stool", ["fabric", "metal", "dark"], stool, 35)
    for k, (x, y) in enumerate(((-0.22, -0.02), (0.13, -0.10)), 1):
        inst(C.int_name(rid, f"Stool{k}"), st, node, (x, y, -0.5), (0, 0, PI / 2 * k))
    # crew personal lockers above the table
    lk = inst(C.int_name(rid, "CrewLockers"), locker(0.46, 0.50, 0.25, 2, key="CrewLocker"), node,
              (-0.15, 0.50, 0.95), (0, 0, PI))
    lk["kind"] = "crew personal lockers"
    hh = handhold(0.4)
    inst(C.int_name(rid, "HandholdDoorN"), hh, node, (0.13, 0.50, 0.75), wall_rot((0, -1, 0), (0, 0, 1)))
    inst(C.int_name(rid, "HandholdDoorS"), hh, node, (-0.36, -1.60, 0.30),
         wall_rot((0, 1, 0), (0, 0, 1)))
    inst(C.int_name(rid, "HandholdBerth"), hh, node, (-0.41, -0.55, 1.40), wall_rot((0, 0, -1), (0, 1, 0)))
    return node


BUILDERS["R04"] = build_R04


# =================================================================== R05 stores locker
def stores_bank(variant):
    """Stores bank against a side wall: x -0.39..0.39 (along the wall), back y=0, front y=0.70,
    z 0..1.98.  'A': two columns of latched doors over two drawer rows; 'B': drawers + three
    open shelves with strapped cargo transfer bags behind a cargo net."""
    W, D, H = 0.78, 0.70, 1.98

    def b(bm):
        bx(bm, -W / 2, W / 2, 0.0, D - 0.02, 0.0, H, 0, 0.008)                  # carcass
        bx(bm, -W / 2, W / 2, D - 0.02, D, 0.0, 0.05, 3)                         # kick plate
        z = 0.06
        for k in range(2):                                                       # drawer rows
            for c in range(2):
                x0 = -W / 2 + 0.01 + c * W / 2
                x1 = x0 + W / 2 - 0.02
                bx(bm, x0, x1, D - 0.02, D, z, z + 0.25, 1, 0.006)
                bx(bm, (x0 + x1) / 2 - 0.09, (x0 + x1) / 2 + 0.09, D, D + 0.02, z + 0.19, z + 0.21, 2, 0.004)
                bx(bm, x0 + 0.03, x0 + 0.13, D, D + 0.002, z + 0.04, z + 0.08, 4)        # label
            z += 0.26
        if variant == "A":
            for r in range(3):
                h = (H - 0.02 - z) / 3
                for c in range(2):
                    x0 = -W / 2 + 0.01 + c * W / 2
                    x1 = x0 + W / 2 - 0.02
                    zz = z + r * h
                    bx(bm, x0, x1, D - 0.02, D, zz + 0.005, zz + h - 0.005, 1, 0.006)
                    hx = x1 - 0.04 if c == 0 else x0 + 0.04
                    bx(bm, hx - 0.012, hx + 0.012, D, D + 0.025, zz + h * 0.5 - 0.07, zz + h * 0.5 + 0.07, 2, 0.005)
                    for k in range(4):                                           # vent slots
                        vz = zz + h - 0.07 - k * 0.025
                        bx(bm, x0 + 0.06, x1 - 0.06, D, D + 0.003, vz - 0.005, vz + 0.005, 3)
                    bx(bm, (x0 + x1) / 2 - 0.06, (x0 + x1) / 2 + 0.06, D, D + 0.002, zz + 0.05, zz + 0.09, 4)
        else:
            bx(bm, -W / 2 + 0.02, W / 2 - 0.02, 0.02, D - 0.02, z, H - 0.03, 3)  # dark interior
            h = (H - 0.03 - z) / 3
            sizes = [(0.34, 0.42, 0.30), (0.30, 0.40, 0.36), (0.68, 0.45, 0.24)]
            for r in range(3):
                zz = z + r * h
                bx(bm, -W / 2 + 0.02, W / 2 - 0.02, 0.02, D - 0.02, zz, zz + 0.02, 0, 0.004)    # shelf
                bx(bm, -W / 2 + 0.02, W / 2 - 0.02, D - 0.04, D - 0.02, zz + 0.02, zz + 0.06, 2, 0.004)  # lip
                w_, l_, h_ = sizes[r]
                xs = (-0.18, 0.18) if w_ < 0.5 else (0.0,)
                for x in xs:                                                     # cargo transfer bags
                    bx(bm, x - w_ / 2, x + w_ / 2, D / 2 - l_ / 2, D / 2 + l_ / 2, zz + 0.02,
                       zz + 0.02 + min(h_, h - 0.06), 5, 0.03, 2)
                    bx(bm, x - 0.06, x + 0.06, D / 2 + l_ / 2, D / 2 + l_ / 2 + 0.006,
                       zz + 0.06, zz + 0.12, 4)                                  # bag tag
                    strap(bm, [(x - 0.04, D / 2 + l_ / 2, zz + 0.04), (x - 0.04, D / 2 + l_ / 2 + 0.005,
                               zz + 0.02 + min(h_, h - 0.06) + 0.004), (x - 0.04, D / 2 - l_ / 2,
                               zz + 0.02 + min(h_, h - 0.06) + 0.004)], w=0.03, t=0.004, side=(1, 0, 0), mi=2)
                for zt in (zz + 0.12, zz + h * 0.6):                             # retaining straps
                    strap(bm, [(-W / 2 + 0.02, D - 0.005, zt), (W / 2 - 0.02, D - 0.005, zt)], w=0.035,
                          t=0.005, side=(0, 0, 1), mi=6)
            # cargo net over the open shelves
            for k in range(7):
                x = -W / 2 + 0.05 + k * (W - 0.1) / 6
                strap(bm, [(x, D + 0.006, z + 0.03), (x, D + 0.006, H - 0.05)], w=0.012, t=0.006,
                      side=(1, 0, 0), mi=6)
            for k in range(6):
                zz = z + 0.03 + k * (H - 0.08 - z) / 5
                strap(bm, [(-W / 2 + 0.03, D + 0.012, zz), (W / 2 - 0.03, D + 0.012, zz)], w=0.012, t=0.006,
                      side=(0, 0, 1), mi=6)
            for x in (-W / 2 + 0.03, W / 2 - 0.03):                              # net anchors
                for zz in (z + 0.03, H - 0.05):
                    add_torus(bm, 0.018, 0.005, 8, 4, Tm((x, D + 0.01, zz)) @ R(PI / 2, 'X'), mi=2)
        for x in (-W / 2, W / 2 - 0.02):                                         # corner guards
            bx(bm, x, x + 0.02, D - 0.03, D + 0.005, 0.0, H, 2, 0.004)
    return lib(f"StoresBank{variant}", ["panel", "equip", "metal", "dark", "screen", "fabric", "rail"], b, None)


def build_R05():
    rid = "R05"
    node = room_node(rid)
    DOOR = (-0.33, 0.33, -0.35, 1.25)
    room_shell(rid, {"N": [DOOR], "S": [DOOR]})
    doorway(C.int_name(rid, "DoorwayR04"), node, "Y", -1.70, -1.60, *DOOR)
    sa, sb = stores_bank("A"), stores_bank("B")
    for nm, me, x, rz in (("StoresPort", sa, -1.10, -PI / 2), ("StoresStbd", sb, 1.10, PI / 2)):
        o = inst(C.int_name(rid, nm), me, node, (x, -2.10, -0.50), (0, 0, rz))
        o["kind"] = "stores_bank"
    inv = mfd(0.16, 0.14)
    inst(C.int_name(rid, "InventoryTerminal"), inv, node, (-0.36, -2.10, 1.05), wall_rot((1, 0, 0), (0, 0, 1)))

    def floor_ties(bm):                                                          # aisle tie-down track
        for x in (-0.25, 0.25):
            bx(bm, x - 0.02, x + 0.02, -2.48, -1.72, -0.50, -0.49, 0, 0.003)
            for k in range(7):
                add_cyl(bm, 0.008, 0.004, 8, Tm((x, -2.43 + k * 0.11, -0.49)), mi=1)
    part(C.int_name(rid, "TieDownTracks"), ["metal", "dark"], floor_ties, node, smooth=None)

    # open spares crate strapped to the starboard tie-down track (lid hinged open on its aft edge)
    def spares(bm):
        w, l, h, t = 0.34, 0.40, 0.26, 0.015
        bx(bm, -w / 2, w / 2, -l / 2, l / 2, 0, t, 0, 0.004)                       # floor
        for sx in (-1, 1):
            bx(bm, sx * w / 2 - t, sx * w / 2, -l / 2, l / 2, 0, h, 0, 0.004)       # sides
            bx(bm, -w / 2, w / 2, sx * l / 2 - t, sx * l / 2, 0, h, 0, 0.004)
            bx(bm, sx * (w / 2 + 0.004) - 0.01, sx * (w / 2 + 0.004) + 0.01, -0.07, 0.07, h * 0.6, h * 0.6 + 0.03, 1,
               0.004)                                                                # handles
        for sy in (-1, 1):                                                          # rim ribs
            bx(bm, -w / 2 - 0.006, w / 2 + 0.006, sy * l / 2 - 0.02, sy * l / 2 + 0.006 * sy, h - 0.03, h, 1, 0.003)
        bx(bm, -w / 2 + t, w / 2 - t, -l / 2 + t, l / 2 - t, t, 0.11, 2, 0.01)       # foam insert
        # lid open, standing up on the aft hinge
        bx(bm, -w / 2, w / 2, -l / 2 - 0.035, -l / 2 - 0.02, h - 0.01, h + l - 0.01, 0, 0.004)
        bx(bm, -w / 2 + 0.03, w / 2 - 0.03, -l / 2 - 0.02, -l / 2 - 0.012, h + 0.03, h + l - 0.05, 2, 0.008)
        for x in (-w / 2 + 0.05, w / 2 - 0.05):
            add_cyl(bm, 0.008, 0.05, 8, Tm((x, -l / 2 - 0.02, h - 0.005)) @ R(PI / 2, 'Y'), mi=1)
        # contents: filter cartridges, spare LRU cards, a valve, a coiled cable, a pump seal kit
        for k in range(3):
            add_cyl(bm, 0.035, 0.16, 16, Tm((-0.10 + k * 0.075, -0.10, 0.145)) @ R(PI / 2, 'X'), mi=3)
            add_cyl(bm, 0.037, 0.012, 16, Tm((-0.10 + k * 0.075, -0.02, 0.145)) @ R(PI / 2, 'X'), mi=1)
        for k in range(4):
            bx(bm, 0.02 + k * 0.032, 0.04 + k * 0.032, 0.03, 0.17, 0.10, 0.215, 4, 0.002)   # cards
            bx(bm, 0.02 + k * 0.032, 0.04 + k * 0.032, 0.03, 0.05, 0.10, 0.225, 1)          # card ejectors
        add_cyl(bm, 0.03, 0.06, 16, Tm((-0.09, 0.10, 0.14)), mi=1)                 # spare valve
        add_cyl(bm, 0.022, 0.12, 8, Tm((-0.09, 0.10, 0.14)) @ R(PI / 2, 'Y'), mi=1)
        add_torus(bm, 0.03, 0.006, 12, 4, Tm((-0.09, 0.10, 0.185)), mi=5)
        add_torus(bm, 0.055, 0.009, 16, 6, Tm((0.07, -0.11, 0.125)), mi=6)          # cable coil
        add_torus(bm, 0.045, 0.009, 16, 6, Tm((0.07, -0.11, 0.142)), mi=6)
        bx(bm, -0.155, -0.02, 0.15, 0.18, 0.11, 0.14, 5, 0.005)                     # seal kit pouch
        for y in (-0.12, 0.12):                                                    # tie-down straps to the track
            strap(bm, [(-w / 2 - 0.004, y, 0.0), (-w / 2 - 0.004, y, h - 0.04), (-w / 2 + 0.02, y, h + 0.004)],
                  w=0.03, t=0.004, side=(0, 1, 0), mi=7)
            strap(bm, [(w / 2 + 0.004, y, 0.0), (w / 2 + 0.004, y, h - 0.04), (w / 2 - 0.02, y, h + 0.004)],
                  w=0.03, t=0.004, side=(0, 1, 0), mi=7)
    cr = part(C.int_name(rid, "SparesCrate"), ["equip", "metal", "dark", "panel", "screen", "red", "rail", "fabric"],
              spares, node, loc=(0.20, -2.06, -0.49), smooth=35)
    cr["kind"] = "open spares crate (filters, LRU cards, valve, cable, seal kit), strapped to the floor track"

    # extinguisher: horizontal quick-release bracket above the R04 doorway (forward wall, facing aft)
    ex = inst(C.int_name(rid, "FireExtinguisher"), extinguisher(), node, (-0.23, -1.775, 1.40), (0, PI / 2, 0))
    ex["kind"] = "fire_extinguisher"

    # med kit above the R06 door (aft wall, facing forward)
    def medkit(bm):
        bx(bm, -0.17, 0.17, -2.50, -2.49, 1.30, 1.49, 2, 0.004)                     # wall plate
        bx(bm, -0.16, 0.16, -2.49, -2.40, 1.31, 1.48, 0, 0.015, 2)                  # case
        bx(bm, -0.025, 0.025, -2.40, -2.398, 1.35, 1.44, 1)                         # red cross
        bx(bm, -0.045, 0.045, -2.40, -2.398, 1.37, 1.42, 1)
        for x in (-0.12, 0.12):
            bx(bm, x - 0.02, x + 0.02, -2.41, -2.395, 1.455, 1.475, 2, 0.003)       # latches
        bx(bm, -0.06, 0.06, -2.405, -2.39, 1.32, 1.335, 2, 0.004)                   # pull handle
    mk = part(C.int_name(rid, "MedKit"), ["panel", "red", "metal"], medkit, node, smooth=None)
    mk["kind"] = "medical kit"

    # wall straps: elastic retaining straps across the starboard bank, anchored to the end walls
    def wall_straps(bm):
        for z in (0.25, 0.85):
            for y in (-1.70, -2.50):
                sgn = -1 if y > -2 else 1
                bx(bm, 0.335, 0.385, y + sgn * 0.012, y, z - 0.04, z + 0.04, 1, 0.004)       # anchor plate
                add_torus(bm, 0.016, 0.004, 8, 4, Tm((0.36, y + sgn * 0.02, z)) @ R(PI / 2, 'Y'), mi=1)
            strap(bm, [(0.36, -1.735, z), (0.375, -1.80, z), (0.375, -2.40, z), (0.36, -2.465, z)], w=0.03,
                  t=0.005, side=(0, 0, 1), mi=0)
            bx(bm, 0.37, 0.385, -2.13, -2.07, z - 0.02, z + 0.02, 1, 0.003)                 # buckle
    part(C.int_name(rid, "WallStraps"), ["rail", "metal"], wall_straps, node, smooth=None)
    hh = handhold(0.4)
    for nm, x, n in (("HandholdPort", -0.40, (1, 0, 0)), ("HandholdStbd", 0.40, (-1, 0, 0))):
        inst(C.int_name(rid, nm), hh, node, (x, -2.10, 1.42), wall_rot(n, (0, 1, 0)))
    return node


BUILDERS["R05"] = build_R05


# =================================================================== R06 engineering bay
def trefoil(bm, M, r=0.09, mi_bg=0, mi_fg=1):
    """Radiation trefoil plate (disc + 3 blades + hub) in the local XY plane facing +Z."""
    add_cyl(bm, r * 1.25, 0.004, 24, M @ Tm((0, 0, 0.002)), mi=mi_bg)
    for k in range(3):
        a = PI / 2 + k * 2 * PI / 3
        add_arc(bm, [(r * 0.28, 0.0045), (r, 0.0045)], a - PI / 6, a + PI / 6, 6, M, mi=mi_fg)
    add_cyl(bm, r * 0.18, 0.006, 16, M @ Tm((0, 0, 0.003)), mi=mi_fg)


def build_R06():
    rid = "R06"
    node = room_node(rid)
    DOOR = (-0.33, 0.33, -0.35, 1.25)
    CRAWL = (-0.40, 0.40, 0.70, 1.45)                    # aft-top opening into C01

    def ceiling_extra(bm):
        for y in (-3.0, -4.3):                           # fire-suppression discharge nozzles
            cylp(bm, (0.75, y, 1.50), (0.75, y, 1.43), 0.012, 8, 2)
            add_revolve(bm, [(0.012, 1.43), (0.03, 1.41), (0.03, 1.40), (0.0, 1.40)], 16, mi=2,
                        M=Tm((0.75, y, 0.0)))
        add_tube(bm, [(0.75, -2.75, 1.465), (0.75, -4.45, 1.465)], 0.012, 8, mi=3)
        for y in (-2.95, -4.25):                         # tray hangers
            for dx in (-0.09, 0.09):
                cylp(bm, (-0.55 + dx, y, 1.42), (-0.55 + dx, y, 1.5), 0.006, 6, 2)
    ceil = room_shell(rid, {"N": [DOOR], "S": [CRAWL]}, {"F": [(-0.55, 0.55, -4.70, -2.60)]},
                      ceiling_extra=ceiling_extra)
    lay_grating(rid, node, -0.55, 0.55, -4.70, -2.60, -0.5, 2, 3)
    inst(C.int_name(rid, "CableTray"), cable_tray(1.6, 0.2, 0.06), ceil, (-0.55, -4.40, 1.36))
    doorway(C.int_name(rid, "DoorwayR05"), node, "Y", -2.60, -2.50, *DOOR)
    hatch = door_leaf_mesh("Hatch_066x160", 0.66, 1.58, 0.06, wheel=True, chamfer=0.10)
    door(C.int_name(rid, "PressureDoorR05"), node, hatch, (0.33, -2.635, -0.34), (0, 0, 1), (-1, 0, 0), (0, -1, 0), 90,
         "R06<->R05", "engineering bay pressure door; swings aft into R06 along the starboard side of the aisle")

    # ---- aft shielded bulkhead: shield plate with cable penetrations + crawlway collar
    def shield(bm):
        y0, y1 = -4.70, -4.62
        hx0, hx1, hz0, hz1 = -0.46, 0.46, 0.64, 1.50
        bx(bm, -1.08, hx0, y0, y1, -0.48, 1.48, 0, 0.012)
        bx(bm, hx1, 1.08, y0, y1, -0.48, 1.48, 0, 0.012)
        bx(bm, hx0, hx1, y0, y1, -0.48, hz0, 0, 0.012)
        for (a, b_) in ((hx0, hx0 + 0.02), (hx1 - 0.02, hx1)):                  # collar lining
            bx(bm, a, b_, y0, y1 + 0.005, hz0, hz1, 1, 0.003)
        bx(bm, hx0, hx1, y0, y1 + 0.005, hz0, hz0 + 0.02, 1, 0.003)
        for x in [-1.03 + k * 0.1975 for k in range(11)]:                       # edge bolts
            for z in (-0.43, 1.43):
                if hx0 - 0.02 < x < hx1 + 0.02 and z > 1:
                    continue
                add_cyl(bm, 0.012, 0.008, 6, Tm((x, y1 + 0.004, z)) @ R(-PI / 2, 'X'), mi=1)
        for z in [-0.43 + k * 0.2325 for k in range(1, 8)]:
            for x in (-1.03, 1.03):
                add_cyl(bm, 0.012, 0.008, 6, Tm((x, y1 + 0.004, z)) @ R(-PI / 2, 'X'), mi=1)
        # penetration ports: flanged glands, cable / pipe stubs entering the plate
        ports = [(-0.95, 1.20, 0.05, 3), (-0.80, 1.20, 0.05, 3), (-0.65, 1.20, 0.05, 3), (-0.95, 1.02, 0.04, 2),
                 (0.62, 1.20, 0.05, 3), (0.78, 1.20, 0.05, 3), (0.94, 1.20, 0.05, 3),
                 (0.62, -0.08, 0.06, 0), (0.92, 0.62, 0.06, 0)]
        for x, z, r, ncab in ports:
            Mp = Tm((x, y1, z)) @ R(-PI / 2, 'X')
            add_cyl(bm, r + 0.03, 0.02, 24, Mp @ Tm((0, 0, 0.01)), mi=1)            # flange
            add_cyl(bm, r + 0.005, 0.05, 24, Mp @ Tm((0, 0, 0.035)), mi=1)          # gland body
            for k in range(6):                                                        # flange bolts
                a = k * PI / 3
                add_cyl(bm, 0.006, 0.01, 6, Mp @ Tm(((r + 0.02) * math.cos(a), (r + 0.02) * math.sin(a), 0.025)), mi=2)
            for k in range(ncab):                                                     # cables out of the gland
                dx = (k - (ncab - 1) / 2) * 0.026
                add_tube(bm, fillet([(x + dx, y1 + 0.06, z), (x + dx, y1 + 0.14, z), (x + dx * 1.5, y1 + 0.18, z + 0.12),
                                     (x + dx * 1.5, y1 + 0.18, 1.46)], 0.05, 2), 0.011, 6, mi=3 if k % 2 else 4)
        trefoil(bm, Tm((0.0, y1, 0.25)) @ R(-PI / 2, 'X'), 0.09, 5, 4)
        bx(bm, -0.17, 0.17, y1, y1 + 0.003, 0.05, 0.11, 5)                          # warning band
    sp = part(C.int_name(rid, "ShieldPlate"), ["metal", "equip", "dark", "dark", "red", "rail"], shield, node,
              smooth=40)
    sp["kind"] = "aft shielded bulkhead plate (reactor shadow-shield side), with cable penetrations"
    inst(C.int_name(rid, "LadderC01"), ladder(1.25, 0.40, 0.14, key="LadderC01"), node, (0.0, -4.62, -0.50))
    hh = handhold(0.4)
    for nm, x in (("HandholdCrawlL", -0.52), ("HandholdCrawlR", 0.52)):
        inst(C.int_name(rid, nm), hh, node, (x, -4.62, 1.05), wall_rot((0, 1, 0), (0, 0, 1)))

    # ---- engineer station (port), facing the aft bulkhead
    def station(bm):
        prof = [(-4.62, -0.50), (-4.32, -0.50), (-4.32, 0.22), (-4.12, 0.22), (-4.12, 0.27), (-4.50, 0.29),
                (-4.58, 1.05), (-4.62, 1.05)]
        add_prism(bm, prof, 0.76, Tm((-0.68, 0, 0)) @ YZX, mi=0)
        bx(bm, -1.05, -0.31, -4.13, -4.10, 0.21, 0.28, 1, 0.006)                  # desk edge
        bx(bm, -1.04, -0.32, -4.33, -4.31, -0.48, 0.18, 1)                       # access panel
        for k in range(5):
            z = -0.38 + k * 0.05
            bx(bm, -0.90, -0.46, -4.315, -4.31, z, z + 0.02, 2)                    # vent louvres
        add_tube(bm, fillet([(-0.95, -4.56, 1.05), (-0.95, -4.56, 1.15), (-0.95, -4.62, 1.2)], 0.04, 2),
                 0.015, 6, mi=2)
    st = part(C.int_name(rid, "EngineerStation"), ["equip", "dark", "dark"], station, node, smooth=None)
    rot_f = wall_rot(Vector((0, 1, 0.105)), (0, -0.105, 1))                    # sloped panel face
    inst(C.int_name(rid, "ReactorMFD"), mfd(0.40, 0.25), st, (-0.68, -4.545, 0.80), rot_f)
    inst(C.int_name(rid, "PropMFD1"), mfd(), st, (-0.94, -4.53, 0.62), rot_f)
    inst(C.int_name(rid, "PropMFD2"), mfd(), st, (-0.42, -4.53, 0.62), rot_f)
    inst(C.int_name(rid, "StationSwitches"), switch_strip(8), st, (-0.68, -4.515, 0.43), rot_f)
    inst(C.int_name(rid, "StationBreakers"), breaker_row(10), st, (-0.68, -4.505, 0.35), rot_f)
    flat = wall_rot((0, 0, 1), (0, 1, 0))
    inst(C.int_name(rid, "StationKeypad"), button_pad(6, 4), st, (-0.80, -4.30, 0.285), flat)
    inst(C.int_name(rid, "StationThrottle"), throttle_mesh(), st, (-0.45, -4.28, 0.28))
    seat = inst(C.int_name(rid, "SeatEngineerStation"), seat_mesh(), node, (-0.68, -3.72, -0.50), (0, 0, PI))
    seat["kind"] = "engineer station seat (on floor rails)"

    # ---- coolant pump skid (starboard aft): pump + motor + pipes + valves
    def skid(bm):
        for x in (0.42, 1.02):
            bx(bm, x - 0.03, x + 0.03, -4.60, -3.55, -0.50, -0.40, 0, 0.006)     # skid rails
        for y in (-4.50, -4.05, -3.65):
            bx(bm, 0.39, 1.05, y - 0.03, y + 0.03, -0.43, -0.38, 0, 0.006)
        bx(bm, 0.50, 0.96, -4.55, -3.60, -0.38, -0.36, 0, 0.004)                 # deck plate
        A = R(-PI / 2, 'X')                                                       # axis along -Y -> +Y
        # pump volute (axis along Y) + pedestal
        bx(bm, 0.62, 0.86, -4.38, -4.18, -0.36, -0.20, 0, 0.01)
        add_revolve(bm, [(0.0, -0.08), (0.15, -0.08), (0.17, -0.05), (0.17, 0.05), (0.15, 0.08), (0.06, 0.08),
                         (0.06, 0.12), (0.0, 0.12)], 24, Tm((0.74, -4.30, -0.02)) @ A, mi=1)
        add_cyl(bm, 0.055, 0.12, 24, Tm((0.74, -4.48, -0.02)) @ A, mi=1)          # suction nozzle
        add_cyl(bm, 0.075, 0.02, 24, Tm((0.74, -4.54, -0.02)) @ A, mi=2)          # flange
        add_cyl(bm, 0.045, 0.16, 24, Tm((0.74 + 0.15, -4.30, 0.06)) @ R(PI / 2, 'Y'), mi=1)     # discharge
        # coupling guard + motor
        bx(bm, 0.66, 0.82, -4.16, -3.98, -0.14, 0.10, 3, 0.01)
        add_cyl(bm, 0.13, 0.36, 24, Tm((0.74, -3.79, -0.02)) @ A, mi=4)          # motor can
        for k in range(8):                                                       # cooling fins
            add_revolve(bm, [(0.13, -0.006), (0.145, -0.006), (0.145, 0.006), (0.13, 0.006)], 24,
                        Tm((0.74, -3.93 + k * 0.04, -0.02)) @ A, closed=True, mi=4)
        add_cyl(bm, 0.10, 0.03, 24, Tm((0.74, -3.595, -0.02)) @ A, mi=2)          # end bell
        bx(bm, 0.83, 0.93, -3.86, -3.72, -0.02, 0.10, 2, 0.01)                   # terminal box
        add_tube(bm, fillet([(0.93, -3.79, 0.05), (1.02, -3.79, 0.05), (1.02, -3.79, 1.43)], 0.06, 2),
                 0.02, 8, mi=3)                                                  # power conduit
        for y in (-3.70, -3.88):
            bx(bm, 0.62, 0.86, y - 0.03, y + 0.03, -0.36, -0.15, 0, 0.006)       # motor feet
        # suction pipe from the bulkhead penetration (0.62, -0.08) down into the pump
        add_tube(bm, fillet([(0.62, -4.62, -0.08), (0.62, -4.56, -0.08), (0.74, -4.56, -0.02)], 0.04, 2),
                 0.05, 16, mi=1)
        # discharge: up from the volute, gate valve, filter, back into the bulkhead at (0.92, 0.62)
        pts = fillet([(0.89, -4.30, 0.06), (0.95, -4.30, 0.06), (0.95, -4.30, 0.45), (0.95, -4.56, 0.62),
                      (0.92, -4.62, 0.62)], 0.08, 3)
        add_tube(bm, pts, 0.045, 16, mi=1)
        for zz in (0.18, 0.32):                                                  # pipe flanges
            add_cyl(bm, 0.065, 0.02, 16, Tm((0.95, -4.30, zz)), mi=2)
        add_cyl(bm, 0.06, 0.12, 16, Tm((0.95, -4.30, 0.25)), mi=4)               # gate valve body
        cylp(bm, (0.95, -4.30, 0.25), (0.95, -4.15, 0.25), 0.012, 8, 2)          # stem
        add_torus(bm, 0.07, 0.01, 16, 6, Tm((0.95, -4.15, 0.25)) @ R(PI / 2, 'X'), mi=5)    # handwheel
        for k in range(3):
            a = k * 2 * PI / 3
            cylp(bm, (0.95, -4.15, 0.25), (0.95 + 0.07 * math.cos(a), -4.15, 0.25 + 0.07 * math.sin(a)),
                 0.006, 6, 2)
        # bypass / drain line with a ball valve and gauge
        add_tube(bm, fillet([(0.74, -4.30, -0.19), (0.50, -4.30, -0.19), (0.50, -4.30, 0.35),
                             (0.50, -4.56, 0.35), (0.50, -4.62, 0.35)], 0.05, 2), 0.022, 8, mi=1)
        add_cyl(bm, 0.035, 0.07, 16, Tm((0.50, -4.30, 0.10)), mi=4)
        bx(bm, 0.50, 0.66, -4.31, -4.29, 0.125, 0.14, 5, 0.004)                   # ball-valve lever
        add_cyl(bm, 0.035, 0.02, 16, Tm((0.50, -4.24, 0.25)) @ R(PI / 2, 'X'), mi=2)    # gauge
        add_cyl(bm, 0.03, 0.004, 16, Tm((0.50, -4.228, 0.25)) @ R(PI / 2, 'X'), mi=6)
        cylp(bm, (0.50, -4.28, 0.25), (0.50, -4.30, 0.25), 0.008, 6, 2)
        bx(bm, 0.40, 0.48, -3.62, -3.56, -0.36, 0.40, 0, 0.006)                  # skid control box
        bx(bm, 0.395, 0.40, -3.61, -3.57, 0.25, 0.35, 7)
    part(C.int_name(rid, "CoolantPumpSkid"), ["equip", "metal", "dark", "dark", "panel", "red", "screen", "screen"],
         skid, node, smooth=35)

    # ---- battery bank (port forward): 2 rows x 4 cells with hold-down straps
    def battery(bm):
        x0, x1, y0, y1 = -1.08, -0.60, -3.32, -2.68
        bx(bm, x0, x1, y0, y1, -0.50, -0.42, 0, 0.008)                           # tray
        for x in (x0, x1 - 0.02):
            bx(bm, x, x + 0.02, y0, y1, -0.42, -0.30, 0, 0.004)                  # side walls
        for y in (y0, y1 - 0.02):
            bx(bm, x0, x1, y, y + 0.02, -0.42, -0.30, 0, 0.004)
        for r, xc in enumerate((-0.955, -0.725)):                                # 2 rows
            for k in range(4):
                yc = y0 + 0.08 + k * 0.155
                bx(bm, xc - 0.10, xc + 0.10, yc - 0.07, yc + 0.07, -0.42, 0.02, 1, 0.012)   # cell
                for dx, mi in ((-0.05, 2), (0.05, 3)):                              # terminals
                    add_cyl(bm, 0.014, 0.025, 8, Tm((xc + dx, yc, 0.032)), mi=mi)
                bx(bm, xc - 0.03, xc + 0.03, yc - 0.071, yc - 0.069, -0.10, -0.04, 4)       # label
            for k in range(3):                                                       # bus links
                yc = y0 + 0.08 + k * 0.155
                bx(bm, xc + (0.035 if k % 2 == 0 else -0.065), xc + (0.065 if k % 2 == 0 else -0.035),
                   yc, yc + 0.155, 0.04, 0.05, 5, 0.003)
            for yy in (y0 + 0.12, y1 - 0.12):                                        # hold-down straps
                strap(bm, [(xc - 0.115, yy, -0.30), (xc - 0.105, yy, 0.03), (xc + 0.105, yy, 0.03),
                           (xc + 0.115, yy, -0.30)], w=0.04, t=0.005, side=(0, 1, 0), mi=6)
        bx(bm, -1.08, -1.04, -3.05, -2.95, 0.02, 0.10, 0, 0.006)                      # main cable gland
        add_tube(bm, fillet([(-1.06, -3.00, 0.10), (-1.06, -3.00, 0.30), (-1.08, -3.00, 0.45)], 0.05, 2),
                 0.02, 8, mi=4)
        bx(bm, -0.62, -0.58, -2.80, -2.70, -0.30, -0.10, 7, 0.006)                    # isolator switch
    part(C.int_name(rid, "BatteryBank"), ["equip", "dark", "red", "dark", "dark", "metal", "fabric", "rail"],
         battery, node, smooth=None)

    # ---- fire suppression canister (starboard wall, forward)
    def canister(bm):
        x, y = 0.97, -3.05
        add_revolve(bm, [(0.0, 0.02), (0.10, 0.02), (0.12, 0.05), (0.12, 0.62), (0.10, 0.68), (0.05, 0.71),
                         (0.0, 0.71)], 24, Tm((x, y, 0.0)), mi=0)
        add_cyl(bm, 0.035, 0.05, 16, Tm((x, y, 0.735)), mi=1)                     # valve head
        bx(bm, x - 0.05, x + 0.05, y - 0.03, y + 0.03, 0.76, 0.83, 1, 0.008)
        bx(bm, x - 0.09, x - 0.02, y - 0.01, y + 0.01, 0.82, 0.835, 3, 0.003)     # manual release
        add_cyl(bm, 0.02, 0.012, 16, Tm((x - 0.06, y, 0.79)) @ R(PI / 2, 'Y'), mi=2)
        for z in (0.15, 0.50):                                                    # wall brackets
            add_revolve(bm, [(0.121, z - 0.015), (0.13, z - 0.015), (0.13, z + 0.015), (0.121, z + 0.015)],
                        24, Tm((x, y, 0.0)), closed=True, mi=1)
            bx(bm, x + 0.11, 1.10, y - 0.02, y + 0.02, z - 0.015, z + 0.015, 1, 0.004)
        add_tube(bm, fillet([(x, y, 0.83), (x, y, 0.95), (0.75, y + 0.05, 1.20), (0.75, -2.75, 1.465)], 0.06, 2),
                 0.012, 8, mi=1)                                                  # discharge line to ceiling
        bx(bm, x - 0.08, x + 0.08, y - 0.121, y - 0.119, 0.30, 0.42, 3)           # label band
    can = part(C.int_name(rid, "FireSuppression"), ["red", "metal", "dark", "rail"], canister, node, smooth=40)
    can["kind"] = "fire suppression canister (discharge to ceiling nozzles)"
    ex = inst(C.int_name(rid, "FireExtinguisher"), extinguisher(), node, (-0.45, -2.675, 0.10))
    ex["kind"] = "fire_extinguisher"
    inst(C.int_name(rid, "HandholdFwd"), hh, node, (0.62, -2.60, 0.95), wall_rot((0, -1, 0), (0, 0, 1)))

    # ---- power distribution board (port wall, between the battery bank and the engineer station)
    def pdb(bm):
        x0, x1, y0, y1, z0_, z1_ = -1.10, -0.96, -4.02, -3.44, 0.28, 1.30
        bx(bm, x0, x1, y0, y1, z0_, z1_, 0, 0.01)                                   # cabinet
        bx(bm, x1, x1 + 0.006, y0 + 0.02, y1 - 0.02, z0_ + 0.02, z1_ - 0.02, 1, 0.004)   # front panel
        bx(bm, x1 + 0.006, x1 + 0.03, y1 - 0.06, y1 - 0.03, 0.70, 0.90, 2, 0.004)         # door handle
        for k in range(3):                                                          # bus-bar conduits to the ceiling
            y = -3.88 + k * 0.08
            add_tube(bm, [(-1.03, y, z1_), (-1.03, y, 1.50)], 0.022, 8, mi=3)
        add_tube(bm, fillet([(-1.03, -3.48, z0_), (-1.03, -3.48, 0.10), (-1.03, -3.20, 0.10), (-1.06, -3.06, 0.10)],
                            0.05, 2), 0.02, 8, mi=3)                              # battery feed
        bx(bm, x1, x1 + 0.02, y0 + 0.03, y0 + 0.13, z1_ - 0.12, z1_ - 0.03, 4, 0.004)    # main isolator
        cylp(bm, (x1 + 0.02, y0 + 0.08, z1_ - 0.075), (x1 + 0.09, y0 + 0.08, z1_ - 0.075), 0.012, 8, 5)
        bx(bm, x1 + 0.006, x1 + 0.008, y0 + 0.16, y1 - 0.16, z1_ - 0.10, z1_ - 0.04, 5)  # warning band
    pd = part(C.int_name(rid, "PowerDistributionBoard"), ["equip", "panel", "metal", "dark", "red", "rail"], pdb,
              node, smooth=None)
    pd["kind"] = "power distribution board (main bus, battery + reactor feeds)"
    rot_p = wall_rot((1, 0, 0), (0, 0, 1))
    for j in range(4):
        inst(C.int_name(rid, f"PDBBreakers{j + 1}"), breaker_row(10), pd, (-0.954, -3.70, 0.42 + j * 0.13), rot_p)
    inst(C.int_name(rid, "PDBDisplay"), mfd(0.16, 0.14), pd, (-0.954, -3.66, 1.08), rot_p)
    return node


BUILDERS["R06"] = build_R06


# =================================================================== C01 dorsal crawlway
def quilt(bm, side, x, y0, y1, z0, z1, pitch=(0.24, 0.215), t=0.028):
    """Insulation quilting: pillowed pads on a wall plane at x (side 'E' faces -X, 'W' +X)
    or on an end wall at y=x (side 'N'/'S'); seams between pads read as stitching."""
    nu = max(1, round((y1 - y0) / pitch[0]))
    nv = max(1, round((z1 - z0) / pitch[1]))
    du, dv = (y1 - y0) / nu, (z1 - z0) / nv
    g = 0.006
    for i in range(nu):
        for j in range(nv):
            a, b_ = y0 + i * du + g, y0 + (i + 1) * du - g
            c, d = z0 + j * dv + g, z0 + (j + 1) * dv - g
            if side == "E":
                bx(bm, x - t, x, a, b_, c, d, 0, 0.011)
            elif side == "W":
                bx(bm, x, x + t, a, b_, c, d, 0, 0.011)
            elif side == "S":
                bx(bm, a, b_, x, x + t, c, d, 0, 0.011)
            else:
                bx(bm, a, b_, x - t, x, c, d, 0, 0.011)
            if side in "EW" and (i + j) % 3 == 0:                                   # stitch buttons
                xx = x - t if side == "E" else x + t
                add_cyl(bm, 0.007, 0.004, 6, Tm((xx, (a + b_) / 2, (c + d) / 2)) @ R(PI / 2, 'Y'), mi=1)


def work_light():
    """Small caged work light, ceiling mounted (origin on the ceiling, hangs down -Z)."""
    def b(bm):
        bx(bm, -0.06, 0.06, -0.06, 0.06, -0.012, 0.0, 0, 0.004)                  # base plate
        add_cyl(bm, 0.035, 0.05, 16, Tm((0, 0, -0.037)), mi=0)                    # body
        add_revolve(bm, [(0.035, -0.062), (0.05, -0.075), (0.05, -0.085), (0.0, -0.085)], 16, mi=1)   # lens
        for k in range(4):                                                        # cage bars
            a = k * PI / 2 + PI / 4
            p = (0.058 * math.cos(a), 0.058 * math.sin(a))
            add_tube(bm, fillet([(p[0] * 0.7, p[1] * 0.7, -0.012), (p[0], p[1], -0.05), (p[0], p[1], -0.09),
                                 (0, 0, -0.10)], 0.02, 2), 0.003, 6, mi=2)
        add_torus(bm, 0.058, 0.003, 16, 4, Tm((0, 0, -0.07)), mi=2)
        add_tube(bm, [(0.05, 0, -0.006), (0.25, 0, -0.006)], 0.006, 6, mi=3)        # conduit
    return lib("WorkLight", ["equip", "light", "metal", "dark"], b, 35)


def build_C01():
    rid = "C01"
    node = room_node(rid)
    _id, _n, x0, x1, y0, y1, z0, z1, _p = ROOMS[rid]
    CRAWL = (-0.40, 0.40, 0.70, 1.45)
    HATCH_Y = (-5.95, -8.05, -9.30)                       # over B05 tanks, B06 reactor, B07 engine bay
    holes_f = [(-0.27, 0.27, y - 0.27, y + 0.27) for y in HATCH_Y]
    # shell: floor, end walls and ceiling panelled; side walls = backing only, covered in quilting
    room_shell(rid, {"N": [CRAWL], "F": holes_f}, {"E": [(-99, 99, -99, 99)], "W": [(-99, 99, -99, 99)]},
               ceiling_extra=lambda bm: [cylp(bm, (sx * 0.27 + dx, y, 1.53), (sx * 0.27 + dx, y, 1.55), 0.005, 6, 2)
                                         for sx in (-1, 1) for y in (-5.75, -7.6, -9.45) for dx in (-0.07, 0.07)],
               light=False)

    def quilting(bm):
        quilt(bm, "E", x1, y0 + 0.02, y1 - 0.02, z0 + 0.02, z1 - 0.02)
        quilt(bm, "W", x0, y0 + 0.02, y1 - 0.02, z0 + 0.02, z1 - 0.02)
        quilt(bm, "S", y0, x0 + 0.04, x1 - 0.04, z0 + 0.02, z1 - 0.30)
    part(C.int_name(rid, "Quilting"), ["foil", "metal"], quilting, node, smooth=None)
    doorway(C.int_name(rid, "DoorwayR06"), node, "Y", -4.80, -4.70, *CRAWL)

    # crawl hatch from R06: hinged on its top edge, propped open up under the crawlway ceiling
    ch = door_leaf_mesh("Hatch_075x080", 0.75, 0.80, 0.05, wheel=False, chamfer=0.10)
    door(C.int_name(rid, "HatchR06"), node, ch, (0.40, -4.83, 1.44), (-1, 0, 0), (0, 0, -1), (0, -1, 0), 90,
         "C01<->R06", "crawlway entry hatch (top of the R06 ladder); hinged on its top edge, latched open "
                      "up against the crawlway ceiling")
    # floor access hatches into the bays below (closed)
    fh = door_leaf_mesh("Hatch_050x050", 0.50, 0.50, 0.04, wheel=False, chamfer=0.06)
    bays = {-5.95: "B05", -8.05: "B06", -9.30: "B07"}

    def frames(bm):
        for y in HATCH_Y:
            for (a, b_, c, d) in ((-0.30, 0.30, y + 0.255, y + 0.30), (-0.30, 0.30, y - 0.30, y - 0.255),
                                  (-0.30, -0.255, y - 0.255, y + 0.255), (0.255, 0.30, y - 0.255, y + 0.255)):
                bx(bm, a, b_, c, d, z0 - SH, z0 + 0.012, 0, 0.004)                 # coaming frame
            bx(bm, -0.12, 0.12, y + 0.31, y + 0.35, z0, z0 + 0.002, 1)               # bay label
    part(C.int_name(rid, "FloorHatchFrames"), ["metal", "rail"], frames, node, smooth=None)
    for k, y in enumerate(HATCH_Y, 1):
        door(C.int_name(rid, f"FloorHatch{k}"), node, fh, (0.25, y + 0.25, z0 - 0.02), (-1, 0, 0), (0, -1, 0),
             (0, 0, 1), 0, f"C01<->{bays[y]} (bay below)", "floor access hatch, closed; hinged on its forward edge")

    # handholds along both sides, cable trays under the ceiling, work light at Light_C01
    hh = handhold(0.4)
    k = 0
    for y in (-5.75, -6.55, -7.35, -8.15, -8.95):
        for sx, side in ((-1, "L"), (1, "R")):
            k += 1
            inst(C.int_name(rid, f"Handhold{side}{k:02d}"), hh, node, (sx * (0.45 - 0.035), y, 1.02),
                 wall_rot((-sx, 0, 0), (0, 1, 0)))
    ceil = bpy.data.objects[C.int_name(rid, "Ceiling")]
    tray = cable_tray(3.85, 0.12, 0.05)
    for sx, side in ((-1, "Port"), (1, "Stbd")):
        inst(C.int_name(rid, f"CableTray{side}"), tray, ceil, (sx * 0.27, -9.55, 1.47))
    p = sock("SOCKET_Light_C01")
    wl = inst(C.int_name(rid, "WorkLight"), work_light(), ceil, (p.x, p.y, z1))
    wl["socket"] = "SOCKET_Light_C01"
    for k, y in enumerate((-6.30, -8.70), 2):                                    # extra work lights fore/aft
        inst(C.int_name(rid, f"WorkLight{k}"), work_light(), ceil, (0.0, y, z1))

    def junction(bm):                                                             # aft-end junction box
        bx(bm, -0.36, 0.36, y0, y0 + 0.12, 1.28, 1.52, 0, 0.01)
        bx(bm, -0.34, 0.34, y0 + 0.12, y0 + 0.125, 1.30, 1.50, 1)
        for sx in (-1, 1):
            bx(bm, sx * 0.27 - 0.07, sx * 0.27 + 0.07, y0 + 0.12, y0 + 0.20, 1.40, 1.47, 0, 0.006)
        bx(bm, -0.08, 0.08, y0 + 0.125, y0 + 0.13, 1.33, 1.37, 2)
    part(C.int_name(rid, "JunctionBox"), ["equip", "dark", "screen"], junction, node, smooth=None)
    return node


BUILDERS["C01"] = build_C01


# =================================================================== driver / report
def run(rooms, out):
    for rid in ROOM_ORDER:
        if rid not in rooms or rid not in BUILDERS:
            continue
        wipe_room(rid)
        BUILDERS[rid]()
        bpy.context.view_layer.update()
        C.save_blend(out)
        print(f"KESTREL_INT room {rid} done; saved {out}", flush=True)


def room_objects(rid):
    return [o for o in INT.all_objects if o.name.split("_")[1:2] == [rid]]


def skin_volume_trees():
    """(distance trees, inside test).  Distance: the detailed skins (fuselage with its cockpit
    opening, sponsons, nacelles) + the canopy glazing/fairing surface (open loft, no fake
    bottom/caps).  Inside: analytic fuselage section, closed sponson/nacelle meshes, closed
    canopy volume."""
    skins = [o for o in bpy.data.collections[C.COL_EXT].all_objects if o.get("kestrel_skin")]
    dist = [E.tree_for(o) for o in skins]
    closed = [E.tree_for(o) for o in skins if "Fuselage" not in o.name]
    rings = E.canopy_rings()
    (y2, r2), (y3, r3) = rings[-2], rings[-1]
    verts = [(x, y, z) for (y, r) in rings[:-1] for (x, z) in r]
    for p, q in zip(r2, E.fairing_end_points(y2, r2, y3, r3)):     # clipped aft fairing (20_exterior)
        verts.append((p[0], y2 - 0.002, p[1]) if q is None else tuple(q))
    n = len(rings[0][1])
    side = [(k * n + i, k * n + i + 1, (k + 1) * n + i + 1, (k + 1) * n + i)
            for k in range(len(rings) - 1) for i in range(n - 1)]
    dist.append(BVHTree.FromPolygons(verts, side))
    K = len(rings)
    chord = [(k * n + n - 1, k * n, (k + 1) * n, (k + 1) * n + n - 1) for k in range(K - 1)]
    vol = side + chord + [tuple(range(n)), tuple(range((K - 1) * n, K * n))]
    bm = bmesh.new()
    bv = [bm.verts.new(v) for v in verts]
    for f in vol:
        bm.faces.new([bv[i] for i in f])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    closed.append(BVHTree.FromBMesh(bm))
    bm.free()

    def inside(p):
        if E.inside_fuselage(p.x, p.z, p.y) and B.FUSELAGE[-1][0] <= p.y <= B.FUSELAGE[0][0]:
            return True
        for tr in closed:
            hit = tr.find_nearest(p)
            if hit[0] is not None and (p - hit[0]).dot(hit[1]) < 0:
                return True
        return False
    return dist, inside


def clearance(objs, trees):
    """Min signed distance of every vertex of objs to the pressure-hull skin; positive = inside."""
    dist, inside = trees
    worst, per = 1e9, {}
    for o in objs:
        if o.type != 'MESH':
            continue
        mw = o.matrix_world
        m = 1e9
        for v in o.data.vertices:
            p = mw @ v.co
            d = min((tr.find_nearest(p)[3] for tr in dist), default=1e9)
            if d < m or not inside(p):
                m = min(m, d if inside(p) else -d)
        per[o.name] = round(m, 4)
        worst = min(worst, m)
    return worst, per


def exterior_intrusions():
    out = []
    boxes = {rid: ROOMS[rid][2:8] for rid in ROOM_ORDER + ["R03a"]}
    for o in bpy.data.collections[C.COL_EXT].all_objects:
        if o.type != 'MESH':
            continue
        mw = o.matrix_world
        for rid, (x0, x1, y0, y1, z0, z1) in boxes.items():
            n = 0
            for v in o.data.vertices:
                p = mw @ v.co
                if x0 - SH < p.x < x1 + SH and y0 - SH < p.y < y1 + SH and z0 - SH < p.z < z1 + SH:
                    n += 1
            if n:
                out.append([rid, o.name, n])
    return out


def report(rooms_checked=ROOM_ORDER):
    bpy.context.view_layer.update()
    rep = {"rooms": {}}
    trees = skin_volume_trees()
    total = 0
    biggest = (0, "")
    for rid in ROOM_ORDER + ["R03a"]:
        objs = room_objects(rid)
        if not objs:
            continue
        meshes = [o for o in objs if o.type == 'MESH']
        t = C.triangle_count(meshes)
        total += t
        for o in meshes:
            tt = C.triangle_count([o])
            if tt > biggest[0]:
                biggest = (tt, o.name)
        worst, per = clearance(meshes, trees)
        rep["rooms"][rid] = {
            "tris": t, "objects": len(objs), "mesh_objects": len(meshes),
            "linked_duplicates": sum(1 for o in meshes if o.data.users > 1),
            "min_skin_clearance_m": round(worst, 4),
            "below_0.05": {k: v for k, v in per.items() if v < 0.05},
        }
    rep["interior_fitted_tris"] = total
    rep["max_single_interior_mesh"] = {"name": biggest[1], "tris": biggest[0]}
    bays = [o for o in INT.all_objects if o.type == 'MESH' and o.name.split("_")[1].startswith("B")]
    rep["bays_blockout_tris"] = C.triangle_count(bays)
    rep["exterior_tris"] = C.triangle_count([o for o in bpy.data.collections[C.COL_EXT].all_objects
                                             if o.type == 'MESH'])
    rep["triangles_total_scene"] = C.triangle_count()
    rep["objects_total"] = len(bpy.context.scene.objects)
    rep["interior_objects"] = len(INT.all_objects)
    rep["doors"] = [{"name": o.name, "connects": o.get("door_connects"), "state": o.get("door_state"),
                     "open_deg": o.get("door_open_deg"), "hinge_world": list(o.get("door_hinge_world")),
                     "hinge_axis": list(o.get("door_hinge_axis")),
                     "origin_world": [round(v, 3) for v in o.matrix_world.translation]}
                    for o in INT.all_objects if o.get("door")]
    rep["exterior_vertices_inside_room_shells"] = exterior_intrusions()
    rep["materials"] = sorted(m.name for m in bpy.data.materials if m.users)
    return rep


if __name__ == "__main__":
    args = C.script_args()
    src = args.get("blend", C.BLEND_PATH)
    out = args.get("out", src)
    rooms = args.get("rooms")
    rooms = set(rooms.split(",")) if isinstance(rooms, str) else set(ROOM_ORDER)
    C.open_blend(src)
    setup()
    run(rooms, out)
    rep = report()
    if args.get("report"):
        with open(args["report"], "w") as f:
            json.dump(rep, f, indent=1)
    print("KESTREL_INT_REPORT " + json.dumps({k: v for k, v in rep.items()}))
