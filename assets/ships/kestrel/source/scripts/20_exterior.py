"""IC-4 "Kestrel" — 20 exterior detail pass.

Runs ON TOP of the blockout built by 10_blockout.py: opens the .blend, deletes
everything in Kestrel_Exterior (blockout placeholders or a previous detail
pass) and rebuilds the exterior in ten feature groups, saving after each one.
Kestrel_Interior and Kestrel_Sockets are never touched (sockets are only READ
for placement).

    blender -b --factory-startup --python 20_exterior.py -- [--blend in.blend] [--out out.blend]
                                                          [--report report.json] [--groups HULL,ENGINES,...]

Defaults: --blend/--out ../kestrel.blend.  Re-runnable / idempotent.
Node names: EXT_<Group>_<Part>[_L|_R] (_R = starboard +X).  Repeated parts are
linked duplicates (shared mesh datablocks) so glTF export instances them.
"""
import importlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True  # keep scripts/ free of __pycache__

import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")
B = importlib.import_module("10_blockout")

PI = math.pi
SIDES = (("L", -1), ("R", 1))
GROUP_ORDER = ["HULL", "ENGINES", "RADIATORS", "RCS", "CANOPY", "GEAR", "WEAPONS",
               "SENSORS", "DOCK", "GREEBLES"]   # DOCK step also builds the HATCH group

EXT = None          # Kestrel_Exterior collection
ROOT = None         # root empty
MAT = {}
PANEL_RAISE = {}    # name -> {(k, i): top offset of that panel above the base loft}
STATS = {}
BUILDERS = {}

# Panel seam profile: groove inset (thickness, depth) then panel lip inset.
GROOVE_T, GROOVE_D, LIP_T = 0.012, -0.008, 0.010
LIP_PLAIN = 0.010                         # ordinary panel  -> top at +0.002
LIP_WHIPPLE = (0.016, 0.022, 0.028)       # Whipple standoff plates -> +0.008..+0.020


# =================================================================== basics
def ename(group, part, side=None):
    return C.ext_name(group, part, side)


def Tm(v):
    return Matrix.Translation(Vector(v))


def R(angle, axis):
    return Matrix.Rotation(angle, 4, axis)


def frame(normal, along=(0.0, 1.0, 0.0)):
    """3x3 rotation: local Z = normal, local Y = `along` projected onto the plane."""
    z = Vector(normal).normalized()
    y = Vector(along)
    y = y - z * y.dot(z)
    if y.length < 1e-6:
        y = Vector((0, 0, 1)) - z * z.z
    y.normalize()
    x = y.cross(z)
    return Matrix((x, y, z)).transposed()


def place(name, data, parent=None, loc=(0, 0, 0), rot=None):
    o = bpy.data.objects.new(name, data)
    EXT.objects.link(o)
    o.parent = parent if parent is not None else ROOT
    o.location = Vector(loc)
    if rot is not None:
        if isinstance(rot, Matrix):
            o.rotation_euler = rot.to_3x3().to_euler()
        else:
            o.rotation_euler = rot
    return o


def empty(name, parent=None, loc=(0, 0, 0), rot=None, size=0.3):
    o = place(name, None, parent, loc, rot)
    o.empty_display_type = 'PLAIN_AXES'
    o.empty_display_size = size
    return o


def finish_mesh(name, bm, mats, smooth_angle=None, recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.normal_update()
    if smooth_angle is not None:
        lim = math.radians(smooth_angle)
        for f in bm.faces:
            f.smooth = True
        for e in bm.edges:
            lf = e.link_faces
            if len(lf) != 2 or lf[0].normal.angle(lf[1].normal, 0.0) > lim:
                e.smooth = False
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    for m in (mats if isinstance(mats, (list, tuple)) else [mats]):
        me.materials.append(m)
    return me


def part(name, mats, build, parent=None, loc=(0, 0, 0), rot=None, smooth=30.0, recalc=True):
    bm = bmesh.new()
    build(bm)
    me = finish_mesh(name, bm, mats, smooth, recalc)
    return place(name, me, parent, loc, rot)


def mesh(name, mats, build, smooth=30.0, recalc=True):
    bm = bmesh.new()
    build(bm)
    return finish_mesh(name, bm, mats, smooth, recalc)


def inst(name, src, parent=None, loc=(0, 0, 0), rot=None):
    """Linked duplicate of src (object or mesh)."""
    me = src.data if isinstance(src, bpy.types.Object) else src
    return place(name, me, parent, loc, rot)


# =================================================================== bmesh primitives
def _new_verts(bm, before):
    return [v for v in bm.verts if v not in before]


def add_box(bm, sx, sy, sz, M=None, bevel=0.0, segs=1, mi=0):
    before = set(bm.verts)
    r = bmesh.ops.create_cube(bm, size=1.0)
    for v in r['verts']:
        v.co = Vector((v.co.x * sx, v.co.y * sy, v.co.z * sz))
    if bevel > 0:
        edges = list({e for v in r['verts'] for e in v.link_edges})
        bmesh.ops.bevel(bm, geom=edges, offset=bevel, segments=segs, profile=0.5,
                        affect='EDGES', clamp_overlap=True)
    nv = _new_verts(bm, before)
    if M is not None:
        bmesh.ops.transform(bm, matrix=M, verts=nv)
    for f in {f for v in nv for f in v.link_faces}:
        f.material_index = mi
    return nv


def add_revolve(bm, prof, segs, M=None, closed=False, cap0=False, cap1=False, mi=0, phase=None):
    """Revolve (r, z) profile around local Z."""
    ph = PI / segs if phase is None else phase
    M = M or Matrix.Identity(4)
    rings = []
    for (r, z) in prof:
        rings.append([bm.verts.new(M @ Vector((r * math.cos(2 * PI * k / segs + ph),
                                               r * math.sin(2 * PI * k / segs + ph), z)))
                      for k in range(segs)])
    faces = []
    nr = len(rings)
    for a in range(nr if closed else nr - 1):
        ra, rb = rings[a], rings[(a + 1) % nr]
        for k in range(segs):
            k2 = (k + 1) % segs
            faces.append(bm.faces.new((ra[k], ra[k2], rb[k2], rb[k])))
    if cap0:
        faces.append(bm.faces.new(list(reversed(rings[0]))))
    if cap1:
        faces.append(bm.faces.new(rings[-1]))
    for f in faces:
        f.material_index = mi
    return faces


def add_cyl(bm, r, h, segs, M=None, mi=0, z0=None):
    """Capped cylinder along local Z, centred (or from z0 to z0+h)."""
    a = -h / 2 if z0 is None else z0
    return add_revolve(bm, [(r, a), (r, a + h)], segs, M, cap0=True, cap1=True, mi=mi)


def add_torus(bm, R0, r, segs, segs_minor, M=None, mi=0):
    prof = [(R0 + r * math.cos(2 * PI * k / segs_minor), r * math.sin(2 * PI * k / segs_minor))
            for k in range(segs_minor)]
    return add_revolve(bm, prof, segs, M, closed=True, mi=mi)


def add_tube(bm, pts, r, segs=8, cap=True, closed=False, M=None, mi=0):
    P = [Vector(p) for p in pts]
    n = len(P)
    T, S = [], []
    for i in range(n):
        if closed or 0 < i < n - 1:
            a = (P[i] - P[i - 1]).normalized()
            b = (P[(i + 1) % n] - P[i]).normalized()
            t = a + b
            t = t.normalized() if t.length > 1e-6 else b
            bend = b - a
            S.append((1.0 / max(a.dot(t), 0.25), bend.normalized() if bend.length > 1e-6 else None))
        else:
            t = (P[1] - P[0]).normalized() if i == 0 else (P[-1] - P[-2]).normalized()
            S.append((1.0, None))
        T.append(t)
    ref = Vector((0, 0, 1)) if abs(T[0].z) < 0.9 else Vector((1, 0, 0))
    N = T[0].cross(ref).normalized()
    rings = []
    for i in range(n):
        if i > 0:
            N = T[i - 1].rotation_difference(T[i]) @ N
            N = (N - T[i] * N.dot(T[i])).normalized()
        Bn = T[i].cross(N)
        s, bend = S[i]
        ring = []
        for k in range(segs):
            ang = 2 * PI * k / segs + PI / segs
            d = N * math.cos(ang) + Bn * math.sin(ang)
            if bend is not None:
                d = d + bend * d.dot(bend) * (s - 1.0)
            p = P[i] + d * r
            ring.append(bm.verts.new(M @ p if M is not None else p))
        rings.append(ring)
    faces = []
    for a in range(n if closed else n - 1):
        ra, rb = rings[a], rings[(a + 1) % n]
        for k in range(segs):
            k2 = (k + 1) % segs
            faces.append(bm.faces.new((ra[k], ra[k2], rb[k2], rb[k])))
    if cap and not closed:
        faces.append(bm.faces.new(list(reversed(rings[0]))))
        faces.append(bm.faces.new(rings[-1]))
    for f in faces:
        f.material_index = mi
    return faces


def add_prism(bm, pts2d, t, M=None, mi=0, z0=None):
    """Polygon in local XY extruded along Z (centred, or z0..z0+t)."""
    M = M or Matrix.Identity(4)
    a = -t / 2 if z0 is None else z0
    lo = [bm.verts.new(M @ Vector((x, y, a))) for (x, y) in pts2d]
    hi = [bm.verts.new(M @ Vector((x, y, a + t))) for (x, y) in pts2d]
    n = len(pts2d)
    faces = [bm.faces.new(list(reversed(lo))), bm.faces.new(hi)]
    for i in range(n):
        j = (i + 1) % n
        faces.append(bm.faces.new((lo[i], lo[j], hi[j], hi[i])))
    for f in faces:
        f.material_index = mi
    return faces


def fillet(pts, rad, n=3):
    P = [Vector(p) for p in pts]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        d1, d2 = a - b, c - b
        l1, l2 = d1.length, d2.length
        ang = d1.angle(d2, PI)
        if ang > PI - 1e-3 or l1 < 1e-6 or l2 < 1e-6:
            out.append(b)
            continue
        tl = min(rad / math.tan(ang / 2), 0.45 * l1, 0.45 * l2)
        p1, p2 = b + d1.normalized() * tl, b + d2.normalized() * tl
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * p1 + 2 * (1 - t) * t * b + t * t * p2)
    out.append(P[-1])
    return out


def split_ring(ring):
    out = []
    for i in range(len(ring)):
        a, b = ring[i], ring[(i + 1) % len(ring)]
        out += [a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)]
    return out


# =================================================================== hull surface queries
TREES = {}
BASE_TREES = {}   # mesh name -> BVH of the un-panelled loft (clean facet normals)


def tree_for(obj):
    bpy.context.view_layer.update()
    mw = obj.matrix_world
    verts = [mw @ v.co for v in obj.data.vertices]
    polys = [tuple(p.vertices) for p in obj.data.polygons]
    return BVHTree.FromPolygons(verts, polys)


def skin_trees():
    if not TREES:
        for o in EXT.all_objects:
            if o.get("kestrel_skin"):
                TREES[o.name] = tree_for(o)
    return TREES


def surf(p, n, only=None, reach=0.8):
    """Ray from p + n*reach along -n onto the skin; returns (hit, normal) or (p, n)."""
    p, n = Vector(p), Vector(n).normalized()
    best, best_name = None, None
    for name, tr in skin_trees().items():
        if only and not any(name.startswith(s) for s in only):
            continue
        hit = tr.ray_cast(p + n * reach, -n, reach * 2.5)
        if hit[0] is not None and (best is None or hit[3] < best[3]):
            best, best_name = hit, name
    if best is None:
        return p, n
    nrm = best[1]
    obj = bpy.data.objects[best_name]
    bt = BASE_TREES.get(obj.data.name)
    if bt is not None:      # take the facet normal from the base loft, not a seam bevel
        mw = obj.matrix_world
        near = bt.find_nearest(mw.inverted() @ best[0])
        if near[0] is not None:
            nrm = (mw.to_3x3() @ near[1]).normalized()
    nrm = nrm if nrm.dot(n) > 0 else -nrm
    return best[0], nrm


def inside_fuselage(x, z, y):
    hw, hh, zc, ch = B.fuselage_section(y)
    return abs(x) <= hw and abs(z - zc) <= hh and abs(x) + abs(z - zc) <= hw + hh - ch


# =================================================================== panelled loft
def panel_loft(name, sections, mat, lip_of, drop=None, split=None):
    """Loft equal-count rings (fore->aft) into a closed hull and cut every quad
    into a panel: a 2-step bevelled seam (groove inset, then lip inset/raise).
    lip_of(key, centre, normal) -> lip depth.  Returns (obj, {key: top_offset}).
    drop(key, [vert coords]) -> True removes that loft quad (an opening; the
    boundary is left open on purpose, e.g. the cockpit opening under the canopy).
    split(bm, grid) may cut loft quads before `drop` runs (new pieces get new keys)."""
    bm = bmesh.new()
    n = len(sections[0][1])
    rings = [[bm.verts.new((u, y, v)) for (u, v) in ring] for (y, ring) in sections]
    grid = {}
    for k in range(len(rings) - 1):
        for i in range(n):
            j = (i + 1) % n
            grid[(k, i)] = bm.faces.new((rings[k][i], rings[k][j], rings[k + 1][j], rings[k + 1][i]))
    for e, rg in ((0, rings[0]), (1, rings[-1])):          # end caps: ring of panels + centre
        ctr = sum((v.co for v in rg), Vector()) / n
        inner = [bm.verts.new(ctr + (v.co - ctr) * 0.55) for v in rg]
        for i in range(n):
            j = (i + 1) % n
            grid[("cap", e, i)] = bm.faces.new((rg[i], rg[j], inner[j], inner[i]))
        grid[("cap", e, "c")] = bm.faces.new(inner)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if split is not None:     # PATCH (task 7): cut quads so an opening can follow a footprint
        split(bm, grid)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    if drop is not None:      # PATCH (interior task): cut openings along panel seams
        gone = [k for k, f in grid.items() if drop(k, [v.co.copy() for v in f.verts])]
        if gone:
            bmesh.ops.delete(bm, geom=[grid.pop(k) for k in gone], context='FACES_ONLY')
            loose = [v for v in bm.verts if not v.link_faces]      # no stray verts in the opening
            if loose:
                bmesh.ops.delete(bm, geom=loose, context='VERTS')
            STATS[f"{name}_openings_quads"] = len(gone)
    bm.normal_update()
    BASE_TREES[name] = BVHTree.FromBMesh(bm)
    tops, groups = {}, {}
    for key, f in grid.items():
        d = lip_of(key, f.calc_center_median(), f.normal.copy())
        tops[key] = GROOVE_D + d
        groups.setdefault(d, []).append(f)
    bmesh.ops.inset_individual(bm, faces=list(grid.values()), thickness=GROOVE_T,
                               depth=GROOVE_D, use_even_offset=True)
    for d, fs in groups.items():
        bmesh.ops.inset_individual(bm, faces=fs, thickness=LIP_T, depth=d, use_even_offset=True)
    me = finish_mesh(name, bm, mat, None, recalc=False)
    BASE_TREES[me.name] = BASE_TREES.pop(name)
    return me, tops


def fuselage_stations():
    ys = [11.5, 11.0, 10.5, 10.05, 9.6, 9.1, 8.6, 8.1, 7.6]
    n = 30
    ys += [7.6 - (17.4 * i / n) for i in range(1, n + 1)]      # -> -9.8
    ys += [-10.4, -11.0]
    return ys


def fuselage_ring(y):
    hw, hh, zc, ch = B.fuselage_section(y)
    return split_ring(C.octagon_ring(hw, hh, ch, 0.0, zc))


def sponson_section(y):
    S = B.SPONSON
    for a, b in zip(S, S[1:]):
        if a[0] >= y >= b[0]:
            t = (a[0] - y) / (a[0] - b[0]) if a[0] != b[0] else 0.0
            return tuple(a[i] + (b[i] - a[i]) * t for i in range(5))
    return S[0] if y > S[0][0] else S[-1]


def whipple_level(k, i):
    return LIP_WHIPPLE[(k * 7 + i * 3) % 3]


# =================================================================== materials / setup
def setup():
    global EXT, ROOT
    cols = C.setup_collections()
    EXT = cols["ext"]
    ROOT = bpy.data.objects[C.ROOT]
    MAT.update(
        hull=C.placeholder_material("MAT_Placeholder_Hull", (0.55, 0.56, 0.58), roughness=0.5),
        dark=C.placeholder_material("MAT_Placeholder_Dark", (0.12, 0.12, 0.13), roughness=0.5),
        glass=C.placeholder_material("MAT_Placeholder_Canopy", (0.05, 0.09, 0.12), roughness=0.1),
        rad=C.placeholder_material("MAT_Placeholder_Radiator", (0.75, 0.72, 0.66), roughness=0.4),
        metal=C.placeholder_material("MAT_Placeholder_Metal", (0.42, 0.42, 0.44), roughness=0.35),
        foil=C.placeholder_material("MAT_Placeholder_Foil", (0.80, 0.58, 0.20), roughness=0.3),
        rail=C.placeholder_material("MAT_Placeholder_Handrail", (0.95, 0.72, 0.04), roughness=0.5),
        scorch=C.placeholder_material("MAT_Placeholder_Scorch", (0.025, 0.022, 0.02), roughness=0.9),
    )
    for m in MAT.values():
        m["kestrel_role"] = {
            "MAT_Placeholder_Hull": "hull_paint", "MAT_Placeholder_Dark": "dark_thermal",
            "MAT_Placeholder_Canopy": "glass", "MAT_Placeholder_Radiator": "radiator",
            "MAT_Placeholder_Metal": "bare_metal", "MAT_Placeholder_Foil": "foil",
            "MAT_Placeholder_Handrail": "eva_yellow", "MAT_Placeholder_Scorch": "scorch",
        }[m.name]


def wipe_exterior():
    ext = bpy.data.collections[C.COL_EXT]
    for o in list(ext.all_objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for me in list(bpy.data.meshes):
        if me.users == 0:
            bpy.data.meshes.remove(me)


def sock(name):
    o = bpy.data.objects[name]
    bpy.context.view_layer.update()
    return o.matrix_world.copy()


# =================================================================== 1 HULL
def handrail_mesh(name, length, stand=0.09, r=0.018):
    """EVA handrail along local Y, standing off local +Z; feet at z=0."""
    L = length / 2

    def build(bm):
        pts = fillet([(0, -L, -0.01), (0, -L, stand), (0, L, stand), (0, L, -0.01)], 0.05, 3)
        add_tube(bm, pts, r, 8)
        for y in (-L, L) + ((0.0,) if length > 0.9 else ()):
            add_cyl(bm, 0.035, 0.012, 8, Tm((0, y, 0.002)))      # foot pads
            if y == 0.0:
                add_tube(bm, [(0, 0, -0.01), (0, 0, stand)], r * 0.9, 8)
    bm = bmesh.new()
    build(bm)
    return finish_mesh(name, bm, MAT["rail"], 50)


def build_hull():
    # ---- fuselage
    nwhip = [0]

    def lip_fuselage(key, c, n):
        sun = (n.z > 0.35) or (c.y > 0.0 and n.z > -0.35) or c.y > 7.55
        if key[0] == "cap":
            return LIP_WHIPPLE[1] if key[1] == 0 else (LIP_WHIPPLE[0] if key[2] == "c" else LIP_PLAIN)
        k, i = key[0], key[1]
        if sun:
            nwhip[0] += 1
            return whipple_level(k, i)
        return LIP_PLAIN

    secs = [(y, fuselage_ring(y)) for y in fuselage_stations()]

    # PATCH (interior task, approved): open the flat dorsal skin under the canopy
    # over the cockpit (every whole panel between fuselage stations that overlaps R01's
    # Y span; all of them lie under the full-width canopy), so the cockpit looks out
    # through the glazing instead of at the skin's underside.
    r01 = next(r for r in B.INTERIOR if r[0] == "R01")
    cy0, cy1 = r01[4], r01[5]
    dropped_y = set()

    def dorsal_z(y):                        # flat dorsal skin height (centre z + half height)
        s_ = B.fuselage_section(y)
        return s_[1] + s_[2]

    def drop_cockpit(key, cos):
        if key[0] == "cap":
            return False
        ys_ = [c.y for c in cos]
        ok = (min(ys_) < cy1 - 1e-4 and max(ys_) > cy0 + 1e-4 and
              all(c.z >= dorsal_z(c.y) - 1e-4 and abs(c.x) <= 0.9 + 1e-4 for c in cos))
        if ok:
            dropped_y.update(round(y, 4) for y in ys_)
        return ok

    # PATCH (task 7): open the flat dorsal skin under the dorsal dock collar so the R03a
    # dock trunk (INT_R03a_DockTrunk, 0.66 m octagon round (-0.47, 1.55), top Z 1.835)
    # opens into the collar barrel and the closed EXT_DOCK_Hatch instead of being capped
    # by the skin.  Opening = the port dorsal panel column X -0.9..0 (existing seams)
    # between Y 1.05 and 2.05 (two extra cuts across the panels of the neighbouring
    # stations); its corners stay 0.67 m from the collar axis, inside the 0.72 m skirt,
    # so from outside the collar still hides the opening completely.
    dk = sock("SOCKET_Dock_Dorsal").translation
    DOCK_OPEN = (-0.9, 0.0, dk.y - 0.5, dk.y + 0.5)

    def in_dock_column(cos):
        x0, x1 = DOCK_OPEN[0], DOCK_OPEN[1]
        return all(c.z >= dorsal_z(c.y) - 1e-4 and x0 - 1e-4 <= c.x <= x1 + 1e-4 for c in cos)

    def split_dock(bm, grid):
        y0, y1 = DOCK_OPEN[2], DOCK_OPEN[3]
        for key in [k_ for k_ in grid if k_[0] != "cap"]:
            f = grid[key]
            cos = [v.co.copy() for v in f.verts]
            ys_ = [c.y for c in cos]
            if not in_dock_column(cos) or max(ys_) <= y0 + 1e-4 or min(ys_) >= y1 - 1e-4:
                continue
            pieces = [f]
            for yc in (y0, y1):
                if min(ys_) + 1e-4 < yc < max(ys_) - 1e-4:
                    for pc in list(pieces):
                        before = set(bm.faces)
                        bmesh.ops.bisect_plane(bm, geom=[pc] + list(pc.edges) + list(pc.verts),
                                               plane_co=(0.0, yc, 0.0), plane_no=(0.0, 1.0, 0.0))
                        pieces += [g for g in bm.faces if g not in before]
            del grid[key]
            kept = 0
            for pc in pieces:
                cy = pc.calc_center_median().y
                if y0 < cy < y1:
                    grid[(key[0], key[1], "dock", len(grid))] = pc
                else:
                    grid[key if kept == 0 else (key[0], key[1], "rest", kept)] = pc
                    kept += 1

    def drop_openings(key, cos):
        if len(key) > 2 and key[2] == "dock":
            return True
        return drop_cockpit(key, cos)

    me, tops = panel_loft(ename("HULL", "Fuselage"), secs, MAT["hull"], lip_fuselage,
                          drop=drop_openings, split=split_dock)
    fus = place(me.name, me)
    fus["kestrel_skin"] = True
    fus["cockpit_opening"] = "flat dorsal panels |X|<=0.9 between fuselage stations overlapping R01 Y span"
    fus["dock_opening"] = "flat dorsal panel column X -0.9..0, Y %.2f..%.2f under EXT_DOCK_Collar" % DOCK_OPEN[2:]
    STATS["dock_opening_xy"] = [list(DOCK_OPEN[:2]), [round(DOCK_OPEN[2], 3), round(DOCK_OPEN[3], 3)]]
    PANEL_RAISE["fuselage"] = tops
    STATS["hull_panels_fuselage"] = len(tops)
    STATS["hull_panels_whipple_sunfacing"] = nwhip[0]

    # cockpit opening coaming: square-section rim following the open skin edge
    ys = sorted(dropped_y)
    ya, yb = ys[0], ys[-1]
    STATS["cockpit_opening_xyz"] = [[-0.9, 0.9], [round(ya, 3), round(yb, 3)],
                                    [round(dorsal_z(y), 3) for y in (ya, yb)]]

    def coaming(bm):
        side = [(B.fuselage_section(y)[3], y, dorsal_z(y) - 0.012) for y in ys]
        loop = side + [(-x, y, z) for (x, y, z) in reversed(side)]
        add_tube(bm, loop, 0.026, 4, closed=True)
    part(ename("HULL", "CockpitCoaming"), MAT["metal"], coaming, smooth=None)

    # dock opening rim: same square-section tube along the open skin edge (task 7)
    def dock_rim(bm):
        x0, x1, y0, y1 = DOCK_OPEN
        loop = [(x0, y0, dorsal_z(y0) - 0.012), (x1, y0, dorsal_z(y0) - 0.012),
                (x1, y1, dorsal_z(y1) - 0.012), (x0, y1, dorsal_z(y1) - 0.012)]
        add_tube(bm, loop, 0.026, 4, closed=True)
    part(ename("HULL", "DockOpeningRim"), MAT["metal"], dock_rim, smooth=None)

    # ---- sponsons (mirror pair, separate meshes)
    sp_ys = [0.0, -0.5, -1.0] + [-1.0 - 0.75 * i for i in range(1, 9)] + [-7.6, -8.2]
    nsp = 0
    for side, sgn in SIDES:
        def lip_sponson(key, c, n):
            if key[0] == "cap":
                return LIP_PLAIN
            return whipple_level(*key) if n.z > 0.35 else LIP_PLAIN
        secs = []
        for y in sp_ys:
            _y, cx, hw, hh, cz = sponson_section(y)
            ch = min(B.SPONSON_CHAMFER, 0.45 * hw, 0.45 * hh)
            secs.append((y, split_ring(C.octagon_ring(hw, hh, ch, sgn * cx, cz))))
        me, tops = panel_loft(ename("HULL", "Sponson", side), secs, MAT["hull"], lip_sponson)
        o = place(me.name, me)
        o["kestrel_skin"] = True
        nsp += len(tops)
    STATS["hull_panels_sponsons"] = nsp
    TREES.clear()

    # ---- access hatches (2 sizes, linked duplicates)
    def hatch_build(w, l):
        def b(bm):
            add_box(bm, w, l, 0.03, Tm((0, 0, 0.005)), bevel=0.008, segs=2)
            add_box(bm, 0.12, 0.05, 0.02, Tm((0, l / 2 - 0.08, 0.025)), bevel=0.004)   # latch
            for x in (-w / 2 + 0.07, w / 2 - 0.07):                                   # hinges
                add_cyl(bm, 0.016, 0.09, 8, Tm((x, -l / 2 - 0.004, 0.012)) @ R(PI / 2, 'Y'))
            for sx in (-1, 1):                                                       # flush screws
                for sy in (-1, 1):
                    add_cyl(bm, 0.012, 0.01, 6, Tm((sx * (w / 2 - 0.03), sy * (l / 2 - 0.03), 0.022)))
        return b
    big = mesh(ename("HULL", "AccessHatch"), MAT["hull"], hatch_build(0.42, 0.55))
    small = mesh(ename("HULL", "AccessPanel"), MAT["hull"], hatch_build(0.28, 0.32))
    spots = [  # (mesh, point guess, normal guess, along)
        (big, (1.8, -1.5, 0.66), (1, 0, 0)), (big, (-1.8, -1.5, 0.66), (-1, 0, 0)),
        (big, (1.8, -7.4, 0.66), (1, 0, 0)), (big, (-1.8, -7.4, 0.66), (-1, 0, 0)),
        (big, (1.8, 1.5, 0.25), (1, 0, 0)),
        (big, (1.25, 8.6, -0.35), (1, 0, 0)), (big, (-1.25, 8.6, -0.35), (-1, 0, 0)),
        (small, (0.0, -0.6, 1.9), (0, 0, 1)), (small, (-0.4, -3.7, 1.9), (0, 0, 1)),
        (small, (2.4, -1.2, -0.75), (1, 0, 0)), (small, (-2.4, -1.2, -0.75), (-1, 0, 0)),
        (small, (0.45, 5.6, -1.9), (0, 0, -1)), (small, (-0.45, -4.6, -1.9), (0, 0, -1)),
    ]
    for idx, (mesh_src, p, n) in enumerate(spots, 1):
        hit, nrm = surf(p, n)
        kind = "AccessHatch" if mesh_src is big else "AccessPanel"
        inst(ename("HULL", f"{kind}{idx:02d}"), mesh_src, loc=hit, rot=frame(nrm))

    # ---- EVA handrails (yellow)
    r10 = handrail_mesh(ename("HULL", "Handrail10"), 1.0)
    r06 = handrail_mesh(ename("HULL", "Handrail06"), 0.6)
    rails = [  # (name part, side, mesh, point, normal, along)
        ("HandrailHatchFwd", "L", r10, (-1.8, 2.35, 0.25), (-1, 0, 0), (0, 0, 1)),
        ("HandrailHatchAft", "L", r10, (-1.8, 0.65, 0.25), (-1, 0, 0), (0, 0, 1)),
        ("HandrailDock", None, r10, (0.5, 1.55, 1.9), (0, 0, 1), (0, 1, 0)),
        ("HandrailDockAft", None, r06, (-0.5, 0.55, 1.9), (0, 0, 1), (1, 0, 0)),
        ("HandrailNose", "L", r10, (-1.8, 7.0, 0.7), (-1, 0, 0), (0, 1, 0)),
        ("HandrailNose", "R", r10, (1.8, 7.0, 0.7), (1, 0, 0), (0, 1, 0)),
        ("HandrailNoseUpper", "L", r06, (-1.4, 9.4, 0.3), (-1, 0, 0.35), (0, 1, 0)),
        ("HandrailNoseUpper", "R", r06, (1.4, 9.4, 0.3), (1, 0, 0.35), (0, 1, 0)),
        ("HandrailSponsonAft", "L", r10, (-2.4, -1.6, -0.35), (-1, 0, 0), (0, 1, 0)),
        ("HandrailSponsonAft", "R", r10, (2.4, -1.6, -0.35), (1, 0, 0), (0, 1, 0)),
        ("HandrailAftBulkhead", None, r10, (0.0, -11.0, 0.15), (0, -1, 0), (1, 0, 0)),
    ]
    for pname, side, me, p, n, along in rails:
        hit, nrm = surf(p, n)
        inst(ename("HULL", pname, side), me, loc=hit, rot=frame(nrm, along))


# =================================================================== 2 ENGINES
NAC_X, NAC_Z = 1.20, -0.50
ENG_Y = -10.9            # nacelle aft bulkhead / thrust structure plane


def bell_radius(z):
    """Bell inner contour in the gimbal frame (z along exhaust): throat 0.20 -> exit 0.95."""
    u = min(max((z - 0.20) / 0.75, 0.0), 1.0)
    return 0.12 + 0.52 * (1.0 - (1.0 - u) ** 1.8)


def build_engines():
    # ---- nacelle: one panelled loft in local coords, linked L/R
    def a_of(y):
        return 0.60 + (0.70 - 0.60) * (-8.6 - y) / 2.2

    ys = [-8.6, -9.15, -9.7, -10.25, -10.8]
    secs = [(y, split_ring(C.octagon_ring(a_of(y), a_of(y), a_of(y) * 0.586))) for y in ys]
    secs.append((-10.9, split_ring(C.octagon_ring(0.74, 0.74, 0.74 * 0.586))))
    me, _tops = panel_loft(ename("ENGINES", "Nacelle"), secs, MAT["dark"],
                           lambda key, c, n: LIP_PLAIN if key[0] == "cap" else
                           (LIP_WHIPPLE[0] if (key[0] + key[1]) % 3 == 0 else LIP_PLAIN))
    for side, sgn in SIDES:
        o = inst(ename("ENGINES", "Nacelle", side), me, loc=(sgn * NAC_X, 0, NAC_Z))
        o["kestrel_skin"] = True
    TREES.clear()

    # ---- louvres on the exposed nacelle flanks (linked)
    def louvre(bm):
        w, l = 0.30, 0.46
        add_box(bm, w, l, 0.02, Tm((0, 0, 0.01)), bevel=0.005)            # frame plate
        for k in range(7):
            y = -l / 2 + 0.06 + k * (l - 0.12) / 6
            add_box(bm, w - 0.05, 0.05, 0.008, Tm((0, y, 0.03)) @ R(math.radians(35), 'X'))
    lv = mesh(ename("ENGINES", "Louvre"), MAT["dark"], louvre, None)
    for side, sgn in SIDES:
        for i, y in enumerate((-9.95, -10.5), 1):
            hit, nrm = surf((sgn * (NAC_X + 0.66), y, NAC_Z), (sgn, 0, 0), only=["EXT_ENGINES_Nacelle"])
            inst(ename("ENGINES", f"Louvre{i}", side), lv, loc=hit, rot=frame(nrm))
        for i, y in enumerate((-9.95, -10.5), 3):
            p = (sgn * (NAC_X + 0.5), y, NAC_Z - 0.5)
            hit, nrm = surf(p, (sgn, 0, -1), only=["EXT_ENGINES_Nacelle"])
            inst(ename("ENGINES", f"Louvre{i}", side), lv, loc=hit, rot=frame(nrm))

    # ---- thrust structure (local +Z = exhaust = world -Y)
    def thrust(bm):
        add_revolve(bm, [(0.40, -0.03), (0.60, -0.03), (0.60, 0.03), (0.40, 0.03)], 32, closed=True)
        add_torus(bm, 0.50, 0.045, 32, 8, Tm((0, 0, 0.05)))
        for k in range(6):
            a = 2 * PI * k / 6
            b = a + PI / 6
            p0 = (0.48 * math.cos(a), 0.48 * math.sin(a), 0.06)
            p1 = (0.25 * math.cos(b), 0.25 * math.sin(b), 0.10)
            add_tube(bm, [p0, p1], 0.022, 8)
            add_box(bm, 0.07, 0.07, 0.05, Tm(p0) @ R(a, 'Z'), bevel=0.006)
        add_revolve(bm, [(0.22, 0.07), (0.30, 0.07), (0.30, 0.13), (0.22, 0.13)], 32, closed=True)
    ts_me = mesh(ename("ENGINES", "ThrustStructure"), MAT["metal"], thrust, 35)

    def gimbal(bm):
        add_torus(bm, 0.27, 0.03, 24, 8)
        for a in (0, PI / 2, PI, 1.5 * PI):
            add_cyl(bm, 0.035, 0.10, 24, Tm((0.29 * math.cos(a), 0.29 * math.sin(a), 0))
                    @ R(PI / 2, 'Y' if a in (0, PI) else 'X'))
    gb_me = mesh(ename("ENGINES", "Gimbal"), MAT["metal"], gimbal, 35)

    def bell(bm):
        outer = [(0.20, -0.14), (0.215, -0.12), (0.215, 0.06), (0.17, 0.14)]
        zs = [0.20 + 0.75 * (i / 16) ** 1.3 for i in range(17)]
        outer += [(bell_radius(z) + 0.016, z) for z in zs]
        inner = [(bell_radius(z), z) for z in reversed(zs)]
        inner += [(0.15, 0.12), (0.17, 0.02), (0.0001, 0.02)]
        add_revolve(bm, outer + [(0.655, 0.955)] + inner, 32, closed=False, cap0=False)
        # chamber dome (closes the profile start) and the regen manifold / hoops
        add_revolve(bm, [(0.0001, -0.20), (0.12, -0.18), (0.20, -0.14)], 32)
        add_torus(bm, 0.225, 0.035, 32, 8, Tm((0, 0, 0.0)))
        add_torus(bm, bell_radius(0.94) + 0.03, 0.022, 32, 8, Tm((0, 0, 0.93)))
        add_torus(bm, bell_radius(0.60) + 0.022, 0.016, 32, 6, Tm((0, 0, 0.60)))
        add_revolve(bm, [(bell_radius(0.45) + 0.01, 0.42), (bell_radius(0.45) + 0.04, 0.43),
                         (bell_radius(0.45) + 0.04, 0.48), (bell_radius(0.48) + 0.01, 0.49)], 32,
                    closed=True)
    bl_me = mesh(ename("ENGINES", "Bell"), MAT["metal"], bell, 35, recalc=False)

    # actuators: anchored on the thrust ring, pushing on the bell collar
    def act_ends(theta):
        p0 = Vector((0.47 * math.cos(theta), 0.47 * math.sin(theta), 0.08))
        rr = bell_radius(0.455) + 0.05
        p1 = Vector((rr * math.cos(theta), rr * math.sin(theta), 0.10 + 0.455))
        return p0, p1
    p0, p1 = act_ends(0.0)
    L = (p1 - p0).length

    def actuator(bm):
        add_cyl(bm, 0.045, L * 0.55, 24, z0=0.03)
        add_cyl(bm, 0.022, L * 0.5, 24, z0=L * 0.5 - 0.03)
        for z, s in ((0.0, 1), (L, -1)):                          # clevis yokes
            add_box(bm, 0.10, 0.04, 0.05, Tm((0, 0, z + s * 0.01)), bevel=0.006)
            for x in (-0.04, 0.04):
                add_box(bm, 0.014, 0.05, 0.07, Tm((x, 0, z)), bevel=0.003)
            add_cyl(bm, 0.012, 0.11, 24, Tm((0, 0, z)) @ R(PI / 2, 'Y'))
    ac_me = mesh(ename("ENGINES", "Actuator"), MAT["metal"], actuator, 35)

    def feeds(bm):
        for theta in (math.radians(210), math.radians(330)):
            c, s = math.cos(theta), math.sin(theta)
            pts = [(0.56 * c, 0.56 * s, -0.05), (0.56 * c, 0.56 * s, 0.02), (0.42 * c, 0.42 * s, 0.10),
                   (0.30 * c, 0.30 * s, 0.10), (0.24 * c, 0.24 * s, 0.10)]
            add_tube(bm, fillet(pts, 0.05, 3), 0.032, 8)
            for z in (0.0,):
                add_cyl(bm, 0.045, 0.03, 8, Tm((0.56 * c, 0.56 * s, z)))      # bellows flange
    fd_me = mesh(ename("ENGINES", "FeedLines"), MAT["metal"], feeds, 50)

    for side, sgn in SIDES:
        ts = inst(ename("ENGINES", "ThrustStructure", side), ts_me,
                  loc=(sgn * NAC_X, ENG_Y, NAC_Z), rot=(PI / 2, 0, 0))
        gb = inst(ename("ENGINES", "Gimbal", side), gb_me, parent=ts, loc=(0, 0, 0.10))
        gb["gimbal_range_deg"] = 6.0
        inst(ename("ENGINES", "Bell", side), bl_me, parent=gb)
        for i, th in enumerate((0.0, PI / 2), 1):
            a0, a1 = act_ends(th)
            inst(ename("ENGINES", f"Actuator{i}", side), ac_me, parent=ts, loc=a0,
                 rot=frame(a1 - a0, (math.cos(th + PI / 2), math.sin(th + PI / 2), 0)))
        inst(ename("ENGINES", "FeedLines", side), fd_me, parent=ts)


BUILDERS["ENGINES"] = build_engines


# =================================================================== 3 RADIATORS
RAD_EDGES = [2.40, 3.69, 4.98, 6.25]
RAD_GAP = 0.03


def build_radiators():
    panels = []
    for i in range(3):
        x0 = RAD_EDGES[i] + (RAD_GAP if i else 0.0)
        x1 = RAD_EDGES[i + 1] - (RAD_GAP if i < 2 else 0.0)
        panels.append((x0, x1))

    knuckle = mesh(ename("RADIATORS", "Hinge"), MAT["metal"], lambda bm: (
        add_cyl(bm, 0.045, 0.24, 24, R(PI / 2, 'X')),
        add_box(bm, 0.16, 0.20, 0.016, Tm((-0.09, 0, 0.0)), bevel=0.004),
        add_box(bm, 0.16, 0.20, 0.016, Tm((0.09, 0, 0.0)), bevel=0.004)), 35)

    def header(bm):
        add_tube(bm, [(0, -1.92, 0), (0, 1.92, 0)], 0.028, 8)
        for y in (-1.5, -0.5, 0.5, 1.5):
            add_box(bm, 0.07, 0.04, 0.06, Tm((0, y, -0.02)), bevel=0.005)
    hdr = mesh(ename("RADIATORS", "Header"), MAT["metal"], header, 50)

    def jumper(bm):
        pts = fillet([(-0.14, 0, 0), (-0.10, 0, 0.12), (0.10, 0, 0.12), (0.14, 0, 0)], 0.06, 3)
        add_tube(bm, pts, 0.02, 8)
    jmp = mesh(ename("RADIATORS", "Jumper"), MAT["metal"], jumper, 50)

    s0 = Vector((2.42, 0, -1.0))
    s1 = Vector((3.35, 0, -0.045))
    SL = (s1 - s0).length

    def strut(bm):
        add_cyl(bm, 0.05, SL * 0.55, 24, z0=0.04)
        add_cyl(bm, 0.028, SL * 0.5, 24, z0=SL * 0.5 - 0.04)
        for z in (0.0, SL):
            add_box(bm, 0.12, 0.05, 0.06, Tm((0, 0, z)), bevel=0.006)
            add_cyl(bm, 0.014, 0.14, 24, Tm((0, 0, z)) @ R(PI / 2, 'Y'))
    st = mesh(ename("RADIATORS", "Strut"), MAT["metal"], strut, 35)
    navh = mesh(ename("RADIATORS", "NavLightHousing"), MAT["metal"], lambda bm: (
        add_revolve(bm, [(0.05, 0.0), (0.05, 0.03), (0.035, 0.06), (0.0001, 0.07)], 24, cap0=True)), 35)

    pmeshes = []
    for i, (x0, x1) in enumerate(panels, 1):
        w = x1 - x0
        pm = mesh(ename("RADIATORS", f"Panel{i}"), MAT["rad"],
                  lambda bm, w=w: add_box(bm, w, 4.0, 0.05, bevel=0.008, segs=1), None)

        def frm(bm, w=w):
            for sx in (-1, 1):
                add_box(bm, 0.05, 4.04, 0.08, Tm((sx * (w / 2 - 0.025), 0, 0)), bevel=0.006)
            for sy in (-1, 1):
                add_box(bm, w - 0.1, 0.05, 0.08, Tm((0, sy * 1.995, 0)), bevel=0.006)
            add_box(bm, w - 0.1, 0.04, 0.07, Tm((0, 0, 0)), bevel=0.006)          # mid rib
        fm = mesh(ename("RADIATORS", f"Panel{i}Frame"), MAT["metal"], frm, None)

        def tubes(bm, w=w):
            for zz in (0.031, -0.031):
                for k in range(16):
                    y = -1.86 + k * (3.72 / 15)
                    if abs(y) < 0.05:
                        continue
                    add_tube(bm, [(-w / 2 + 0.06, y, zz), (w / 2 - 0.06, y, zz)], 0.011, 8)
        tm = mesh(ename("RADIATORS", f"Panel{i}Tubes"), MAT["rad"], tubes, 50)
        pmeshes.append((pm, fm, tm, x0, x1))

    for side, sgn in SIDES:
        for i, (pm, fm, tm, x0, x1) in enumerate(pmeshes, 1):
            p = inst(ename("RADIATORS", f"Panel{i}", side), pm, loc=(sgn * (x0 + x1) / 2, -5.0, 0.0))
            inst(ename("RADIATORS", f"Panel{i}Frame", side), fm, parent=p)
            inst(ename("RADIATORS", f"Panel{i}Tubes", side), tm, parent=p)
            if i > 1:
                inst(ename("RADIATORS", f"Panel{i}Header", side), hdr,
                     loc=(sgn * (x0 + 0.08), -5.0, 0.075))
        n = 0
        for xj in RAD_EDGES[:3]:
            for y in (-6.4, -5.0, -3.6):
                n += 1
                inst(ename("RADIATORS", f"Hinge{n}", side), knuckle, loc=(sgn * xj, y, 0.0))
        n = 0
        for xj in RAD_EDGES[1:3]:
            for y in (-6.75, -3.25):
                n += 1
                inst(ename("RADIATORS", f"Jumper{n}", side), jmp, loc=(sgn * xj, y, 0.075))
        for i, y in enumerate((-6.2, -3.8), 1):
            a = Vector((sgn * s0.x, y, s0.z))
            b = Vector((sgn * s1.x, y, s1.z))
            inst(ename("RADIATORS", f"Strut{i}", side), st, loc=a, rot=frame(b - a, (0, 1, 0)))
        inst(ename("RADIATORS", "NavLightHousing", side), navh, loc=(sgn * 6.25, -5.0, 0.0),
             rot=frame((sgn, 0, 0)))
        # coolant manifolds (supply / return), mirrored geometry -> separate meshes
        sup = [(sgn * 1.70, -7.5, 0.58), (sgn * 2.02, -7.5, 0.58), (sgn * 2.30, -7.5, 0.26),
               (sgn * 2.47, -7.25, 0.12), (sgn * 2.47, -3.0, 0.12)]
        ret = [(sgn * 1.70, -7.85, 0.48), (sgn * 2.02, -7.85, 0.48), (sgn * 2.34, -7.85, 0.20),
               (sgn * 2.58, -7.5, 0.10), (sgn * 2.58, -3.0, 0.10)]
        for nm, pts, r in (("ManifoldSupply", sup, 0.040), ("ManifoldReturn", ret, 0.034)):
            def mb(bm, pts=pts, r=r):
                add_tube(bm, fillet(pts, 0.12, 4), r, 8)
                for y in (-6.0, -4.5):
                    add_box(bm, 0.08, 0.05, 0.08, Tm((pts[-1][0], y, pts[-1][2] - 0.03)), bevel=0.006)
            part(ename("RADIATORS", nm, side), MAT["metal"], mb, smooth=50)


BUILDERS["RADIATORS"] = build_radiators


# =================================================================== 4 RCS
def build_rcs():
    def nozzle(bm):
        prof = [(0.022, -0.01), (0.026, 0.0), (0.05, 0.085), (0.054, 0.09),
                (0.046, 0.09), (0.020, 0.012), (0.012, 0.0), (0.012, -0.01)]
        add_revolve(bm, prof, 8, closed=True)
    nz = mesh(ename("RCS", "Nozzle"), MAT["metal"], nozzle, 50)

    def housing(bm):
        add_box(bm, 0.24, 0.24, 0.22, Tm((0, 0, 0.03)), bevel=0.02, segs=2)
        add_box(bm, 0.30, 0.30, 0.03, Tm((0, 0, 0.01)), bevel=0.008)
        for x in (-0.11, 0.11):
            for y in (-0.11, 0.11):
                add_cyl(bm, 0.012, 0.012, 6, Tm((x * 1.2, y * 1.2, 0.03)))
    hs = mesh(ename("RCS", "Housing"), MAT["metal"], housing, 35)
    sc = mesh(ename("RCS", "Scorch"), MAT["scorch"],
              lambda bm: add_revolve(bm, [(0.34, -0.03), (0.34, 0.012), (0.30, 0.02)], 24, cap0=True,
                                     cap1=True), 35)
    noz_pose = [((0.12, 0, 0.09), (1, 0, 0)), ((-0.12, 0, 0.09), (-1, 0, 0)),
                ((0, 0.12, 0.09), (0, 1, 0)), ((0, -0.12, 0.09), (0, -1, 0))]
    for so in sorted(o for o in bpy.data.objects.keys() if o.startswith("SOCKET_RCS_Quad_")):
        mw = sock(so)
        loc = mw.translation
        n0 = (mw.to_3x3() @ Vector((0, 0, 1))).normalized()
        hit, nrm = surf(loc, n0, only=["EXT_HULL_Fuselage"])
        tag = so[len("SOCKET_RCS_"):]               # Quad_FwdUp_L
        base, side = tag.rsplit("_", 1)
        q = inst(ename("RCS", base, side), hs, loc=hit, rot=frame(nrm, (0, 1, 0)))
        q["socket"] = so
        inst(ename("RCS", base + "_Scorch", side), sc, parent=q)
        for k, (p, d) in enumerate(noz_pose, 1):
            inst(ename("RCS", f"{base}_Nozzle{k}", side), nz, parent=q, loc=p,
                 rot=frame(d, (0, 0, 1)))


BUILDERS["RCS"] = build_rcs


# =================================================================== 5 CANOPY
def canopy_rings():
    """Per station: y and the 7-point ring [bR, p1R, p2R, c, p2L, p1L, bL] where b is
    where the canopy flank meets the hull skin."""
    out = []
    for (y, ws, top) in B.CANOPY_STATIONS:
        k = (top - 1.20) / (2.30 - 1.20)
        raw = [(x * ws, 1.20 + (z - 1.20) * k) for (x, z) in B.CANOPY_RING]
        p0, p1, p2 = Vector(raw[0]), Vector(raw[1]), Vector(raw[2])
        if inside_fuselage(p1.x, p1.y, y):
            b = p0
        else:
            lo, hi = 0.0, 3.0
            for _ in range(40):
                mid = (lo + hi) / 2
                q = p1 + (p0 - p1) * mid
                if inside_fuselage(q.x, q.y, y):
                    hi = mid
                else:
                    lo = mid
            b = p1 + (p0 - p1) * (hi + 0.02)
        ring = [(b.x, b.y), (p1.x, p1.y), (p2.x, p2.y), (0.0, p2.y),
                (-p2.x, p2.y), (-p1.x, p1.y), (-b.x, b.y)]
        out.append((y, ring))
    return out


def fairing_end_points(y2, r2, y3, r3):
    """PATCH (interior task): for each fairing line r2[i]@y2 -> r3[i]@y3, the 3D point
    where it enters the fuselage (None if it starts inside, i.e. at the canopy base)."""
    out = []
    for p, q in zip(r2, r3):
        a, b = Vector((p[0], y2, p[1])), Vector((q[0], y3, q[1]))
        if inside_fuselage(a.x, a.z, a.y):
            out.append(None)
            continue
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            c = a.lerp(b, mid)
            if inside_fuselage(c.x, c.z, c.y):
                hi = mid
            else:
                lo = mid
        out.append(a.lerp(b, min(1.0, hi + 0.02)))     # 2 % into the skin, like the canopy base
    return out


def build_canopy():
    rings = canopy_rings()
    (y0, r0), (y1, r1), (y2, r2), (y3, r3) = rings

    def V(y, p, out=0.0):
        return Vector((p[0], y, p[1] + out))

    def pane(name, quads, thick=0.018):
        def b(bm):
            vm = {}

            def vv(co):
                key = tuple(round(c, 5) for c in co)
                if key not in vm:
                    vm[key] = bm.verts.new(co)
                return vm[key]
            fs = []
            for q in quads:
                fs.append(bm.faces.new([vv(c) for c in q]))
            bm.normal_update()
            ctr = Vector((0, sum(q[0].y for q in quads) / len(quads), 1.0))
            for f in fs:
                if f.normal.dot(f.calc_center_median() - ctr) < 0:
                    f.normal_flip()
            bm.normal_update()
            # thicken OUTWARD (keeps interior clearance)
            ext = bmesh.ops.extrude_face_region(bm, geom=fs)
            nv = [e for e in ext['geom'] if isinstance(e, bmesh.types.BMVert)]
            for v in nv:
                nrm = Vector((0, 0, 0))
                for f in v.link_faces:
                    if f in fs:
                        nrm += f.normal
                if nrm.length == 0:
                    nrm = v.normal
                v.co += nrm.normalized() * thick
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        return part(name, MAT["glass"], b, smooth=None, recalc=False)

    def strip(ya, ra, yb, rb, i0, i1):
        return [[V(ya, ra[i]), V(ya, ra[i + 1]), V(yb, rb[i + 1]), V(yb, rb[i])] for i in range(i0, i1)]

    pane(ename("CANOPY", "Windscreen", "R"), strip(y0, r0, y1, r1, 0, 3))
    pane(ename("CANOPY", "Windscreen", "L"), strip(y0, r0, y1, r1, 3, 6))
    pane(ename("CANOPY", "Side", "R"), strip(y1, r1, y2, r2, 0, 2))
    pane(ename("CANOPY", "Side", "L"), strip(y1, r1, y2, r2, 4, 6))
    pane(ename("CANOPY", "Top"), strip(y1, r1, y2, r2, 2, 4))

    # aft fairing (painted metal) closing the cab down into the hull.
    # PATCH (interior task, approved): each fairing line stops where it enters the
    # fuselage, so the fairing no longer dips into R02 (it used to close at Y 3.2, Z 1.2).
    def fairing(bm):
        ra = [bm.verts.new(V(y2, p)) for p in r2]
        rb = []
        for va, q in zip(ra, fairing_end_points(y2, r2, y3, r3)):
            rb.append(va if q is None else bm.verts.new(q))
        for i in range(6):
            quad = []
            for v in (ra[i], ra[i + 1], rb[i + 1], rb[i]):
                if v not in quad:
                    quad.append(v)
            if len(quad) >= 3:
                bm.faces.new(quad)
        cap = []
        for v in reversed(rb):
            if v not in cap:
                cap.append(v)
        bm.faces.new(cap)
        bm.faces.new(ra)
    part(ename("CANOPY", "AftFairing"), MAT["hull"], fairing, smooth=None)

    # frames: square-section bars on every pane edge
    def frames(bm):
        bar = 0.045
        for (y, r) in ((y0, r0), (y1, r1), (y2, r2)):
            add_tube(bm, [V(y, p, 0.01) for p in r], bar, 4, cap=True)
        for i in (1, 2, 4, 5):
            add_tube(bm, [V(y0, r0[i], 0.01), V(y1, r1[i], 0.01), V(y2, r2[i], 0.01)], bar, 4)
        add_tube(bm, [V(y0, r0[3], 0.01), V(y1, r1[3], 0.01)], bar, 4)
    part(ename("CANOPY", "Frame"), MAT["metal"], frames, smooth=None)

    def base(bm):
        for i in (0, 6):
            pts = [V(y, r[i], 0.0) for (y, r) in rings[:3]]     # PATCH: rail ends at the fairing (Y 4.4)
            add_tube(bm, pts, 0.055, 6)
        # sill bolts every ~0.35 m along the base rail
    part(ename("CANOPY", "BaseFrame"), MAT["metal"], base, smooth=None)


BUILDERS["CANOPY"] = build_canopy


# =================================================================== 6 GEAR
def build_gear():
    asm = empty(ename("GEAR", "Assembly"), size=0.6)
    asm["note"] = "deployed; hide this node to stow the gear"

    # parts in each leg's local frame: origin at the top pivot, Z up

    def build_leg(prefix, side, top, r_out, len_out, r_in, len_in, foot):
        z_out_bot = -len_out
        z_in_bot = z_out_bot - len_in + 0.12
        meshes = {}

        def leg(bm):
            add_cyl(bm, r_out, len_out, 24, z0=-len_out)
            add_revolve(bm, [(r_out + 0.025, -len_out - 0.02), (r_out + 0.025, -len_out + 0.06)],
                        24, cap0=True, cap1=True)                                         # gland nut
            add_revolve(bm, [(r_out + 0.03, -0.12), (r_out + 0.03, 0.0)], 24, cap0=True, cap1=True)
            add_cyl(bm, 0.05, 0.42, 24, Tm((0, 0, -0.05)) @ R(PI / 2, 'Y'))               # trunnion
            add_box(bm, 0.06, 0.05, 0.10, Tm((0, r_out + 0.02, -len_out + 0.06)), bevel=0.008)
        meshes["Leg"] = mesh(ename("GEAR", prefix + "Leg"), MAT["metal"], leg, 35)

        def slider(bm):
            add_cyl(bm, r_in, len_in, 24, z0=z_out_bot - len_in + 0.12)
            add_revolve(bm, [(r_in + 0.02, z_in_bot - 0.03), (r_in + 0.02, z_in_bot + 0.06)], 24,
                        cap0=True, cap1=True)
            add_box(bm, 0.06, 0.05, 0.08, Tm((0, r_in + 0.03, z_in_bot + 0.04)), bevel=0.008)
        meshes["Slider"] = mesh(ename("GEAR", prefix + "Slider"), MAT["metal"], slider, 35)

        knee_y = r_out + 0.20
        zt, zb = -len_out + 0.06, z_in_bot + 0.04
        zk = (zt + zb) / 2

        def link_bar(bm, za, zb2):
            b = Vector((0, knee_y, zk))
            a = Vector((0, r_out + 0.03, za)) if zb2 is None else Vector((0, r_in + 0.03, zb2))
            d = (b - a)
            L = d.length
            M = Tm(a) @ frame(Vector((1, 0, 0)), d).to_4x4()
            add_prism(bm, [(-0.025, 0), (0.025, 0), (0.02, L), (-0.02, L)], 0.025, M)
            add_cyl(bm, 0.022, 0.06, 24, Tm(a) @ R(PI / 2, 'Y'))
            add_cyl(bm, 0.022, 0.06, 24, Tm(b) @ R(PI / 2, 'Y'))
        meshes["TorqueLinkUpper"] = mesh(ename("GEAR", prefix + "TorqueLinkUpper"), MAT["metal"],
                                         lambda bm: link_bar(bm, zt, None), 35)
        meshes["TorqueLinkLower"] = mesh(ename("GEAR", prefix + "TorqueLinkLower"), MAT["metal"],
                                         lambda bm: link_bar(bm, None, zb), 35)

        def hyd(bm):
            x = r_out + 0.02
            add_tube(bm, fillet([(x * 0.7, -x, 0.1), (x * 0.7, -x, zt - 0.05), (x * 0.7, -x - 0.10, zk),
                                 (x * 0.7, -r_in - 0.03, zb + 0.06)], 0.06, 3), 0.012, 6)
            add_tube(bm, fillet([(-x * 0.7, -x, 0.1), (-x * 0.7, -x, zt - 0.05)], 0.05, 2), 0.012, 6)
            for z in (-0.3, -0.6):
                if z > zt:
                    add_box(bm, 0.2, 0.03, 0.03, Tm((0, -x, z)), bevel=0.005)
        meshes["HydLines"] = mesh(ename("GEAR", prefix + "HydLines"), MAT["dark"], hyd, 50)

        def brace(bm):
            a = (0, 0, -len_out * 0.45)
            b = (0, -0.9, 0.05)
            add_tube(bm, [a, b], 0.04, 8)
            add_cyl(bm, 0.05, 0.12, 24, Tm(b) @ R(PI / 2, 'Y'))
            add_revolve(bm, [(r_out + 0.02, -len_out * 0.45 - 0.04), (r_out + 0.02, -len_out * 0.45 + 0.04)],
                        24, cap0=True, cap1=True)
        meshes["DragBrace"] = mesh(ename("GEAR", prefix + "DragBrace"), MAT["metal"], brace, 40)

        if foot == "wheel":
            wz = C.Z_GROUND - top + 0.28          # local axle height (tyre radius 0.28)

            def fork(bm):
                add_box(bm, 0.36, 0.10, 0.06, Tm((0, 0, zb - 0.02)), bevel=0.01, segs=2)
                for x in (-0.15, 0.15):
                    add_box(bm, 0.04, 0.10, zb - 0.02 - wz + 0.06, Tm((x, 0, (zb - 0.02 + wz) / 2)),
                            bevel=0.008)
                add_cyl(bm, 0.03, 0.36, 24, Tm((0, 0, wz)) @ R(PI / 2, 'Y'))
            meshes["Fork"] = mesh(ename("GEAR", prefix + "Fork"), MAT["metal"], fork, 35)

            def wheel(bm):
                tyre = [(0.20, -0.09), (0.25, -0.09), (0.275, -0.07), (0.28, -0.03), (0.28, 0.03),
                        (0.275, 0.07), (0.25, 0.09), (0.20, 0.09)]
                add_revolve(bm, tyre, 32, closed=True, M=R(PI / 2, 'Y'))
                add_revolve(bm, [(0.05, -0.07), (0.20, -0.07), (0.20, 0.07), (0.05, 0.07)], 32,
                            closed=True, M=R(PI / 2, 'Y'), mi=1)
                add_cyl(bm, 0.07, 0.17, 24, R(PI / 2, 'Y'), mi=1)
            meshes["Wheel"] = mesh(ename("GEAR", prefix + "Wheel"), [MAT["dark"], MAT["metal"]], wheel, 35)
        else:
            fz = C.Z_GROUND - top                 # local ground height

            def footpad(bm):
                add_revolve(bm, [(0.0001, fz), (0.40, fz), (0.42, fz + 0.025), (0.40, fz + 0.05),
                                 (0.20, fz + 0.11), (0.09, fz + 0.13), (0.0001, fz + 0.13)], 32)
                add_revolve(bm, [(0.0001, fz + 0.12), (0.07, fz + 0.13), (0.085, fz + 0.18),
                                 (0.07, fz + 0.23), (0.0001, fz + 0.24)], 24)            # ball joint
                add_tube(bm, [(0, 0, zb), (0, 0, fz + 0.2)], r_in * 0.85, 8)
                for k in range(4):                                                       # pad ribs
                    a = PI / 4 + k * PI / 2
                    add_box(bm, 0.30, 0.02, 0.06, Tm((0.2 * math.cos(a), 0.2 * math.sin(a), fz + 0.08))
                            @ R(a, 'Z') @ R(-0.3, 'Y'), bevel=0.004)
            meshes["Footpad"] = mesh(ename("GEAR", prefix + "Footpad"), MAT["metal"], footpad, 35)
        return meshes

    nose = build_leg("Nose", None, -1.75, 0.10, 0.70, 0.07, 0.62, "wheel")
    main = build_leg("Main", "L", -1.60, 0.13, 0.85, 0.095, 0.72, "pad")
    for name, (x, y) in B.GEAR.items():
        side = name.split("_")[1] if "_" in name else None
        ms, top, prefix = (nose, -1.75, "Nose") if name == "Nose" else (main, -1.60, "Main")
        leg = inst(ename("GEAR", prefix + "Leg", side), ms["Leg"], parent=asm, loc=(x, y, top))
        leg["socket"] = f"SOCKET_Gear_Contact_{name}"
        for part_name, me in ms.items():
            if part_name == "Leg":
                continue
            loc = (0, 0, C.Z_GROUND - top + 0.28) if part_name == "Wheel" else (0, 0, 0)
            inst(ename("GEAR", prefix + part_name, side), me, parent=leg, loc=loc)


BUILDERS["GEAR"] = build_gear


# =================================================================== 7 WEAPONS
def build_weapons():
    # ---- chin turret: base -> yaw -> pitch -> twin barrels
    hit, nrm = surf((0, 8.7, -1.75), (0, 0, -1), only=["EXT_HULL_Fuselage"])
    base = part(ename("WEAPONS", "ChinTurret"), MAT["metal"], lambda bm: add_revolve(
        bm, [(0.0001, 0.05), (0.52, 0.05), (0.52, -0.01), (0.47, -0.03), (0.0001, -0.03)], 32,
        closed=False), loc=(0, 8.7, hit.z), rot=(PI, 0, 0), smooth=35)
    base["socket"] = "SOCKET_Turret_Chin"

    # children of the base live in its local frame: local +Z points DOWN (world -Z)
    def yaw_b(bm):
        add_revolve(bm, [(0.0001, 0.0), (0.44, 0.0), (0.44, 0.08), (0.40, 0.16), (0.30, 0.22),
                         (0.0001, 0.24)], 32)
        for x in (-0.24, 0.24):
            add_box(bm, 0.06, 0.30, 0.20, Tm((x, 0, 0.24)), bevel=0.012, segs=2)
    yaw = part(ename("WEAPONS", "ChinTurretYaw"), MAT["dark"], yaw_b, parent=base, loc=(0, 0, 0.03),
               smooth=35)

    def pitch_b(bm):
        # base frame is flipped: local -Y = world forward
        add_box(bm, 0.40, 0.42, 0.17, Tm((0, -0.04, 0)), bevel=0.02, segs=2)
        add_cyl(bm, 0.05, 0.54, 24, R(PI / 2, 'Y'))                     # trunnion
        add_box(bm, 0.32, 0.08, 0.13, Tm((0, -0.27, 0)), bevel=0.012, segs=2)  # mantlet
        add_box(bm, 0.12, 0.16, 0.06, Tm((0, 0.05, 0.10)), bevel=0.01)        # feed chute
    pitch = part(ename("WEAPONS", "ChinTurretPitch"), MAT["dark"], pitch_b, parent=yaw,
                 loc=(0, 0, 0.26), smooth=35)
    pitch["axis"] = "local X"

    def barrel(bm):
        add_cyl(bm, 0.055, 0.30, 24, R(-PI / 2, 'X'), z0=0.28)           # recoil sleeve
        add_cyl(bm, 0.033, 1.20, 24, R(-PI / 2, 'X'), z0=0.30)
        add_revolve(bm, [(0.02, 1.48), (0.05, 1.48), (0.05, 1.58), (0.02, 1.58)], 24, closed=True,
                    M=R(-PI / 2, 'X'))                                    # muzzle brake
        for z in (1.50, 1.54):
            add_box(bm, 0.11, 0.012, 0.03, Tm((0, z, 0)), bevel=0.003)
    bm_me = mesh(ename("WEAPONS", "ChinTurretBarrel"), MAT["metal"], barrel, 35)
    # base is flipped (rot X 180): local +Y = world -Y, so barrels point along local -Y
    for side, sgn in SIDES:
        inst(ename("WEAPONS", "ChinTurretBarrel", side), bm_me, parent=pitch,
             loc=(sgn * 0.10, 0, 0.0), rot=(0, 0, PI))

    # ---- sponson pylons + missiles (2 per sponson, at the hardpoint station Y -5.8)
    MZ = -1.98

    def pylon(top_z):
        h = MZ + 0.11 - top_z      # negative

        def b(bm):
            pts = [(-0.70, 0.0), (0.60, 0.0), (0.45, h), (-0.55, h)]
            yz_plane = Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1)))  # (y, z), thick X
            add_prism(bm, pts, 0.07, yz_plane)
            add_box(bm, 0.12, 0.9, 0.05, Tm((0, -0.05, h + 0.02)), bevel=0.01)      # ejector rack
            for y in (-0.35, 0.25):
                add_box(bm, 0.05, 0.04, 0.06, Tm((0, y, h - 0.01)), bevel=0.005)
        return b
    py_out = mesh(ename("WEAPONS", "PylonOuter"), MAT["dark"], pylon(-1.62), 35)
    py_in = mesh(ename("WEAPONS", "PylonInner"), MAT["dark"], pylon(-1.38), 35)

    def body(bm):
        add_cyl(bm, 0.11, 1.75, 24, R(-PI / 2, 'X'), z0=-1.0)
        add_revolve(bm, [(0.11, -1.0), (0.085, -1.06), (0.06, -1.08), (0.05, -1.04)], 24,
                    M=R(-PI / 2, 'X'), cap1=True)                         # boat tail / nozzle
        add_revolve(bm, [(0.112, 0.70), (0.112, 0.75)], 24, M=R(-PI / 2, 'X'))
        for y in (0.25, -0.35):
            add_box(bm, 0.03, 0.08, 0.03, Tm((0, y, 0.115)), bevel=0.005)   # lugs
    ms_body = mesh(ename("WEAPONS", "Missile"), MAT["hull"], body, 35)
    ms_seek = mesh(ename("WEAPONS", "MissileSeeker"), [MAT["dark"], MAT["glass"]], lambda bm: (
        add_revolve(bm, [(0.11, 0.0), (0.10, 0.12), (0.08, 0.22), (0.05, 0.29)], 24, M=R(-PI / 2, 'X')),
        add_revolve(bm, [(0.05, 0.29), (0.035, 0.32), (0.0001, 0.335)], 24, M=R(-PI / 2, 'X'), mi=1)), 35)

    def fins(bm):
        for k in range(4):
            a = PI / 4 + k * PI / 2
            M = R(a, 'Y')
            add_prism(bm, [(0.10, -0.98), (0.24, -0.90), (0.24, -0.78), (0.10, -0.62)], 0.012, M)
            add_prism(bm, [(0.10, 0.05), (0.15, 0.15), (0.15, 0.35), (0.10, 0.45)], 0.01, M)
    ms_fins = mesh(ename("WEAPONS", "MissileFins"), MAT["dark"], fins, None)
    n = 0
    for side, sgn in SIDES:
        for kind, x, pm in (("Outer", 1.925, py_out), ("Inner", 1.35, py_in)):
            top = -1.62 if kind == "Outer" else -1.38
            py = inst(ename("WEAPONS", f"Pylon{kind}", side), pm, loc=(sgn * x, -5.8, top))
            if kind == "Outer":
                py["socket"] = "SOCKET_Hardpoint_3_SponsonL" if side == "L" else "SOCKET_Hardpoint_4_SponsonR"
            n += 1
            m = inst(ename("WEAPONS", f"Missile_{n}"), ms_body, loc=(sgn * x, -5.8, MZ))
            m["pylon"] = py.name
            inst(ename("WEAPONS", f"Missile_{n}_Seeker"), ms_seek, parent=m, loc=(0, 0.75, 0))
            inst(ename("WEAPONS", f"Missile_{n}_Fins"), ms_fins, parent=m)

    # ---- ventral ordnance bay doors (closed) + surround
    def door(bm):
        w, l = 0.835, 4.50
        add_box(bm, w, l, 0.03, bevel=0.008, segs=2)
        for y in (-1.5, -0.5, 0.5, 1.5):
            add_box(bm, w - 0.1, 0.05, 0.02, Tm((0, y, -0.022)), bevel=0.005)
        add_box(bm, 0.05, l - 0.1, 0.02, Tm((w / 2 - 0.05, 0, -0.022)), bevel=0.005)
        for y in (-1.8, 0.0, 1.8):                                      # hinge knuckles (outer edge)
            add_cyl(bm, 0.025, 0.25, 24, Tm((w / 2 + 0.01, y, 0.0)) @ R(PI / 2, 'X'))
    dm = mesh(ename("WEAPONS", "BayDoor"), MAT["hull"], door, 35)
    zc = -1.90 - 0.02
    for side, sgn in SIDES:
        inst(ename("WEAPONS", "BayDoor", side), dm, loc=(sgn * 0.425, 0.30, zc),
             rot=(0, 0, PI) if sgn < 0 else (0, 0, 0))     # hinge on the outer edge
    part(ename("WEAPONS", "BayFrame"), MAT["metal"], lambda bm: (
        add_box(bm, 1.90, 0.06, 0.03, Tm((0, 2.58, zc))), add_box(bm, 1.90, 0.06, 0.03, Tm((0, -1.98, zc))),
        add_box(bm, 0.06, 4.62, 0.03, Tm((0.89, 0.30, zc))), add_box(bm, 0.06, 4.62, 0.03, Tm((-0.89, 0.30, zc)))),
        smooth=None)


BUILDERS["WEAPONS"] = build_weapons


# =================================================================== 8 SENSORS
def build_sensors():
    # ---- nose radar fairing (ogive cap over the flat nose face)
    hw, hh, zc, ch = B.fuselage_section(11.5)
    L, Rb = 0.34, 0.43
    prof = [(Rb * math.sqrt(max(1 - (z / L) ** 2, 0.0)) if z < L else 0.0001, z)
            for z in [L * (i / 10) ** 0.8 for i in range(11)]]
    prof[-1] = (0.0001, L)
    part(ename("SENSORS", "RadarFairing"), MAT["dark"], lambda bm: (
        add_revolve(bm, [(0.0001, -0.02), (Rb + 0.02, -0.02)] + [(Rb + 0.02, 0.03)] + prof[1:], 32)),
        loc=(0, 11.5, zc), rot=(-PI / 2, 0, 0), smooth=35)
    part(ename("SENSORS", "RadarFairingRing"), MAT["metal"], lambda bm: (
        add_revolve(bm, [(Rb - 0.01, -0.03), (Rb + 0.05, -0.03), (Rb + 0.05, 0.035), (Rb - 0.01, 0.035)],
                    32, closed=True)), loc=(0, 11.5, zc), rot=(-PI / 2, 0, 0), smooth=35)

    # ---- star trackers (cube + baffle + lens), on the upper nose chamfers
    def tracker(bm):
        add_box(bm, 0.20, 0.20, 0.16, Tm((0, 0, 0.08)), bevel=0.012, segs=2)
        M = Tm((0, 0.02, 0.14)) @ R(math.radians(-25), 'X')
        add_revolve(bm, [(0.06, 0.0), (0.075, 0.0), (0.10, 0.20), (0.09, 0.20), (0.06, 0.02)], 24,
                    M=M, closed=True)
        add_cyl(bm, 0.062, 0.01, 24, M @ Tm((0, 0, 0.02)), mi=1)
        for x in (-0.07, 0.07):
            add_box(bm, 0.03, 0.12, 0.012, Tm((x, 0, 0.006)), bevel=0.003)
    tr = mesh(ename("SENSORS", "StarTracker"), [MAT["metal"], MAT["glass"]], tracker, 35)
    for side, sgn in SIDES:
        hw, hh, zc, ch = B.fuselage_section(10.15)
        p = (sgn * (hw - ch / 2), 10.15, zc + hh - ch / 2)
        hit, nrm = surf(p, (sgn, 0, 1), only=["EXT_HULL_Fuselage"])
        inst(ename("SENSORS", "StarTracker", side), tr, loc=hit, rot=frame(nrm, (0, 1, 0)))

    # ---- lidar head: pedestal + rotating drum
    hit, nrm = surf((0, 10.75, 0.4), (0, 0, 1), only=["EXT_HULL_Fuselage"])
    lid = part(ename("SENSORS", "Lidar"), MAT["metal"], lambda bm: (
        add_box(bm, 0.30, 0.30, 0.14, Tm((0, 0.0, 0.05)), bevel=0.015, segs=2),
        add_box(bm, 0.24, 0.24, 0.04, Tm((0, 0, 0.13)), bevel=0.008)),
        loc=hit, rot=frame(nrm, (0, 1, 0)), smooth=35)
    lid["note"] = "pedestal sits on the nose facet; the drum spins about local Z"
    part(ename("SENSORS", "LidarDrum"), [MAT["dark"], MAT["glass"]], lambda bm: (
        add_cyl(bm, 0.10, 0.05, 24, z0=0.0),
        add_revolve(bm, [(0.098, 0.05), (0.098, 0.10)], 24, mi=1),
        add_cyl(bm, 0.10, 0.04, 24, z0=0.10),
        add_box(bm, 0.05, 0.03, 0.03, Tm((0, 0.0, 0.155)), bevel=0.004)),
        parent=lid, loc=(0, 0, 0.15), smooth=35)

    # ---- dorsal gimballed high-gain dish: mast -> yoke (azimuth) -> dish (elevation) -> feed
    top = surf((0, -6.0, 1.9), (0, 0, 1), only=["EXT_HULL_Fuselage"])[0].z
    mast = part(ename("SENSORS", "DorsalDishMast"), MAT["metal"], lambda bm: (
        add_revolve(bm, [(0.0001, -0.03), (0.22, -0.03), (0.22, 0.0), (0.10, 0.02), (0.10, 0.04),
                         (0.14, 0.04), (0.14, 0.06), (0.0001, 0.06)], 24),),
        loc=(0, -6.0, top), smooth=35)
    yoke = part(ename("SENSORS", "DishYoke"), MAT["metal"], lambda bm: (
        add_cyl(bm, 0.13, 0.03, 24, z0=0.0),
        add_box(bm, 1.30, 0.06, 0.04, Tm((0, 0, 0.05)), bevel=0.008),
        add_box(bm, 0.05, 0.08, 0.12, Tm((0.635, 0, 0.10)), bevel=0.008),
        add_box(bm, 0.05, 0.08, 0.12, Tm((-0.635, 0, 0.10)), bevel=0.008),
        add_cyl(bm, 0.035, 0.08, 24, Tm((0.60, 0, 0.12)) @ R(PI / 2, 'Y')),
        add_cyl(bm, 0.035, 0.08, 24, Tm((-0.60, 0, 0.12)) @ R(PI / 2, 'Y'))),
        parent=mast, loc=(0, 0, 0.06), smooth=35)
    yoke["axis"] = "azimuth, local Z"

    def dish(bm):
        rd, depth = 0.55, 0.09
        zs = [(rd * (i / 8), -0.07 + depth * (i / 8) ** 2) for i in range(9)]
        outer = [(max(r, 0.0001), z) for r, z in zs]
        back = [(max(r, 0.0001), z - 0.02) for r, z in reversed(zs)]
        add_revolve(bm, outer + [(rd + 0.01, outer[-1][1] + 0.005)] + back, 32)
        add_torus(bm, rd, 0.012, 32, 6, Tm((0, 0, zs[-1][1])))
        for k in range(4):                                     # back ribs
            a = k * PI / 2
            add_box(bm, 0.40, 0.02, 0.03, Tm((0.25 * math.cos(a), 0.25 * math.sin(a), -0.07)) @ R(a, 'Z'),
                    bevel=0.004)
        add_cyl(bm, 0.07, 0.08, 24, z0=-0.13)                   # elevation hub
        add_cyl(bm, 0.03, 1.12, 24, Tm((0, 0, -0.09)) @ R(PI / 2, 'Y'))
    dsh = part(ename("SENSORS", "DorsalDish"), MAT["hull"], dish, parent=yoke, loc=(0, 0, 0.21), smooth=35)
    dsh["axis"] = "elevation, local X"

    def feed(bm):
        for k in range(3):
            a = PI / 2 + k * 2 * PI / 3
            add_tube(bm, [(0.50 * math.cos(a), 0.50 * math.sin(a), 0.01),
                          (0.04 * math.cos(a), 0.04 * math.sin(a), 0.085)], 0.01, 6)
        add_revolve(bm, [(0.0001, 0.06), (0.035, 0.06), (0.045, 0.095), (0.08, 0.11), (0.0001, 0.105)], 24)
    part(ename("SENSORS", "DishFeed"), MAT["metal"], feed, parent=dsh, smooth=40)

    # ---- blade antennas (dorsal + ventral, linked)
    def blade(bm):
        add_prism(bm, [(-0.16, 0.0), (0.16, 0.0), (0.06, 0.22), (-0.02, 0.22)], 0.02,
                  Matrix(((0, 0, 1, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))))
        add_box(bm, 0.06, 0.38, 0.02, Tm((0, 0, 0.01)), bevel=0.005)
    bl = mesh(ename("SENSORS", "BladeAntenna"), MAT["dark"], blade, None)
    for i, (p, n) in enumerate((((0, -2.6, 1.9), (0, 0, 1)), ((0, -6.6, -1.9), (0, 0, -1))), 1):
        hit, nrm = surf(p, n, only=["EXT_HULL_Fuselage"])
        inst(ename("SENSORS", f"BladeAntenna{i}"), bl, loc=hit, rot=frame(nrm, (0, -1, 0)))

    # ---- side phased arrays: panel + 8 x 6 instanced raised cells per side
    PW, PL, NX, NY = 0.84, 1.20, 6, 8
    pan = mesh(ename("SENSORS", "PhasedArray"), MAT["dark"], lambda bm: (
        add_box(bm, PW, PL, 0.04, Tm((0, 0, 0.02)), bevel=0.01, segs=2),), 35)
    cell = mesh(ename("SENSORS", "PhasedArrayCell"), MAT["metal"], lambda bm: (
        add_box(bm, 0.11, 0.125, 0.02, Tm((0, 0, 0.01)), bevel=0.004, segs=1),), None)
    for side, sgn in SIDES:
        hit, nrm = surf((sgn * 1.8, 3.6, 0.25), (sgn, 0, 0), only=["EXT_HULL_Fuselage"])
        pa = inst(ename("SENSORS", "PhasedArray", side), pan, loc=hit, rot=frame(nrm, (0, 1, 0)))
        n = 0
        for ix in range(NX):
            for iy in range(NY):
                n += 1
                x = -PW / 2 + 0.07 + ix * (PW - 0.14) / (NX - 1)
                y = -PL / 2 + 0.08 + iy * (PL - 0.16) / (NY - 1)
                inst(ename("SENSORS", f"PhasedArrayCell{n:02d}", side), cell, parent=pa, loc=(x, y, 0.04))


BUILDERS["SENSORS"] = build_sensors


# =================================================================== 9 DOCK + HATCH
def build_dock_hatch():
    so = sock("SOCKET_Dock_Dorsal").translation          # (-0.45, 1.55, 2.15)
    zt = so.z

    def collar(bm):
        # skirt grows out of the hull (bottom buried), flange, collar barrel to the docking plane
        # PATCH (interior task): skirt bottom raised 1.45 -> 1.60 (stays buried in the
        # hull, now clears the R03/R03a ceilings at Z 1.5 + 0.05 shell)
        add_revolve(bm, [(0.50, 1.60 - zt), (0.72, 1.60 - zt), (0.72, 1.98 - zt), (0.66, 2.00 - zt),
                         (0.62, 2.02 - zt), (0.62, -0.03), (0.50, -0.03)], 32, closed=True)
        for k in range(16):                                    # flange bolts
            a = 2 * PI * k / 16
            add_cyl(bm, 0.016, 0.02, 6, Tm((0.69 * math.cos(a), 0.69 * math.sin(a), 1.985 - zt)))
    col = part(ename("DOCK", "Collar"), MAT["metal"], collar, loc=so, smooth=35)
    col["socket"] = "SOCKET_Dock_Dorsal"

    def ring(bm):
        add_revolve(bm, [(0.47, -0.04), (0.64, -0.04), (0.64, 0.0), (0.60, 0.02), (0.47, 0.02)], 32, closed=True)
        for k in range(3):                                     # guide petals
            a = k * 2 * PI / 3
            M = R(a, 'Z') @ Tm((0.555, 0, 0.0)) @ R(PI / 2, 'Z') @ Matrix(
                ((1, 0, 0, 0), (0, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
            add_prism(bm, [(-0.16, 0.0), (0.16, 0.0), (0.05, 0.13), (-0.05, 0.13)], 0.03, M)
    rg = part(ename("DOCK", "CaptureRing"), MAT["metal"], ring, parent=col, smooth=35)

    def notch(bm):   # capture latch in a notch cut through the ring lip
        add_box(bm, 0.09, 0.07, 0.05, Tm((0, 0, 0.0)), bevel=0.006)
        add_box(bm, 0.04, 0.03, 0.05, Tm((0, -0.02, 0.04)), bevel=0.004)
    lt = mesh(ename("DOCK", "Latch"), MAT["dark"], notch, 35)
    for k in range(3):
        a = PI / 3 + k * 2 * PI / 3
        inst(ename("DOCK", f"Latch{k + 1}"), lt, parent=rg, loc=(0.55 * math.cos(a), 0.55 * math.sin(a), 0.0),
             rot=(0, 0, a + PI / 2))
    part(ename("DOCK", "Hatch"), MAT["hull"], lambda bm: (
        add_cyl(bm, 0.47, 0.03, 32, z0=-0.06),
        add_box(bm, 0.30, 0.05, 0.03, Tm((0, 0, -0.02)), bevel=0.008)), parent=col, smooth=35)

    # ---- port crew hatch: disc + frame + ladder rungs
    so = sock("SOCKET_Hatch_Crew").translation           # (-1.84, 1.50, 0.25)
    hit, nrm = surf((-1.8, so.y, so.z), (-1, 0, 0), only=["EXT_HULL_Fuselage"])
    F = frame(nrm, (0, 0, 1))
    frm = part(ename("HATCH", "CrewFrame"), MAT["metal"], lambda bm: (
        add_revolve(bm, [(0.53, -0.03), (0.70, -0.03), (0.70, 0.02), (0.66, 0.05), (0.53, 0.05)], 32,
                    closed=True),
        [add_cyl(bm, 0.014, 0.015, 6, Tm((0.615 * math.cos(2 * PI * k / 20), 0.615 * math.sin(2 * PI * k / 20),
                                             0.055))) for k in range(20)]), loc=hit, rot=F, smooth=35)
    frm["socket"] = "SOCKET_Hatch_Crew"

    def disc(bm):
        add_cyl(bm, 0.52, 0.05, 32, z0=-0.02)
        add_revolve(bm, [(0.30, 0.03), (0.42, 0.03), (0.42, 0.045), (0.30, 0.045)], 32, closed=True)
        add_box(bm, 0.36, 0.05, 0.035, Tm((0, 0, 0.045)), bevel=0.008)       # handle bar
        add_cyl(bm, 0.07, 0.05, 24, Tm((0, 0, 0.05)))                          # latch hub
        add_cyl(bm, 0.035, 0.24, 24, Tm((0.0, 0.56, 0.02)) @ R(PI / 2, 'Y'))     # hinge
    hd = part(ename("HATCH", "Crew"), [MAT["hull"]], disc, parent=frm, smooth=35)
    hd["note"] = "hinged at the top (local +Y); swings outboard"

    def rung(bm):
        pts = fillet([(0, -0.22, -0.01), (0, -0.22, 0.11), (0, 0.22, 0.11), (0, 0.22, -0.01)], 0.04, 3)
        add_tube(bm, pts, 0.017, 8)
        for y in (-0.22, 0.22):
            add_cyl(bm, 0.035, 0.012, 8, Tm((0, y, 0.002)))
    rm = mesh(ename("HATCH", "Rung"), MAT["rail"], rung, 50)
    rungs = [(-1.8, -0.55, (-1, 0, 0)), (-1.8, -0.85, (-1, 0, 0)), (-1.55, -1.30, (-1, 0, -1)),
             (-1.25, -1.60, (-1, 0, -1))]
    for i, (x, z, n) in enumerate(rungs, 1):
        h, nn = surf((x, so.y, z), n, only=["EXT_HULL_Fuselage"])
        inst(ename("HATCH", f"Rung{i}"), rm, loc=h, rot=frame(nn, (0, 1, 0)))


BUILDERS["DOCK"] = build_dock_hatch


# =================================================================== 10 GREEBLES
def bolt_rows():
    """Instanced bolt rows on the circumferential seams of the constant fuselage
    section and the sponsons (one row each side of a seam, on the panel lip)."""
    def row(n, pitch):
        def b(bm):
            for k in range(n):
                x = (k - (n - 1) / 2) * pitch
                add_revolve(bm, [(0.017, 0.0), (0.017, 0.007), (0.012, 0.012)], 6, M=Tm((x, 0, 0)),
                            cap0=True, cap1=True)
        return b
    wide = mesh(ename("GREEBLES", "BoltRow8"), MAT["metal"], row(8, 0.10), None)
    narrow = mesh(ename("GREEBLES", "BoltRow5"), MAT["metal"], row(5, 0.10), None)

    def blocked(p):
        x, y, z = p
        ax = abs(x)
        return any((
            3.0 < y < 9.9 and ax < 1.35 and z > 1.25,               # canopy
            0.7 < y < 2.4 and -1.25 < x < 0.35 and z > 1.45,       # dock
            x < -1.6 and 0.6 < y < 2.4 and -1.0 < z < 1.05,        # crew hatch
            ax > 1.6 and 2.8 < y < 4.4 and -0.45 < z < 0.95,       # phased arrays
            -8.35 < y < 0.15 and ax > 1.35 and z < 0.4,            # under the sponsons
            y < -8.4 and ax > 0.8 and z < 0.35,                    # nacelles
            -6.8 < y < -5.2 and ax < 0.75 and z > 1.5,             # dish
            -9.0 < y < -6.9 and ax < 0.62 and z > 1.5,             # foil tanks
            -2.15 < y < 2.75 and ax < 1.0 and z < -1.5,            # bay doors
            7.0 < y < 8.2 and ax > 1.6 and 0.5 < z < 0.95,         # nose handrails
            (abs(y - 9.0) < 0.45 or abs(y + 9.4) < 0.45) and 1.0 < ax < 1.7 and abs(z) > 0.9,  # RCS
            abs(y + 1.5) < 0.4 and ax > 1.6 and 0.3 < z < 1.0,     # access hatches
            abs(y + 7.4) < 0.4 and ax > 1.6 and 0.3 < z < 1.0,
            abs(y - 1.5) < 0.4 and x > 1.6 and -0.1 < z < 0.6,
            abs(y + 0.6) < 0.3 and ax < 0.3 and z > 1.5,
            abs(y + 3.7) < 0.3 and -0.75 < x < -0.05 and z > 1.5,
            abs(y + 2.6) < 0.3 and ax < 0.3 and z > 1.5,           # blade antenna
            abs(y - 5.6) < 0.3 and 0.2 < x < 0.7 and z < -1.5,
            abs(y + 4.6) < 0.3 and -0.7 < x < -0.2 and z < -1.5,
            abs(y + 6.6) < 0.3 and ax < 0.3 and z < -1.5,
        ))

    count = 0
    ys = fuselage_stations()
    tops = PANEL_RAISE["fuselage"]
    ring = fuselage_ring(0.0)
    nr = len(ring)
    for k in range(9, 38):                 # interior seams of the constant section
        if k % 2:
            continue
        y_seam = ys[k]
        for off, kk in ((0.075, k - 1), (-0.075, k)):
            y = y_seam + off
            for i in range(nr):
                a, b = Vector((ring[i][0], 0, ring[i][1])), Vector((ring[(i + 1) % nr][0], 0, ring[(i + 1) % nr][1]))
                t = (b - a)
                w = t.length
                mid = (a + b) / 2
                n = Vector((t.z, 0, -t.x)).normalized()
                if n.dot(mid) < 0:
                    n = -n
                if (kk, i) not in tops:        # panel removed (cockpit opening)
                    continue
                p = mid + n * tops[(kk, i)] + Vector((0, y, 0))
                if blocked(p):
                    continue
                me = wide if w > 0.85 else narrow
                count += 1
                inst(ename("GREEBLES", f"BoltRow{count:03d}"), me, loc=p, rot=frame(n, (0, 0, 1) if abs(n.z) < 0.5 else (1, 0, 0)))
    STATS["bolt_rows"] = count
    STATS["bolts"] = sum(8 if o.data == wide else 5 for o in EXT.all_objects
                         if o.name.startswith("EXT_GREEBLES_BoltRow") and o.type == 'MESH' and o.data in (wide, narrow))


def build_greebles():
    # ---- spine conduits (+ clamps), starboard pair and a port single run
    def run_pts(x, y0):
        return fillet([(x, y0, 1.80), (x, y0 - 0.15, 1.985), (x, -9.80, 1.985), (x, -10.65, 1.74),
                       (x, -10.85, 1.50)], 0.12, 4)
    for nm, x, r, y0 in (("SpineConduitA", 0.66, 0.040, 3.05), ("SpineConduitB", 0.77, 0.030, 3.05),
                         ("SpineConduitC", -0.70, 0.036, 0.45)):
        part(ename("GREEBLES", nm), MAT["metal"], lambda bm, x=x, r=r, y0=y0: add_tube(bm, run_pts(x, y0), r, 8),
             smooth=50)
    clamp2 = mesh(ename("GREEBLES", "ConduitClamp2"), MAT["dark"], lambda bm: (
        add_box(bm, 0.26, 0.06, 0.05, Tm((0, 0, 0.025)), bevel=0.006),
        add_box(bm, 0.26, 0.05, 0.03, Tm((0, 0, 0.115)), bevel=0.006),
        add_box(bm, 0.03, 0.05, 0.12, Tm((-0.12, 0, 0.07)), bevel=0.004),
        add_box(bm, 0.03, 0.05, 0.12, Tm((0.12, 0, 0.07)), bevel=0.004)), None)
    clamp1 = mesh(ename("GREEBLES", "ConduitClamp1"), MAT["dark"], lambda bm: (
        add_box(bm, 0.14, 0.06, 0.05, Tm((0, 0, 0.025)), bevel=0.006),
        add_box(bm, 0.14, 0.05, 0.03, Tm((0, 0, 0.115)), bevel=0.006)), None)
    n = 0
    top = 1.9 + GROOVE_D + LIP_WHIPPLE[0] - 0.004
    for y in [2.6 - 0.95 * i for i in range(13)]:
        if -6.75 < y < -5.25:
            continue
        n += 1
        inst(ename("GREEBLES", f"ConduitClamp{n:02d}"), clamp2, loc=(0.715, y, top))
        if y < 0.2:
            n += 1
            inst(ename("GREEBLES", f"ConduitClamp{n:02d}"), clamp1, loc=(-0.70, y, top))

    # ---- sponson pipe runs (mirrored -> separate meshes)
    for side, sgn in SIDES:
        pts = [(sgn * 2.29, -0.3, 0.02), (sgn * 2.29, -0.5, 0.19), (sgn * 2.29, -6.9, 0.19),
               (sgn * 2.20, -7.2, 0.10)]
        part(ename("GREEBLES", "SponsonPipe", side), MAT["metal"], lambda bm, pts=pts: (
            add_tube(bm, fillet(pts, 0.1, 3), 0.03, 8),
            add_tube(bm, fillet([(p[0] - sgn * 0.07, p[1], p[2] + 0.01) for p in pts], 0.1, 3), 0.02, 8)),
            smooth=50)

    # ---- louvred vents
    def vent(bm):
        add_box(bm, 0.30, 0.42, 0.07, Tm((0, 0, 0.035)), bevel=0.01, segs=2)
        for k in range(6):
            add_box(bm, 0.24, 0.045, 0.012, Tm((0, -0.15 + k * 0.06, 0.075)) @ R(math.radians(30), 'X'))
    vm = mesh(ename("GREEBLES", "Vent"), MAT["dark"], vent, None)
    vents = [((1.92, -1.2, 0.3), (0, 0, 1)), ((-1.92, -1.2, 0.3), (0, 0, 1)),
             ((1.92, -2.3, 0.3), (0, 0, 1)), ((-1.92, -2.3, 0.3), (0, 0, 1)),
             ((1.8, -4.9, 0.66), (1, 0, 0)), ((-1.8, -4.9, 0.66), (-1, 0, 0)),
             ((0.45, -8.3, -1.9), (0, 0, -1)), ((-0.45, -8.3, -1.9), (0, 0, -1)),
             ((1.8, 5.2, -0.5), (1, 0, 0)), ((-1.8, 5.2, -0.5), (-1, 0, 0)),
             ((0.55, -11.0, 0.75), (0, -1, 0)), ((-0.55, -11.0, 0.75), (0, -1, 0))]
    for i, (p, nn) in enumerate(vents, 1):
        h, nrm = surf(p, nn)
        inst(ename("GREEBLES", f"Vent{i:02d}"), vm, loc=h, rot=frame(nrm, (0, 1, 0)))

    # ---- fill / drain ports (capped)
    def port(bm):
        add_cyl(bm, 0.075, 0.02, 24, z0=0.0)
        add_cyl(bm, 0.05, 0.045, 24, z0=0.02)
        add_cyl(bm, 0.03, 0.02, 6, z0=0.065)
        add_tube(bm, [(0.05, 0, 0.05), (0.09, 0, 0.05), (0.09, 0, 0.02)], 0.006, 6)    # lanyard
    pm = mesh(ename("GREEBLES", "Port"), MAT["metal"], port, 35)
    ports = [((2.4, -1.9, -1.05), (1, 0, 0)), ((-2.4, -1.9, -1.05), (-1, 0, 0)),
             ((2.4, -2.25, -1.05), (1, 0, 0)), ((-2.4, -2.25, -1.05), (-1, 0, 0)),
             ((0.55, -9.2, -1.9), (0, 0, -1)), ((-0.55, -9.2, -1.9), (0, 0, -1)),
             ((0.0, -9.6, -1.9), (0, 0, -1)), ((1.8, -8.0, 0.75), (1, 0, 0)),
             ((-1.8, -8.0, 0.75), (-1, 0, 0)), ((0.0, 6.4, -1.9), (0, 0, -1)),
             ((0.0, -11.0, -0.95), (0, -1, 0)), ((0.25, -11.0, -1.2), (0, -1, 0))]
    for i, (p, nn) in enumerate(ports, 1):
        h, nrm = surf(p, nn)
        inst(ename("GREEBLES", f"Port{i:02d}"), pm, loc=h, rot=frame(nrm, (0, 1, 0)))

    # ---- foil-blanket tanks (puffy boxes) on the exposed aft surfaces
    def puffy(sx, sy, sz):
        def b(bm):
            r = bmesh.ops.create_cube(bm, size=1.0)
            bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=4, use_grid_fill=True)
            for v in bm.verts:
                c = v.co.copy()
                sph = c.normalized() * 0.5 * math.sqrt(3) * 0.62
                v.co = c.lerp(sph, 0.18)
                v.co.x *= sx
                v.co.y *= sy
                v.co.z *= sz
                v.co.z += sz * 0.45
            # quilting: pull the grid lines in a little
            for v in bm.verts:
                if abs(abs(v.co.x) - 0) < 1e-4 or abs(abs(v.co.y) - 0) < 1e-4:
                    v.co *= 0.985
        return b
    tk = mesh(ename("GREEBLES", "FoilTank"), MAT["foil"], puffy(0.42, 1.0, 0.28), 60)
    ts = mesh(ename("GREEBLES", "FoilTankSmall"), MAT["foil"], puffy(0.36, 0.50, 0.22), 60)
    strap = mesh(ename("GREEBLES", "TankStrap"), MAT["metal"], lambda bm: add_tube(
        bm, [(-0.25, 0, 0.0), (-0.24, 0, 0.24), (-0.16, 0, 0.30), (0.16, 0, 0.30), (0.24, 0, 0.24), (0.25, 0, 0.0)],
        0.012, 6), 50)
    tanks = [(tk, (0.33, -7.85, 1.9), (0, 0, 1)), (tk, (-0.33, -7.85, 1.9), (0, 0, 1))]
    for side, sgn in SIDES:
        hw, hh, zc, ch = B.fuselage_section(-10.35)
        tanks.append((ts, (sgn * (hw - ch / 2), -10.35, zc + hh - ch / 2), (sgn, 0, 1)))
        tanks.append((ts, (sgn * 1.55, -10.35, -1.2), (sgn, 0, -1)))
    for i, (me, p, nn) in enumerate(tanks, 1):
        h, nrm = surf(p, nn, only=["EXT_HULL_Fuselage"])
        t = inst(ename("GREEBLES", f"FoilTank{i:02d}"), me, loc=h - nrm * 0.01, rot=frame(nrm, (0, 1, 0)))
        if me is tk:
            for j, yy in enumerate((-0.3, 0.3), 1):
                inst(ename("GREEBLES", f"FoilTank{i:02d}Strap{j}"), strap, parent=t, loc=(0, yy, 0))

    bolt_rows()


BUILDERS["GREEBLES"] = build_greebles


# =================================================================== driver
def tri_count_group(prefix):
    objs = [o for o in EXT.all_objects if o.type == 'MESH' and o.name.startswith(prefix)]
    return C.triangle_count(objs), len(objs)



def run(groups, out):
    for g in GROUP_ORDER:
        if g not in groups or g not in BUILDERS:
            continue
        BUILDERS[g]()
        bpy.context.view_layer.update()
        C.save_blend(out)
        print(f"KESTREL_EXT group {g} done; saved {out}", flush=True)


BUILDERS["HULL"] = build_hull


def fit_check():
    """Interior volumes vs the new skin: fuselage, sponsons, nacelles (detailed
    meshes, i.e. measured to the bottom of the seam grooves) plus the canopy
    volume (closed loft through the pane rings).  Same sampling as 10_blockout."""
    trees = []
    for o in EXT.all_objects:
        if o.get("kestrel_skin"):
            trees.append(tree_for(o))
    bm = bmesh.new()
    rings = canopy_rings()
    vr = [[bm.verts.new((x, y, z)) for (x, z) in r] for (y, r) in rings]
    for a, b in zip(vr, vr[1:]):
        for i in range(len(a)):
            j = (i + 1) % len(a)
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.faces.new(vr[0])
    bm.faces.new(vr[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    trees.append(BVHTree.FromBMesh(bm))
    bm.free()

    def depth(p):
        best = -1e9
        for tr in trees:
            hit = tr.find_nearest(p)
            if hit[0] is None:
                continue
            loc, nrm, _i, dist = hit
            best = max(best, dist if (p - loc).dot(nrm) < 0 else -dist)
        return best
    out = {}
    for o in bpy.data.collections[C.COL_INT].all_objects:
        if o.type != 'MESH':
            continue
        lo, hi = C.world_bbox([o])
        pts = [Vector((lo.x + (hi.x - lo.x) * fx, lo.y + (hi.y - lo.y) * fy, lo.z + (hi.z - lo.z) * fz))
               for fx in (0, .5, 1) for fy in (0, .5, 1) for fz in (0, .5, 1) if (fx, fy, fz) != (.5, .5, .5)]
        out[o.name] = round(min(depth(p) for p in pts), 3)
    return out


def report():
    bpy.context.view_layer.update()
    rep = {"groups": {}, "stats": STATS}
    total = 0
    for g in GROUP_ORDER + ["HATCH"]:
        t, n = tri_count_group(f"EXT_{g}_")
        rep["groups"][g] = {"tris": t, "meshes": n}
        total += t
    ext_objs = list(EXT.all_objects)
    rep["exterior_tris"] = C.triangle_count([o for o in ext_objs if o.type == 'MESH'])
    rep["sum_groups"] = total
    mx = max(((C.triangle_count([o]), o.name) for o in ext_objs if o.type == 'MESH'), default=(0, ""))
    rep["max_single_mesh"] = {"name": mx[1], "tris": mx[0]}
    rep["objects"] = {"exterior": len(ext_objs), "total": len(bpy.context.scene.objects),
                      "unique_ext_meshes": len({o.data.name for o in ext_objs if o.type == 'MESH'})}
    rep["triangles_total_scene"] = C.triangle_count()
    hull = [o for o in ext_objs if o.type == 'MESH' and not o.name.startswith(("EXT_RADIATORS", "EXT_GEAR"))]
    lo, hi = C.world_bbox(hull)
    size = hi - lo
    rep["hull_bbox_no_radiators_no_gear"] = {"min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi],
                                             "size": [round(v, 3) for v in size]}
    spec = (C.SPEC["beam"], C.SPEC["length"], C.SPEC["height"])
    rep["envelope_dev_pct"] = [round(100 * (s - r) / r, 2) for s, r in zip(size, spec)]
    lo2, hi2 = C.world_bbox([o for o in ext_objs if o.type == 'MESH'])
    fit = fit_check()
    rep["interior_min_clearance"] = fit
    rep["interior_all_fit_0.05"] = all(v >= B.WALL_CLEARANCE - 1e-6 for v in fit.values())
    rep["bbox_all_exterior"] = {"min": [round(v, 3) for v in lo2], "max": [round(v, 3) for v in hi2]}
    return rep


if __name__ == "__main__":
    args = C.script_args()
    src = args.get("blend", C.BLEND_PATH)
    out = args.get("out", src)
    groups = args.get("groups")
    groups = set(groups.split(",")) if isinstance(groups, str) else set(GROUP_ORDER)
    C.open_blend(src)
    setup()
    wipe_exterior()
    setup()
    run(groups, out)
    rep = report()
    if args.get("report"):
        with open(args["report"], "w") as f:
            json.dump(rep, f, indent=1)
    print("KESTREL_EXT_REPORT " + json.dumps(rep))
