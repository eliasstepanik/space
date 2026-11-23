"""IC-4 "Kestrel" — 10 blockout & layout lock.

Builds the dimensioned blockout from an EMPTY scene and saves it.

    blender -b --factory-startup --python 10_blockout.py -- [--out /path/file.blend] [--report report.json]

Defaults: --out ../kestrel.blend.  Re-runnable: the script wipes the file first.
Frame: Blender Z-up, nose +Y, starboard +X, metres.  Origin = hull centre line,
mid-length, Z=0 at fuselage mid-height.  Ground (on gear) at Z = -3.3.
"""
import importlib
import json
import math
import os
import sys

sys.dont_write_bytecode = True  # keep scripts/ free of __pycache__

import bpy
from mathutils import Vector, Euler
from mathutils.bvhtree import BVHTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")

# ============================================================ HULL GEOMETRY
# Fuselage loft: (y, half_width, half_height, z_centre). Chamfer = hw/2 so the
# full section is an octagon 3.6 m wide x 3.8 m tall with 0.9 m chamfers.
FUSELAGE = [
    (11.5, 0.40, 0.40, -0.45),   # nose tip (flat sensor face)
    (9.6, 1.35, 1.40, -0.20),
    (7.6, 1.80, 1.90, 0.00),     # full section starts
    (-9.8, 1.80, 1.90, 0.00),    # full section ends
    (-11.0, 1.45, 1.55, 0.00),   # aft taper (engine bay)
]


def chamfer_of(hw, hh):
    return 0.5 * min(hw, hh)


def fuselage_section(y):
    """Interpolated (hw, hh, zc, chamfer) of the fuselage at station y."""
    secs = FUSELAGE
    if y >= secs[0][0]:
        s = secs[0]
    elif y <= secs[-1][0]:
        s = secs[-1]
    else:
        for a, b in zip(secs, secs[1:]):
            if a[0] >= y >= b[0]:
                t = (a[0] - y) / (a[0] - b[0])
                s = tuple(a[i] + (b[i] - a[i]) * t for i in range(4))
                break
    return s[1], s[2], s[3], chamfer_of(s[1], s[2])


# Canopy: faceted cab on the forward dorsal hull. Ring (x, z) at full size.
CANOPY_RING = [(1.25, 1.20), (1.00, 2.00), (0.70, 2.30),
               (-0.70, 2.30), (-1.00, 2.00), (-1.25, 1.20)]
CANOPY_STATIONS = [  # (y, width scale, top z)
    (9.7, 0.50, 1.25),
    (8.0, 1.00, 2.30),
    (4.4, 1.00, 2.30),
    (3.2, 0.90, 1.70),
]

# Sponsons (starboard; port is mirrored): chamfered box X 1.45..2.4, Z -1.7..0.3
SPONSON = [  # (y, cx, hw, hh, cz)
    (0.0, 1.70, 0.25, 0.60, -0.70),
    (-1.0, 1.925, 0.475, 1.00, -0.70),
    (-7.0, 1.925, 0.475, 1.00, -0.70),
    (-8.2, 1.75, 0.30, 0.65, -0.70),
]
SPONSON_CHAMFER = 0.30

# ============================================================ INTERIOR LAYOUT
# (id, name, xmin, xmax, ymin, ymax, zmin, zmax, purpose) — fore to aft.
INTERIOR = [
    ("B01", "NoseSensorBay",        -0.50, 0.50,  9.00, 10.60, -0.85, 0.35,
     "radar / optical sensor stack behind the flat nose face"),
    ("B02", "NoseGearWell",         -0.45, 0.45,  6.30,  8.20, -1.70, -0.45,
     "nose gear retracts aft-up under the cockpit"),
    ("R01", "Cockpit",              -0.75, 0.75,  4.40,  7.90,  0.55, 2.15,
     "tandem: pilot seat Y 6.75, engineer seat Y 4.95, raised under canopy"),
    ("R02", "AvionicsLifeSupport",  -1.10, 1.10,  2.60,  4.30, -0.50, 1.50,
     "avionics racks + ECLSS; ladder up into the cockpit"),
    ("R03", "AirlockVestibule",     -0.05, 1.10,  0.60,  2.50, -0.50, 1.50,
     "suit-up vestibule, starboard of the airlock chamber"),
    ("R03a", "AirlockChamber",      -1.10, -0.15, 0.60,  2.50, -0.50, 1.50,
     "sealed chamber, port hull hatch + ceiling hatch to dorsal dock"),
    ("R04", "CrewQuarters",         -1.10, 1.10, -1.60,  0.50, -0.50, 1.50,
     "2 bunks, galley, hygiene"),
    ("R05", "StoresLocker",         -1.10, 1.10, -2.50, -1.70, -0.50, 1.50,
     "consumables / spares lockers"),
    ("R06", "EngineeringBay",       -1.10, 1.10, -4.70, -2.60, -0.50, 1.50,
     "engineer station, reactor + propulsion control; crawlway entry aft-top"),
    ("C01", "DorsalCrawlway",       -0.45, 0.45, -9.60, -4.80,  0.65, 1.55,
     "0.9 x 0.9 m crawlway along the dorsal spine, R06 -> B07"),
    ("B03", "OrdnanceBay",          -0.85, 0.85, -2.00,  2.60, -1.75, -0.60,
     "ventral weapons bay under R03/R04, 2 internal hardpoints"),
    ("B04", "MainGearWell_L",       -2.25, -1.60, -4.40, -2.20, -1.45, -0.25,
     "port main gear well in sponson"),
    ("B04", "MainGearWell_R",        1.60, 2.25, -4.40, -2.20, -1.45, -0.25,
     "starboard main gear well in sponson"),
    ("B05", "PropellantTankBay",    -1.00, 1.00, -7.10, -4.80, -1.60, 0.50,
     "reaction-mass tankage under the crawlway"),
    ("B06", "ReactorCompartment",   -1.00, 1.00, -8.90, -7.20, -1.60, 0.50,
     "fission reactor + shadow shield, under the crawlway"),
    ("B07", "EngineBay",            -1.00, 1.00, -10.70, -9.00, -1.30, 0.50,
     "turbopumps / engine feeds, feeds both nacelles"),
]
ROOM_IDS = ["B01", "B02", "R01", "R02", "R03", "R04", "R05", "R06",
            "C01", "B03", "B04", "B05", "B06", "B07"]
WALL_CLEARANCE = 0.05   # min metres between an interior volume and the hull skin

GEAR = {  # name: (x, y)
    "Nose": (0.0, 7.30),
    "Main_L": (-1.925, -3.30),
    "Main_R": (1.925, -3.30),
}


def build():
    scn = C.reset_scene()
    cols = C.setup_collections()
    ext, inn, sock = cols["ext"], cols["int"], cols["sock"]

    m_hull = C.placeholder_material("MAT_Placeholder_Hull", (0.55, 0.56, 0.58), roughness=0.5)
    m_dark = C.placeholder_material("MAT_Placeholder_Dark", (0.12, 0.12, 0.13), roughness=0.5)
    m_glass = C.placeholder_material("MAT_Placeholder_Canopy", (0.05, 0.09, 0.12), roughness=0.1)
    m_rad = C.placeholder_material("MAT_Placeholder_Radiator", (0.75, 0.72, 0.66), roughness=0.4)
    m_room = C.placeholder_material("MAT_Placeholder_Room", (0.20, 0.45, 0.85))
    m_bay = C.placeholder_material("MAT_Placeholder_Bay", (0.90, 0.50, 0.15))
    m_crawl = C.placeholder_material("MAT_Placeholder_Crawl", (0.25, 0.75, 0.35))
    m_lock = C.placeholder_material("MAT_Placeholder_Airlock", (0.85, 0.20, 0.25))

    root = C.empty_object(C.ROOT, (0, 0, 0), cols["root"], display='ARROWS', size=2.0)
    root["kestrel_spec"] = json.dumps(C.SPEC)

    def own(obj):
        obj.parent = root
        return obj

    # ---------------------------------------------------------------- HULL
    secs = [(y, C.octagon_ring(hw, hh, chamfer_of(hw, hh), 0.0, zc))
            for (y, hw, hh, zc) in FUSELAGE]
    own(C.loft_object(C.ext_name("HULL", "Fuselage"), secs, ext, m_hull))

    for side, sgn in (("L", -1), ("R", 1)):
        rings = []
        for (y, cx, hw, hh, cz) in SPONSON:
            ch = min(SPONSON_CHAMFER, 0.45 * hw, 0.45 * hh)
            rings.append((y, C.octagon_ring(hw, hh, ch, sgn * cx, cz)))
        own(C.loft_object(C.ext_name("HULL", "Sponson", side), rings, ext, m_hull))

    # ---------------------------------------------------------------- CANOPY
    rings = []
    for (y, ws, top) in CANOPY_STATIONS:
        k = (top - 1.20) / (2.30 - 1.20)
        rings.append((y, [(x * ws, 1.20 + (z - 1.20) * k) for (x, z) in CANOPY_RING]))
    own(C.loft_object(C.ext_name("CANOPY", "Glazing"), rings, ext, m_glass))

    # ---------------------------------------------------------------- ENGINES
    for side, sgn in (("L", -1), ("R", 1)):
        rings = []
        for (y, a) in ((-8.6, 0.60), (-10.8, 0.70), (-11.5, 0.80)):
            rings.append((y, C.octagon_ring(a, a, a * 0.586, sgn * 1.20, -0.50)))
        own(C.loft_object(C.ext_name("ENGINES", "Nacelle", side), rings, ext, m_dark))

    # ---------------------------------------------------------------- RADIATORS
    # 3 spanwise panels each side, hinged at the sponson outer edge (X 2.4),
    # deployed flat: tip at X 6.25 -> span 12.5 m.  Chord Y -7.0 .. -3.0.
    edges = [2.40, 3.69, 4.98, 6.25]
    gap = 0.03
    for side, sgn in (("L", -1), ("R", 1)):
        for i in range(3):
            x0, x1 = edges[i] + (gap if i else 0.0), edges[i + 1] - (gap if i < 2 else 0.0)
            xa, xb = sorted((sgn * x0, sgn * x1))
            own(C.box_object(C.ext_name("RADIATORS", f"Panel{i + 1}", side),
                             xa, xb, -7.0, -3.0, -0.03, 0.03, ext, m_rad))

    # ---------------------------------------------------------------- GEAR (stubs, deployed)
    for name, (x, y) in GEAR.items():
        top = -1.75 if name == "Nose" else -1.60
        side = name.split("_")[1] if "_" in name else None
        part = "NoseLeg" if name == "Nose" else "MainLeg"
        own(C.box_object(C.ext_name("GEAR", part, side),
                         x - 0.15, x + 0.15, y - 0.25, y + 0.25, C.Z_GROUND, top, ext, m_dark))

    # ---------------------------------------------------------------- WEAPONS / SENSORS / DOCK / HATCH
    t = C.cylinder_object(C.ext_name("WEAPONS", "ChinTurret"), 0.45, 0.35, 8, ext, m_dark)
    t.location = (0.0, 8.7, -1.75)    # Z -1.925 .. -1.575
    own(t)

    mast = C.cylinder_object(C.ext_name("SENSORS", "DorsalDishMast"), 0.10, 0.20, 8, ext, m_dark)
    mast.location = (0.0, -6.0, 1.98)
    own(mast)
    dish = C.cylinder_object(C.ext_name("SENSORS", "DorsalDish"), 0.60, 0.20, 16, ext, m_hull)
    dish.location = (0.0, -6.0, 2.15)
    own(dish)

    dock = C.cylinder_object(C.ext_name("DOCK", "Collar"), 0.60, 0.45, 8, ext, m_dark)
    dock.location = (-0.45, 1.55, 1.925)    # Z 1.70 .. 2.15
    own(dock)

    own(C.box_object(C.ext_name("HATCH", "Crew"), -1.84, -1.78, 0.95, 2.05, -0.45, 0.95, ext, m_dark))

    # ---------------------------------------------------------------- INTERIOR
    for (rid, name, x0, x1, y0, y1, z0, z1, purpose) in INTERIOR:
        mat = (m_crawl if rid.startswith("C") else m_lock if rid.startswith("R03")
               else m_room if rid.startswith("R") else m_bay)
        o = own(C.box_object(C.int_name(rid, name), x0, x1, y0, y1, z0, z1, inn, mat))
        o["room_id"] = rid
        o["room_name"] = name
        o["purpose"] = purpose
        o["dims_m"] = [round(x1 - x0, 3), round(y1 - y0, 3), round(z1 - z0, 3)]

    # ---------------------------------------------------------------- SOCKETS
    def socket(kind, name, loc, rot=(0, 0, 0), display='ARROWS', **props):
        o = C.empty_object(C.socket_name(kind, name), loc, sock, display=display,
                           size=0.3, rotation=rot, props=dict(socket_kind=kind, **props))
        return own(o)

    look_fwd = (math.pi / 2, 0.0, 0.0)      # camera-style: -Z looks along +Y
    socket("Camera", "PilotEye", (0.0, 6.85, 1.85), look_fwd)
    socket("Camera", "EngEye", (0.0, 5.05, 1.85), look_fwd)
    socket("Seat", "Pilot", (0.0, 6.75, 0.55))
    socket("Seat", "Engineer", (0.0, 4.95, 0.55))
    # Engines: local -Z = exhaust direction (-Y)
    for side, sgn in (("L", -1), ("R", 1)):
        socket("Engine", f"Main_{side}", (sgn * 1.20, -11.5, -0.50), (-math.pi / 2, 0, 0),
               thrust_axis="+Y")

    # RCS quads on the mid-points of the four chamfer faces, fore and aft.
    for station, y in (("Fwd", 9.0), ("Aft", -9.4)):
        hw, hh, zc, ch = fuselage_section(y)
        for vert, vs in (("Up", 1), ("Dn", -1)):
            for side, sgn in (("L", -1), ("R", 1)):
                x = sgn * (hw - ch / 2)
                z = zc + vs * (hh - ch / 2)
                n = Vector((sgn, 0, vs)).normalized()
                rot = n.to_track_quat('Z', 'Y').to_euler()
                socket("RCS", f"Quad_{station}{vert}_{side}", (x, y, z), tuple(rot))

    down = (math.pi, 0.0, 0.0)  # local +Z points -Z world
    socket("Hardpoint", "1_BayL", (-0.40, 0.30, -0.65), down, internal=True)
    socket("Hardpoint", "2_BayR", (0.40, 0.30, -0.65), down, internal=True)
    socket("Hardpoint", "3_SponsonL", (-1.925, -5.80, -1.70), down, internal=False)
    socket("Hardpoint", "4_SponsonR", (1.925, -5.80, -1.70), down, internal=False)
    socket("Turret", "Chin", (0.0, 8.70, -1.925), down)
    socket("Dock", "Dorsal", (-0.45, 1.55, 2.15))
    socket("Hatch", "Crew", (-1.84, 1.50, 0.25), (0, -math.pi / 2, 0))
    for name, (x, y) in GEAR.items():
        socket("Gear", f"Contact_{name}", (x, y, C.Z_GROUND), display='SPHERE')
    socket("Nav", "Light_L", (-6.25, -5.0, 0.0), display='SPHERE', color="red")
    socket("Nav", "Light_R", (6.25, -5.0, 0.0), display='SPHERE', color="green")
    socket("Nav", "Light_Tail", (0.0, -11.0, 1.55), display='SPHERE', color="white")
    # One ceiling light per room ID (14).  B04 light sits in the port well,
    # R03 light in the vestibule (R03a is lit from it through the inner door).
    placed = set()
    for (rid, name, x0, x1, y0, y1, z0, z1, _p) in INTERIOR:
        if rid in placed or rid == "R03a":
            continue
        placed.add(rid)
        socket("Light", rid, ((x0 + x1) / 2, (y0 + y1) / 2, z1 - 0.05), display='SINGLE_ARROW',
               room=f"{rid}_{name}")

    # ---------------------------------------------------------------- PREVIEW RIG (not under Kestrel)
    rig = C.ensure_collection(C.COL_RIG)

    def camera(name, loc, target, ortho=None, lens=40):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.clip_end = 500
        if ortho:
            cd.type = 'ORTHO'
            cd.ortho_scale = ortho
        o = bpy.data.objects.new(name, cd)
        o.location = loc
        if target is not None:
            d = Vector(target) - Vector(loc)
            o.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        rig.objects.link(o)
        return o

    c34 = camera("CAM_Preview_34", (18.0, 19.0, 13.0), (0.0, -1.2, -0.8), lens=40)
    ctop = camera("CAM_Preview_Top", (0.0, 0.0, 40.0), None, ortho=25.0)
    ctop.rotation_euler = (0.0, 0.0, math.pi / 2)    # nose to image right
    scn.camera = c34

    sun_d = bpy.data.lights.new("LGT_Preview_Sun", 'SUN')
    sun_d.energy = 4.0
    sun_d.angle = math.radians(2.0)
    sun = bpy.data.objects.new("LGT_Preview_Sun", sun_d)
    sun.rotation_euler = Euler((math.radians(40), math.radians(15), math.radians(35)))
    rig.objects.link(sun)

    world = bpy.data.worlds.new("WLD_Preview")
    if not world.node_tree:          # Blender 5.x worlds already have nodes
        world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.32, 0.36, 0.42, 1.0)
    bg.inputs["Strength"].default_value = 0.8
    scn.world = world

    r = scn.render
    r.engine = 'CYCLES'
    scn.cycles.samples = 64
    scn.cycles.use_denoising = True
    r.resolution_x, r.resolution_y, r.resolution_percentage = 1280, 720, 100
    r.image_settings.file_format = 'JPEG'
    r.image_settings.quality = 92
    scn.view_settings.view_transform = 'AgX'

    return root


# ============================================================ CHECKS
def _bvh(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(obj, dg)   # local space; all hull objects have identity rotation/scale


def fit_report():
    """For each interior volume: min clearance of its sample points to the hull skin
    (fuselage, sponsons, canopy, nacelles); negative = pokes out."""
    hulls = [o for o in bpy.data.objects if o.type == 'MESH' and (
        o.name.startswith("EXT_HULL_") or o.name.startswith("EXT_CANOPY_")
        or o.name.startswith("EXT_ENGINES_"))]
    trees = [(o, _bvh(o)) for o in hulls]

    def signed_depth(p):
        best = -1e9
        for o, tree in trees:
            lp = o.matrix_world.inverted() @ p
            hit = tree.find_nearest(lp)
            if hit[0] is None:
                continue
            loc, nrm, _i, dist = hit
            inside = (lp - loc).dot(nrm) < 0
            best = max(best, dist if inside else -dist)
        return best

    out = []
    ints = sorted([o for o in bpy.data.objects if o.name.startswith("INT_")],
                  key=lambda o: -o.location.y)
    for o in ints:
        lo, hi = C.world_bbox([o])
        pts = []
        for fx in (0, 0.5, 1):
            for fy in (0, 0.5, 1):
                for fz in (0, 0.5, 1):
                    if (fx, fy, fz) == (0.5, 0.5, 0.5):
                        continue
                    pts.append(Vector((lo.x + (hi.x - lo.x) * fx, lo.y + (hi.y - lo.y) * fy,
                                       lo.z + (hi.z - lo.z) * fz)))
        clear = min(signed_depth(p) for p in pts)
        out.append({
            "name": o.name,
            "dims_xyz_m": [round(hi.x - lo.x, 3), round(hi.y - lo.y, 3), round(hi.z - lo.z, 3)],
            "x": [round(lo.x, 3), round(hi.x, 3)],
            "y": [round(lo.y, 3), round(hi.y, 3)],
            "z": [round(lo.z, 3), round(hi.z, 3)],
            "volume_m3": round((hi.x - lo.x) * (hi.y - lo.y) * (hi.z - lo.z), 2),
            "min_clearance_m": round(clear, 3),
            "fits": clear >= WALL_CLEARANCE - 1e-6,
        })
    # pairwise overlaps
    overlaps = []
    boxes = [(o.name, *C.world_bbox([o])) for o in ints]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            n1, a0, a1 = boxes[i]
            n2, b0, b1 = boxes[j]
            d = [min(a1[k], b1[k]) - max(a0[k], b0[k]) for k in range(3)]
            if all(v > 1e-4 for v in d):
                overlaps.append([n1, n2, [round(v, 3) for v in d]])
    return out, overlaps


def summary():
    scn = bpy.context.scene
    bpy.context.view_layer.update()     # refresh matrix_world before measuring
    ext = [o for o in scn.objects if o.name.startswith("EXT_")]
    hull_main = [o for o in ext if not o.name.startswith(("EXT_RADIATORS", "EXT_GEAR"))]
    lo, hi = C.world_bbox(hull_main)
    lo_all, hi_all = C.world_bbox(ext)
    fit, overlaps = fit_report()
    names = sorted(o.name for o in scn.objects)
    return {
        "triangles_total": C.triangle_count(),
        "triangles_exterior": C.triangle_count(ext),
        "triangles_interior": C.triangle_count([o for o in scn.objects if o.name.startswith("INT_")]),
        "hull_bbox_no_radiators_no_gear": {"min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi],
                                           "size": [round(v, 3) for v in (hi - lo)]},
        "bbox_all_exterior": {"min": [round(v, 3) for v in lo_all], "max": [round(v, 3) for v in hi_all],
                              "size": [round(v, 3) for v in (hi_all - lo_all)]},
        "interior": fit,
        "interior_overlaps": overlaps,
        "counts": {
            "EXT": sum(n.startswith("EXT_") for n in names),
            "INT": sum(n.startswith("INT_") for n in names),
            "SOCKET": sum(n.startswith("SOCKET_") for n in names),
            "SOCKET_Light": sum(n.startswith("SOCKET_Light_") for n in names),
            "SOCKET_RCS": sum(n.startswith("SOCKET_RCS_") for n in names),
            "objects": len(names),
        },
        "objects": names,
    }


if __name__ == "__main__":
    args = C.script_args()
    out = args.get("out", C.BLEND_PATH)
    build()
    rep = summary()
    C.save_blend(out)
    rep["saved"] = out
    if args.get("report"):
        with open(args["report"], "w") as f:
            json.dump(rep, f, indent=1)
    print("KESTREL_REPORT " + json.dumps({k: v for k, v in rep.items() if k != "objects"}))
