"""IC-4 "Kestrel" — 31 technical bay fit-out (B01-B07).

Runs ON TOP of 30_interior.py output: opens the .blend, replaces the blockout box of
every technical bay with a fitted bay and saves after each bay.  Rooms R01-R06, R03a,
C01, Kestrel_Exterior and Kestrel_Sockets are left untouched (sockets / exterior parts
are only READ for placement; the B03 missiles are linked duplicates of the exterior
missile meshes so the two always match).

    blender -b --factory-startup --python 31_bays.py -- [--blend in.blend] [--out out.blend]
                                                       [--bays B01,B02,...] [--report report.json]

Defaults: --blend/--out ../kestrel.blend.  Re-runnable: each bay step first deletes
everything named INT_<ID>_* of that bay (B04 = both wells).

Hierarchy:  Kestrel > INT_<ID>_<Name> (empty, origin at the bay floor centre, props
room_id / clear_volume / fitted) > INT_<ID>_<Part>.  Meshes are built in world
coordinates like the rooms of 30_interior.py; repeats are linked duplicates of
INTM_* library meshes (B04_L is a mirrored linked duplicate of B04_R, scale X -1).
Parts that a cutaway preview should hide carry the custom prop "cutaway".

Clearance: the report measures every bay vertex against the OUTER hull envelope (the
union of fuselage, sponsons, nacelles and canopy; skin faces buried inside another
closed hull volume - e.g. the fuselage side wall running through a sponson or the
nacelle walls inside the aft fuselage - are not envelope and are ignored) and against
the shell boxes of the fitted rooms (clear box + 0.05 m shell).
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
I = importlib.import_module("30_interior")
B, E = I.B, I.E

PI = math.pi
Tm, R = E.Tm, E.R
add_box, add_cyl, add_revolve, add_torus, add_tube, fillet = (
    E.add_box, E.add_cyl, E.add_revolve, E.add_torus, E.add_tube, E.fillet)
bx, cylp, strap, part, lib, inst, place = I.bx, I.cylp, I.strap, I.part, I.lib, I.inst, I.place

BAY_ORDER = ["B01", "B02", "B03", "B04", "B05", "B06", "B07"]
BAYS = {}                                    # (id, name) -> box
for _r in B.INTERIOR:
    if _r[0].startswith("B"):
        BAYS[(_r[0], _r[1])] = _r
ROOM_BOXES = {r[0]: r[2:8] for r in B.INTERIOR if not r[0].startswith("B")}
BUILDERS = {}
MIRROR_X = Matrix.Scale(-1.0, 4, (1.0, 0.0, 0.0))
AY = R(-PI / 2, 'X')                         # local +Z -> world +Y
AX = R(PI / 2, 'Y')                          # local +Z -> world +X


# =================================================================== nodes / small helpers
def bay_node(rid, name):
    _id, _n, x0, x1, y0, y1, z0, z1, purpose = BAYS[(rid, name)]
    o = bpy.data.objects.new(C.int_name(rid, name), None)
    I.INT.objects.link(o)
    o.parent = I.ROOT
    o.location = ((x0 + x1) / 2, (y0 + y1) / 2, z0)
    o.empty_display_type = 'CUBE'
    o.empty_display_size = 0.25
    o["room_id"] = rid
    o["room_name"] = name
    o["purpose"] = purpose
    o["dims_m"] = [round(x1 - x0, 3), round(y1 - y0, 3), round(z1 - z0, 3)]
    o["clear_volume"] = [[x0, x1], [y0, y1], [z0, z1]]
    o["fitted"] = True
    return o


def N(rid, name):
    return C.int_name(rid, name)


def pipe(bm, pts, r, rad=0.08, segs=10, mi=0, n=3):
    segs = min(segs, 10)                                   # bay plumbing budget: <= 10 sides per pipe
    add_tube(bm, fillet(pts, rad, n) if len(pts) > 2 else [Vector(p) for p in pts], r, segs, mi=mi)


def flange(bm, p, d, r, t=0.02, mi=1, bolts=6):
    """Pipe flange (disc) at p, normal d, with bolt heads."""
    M = Tm(p) @ I.align_z(d)
    add_cyl(bm, r, t, 16, M, mi=mi)
    for k in range(bolts):
        a = 2 * PI * k / bolts
        add_cyl(bm, 0.006, t + 0.012, 6, M @ Tm(((r - 0.012) * math.cos(a), (r - 0.012) * math.sin(a), 0)), mi=mi)


def valve(bm, p, d, r, up=(0, 0, 1), mi_body=1, mi_wheel=4):
    """In-line gate valve on a pipe along d at p: body, bonnet along `up`, handwheel."""
    d, up = Vector(d).normalized(), Vector(up).normalized()
    add_cyl(bm, r * 1.6, r * 3.0, 12, Tm(p) @ I.align_z(d), mi=mi_body)
    q = Vector(p) + up * r * 3.2
    cylp(bm, p, q, r * 0.7, 8, mi_body)
    add_torus(bm, r * 1.8, r * 0.28, 16, 4, Tm(q) @ I.align_z(up), mi=mi_wheel)
    for k in range(3):
        a = 2 * PI * k / 3
        w = I.align_z(up).to_3x3() @ Vector((math.cos(a), math.sin(a), 0)) * r * 1.8
        cylp(bm, q, q + w, r * 0.18, 4, mi_wheel)


def frame_rect(bm, y, x0, x1, z0, z1, d=0.06, t=0.012, f=0.045, sides="LRTB", mi=0, mi_f=0):
    """Ring frame (channel: web + inner flange) at station y hugging the box x0..x1, z0..z1."""
    if "T" in sides:
        bx(bm, x0, x1, y - t / 2, y + t / 2, z1 - d, z1, mi)
        bx(bm, x0 + d, x1 - d, y - f / 2, y + f / 2, z1 - d, z1 - d + 0.008, mi_f)
    if "B" in sides:
        bx(bm, x0, x1, y - t / 2, y + t / 2, z0, z0 + d, mi)
        bx(bm, x0 + d, x1 - d, y - f / 2, y + f / 2, z0 + d - 0.008, z0 + d, mi_f)
    za = z0 + (d if "B" in sides else 0.0)
    zb = z1 - (d if "T" in sides else 0.0)
    if "L" in sides:
        bx(bm, x0, x0 + d, y - t / 2, y + t / 2, z0, z1, mi)
        bx(bm, x0 + d - 0.008, x0 + d, y - f / 2, y + f / 2, za, zb, mi_f)
    if "R" in sides:
        bx(bm, x1 - d, x1, y - t / 2, y + t / 2, z0, z1, mi)
        bx(bm, x1 - d, x1 - d + 0.008, y - f / 2, y + f / 2, za, zb, mi_f)


def stringer_y(bm, x, z, y0, y1, w=0.03, h=0.03, mi=0):
    bx(bm, x - w / 2, x + w / 2, y0, y1, z - h / 2, z + h / 2, mi, 0.003)


def rad_plate(bm, M, r=0.07, w=0.20, h=0.26):
    """Radiation hazard marking plate in the local XY plane facing +Z: yellow plate, black
    trefoil, black/yellow hazard stripe along the bottom.  mi: 0 yellow, 1 black."""
    bx(bm, -w / 2, w / 2, -h / 2, h / 2, -0.004, 0.0, 0, 0.002, M=M)
    I.trefoil(bm, M @ Tm((0, 0.025, 0)), r, 0, 1)
    for k in range(5):
        x = -w / 2 + 0.02 + k * (w - 0.04) / 5
        bx(bm, x, x + (w - 0.04) / 10, -h / 2 + 0.015, -h / 2 + 0.05, 0.0, 0.002, 1, M=M)


def rad_plate_mesh():
    return lib("BayRadPlate", ["rail", "dark"], lambda bm: rad_plate(bm, Matrix.Identity(4)), None)


def light_at(rid, node, dx=0.0, name="WorkLight"):
    """Caged work light (30_interior library mesh) at SOCKET_Light_<rid>, optionally shifted in X
    (bays where the socket sits right under a crawlway floor hatch)."""
    p = I.sock(f"SOCKET_Light_{rid}")
    z1 = p.z + 0.05
    o = inst(N(rid, name), I.work_light(), node, (p.x + dx, p.y, z1))
    o["socket"] = f"SOCKET_Light_{rid}"
    return o


def wipe_bay(rid):
    I.wipe_room(rid)


# =================================================================== B01 nose sensor bay
def build_B01():
    rid = "B01"
    node = bay_node(rid, "NoseSensorBay")
    x0, x1, y0, y1, z0, z1 = -0.5, 0.5, 9.0, 10.6, -0.85, 0.35
    DC = Vector((0.0, 10.82, -0.37))                     # dish vertex (looks +Y through the radar fairing)

    def structure(bm):
        for y in (9.05, 10.0, 10.55):
            frame_rect(bm, y, x0 + 0.02, x1 - 0.02, z0 + 0.02, z1 - 0.02, 0.055, 0.012, 0.04, mi=0, mi_f=1)
        # nose ring bulkhead that carries the radar fairing (just aft of the fairing ring)
        zc = -0.43
        add_revolve(bm, [(0.30, -0.015), (0.37, -0.015), (0.37, 0.015), (0.30, 0.015)], 32,
                    Tm((0, 11.28, zc)) @ AY, closed=True, mi=0)
        for k in range(8):
            a = 2 * PI * k / 8 + PI / 8
            add_cyl(bm, 0.008, 0.04, 6, Tm((0.335 * math.cos(a), 11.30, zc + 0.335 * math.sin(a))) @ AY, mi=1)
        # longerons: bay frame corners -> nose ring
        for sx in (-1, 1):
            for (za, zb) in ((z0 + 0.06, zc - 0.25), (z1 - 0.06, zc + 0.25)):
                cylp(bm, (sx * 0.40, 10.55, za), (sx * 0.24, 11.27, zb), 0.018, 8, 0)
        # equipment decks: floor deck + upper shelf, with edge lips
        bx(bm, x0 + 0.02, x1 - 0.02, 9.06, 10.54, z0, z0 + 0.02, 0, 0.004)
        bx(bm, x0 + 0.04, x1 - 0.04, 9.07, 9.72, -0.31, -0.29, 0, 0.004)
        for sx in (-1, 1):
            bx(bm, sx * 0.46 - 0.006, sx * 0.46 + 0.006, 9.07, 9.72, -0.29, -0.26, 1)
            cylp(bm, (sx * 0.44, 9.70, -0.31), (sx * 0.44, 9.70, z0 + 0.02), 0.012, 8, 1)   # shelf posts
        for sx in (-1, 1):                                 # side stringers
            for z in (-0.55, 0.0):
                stringer_y(bm, sx * 0.47, z, 9.05, 10.55, 0.025, 0.03)
    part(N(rid, "Structure"), ["equip", "metal"], structure, node, smooth=None)

    def ceiling(bm):
        bx(bm, x0, x1, y0, y1, z1, z1 + 0.02, 0, 0.004)
        for k in range(4):
            y = 9.2 + k * 0.4
            bx(bm, x0 + 0.03, x1 - 0.03, y - 0.15, y + 0.15, z1 - 0.004, z1, 1, 0.002)
    ce = part(N(rid, "Ceiling"), ["equip", "panel"], ceiling, node, smooth=None)
    ce["cutaway"] = "hide to look into the bay from above"

    # radar dish on an az/el yoke, looking forward through the radar fairing
    def dish(bm):
        f, rr = 0.22, 0.32
        prof = [(r, r * r / (4 * f)) for r in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.29, rr]]
        back = [(r, z - 0.014) for (r, z) in reversed(prof)]
        M = Tm(DC) @ AY
        add_revolve(bm, prof + back, 32, M, closed=True, mi=0)                  # reflector shell
        add_torus(bm, rr, 0.012, 32, 6, M @ Tm((0, 0, rr * rr / (4 * f))), mi=1)   # rim
        for k in range(6):                                                       # back ribs
            a = 2 * PI * k / 6
            pts = [M @ Vector((r * math.cos(a), r * math.sin(a), r * r / (4 * f) - 0.03)) for r in (0.06, 0.18, 0.30)]
            add_tube(bm, pts, 0.01, 4, mi=1)
        add_cyl(bm, 0.07, 0.08, 16, M @ Tm((0, 0, -0.05)), mi=2)                  # rotary joint hub
        hf = M @ Vector((0, 0, f))                                               # feed horn at focus
        add_revolve(bm, [(0.022, -0.06), (0.026, -0.02), (0.05, 0.03), (0.046, 0.032), (0.02, -0.02)],
                    12, Tm(hf) @ AY @ R(PI, 'X'), closed=True, mi=1)
        for k in range(3):                                                       # feed struts
            a = PI / 2 + 2 * PI * k / 3
            p = M @ Vector((0.30 * math.cos(a), 0.30 * math.sin(a), 0.30 ** 2 / (4 * f)))
            cylp(bm, p, hf + (p - hf).normalized() * 0.03, 0.007, 6, 1)
        # waveguide from the horn down one strut, round the rim and back into the rotary joint
        a = PI / 2
        p_rim = M @ Vector((0.31 * math.cos(a), 0.31 * math.sin(a), 0.31 ** 2 / (4 * f) + 0.02))
        wg = [hf + Vector((0, -0.04, 0.03)), p_rim, p_rim + Vector((0, -0.16, 0)),
              M @ Vector((0, 0.10, -0.06))]
        add_tube(bm, fillet(wg, 0.04, 2), 0.014, 4, mi=2)
        # az/el yoke: elevation trunnions at the rim, yoke arms back to the azimuth drive
        yt = DC.y + 0.06
        for sx in (-1, 1):
            cylp(bm, (sx * 0.325, yt, DC.z), (sx * 0.37, yt, DC.z), 0.025, 12, 2)            # trunnion
            bx(bm, sx * 0.37, sx * 0.40, yt - 0.04, yt + 0.04, DC.z - 0.05, DC.z + 0.05, 2, 0.006)
            pipe(bm, [(sx * 0.385, yt, DC.z), (sx * 0.385, 10.66, DC.z), (sx * 0.10, 10.66, DC.z)], 0.018, 0.06, 8, 2)
        add_cyl(bm, 0.075, 0.10, 16, Tm((0, 10.62, DC.z)) @ AY, mi=2)              # azimuth drive
        add_cyl(bm, 0.05, 0.08, 12, Tm((0, 10.62, DC.z - 0.09)), mi=1)            # drive motor (below)
        bx(bm, -0.12, 0.12, 10.54, 10.57, DC.z - 0.12, DC.z + 0.12, 1, 0.006)      # mount plate on frame 10.55
    d = part(N(rid, "RadarDish"), ["metal", "equip", "dark"], dish, node, smooth=35)
    d["kind"] = "nose radar dish on az/el yoke, feed horn + waveguide, behind the radar fairing"

    # T/R unit + waveguide run to the dish rotary joint
    def trunit(bm):
        bx(bm, -0.19, 0.19, 10.06, 10.42, z0 + 0.02, -0.56, 0, 0.01)
        for k in range(9):                                                       # cooling fins
            x = -0.16 + k * 0.04
            bx(bm, x - 0.004, x + 0.004, 10.09, 10.39, -0.56, -0.52, 0)
        for k in range(4):                                                       # connectors (aft face)
            add_cyl(bm, 0.018, 0.03, 8, Tm((-0.12 + k * 0.08, 10.045, -0.70)) @ AY, mi=2)
        bx(bm, -0.05, 0.05, 10.42, 10.425, -0.70, -0.62, 3)                       # status window
        wg = [(0.0, 10.30, -0.52), (0.0, 10.30, -0.47), (0.0, 10.47, -0.47), (0.0, 10.47, DC.z), (0.0, 10.57, DC.z)]
        add_tube(bm, fillet([Vector(p) for p in wg], 0.035, 2), 0.016, 4, mi=1)
        flange(bm, (0.0, 10.30, -0.53), (0, 0, 1), 0.03, 0.012, 1, 4)
    part(N(rid, "TRUnit"), ["equip", "metal", "dark", "screen"], trunit, node, smooth=None)

    # avionics LRUs (linked duplicates of the R02 flight-computer box) on the deck + shelf,
    # connector faces aft, harnesses up into the ceiling tray and aft to the bulkhead gland
    fc = I.flight_computer()
    for k, (x, z) in enumerate(((-0.235, z0 + 0.02), (0.235, z0 + 0.02), (-0.235, -0.29), (0.235, -0.29)), 1):
        o = inst(N(rid, f"SensorLRU{k}"), fc, node, (x, 9.62, z), (0, 0, PI))
        o["kind"] = "sensor processing LRU (connector face aft)"
    tray = I.cable_tray(1.45, 0.14, 0.05)
    inst(N(rid, "CableTray"), tray, node, (-0.27, 9.04, z1 - 0.10))

    def harness(bm):
        for k, (x, z) in enumerate(((-0.235, z0 + 0.02), (0.235, z0 + 0.02), (-0.235, -0.29), (0.235, -0.29))):
            for j in range(3):
                xc = x - 0.11 + j * 0.055
                zc = z + 0.095
                zt = z1 - 0.07 + j * 0.012
                pts = [(xc, 9.19, zc), (xc, 9.12, zc), (xc, 9.12 - 0.0, zc + 0.08), (xc + (-0.27 - xc) * 0.6, 9.10 + 0.01 * j,
                                                                                zt - 0.06), (-0.27 + 0.02 * (j - 1), 9.10, zt)]
                pts = [p for p in pts]
                pts[2] = (xc, 9.10, min(zc + 0.08, zt - 0.10))
                add_tube(bm, fillet([Vector(p) for p in pts], 0.03, 2), 0.009, 6, mi=k % 2)
        # T/R unit harness forward along the floor deck, then up the frame to the tray
        for j in range(3):
            xc = -0.12 + j * 0.08
            add_tube(bm, fillet([Vector(p) for p in ((xc, 10.03, -0.70), (xc, 9.95, -0.70), (-0.42, 9.95, -0.70),
                                                     (-0.42, 9.95, z1 - 0.06), (-0.30 + 0.02 * j, 9.95, z1 - 0.06))],
                                0.04, 2), 0.009, 6, mi=j % 2)
        # aft bulkhead gland plate: everything leaves the bay through Y 9.0 into the cockpit wiring
        bx(bm, -0.40, -0.12, 9.0, 9.025, z1 - 0.16, z1 - 0.02, 2, 0.004)
        for j in range(4):
            add_cyl(bm, 0.02, 0.03, 10, Tm((-0.36 + j * 0.06, 9.035, z1 - 0.07)) @ AY, mi=2)
    part(N(rid, "CableHarness"), ["dark", "red", "metal"], harness, node, smooth=40)
    light_at(rid, node, dx=0.18)
    return node


BUILDERS["B01"] = build_B01


# =================================================================== landing-gear wells (B02, B04)
def gear_well(rid, node, sfx, x0, x1, y0, y1, z0, z1, leg, leg_top, drag, act_top, hyd, r_cyl, frames_y,
              well_skin_z0):
    """Open-bottomed gear well in world coordinates.  leg = (x, y) of the strut axis; leg_top = z of the
    exterior leg top; drag = (y, z) of the drag-brace upper end; act_top = (y, z) of the retraction
    actuator's ceiling clevis; hyd = (x, y) where the exterior hydraulic lines reach the well."""
    lx, ly = leg
    xc = (x0 + x1) / 2

    def structure(bm):
        t = 0.015
        # skins: side walls, roof, end walls (outside the clear volume), open below
        for sx, xw in ((-1, x0), (1, x1)):
            bx(bm, xw - (t if sx < 0 else 0), xw + (t if sx > 0 else 0), y0 - t, y1 + t, well_skin_z0, z1 + t, 0, 0.003)
        bx(bm, x0 - t, x1 + t, y0 - t, y1 + t, z1, z1 + t, 0, 0.003)
        for yw, s in ((y0, -1), (y1, 1)):
            bx(bm, x0, x1, yw - (t if s < 0 else 0), yw + (t if s > 0 else 0), well_skin_z0, z1, 0, 0.003)
        # well-opening sill (doubler round the open bottom)
        for (a, b_, c, d) in ((x0, x1, y0, y0 + 0.05), (x0, x1, y1 - 0.05, y1), (x0, x0 + 0.05, y0, y1),
                              (x1 - 0.05, x1, y0, y1)):
            bx(bm, a, b_, c, d, well_skin_z0, well_skin_z0 + 0.03, 1, 0.004)
        # U-frames (walls + roof) and stringers on the walls / roof
        for y in frames_y:
            frame_rect(bm, y, x0, x1, well_skin_z0, z1, 0.05, 0.012, 0.04, sides="LRT", mi=1, mi_f=1)
        for sx, xw in ((-1, x0), (1, x1)):
            for z in [well_skin_z0 + 0.16 + k * (z1 - well_skin_z0 - 0.3) / 2 for k in range(3)]:
                stringer_y(bm, xw - sx * 0.02, z, y0, y1, 0.04, 0.025, 1)
        for x in (x0 + 0.25 * (x1 - x0), x1 - 0.25 * (x1 - x0)):
            bx(bm, x - 0.015, x + 0.015, y0, y1, z1 - 0.04, z1, 1, 0.003)
        # end-wall stiffeners
        for yw, s in ((y0, 1), (y1, -1)):
            for x in (x0 + (x1 - x0) / 3, x1 - (x1 - x0) / 3):
                bx(bm, x - 0.012, x + 0.012, yw, yw + s * 0.04, well_skin_z0, z1, 1)
    part(N(rid, f"WellStructure{sfx}"), ["panel", "metal"], structure, node, smooth=None)

    def strut(bm):
        zt = z1 - 0.42                                                          # trunnion height
        zb = leg_top - 0.03                                                     # overlaps the exterior leg
        # oleo outer cylinder, gland nut, chrome piston band (lower end), trunnion collar
        add_revolve(bm, [(0.0, zb), (r_cyl * 0.95, zb), (r_cyl, zb + 0.02), (r_cyl, zt - 0.05), (r_cyl * 1.15, zt - 0.03),
                         (r_cyl * 1.15, zt + 0.06), (r_cyl * 0.8, zt + 0.10), (0.0, zt + 0.12)], 24,
                    Tm((lx, ly, 0)), mi=0)
        add_torus(bm, r_cyl + 0.006, 0.012, 24, 6, Tm((lx, ly, zb + 0.10)), mi=1)          # gland nut
        add_torus(bm, r_cyl + 0.006, 0.010, 24, 6, Tm((lx, ly, (zb + zt) / 2)), mi=1)      # collar
        cylp(bm, (lx, ly - r_cyl, (zb + zt) / 2), (lx, ly - r_cyl - 0.06, (zb + zt) / 2), 0.02, 8, 1)   # act. lug
        # trunnion axle across the well with bearing blocks on the side walls
        cylp(bm, (x0 + 0.02, ly, zt), (x1 - 0.02, ly, zt), 0.035, 16, 1)
        for xw, s in ((x0, 1), (x1, -1)):
            bx(bm, xw, xw + s * 0.07, ly - 0.08, ly + 0.08, zt - 0.08, zt + 0.08, 2, 0.01)
            for dy in (-0.05, 0.05):
                for dz in (-0.05, 0.05):
                    add_cyl(bm, 0.008, 0.015, 6, Tm((xw + s * 0.075, ly + dy, zt + dz)) @ AX, mi=1)
        # torque links (scissor) on the forward face of the cylinder (same side as the exterior
        # links), above the gland; the retraction actuator works on the opposite (aft) face
        za, zm, zc = zb + 0.12, zb + 0.30, zb + 0.48
        for k, (p, q) in enumerate((((lx, ly + r_cyl + 0.02, za), (lx, ly + r_cyl + 0.20, zm)),
                                    ((lx, ly + r_cyl + 0.20, zm), (lx, ly + r_cyl + 0.02, zc)))):
            for dx in (-0.025, 0.025):
                cylp(bm, (p[0] + dx, p[1], p[2]), (q[0] + dx, q[1], q[2]), 0.012, 6, 1)
        for (yy, zz) in ((ly + r_cyl + 0.02, za), (ly + r_cyl + 0.20, zm), (ly + r_cyl + 0.02, zc)):
            cylp(bm, (lx - 0.04, yy, zz), (lx + 0.04, yy, zz), 0.014, 8, 2)
        # retraction actuator: roof clevis (aft of the strut) -> strut lug
        pa = Vector((lx, act_top[0], act_top[1]))
        pb = Vector((lx, ly - r_cyl - 0.06, (zb + zt) / 2))
        bx(bm, lx - 0.06, lx + 0.06, pa.y - 0.06, pa.y + 0.06, z1 - 0.03, z1, 2, 0.006)
        cylp(bm, (lx - 0.05, pa.y, pa.z), (lx + 0.05, pa.y, pa.z), 0.015, 8, 2)
        mid = pa.lerp(pb, 0.62)
        cylp(bm, pa, mid, 0.045, 16, 3)                                            # actuator body
        cylp(bm, mid, pb, 0.022, 12, 1)                                            # piston rod
        add_torus(bm, 0.046, 0.008, 16, 4, Tm(mid) @ I.align_z(pb - pa), mi=2)
        # drag-brace upper fitting (pin across) where the exterior drag brace ends
        dy, dz = drag
        bx(bm, lx - 0.09, lx + 0.09, dy - 0.05, dy + 0.07, dz + 0.04, dz + 0.16, 2, 0.01)
        cylp(bm, (lx - 0.11, dy, dz + 0.06), (lx + 0.11, dy, dz + 0.06), 0.022, 12, 1)
        for sx in (-1, 1):
            bx(bm, lx + sx * 0.07 - 0.012, lx + sx * 0.07 + 0.012, dy - 0.03, dy + 0.03, dz - 0.02, dz + 0.10, 2)
        # uplock hook box on the roof
        bx(bm, lx - 0.05, lx + 0.05, ly + 0.22, ly + 0.36, z1 - 0.14, z1, 2, 0.008)
        add_torus(bm, 0.035, 0.01, 12, 4, Tm((lx, ly + 0.29, z1 - 0.16)) @ AX, mi=1)
    st = part(N(rid, f"Strut{sfx}"), ["panel", "metal", "equip", "dark"], strut, node, smooth=35)
    st["kind"] = "landing-gear oleo upper cylinder, trunnion, torque links, retraction actuator"

    def hydraulics(bm):
        hx, hy = hyd
        xw = x0 + 0.05
        for j in range(3):
            dz = j * 0.035
            # along the port wall from the forward end, down to the actuator / strut
            pts = [(xw, y1 - 0.01, z1 - 0.10 - dz), (xw, ly + 0.10 + 0.03 * j, z1 - 0.10 - dz),
                   (xw + 0.10, ly + 0.10 + 0.03 * j, z1 - 0.10 - dz)]
            if j < 2:                                               # supply / return to the actuator
                pa = Vector((lx, act_top[0], act_top[1]))
                pts = [(xw, y1 - 0.01, z1 - 0.10 - dz), (xw, pa.y + 0.03 * j, z1 - 0.10 - dz),
                       (lx - 0.06, pa.y + 0.03 * j, z1 - 0.10 - dz), (lx - 0.06, pa.y + 0.03 * j, pa.z - 0.08)]
                pipe(bm, pts, 0.008, 0.04, 6, 0)
            else:                                                   # brake / downlock line down the strut
                pts = [(xw, y1 - 0.01, z1 - 0.10 - dz), (xw, hy, z1 - 0.10 - dz), (hx, hy, z1 - 0.18),
                       (hx, hy, leg_top - 0.05)]
                pipe(bm, pts, 0.008, 0.05, 6, 0)
        for k in range(4):                                          # P-clips along the wall run
            y = y1 - 0.15 - k * (y1 - ly) / 4
            bx(bm, xw - 0.03, xw + 0.012, y - 0.012, y + 0.012, z1 - 0.20, z1 - 0.06, 1)
        add_cyl(bm, 0.035, 0.10, 12, Tm((xw + 0.02, y1 - 0.12, z1 - 0.30)), mi=1)   # accumulator / filter
    part(N(rid, f"HydraulicLines{sfx}"), ["rail", "metal"], hydraulics, node, smooth=35)


def build_B02():
    rid = "B02"
    node = bay_node(rid, "NoseGearWell")
    gear_well(rid, node, "", -0.45, 0.45, 6.3, 8.2, -1.70, -0.45, (0.0, 7.31), -1.75, (6.36, -1.66),
              (6.75, -0.50), (0.09, 7.14), 0.085, (6.38, 6.85, 7.75, 8.12), -1.70)
    light_at(rid, node, dx=0.22)
    return node


BUILDERS["B02"] = build_B02


# =================================================================== B03 ventral ordnance bay
def build_B03():
    rid = "B03"
    node = bay_node(rid, "OrdnanceBay")
    x0, x1, y0, y1, z0, z1 = -0.85, 0.85, -2.0, 2.6, -1.75, -0.6
    ZD = -1.80                                                     # inner face of the closed doors

    def rib(bm):                                                   # one bay rib (walls + roof), at y = 0
        frame_rect(bm, 0.0, x0, x1, ZD + 0.03, z1, 0.07, 0.014, 0.05, sides="LRT", mi=0, mi_f=1)
        for sx in (-1, 1):                                         # lightening holes read as dark discs
            for z in (-1.05, -1.40):
                add_cyl(bm, 0.022, 0.016, 12, Tm((sx * (0.85 - 0.035), 0.0, z)) @ AY, mi=2)
    rib_me = lib("B03Rib", ["equip", "metal", "dark"], rib, None)
    for k, y in enumerate([y0 + 0.05 + i * (y1 - y0 - 0.1) / 8 for i in range(9)], 1):
        inst(N(rid, f"Rib{k:02d}"), rib_me, node, (0, y, 0))

    def walls(bm):
        t = 0.012
        for sx, xw in ((-1, x0), (1, x1)):
            bx(bm, xw - (t if sx < 0 else 0), xw + (t if sx > 0 else 0), y0, y1, ZD, z1 + t, 0, 0.003)
            for z in (-0.85, -1.20, -1.55):
                stringer_y(bm, xw - sx * 0.02, z, y0, y1, 0.04, 0.025, 1)
        bx(bm, x0 - t, x1 + t, y0, y1, z1, z1 + t, 0, 0.003)
        for yw, s in ((y0, -1), (y1, 1)):
            bx(bm, x0, x1, yw - (t if s < 0 else 0), yw + (t if s > 0 else 0), ZD, z1, 0, 0.003)
            for x in (-0.55, 0.0, 0.55):
                bx(bm, x - 0.015, x + 0.015, yw, yw - s * 0.05, ZD, z1, 1)
        # hardpoint beams under the roof (carry the ejector racks at the internal hardpoint sockets)
        for x in (-0.40, 0.40):
            bx(bm, x - 0.05, x + 0.05, y0 + 0.05, y1 - 0.05, z1 - 0.10, z1, 1, 0.006)
    part(N(rid, "Structure"), ["panel", "metal"], walls, node, smooth=None)

    # launcher rails at the internal hardpoint sockets (ejector rack + rail + hooks + umbilical)
    def launcher(bm):                                              # local: socket at origin, rail along +Y
        bx(bm, -0.07, 0.07, -0.45, 0.45, -0.09, 0.0, 0, 0.01)                  # ejector rack
        for y in (-0.35, 0.35):
            add_cyl(bm, 0.03, 0.06, 12, Tm((0, y, -0.12)), mi=1)              # ejector pistons
        bx(bm, -0.035, 0.035, -0.85, 0.85, -0.17, -0.09, 1, 0.008)             # launch rail
        for y in (-0.62, 0.0, 0.62):
            bx(bm, -0.02, 0.02, y - 0.02, y + 0.02, -0.20, -0.17, 1)            # hooks / lugs
        for sx in (-1, 1):                                                       # sway braces
            cylp(bm, (sx * 0.06, -0.25, -0.08), (sx * 0.10, -0.25, -0.185), 0.008, 6, 1)
            cylp(bm, (sx * 0.06, 0.30, -0.08), (sx * 0.10, 0.30, -0.185), 0.008, 6, 1)
            add_cyl(bm, 0.018, 0.01, 8, Tm((sx * 0.10, -0.25, -0.19)), mi=2)
            add_cyl(bm, 0.018, 0.01, 8, Tm((sx * 0.10, 0.30, -0.19)), mi=2)
        pipe(bm, [(0.05, 0.45, -0.05), (0.05, 0.60, -0.05), (0.05, 0.70, -0.20)], 0.012, 0.05, 6, 3)   # umbilical
        bx(bm, 0.03, 0.08, 0.68, 0.74, -0.24, -0.19, 3, 0.006)
    lm = lib("B03Launcher", ["equip", "metal", "dark", "red"], launcher, 35)
    for k, sn in enumerate(("SOCKET_Hardpoint_1_BayL", "SOCKET_Hardpoint_2_BayR"), 1):
        p = I.sock(sn)
        o = inst(N(rid, f"LauncherRail{k}"), lm, node, (p.x, p.y, p.z))
        o["socket"] = sn
        o["kind"] = "ejector rack + launch rail at the internal hardpoint"

    # missiles: linked duplicates of the exterior sponson missile (body, fins, seeker), hung under the
    # rails.  Exterior missile 1 hangs 0.15 m below its socket at (-1.925, -5.80, -1.70).
    ref = I.sock("SOCKET_Hardpoint_3_SponsonL")
    for k, sn in enumerate(("SOCKET_Hardpoint_1_BayL", "SOCKET_Hardpoint_2_BayR"), 1):
        p = I.sock(sn)
        delta = Vector((p.x - ref.x, p.y - ref.y, (p.z - 0.20) - (ref.z - 0.15)))
        for suffix in ("", "_Fins", "_Seeker"):
            src = bpy.data.objects["EXT_WEAPONS_Missile_1" + suffix]
            o = place(N(rid, f"Missile{k}{suffix.replace('_', '')}"), src.data, node,
                      (0, 0, 0), Tm(delta) @ src.matrix_world)
            o["kind"] = "missile (linked duplicate of the exterior sponson missile mesh)"

    # inner face of the closed bay doors (split on the centreline, hinged at the bay walls)
    def doors(bm):
        for sx in (-1, 1):
            xa, xb = sorted((sx * 0.006, sx * 0.845))
            bx(bm, xa, xb, y0 + 0.02, y1 - 0.02, ZD - 0.015, ZD, 0, 0.004)            # inner skin
            for y in [y0 + 0.30 + i * 0.50 for i in range(9)]:                       # transverse hat stiffeners
                if y > y1 - 0.1:
                    continue
                bx(bm, xa + 0.03, xb - 0.03, y - 0.02, y + 0.02, ZD, ZD + 0.035, 1, 0.006)
            for x in (sx * 0.28, sx * 0.60):                                         # longitudinal stiffeners
                bx(bm, x - 0.015, x + 0.015, y0 + 0.05, y1 - 0.05, ZD, ZD + 0.03, 1, 0.004)
            for y in (y0 + 0.40, 0.3, y1 - 0.40):                                      # hinges at the wall
                cylp(bm, (sx * 0.85, y - 0.08, ZD + 0.02), (sx * 0.85, y + 0.08, ZD + 0.02), 0.018, 10, 2)
                bx(bm, sx * 0.80 - 0.02, sx * 0.80 + 0.02, y - 0.06, y + 0.06, ZD, ZD + 0.03, 2)
            # door actuators: wall clevis -> door horn
            for y in (y0 + 0.9, y1 - 0.9):
                pa = Vector((sx * 0.80, y, -1.25))
                pb = Vector((sx * 0.55, y, ZD + 0.04))
                m = pa.lerp(pb, 0.55)
                cylp(bm, pa, m, 0.03, 12, 3)
                cylp(bm, m, pb, 0.014, 8, 2)
                bx(bm, sx * 0.83 - 0.02, sx * 0.83 + 0.02, y - 0.04, y + 0.04, -1.30, -1.20, 2)
                bx(bm, sx * 0.55 - 0.03, sx * 0.55 + 0.03, y - 0.03, y + 0.03, ZD, ZD + 0.06, 2)
        bx(bm, -0.006, 0.006, y0 + 0.02, y1 - 0.02, ZD - 0.01, ZD + 0.01, 4)            # centre seal
    dr = part(N(rid, "DoorsInner"), ["panel", "metal", "rail", "equip", "dark"], doors, node, smooth=None)
    dr["kind"] = "inner face of the closed ventral bay doors (EXT_WEAPONS_BayDoor_L/R), hinges + actuators"
    light_at(rid, node)
    return node


BUILDERS["B03"] = build_B03


# =================================================================== B04 main gear wells (L/R)
def build_B04():
    rid = "B04"
    nr = bay_node(rid, "MainGearWell_R")
    nl = bay_node(rid, "MainGearWell_L")
    before = set(bpy.data.objects)
    gear_well(rid, nr, "_R", 1.60, 2.25, -4.40, -2.20, -1.45, -0.25, (1.925, -3.30), -1.60, (-4.22, -1.52),
              (-3.95, -0.30), (2.02, -3.48), 0.09, (-4.35, -3.85, -2.75, -2.25), -1.45)
    made = [o for o in bpy.data.objects if o not in before]
    bpy.context.view_layer.update()
    # port well = mirrored linked duplicates of the starboard well (scale X -1 about the centreline)
    for o in made:
        lo = place(o.name[:-2] + "_L", o.data, nl, (0, 0, 0), MIRROR_X @ o.matrix_world)
        for k in o.keys():
            lo[k] = o[k]
        lo["mirror_of"] = o.name
    pl = I.sock("SOCKET_Light_B04")                       # socket sits in the port well
    wl = inst(N(rid, "WorkLight_L"), I.work_light(), nl, (pl.x - 0.15, pl.y + 0.45, pl.z + 0.05))
    wl["socket"] = "SOCKET_Light_B04"
    inst(N(rid, "WorkLight_R"), I.work_light(), nr, (-pl.x + 0.15, pl.y + 0.45, pl.z + 0.05))
    return nr


BUILDERS["B04"] = build_B04


# =================================================================== B05 propellant tank bay
TANK_Y = (-5.30, -6.60)
TANK_Z = -0.62
SPHERES = [(sx * 0.70, y, 0.24) for y in TANK_Y for sx in (-1, 1)]
FEED_X, FEED_Z = 0.80, -1.40                               # feed-line run B05 -> B06 -> B07


def puffy_tank(bm, L, Rr, nu=32, nv=32, dome=0.30, puff=0.018, pu=0.19, pv=14, mi=0):
    """Foil-wrapped (MLI) tank along local X: cylinder + elliptic domes; the radius is pillowed
    between stitch lines (pitch pu along the axis, pv pillows round) so it reads as quilted foil."""
    rings = []
    for i in range(1, nu):
        u = -L / 2 + L * i / nu
        a = abs(u) - (L / 2 - dome)
        r = Rr * math.sqrt(max(0.0, 1 - (a / dome) ** 2)) if a > 0 else Rr
        ring = []
        for j in range(nv):
            v = 2 * PI * j / nv
            p = abs(math.sin(PI * u / pu)) * abs(math.sin(v * pv / 2))
            rr = r + puff * p * (r / Rr)
            ring.append(bm.verts.new((u, rr * math.cos(v), rr * math.sin(v))))
        rings.append(ring)
    p0 = bm.verts.new((-L / 2, 0, 0))
    p1 = bm.verts.new((L / 2, 0, 0))
    for a, b_ in zip(rings, rings[1:]):
        for j in range(nv):
            f = bm.faces.new((a[j], a[(j + 1) % nv], b_[(j + 1) % nv], b_[j]))
            f.material_index = mi
    for j in range(nv):
        bm.faces.new((p0, rings[0][(j + 1) % nv], rings[0][j])).material_index = mi
        bm.faces.new((p1, rings[-1][j], rings[-1][(j + 1) % nv])).material_index = mi


def build_B05():
    rid = "B05"
    node = bay_node(rid, "PropellantTankBay")
    x0, x1, y0, y1, z0, z1 = -1.0, 1.0, -7.1, -4.8, -1.6, 0.5
    L, RR = 1.84, 0.42

    def tank(bm):                                              # local: axis along X, centre at origin
        puffy_tank(bm, L, RR, mi=0)
        for u in (-0.55, 0.0, 0.55):                           # strap bands over the foil
            add_revolve(bm, [(RR + 0.012, -0.022), (RR + 0.028, -0.022), (RR + 0.028, 0.022), (RR + 0.012, 0.022)],
                        32, Tm((u, 0, 0)) @ R(PI / 2, 'Y'), closed=True, mi=1)
            bx(bm, u - 0.03, u + 0.03, -0.05, 0.05, -RR - 0.06, -RR - 0.01, 1, 0.006)   # band buckle / saddle pad
        for x in (-0.30, 0.30):                                # sump outlets (bottom)
            add_cyl(bm, 0.045, 0.06, 16, Tm((x, 0, -RR - 0.02)), mi=1)
        add_cyl(bm, 0.05, 0.06, 16, Tm((0.55, 0, RR + 0.02)), mi=1)          # pressurant inlet boss
        for sx in (-1, 1):                                     # pole fittings
            add_cyl(bm, 0.06, 0.04, 16, Tm((sx * (L / 2 + 0.01), 0, 0)) @ R(PI / 2, 'Y'), mi=1)
        bx(bm, -0.16, 0.16, -0.003, 0.003, 0.18, 0.30, 2)                     # label (on the band side)
    tm = lib("B05PropTank", ["foil", "metal", "panel"], tank, 40)
    for k, y in enumerate(TANK_Y, 1):
        o = inst(N(rid, f"PropTank{k}"), tm, node, (0, y, TANK_Z))
        o["kind"] = "foil-wrapped (MLI) propellant tank with strap bands"

    def sphere(bm):
        add_revolve(bm, [(math.sin(a) * 0.19, -math.cos(a) * 0.19) for a in
                         [PI * i / 10 for i in range(11)]], 20, mi=0)
        add_torus(bm, 0.19, 0.008, 20, 4, mi=1)                                # girth weld band
        add_cyl(bm, 0.035, 0.05, 12, Tm((0, 0, 0.21)), mi=1)                  # bosses
        add_cyl(bm, 0.035, 0.05, 12, Tm((0, 0, -0.21)), mi=1)
        add_torus(bm, 0.13, 0.012, 16, 4, Tm((0, 0, -0.14)), mi=1)             # cradle ring
        for k in range(4):
            a = PI / 4 + k * PI / 2
            cylp(bm, (0.13 * math.cos(a), 0.13 * math.sin(a), -0.14), (0.16 * math.cos(a), 0.16 * math.sin(a), -0.26),
                 0.01, 6, 1)
    sm = lib("B05PressSphere", ["equip", "metal"], sphere, 40)
    for k, p in enumerate(SPHERES, 1):
        o = inst(N(rid, f"PressurantSphere{k}"), sm, node, p)
        o["kind"] = "spherical pressurant tank (COPV)"

    def truss(bm):
        # floor longerons + cross beams, saddle posts and diagonals under the tanks
        for sx in (-1, 1):
            bx(bm, sx * 0.75 - 0.03, sx * 0.75 + 0.03, y0 + 0.05, y1 - 0.05, z0 + 0.02, z0 + 0.10, 0, 0.006)
            bx(bm, sx * 0.95 - 0.025, sx * 0.95 + 0.025, y0 + 0.05, y1 - 0.05, -0.08, -0.02, 0, 0.005)  # upper
            bx(bm, sx * 0.95 - 0.025, sx * 0.95 + 0.025, y0 + 0.05, y1 - 0.05, z1 - 0.06, z1 - 0.01, 0, 0.005)
        for y in TANK_Y:
            bx(bm, -0.95, 0.95, y - 0.03, y + 0.03, z0 + 0.02, z0 + 0.10, 0, 0.006)
            bx(bm, -0.95, 0.95, y - 0.025, y + 0.025, -0.08, -0.02, 0, 0.005)  # upper cross beam
            for u in (-0.55, 0.55):
                zt = TANK_Z - RR - 0.06
                cylp(bm, (u, y, z0 + 0.10), (u, y, zt), 0.025, 10, 1)            # saddle posts
                bx(bm, u - 0.06, u + 0.06, y - 0.20, y + 0.20, zt - 0.03, zt, 1, 0.01)
                for s in (-1, 1):
                    cylp(bm, (u, y + s * 0.18, zt - 0.03), (u + (0.75 * (1 if u > 0 else -1) - u), y, z0 + 0.10),
                         0.015, 8, 1)
            for sx in (-1, 1):                                                   # side posts
                cylp(bm, (sx * 0.975, y, z0 + 0.10), (sx * 0.975, y, z1 - 0.06), 0.02, 10, 1)
                cylp(bm, (sx * 0.975, y, -0.08), (sx * 0.75, y, z0 + 0.10), 0.015, 8, 1)
        # light cross-member under the C01 floor (work light)
        bx(bm, 0.30, 0.95, -5.98, -5.92, z1 - 0.06, z1 - 0.01, 0, 0.004)
        # ladder landing (grating frame + bars) between the tanks under the C01 hatch
        zl = -1.25
        for (a, b_, c, d) in ((-0.36, 0.36, -6.12, -6.09), (-0.36, 0.36, -5.81, -5.78), (-0.36, -0.33, -6.12, -5.78),
                              (0.33, 0.36, -6.12, -5.78)):
            bx(bm, a, b_, c, d, zl - 0.03, zl, 1)
        for k in range(1, 12):
            x = -0.36 + 0.72 * k / 12
            bx(bm, x - 0.003, x + 0.003, -6.09, -5.81, zl - 0.026, zl - 0.002, 1)
        bx(bm, -0.75, 0.75, -5.98, -5.92, z0 + 0.02, z0 + 0.10, 0, 0.006)     # landing cross beam
        for sx in (-1, 1):
            for y in (-6.10, -5.80):
                cylp(bm, (sx * 0.12, y, zl - 0.03), (sx * 0.12, -5.95, z0 + 0.10), 0.014, 8, 1)
    part(N(rid, "SupportTruss"), ["equip", "metal"], truss, node, smooth=None)

    def ladder(bm):                                           # C01 hatch (Y -5.95) down to the landing
        za, zb = -1.25, 0.595
        for x in (-0.19, 0.19):
            bx(bm, x - 0.018, x + 0.018, -5.97, -5.93, za, zb, 0, 0.005)
        n = int((zb - za - 0.15) / 0.28)
        for i in range(n + 1):
            z = za + 0.2 + i * 0.28
            if z < zb - 0.05:
                cylp(bm, (-0.19, -5.95, z), (0.19, -5.95, z), 0.014, 8, 1)
    lad = part(N(rid, "LadderC01"), ["metal", "rail"], ladder, node, smooth=None)
    lad["kind"] = "ladder from the C01 floor hatch (centre Y -5.95, hinge Y -5.70) down to the tank-bay landing"

    def plumbing(bm):
        # sump lines under the tanks -> aft manifold -> feed lines out through the B06 shield glands
        for sx in (-1, 1):
            x = sx * 0.30
            for y in TANK_Y:
                pipe(bm, [(x, y, TANK_Z - RR - 0.05), (x, y, FEED_Z)], 0.03, 0.05, 10, 0)
                flange(bm, (x, y, TANK_Z - RR - 0.07), (0, 0, 1), 0.05, 0.012, 1, 6)
            pipe(bm, [(x, TANK_Y[0] + 0.03, FEED_Z), (x, -6.95, FEED_Z), (sx * FEED_X, -6.95, FEED_Z),
                      (sx * FEED_X, -7.16, FEED_Z)], 0.035, 0.08, 12, 0)
            valve(bm, (x, -6.30, FEED_Z), (0, 1, 0), 0.035)                    # isolation valves
            flange(bm, (sx * FEED_X, -7.08, FEED_Z), (0, 1, 0), 0.06, 0.015, 1, 6)
        pipe(bm, [(-0.30, -6.80, FEED_Z), (0.30, -6.80, FEED_Z)], 0.025, 0.05, 10, 0)   # cross-feed
        valve(bm, (0.0, -6.80, FEED_Z), (1, 0, 0), 0.025)
        # pressurant: sphere bottom boss -> regulator -> tank pressurant boss
        for (x, y, z) in SPHERES:
            sx = 1 if x > 0 else -1
            yo = y + 0.10
            pts = [(x, y, z - 0.235), (x, yo, z - 0.235), (x, yo, -0.12), (0.55 * sx, yo, -0.12), (0.55 * sx, y, -0.12),
                   (0.55 * sx, y, TANK_Z + RR + 0.04)]
            pipe(bm, pts, 0.012, 0.03, 8, 0)
            bx(bm, 0.63 * sx - 0.04, 0.63 * sx + 0.04, yo - 0.035, yo + 0.035, -0.16, -0.08, 2, 0.008)    # regulator
        for sx in (-1, 1):                                                         # balance lines
            pipe(bm, [(sx * 0.70, TANK_Y[0] + 0.10, -0.12), (sx * 0.70, TANK_Y[1] + 0.10, -0.12)], 0.01, 0.04, 6, 0)
    pl = part(N(rid, "FeedLines"), ["metal", "rail", "equip", "dark"], plumbing, node, smooth=35)
    pl["kind"] = "sump + feed lines (aft through the B06 shield glands), cross-feed, pressurant lines + regulators"
    light_at(rid, node, dx=0.55)
    return node


BUILDERS["B05"] = build_B05


# =================================================================== B06 reactor compartment
R06_PORTS = [(-0.95, 1.20, 0.05), (-0.80, 1.20, 0.05), (-0.65, 1.20, 0.05), (-0.95, 1.02, 0.04),
             (0.62, 1.20, 0.05), (0.78, 1.20, 0.05), (0.94, 1.20, 0.05), (0.62, -0.08, 0.06), (0.92, 0.62, 0.06)]
RC = Vector((0.0, -8.15, 0.0))                              # reactor axis (x, y)


def build_B06():
    rid = "B06"
    node = bay_node(rid, "ReactorCompartment")
    x0, x1, y0, y1, z0, z1 = -1.0, 1.0, -8.9, -7.2, -1.6, 0.5
    SY0, SY1 = -7.22, -7.40                                  # shadow-shield plate faces

    def shield(bm):
        # laminated shadow shield: main plate across the bay + shoulders either side of the crawlway,
        # so the R06 aft-bulkhead penetrations (same X/Z) continue straight through it
        for (a, b_, c, d) in ((-0.97, 0.97, -1.57, 0.47), (-1.05, -0.55, 0.47, 1.45), (0.55, 1.05, 0.47, 1.45)):
            bx(bm, a, b_, SY1, SY0, c, d, 0, 0.012)
            bx(bm, a + 0.02, b_ - 0.02, SY0, SY0 + 0.004, c + 0.02, d - 0.02, 1, 0.003)   # fwd face doubler
            bx(bm, a + 0.02, b_ - 0.02, SY1 - 0.004, SY1, c + 0.02, d - 0.02, 1, 0.003)   # aft face doubler
            for k in range(1, 3):                                                       # laminate lines
                yy = SY0 - k * (SY0 - SY1) / 3
                bx(bm, a - 0.002, a + 0.002, yy - 0.002, yy + 0.002, c, d, 2)
                bx(bm, b_ - 0.002, b_ + 0.002, yy - 0.002, yy + 0.002, c, d, 2)
            nb = max(2, int((b_ - a) / 0.2))
            for i in range(nb + 1):                                                     # edge bolts (fwd face)
                x = a + 0.04 + (b_ - a - 0.08) * i / nb
                for z in (c + 0.04, d - 0.04):
                    add_cyl(bm, 0.012, 0.01, 4, Tm((x, SY0 + 0.006, z)) @ AY, mi=1)
        ports = [(x, z, r) for (x, z, r) in R06_PORTS] + [(-FEED_X, FEED_Z, 0.035), (FEED_X, FEED_Z, 0.035)]
        for (x, z, r) in ports:
            for yf, s in ((SY0, 1), (SY1, -1)):
                Mp = Tm((x, yf, z)) @ AY @ (Matrix.Identity(4) if s > 0 else R(PI, 'X'))
                add_cyl(bm, r + 0.03, 0.02, 16, Mp @ Tm((0, 0, 0.01)), mi=1)            # flange
                add_cyl(bm, r + 0.005, 0.05, 12, Mp @ Tm((0, 0, 0.035)), mi=1)          # gland body
                for k in range(4):
                    a = k * PI / 2 + PI / 4
                    add_cyl(bm, 0.006, 0.01, 4, Mp @ Tm(((r + 0.02) * math.cos(a), (r + 0.02) * math.sin(a), 0.025)),
                            mi=2)
        # lifting eyes on top of the main plate
        for x in (-0.80, 0.80):
            add_torus(bm, 0.035, 0.01, 12, 4, Tm((x, (SY0 + SY1) / 2, 0.51)) @ AX, mi=1)
    sp = part(N(rid, "ShieldPlate"), ["metal", "equip", "dark"], shield, node, smooth=40)
    sp["kind"] = ("laminated shadow shield toward R06; 9 penetration glands at the same X/Z as the R06 aft-bulkhead "
                  "penetrations + 2 feed-line glands")

    def conduits(bm):                                       # R06 aft-bulkhead penetrations -> shield glands
        for (x, z, r) in R06_PORTS:
            if z < 0.5:
                continue                                    # (0.62, -0.08) port: gland only (tank bay below)
            add_tube(bm, [Vector((x, -4.80, z)), Vector((x, SY0 + 0.05, z))], r * 0.8, 8, mi=0)
            for y in (-5.4, -6.1, -6.8):                    # P-clips
                add_torus(bm, r * 0.8 + 0.006, 0.006, 12, 4, Tm((x, y, z)) @ AY, mi=1)
        for sx in (-1, 1):                                  # hangers for the conduit bundles
            for y in (-5.4, -6.1, -6.8):
                bx(bm, sx * 0.60, sx * 1.0, y - 0.015, y + 0.015, 1.27, 1.29, 1)
                bx(bm, sx * 0.57 - 0.01, sx * 0.57 + 0.01, y - 0.015, y + 0.015, 0.55, 1.29, 1)
    cd = part(N(rid, "Conduits"), ["dark", "metal"], conduits, node, smooth=40)
    cd["kind"] = "cable conduits from the R06 aft-bulkhead penetrations to the shield glands (run beside C01)"

    def reactor(bm):
        cx, cy = RC.x, RC.y
        M = Tm((cx, cy, 0))
        add_revolve(bm, [(0.0, -1.50), (0.20, -1.48), (0.32, -1.43), (0.36, -1.36), (0.36, -0.64), (0.42, -0.64),
                         (0.42, -0.58), (0.34, -0.58), (0.28, -0.52), (0.12, -0.48), (0.0, -0.47)], 32, M, mi=0)
        for z in (-1.30, -1.00, -0.75):                                             # vessel rings
            add_torus(bm, 0.362, 0.01, 32, 4, M @ Tm((0, 0, z)), mi=1)
        # control drum ring: 8 drums round the core, drive ring + motors on top
        add_revolve(bm, [(0.36, -0.62), (0.57, -0.62), (0.57, -0.58), (0.36, -0.58)], 32, M, closed=True, mi=1)
        add_revolve(bm, [(0.36, -1.40), (0.57, -1.40), (0.57, -1.37), (0.36, -1.37)], 32, M, closed=True, mi=1)
        for k in range(8):
            a = PI / 8 + k * PI / 4
            p = (cx + 0.47 * math.cos(a), cy + 0.47 * math.sin(a))
            add_cyl(bm, 0.065, 0.74, 12, Tm((p[0], p[1], -1.0)), mi=2)                # drum
            bx(bm, -0.005, 0.005, -0.066, 0.066, -0.36, 0.36, 3,                        # absorber face stripe
               M=Tm((p[0], p[1], -1.0)) @ R(a, 'Z') @ Tm((0.062, 0, 0)))
            add_cyl(bm, 0.035, 0.09, 12, Tm((p[0], p[1], -0.535)), mi=1)              # drive motor
            add_cyl(bm, 0.02, 0.03, 8, Tm((p[0], p[1], -0.475)), mi=3)
        # instrument / control cables from the head up to a junction on the catwalk frame
        for k in range(3):
            a = PI / 2 + (k - 1) * 0.35
            q0 = Vector((cx + 0.10 * math.cos(a), cy + 0.10 * math.sin(a), -0.48))
            add_tube(bm, fillet([q0, q0 + Vector((0, 0, 0.06)), Vector((0.55 + 0.03 * k, -7.78, -0.38)),
                                 Vector((0.85, -7.78, -0.38))], 0.05, 2), 0.01, 6, mi=3)
        # support skirt legs to the floor beams
        for k in range(4):
            a = k * PI / 2
            p0 = Vector((cx + 0.30 * math.cos(a), cy + 0.30 * math.sin(a), -1.45))
            p1 = Vector((cx + 0.62 * math.cos(a), cy + 0.62 * math.sin(a), z0 + 0.08))
            cylp(bm, p0, p1, 0.03, 10, 1)
    rv = part(N(rid, "ReactorVessel"), ["equip", "metal", "panel", "dark"], reactor, node, smooth=35)
    rv["kind"] = "reactor vessel with control-drum ring (8 drums + drive motors)"

    def structure(bm):
        frame_rect(bm, -8.86, x0 + 0.03, x1 - 0.03, z0 + 0.03, z1 - 0.03, 0.06, 0.012, 0.045, mi=0, mi_f=1)
        for sx in (-1, 1):
            for z in (z0 + 0.06, z1 - 0.06):
                stringer_y(bm, sx * 0.94, z, -8.86, SY1, 0.05, 0.05)
        for y in (-7.60, -8.70):                                                     # floor beams
            bx(bm, -0.95, 0.95, y - 0.03, y + 0.03, z0 + 0.03, z0 + 0.09, 0, 0.005)
        bx(bm, -0.03, 0.03, -8.70, -7.60, z0 + 0.03, z0 + 0.09, 0, 0.005)
        for sx in (-1, 1):
            bx(bm, sx * 0.62 - 0.03, sx * 0.62 + 0.03, -8.80, -7.50, z0 + 0.03, z0 + 0.09, 0, 0.005)
        # catwalk across the reactor head under the C01 hatch (centre Y -8.05)
        zc = -0.30
        for y in (-8.36, -7.74):
            bx(bm, -0.95, 0.95, y - 0.02, y + 0.02, zc - 0.10, zc, 0, 0.004)            # edge beams
        for sx in (-1, 1):
            cylp(bm, (sx * 0.94, -8.36, zc), (sx * 0.94, -8.36, z1 - 0.06), 0.02, 8, 0)  # hangers
            cylp(bm, (sx * 0.94, -7.74, zc), (sx * 0.94, -7.74, z1 - 0.06), 0.02, 8, 0)
            bx(bm, sx * 0.95 - 0.03, sx * 0.95 + 0.03, -8.40, -7.70, z1 - 0.08, z1 - 0.02, 0, 0.005)
        # handrails: forward edge full width, aft edge with a gap at the ladder
        for (xa, xb, y) in ((-0.92, 0.92, -7.76), (-0.92, -0.28, -8.34), (0.28, 0.92, -8.34)):
            cylp(bm, (xa, y, zc + 0.55), (xb, y, zc + 0.55), 0.016, 8, 2)
            cylp(bm, (xa, y, zc + 0.28), (xb, y, zc + 0.28), 0.012, 8, 2)
            for x in (xa, xb):
                cylp(bm, (x, y, zc), (x, y, zc + 0.56), 0.016, 8, 2)
            bx(bm, xa, xb, y - 0.004, y + 0.004, zc, zc + 0.08, 2)                      # toe board
        # ladder: catwalk -> C01 floor hatch (aft edge of the hatch opening)
        for x in (-0.19, 0.19):
            bx(bm, x - 0.018, x + 0.018, -8.30, -8.26, zc, 0.595, 1, 0.005)
        for i in range(3):
            z = zc + 0.22 + i * 0.26
            cylp(bm, (-0.19, -8.28, z), (0.19, -8.28, z), 0.014, 8, 2)
    st = part(N(rid, "Structure"), ["equip", "metal", "rail"], structure, node, smooth=None)
    st["kind"] = "bay frame + floor beams, catwalk frame / handrails / ladder from the C01 hatch"
    gm = I.grate_tile(0.95, 0.58)
    for k, x in enumerate((-0.475, 0.475), 1):
        inst(N(rid, f"CatwalkGrate{k}"), gm, node, (x, -8.05, -0.30))

    def coolant(bm):                                       # primary loop: hot + cold legs aft to B07
        for z, r, mi in ((-0.75, 0.05, 0), (-1.10, 0.045, 1)):
            p0 = Vector((RC.x, RC.y - 0.34, z))
            pipe(bm, [p0, Vector((RC.x, -8.97, z))], r, 0.05, 14, mi)
            flange(bm, (RC.x, RC.y - 0.37, z), (0, 1, 0), r + 0.03, 0.02, 2, 8)
            flange(bm, (RC.x, -8.88, z), (0, 1, 0), r + 0.03, 0.02, 2, 8)
            add_torus(bm, r + 0.004, 0.006, 16, 4, Tm((RC.x, -8.72, z)) @ AY, mi=2)
        valve(bm, (RC.x, -8.75, -1.10), (0, 1, 0), 0.04, up=(1, 0, 0), mi_body=2, mi_wheel=3)
        # feed lines from B05 continue along the lower corners to B07
        for sx in (-1, 1):
            pipe(bm, [(sx * FEED_X, SY1 - 0.04, FEED_Z), (sx * FEED_X, -8.97, FEED_Z)], 0.035, 0.05, 12, 4)
            for y in (-7.8, -8.5):
                add_torus(bm, 0.041, 0.006, 12, 4, Tm((sx * FEED_X, y, FEED_Z)) @ AY, mi=2)
                bx(bm, sx * FEED_X - 0.015, sx * FEED_X + 0.015, y - 0.01, y + 0.01, z0 + 0.06, FEED_Z - 0.04, 2)
    cl = part(N(rid, "CoolantLoop"), ["red", "equip", "metal", "rail", "metal"], coolant, node, smooth=35)
    cl["kind"] = "primary coolant hot/cold legs to B07 + B05->B07 feed lines"

    rp = rad_plate_mesh()
    face_fwd = R(-PI / 2, 'X')                     # local +Z -> world +Y (plate faces forward)
    face_aft = R(PI / 2, 'X')                      # local +Z -> world -Y
    for k, (loc, fwd) in enumerate((((-0.40, SY0 + 0.004, 0.05), True), ((0.40, SY0 + 0.004, 0.05), True),
                                    ((0.80, SY0 + 0.004, 1.10), True), ((0.60, -7.735, -0.18), True),
                                    ((-0.60, -8.365, -0.18), False)), 1):
        o = inst(N(rid, f"RadHazardPlate{k}"), rp, node, loc, face_fwd if fwd else face_aft)
        o["kind"] = "radiation hazard marking plate"
    light_at(rid, node, dx=-0.55)
    return node


BUILDERS["B06"] = build_B06


# =================================================================== B07 engine bay
ENG = [(-1.2, -0.5), (1.2, -0.5)]                          # engine axes (x, z): SOCKET_Engine_Main_L/R
ACTS = [(-0.743, -0.50), (-1.20, -0.043), (1.657, -0.50), (1.20, -0.043)]   # exterior actuator lines
PUMP_X, PUMP_Y, PUMP_Z = 0.52, -9.75, -0.72


def build_B07():
    rid = "B07"
    node = bay_node(rid, "EngineBay")
    x0, x1, y0, y1, z0, z1 = -1.0, 1.0, -10.7, -9.0, -1.3, 0.5
    YT = -10.60                                               # thrust-frame plane

    def thrust_frame(bm):
        for (ex, ez) in ENG:                                  # engine mount rings
            add_revolve(bm, [(0.30, -0.04), (0.40, -0.04), (0.40, 0.04), (0.30, 0.04)], 32,
                        Tm((ex, YT, ez)) @ AY, closed=True, mi=0)
            for k in range(8):
                a = 2 * PI * k / 8
                add_cyl(bm, 0.012, 0.10, 6, Tm((ex + 0.35 * math.cos(a), YT, ez + 0.35 * math.sin(a))) @ AY, mi=1)
        bx(bm, -1.30, 1.30, YT - 0.04, YT + 0.04, 0.36, 0.44, 0, 0.008)                # top beam
        bx(bm, -0.98, 0.98, YT - 0.04, YT + 0.04, -1.24, -1.16, 0, 0.008)              # bottom beam
        bx(bm, -0.04, 0.04, YT - 0.04, YT + 0.04, -1.16, 0.36, 0, 0.006)               # centre post
        for sx in (-1, 1):
            cylp(bm, (sx * 0.04, YT, 0.36), (sx * 0.86, YT, -0.30), 0.03, 10, 0)       # diagonals to rings
            cylp(bm, (sx * 0.04, YT, -1.16), (sx * 0.88, YT, -0.72), 0.03, 10, 0)
            cylp(bm, (sx * 1.2, YT, -0.10), (sx * 1.2, YT, 0.36), 0.035, 10, 0)        # ring -> top beam
            cylp(bm, (sx * 0.98, YT, -1.16), (sx * 1.05, YT, -0.84), 0.03, 10, 0)
            # longerons: thrust path forward to the bay frame at Y -9.05
            for (xa, za, xb, zb) in ((0.88, -0.25, 0.92, 0.42), (0.88, -0.75, 0.92, -1.22)):
                cylp(bm, (sx * xa, YT + 0.02, za), (sx * xb, -9.06, zb), 0.028, 10, 0)
        frame_rect(bm, -9.05, -0.97, 0.97, z0 + 0.03, z1 - 0.03, 0.06, 0.012, 0.045, mi=0, mi_f=1)
        for sx in (-1, 1):                                                          # upper / lower longerons
            for z in (z0 + 0.06, z1 - 0.06):
                stringer_y(bm, sx * 0.94, z, YT + 0.04, -9.05, 0.05, 0.05)
        # gimbal blocks (cardan cross) on each ring centre, meeting the top of the exterior bell (Y -10.80)
        for (ex, ez) in ENG:
            for s in (-1, 1):
                cylp(bm, (ex + s * 0.30, YT, ez), (ex + s * 0.10, -10.70, ez), 0.03, 10, 2)   # spider arms
            bx(bm, ex - 0.07, ex + 0.07, -10.79, -10.65, ez - 0.07, ez + 0.07, 2, 0.01)
            cylp(bm, (ex - 0.12, -10.72, ez), (ex + 0.12, -10.72, ez), 0.025, 12, 1)
            cylp(bm, (ex, -10.72, ez - 0.12), (ex, -10.72, ez + 0.12), 0.025, 12, 1)
    tf = part(N(rid, "ThrustFrame"), ["equip", "metal", "dark"], thrust_frame, node, smooth=None)
    tf["kind"] = "thrust frame: engine mount rings, beams, longerons, gimbal blocks at the bell tops"

    def actuator(bm):                                         # local: rod end at origin, body forward (+Y)
        bx(bm, -0.07, 0.07, 0.40, 0.46, -0.07, 0.07, 1, 0.01)                         # yoke base plate
        for s in (-1, 1):
            bx(bm, s * 0.055 - 0.012, s * 0.055 + 0.012, 0.32, 0.42, -0.05, 0.05, 1, 0.006)   # clevis cheeks
        cylp(bm, (-0.08, 0.36, 0), (0.08, 0.36, 0), 0.015, 8, 2)                      # clevis pin
        add_revolve(bm, [(0.0, 0.33), (0.03, 0.33), (0.048, 0.31), (0.048, 0.04), (0.035, 0.02), (0.0, 0.02)], 16,
                    AY, mi=0)                                                           # body (along Y)
        cylp(bm, (0, 0.02, 0), (0, -0.20, 0), 0.02, 10, 2)                            # rod
        add_torus(bm, 0.03, 0.01, 12, 4, Tm((0, -0.22, 0)) @ AX, mi=2)                 # rod-end eye
        pipe(bm, [(0.048, 0.28, 0.0), (0.09, 0.28, 0.0), (0.09, 0.42, 0.05)], 0.008, 0.03, 6, 3)
        pipe(bm, [(0.048, 0.08, 0.0), (0.10, 0.08, 0.0), (0.10, 0.42, -0.05)], 0.008, 0.03, 6, 3)
    am = lib("B07GimbalActuator", ["equip", "metal", "dark", "rail"], actuator, 35)
    for k, (ax, az) in enumerate(ACTS, 1):
        o = inst(N(rid, f"GimbalActuator{k}"), am, node, (ax, -10.58, az))
        o["kind"] = "gimbal actuator + yoke, in line with the exterior actuator behind the aft bulkhead"
    # yoke support struts back to the thrust frame
    def yoke_struts(bm):
        for (ax, az) in ACTS:
            ex = -1.2 if ax < 0 else 1.2
            ez = -0.5
            d = Vector((ax - ex, 0, az - ez)).normalized()
            p0 = Vector((ax, -10.20, az))
            p1 = Vector((ex, YT + 0.04, ez)) + d * 0.36
            cylp(bm, p0, p1, 0.02, 8, 0)
    part(N(rid, "YokeStruts"), ["equip"], yoke_struts, node, smooth=None)

    def turbopump(bm):                                        # local: axis +Y forward, origin pump centre
        add_revolve(bm, [(0.0, 0.22), (0.07, 0.22), (0.13, 0.17), (0.16, 0.10), (0.16, -0.04), (0.12, -0.07),
                         (0.13, -0.10), (0.13, -0.28), (0.09, -0.32), (0.0, -0.33)], 24, AY, mi=0)
        add_torus(bm, 0.14, 0.055, 24, 10, Tm((0, 0.19, 0)) @ AY, mi=0)              # inlet volute
        add_tube(bm, [Vector((0.0, 0.19, -0.14)), Vector((0.0, 0.19, -0.30))], 0.045, 12, mi=0)   # inlet stub
        flange(bm, (0.0, 0.19, -0.30), (0, 0, 1), 0.07, 0.015, 1, 6)
        add_tube(bm, [Vector((0.0, 0.03, 0.15)), Vector((0.0, 0.03, 0.17))], 0.04, 12, mi=0)
        flange(bm, (0.0, 0.03, 0.17), (0, 0, 1), 0.065, 0.015, 1, 6)                  # discharge flange
        for z in (-0.10, 0.10):                                                       # casing ribs
            add_torus(bm, 0.162, 0.008, 24, 4, Tm((0, z * 0.5, 0)) @ AY, mi=1)
        add_tube(bm, [Vector((0.0, -0.33, 0.0)), Vector((0.0, -0.42, 0.0))], 0.05, 12, mi=1)  # turbine exhaust
        for s in (-1, 1):                                                             # mounting feet
            bx(bm, s * 0.10 - 0.03, s * 0.10 + 0.03, -0.20, 0.05, -0.24, -0.12, 1, 0.006)
        bx(bm, -0.06, 0.06, -0.16, -0.06, 0.12, 0.17, 2, 0.006)                       # sensor box
    tp = lib("B07Turbopump", ["equip", "metal", "dark"], turbopump, 35)
    for k, sx in enumerate((-1, 1), 1):
        o = inst(N(rid, f"Turbopump{k}"), tp, node, (sx * PUMP_X, PUMP_Y, PUMP_Z))
        o["kind"] = "turbopump with inlet volute"

    def plumbing(bm):
        zi = PUMP_Z - 0.30
        for sx in (-1, 1):
            px = sx * PUMP_X
            # feed line from B05 (through B06) up into the volute inlet
            pipe(bm, [(sx * FEED_X, -8.95, FEED_Z), (sx * FEED_X, PUMP_Y + 0.19, FEED_Z), (px, PUMP_Y + 0.19, FEED_Z),
                      (px, PUMP_Y + 0.19, zi - 0.01)], 0.035, 0.10, 12, 0)
            valve(bm, (sx * FEED_X, -9.35, FEED_Z), (0, 1, 0), 0.035, up=(0, 0, 1), mi_body=1, mi_wheel=3)
            # discharge: pump top -> aft, down/outboard, aft through the ring to the gimbal flex joint
            ex = sx * 1.2
            pipe(bm, [(px, PUMP_Y + 0.03, PUMP_Z + 0.17), (px, PUMP_Y + 0.03, -0.40), (px, -10.00, -0.40),
                      (ex, -10.12, -0.74), (ex, -10.84, -0.74)], 0.04, 0.10, 12, 0)
            for k in range(4):                                                          # flex bellows
                add_torus(bm, 0.046, 0.01, 16, 4, Tm((ex, -10.40 - k * 0.025, -0.74)) @ AY, mi=1)
        # regenerative heat exchanger fed by the primary coolant loop (from B06), turbine drive lines
        add_revolve(bm, [(0.0, 0.28), (0.10, 0.28), (0.15, 0.22), (0.15, -0.30), (0.10, -0.36), (0.0, -0.36)], 24,
                    Tm((0, -9.70, -1.0)) @ AY, mi=2)
        for y in (-9.55, -9.80, -10.0):
            add_torus(bm, 0.152, 0.008, 24, 4, Tm((0, y, -1.0)) @ AY, mi=1)
        pipe(bm, [(0.0, -8.95, -0.75), (0.0, -9.25, -0.75), (0.0, -9.35, -0.88), (0.0, -9.46, -0.88)], 0.05, 0.08, 14, 4)
        pipe(bm, [(0.0, -8.95, -1.10), (0.0, -9.46, -1.10)], 0.045, 0.06, 14, 4)
        for sx in (-1, 1):
            pipe(bm, [(sx * 0.10, -10.04, -1.0), (sx * 0.10, -10.14, -1.0), (sx * (PUMP_X - 0.05), -10.14, -1.0),
                      (sx * (PUMP_X - 0.05), -10.0, PUMP_Z - 0.10)], 0.03, 0.05, 10, 1)
            pipe(bm, [(sx * PUMP_X, PUMP_Y - 0.42, PUMP_Z), (sx * PUMP_X, -10.40, PUMP_Z - 0.25),
                      (sx * PUMP_X, -10.62, PUMP_Z - 0.25)], 0.045, 0.06, 12, 1)       # turbine exhaust to the shield
    pl = part(N(rid, "FeedLines"), ["metal", "rail", "equip", "dark", "red"], plumbing, node, smooth=35)
    pl["kind"] = "feed lines from B05 to the turbopump volutes, discharge lines to the gimbals, HX + coolant legs"

    def heat_shield(bm):                                       # quilted foil blanket on the aft bulkhead
        ya, yb = -10.69, -10.665
        pu, pv = 0.21, 0.195
        keep_out = [(ax, az, 0.10) for (ax, az) in ACTS] + [(sx * PUMP_X, PUMP_Z - 0.25, 0.08) for sx in (-1, 1)]
        nu, nv = int(1.94 / pu), int(1.74 / pv)
        for i in range(nu):
            for j in range(nv):
                a = -0.97 + i * 1.94 / nu + 0.005
                b_ = -0.97 + (i + 1) * 1.94 / nu - 0.005
                c = -1.27 + j * 1.74 / nv + 0.005
                d = -1.27 + (j + 1) * 1.74 / nv - 0.005
                if any(a - r < kx < b_ + r and c - r < kz < d + r for (kx, kz, r) in keep_out):
                    continue
                bx(bm, a, b_, ya, yb, c, d, 0)                                            # pad
                bx(bm, a + 0.025, b_ - 0.025, yb, yb + 0.008, c + 0.025, d - 0.025, 0)     # pillowed centre
        for (kx, kz, r) in keep_out:                                                   # collars round cut-outs
            add_torus(bm, r - 0.02, 0.012, 16, 4, Tm((kx, -10.675, kz)) @ AY, mi=1)
        for sx in (-1, 1):                                                             # side blankets (lower)
            for j in range(4):
                c = -1.27 + j * 0.30
                bx(bm, sx * 0.995 - 0.0125, sx * 0.995 + 0.0125, -10.66, -10.25, c + 0.005, c + 0.295, 0, 0.01)
    hs = part(N(rid, "HeatShield"), ["foil", "metal"], heat_shield, node, smooth=None)
    hs["kind"] = "quilted foil heat shielding on the aft bulkhead (cut-outs round actuators / exhaust)"

    def platform(bm):                                          # work platform under the C01 hatch (Y -9.30)
        zp = -0.15
        for (a, b_, c, d) in ((-0.45, 0.45, -9.60, -9.57), (-0.45, 0.45, -9.13, -9.10), (-0.45, -0.42, -9.60, -9.10),
                              (0.42, 0.45, -9.60, -9.10)):
            bx(bm, a, b_, c, d, zp - 0.05, zp, 0, 0.004)
        for sx in (-1, 1):
            for y in (-9.585, -9.115):
                cylp(bm, (sx * 0.435, y, zp), (sx * 0.435, y, z1 - 0.06), 0.016, 8, 0)   # hangers
            bx(bm, sx * 0.94 - 0.03, sx * 0.94 + 0.03, -9.62, -9.08, z1 - 0.08, z1 - 0.02, 0, 0.004)
        for y in (-9.585, -9.115):
            bx(bm, -0.97, 0.97, y - 0.02, y + 0.02, z1 - 0.08, z1 - 0.02, 0, 0.004)    # hanger beams
        for sx in (-1, 1):                                                              # side handrails
            cylp(bm, (sx * 0.45, -9.58, zp + 0.48), (sx * 0.45, -9.12, zp + 0.48), 0.015, 8, 1)
        for x in (-0.19, 0.19):                                                         # ladder to C01
            bx(bm, x - 0.018, x + 0.018, -9.56, -9.52, zp, 0.595, 1, 0.005)
        for i in range(3):
            cylp(bm, (-0.19, -9.54, zp + 0.18 + i * 0.22), (0.19, -9.54, zp + 0.18 + i * 0.22), 0.014, 8, 1)
    pf = part(N(rid, "WorkPlatform"), ["metal", "rail"], platform, node, smooth=None)
    pf["kind"] = "work platform + ladder under the C01 floor hatch (centre Y -9.30, hinge Y -9.05)"
    gm = I.grate_tile(0.42, 0.44)
    for k, x in enumerate((-0.21, 0.21), 1):
        inst(N(rid, f"PlatformGrate{k}"), gm, node, (x, -9.35, -0.15))
    light_at(rid, node, dx=0.62)
    return node


BUILDERS["B07"] = build_B07


# =================================================================== driver / checks
def run(bays, out):
    for rid in BAY_ORDER:
        if rid not in bays or rid not in BUILDERS:
            continue
        wipe_bay(rid)
        BUILDERS[rid]()
        bpy.context.view_layer.update()
        C.save_blend(out)
        print(f"KESTREL_BAY {rid} done; saved {out}", flush=True)


def bay_objects(rid):
    return [o for o in I.INT.all_objects if o.name.split("_")[1:2] == [rid]]


def envelope():
    """(distance tree of the outer hull envelope, inside test of the union)."""
    _dist, inside = I.skin_volume_trees()
    skins = [o for o in bpy.data.collections[C.COL_EXT].all_objects if o.get("kestrel_skin")]
    closed = [(o, E.tree_for(o)) for o in skins if "Fuselage" not in o.name]

    def inside_other(p, me):
        if "Fuselage" not in me.name and E.inside_fuselage(p.x, p.z, p.y) and \
                B.FUSELAGE[-1][0] <= p.y <= B.FUSELAGE[0][0]:
            return True
        for o, tr in closed:
            if o is me:
                continue
            hit = tr.find_nearest(p)
            if hit[0] is not None and (p - hit[0]).dot(hit[1]) < -0.01:
                return True
        return False
    verts, polys = [], []
    for o in skins:
        mw = o.matrix_world
        base = len(verts)
        verts += [mw @ v.co for v in o.data.vertices]
        for f in o.data.polygons:
            c = mw @ f.center
            nrm = (mw.to_3x3() @ f.normal).normalized()
            if not inside_other(c - nrm * 0.0, o):
                polys.append(tuple(base + i for i in f.vertices))
    return BVHTree.FromPolygons(verts, polys), inside


def clearance(objs, env):
    tree, inside = env
    worst, per = 1e9, {}
    for o in objs:
        if o.type != 'MESH':
            continue
        mw = o.matrix_world
        m = 1e9
        for v in o.data.vertices:
            p = mw @ v.co
            d = tree.find_nearest(p)[3]
            if d < m:
                m = min(m, d if inside(p) else -d)
            elif not inside(p):
                m = min(m, -d)
        per[o.name] = round(m, 4)
        worst = min(worst, m)
    return worst, per


def room_hits(objs):
    out = []
    for o in objs:
        if o.type != 'MESH':
            continue
        mw = o.matrix_world
        for rid, (x0, x1, y0, y1, z0, z1) in ROOM_BOXES.items():
            n = 0
            for v in o.data.vertices:
                p = mw @ v.co
                s = I.SH
                if x0 - s < p.x < x1 + s and y0 - s < p.y < y1 + s and z0 - s < p.z < z1 + s:
                    n += 1
            if n:
                out.append([o.name, rid, n])
    return out


def report():
    bpy.context.view_layer.update()
    env = envelope()
    rep = {"bays": {}}
    total, biggest = 0, (0, "")
    for rid in BAY_ORDER:
        objs = bay_objects(rid)
        meshes = [o for o in objs if o.type == 'MESH']
        t = C.triangle_count(meshes)
        total += t
        for o in meshes:
            tt = C.triangle_count([o])
            if tt > biggest[0]:
                biggest = (tt, o.name)
        worst, per = clearance(meshes, env)
        rep["bays"][rid] = {
            "tris": t, "objects": len(objs), "mesh_objects": len(meshes),
            "linked_duplicates": sum(1 for o in meshes if o.data.users > 1),
            "min_envelope_clearance_m": round(worst, 4),
            "below_0.05": {k: v for k, v in per.items() if v < 0.05},
            "room_shell_hits": room_hits(meshes),
        }
    rep["bays_tris"] = total
    rep["max_single_bay_mesh"] = {"name": biggest[1], "tris": biggest[0]}
    allint = [o for o in I.INT.all_objects if o.type == 'MESH']
    rep["interior_plus_bays_tris"] = C.triangle_count(allint)
    big = max(((C.triangle_count([o]), o.name) for o in bpy.context.scene.objects if o.type == 'MESH'))
    rep["max_single_mesh_scene"] = {"name": big[1], "tris": big[0]}
    rep["exterior_tris"] = C.triangle_count([o for o in bpy.data.collections[C.COL_EXT].all_objects
                                             if o.type == 'MESH'])
    rep["triangles_total_scene"] = C.triangle_count()
    rep["objects_total"] = len(bpy.context.scene.objects)
    rep["interior_objects"] = len(I.INT.all_objects)
    return rep


if __name__ == "__main__":
    args = C.script_args()
    src = args.get("blend", C.BLEND_PATH)
    out = args.get("out", src)
    bays = args.get("bays")
    bays = set(bays.split(",")) if isinstance(bays, str) else set(BAY_ORDER)
    C.open_blend(src)
    I.setup()
    run(bays, out)
    if not args.get("noreport"):
        rep = report()
        if args.get("report"):
            with open(args["report"], "w") as f:
                json.dump(rep, f, indent=1)
        print("KESTREL_BAY_REPORT " + json.dumps(rep))
