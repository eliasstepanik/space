"""IC-4 "Kestrel" — 40 final materials, UVs, procedural textures and decals.

Runs ON TOP of 31_bays.py output (re-runnable on its own output):

    blender -b --factory-startup --python 40_materials.py -- [--blend in.blend] [--out out.blend]
                                                           [--report report.json]

Defaults: --blend/--out ../kestrel.blend.  Geometry is never touched: no object is
moved/scaled, no vertex/face is added or removed.  The script only
  * replaces every MAT_Placeholder_* slot by one of the 16 final M_* materials
    (materials follow the mesh, so linked duplicates stay linked),
  * re-assigns face material indices on a few meshes (fuselage belly -> grey paint,
    decal faces -> M_Decals),
  * writes one UV layer "UVMap" per mesh (numpy cube projection, 1 UV = 2 m for the
    tileable sets, 1 UV = 0.5 m for fabric; decal / screen faces are planar-projected
    into their atlas cells),
  * generates 6 textures procedurally with numpy (tileable FFT noise, stamped panel
    features, a built-in 5x7 bitmap font), saves them as JPEG/PNG and packs them.

Materials read correctly from their factors alone (base colour, metallic,
roughness, emission); textures only add detail (ORM channels are ~1.0 on average and
are multiplied by the factors, normal maps are detail).
"""
import importlib
import json
import math
import os
import sys
import zlib

sys.dont_write_bytecode = True

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
C = importlib.import_module("00_common")
B = importlib.import_module("10_blockout")

TEX_DIR = "/tmp/kestrel_tex"          # scratch; images are packed into the .blend
UV_NAME = "UVMap"
S_TILE = 0.5                          # UV units per metre (1 UV = 2 m)
S_FABRIC = 2.0                        # 1 UV = 0.5 m

# =================================================================== final material set
#   name: base (linear), metallic, roughness, texture set, normal strength, extras
MATS = {
    "M_HullPaint_White":   dict(base=(0.66, 0.67, 0.68), metal=0.0, rough=0.42, tex="hull", nstr=1.0),
    "M_HullPaint_Grey":    dict(base=(0.25, 0.26, 0.28), metal=0.0, rough=0.48, tex="hull", nstr=1.0),
    "M_HullBare_Metal":    dict(base=(0.56, 0.57, 0.58), metal=1.0, rough=0.32, tex="hull", nstr=0.6),
    "M_Thermal_Dark":      dict(base=(0.032, 0.032, 0.035), metal=0.0, rough=0.62, tex="hull", nstr=0.5),
    "M_Radiator":          dict(base=(0.78, 0.77, 0.73), metal=0.0, rough=0.30, tex="hull", nstr=0.4),
    "M_MLI_Foil":          dict(base=(0.86, 0.62, 0.24), metal=1.0, rough=0.26, tex="int", nstr=1.6),
    "M_Glass_Canopy":      dict(base=(0.05, 0.09, 0.12), metal=0.0, rough=0.04, alpha=0.28),
    "M_Decals":            dict(base=(1.0, 1.0, 1.0), metal=0.0, rough=0.50, decal=True),
    "M_Rubber_Dark":       dict(base=(0.04, 0.04, 0.042), metal=0.0, rough=0.80),
    "M_Paint_SafetyYellow": dict(base=(0.86, 0.56, 0.02), metal=0.0, rough=0.50, tex="hull", nstr=0.5),
    "M_Int_Panel":         dict(base=(0.62, 0.63, 0.61), metal=0.0, rough=0.55, tex="int", nstr=1.0),
    "M_Int_Floor":         dict(base=(0.15, 0.155, 0.16), metal=0.0, rough=0.75, tex="int", nstr=1.3),
    "M_Int_Fabric":        dict(base=(0.19, 0.26, 0.35), metal=0.0, rough=0.90, tex="int", nstr=0.9),
    "M_Int_Equipment":     dict(base=(0.30, 0.32, 0.34), metal=0.0, rough=0.45, tex="int", nstr=0.6),
    "M_Screens_Emissive":  dict(base=(0.02, 0.03, 0.035), metal=0.0, rough=0.15,
                                emit=(0.55, 0.85, 1.0), estr=3.0, screen=True),
    "M_Lights_Emissive":   dict(base=(0.95, 0.93, 0.88), metal=0.0, rough=0.50,
                                emit=(1.0, 0.92, 0.80), estr=12.0),
}

GREY_HULL_MESHES = {"EXT_HULL_Sponson_L", "EXT_HULL_Sponson_R", "EXT_HULL_AccessHatch",
                    "EXT_HULL_AccessPanel", "EXT_WEAPONS_BayDoor", "EXT_DOCK_Hatch"}
RUBBER_EXT_MESHES = {"EXT_GEAR_MainHydLines", "EXT_GEAR_NoseHydLines", "EXT_GEAR_NoseWheel"}


def final_for(mesh_name, ph):
    """Placeholder material name -> final material name (context = mesh name)."""
    if ph.startswith("M_"):
        return ph
    key = ph.replace("MAT_Placeholder_", "")
    ext = mesh_name.startswith("EXT_")
    if key == "Hull":
        return "M_HullPaint_Grey" if mesh_name in GREY_HULL_MESHES else "M_HullPaint_White"
    if key == "Dark":
        if not ext or mesh_name in RUBBER_EXT_MESHES:
            return "M_Rubber_Dark"
        return "M_Thermal_Dark"
    return {
        "Metal": "M_HullBare_Metal", "Canopy": "M_Glass_Canopy", "Radiator": "M_Radiator",
        "Foil": "M_MLI_Foil", "Handrail": "M_Paint_SafetyYellow", "Scorch": "M_Thermal_Dark",
        "Airlock": "M_Paint_SafetyYellow", "IntPanel": "M_Int_Panel", "Floor": "M_Int_Floor",
        "Fabric": "M_Int_Fabric", "Equipment": "M_Int_Equipment", "Screen": "M_Screens_Emissive",
        "IntLight": "M_Lights_Emissive",
        # blockout-only placeholders (not used after 31_bays, mapped for completeness)
        "Room": "M_Int_Panel", "Bay": "M_Int_Equipment", "Crawl": "M_Int_Panel",
    }[key]


# =================================================================== numpy helpers
def srgb(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def fft_noise(n, beta, lo, hi, seed):
    """Tileable (periodic) band-limited 1/f^beta noise, zero mean, unit std."""
    rng = np.random.default_rng(seed)
    F = np.fft.rfft2(rng.standard_normal((n, n)))
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.rfftfreq(n)[None, :] * n
    f = np.sqrt(fx * fx + fy * fy)
    amp = np.where((f >= lo) & (f <= hi), 1.0 / np.maximum(f, 1.0) ** beta, 0.0)
    out = np.fft.irfft2(F * amp, s=(n, n))
    out -= out.mean()
    return out / (out.std() + 1e-9)


def blur_wrap(a, r):
    """Periodic box blur (3 passes ~ gaussian)."""
    if r < 1:
        return a
    for _ in range(3):
        for ax in (0, 1):
            acc = np.zeros_like(a)
            for k in range(-r, r + 1):
                acc += np.roll(a, k, axis=ax)
            a = acc / (2 * r + 1)
    return a


def blur_clamp(a, r):
    if r < 1:
        return a
    for _ in range(2):
        for ax in (0, 1):
            pad = [(0, 0)] * a.ndim
            pad[ax] = (r + 1, r)
            p = np.pad(a, pad, mode="edge")
            cs = np.cumsum(p, axis=ax)
            hi = np.take(cs, np.arange(2 * r + 1, cs.shape[ax]), axis=ax)
            lo = np.take(cs, np.arange(0, cs.shape[ax] - 2 * r - 1), axis=ax)
            a = (hi - lo) / (2 * r + 1)
    return a


def height_to_normal(h, k):
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5 * k
    dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5 * k
    nz = np.ones_like(h)
    ln = np.sqrt(dx * dx + dy * dy + 1.0)
    return np.stack([-dx / ln, -dy / ln, nz / ln], axis=-1) * 0.5 + 0.5


def stamp_wrap(dst, y0, x0, patch, mode="add"):
    n = dst.shape[0]
    ys = (np.arange(patch.shape[0]) + y0) % n
    xs = (np.arange(patch.shape[1]) + x0) % n
    sub = dst[np.ix_(ys, xs)]
    if mode == "add":
        sub = sub + patch
    elif mode == "min":
        sub = np.minimum(sub, patch)
    elif mode == "max":
        sub = np.maximum(sub, patch)
    dst[np.ix_(ys, xs)] = sub


def disc(r, soft=1.0):
    R = int(math.ceil(r + 2))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    d = np.sqrt(xx * xx + yy * yy)
    return np.clip((r - d) / soft + 0.5, 0, 1), d


# =================================================================== bitmap font (5x7)
FONT_SRC = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#", "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.", "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####", "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.####", "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.", "J": "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", "N": "#...#|#...#|##..#|#.#.#|#..##|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.", "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.", "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|#.#.#|.#.#.", "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "0": ".###.|#...#|#..##|#.#.#|##..#|#...#|.###.", "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####", "3": "#####|...#.|..#..|...#.|....#|#...#|.###.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.", "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.", "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.", "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    "-": ".....|.....|.....|#####|.....|.....|.....", ".": ".....|.....|.....|.....|.....|.##..|.##..",
    "/": "....#|....#|...#.|..#..|.#...|#....|#....", ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "%": "##...|##..#|...#.|..#..|.#...|#..##|...##", "+": ".....|..#..|..#..|#####|..#..|..#..|.....",
    "=": ".....|.....|#####|.....|#####|.....|.....", ">": ".#...|..#..|...#.|....#|...#.|..#..|.#...",
    "<": "...#.|..#..|.#...|#....|.#...|..#..|...#.", "^": "..#..|..#..|.#.#.|.#.#.|#...#|#...#|#####",
    " ": ".....|.....|.....|.....|.....|.....|.....",
}
FONT = {k: np.array([[c == "#" for c in row] for row in v.split("|")], dtype=bool) for k, v in FONT_SRC.items()}


def text_bitmap(s):
    s = s.upper()
    cols = []
    for ch in s:
        g = FONT.get(ch, FONT[" "])
        cols.append(g)
        cols.append(np.zeros((7, 1), bool))
    return np.concatenate(cols[:-1], axis=1) if cols else np.zeros((7, 1), bool)


def text_width(s, height):
    return (len(s) * 6 - 1) * height / 7.0


# =================================================================== supersampled canvas
class Canvas:
    """RGBA canvas, origin top-left, y down, coordinates in output pixels; colours are
    display (sRGB) values.  Drawn at ss x supersampling, box-filtered on finish()."""

    def __init__(self, w, h, bg=(0, 0, 0, 1), ss=3):
        self.w, self.h, self.ss = w, h, ss
        self.a = np.empty((h * ss, w * ss, 4), np.float32)
        self.a[:] = bg

    def _win(self, x0, y0, x1, y1):
        s = self.ss
        X0, Y0 = max(int(math.floor(x0 * s)), 0), max(int(math.floor(y0 * s)), 0)
        X1, Y1 = min(int(math.ceil(x1 * s)), self.w * s), min(int(math.ceil(y1 * s)), self.h * s)
        if X1 <= X0 or Y1 <= Y0:
            return None
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        return (Y0, Y1, X0, X1), (xx + 0.5) / s, (yy + 0.5) / s

    def _put(self, win, mask, col):
        (Y0, Y1, X0, X1) = win
        col = np.array(col if len(col) == 4 else (*col, 1.0), np.float32)
        m = mask[..., None].astype(np.float32) * col[3]
        sub = self.a[Y0:Y1, X0:X1]
        sub[..., :3] = sub[..., :3] * (1 - m) + col[:3] * m
        sub[..., 3:] = sub[..., 3:] * (1 - m) + m
        self.a[Y0:Y1, X0:X1] = sub

    def rect(self, x0, y0, x1, y1, col):
        r = self._win(x0, y0, x1, y1)
        if r:
            win, X, Y = r
            self._put(win, (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1), col)

    def frame(self, x0, y0, x1, y1, t, col):
        self.rect(x0, y0, x1, y0 + t, col)
        self.rect(x0, y1 - t, x1, y1, col)
        self.rect(x0, y0, x0 + t, y1, col)
        self.rect(x1 - t, y0, x1, y1, col)

    def circle(self, cx, cy, r, col, width=None, a0=None, a1=None):
        r_out = r + (width or 0) / 2 if width else r
        res = self._win(cx - r_out - 1, cy - r_out - 1, cx + r_out + 1, cy + r_out + 1)
        if not res:
            return
        win, X, Y = res
        d = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        m = (np.abs(d - r) <= width / 2) if width else (d <= r)
        if a0 is not None:
            ang = np.degrees(np.arctan2(-(Y - cy), X - cx)) % 360
            m &= ((ang - a0) % 360) <= ((a1 - a0) % 360)
        self._put(win, m, col)

    def line(self, x0, y0, x1, y1, w, col):
        res = self._win(min(x0, x1) - w, min(y0, y1) - w, max(x0, x1) + w, max(y0, y1) + w)
        if not res:
            return
        win, X, Y = res
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy + 1e-9
        t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
        d = np.sqrt((X - x0 - t * dx) ** 2 + (Y - y0 - t * dy) ** 2)
        self._put(win, d <= w / 2, col)

    def poly(self, pts, col):
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        res = self._win(min(xs), min(ys), max(xs), max(ys))
        if not res:
            return
        win, X, Y = res
        inside = np.zeros(X.shape, bool)
        n = len(pts)
        for i in range(n):                       # even-odd ray casting
            (xa, ya), (xb, yb) = pts[i], pts[(i + 1) % n]
            cond = ((ya > Y) != (yb > Y))
            xint = xa + (Y - ya) * (xb - xa) / ((yb - ya) if yb != ya else 1e-9)
            inside ^= cond & (X < xint)
        self._put(win, inside, col)

    def stripes(self, x0, y0, x1, y1, period, col, phase=0.0):
        res = self._win(x0, y0, x1, y1)
        if res:
            win, X, Y = res
            m = (((X + Y) / period + phase) % 1.0) < 0.5
            self._put(win, m & (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1), col)

    def text(self, s, x, y, height, col, align="left", bold=0.12):
        """y = top of the caps; height = cap height.  Rounded-pixel stencil glyphs."""
        bm = text_bitmap(s)
        cell = height / 7.0
        wpx = bm.shape[1] * cell
        if align == "center":
            x -= wpx / 2
        elif align == "right":
            x -= wpx
        res = self._win(x - cell, y - cell, x + wpx + cell, y + height + cell)
        if not res:
            return
        win, X, Y = res
        gx = np.floor((X - x) / cell).astype(int)
        gy = np.floor((Y - y) / cell).astype(int)
        ok = (gx >= 0) & (gx < bm.shape[1]) & (gy >= 0) & (gy < 7)
        m = np.zeros(X.shape, np.float32)
        m[ok] = bm[gy[ok], gx[ok]]
        r = max(1, int(round(cell * self.ss * 0.30)))
        m = blur_clamp(m, r)
        self._put(win, m > (0.5 - bold), col)

    def finish(self):
        s = self.ss
        a = self.a.reshape(self.h, s, self.w, s, 4).mean(axis=(1, 3))
        return a                                   # top-row-first


# =================================================================== tileable textures
def hull_textures(n=1024, seed=7):
    """Tile = 2 m.  Height: oil-canning, orange peel, small access plates with
    fasteners, rivet rows, scratches.  ORM: groove AO, roughness wear/streaks, metal ~1."""
    rng = np.random.default_rng(seed)
    h = 0.55 * fft_noise(n, 2.0, 2, 10, seed + 1)            # oil canning
    h += 0.035 * fft_noise(n, 0.3, 160, 420, seed + 2)        # orange peel
    groove = np.zeros((n, n), np.float32)
    raised = np.zeros((n, n), np.float32)
    # small access plates (outline groove + slightly raised plate + corner fasteners)
    for _ in range(16):
        w, hh = int(rng.integers(40, 170)), int(rng.integers(40, 130))
        x0, y0 = int(rng.integers(0, n)), int(rng.integers(0, n))
        pw = np.zeros((hh + 8, w + 8), np.float32)
        pw[4:-4, 4:-4] = 1.0
        inner = np.zeros_like(pw)
        inner[6:-6, 6:-6] = 1.0
        outline = np.clip(pw - inner, 0, 1)
        stamp_wrap(groove, y0, x0, outline, "max")
        stamp_wrap(raised, y0, x0, inner * 0.18, "max")
        dk, _d = disc(3.2)
        for (cy, cx) in ((12, 12), (12, w - 4), (hh - 4, 12), (hh - 4, w - 4)):
            stamp_wrap(raised, y0 + cy - dk.shape[0] // 2, x0 + cx - dk.shape[1] // 2, dk * 0.9, "max")
    # rivet rows (stiffeners behind the skin): 4 cm pitch
    dk, _d = disc(2.4)
    for _ in range(5):
        horiz = bool(rng.integers(0, 2))
        c0 = int(rng.integers(0, n))
        a0 = int(rng.integers(0, n))
        length = int(rng.integers(n // 3, n))
        for k in range(0, length, 20):
            yy, xx = (c0, a0 + k) if horiz else (a0 + k, c0)
            stamp_wrap(raised, yy - dk.shape[0] // 2, xx - dk.shape[1] // 2, dk * 0.6, "max")
    # scratches
    scratch = np.zeros((n, n), np.float32)
    for _ in range(70):
        L = int(rng.integers(30, 220))
        ang = rng.uniform(0, math.pi)
        x0, y0 = rng.uniform(0, n), rng.uniform(0, n)
        t = np.linspace(0, L, L * 2)
        xs = (x0 + np.cos(ang) * t).astype(int) % n
        ys = (y0 + np.sin(ang) * t).astype(int) % n
        scratch[ys, xs] = 1.0
    h = h - 1.0 * groove + raised - 0.25 * scratch
    nrm = height_to_normal(blur_wrap(h, 1), 2.2)

    gsoft = blur_wrap(groove, 3)
    ao = np.clip(1.0 - 0.55 * gsoft - 0.06 * np.clip(-fft_noise(n, 2.0, 2, 10, seed + 1), 0, 3), 0.55, 1.0)
    streak = fft_noise(n, 1.2, 1, 64, seed + 5)
    streak = blur_wrap(np.roll(streak, 0, 0), 1)
    # vertical-ish streaking: average along one axis for anisotropy
    vstreak = np.cumsum(fft_noise(n, 0.8, 8, 200, seed + 6), axis=0)
    vstreak = (vstreak - vstreak.mean()) / (vstreak.std() + 1e-9)
    rough = 0.88 + 0.05 * streak + 0.03 * vstreak - 0.25 * blur_wrap(scratch, 1) + 0.06 * gsoft
    rough = np.clip(rough, 0.55, 1.0)
    metal = np.clip(1.0 - 0.12 * (fft_noise(n, 0.0, 200, 512, seed + 8) > 2.2), 0.8, 1.0)
    orm = np.stack([ao, rough, metal], axis=-1)
    return nrm, orm


def int_textures(n=1024, seed=21):
    """Tile = 2 m (512 px/m).  1.0 x 0.5 m panels: bevelled seams, corner fasteners,
    one perforated vent/grating panel per tile, fine weave micro relief."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    pw, ph = 512, 256
    dxs = np.minimum(xx % pw, pw - (xx % pw))
    dys = np.minimum(yy % ph, ph - (yy % ph))
    dseam = np.minimum(dxs, dys)
    seam = np.clip((3.5 - dseam) / 2.0, 0, 1)                  # 7 px (~14 mm) bevelled groove
    h = -1.2 * seam
    h += 0.04 * np.sin(2 * math.pi * xx / 8) * np.sin(2 * math.pi * yy / 8)   # weave hint
    h += 0.025 * np.sin(2 * math.pi * (xx + yy) / 16)
    h += 0.06 * fft_noise(n, 0.5, 64, 300, seed + 1)
    h += 0.20 * fft_noise(n, 2.0, 2, 8, seed + 2)
    # per-panel roughness offset
    pid = (xx // pw).astype(int) * 7 + (yy // ph).astype(int) * 3
    panel_r = (np.sin(pid * 12.9898) * 43758.5453) % 1.0
    # fasteners near panel corners and mid edges
    dk, d = disc(5.0)
    hexd = np.clip(1.0 - d / 2.5, 0, 1)
    fast = np.zeros((n, n), np.float32)
    for px in range(0, n, pw):
        for py in range(0, n, ph):
            for (ox, oy) in ((16, 16), (pw - 16, 16), (16, ph - 16), (pw - 16, ph - 16), (pw // 2, 16), (pw // 2, ph - 16)):
                stamp_wrap(fast, py + oy - dk.shape[0] // 2, px + ox - dk.shape[1] // 2, dk * 0.7 - hexd * 0.5 * dk, "max")
    h += fast
    # perforated vent / grating panel (one per tile)
    vx0, vy0 = 512 + 64, 256 * 2 + 48
    vent = np.zeros((n, n), np.float32)
    for r_ in range(8):
        for c_ in range(6):
            x0 = vx0 + c_ * 64
            y0 = vy0 + r_ * 20
            vent[y0:y0 + 8, x0:x0 + 52] = 1.0
    vent = blur_wrap(vent, 1)
    h -= 1.0 * vent
    nrm = height_to_normal(blur_wrap(h, 1), 1.8)
    ao = np.clip(1.0 - 0.5 * blur_wrap(seam, 4) - 0.45 * blur_wrap(vent, 2) - 0.2 * blur_wrap((fast > 0.1).astype(np.float32), 2), 0.45, 1.0)
    rough = np.clip(0.90 + 0.06 * (panel_r - 0.5) + 0.04 * fft_noise(n, 1.0, 4, 128, seed + 3) + 0.05 * seam, 0.6, 1.0)
    metal = np.ones((n, n), np.float32)
    orm = np.stack([ao, rough, metal], axis=-1)
    return nrm, orm


# =================================================================== screen atlas
SCREEN_CELLS = ["ADI", "RADAR", "ENG", "SYS", "NAV", "REACTOR", "COMMS", "KEYS", "GLOW"]
SCR_N = 1024
SCR_GRID = 3
SCR_CELL = SCR_N // SCR_GRID          # 341


def screen_cell_rect(name):
    i = SCREEN_CELLS.index(name)
    c, r = i % SCR_GRID, i // SCR_GRID      # r counted from top
    x0 = c * SCR_CELL
    y0_top = r * SCR_CELL
    return x0, y0_top, SCR_CELL, SCR_CELL


def draw_screen(name, s):
    cv = Canvas(s, s, bg=(0.015, 0.025, 0.03, 1))
    G, Gd, W, A, Cy = (0.35, 1.0, 0.55), (0.10, 0.35, 0.18), (0.95, 0.98, 1.0), (1.0, 0.75, 0.25), (0.35, 0.85, 1.0)
    m = 10
    if name not in ("GLOW", "KEYS"):
        cv.frame(4, 4, s - 4, s - 4, 2, (0.2, 0.45, 0.5))
    if name == "ADI":
        cx, cy, R = s / 2, s / 2 + 8, s * 0.36
        cv.circle(cx, cy, R, (0.08, 0.22, 0.42))                       # sky
        cv.poly([(cx - R, cy + 12), (cx + R, cy - 18), (cx + R, cy + R), (cx - R, cy + R)], (0.30, 0.18, 0.08))
        # mask ground outside the ball: redraw ring of background
        cv.circle(cx, cy, R + 40, (0.015, 0.025, 0.03), width=80)
        cv.line(cx - R, cy + 12, cx + R, cy - 18, 3, W)                 # horizon (banked)
        for k in (-2, -1, 1, 2):
            o = k * 26
            cv.line(cx - 30, cy - 3 - o, cx + 30, cy - 3 - o - 4.5, 2, W)
            cv.text(f"{abs(k) * 10}", cx + 40, cy - 9 - o, 10, W)
        cv.circle(cx, cy, R, W, width=3)
        cv.line(cx - 70, cy, cx - 25, cy, 5, A)
        cv.line(cx + 25, cy, cx + 70, cy, 5, A)
        cv.circle(cx, cy, 5, A)
        cv.rect(m, m, s - m, 44, (0.02, 0.05, 0.06))
        for i, hd in enumerate(("33", "N", "3", "6", "E")):
            cv.text(hd, 40 + i * 65, 18, 16, W, align="center")
        cv.line(s / 2, 40, s / 2, 54, 3, A)
        cv.text("SPD 212", m + 6, s - 34, 14, G)
        cv.text("ALT 18.4K", s - m - 6, s - 34, 14, G, align="right")
    elif name == "RADAR":
        cx, cy, R = s / 2, s / 2 + 6, s * 0.40
        cv.circle(cx, cy, R, (0.03, 0.12, 0.06))
        cv.circle(cx, cy, R, G, a0=60, a1=95)                         # sweep wedge
        cv.circle(cx, cy, R, (0.15, 0.55, 0.30), a0=35, a1=60)
        for f in (0.33, 0.66, 1.0):
            cv.circle(cx, cy, R * f, G, width=2)
        cv.line(cx - R, cy, cx + R, cy, 1.5, Gd)
        cv.line(cx, cy - R, cx, cy + R, 1.5, Gd)
        for (bx, by, col) in ((0.3, -0.5, A), (-0.45, 0.2, W), (0.55, 0.35, A), (-0.2, -0.7, W)):
            cv.circle(cx + bx * R, cy + by * R, 6, col)
        cv.text("RNG 40", m + 6, m + 8, 14, G)
        cv.text("TRK 4", s - m - 6, m + 8, 14, A, align="right")
    elif name == "ENG":
        cv.text("ENGINES", s / 2, 16, 18, W, align="center")
        for i, (lab, val) in enumerate((("L", 0.87), ("R", 0.84))):
            x0 = 70 + i * 140
            cv.frame(x0, 60, x0 + 60, 260, 3, G)
            cv.rect(x0 + 6, 260 - 194 * val, x0 + 54, 254, G)
            cv.text(lab, x0 + 30, 272, 16, W, align="center")
            cv.text(f"{int(val * 100)}%", x0 + 30, 298, 14, A, align="center")
        cv.line(m, 50, s - m, 50, 2, Gd)
    elif name == "SYS":
        cv.text("SYSTEMS", s / 2, 16, 18, W, align="center")
        rows = (("PWR", "OK", G), ("ECLSS", "OK", G), ("RCS", "OK", G), ("COMMS", "OK", G),
                ("HYD", "LOW", A), ("REACT", "62%", Cy), ("CABIN", "101K", Cy))
        for i, (a, b, col) in enumerate(rows):
            y = 56 + i * 38
            cv.text(a, 24, y, 16, W)
            cv.rect(s - 120, y - 6, s - 24, y + 22, (0.04, 0.10, 0.08))
            cv.frame(s - 120, y - 6, s - 24, y + 22, 2, col)
            cv.text(b, s - 72, y, 14, col, align="center")
    elif name == "NAV":
        for k in range(1, 8):
            cv.line(k * s / 8, m, k * s / 8, s - m, 1, (0.05, 0.18, 0.12))
            cv.line(m, k * s / 8, s - m, k * s / 8, 1, (0.05, 0.18, 0.12))
        pts = [(40, s - 50), (110, s - 120), (170, s - 150), (230, 140), (290, 70)]
        for a, b in zip(pts[:-1], pts[1:]):
            cv.line(*a, *b, 3, Cy)
        for p in pts[1:]:
            cv.circle(*p, 6, W, width=2)
        cv.poly([(40, s - 70), (28, s - 38), (52, s - 38)], A)
        cv.text("WP3 12.4KM", m + 8, m + 8, 14, W)
        cv.text("ETA 04:12", s - m - 8, s - 30, 14, G, align="right")
    elif name == "REACTOR":
        cx, cy, R = s / 2, s / 2 - 10, s * 0.28
        cv.circle(cx, cy, R, Cy, width=4)
        cv.circle(cx, cy, R * 0.7, (0.05, 0.25, 0.30))
        for k in range(12):
            a = k * math.pi / 6
            cv.circle(cx + math.cos(a) * R * 0.45, cy + math.sin(a) * R * 0.45, 9, G if k % 4 else A)
        cv.circle(cx, cy, 14, W)
        cv.text("CORE 612K", s / 2, s - 70, 16, W, align="center")
        cv.rect(30, s - 40, s - 30, s - 24, (0.05, 0.15, 0.10))
        cv.rect(30, s - 40, 30 + (s - 60) * 0.62, s - 24, G)
        cv.text("REACTOR", s / 2, 14, 16, Cy, align="center")
    elif name == "COMMS":
        cv.text("COMMS LOG", 18, 16, 16, W)
        rng = np.random.default_rng(3)
        for i in range(9):
            y = 52 + i * 30
            cv.text(f"{i + 10:02d}:4{i}", 18, y, 12, G)
            ln = int(rng.integers(3, 8))
            for j in range(ln):
                cv.rect(110 + j * 30, y + 2, 110 + j * 30 + int(rng.integers(10, 26)), y + 10, (0.5, 0.8, 0.9))
    elif name == "KEYS":
        for r_ in range(6):
            for c_ in range(6):
                x0, y0 = 14 + c_ * 54, 14 + r_ * 54
                cv.rect(x0, y0, x0 + 44, y0 + 44, (0.25, 0.55, 0.65))
                cv.rect(x0 + 10, y0 + 18, x0 + 34, y0 + 26, (0.9, 0.98, 1.0))
    elif name == "GLOW":
        cv.rect(0, 0, s, s, (0.55, 0.80, 0.90))
    if name not in ("GLOW", "KEYS"):           # faint scanlines
        sl = cv.finish()
        sl[::3] *= 0.85
        return sl
    return cv.finish()


def screen_atlas():
    atlas = np.zeros((SCR_N, SCR_N, 3), np.float32)
    atlas[:] = 0.015
    for nm in SCREEN_CELLS:
        x0, y0t, w, h = screen_cell_rect(nm)
        cell = draw_screen(nm, w)[..., :3]
        atlas[y0t:y0t + h, x0:x0 + w] = cell
    return np.flipud(atlas)        # -> row 0 = bottom (Blender pixel order)


# =================================================================== image io
def save_and_pack(name, arr, fmt, colorspace):
    """arr: (h, w, 3|4) row 0 = bottom, display/data values 0..1."""
    os.makedirs(TEX_DIR, exist_ok=True)
    h, w, ch = arr.shape
    old = bpy.data.images.get(name)
    if old:
        bpy.data.images.remove(old)
    tmp = bpy.data.images.new("TMP_" + name, w, h, alpha=(ch == 4))
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :ch] = np.clip(arr, 0, 1)
    tmp.pixels.foreach_set(rgba.ravel())
    ext = "jpg" if fmt == "JPEG" else "png"
    path = os.path.join(TEX_DIR, f"{name}.{ext}")
    tmp.filepath_raw = path
    tmp.file_format = fmt
    scn = bpy.context.scene
    if fmt == "JPEG":
        scn.render.image_settings.quality = 90
    tmp.save(filepath=path, quality=90) if fmt == "JPEG" else tmp.save(filepath=path)
    bpy.data.images.remove(tmp)
    img = bpy.data.images.load(path)
    img.name = name
    img.colorspace_settings.name = colorspace
    img.pack()
    img.filepath = f"//textures/{name}.{ext}"
    img["kestrel_generated"] = True
    return img


# =================================================================== materials
def gltf_output_group():
    name = "glTF Material Output"
    g = bpy.data.node_groups.get(name)
    if g is None:
        g = bpy.data.node_groups.new(name, "ShaderNodeTree")
        g.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
        g.interface.new_socket("Thickness", in_out="INPUT", socket_type="NodeSocketFloat")
    return g


def build_material(name, spec, imgs):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (600, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (250, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    base = (*spec["base"], 1.0)
    bsdf.inputs["Base Color"].default_value = base
    bsdf.inputs["Metallic"].default_value = spec["metal"]
    bsdf.inputs["Roughness"].default_value = spec["rough"]
    mat.diffuse_color = base
    mat.metallic = spec["metal"]
    mat.roughness = spec["rough"]
    L = nt.links.new
    uv = nt.nodes.new("ShaderNodeUVMap")
    uv.uv_map = UV_NAME
    uv.location = (-900, 0)

    def tex(img, loc, interp="Linear"):
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img
        t.location = loc
        t.interpolation = interp
        L(uv.outputs["UV"], t.inputs["Vector"])
        return t

    tset = spec.get("tex")
    if tset:
        tn = tex(imgs[f"T_{tset.capitalize()}_Normal"], (-600, -350))
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-250, -350)
        nm.uv_map = UV_NAME
        nm.inputs["Strength"].default_value = spec.get("nstr", 1.0)
        L(tn.outputs["Color"], nm.inputs["Color"])
        L(nm.outputs["Normal"], bsdf.inputs["Normal"])
        to = tex(imgs[f"T_{tset.capitalize()}_ORM"], (-600, 50))
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.location = (-300, 50)
        L(to.outputs["Color"], sep.inputs["Color"])
        mr = nt.nodes.new("ShaderNodeMath")
        mr.operation = "MULTIPLY"
        mr.location = (-80, 80)
        L(sep.outputs["Green"], mr.inputs[0])
        mr.inputs[1].default_value = spec["rough"]
        L(mr.outputs["Value"], bsdf.inputs["Roughness"])
        if True:   # always route B (even x0) so the exporter keeps ONE shared ORM image
            mm = nt.nodes.new("ShaderNodeMath")
            mm.operation = "MULTIPLY"
            mm.location = (-80, 200)
            L(sep.outputs["Blue"], mm.inputs[0])
            mm.inputs[1].default_value = spec["metal"]
            L(mm.outputs["Value"], bsdf.inputs["Metallic"])
        g = nt.nodes.new("ShaderNodeGroup")
        g.node_tree = gltf_output_group()
        g.location = (250, -450)
        L(sep.outputs["Red"], g.inputs["Occlusion"])
    if spec.get("decal"):
        td = tex(imgs["T_Decal_Atlas"], (-600, 300))
        L(td.outputs["Color"], bsdf.inputs["Base Color"])
        rnd = nt.nodes.new("ShaderNodeMath")
        rnd.operation = "ROUND"                     # glTF exporter -> alphaMode MASK, cutoff 0.5
        rnd.location = (-250, 300)
        L(td.outputs["Alpha"], rnd.inputs[0])
        L(rnd.outputs["Value"], bsdf.inputs["Alpha"])
        mat.surface_render_method = "DITHERED"
    if "alpha" in spec:
        bsdf.inputs["Alpha"].default_value = spec["alpha"]
        mat.surface_render_method = "BLENDED"
        mat.use_backface_culling = False
    if "emit" in spec:
        bsdf.inputs["Emission Strength"].default_value = spec["estr"]
        bsdf.inputs["Emission Color"].default_value = (*spec["emit"], 1.0)
        if spec.get("screen"):
            ts = tex(imgs["T_Screen_Atlas"], (-600, -800))
            mix = nt.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.location = (-250, -800)
            mix.inputs["Factor"].default_value = 1.0
            L(ts.outputs["Color"], mix.inputs[6])
            mix.inputs[7].default_value = (*spec["emit"], 1.0)
            L(mix.outputs[2], bsdf.inputs["Emission Color"])
    mat["kestrel_final"] = True
    for k in ("kestrel_placeholder",):
        if k in mat:
            del mat[k]
    return mat


# =================================================================== mesh helpers
def mesh_users():
    users = {}
    for o in bpy.data.objects:
        if o.type == 'MESH':
            users.setdefault(o.data.name, []).append(o)
    return users


def slot_index(me, mat):
    for i, m in enumerate(me.materials):
        if m == mat:
            return i
    me.materials.append(mat)
    return len(me.materials) - 1


def poly_arrays(me):
    npoly = len(me.polygons)
    nrm = np.empty(npoly * 3, np.float32)
    me.polygons.foreach_get("normal", nrm)
    cen = np.empty(npoly * 3, np.float32)
    me.polygons.foreach_get("center", cen)
    mi = np.empty(npoly, np.int32)
    me.polygons.foreach_get("material_index", mi)
    ls = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_start", ls)
    lt = np.empty(npoly, np.int32)
    me.polygons.foreach_get("loop_total", lt)
    return nrm.reshape(-1, 3), cen.reshape(-1, 3), mi, ls, lt


def loop_coords(me):
    nv = len(me.vertices)
    co = np.empty(nv * 3, np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    vi = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", vi)
    return co, vi


def cube_project(me, fabric_idx):
    """Per-face dominant-axis projection, metric scale.  Opposite faces are flipped so
    the texture is never mirrored."""
    nrm, cen, mi, ls, lt = poly_arrays(me)
    co, vi = loop_coords(me)
    lp = np.repeat(np.arange(len(ls)), lt)
    p = co[vi]
    n = nrm[lp]
    ax = np.argmax(np.abs(n), axis=1)
    sgn = np.sign(n[np.arange(len(n)), ax])
    sgn[sgn == 0] = 1
    u = np.where(ax == 0, p[:, 1] * sgn, np.where(ax == 1, -p[:, 0] * sgn, p[:, 0] * sgn))
    v = np.where(ax == 2, p[:, 1], p[:, 2])
    scale = np.full(len(lp), S_TILE, np.float32)
    if fabric_idx:
        scale[np.isin(mi[lp], list(fabric_idx))] = S_FABRIC
    uv = np.stack([u * scale, v * scale], axis=1).astype(np.float32)
    layer = me.uv_layers.get(UV_NAME)
    if layer is None:
        for l in list(me.uv_layers):
            me.uv_layers.remove(l)
        layer = me.uv_layers.new(name=UV_NAME)
    me.uv_layers.active = layer
    layer.data.foreach_set("uv", uv.ravel())
    return layer


def planar_basis(n, up=(0.0, 0.0, 1.0)):
    n = Vector(n).normalized()
    upv = Vector(up)
    if abs(n.dot(upv)) > 0.9:
        upv = Vector((0.0, 1.0, 0.0))
    vdir = (upv - n * upv.dot(n)).normalized()
    udir = vdir.cross(n).normalized()       # text reads left->right for a viewer facing the face
    return udir, vdir


def faces_uv_extent(me, faces, udir, vdir):
    co = [me.vertices[i].co for f in faces for i in me.polygons[f].vertices]
    us = [c.dot(udir) for c in co]
    vs = [c.dot(vdir) for c in co]
    return min(us), max(us), min(vs), max(vs)


def write_planar_uv(me, faces, udir, vdir, ext, rect):
    """Map the faces' planar extent ext=(u0,u1,v0,v1) [m] onto rect=(U0,V0,U1,V1) [uv]."""
    layer = me.uv_layers[UV_NAME]
    u0, u1, v0, v1 = ext
    U0, V0, U1, V1 = rect
    for f in faces:
        p = me.polygons[f]
        for li in range(p.loop_start, p.loop_start + p.loop_total):
            c = me.vertices[me.loops[li].vertex_index].co
            a = (c.dot(udir) - u0) / max(u1 - u0, 1e-6)
            b = (c.dot(vdir) - v0) / max(v1 - v0, 1e-6)
            layer.data[li].uv = (U0 + a * (U1 - U0), V0 + b * (V1 - V0))


def world_aabb(o):
    bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
    return (Vector([min(v[k] for v in bb) for k in range(3)]),
            Vector([max(v[k] for v in bb) for k in range(3)]))


def face_world_aabb(o, me, f):
    vs = [o.matrix_world @ me.vertices[i].co for i in me.polygons[f].vertices]
    return (Vector([min(v[k] for v in vs) for k in range(3)]),
            Vector([max(v[k] for v in vs) for k in range(3)]))


def aabb_hit(a, b, pad=0.02):
    return all(a[0][k] - pad < b[1][k] and b[0][k] - pad < a[1][k] for k in range(3))


# =================================================================== decal targets
ROOM_TITLE = {"R01": "R01 COCKPIT", "R02": "R02 AVIONICS / ECLSS", "R03": "R03 AIRLOCK VEST.",
              "R03a": "R03A AIRLOCK", "R04": "R04 CREW QUARTERS", "R05": "R05 STORES",
              "R06": "R06 ENGINEERING", "C01": "C01 CRAWLWAY"}


def wn(o, p):
    return (o.matrix_world.to_3x3() @ p.normal).normalized()


def wc(o, p):
    return o.matrix_world @ p.center


def find_targets():
    """Return a list of decal targets: dict(obj, faces, n (world), kind, text, ppm).
    Face selection is done in world space (objects may be parented / rotated)."""
    T = []
    ext_objs = [o for o in bpy.data.collections[C.COL_EXT].all_objects if o.type == 'MESH']
    fus = bpy.data.objects["EXT_HULL_Fuselage"]
    me = fus.data
    avoid = [(o.name, world_aabb(o)) for o in ext_objs
             if o.name != "EXT_HULL_Fuselage" and "BoltRow" not in o.name]

    def free(o, f):
        fa = face_world_aabb(o, o.data, f)
        return not any(aabb_hit(fa, b) for nm, b in avoid if nm != o.name)

    # --- ship ID on a chamfer band, both sides (longest greeble-free run, upper band on ties)
    for side, sx in (("R", 1.0), ("L", -1.0)):
        nn = Vector((0.7071 * sx, 0.0, 0.7071))
        best = None
        for zlo, zhi in ((1.0, 1.47), (1.46, 1.92)):
            cand = []
            for p in me.polygons:
                c = wc(fus, p)
                if wn(fus, p).dot(nn) > 0.98 and zlo < c.z < zhi and -7.5 < c.y < 6.5 and p.area > 0.05:
                    cand.append((c.y, p.index))
            cand = sorted([(y, f) for (y, f) in cand if free(fus, f)])
            runs, cur, last = [], [], None
            for y, f in cand:
                if cur and y - last > 0.62:
                    runs.append(cur)
                    cur = []
                cur.append(f)
                last = y
            if cur:
                runs.append(cur)
            for r in runs:
                ln = len(r)
                if best is None or ln > len(best[0]) or (ln == len(best[0]) and zlo > best[1]):
                    best = (r, zlo)
        if best and len(best[0]) >= 4:
            run = best[0]
            if len(run) > 8:                         # ~4.6 m window, centred nearest y = +1.5
                ys = [wc(fus, me.polygons[f]).y for f in run]
                k = min(range(len(run) - 7), key=lambda i: abs((ys[i] + ys[i + 7]) / 2 - 1.5))
                run = run[k:k + 8]
            T.append(dict(obj=fus, faces=run, n=nn, kind="ID", text="IC-4 KESTREL", ppm=230, key="ID"))

    # --- reactor radiation markers on the upper chamfer bands above B06 (spine is full of conduits)
    for sx in (1.0, -1.0):
        nn = Vector((0.7071 * sx, 0.0, 0.7071))
        cand = []
        for p in me.polygons:
            c = wc(fus, p)
            if wn(fus, p).dot(nn) > 0.98 and 1.46 < c.z < 1.92 and -8.7 < c.y < -7.4 and p.area > 0.05:
                cand.append(p.index)
        cand = [f for f in cand if free(fus, f)]
        if cand:
            T.append(dict(obj=fus, faces=cand, n=nn, kind="RAD_HULL", text="REACTOR", ppm=300, key="RADH"))

    def outer_faces(o, nn, tol, largest=False):
        md = o.data
        sel = [(wc(o, p).dot(nn), p.index) for p in md.polygons if wn(o, p).dot(nn) > 0.98]
        if not sel:
            return []
        if largest:                                   # plane carrying the most area
            acc = {}
            for d, f in sel:
                acc[round(d / tol)] = acc.get(round(d / tol), 0.0) + md.polygons[f].area
            kbest = max(acc, key=acc.get)
            return [f for d, f in sel if round(d / tol) == kbest]
        dmax = max(d for d, _ in sel)
        return [f for d, f in sel if d > dmax - tol]

    # --- crew hatch: hazard stripes on the outer frame face, caution placard on the leaf
    fr = bpy.data.objects["EXT_HATCH_CrewFrame"]
    faces = outer_faces(fr, Vector((-1, 0, 0)), 0.005)
    if faces:
        T.append(dict(obj=fr, faces=faces, n=Vector((-1, 0, 0)), kind="STRIPES", ppm=380))
    hc = bpy.data.objects["EXT_HATCH_Crew"]
    faces = outer_faces(hc, Vector((-1, 0, 0)), 0.004, largest=True)
    if faces:
        T.append(dict(obj=hc, faces=faces, n=Vector((-1, 0, 0)), kind="CAUTION", ppm=420))

    # --- docking collar top ring: hazard stripes
    dc = bpy.data.objects["EXT_DOCK_Collar"]
    faces = outer_faces(dc, Vector((0, 0, 1)), 0.005)
    if faces:
        T.append(dict(obj=dc, faces=faces, n=Vector((0, 0, 1)), kind="STRIPES", ppm=300, up=(0, 1, 0)))

    # --- nav-light markers on the sponson outboard walls (port red / starboard green)
    for side, sx in (("L", -1.0), ("R", 1.0)):
        sp = bpy.data.objects[f"EXT_HULL_Sponson_{side}"]
        faces = [p.index for p in sp.data.polygons if wn(sp, p).x * sx > 0.97 and p.area > 0.2]
        faces = [f for f in faces if free(sp, f)]
        if faces:                                    # the free wall panel nearest the bow
            faces = [max(faces, key=lambda f: wc(sp, sp.data.polygons[f]).y)]
            T.append(dict(obj=sp, faces=faces, n=Vector((sx, 0, 0)), kind="NAV_" + side, ppm=300))

    # --- interior: radiation plates (yellow plate face of the B06 hazard plates; shared mesh)
    rp = bpy.data.objects.get("INT_B06_RadHazardPlate1")
    if rp:
        names = [m.name if m else "" for m in rp.data.materials]
        ok = {"MAT_Placeholder_Handrail", "M_Paint_SafetyYellow", "M_Decals"}
        n0 = None
        for p in rp.data.polygons:                     # largest yellow face direction = plate front
            if names[p.material_index] in ok and (n0 is None or p.area > n0[1]):
                n0 = (wn(rp, p), p.area)
        if n0:
            faces = [p.index for p in rp.data.polygons
                     if names[p.material_index] in ok and wn(rp, p).dot(n0[0]) > 0.98]
            T.append(dict(obj=rp, faces=faces, n=n0[0], kind="RAD_PLATE", ppm=1400))

    # --- room ID plates on the doorway head trims (each side shows the room beyond)
    rooms = {r[0]: r for r in B.INTERIOR}
    for o in bpy.data.collections[C.COL_INT].all_objects:
        if o.type != 'MESH' or "_Doorway" not in o.name:
            continue
        a = o.name.split("_")[1]
        b = o.name.split("_Doorway")[1]
        if a not in rooms or b not in rooms:
            continue
        ca = Vector(((rooms[a][2] + rooms[a][3]) / 2, (rooms[a][4] + rooms[a][5]) / 2, 0))
        cb = Vector(((rooms[b][2] + rooms[b][3]) / 2, (rooms[b][4] + rooms[b][5]) / 2, 0))
        d = cb - ca
        axis = Vector((1, 0, 0)) if abs(d.x) > abs(d.y) else Vector((0, 1, 0))
        md = o.data
        mw = o.matrix_world
        zmax = max((mw @ v.co).z for v in md.vertices)
        for sgn in (1, -1):
            nn = axis * sgn
            cand = []
            for p in md.polygons:
                if wn(o, p).dot(nn) < 0.98:
                    continue
                vz = [(mw @ md.vertices[i].co).z for i in p.vertices]
                if min(vz) < zmax - 0.1 or max(vz) - min(vz) < 0.04:
                    continue
                cand.append(p.index)
            if not cand:
                continue
            f = max(cand, key=lambda f: wc(o, md.polygons[f]).dot(nn))
            # the face looks toward the side it is read from -> show the room behind it
            toward_b = nn.dot(d) > 0
            title = ROOM_TITLE[a] if toward_b else ROOM_TITLE[b]
            T.append(dict(obj=o, faces=[f], n=nn, kind="ROOM", text=title, ppm=900))
    return T


# =================================================================== decal atlas
DEC_N = 2048
PAD = 10


def trefoil(cv, cx, cy, R, col, bg):
    """ISO 361 radiation trefoil: centre disc + three 60 deg blades."""
    cv.circle(cx, cy, R * 0.2, col)
    for a in (30, 150, 270):
        cv.circle(cx, cy, R, col, a0=a - 30, a1=a + 30)
    cv.circle(cx, cy, R * 0.3, bg, width=R * 0.2)


def draw_decal(t, w, h):
    kind = t["kind"]
    white = tuple(srgb(MATS["M_HullPaint_White"]["base"])) + (1.0,)
    grey = tuple(srgb(MATS["M_HullPaint_Grey"]["base"])) + (1.0,)
    yellow = tuple(srgb(MATS["M_Paint_SafetyYellow"]["base"])) + (1.0,)
    ink = (0.07, 0.075, 0.08, 1.0)
    if kind == "ID":
        cv = Canvas(w, h, bg=white)
        th = h * 0.42
        tw = text_width(t["text"], th)
        if tw > w * 0.9:
            th *= w * 0.9 / tw
        cv.text(t["text"], w / 2, h * 0.22, th, ink, align="center")
        cv.rect(w * 0.05, h * 0.74, w * 0.95, h * 0.78, (0.16, 0.22, 0.32, 1))
        cv.text("INTERCEPTOR  REG 04-117", w * 0.95, h * 0.82, h * 0.09, ink, align="right")
    elif kind == "STRIPES":
        cv = Canvas(w, h, bg=yellow)
        per = 0.14 * t["ppm"]
        cv.stripes(0, 0, w, h, per, ink)
    elif kind == "CAUTION":
        cv = Canvas(w, h, bg=grey)
        m = w * 0.08
        cv.rect(m, h * 0.10, w - m, h * 0.26, yellow)
        cv.text("CAUTION", w / 2, h * 0.135, h * 0.10, ink, align="center")
        lines = ("AIRLOCK HATCH", "VERIFY ^P = 0", "BEFORE OPENING")
        for i, s in enumerate(lines):
            cv.text(s, w / 2, h * (0.34 + i * 0.09), h * 0.05, white, align="center")
        cv.rect(w * 0.30, h * 0.66, w * 0.70, h * 0.82, (0.75, 0.08, 0.05, 1))
        cv.text("EMERG PULL", w / 2, h * 0.71, h * 0.05, white, align="center")
        cv.stripes(0, h * 0.92, w, h, 0.12 * t["ppm"], yellow)
    elif kind == "RAD_PLATE":
        cv = Canvas(w, h, bg=yellow)
        purple = (0.45, 0.05, 0.35, 1)
        R = min(w, h) * 0.32
        trefoil(cv, w / 2, h * 0.42, R, purple, yellow)
        cv.frame(3, 3, w - 3, h - 3, max(2, w * 0.02), ink)
        cv.text("RADIATION", w / 2, h * 0.80, min(h * 0.09, w * 0.075), ink, align="center")
    elif kind == "RAD_HULL":
        cv = Canvas(w, h, bg=white)
        R = h * 0.26
        cx, cy = h * 0.55, h * 0.5
        cv.circle(cx, cy, R * 1.45, yellow)
        cv.circle(cx, cy, R * 1.45, ink, width=R * 0.1)
        trefoil(cv, cx, cy, R, ink, yellow)
        x0 = cx + R * 1.7
        avail = w - x0 - w * 0.04
        title = t.get("text", "REACTOR")
        th = min(h * 0.22, avail / max(text_width(title, 1.0), 1e-6))
        th2 = min(th * 0.6, avail / max(text_width("NO EVA ACCESS", 1.0), 1e-6))
        cv.text(title, x0, h * 0.30, th, ink)
        cv.text("NO EVA ACCESS", x0, h * 0.30 + th * 1.35, th2, ink)
    elif kind.startswith("NAV_"):
        port = kind.endswith("L")
        cv = Canvas(w, h, bg=grey)
        col = (0.85, 0.07, 0.05, 1) if port else (0.05, 0.70, 0.20, 1)
        s = min(w, h)
        cx, cy = w / 2, h * 0.36
        cv.circle(cx, cy, s * 0.24, white)
        cv.circle(cx, cy, s * 0.19, col)
        lab = "NAV " + ("PORT" if port else "STBD")
        th = min(h * 0.12, w * 0.84 / text_width(lab, 1.0))
        cv.text(lab, w / 2, h * 0.70, th, white, align="center")
    elif kind == "ROOM":
        # the lintel face runs behind the door-frame trims: keep the content in the
        # middle 72 % so the jambs never clip the room number
        cv = Canvas(w, h, bg=(0.10, 0.12, 0.14, 1))
        th = h * 0.58
        tw = text_width(t["text"], th)
        avail = w * 0.72 - h
        if tw > avail:
            th *= avail / tw
            tw = text_width(t["text"], th)
        x0 = (w - tw - h * 0.8) / 2
        cv.rect(x0, 0, x0 + h * 0.45, h, yellow)
        cv.text(t["text"], x0 + h * 0.8, (h - th) / 2, th, (0.93, 0.94, 0.95, 1))
    else:
        cv = Canvas(w, h, bg=white)
    return cv.finish()


def build_decals(targets):
    """Measure every target, pack cells, draw atlas, assign faces + UVs.  Returns image + stats."""
    cells = []
    shared = {}
    # deterministic order (fresh chain vs re-run iterate objects differently)
    kind_rank = {}
    for t in targets:
        kind_rank.setdefault(t["kind"], len(kind_rank))
    targets.sort(key=lambda t: (kind_rank[t["kind"]], t["obj"].name, t.get("text", ""), min(t["faces"])))
    for t in targets:
        o = t["obj"]
        me = o.data
        n_local = (o.matrix_world.to_3x3().inverted() @ t["n"]).normalized()
        up_w = Vector(t.get("up", (0, 0, 1)))
        up_local = (o.matrix_world.to_3x3().inverted() @ up_w).normalized()
        udir, vdir = planar_basis(n_local, up_local)
        ext = faces_uv_extent(me, t["faces"], udir, vdir)
        wm, hm = ext[1] - ext[0], ext[3] - ext[2]
        t.update(udir=udir, vdir=vdir, ext=ext, wm=wm, hm=hm)
        key = t.get("key")
        if key and key in shared:                   # same cell for both hull sides
            t["cell"] = shared[key]
            continue
        ppm = t["ppm"]
        w = int(min(max(wm * ppm, 32), 1500))
        hpx = int(min(max(hm * ppm, 16), 1500))
        cell = dict(w=w, h=hpx, t=t)
        cells.append(cell)
        t["cell"] = cell
        if key:
            shared[key] = cell
    # shelf packing (tallest first)
    x = y = shelf_h = 0
    for c in sorted(cells, key=lambda c: -c["h"]):
        W, H = c["w"] + 2 * PAD, c["h"] + 2 * PAD
        if x + W > DEC_N:
            x, y = 0, y + shelf_h
            shelf_h = 0
        c["x"], c["y"] = x + PAD, y + PAD        # y from the top
        x += W
        shelf_h = max(shelf_h, H)
    if y + shelf_h > DEC_N:
        raise RuntimeError("decal atlas overflow")
    atlas = np.zeros((DEC_N, DEC_N, 4), np.float32)
    atlas[..., :3] = 0.5
    for c in cells:
        img = draw_decal(c["t"], c["w"], c["h"])
        img = np.pad(img, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")   # bleed for mips/MASK
        img[..., 3] = 1.0
        y0, x0 = c["y"] - PAD, c["x"] - PAD
        atlas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    atlas = np.flipud(atlas)
    # face assignment + UVs (half-texel inset)
    dec = bpy.data.materials["M_Decals"]
    stats = []
    for t in targets:
        o, me, c = t["obj"], t["obj"].data, t["cell"]
        idx = slot_index(me, dec)
        for f in t["faces"]:
            me.polygons[f].material_index = idx
        U0 = (c["x"] + 0.5) / DEC_N
        U1 = (c["x"] + c["w"] - 0.5) / DEC_N
        V1 = 1.0 - (c["y"] + 0.5) / DEC_N
        V0 = 1.0 - (c["y"] + c["h"] - 0.5) / DEC_N
        write_planar_uv(me, t["faces"], t["udir"], t["vdir"], t["ext"], (U0, V0, U1, V1))
        stats.append(dict(kind=t["kind"], obj=o.name, faces=len(t["faces"]),
                          size_m=[round(t["wm"], 3), round(t["hm"], 3)], cell_px=[c["w"], c["h"]],
                          text=t.get("text", "")))
    return atlas, stats, len(cells)


# =================================================================== screens
SCREEN_OVERRIDE = {"INTM_MFD_40x25": ["REACTOR"], "INTM_MFD_25x23": ["ADI"], "INTM_MFD_16x14": ["RADAR"],
                   "INT_R01_MainPanel": ["ADI", "RADAR", "NAV", "ENG", "SYS", "COMMS"],
                   "INT_R06_CoolantPumpSkid": ["ENG"], "INT_R03_SuitServicePanel": ["SYS"],
                   "INT_R04_Galley": ["COMMS"], "INT_C01_JunctionBox": ["SYS"],
                   "INT_R05_SparesCrate": ["COMMS"], "INT_B01_TRUnit": ["RADAR"],
                   "INTM_FlightComputer": ["SYS"]}
DISPLAY_CYCLE = ["SYS", "COMMS", "ENG", "NAV"]


def screen_uvs(me, scr_mat, up_local=(0.0, 0.0, 1.0)):
    idx = {i for i, m in enumerate(me.materials) if m == scr_mat}
    faces = [p.index for p in me.polygons if p.material_index in idx]
    if not faces:
        return 0, 0
    parent = {}

    def find(a):
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a
    for f in faces:
        vs = list(me.polygons[f].vertices)
        ra = find(vs[0])
        for v in vs[1:]:
            rb = find(v)
            if rb != ra:
                parent[rb] = ra
    islands = {}
    for f in faces:
        islands.setdefault(find(me.polygons[f].vertices[0]), []).append(f)
    isl = list(islands.values())
    cyc = SCREEN_OVERRIDE.get(me.name, DISPLAY_CYCLE)
    big = small = 0
    # order displays left->right, top->bottom for a stable assignment
    isl.sort(key=lambda fs: (-round(sum(me.polygons[f].center.z for f in fs) / len(fs), 1),
                             sum(me.polygons[f].center.x for f in fs) / len(fs)))
    k = 0
    gx0, gy0t, gw, gh = screen_cell_rect("GLOW")
    glow_rect = ((gx0 + 60) / SCR_N, 1 - (gy0t + gh - 60) / SCR_N, (gx0 + gw - 60) / SCR_N, 1 - (gy0t + 60) / SCR_N)
    for fs in isl:
        # screens are thin boxes: the two largest (front/back) faces carry the display
        # (front and back are equal-area, so map both; the hidden one is simply mirrored)
        f0 = max(fs, key=lambda f: (round(me.polygons[f].area, 6), me.polygons[f].normal.z))
        n0 = me.polygons[f0].normal.copy()
        front = [f for f in fs if abs(me.polygons[f].normal.dot(n0)) > 0.9]
        rest = [f for f in fs if f not in front]
        udir, vdir = planar_basis(n0, up_local)
        ext = faces_uv_extent(me, front, udir, vdir)
        wm, hm = ext[1] - ext[0], ext[3] - ext[2]
        area = sum(me.polygons[f].area for f in front)
        if area > 0.003 and min(wm, hm) > 0.04:
            name = cyc[k % len(cyc)]
            k += 1
            big += 1
            x0, y0t, w, h = screen_cell_rect(name)
            ins = 3
            U0, U1 = (x0 + ins) / SCR_N, (x0 + w - ins) / SCR_N
            V1, V0 = 1 - (y0t + ins) / SCR_N, 1 - (y0t + h - ins) / SCR_N
            write_planar_uv(me, front, udir, vdir, ext, (U0, V0, U1, V1))
        else:
            small += 1
            write_planar_uv(me, front, udir, vdir, ext, glow_rect)
        if rest:
            u2, v2 = planar_basis(me.polygons[rest[0]].normal, up_local)
            write_planar_uv(me, rest, u2, v2, faces_uv_extent(me, rest, u2, v2), glow_rect)
    return big, small


# =================================================================== main
def run(out, report_path=None):
    rep = {}
    # ---------------- tileable textures + screen atlas
    imgs = {}
    hn, ho = hull_textures()
    imgs["T_Hull_Normal"] = save_and_pack("T_Hull_Normal", hn, "JPEG", "Non-Color")
    imgs["T_Hull_ORM"] = save_and_pack("T_Hull_ORM", ho, "JPEG", "Non-Color")
    inn, ino = int_textures()
    imgs["T_Int_Normal"] = save_and_pack("T_Int_Normal", inn, "JPEG", "Non-Color")
    imgs["T_Int_ORM"] = save_and_pack("T_Int_ORM", ino, "JPEG", "Non-Color")
    imgs["T_Screen_Atlas"] = save_and_pack("T_Screen_Atlas", screen_atlas(), "JPEG", "sRGB")
    # decal atlas placeholder so the material can be built; replaced below
    if "T_Decal_Atlas" not in bpy.data.images:
        bpy.data.images.new("T_Decal_Atlas", 4, 4, alpha=True)
    imgs["T_Decal_Atlas"] = bpy.data.images["T_Decal_Atlas"]
    print("KESTREL_MAT textures done", flush=True)

    # ---------------- materials + slot remap
    mats = {name: build_material(name, spec, imgs) for name, spec in MATS.items()}
    users = mesh_users()
    remap = {}
    for me in bpy.data.meshes:
        if me.name not in users:
            continue
        for i, m in enumerate(me.materials):
            if m is None:
                continue
            fin = final_for(me.name, m.name)
            remap.setdefault(m.name, set()).add(fin)
            me.materials[i] = mats[fin]
    rep["placeholder_map"] = {k: sorted(v) for k, v in sorted(remap.items())}

    # fuselage two-tone: belly + lower chamfers grey
    fus = bpy.data.objects["EXT_HULL_Fuselage"].data
    gi = slot_index(fus, mats["M_HullPaint_Grey"])
    wi = slot_index(fus, mats["M_HullPaint_White"])
    nbelly = 0
    for p in fus.polygons:
        if fus.materials[p.material_index] in (mats["M_HullPaint_White"], mats["M_HullPaint_Grey"]):
            if p.normal.z < -0.3:
                p.material_index = gi
                nbelly += 1
            else:
                p.material_index = wi
    rep["fuselage_belly_faces_grey"] = nbelly

    # ---------------- UVs (cube projection for every mesh)
    nmesh = 0
    for me in bpy.data.meshes:
        if me.name not in users:
            continue
        fab = {i for i, m in enumerate(me.materials) if m == mats["M_Int_Fabric"]}
        cube_project(me, fab)
        nmesh += 1
    rep["uv_cube_projected_meshes"] = nmesh

    # ---------------- screens
    nb = ns = 0
    for me in bpy.data.meshes:
        if me.name in users and mats["M_Screens_Emissive"] in list(me.materials):
            o0 = users[me.name][0]
            up_local = tuple((o0.matrix_world.to_3x3().inverted() @ Vector((0, 0, 1))).normalized())
            b, s = screen_uvs(me, mats["M_Screens_Emissive"], up_local)
            nb += b
            ns += s
    rep["screen_islands"] = {"display": nb, "glow": ns}

    # ---------------- decals
    targets = find_targets()
    atlas, dstats, ncells = build_decals(targets)
    old = bpy.data.images.get("T_Decal_Atlas")
    if old:
        bpy.data.images.remove(old)
    imgs["T_Decal_Atlas"] = save_and_pack("T_Decal_Atlas", atlas, "PNG", "sRGB")
    for n in mats["M_Decals"].node_tree.nodes:
        if n.type == 'TEX_IMAGE':
            n.image = imgs["T_Decal_Atlas"]
    rep["decals"] = dstats
    rep["decal_cells"] = ncells
    print("KESTREL_MAT decals done", flush=True)

    # ---------------- purge placeholders / orphans
    for m in list(bpy.data.materials):
        if m.users == 0 or m.name.startswith("MAT_Placeholder_"):
            if m.users == 0:
                bpy.data.materials.remove(m)
    for im in list(bpy.data.images):
        if im.users == 0:
            bpy.data.images.remove(im)
    rep.update(report())
    C.save_blend(out)
    rep["saved"] = out
    if report_path:
        with open(report_path, "w") as f:
            json.dump(rep, f, indent=1)
    print("KESTREL_MAT_REPORT " + json.dumps(rep), flush=True)
    return rep


def report():
    users = mesh_users()
    meshes = [me for me in bpy.data.meshes if me.name in users]
    objs = [o for o in bpy.data.objects if o.type == 'MESH']
    mat_objs, mat_meshes = {}, {}
    for o in objs:
        for m in {s.material.name for s in o.material_slots if s.material}:
            mat_objs[m] = mat_objs.get(m, 0) + 1
    for me in meshes:
        for m in {m.name for m in me.materials if m}:
            mat_meshes[m] = mat_meshes.get(m, 0) + 1
    tex = []
    total = 0
    for im in bpy.data.images:
        if im.packed_file:
            sz = im.packed_file.size
            total += sz
            tex.append(dict(name=im.name, size=list(im.size), format=im.file_format, packed_bytes=sz,
                            colorspace=im.colorspace_settings.name))
    no_uv = [me.name for me in meshes if len(me.uv_layers) == 0]
    tris = C.triangle_count()
    return dict(
        objects=len(bpy.data.objects), mesh_objects=len(objs), meshes=len(meshes), triangles=tris,
        materials=sorted(m.name for m in bpy.data.materials),
        material_object_count=dict(sorted(mat_objs.items())),
        material_mesh_count=dict(sorted(mat_meshes.items())),
        textures=tex, textures_total_bytes=total, textures_total_MB=round(total / 1048576, 3),
        meshes_without_uv=no_uv,
        placeholders_left=[m.name for m in bpy.data.materials if m.name.startswith("MAT_Placeholder_")],
    )


if __name__ == "__main__":
    args = C.script_args()
    src = args.get("blend", C.BLEND_PATH)
    out = args.get("out", src)
    C.open_blend(src)
    run(out, args.get("report"))
