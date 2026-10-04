#!/usr/bin/env python3
"""
ARMR Assignment 1 - Part B asset pipeline (pure Python / NumPy, no Blender needed).

Stages (mirrors the Blender workflow in the assignment brief):
  1. AUTHOR   : procedural ornamental vase, ~737k triangles (surface of revolution + relief detail)
  2. DECIMATE : vertex-clustering decimation  ->  6,144 triangles
  3. BAKE     : high-poly -> low-poly tangent-space NORMAL map + baked DIFFUSE/AO map (1024x1024)
  4. EXPORT   : glTF binary (.glb) + metrics.json (triangles, draw calls, file size, GPU memory)
Draco compression is the final step (see optimize_draco.sh) because it needs an external encoder.

Run:  python3 make_asset.py [output_dir]
"""
import os, sys, json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import uniform_filter1d, map_coordinates
from PIL import Image
from glbtools import GLBBuilder

OUT = sys.argv[1] if len(sys.argv) > 1 else "../assets"
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ resolution
SU, SV = 64, 48            # LOW-poly grid segments (around, along profile)
KU, KV = 12, 10            # decimation ratio per axis
SUH, SVH = SU * KU, SV * KV  # HIGH-poly grid segments (768 x 480)
TEX = 1024

# ------------------------------------------------------------------ vase profile (r, y)
ctrl = np.array([(0, 0), (.46, 0), (.54, .03), (.50, .09), (.56, .24), (.80, .58), (.88, .85), (.80, 1.10),
                 (.52, 1.36), (.38, 1.55), (.40, 1.72), (.55, 1.86), (.58, 1.92), (.50, 1.95), (.44, 1.90),
                 (.34, 1.78), (.30, 1.55), (.28, 1.30), (0, 1.25)], float)
t = np.r_[0, np.cumsum(np.hypot(*np.diff(ctrl, axis=0).T))]
fr, fy = PchipInterpolator(t, ctrl[:, 0]), PchipInterpolator(t, ctrl[:, 1])
td = np.linspace(0, t[-1], 40001)
rd, yd = fr(td), fy(td)
sd = np.r_[0, np.cumsum(np.hypot(np.diff(rd), np.diff(yd)))]
L = sd[-1]; vd = sd / L                                  # arclength parameter v in [0,1]
drd, dyd = np.gradient(rd, sd), np.gradient(yd, sd)
V_TOP = vd[np.argmax(yd)]                                # v where the outer wall ends (rim top)

def prof(v):
    return (np.interp(v, vd, rd), np.interp(v, vd, yd), np.interp(v, vd, drd), np.interp(v, vd, dyd))

def smooth(a, b, x):
    x = np.clip((x - a) / (b - a), 0, 1); return x * x * (3 - 2 * x)

rng = np.random.default_rng(7)
NZ = [(int(rng.integers(5, 40)), rng.uniform(8, 60), rng.uniform(0, 6.28), rng.uniform(0, 6.28)) for _ in range(6)]

def decor(u, v):
    """Relief height h(u,v) and masks. Single source of truth for high-poly geometry AND the baked textures."""
    r, y, _, _ = prof(v); outer = (v < V_TOP).astype(float)
    # (1) diamond stud lattice on the belly (24 around, 3 rows)
    a, b = 24 * u + (y - .28) / .22, 24 * u - (y - .28) / .22
    da, db = np.abs((a % 1) - .5) * 2, np.abs((b % 1) - .5) * 2
    stud = smooth(0, .75, 1 - np.maximum(da, db))
    m1 = smooth(.28, .34, y) * (1 - smooth(.84, .90, y)) * outer
    # (2) round beads around the neck (20 around)
    dx = ((20 * u) % 1 - .5) * 2 * np.pi * np.maximum(r, .05) / 20; dy = y - 1.53
    bead = np.exp(-(dx ** 2 + dy ** 2) / (2 * .035 ** 2)) * outer
    # (3) concentric ridges on foot and shoulder
    ring = ((.5 + .5 * np.cos(2 * np.pi * (y - .05) / .04)) ** 3 * smooth(.04, .06, y) * (1 - smooth(.18, .20, y))
            + (.5 + .5 * np.cos(2 * np.pi * (y - .98) / .04)) ** 3 * smooth(.97, .99, y) * (1 - smooth(1.12, 1.14, y))) * outer
    # (4) fine hammered noise (integer frequency around -> seamless)
    noise = sum(np.sin(2 * np.pi * k * u + f * v + p) * np.sin(f * .7 * v + q) for k, f, p, q in NZ) / len(NZ)
    pole_fade = smooth(0, .01, v) * (1 - smooth(.99, 1, v))
    h = (.035 * stud * m1 + .030 * bead + .012 * ring + .0022 * noise * outer) * pole_fade
    return h, stud * m1, bead, ring, noise, outer

def color(u, v):
    """sRGB albedo (with baked cavity/AO) at (u,v)."""
    h, stud, bead, ring, noise, outer = decor(u, v)
    r, y, _, _ = prof(v)
    teal, bronze = np.array([.10, .34, .40]), np.array([.50, .30, .17])
    gold, ivory, cream = np.array([.86, .66, .24]), np.array([.93, .87, .72]), np.array([.86, .80, .68])
    c = teal + (bronze - teal) * np.clip(smooth(.95, .97, y) * (1 - smooth(1.12, 1.14, y)) + (1 - smooth(.2, .22, y)), 0, 1)[..., None]
    c = c + (gold - c) * stud[..., None]
    c = c + (ivory - c) * smooth(.2, .8, bead)[..., None]
    c = c + (gold - c) * smooth(.9, .97, v)[..., None] * (v < V_TOP)[..., None]
    c = np.where((v >= V_TOP)[..., None], cream, c)
    ao = 1 - .45 * (1 - np.maximum(stud, 0)) * smooth(.28, .34, y)[..., None][..., 0] * (1 - smooth(.84, .90, y)) * outer
    ao = ao * (1 - .25 * ring)
    return np.clip(c * ao[..., None] * (1 + .05 * noise[..., None]), 0, 1)

def srgb2lin(c):
    return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)

# ------------------------------------------------------------------ 1. AUTHOR high-poly
def grid_uv(su, sv):
    u, v = np.meshgrid(np.linspace(0, 1, su + 1), np.linspace(0, 1, sv + 1)); return u, v

uh, vh = grid_uv(SUH, SVH)
r0, y0, dr0, dy0 = prof(vh); th = 2 * np.pi * uh
nr, ny = dy0, -dr0; nn = np.hypot(nr, ny); nr, ny = nr / nn, ny / nn          # profile-plane normal
N0 = np.stack([nr * np.cos(th), ny, nr * np.sin(th)], -1)
P0 = np.stack([r0 * np.cos(th), y0, r0 * np.sin(th)], -1)
hh = decor(uh, vh)[0]
PH = P0 + hh[..., None] * N0
du = np.gradient(PH, axis=1); dv = np.gradient(PH, axis=0)
NH = np.cross(dv, du); nl = np.linalg.norm(NH, axis=-1, keepdims=True)
NH = np.where(nl > 1e-9, NH / np.maximum(nl, 1e-12), N0)
NH[:, -1] = NH[:, 0]; PH[:, -1] = PH[:, 0]                                   # seam
CH = color(uh, vh)

def faces(su, sv):
    i, j = np.meshgrid(np.arange(su), np.arange(sv))
    a = (j * (su + 1) + i); b = a + 1; c = a + su + 1; d = c + 1
    return np.stack([np.stack([a, c, b], -1), np.stack([b, c, d], -1)], 2).reshape(-1, 3)

# sanity: winding must agree with outward normals
Fh = faces(SUH, SVH); p = PH.reshape(-1, 3)
fn = np.cross(p[Fh[:, 1]] - p[Fh[:, 0]], p[Fh[:, 2]] - p[Fh[:, 0]])
agree = (np.einsum("ij,ij->i", fn, NH.reshape(-1, 3)[Fh[:, 0]]) > 0).mean()
assert agree > .98, f"winding disagrees with normals ({agree:.3f})"

def high_prim_rows(b, j0, j1, mat):
    sl = slice(j0 * (SUH + 1), (j1 + 1) * (SUH + 1))
    rows = j1 - j0; f = faces(SUH, rows)
    return b.primitive(PH.reshape(-1, 3)[sl].astype("f4"), NH.reshape(-1, 3)[sl].astype("f4"),
                       f.astype("u4"), col=srgb2lin(CH.reshape(-1, 3)[sl]).astype("f4"), material=mat)

b = GLBBuilder()
jf = int(np.searchsorted(yd, .22) / len(vd) * SVH); jb = int(np.argmax(vh[:, 0] >= .62))
jf, jb = (jf // KV) * KV, (jb // KV) * KV
for nm, col in (("foot", [.5, .3, .17]), ("body", [.1, .34, .4]), ("neck_rim_inside", [.86, .8, .68])):
    b.materials.append({"name": nm, "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": .1, "roughnessFactor": .55}})
prims = [high_prim_rows(b, 0, jf, 0), high_prim_rows(b, jf, jb, 1), high_prim_rows(b, jb, SVH, 2)]
HIGH = os.path.join(OUT, "vase_high_poly.glb"); b.save(HIGH, prims, "vase_high")

# ------------------------------------------------------------------ 2. DECIMATE (vertex clustering)
core = PH[:, :-1]
core = uniform_filter1d(core, 2 * KU // 2 + 1, axis=1, mode="wrap")
core = uniform_filter1d(core, KV + 1, axis=0, mode="nearest")
PL = core[::KV, ::KU]; PL = np.concatenate([PL, PL[:, :1]], 1)              # (SV+1, SU+1, 3)
PL[0, :, [0, 2]] = 0; PL[-1, :, [0, 2]] = 0                                  # keep the poles closed
Fl = faces(SU, SV)
ul, vl = grid_uv(SU, SV)

def vertex_normals(P, F, su):
    p = P.reshape(-1, 3); fnn = np.cross(p[F[:, 1]] - p[F[:, 0]], p[F[:, 2]] - p[F[:, 0]])
    n = np.zeros_like(p)
    for k in range(3): np.add.at(n, F[:, k], fnn)
    n = n.reshape(P.shape); n[:, 0] += n[:, -1]; n[:, -1] = n[:, 0]          # weld the seam
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)

NL = vertex_normals(PL, Fl, SU)
thl = 2 * np.pi * ul
TL = np.stack([-np.sin(thl), np.zeros_like(thl), np.cos(thl)], -1)           # revolution tangent
TL = TL - NL * np.sum(TL * NL, -1, keepdims=True); TL /= np.linalg.norm(TL, axis=-1, keepdims=True)

# ------------------------------------------------------------------ 3. BAKE
def interp_tri(A, u, v):
    """Interpolate a low-poly grid attribute exactly as the GPU would (same triangle split)."""
    x, y = u * SU, v * SV
    i = np.clip(np.floor(x).astype(int), 0, SU - 1); j = np.clip(np.floor(y).astype(int), 0, SV - 1)
    fu, fv = (x - i)[..., None], (y - j)[..., None]
    a, bb, c, d = A[j, i], A[j, i + 1], A[j + 1, i], A[j + 1, i + 1]
    return np.where(fu + fv <= 1, a + fu * (bb - a) + fv * (c - a), d + (1 - fu) * (c - d) + (1 - fv) * (bb - d))

tu, tv = np.meshgrid((np.arange(TEX) + .5) / TEX, (np.arange(TEX) + .5) / TEX)
Nl = interp_tri(NL, tu, tv); Tl = interp_tri(TL, tu, tv)
Nl /= np.linalg.norm(Nl, axis=-1, keepdims=True)
Tl = Tl - Nl * np.sum(Tl * Nl, -1, keepdims=True); Tl /= np.linalg.norm(Tl, axis=-1, keepdims=True)
Bl = np.cross(Nl, Tl)                                                         # glTF: B = cross(N,T)*w, w=+1
hx, hy = tu * SUH, tv * SVH
Nh = np.stack([map_coordinates(NH[..., k], [hy, hx], order=1, mode="nearest") for k in range(3)], -1)
Nh /= np.linalg.norm(Nh, axis=-1, keepdims=True)
nts = np.stack([np.sum(Nh * Tl, -1), np.sum(Nh * Bl, -1), np.sum(Nh * Nl, -1)], -1)
normal_img = Image.fromarray(np.uint8(np.clip((nts * .5 + .5) * 255 + .5, 0, 255)))
diffuse_img = Image.fromarray(np.uint8(np.clip(color(tu, tv) * 255 + .5, 0, 255)))
normal_img.save(os.path.join(OUT, "vase_normal.png"), optimize=True)
diffuse_img.save(os.path.join(OUT, "vase_diffuse.jpg"), quality=88, optimize=True)

# ------------------------------------------------------------------ 4. EXPORT optimised GLB
b = GLBBuilder()
b.materials.append({"name": "vase_baked", "pbrMetallicRoughness": {
    "baseColorTexture": {"index": b.texture(b.image(diffuse_img, "JPEG"))},
    "metallicFactor": .1, "roughnessFactor": .55}, "normalTexture": {"index": b.texture(b.image(normal_img, "PNG")), "scale": 1.0}})
tan = np.concatenate([TL.reshape(-1, 3), np.ones((TL.size // 3, 1))], 1).astype("f4")
prim = b.primitive(PL.reshape(-1, 3).astype("f4"), NL.reshape(-1, 3).astype("f4"), Fl.astype("u2"),
                   uv=np.stack([ul, vl], -1).reshape(-1, 2).astype("f4"), tan=tan, material=0)
LOW = os.path.join(OUT, "vase_optimized.glb"); b.save(LOW, [prim], "vase_optimized")

# ------------------------------------------------------------------ metrics
def mb(p): return os.path.getsize(p) / 1048576
hv, lv = (SUH + 1) * (SVH + 1), (SU + 1) * (SV + 1)
M = {
    "before": {"file": "vase_high_poly.glb", "triangles": len(Fh), "vertices": hv, "draw_calls": 3, "materials": 3,
               "textures": 0, "file_size_mb": round(mb(HIGH), 2),
               "gpu_mem_est_mb": round((hv * 36 + len(Fh) * 12) / 1048576, 1)},
    "after": {"file": "vase_optimized.glb", "triangles": len(Fl), "vertices": lv, "draw_calls": 1, "materials": 1,
              "textures": 2, "file_size_mb": round(mb(LOW), 2),
              "gpu_mem_est_mb": round((lv * 48 + len(Fl) * 6 + 2 * TEX * TEX * 4 * 4 / 3) / 1048576, 1)},
    "textures": {"diffuse_jpg_mb": round(os.path.getsize(os.path.join(OUT, "vase_diffuse.jpg")) / 1048576, 2),
                 "normal_png_mb": round(os.path.getsize(os.path.join(OUT, "vase_normal.png")) / 1048576, 2), "resolution": TEX},
    "decimation": {"ratio_triangles": round(len(Fh) / len(Fl), 1), "method": "vertex clustering (12x10 cells)"},
}
M["after"]["triangle_reduction_pct"] = round(100 * (1 - len(Fl) / len(Fh)), 2)
M["after"]["size_reduction_pct"] = round(100 * (1 - M["after"]["file_size_mb"] / M["before"]["file_size_mb"]), 1)
json.dump(M, open(os.path.join(OUT, "metrics.json"), "w"), indent=2)
print(json.dumps(M, indent=2))
