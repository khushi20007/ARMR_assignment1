#!/usr/bin/env python3
"""Software renderer used to make the report figures. Reads the exported .glb files (not the in-memory data),
so the figures show exactly what is inside the delivered files."""
import sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from glbtools import read_glb

ASSETS = sys.argv[1] if len(sys.argv) > 1 else "../assets"
FIGS = sys.argv[2] if len(sys.argv) > 2 else "../figures"
os.makedirs(FIGS, exist_ok=True)
W, H = 900, 1100

def rot(ax, ay):
    cx, sx, cy, sy = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

R = rot(np.radians(16), np.radians(-28))
LIGHT = np.array([-.45, .65, .62]); LIGHT /= np.linalg.norm(LIGHT)
HALF = np.array([0, 0, 1.]) + LIGHT; HALF /= np.linalg.norm(HALF)

def lin2s(c): return np.where(c <= .0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - .055)
def s2lin(c): return np.where(c <= .04045, c / 12.92, ((c + .055) / 1.055) ** 2.4)

def bilinear(img, uv):
    a = np.asarray(img, np.float32) / 255; h, w = a.shape[:2]
    x = np.mod(uv[:, 0], 1) * w - .5; y = np.clip(uv[:, 1], 0, 1) * h - .5
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    g = lambda yy, xx: a[np.clip(yy, 0, h - 1), xx % w]
    return (g(y0, x0) * (1 - fx) * (1 - fy) + g(y0, x0 + 1) * fx * (1 - fy) + g(y0 + 1, x0) * (1 - fx) * fy + g(y0 + 1, x0 + 1) * fx * fy)

def render(glb, mode, K, scale=None, center=(0, .95, 0), splat=1, zoom_box=None):
    """mode: 'vertex' (vertex colours), 'flat' (single colour, vertex normals), 'baked' (diffuse+normal maps)."""
    d = read_glb(glb); pos_l, nrm_l, col_l = [], [], []
    for pr in d["prims"]:
        P = pr["POSITION"].astype(np.float64); N = pr["NORMAL"].astype(np.float64); F = pr["indices"].reshape(-1, 3)
        fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]) @ R.T
        F = F[fn[:, 2] > 0]                                        # back-face culling
        bc = np.array([(i, j, K - i - j) for i in range(K + 1) for j in range(K + 1 - i)], float) / K
        pos_l.append(P); nrm_l.append(N)
        # sample barycentric lattice on every front-facing triangle
        w = bc[None]; 
        pts = np.einsum("tk,fkc->ftc", np.ones((1, 1)), np.zeros((len(F), 1, 3))) if False else None
        A = np.einsum("pk,fkc->fpc", bc, np.stack([P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]], 1)).reshape(-1, 3)
        An = np.einsum("pk,fkc->fpc", bc, np.stack([N[F[:, 0]], N[F[:, 1]], N[F[:, 2]]], 1)).reshape(-1, 3)
        An /= np.maximum(np.linalg.norm(An, axis=1, keepdims=True), 1e-9)
        if mode == "vertex":
            C = np.einsum("pk,fkc->fpc", bc, np.stack([pr["COLOR_0"][F[:, 0]], pr["COLOR_0"][F[:, 1]], pr["COLOR_0"][F[:, 2]]], 1)).reshape(-1, 3)
        elif mode == "flat":
            C = np.tile(s2lin(np.array([.10, .34, .40])), (len(A), 1))
        else:
            UV = np.einsum("pk,fkc->fpc", bc, np.stack([pr["TEXCOORD_0"][F[:, 0]], pr["TEXCOORD_0"][F[:, 1]], pr["TEXCOORD_0"][F[:, 2]]], 1)).reshape(-1, 2)
            Tn = np.einsum("pk,fkc->fpc", bc, np.stack([pr["TANGENT"][F[:, 0], :3], pr["TANGENT"][F[:, 1], :3], pr["TANGENT"][F[:, 2], :3]], 1)).reshape(-1, 3)
            Tn = Tn - An * np.sum(Tn * An, 1, keepdims=True); Tn /= np.maximum(np.linalg.norm(Tn, axis=1, keepdims=True), 1e-9)
            Bn = np.cross(An, Tn)
            m = bilinear(d["images"][1], UV) * 2 - 1
            An = m[:, :1] * Tn + m[:, 1:2] * Bn + m[:, 2:3] * An
            An /= np.maximum(np.linalg.norm(An, axis=1, keepdims=True), 1e-9)
            C = s2lin(bilinear(d["images"][0], UV))
        Q = (A - np.array(center)) @ R.T; Nr = An @ R.T
        diff = np.clip(Nr @ LIGHT, 0, 1)[:, None]; spec = (np.clip(Nr @ HALF, 0, 1) ** 40)[:, None] * .25
        amb = (.28 + .12 * Nr[:, 1:2])
        col = C * (amb + .95 * diff) + spec
        pos_l[-1] = None; nrm_l[-1] = None
        col_l.append((Q, lin2s(np.clip(col, 0, 1))))
    Q = np.concatenate([q for q, _ in col_l]); col = np.concatenate([c for _, c in col_l])
    s = scale or H * .43 / 1.1
    px = (W / 2 + Q[:, 0] * s).astype(int); py = (H / 2 - Q[:, 1] * s).astype(int)
    img = np.zeros((H, W, 3), np.float32) + .93; zb = np.full((H, W), -1e9, np.float32)
    order = np.argsort(Q[:, 2])
    for dx in range(splat):
        for dy in range(splat):
            x, y = px[order] + dx, py[order] + dy; ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
            x, y, zz, cc = x[ok], y[ok], Q[order][ok, 2], col[order][ok]
            # painter order (far->near) with last-write-wins, guarded by depth test
            flat = y * W + x
            best = np.full(H * W, -1e9, np.float32); np.maximum.at(best, flat, zz.astype(np.float32))
            win = zz >= best[flat] - 1e-6
            im = img.reshape(-1, 3); im[flat[win]] = cc[win]
    return (img * 255).astype(np.uint8)

def label(im, text, sub):
    pil = Image.fromarray(im); dr = ImageDraw.Draw(pil)
    try:
        f1 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        f2 = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    dr.text((24, 18), text, fill=(20, 20, 20), font=f1); dr.text((24, 58), sub, fill=(70, 70, 70), font=f2)
    return pil

def main():
    hi, lo = os.path.join(ASSETS, "vase_high_poly.glb"), os.path.join(ASSETS, "vase_optimized.glb")
    a = render(hi, "vertex", K=3, splat=2)
    b = render(lo, "flat", K=36, splat=2)
    c = render(lo, "baked", K=36, splat=2)
    import json; M = json.load(open(os.path.join(ASSETS, "metrics.json")))
    t = [label(a, "(a) High-poly source", f"{M['before']['triangles']:,} tris | {M['before']['file_size_mb']} MB"),
         label(b, "(b) Decimated, no bake", f"{M['after']['triangles']:,} tris | smooth normals only"),
         label(c, "(c) Decimated + baked maps", f"{M['after']['triangles']:,} tris | {M['after']['file_size_mb']} MB")]
    sheet = Image.new("RGB", (W * 3, H), (255, 255, 255))
    for i, im in enumerate(t): sheet.paste(im, (i * W, 0))
    sheet.save(os.path.join(FIGS, "fig_comparison_full.png"))
    # zoomed detail row (belly lattice) - crop the same region from each render
    box = (int(W * .30), int(H * .50), int(W * .80), int(H * .70))
    z = Image.new("RGB", ((box[2] - box[0]) * 3 * 2, (box[3] - box[1]) * 2), (255, 255, 255))
    for i, im in enumerate([a, b, c]):
        z.paste(Image.fromarray(im).crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2), Image.LANCZOS), (i * (box[2] - box[0]) * 2, 0))
    z.save(os.path.join(FIGS, "fig_comparison_zoom.png"))
    print("figures written")

if __name__ == "__main__":
    main()
