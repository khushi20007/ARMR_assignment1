#!/usr/bin/env python3
"""Generate a feature-rich, asymmetric image-tracking target (good corner/edge density for MindAR)."""
import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
out = sys.argv[1] if len(sys.argv) > 1 else "../web/assets/target.png"
W, H = 1024, 768
rng = np.random.default_rng(2026)
bg = rng.normal(0.5, 0.18, (H // 8, W // 8)); bg = np.kron(bg, np.ones((8, 8)))
img = Image.fromarray(np.uint8(np.clip(bg, 0, 1) * 255)).convert("RGB").filter(ImageFilter.GaussianBlur(2))
d = ImageDraw.Draw(img)
pal = [(15, 25, 60), (230, 70, 40), (245, 200, 50), (30, 150, 140), (250, 245, 235), (120, 40, 140), (20, 20, 20)]
for _ in range(170):                                   # random polygons / circles / lines
    k = rng.integers(0, 3); c = pal[rng.integers(0, len(pal))]
    x, y = rng.integers(0, W), rng.integers(0, H); s = int(rng.integers(30, 150))
    if k == 0: d.polygon([(x + int(rng.integers(-s, s)), y + int(rng.integers(-s, s))) for _ in range(int(rng.integers(3, 6)))], fill=c, outline=(0, 0, 0))
    elif k == 1: d.ellipse([x, y, x + s, y + int(s * rng.uniform(.5, 1.4))], fill=c, outline=(255, 255, 255), width=3)
    else: d.line([x, y, x + int(rng.integers(-3 * s, 3 * s)), y + int(rng.integers(-3 * s, 3 * s))], fill=c, width=int(rng.integers(3, 9)))
d.rectangle([14, 14, W - 15, H - 15], outline=(0, 0, 0), width=14)           # border
d.rectangle([40, 40, W - 41, H - 41], outline=(255, 255, 255), width=5)
try: f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 74); g = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
except Exception: f = g = ImageFont.load_default()
d.rounded_rectangle([90, 300, 760, 470], 18, fill=(10, 10, 10), outline=(245, 200, 50), width=6)
d.text((120, 318), "ARMR  Asg-1", fill=(245, 200, 50), font=f); d.text((122, 408), "scan me  |  WebAR  |  MindAR", fill=(255, 255, 255), font=g)
img.save(out, optimize=True); print("target ->", out)
