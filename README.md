# ARMR Assignment 1 - WebAR vase (MindAR.js + optimized glTF)

## What is in this folder
| Path | Purpose |
|---|---|
| `report/ARMR_Assignment1_Report.docx` / `.pdf` | 3-page survey + optimization report (Parts A, B, C) |
| `vase_optimized.glb` | **Final optimized asset** (6,144 tris, 1 draw call, 0.65 MB) - Draco build is created in step 5 |
| `web/` | The AR site. **This folder's contents are the GitHub repo root** |
| `assets/` | High-poly source (`vase_high_poly.glb`, 21 MB), baked textures, `metrics.json` |
| `pipeline/` | `make_asset.py` (decimate + bake + export), `render_figs.py`, `make_target.py`, `optimize_draco.sh`, `blender_pipeline.py` |
| `figures/` | Before/after renders used in the report |

## What was done here vs. what you must do
Done and measured: high-poly generation, decimation, normal + diffuse bake, GLB export, all metrics in Table 2, report text, demo page.
**Not possible in the authoring environment (no Blender, no internet, no phone) - you must do these:**
1. Compile the tracking target -> `targets.mind` (step 2).
2. Deploy to GitHub Pages and get the live URL (step 4).
3. Run the Draco step and fill the yellow column in Table 2 (step 5).
4. Test on real phones and fill Table 3 (step 6).
5. Fill your name / roll no. / URLs (yellow fields) in the report.
The page itself has **not** been run in a browser here; test it before submitting.

## Steps
1. *(optional)* Rebuild everything: `cd pipeline && python3 make_asset.py ../assets && python3 render_figs.py ../assets ../figures` (needs numpy, scipy, pillow).
2. **Compile target:** open https://hiukim.github.io/mind-ar-js-doc/tools/compile , upload `web/assets/target.png`, press Start, download `targets.mind`, save as `web/assets/targets.mind`. Check the feature-point preview is well spread over the image.
3. **Local test:** `cd web && python3 -m http.server 8000`, open http://localhost:8000 on your laptop (localhost is allowed camera access), show `target.png` on another screen.
4. **GitHub Pages:** create a public repo, upload the *contents of `web/`* (including `.nojekyll`, `assets/targets.mind`), then Settings -> Pages -> Deploy from branch `main` / root. URL: `https://<user>.github.io/<repo>/`. Open it on your phone, allow the camera, aim at the target.
5. **Draco:** `cd pipeline && bash optimize_draco.sh`. Test with `...?model=vase_optimized_draco.glb`. Copy the new file size into Table 2, and use the Draco file as `vase_optimized.glb` in the final submission if your instructor requires "Draco .glb".
6. **Measure:** open `...?debug=1` on each phone, hold the target steady 30 s; read FPS from the on-screen counter / stats panel (`fpsLog` in the console for samples). Record lock time and stability in Table 3.
7. **Blender (recommended):** the brief says Blender. Import `assets/vase_high_poly.glb`, apply Decimate (Collapse, ratio 0.0083), Smart UV, bake Normal + Diffuse (selected-to-active), export GLB with Draco. `pipeline/blender_pipeline.py` automates this (untested - adjust to your Blender version). If you do this, add your screenshots and update the Part B text so it describes what you actually did.

## Troubleshooting
- Black screen / no camera: page must be HTTPS (or localhost) and camera permission granted.
- Model never appears: `targets.mind` missing or wrong path; open DevTools console.
- Model sideways/too big: edit `rotation`, `scale`, `position` on the `<a-gltf-model>` in `index.html`.
- CDN versions are pinned (A-Frame 1.5.0, MindAR 1.2.5); if a CDN is blocked, download the two scripts and serve them locally.
