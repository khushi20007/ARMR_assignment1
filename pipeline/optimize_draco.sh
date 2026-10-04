#!/usr/bin/env bash
# Final pipeline step: Draco-compress the optimised .glb  (needs Node.js + internet once for npx)
# Run from the pipeline/ folder:   bash optimize_draco.sh
set -e
IN=../web/assets/vase_optimized.glb
OUT=../web/assets/vase_optimized_draco.glb

# Option 1 - gltf-pipeline (Cesium)
npx --yes gltf-pipeline -i "$IN" -o "$OUT" -d

# Option 2 - if option 1 complains about the TANGENT attribute, use glTF-Transform instead:
# npx --yes @gltf-transform/cli draco "$IN" "$OUT" --method edgebreaker
#
# Optional: shrink textures too (they are ~70% of the file):
# npx --yes @gltf-transform/cli webp "$OUT" ../web/assets/vase_optimized_draco_webp.glb

ls -l "$IN" "$OUT"
echo "Now test with:  index.html?model=vase_optimized_draco.glb   (decoder is loaded from gstatic automatically)"
