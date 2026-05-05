#!/usr/bin/env bash
# No-op: cricket's single-arm FR3 ships a hand-tuned spherization
# already, and tools/build_bimanual_urdf.py just stamps two copies into
# the bimanual cell. There's nothing to decompose / re-spherize unless
# you want to replace the meshes — see scripts/build_robot.sh.
set -euo pipefail
echo "[decompose_robot] no-op: bimanual_fr3 inherits cricket's pre-spherized FR3"
