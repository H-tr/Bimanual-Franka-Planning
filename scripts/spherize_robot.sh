#!/usr/bin/env bash
# No-op: cricket's single-arm FR3 ships a hand-tuned spherization
# already (see third_party/cricket/resources/fr3/fr3_spherized.urdf).
# tools/build_bimanual_urdf.py composes two copies into the bimanual
# cell, preserving the original sphere density per arm.
set -euo pipefail
echo "[spherize_robot] no-op: bimanual_fr3 inherits cricket's pre-spherized FR3"
