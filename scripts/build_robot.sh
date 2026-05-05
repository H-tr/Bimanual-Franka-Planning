#!/usr/bin/env bash
# Build planning-ready bimanual FR3 description from cricket's single-arm
# FR3 source. All project-specific paths live in tools/build_bimanual_urdf.py;
# this shell wrapper is here so the build pipeline matches the autolife
# project layout (pixi run build-robot).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

python "$ROOT/tools/build_bimanual_urdf.py" "$@"
