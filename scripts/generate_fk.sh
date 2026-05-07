#!/usr/bin/env bash
# Generate the FK + collision-checking C++ headers via cricket for every
# robot description registered under ``ext/ompl_vamp/robot/*.json``.  All
# inputs (urdf/srdf) and outputs live inside the project; third_party
# submodules are only read from, never written.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

CRICKET_BIN="$ROOT/third_party/cricket/build/fkcc_gen"
OUTDIR="$ROOT/ext/ompl_vamp/robot"

if [ ! -x "$CRICKET_BIN" ]; then
    echo "[generate_fk] Building cricket first..." >&2
    cmake --build "$ROOT/third_party/cricket/build"
fi

# fkcc_gen writes its output into CWD using each config's "output"
# filename.  Run from the target dir so headers land there directly.
cd "$OUTDIR"

# Iterate over every robot config (skip cricket's intermediate
# ``output.json`` dump that gets re-emitted on every invocation).
for cfg in "$OUTDIR"/*.json; do
    name="$(basename "$cfg" .json)"
    if [ "$name" = "output" ]; then
        continue
    fi
    echo "[generate_fk] Generating FK header for $name..."
    "$CRICKET_BIN" "$cfg"
    echo "[generate_fk] Wrote $OUTDIR/$name.hh"
done

# Clean up cricket's per-run intermediate dump.
rm -f "$OUTDIR/output.json"
