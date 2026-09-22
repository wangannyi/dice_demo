#!/usr/bin/env bash
set -euo pipefail
PLANAR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
exec "${DICE_VISION_PYTHON:-/usr/bin/python3}" \
  "$PLANAR_ROOT/cup_grasp_demo/flow/planar_shake_cli.py" "$@"
