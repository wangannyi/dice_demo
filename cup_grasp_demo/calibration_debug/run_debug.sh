#!/usr/bin/env bash
set -euo pipefail
DEBUG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPENBLAS_NUM_THREADS=1
export QT_X11_NO_MITSHM=1
export PYTHONNOUSERSITE=1
exec "${DICE_VISION_PYTHON:-/home/test2/.venv-grasp/bin/python}" \
  "$DEBUG_ROOT/cup_grasp_demo/calibration_debug/debug.py" "$@"
