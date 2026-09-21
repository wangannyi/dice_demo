#!/usr/bin/env bash
set -euo pipefail
JOINT_TEST_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPENBLAS_NUM_THREADS=1
export PYTHONNOUSERSITE=1
exec "${DICE_VISION_PYTHON:-/home/test2/.venv-grasp/bin/python}" \
  "$JOINT_TEST_ROOT/cup_grasp_demo/calibration_debug/joint_test.py" "$@"
