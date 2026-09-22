#!/usr/bin/env bash
set -euo pipefail
DEBUG_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export OPENBLAS_NUM_THREADS=1
export QT_X11_NO_MITSHM=1
export PYTHONNOUSERSITE=1
if [[ ${1:-} == pipeline ]]; then
  execute=0
  control_mode=0
  previous=
  for argument in "$@"; do
    if [[ $argument == --execute ]]; then
      execute=1
    fi
    if [[ $previous == --mode && $argument == control ]]; then
      control_mode=1
    fi
    previous=$argument
  done
  if (( execute )); then
    if (( control_mode )); then
      bash "$DEBUG_ROOT/scripts/ensure_can_link.sh" >&2
    else
      bash "$DEBUG_ROOT/scripts/ensure_can_link.sh"
    fi
  fi
fi
exec "${DICE_VISION_PYTHON:-/usr/bin/python3}" \
  "$DEBUG_ROOT/cup_grasp_demo/calibration_debug/debug.py" "$@"
