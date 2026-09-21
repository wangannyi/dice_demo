#!/usr/bin/env bash
# Source this file from any directory; do not modify shell HOME.
if [ -n "${ZSH_VERSION:-}" ]; then
    DICE_ENV_SOURCE="${(%):-%x}"
else
    DICE_ENV_SOURCE="${BASH_SOURCE[0]}"
fi
DICE_ROOT="$(cd -- "$(dirname -- "$DICE_ENV_SOURCE")/.." && pwd)"
unset DICE_ENV_SOURCE
export DICE_ROOT
export DICE_VISION_PYTHON="${DICE_VISION_PYTHON:-$HOME/.venv-grasp/bin/python}"
export DICE_SDK_PYTHON="${DICE_SDK_PYTHON:-$HOME/agilex-api-test/venv/bin/python}"
export NERO_SDK_DIR="${NERO_SDK_DIR:-$HOME/agilex-api-test/pyAgxArm}"
export CALIB_PYTHON="${CALIB_PYTHON:-$DICE_VISION_PYTHON}"
export PYTHONPATH="$DICE_ROOT:$NERO_SDK_DIR:$DICE_ROOT/nero_calibration/.deps${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export QT_X11_NO_MITSHM=1
export PYTHONNOUSERSITE=1
