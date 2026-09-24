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
# Use the host's Python by default.  A deployment can override DICE_PYTHON,
# but the repository never searches user-specific virtualenv locations.
_python_default="$(command -v python3 || true)"
if [[ -z "$_python_default" ]]; then
    echo "python3 was not found in PATH" >&2
    return 1 2>/dev/null || exit 1
fi
export DICE_PYTHON="${DICE_PYTHON:-$_python_default}"
export DICE_VISION_PYTHON="${DICE_VISION_PYTHON:-$DICE_PYTHON}"
export DICE_SDK_PYTHON="${DICE_SDK_PYTHON:-$DICE_PYTHON}"
export NERO_SDK_DIR="${NERO_SDK_DIR:-$DICE_ROOT/third_party/pyAgxArm}"
export CALIB_PYTHON="${CALIB_PYTHON:-$DICE_VISION_PYTHON}"
_python_paths=("$DICE_ROOT" "$NERO_SDK_DIR")
[[ -d "$DICE_ROOT/third_party/python" ]] && _python_paths+=("$DICE_ROOT/third_party/python")
[[ -n "${DICE_PYTHON_EXTRA:-}" ]] && _python_paths+=("$DICE_PYTHON_EXTRA")
_joined_path="$(IFS=:; echo "${_python_paths[*]}")"
export PYTHONPATH="$_joined_path${PYTHONPATH:+:$PYTHONPATH}"
unset _python_default _python_paths _joined_path
export OPENBLAS_NUM_THREADS=1
export QT_X11_NO_MITSHM=1
export PYTHONNOUSERSITE=1
