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
# Board layout from the original test2 environment uses two dedicated venvs
# under $HOME. On boards without that layout (e.g. main K3), fall back to the
# system python and the repo-local vendored dependencies: vendor-site/ carries
# pyrealsense2 + pyAgxArm, nero_calibration/.deps/ carries python-can.
_vision_default="$HOME/.venv-grasp/bin/python"
[[ -x "$_vision_default" ]] || _vision_default="/usr/bin/python3"
export DICE_VISION_PYTHON="${DICE_VISION_PYTHON:-$_vision_default}"
_sdk_default="$HOME/agilex-api-test/venv/bin/python"
[[ -x "$_sdk_default" ]] || _sdk_default="/usr/bin/python3"
export DICE_SDK_PYTHON="${DICE_SDK_PYTHON:-$_sdk_default}"
_sdk_dir_default="$HOME/agilex-api-test/pyAgxArm"
[[ -d "$_sdk_dir_default" ]] || _sdk_dir_default="$DICE_ROOT/vendor-site/pyAgxArm"
export NERO_SDK_DIR="${NERO_SDK_DIR:-$_sdk_dir_default}"
export CALIB_PYTHON="${CALIB_PYTHON:-$DICE_VISION_PYTHON}"
_vendor_site=""
[[ -d "$DICE_ROOT/vendor-site" ]] && _vendor_site=":$DICE_ROOT/vendor-site"
export PYTHONPATH="$DICE_ROOT:$NERO_SDK_DIR:$DICE_ROOT/nero_calibration/.deps$_vendor_site${PYTHONPATH:+:$PYTHONPATH}"
unset _vision_default _sdk_default _sdk_dir_default _vendor_site
export OPENBLAS_NUM_THREADS=1
export QT_X11_NO_MITSHM=1
export PYTHONNOUSERSITE=1
