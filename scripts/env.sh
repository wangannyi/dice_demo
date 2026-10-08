#!/usr/bin/env bash
# Source this file from any directory; do not modify shell HOME.
# `source scripts/env.sh --system` discards stale project/virtualenv overrides.
_dice_system_reset=0
if [[ "${1:-}" == "--system" ]]; then
    if type deactivate >/dev/null 2>&1; then
        deactivate
    fi
    unset DICE_PYTHON DICE_VISION_PYTHON DICE_SDK_PYTHON CALIB_PYTHON
    unset NERO_SDK_DIR DICE_PYTHON_EXTRA PYTHONPATH PYTHONHOME
    _dice_system_reset=1
fi
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
if [[ "$_dice_system_reset" == 1 && -x /usr/bin/python3 ]]; then
    _python_default=/usr/bin/python3
else
    _python_default="$(command -v python3 || true)"
fi
if [[ -z "$_python_default" ]]; then
    echo "python3 was not found in PATH" >&2
    return 1 2>/dev/null || exit 1
fi
export DICE_PYTHON="${DICE_PYTHON:-$_python_default}"
export DICE_VISION_PYTHON="${DICE_VISION_PYTHON:-$DICE_PYTHON}"
export DICE_SDK_PYTHON="${DICE_SDK_PYTHON:-$DICE_PYTHON}"
export NERO_SDK_DIR="${NERO_SDK_DIR:-$DICE_ROOT/third_party/pyAgxArm}"
export CALIB_PYTHON="${CALIB_PYTHON:-$DICE_VISION_PYTHON}"
if ! _dice_locked_env="$(/usr/bin/python3 "$DICE_ROOT/scripts/k3_runtime.py" shell)"; then
    return 1 2>/dev/null || exit 1
fi
# The helper emits only shlex-quoted exports from the verified dependency lock.
eval "$_dice_locked_env"
unset _dice_locked_env
_python_paths=("$DICE_ROOT" "$NERO_SDK_DIR")
[[ -n "${DICE_LOCKED_PYTHON:-}" ]] && _python_paths=("$DICE_LOCKED_PYTHON" "${_python_paths[@]}")
[[ -d "$DICE_ROOT/third_party/python" ]] && _python_paths+=("$DICE_ROOT/third_party/python")
[[ -n "${DICE_PYTHON_EXTRA:-}" ]] && _python_paths+=("$DICE_PYTHON_EXTRA")
_joined_path="$(IFS=:; echo "${_python_paths[*]}")"
export PYTHONPATH="$_joined_path${PYTHONPATH:+:$PYTHONPATH}"
unset _dice_system_reset _python_default _python_paths _joined_path
export OPENBLAS_NUM_THREADS=1
export QT_X11_NO_MITSHM=1
export PYTHONNOUSERSITE=1
# Resolve with the selected interpreter before it starts loading native code.
# The alias directory binds the EP's generic libonnxruntime.so dependency to
# the exact core bundled with Python ORT, ahead of stale /usr/local libraries.
if ! _dice_native_paths="$("$DICE_VISION_PYTHON" "$DICE_ROOT/scripts/runtime_library_path.py")"; then
    echo "Unable to select matching K3 inference libraries" >&2
    unset _dice_native_paths
    return 1 2>/dev/null || exit 1
fi
if [[ -n "$_dice_native_paths" ]]; then
    _dice_native_merged="$_dice_native_paths"
    _dice_native_old="${LD_LIBRARY_PATH:-}"
    while [[ -n "$_dice_native_old" ]]; do
        _dice_native_entry="${_dice_native_old%%:*}"
        if [[ ":$_dice_native_merged:" != *":$_dice_native_entry:"* && -n "$_dice_native_entry" ]]; then
            _dice_native_merged="$_dice_native_merged:$_dice_native_entry"
        fi
        if [[ "$_dice_native_old" == *:* ]]; then
            _dice_native_old="${_dice_native_old#*:}"
        else
            _dice_native_old=""
        fi
    done
    export LD_LIBRARY_PATH="$_dice_native_merged"
fi
unset _dice_native_paths _dice_native_merged _dice_native_old _dice_native_entry
