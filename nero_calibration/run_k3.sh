#!/bin/sh
set -eu
CALIB_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CALIB_PYTHON=${CALIB_PYTHON:-"$HOME/.venv-grasp/bin/python"}
NERO_SDK_DIR=${NERO_SDK_DIR:-"$HOME/agilex-api-test/pyAgxArm"}
export PYTHONPATH="$CALIB_DIR/.deps:$NERO_SDK_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec "$CALIB_PYTHON" "$CALIB_DIR/calibrate.py" "$@"
