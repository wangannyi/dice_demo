#!/bin/sh
set -eu
GRASP_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
GRASP_ROOT=$(dirname -- "$GRASP_DIR")
GRASP_PYTHON=${GRASP_PYTHON:-"$HOME/.venv-grasp/bin/python"}
NERO_SDK_DIR=${NERO_SDK_DIR:-"$HOME/agilex-api-test/pyAgxArm"}
export PYTHONPATH="$GRASP_ROOT:$GRASP_DIR/.deps:$NERO_SDK_DIR${PYTHONPATH:+:$PYTHONPATH}"
if [ "${1:-}" = "execute" ]; then
  shift
  exec "$GRASP_PYTHON" "$GRASP_DIR/execute.py" "$@"
fi
if [ "${1:-}" = "pipeline" ]; then
  shift
  NERO_CONTROL_PYTHON=${NERO_CONTROL_PYTHON:-"$HOME/agilex-api-test/venv/bin/python"}
  exec "$GRASP_PYTHON" -m cup_grasp_demo.pipeline \
    --python "$NERO_CONTROL_PYTHON" --camera-python "$GRASP_PYTHON" "$@"
fi
exec "$GRASP_PYTHON" "$GRASP_DIR/grasp.py" "$@"
