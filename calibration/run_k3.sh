#!/usr/bin/env bash
set -euo pipefail
CALIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$CALIB_DIR/../scripts/env.sh"
exec "$CALIB_PYTHON" "$CALIB_DIR/calibrate.py" "$@"
