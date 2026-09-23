#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
source scripts/env.sh
exec "$CALIB_PYTHON" calibration/workflow.py "$@"
