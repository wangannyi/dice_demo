#!/usr/bin/env bash
set -euo pipefail
FEEDBACK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FEEDBACK_ROOT/scripts/env.sh"
exec "$DICE_VISION_PYTHON" "$FEEDBACK_ROOT/scripts/result_feedback.py" "$@"
