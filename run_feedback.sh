#!/usr/bin/env bash
set -euo pipefail
# 猜拳预备位置：./run_feedback.sh rps-ready（预览）
# 执行动作：./run_feedback.sh rps-ready --execute
# 交互菜单：./run_feedback.sh --execute
FEEDBACK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$FEEDBACK_ROOT/scripts/env.sh"
exec "$DICE_VISION_PYTHON" "$FEEDBACK_ROOT/scripts/result_feedback.py" "$@"
