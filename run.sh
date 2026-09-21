#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$DICE_ROOT"
if [[ ${1:-} == '--help' || $# == 0 ]]; then
  echo 'Usage: ./run.sh {step|auto|fast} [--show] [--execute] [--until place]'
  echo 'DICE_CONFIG overrides configs/green_cup.json; DICE_RUN overrides datasets/green_current.'
  echo 'Without --execute, only the flow is printed. No hardware is opened.'
  exit 0
fi
mode=$1; shift
case "$mode" in step|auto|fast) ;; *) echo 'Mode must be step, auto or fast' >&2; exit 2;; esac
exec "$DICE_ROOT/cup_grasp_demo/calibration_debug/run_debug.sh" pipeline \
  --config "${DICE_CONFIG:-$DICE_ROOT/configs/green_cup.json}" \
  --session "${DICE_RUN:-$DICE_ROOT/cup_grasp_demo/datasets/green_current}" \
  --mode "$mode" --until place "$@"
