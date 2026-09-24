#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$DICE_VISION_PYTHON" - <<'PY'
import importlib, json, os, sys
from pathlib import Path
root=Path(os.environ['DICE_ROOT'])
raw=json.loads((root/'configs/green_cup.json').read_text())
for name in ('numpy','scipy','cv2','pyrealsense2','onnxruntime','can','wrapt','packaging','typing_extensions','pyAgxArm'):
    m=importlib.import_module(name)
    print(name, getattr(m,'__version__','installed'), Path(m.__file__).resolve())
import cv2
assert hasattr(cv2,'aruco'), 'OpenCV contrib/aruco required'
from cup_grasp_demo.flow.core import load_config
from cup_grasp_demo.flow.green_pipeline import validate
root=Path(os.environ['DICE_ROOT']); cfg=load_config(root/'configs/green_cup.json');validate(cfg)
for key in ('home','calibration','tcp_candidate','orientation_reference','grasp_config'):
    assert Path(cfg[key]).is_file(), cfg[key]
for key in ('reference','home_table_scene','joint_test_config'):
    assert (root/cfg['green_cup'][key]).is_file(), key
for relative in ('calibration/auto_collect.py',
                 'calibration/config/board_hand_redcloth.json',
                 'calibration/config/board_hand_redcloth_cover_fixed.json',
                 'calibration/config/board_reference_redcloth.json'):
    assert (root/relative).is_file(), relative
assert (root/cfg['green_cup']['perception']['model']).is_file()
sdk=(root/'third_party/pyAgxArm/pyAgxArm').resolve()
assert sdk.is_dir(), sdk
assert Path(importlib.import_module('pyAgxArm').__file__).resolve().is_relative_to(sdk)
from cup_grasp_demo.flow.core import Screen
Screen(table_only=True)
print('Config, model paths and geometry imports OK; no devices opened.')
PY
if [[ "$DICE_SDK_PYTHON" != "$DICE_VISION_PYTHON" ]]; then
  "$DICE_SDK_PYTHON" - <<'PY'
import can
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
print('SDK imports OK; no CAN connection opened.')
PY
fi
