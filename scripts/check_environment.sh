#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
"$DICE_VISION_PYTHON" - <<'PY'
import sys, importlib, json, os
from pathlib import Path
root=Path(os.environ['DICE_ROOT'])
raw=json.loads((root/'configs/green_cup.json').read_text())
sys.path.append(raw['green_cup']['perception']['ort_package_dir'])
for name in ('numpy','scipy','cv2','pyrealsense2','onnxruntime'):
    m=importlib.import_module(name)
    print(name, getattr(m,'__version__','installed'))
import cv2
assert hasattr(cv2,'aruco'), 'OpenCV contrib/aruco required'
from cup_grasp_demo.flow.core import load_config
from cup_grasp_demo.flow.green_pipeline import validate
root=Path(os.environ['DICE_ROOT']); cfg=load_config(root/'configs/green_cup.json');validate(cfg)
for key in ('home','calibration','tcp_candidate','orientation_reference','grasp_config'):
    assert Path(cfg[key]).is_file(), cfg[key]
for key in ('reference','home_table_scene','joint_test_config'):
    assert (root/cfg['green_cup'][key]).is_file(), key
assert (root/cfg['green_cup']['perception']['model']).is_file()
from cup_grasp_demo.flow.core import Screen
Screen(table_only=True)
print('Config, model paths and geometry imports OK; no devices opened.')
PY
"$DICE_SDK_PYTHON" - <<'PY'
import can
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
print('SDK imports OK; no CAN connection opened.')
PY
