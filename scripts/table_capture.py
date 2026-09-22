#!/usr/bin/env python3
"""独立桌面采集（替代旧 run_planar_shake.sh table-capture）。

仅采集红布桌面平面并写入 planar_table_scene.json，不识别杯子、不移动机械臂。
配合 scripts/register_home_table.py 完成桌面登记（见 docs/CALIBRATION.md）。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cup_grasp_demo.flow import planar_scene
from cup_grasp_demo.flow.core import load_config
from cup_grasp_demo.flow.session_storage import session_lock
from types import SimpleNamespace


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    args = parser.parse_args(argv)
    load_config(args.config)  # 校验配置
    with session_lock(args.session):
        ns = SimpleNamespace(config=args.config, session=args.session)
        return planar_scene.capture(ns) or 0


if __name__ == '__main__':
    raise SystemExit(main())
