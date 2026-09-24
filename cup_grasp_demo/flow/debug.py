"""Operator-run flange/TCP comparison. Default commands never move the arm."""

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if __name__ == '__main__':
    from cup_grasp_demo.flow.green_startup import launch
    launch(sys.argv[1:])

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cup_grasp_demo.flow.core import (  # noqa: E402
    digest, load_config, read_json, write_json,
)
from cup_grasp_demo.flow.contact_geometry import contact_direction  # noqa: E402
from cup_grasp_demo.flow.session_storage import dispatch  # noqa: E402
from cup_grasp_demo.planning import load_calibration  # noqa: E402
from vision.capture.frame_io import load_batch  # noqa: E402
from vision.geometry.table_plane import Config, _plane, deproject  # noqa: E402
from cup_grasp_demo.flow.transforms import inverse, matrix, pose_matrix  # noqa: E402
from nero_revo2_control.kinematics import load_model  # noqa: E402

HERE = Path(__file__).resolve().parent


def new_run(session, name):
    directory = Path(session) / 'runs' / (time.strftime('%Y%m%d_%H%M%S') + '_' + name + '_' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def bridge(command, output, cfg, request=None):
    python = os.environ.get('DICE_SDK_PYTHON', sys.executable)
    argv = [python, str(HERE / 'hardware.py'), command, '--channel', cfg['channel'],
            '--output', str(output)]
    if request:
        argv += ['--request', str(request), '--sha256', digest(request)]
    with Path(output).with_suffix('.log').open('x') as log:
        proc = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=log, check=False)
    if proc.returncode:
        detail = read_json(output).get('error') if Path(output).exists() else 'SDK bridge failed'
        raise RuntimeError(f'{detail}; 日志：{Path(output).with_suffix(".log")}')
    return read_json(output)


def capture_rgbd(output, cfg, *, frames=5):
    from cup_grasp_demo.flow.green_capture import capture_arguments
    extra = capture_arguments(cfg)
    subprocess.run([sys.executable, str(ROOT / 'vision/capture/realsense_session.py'),
                    '--serial', cfg['serial'], '--frames', str(frames), '--output', str(output), *extra],
                   cwd=ROOT, check=True)


def ready(snapshot):
    if (snapshot.get('arm_status') != 0 or snapshot.get('motion_status', 0) != 0
            or snapshot.get('joints_enabled') != [True] * 7
            or snapshot.get('ctrl_mode') not in (1, 3)):
        raise ValueError('机械臂必须停稳、全部使能且无错误')


def show(path, enabled, *, timeout_s=None):
    print(f'图像：{path}', flush=True)
    if enabled:
        if not os.environ.get('DISPLAY'):
            raise RuntimeError('没有 DISPLAY；请从 PC 使用 ssh -X 登录，或去掉 --show 查看保存图像')
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot load preview image')
        title = 'Dice calibration debug - press any key'
        if timeout_s is None:
            cv2.imshow(title, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Preview timeout must be positive')
        title = 'Dice preview - 3s; Esc/Q exits'
        try:
            cv2.imshow(title, image)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                key = cv2.waitKey(50)
                if key >= 0:
                    if (key & 0xff) in (27, ord('q'), ord('Q')):
                        raise InterruptedError('用户退出图像预览，pipeline 已停止')
                    break
                try:
                    visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
                except cv2.error as exc:
                    # Qt destroys guiReceiver when the user closes the last window.
                    if 'NULL guiReceiver' in str(exc) or 'NULL window' in str(exc):
                        break
                    raise
                if visible < 1:
                    break
        except cv2.error as exc:
            raise RuntimeError(f'图像预览失败，pipeline 已停止：{exc}') from exc
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error as exc:
                if 'NULL guiReceiver' not in str(exc) and 'NULL window' not in str(exc):
                    raise RuntimeError(f'图像窗口清理失败：{exc}') from exc


def camera_transform(meta, cfg):
    if meta['serial'] != cfg['serial']:
        raise ValueError('Wrong camera serial')
    calibration = load_calibration(cfg['calibration'], expected_camera_serial=meta['serial'],
                                   allow_provisional=cfg['allow_provisional_calibration'])
    intr = meta['intrinsics']
    k = [[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]]
    camera = calibration['camera']
    if ((intr['width'], intr['height']) != (camera['width'], camera['height'])
            or not np.allclose(k, camera['camera_matrix'], atol=1e-4, rtol=0)):
        raise ValueError('Camera intrinsics differ from calibration')
    return matrix(calibration['T_base_camera']), calibration['quality_passed']


def validate_plan(plan, session_dir, cfg, now=None):
    if plan.get('kind') != 'flange_tcp_debug_plan' or plan.get('offline_only'):
        raise ValueError('Only a live Debug plan can execute; replay plans are read-only')
    if plan.get('blockers') or not plan.get('screen_passed'):
        raise ValueError('Plan is blocked')
    if digest(session_dir / 'session.json') != plan['session_sha256']:
        raise ValueError('Session changed after planning')
    age = (time.time() if now is None else now) - plan['created_epoch_s']
    if not 0 <= age <= cfg['plan_max_age_s']:
        raise ValueError('计划过期，请用当前关节位置重新 plan')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('green-detect', help='只采集和定位绿色杯；不连接 CAN、不运动')
    p.add_argument('--config', type=Path, default=HERE / 'green_open_cup/config.json')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('pipeline', help='绿杯抓取与摇晃；fast 连续、control 常驻')
    p.add_argument('--config', type=Path, default=ROOT / 'configs/green_cup.json')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--mode', choices=('step', 'auto', 'fast', 'control'), default='step')
    p.add_argument('--until', choices=('ready', 'grip', 'shake-plan', 'shake', 'place'),
                   help='默认 step/auto 到 grip，fast 到 shake')
    p.add_argument('--execute', action='store_true', help='允许执行；省略时只显示流程，不访问硬件')
    p.add_argument('--resume', action='store_true', help='继续正常暂停的流程；失败或中断后不自动重放')
    p.add_argument('--status', action='store_true', help='只读查看最近一次流程状态')
    p.add_argument('--show', action='store_true', help='仅 STEP 模式弹出 X11 图像；auto/fast 只保存图像')
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == 'green-detect':
        if load_config(args.config).get('pipeline_strategy') != 'green_open_cup':
            raise ValueError('green-detect requires green_open_cup configuration')
        from cup_grasp_demo.flow.green_pipeline import Workflow
        args.mode = 'fast'
        def green_detect_only(args):
            flow = Workflow(args)
            flow.capture()
            print('绿色杯定位完成；未连接 CAN、未发送运动指令')
            return 0
        return dispatch(args, green_detect_only)
    if args.command == 'pipeline':
        if load_config(args.config).get('pipeline_strategy') == 'green_open_cup':
            if args.mode in ('step', 'auto'):
                raise ValueError('绿色杯流程仅支持 fast/control（step/auto 已移除）')
            if args.until is None:
                args.until = 'place'
            if args.mode == 'control':
                from cup_grasp_demo.flow.green_control import run
                return run(args)
            from cup_grasp_demo.flow.green_pipeline import run
            return run(args)
        raise ValueError('本仓库只保留绿杯流程（green_open_cup）')



if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'DEBUG STOP: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print('已中断；若正在运动，请检查 SDK 日志及实际保持状态。', file=sys.stderr)
        raise SystemExit(130)
