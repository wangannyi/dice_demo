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
    from cup_grasp_demo.calibration_debug.green_startup import launch
    launch(sys.argv[1:])

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cup_grasp_demo.calibration_debug.core import (  # noqa: E402
    Screen, configured_tcp, digest, load_config, make_plan, measured_offset, read_json, write_json,
)
from cup_grasp_demo.calibration_debug.contact_geometry import contact_direction  # noqa: E402
from cup_grasp_demo.calibration_debug.cup_recheck import verify_cup  # noqa: E402
from cup_grasp_demo.calibration_debug.cup_selection import select_cup  # noqa: E402
from cup_grasp_demo.calibration_debug.tcp_overlay import draw_tcp  # noqa: E402
from cup_grasp_demo.calibration_debug.session_storage import (  # noqa: E402
    CAPTURE_PENDING, dispatch, prepare_plan, reset_capture,
)
from cup_grasp_demo.hand_geometry import RightRevo2Model  # noqa: E402
from cup_grasp_demo.planning import load_calibration  # noqa: E402
from cup_grasp_demo.side_grasp.preview_index import load_batch  # noqa: E402
from dice_cup_localization.geometry import Config, _plane, deproject  # noqa: E402
from nero_calibration.core import inverse, matrix, pose_matrix  # noqa: E402
from nero_revo2_control.kinematics import load_model  # noqa: E402

HERE = Path(__file__).resolve().parent


def new_run(session, name):
    directory = Path(session) / 'runs' / (time.strftime('%Y%m%d_%H%M%S') + '_' + name + '_' + uuid.uuid4().hex[:6])
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def bridge(command, output, cfg, request=None):
    python = os.environ.get('DICE_SDK_PYTHON', '/home/test2/agilex-api-test/venv/bin/python')
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
    from cup_grasp_demo.calibration_debug.green_capture import capture_arguments
    extra = capture_arguments(cfg)
    subprocess.run([sys.executable, str(ROOT / 'dice_cup_localization/capture_rgbd.py'),
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


def target_overlay(image_path, output, session, point, snapshot, hand_state='unknown'):
    """Draw the current feedback-derived TCP separately from the frozen target."""
    image, report = draw_tcp(cv2.imread(str(image_path)), session, snapshot, point, hand_state)
    if not cv2.imwrite(str(output), image):
        raise OSError('Could not write TCP overlay')
    write_json(Path(output).with_suffix('.json'), report)


def capture_with_feedback(output, cfg, *, bridge_fn=None, capture_fn=None):
    """Bracket the camera batch with read-only joint feedback; require a stopped arm."""
    bridge_fn = bridge if bridge_fn is None else bridge_fn
    capture_fn = capture_rgbd if capture_fn is None else capture_fn
    before = bridge_fn('snapshot', output.with_name(output.name + '_before.json'), cfg)
    ready(before)
    capture_fn(output, cfg)
    after = bridge_fn('snapshot', output.with_name(output.name + '_after.json'), cfg)
    ready(after)
    q0, q1 = np.asarray(before['joints_rad']), np.asarray(after['joints_rad'])
    if (q0.shape != (7,) or q1.shape != (7,) or not np.isfinite([q0, q1]).all()
            or np.max(np.abs(q1 - q0)) > math.radians(.2)):
        raise ValueError('采样时关节反馈无效或机械臂发生运动，不能标注当前 TCP')
    return after


def check_overlay_camera(image_path, session):
    meta = read_json(Path(image_path).with_suffix('.json'))
    if meta['serial'] != session['config']['serial'] or meta['intrinsics'] != session['intrinsics']:
        raise ValueError('当前相机或内参与冻结会话不同，不能投影 TCP')


def tcp_view(args):
    """Read-only view at any stopped arm pose; no cup localization or motion."""
    if args.session:
        session, cfg = verify_session(args.session)
        if session['replay_only']:
            raise ValueError('实时 TCP 查看不能使用离线回放会话')
    else:
        cfg = load_config(args.config)
        session = None
    run = new_run(args.output or args.session or ROOT / 'cup_grasp_demo/datasets', 'tcp_view')
    feedback = capture_with_feedback(run / 'rgbd', cfg)
    meta = read_json(run / 'rgbd/frame_000.json')
    cam, _ = camera_transform(meta, cfg)
    if session is None:
        session = dict(T_base_camera=cam.tolist(), intrinsics=meta['intrinsics'], config=cfg,
                       T_flange_tcp=configured_tcp(cfg).tolist())
    check_overlay_camera(run / 'rgbd/frame_000.png', session)
    target_overlay(run / 'rgbd/frame_000.png', run / 'tcp_view.png', session,
                   session.get('contact_base_m'), feedback)
    print('紫圈为实时关节反馈计算的全张开手形 TCP；闭手后仅作参考，不代表实际食指点。')
    show(run / 'tcp_view.png', args.show)
    return 0


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


def source_hashes(cfg):
    from cup_grasp_demo.calibration_debug.cup_perception import model_path
    hand = RightRevo2Model()
    paths = [Path(cfg[key]) for key in ('home', 'calibration', 'tcp_candidate', 'orientation_reference', 'grasp_config')]
    paths += [Path(x['path']) for x in hand.provenance['model_sources']]
    paths += [item[-1] for item in hand.collision.values()]
    description = ROOT / 'agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero'
    paths += list((description / 'meshes').glob('link*.stl'))
    paths += [description / 'meshes/revo2_flange.stl', description / 'urdf/nero_description.urdf']
    paths += [ROOT / 'nero_revo2_control/models/nero_description.urdf']
    cup_model = model_path(cfg)
    if cup_model is not None:
        paths.append(cup_model)
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def verify_session(directory):
    if (directory / CAPTURE_PENDING).exists():
        raise ValueError('本 RUN 最新采集尚未成功，请重新 capture 后再规划')
    session = read_json(directory / 'session.json')
    for filename, expected in session['source_hashes'].items():
        if digest(ROOT / filename) != expected:
            raise ValueError(f'参数/模型已变更，请重新 capture：{filename}')
    for filename, expected in session['capture_hashes'].items():
        if digest(directory / filename) != expected:
            raise ValueError(f'采样数据已变更：{filename}')
    cfg = dict(session['config'])
    for key in ('home', 'calibration', 'tcp_candidate', 'orientation_reference', 'grasp_config'):
        cfg[key] = str(ROOT / cfg[key])
    return session, cfg


def capture(args, *, prepared=None, snapshot=None):
    reset_capture(args.session, args.replay)
    cfg = load_config(args.config)
    if args.replay:
        shutil.copytree(args.replay / 'rgbd', args.session / 'rgbd')
        before = after = json.loads((args.replay / 'joints_after.jsonl').read_text().splitlines()[-1])
    else:
        if snapshot is not None and 0 <= time.time() - snapshot.get('observed_epoch_s', 0) <= 2:
            before = snapshot
            write_json(args.session / 'before.json', before)
        else:
            before = bridge('snapshot', args.session / 'before.json', cfg)
        ready(before)
        home = read_json(cfg['home'])['joints_rad']
        if max(abs(a - b) for a, b in zip(home, before['joints_rad'])) > math.radians(1):
            raise ValueError('先回保存的 HOME，再采集杯位')
        capture_started_monotonic_s = time.monotonic()
        capture_rgbd(args.session / 'rgbd', cfg)
        after = bridge('snapshot', args.session / 'after.json', cfg)
    ready(after)
    if max(abs(a - b) for a, b in zip(before['joints_rad'], after['joints_rad'])) > math.radians(.2):
        raise ValueError('采样时机械臂发生运动')
    meta, depth, image, _ = load_batch(args.session)
    cam, quality = camera_transform(meta, cfg)
    try:
        geom, mask, points = select_cup(depth, image, meta, cfg, output=args.session)
    except ValueError:
        preview = args.session / 'cup_candidates.png'
        if preview.exists():
            show(preview, args.show)
        raise
    arm = load_model()
    ref_q = read_json(cfg['orientation_reference'])['joints_rad']
    rot = np.array(arm.fk(ref_q))[:3, :3]
    candidate = read_json(cfg['tcp_candidate'])
    tcp = configured_tcp(cfg)
    hand = RightRevo2Model()
    link = hand.link_transforms_from_flange(np.eye(4))[candidate['link']]
    palm_normal = link[:3, 0]
    normal = cam[:3, :3] @ geom['normal_camera']
    center = cam[:3, :3] @ geom['center_camera_m'] + cam[:3, 3]
    support = center - geom['contact_height_m'] * normal
    outward, rot, direction_info = contact_direction(rot, palm_normal, normal, cfg.get('contact_direction'))
    contact = center + geom['radius_m'] * outward
    delta = points - np.array(geom['center_camera_m'])
    nc = np.array(geom['normal_camera'])
    radius = float(np.linalg.norm(delta - (delta @ nc)[:, None] * nc, axis=1).max())
    scene = dict(cup_support_base_m=support.tolist(), cup_normal_base=normal.tolist(),
                 outward_base=outward.tolist(), cup_envelope_radius_m=radius, geometry=geom)
    if direction_info is not None:
        scene['contact_direction'] = direction_info
    overlay = image.copy()
    overlay[mask] = (.6 * overlay[mask] + [0, 72, 0]).astype('uint8')
    inv = inverse(cam)
    if direction_info is not None:
        intr = meta['intrinsics']
        ends = []
        for point in (center - geom['radius_m'] * outward, contact):
            p = (inv @ np.r_[point, 1])[:3]
            ends.append(tuple(np.rint([intr['fx'] * p[0] / p[2] + intr['cx'],
                                      intr['fy'] * p[1] / p[2] + intr['cy']]).astype(int)))
        cv2.line(overlay, ends[0], ends[1], (255, 180, 0), 2)
    side = cfg.get('side_grasp', {})
    gap = float(side['close_gap_mm']) if side.get('strategy') == 'direct_close' else 50.
    for label, point, color in [('contact', contact, (0, 0, 255)),
                                (f'outside{gap:g}', contact + gap / 1000 * outward, (255, 255, 0))]:
        p = (inv @ np.r_[point, 1])[:3]
        intr = meta['intrinsics']
        uv = tuple(np.rint([intr['fx'] * p[0] / p[2] + intr['cx'],
                           intr['fy'] * p[1] / p[2] + intr['cy']]).astype(int))
        cv2.circle(overlay, uv, 5, color, -1)
        cv2.putText(overlay, label, (uv[0] + 6, uv[1]), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
    height_label = Fraction(cfg['contact_height_fraction']).limit_denominator(100)
    cv2.putText(overlay, f'DEBUG {height_label}H - verify silver cup', (10, 465),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
    portable = dict(cfg)
    for key in ('home', 'calibration', 'tcp_candidate', 'orientation_reference', 'grasp_config'):
        portable[key] = str(Path(cfg[key]).relative_to(ROOT))
    session = dict(schema=1, kind='frozen_flange_tcp_comparison', capture_id=uuid.uuid4().hex,
                   captured_epoch_s=time.time(),
                   replay_only=bool(args.replay), config=portable, source_hashes=source_hashes(cfg),
                   capture_hashes={str(p.relative_to(args.session)): digest(p)
                                   for p in sorted((args.session / 'rgbd').glob('*'))},
                   captured_q_rad=after['joints_rad'], contact_base_m=contact.tolist(),
                   T_base_camera=cam.tolist(), intrinsics=meta['intrinsics'],
                   R_base_flange=rot.tolist(), T_flange_tcp=tcp.tolist(), scene=scene,
                   calibration_quality_passed=quality, physical_tcp_verified=False,
                   segmentation=('YOLO-seg cap mask + registered depth in red workspace'
                                 if geom.get('segmentation') else
                                 'Depth geometry in red workspace; operator verifies silver cup'))
    overlay, tcp_report = draw_tcp(overlay, session, after)
    tcp_report['replay_only'] = bool(args.replay)
    if args.replay:
        cv2.putText(overlay, 'REPLAY - saved feedback, NOT live', (8, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)
    if not cv2.imwrite(str(args.session / 'target.png'), overlay):
        raise OSError('Could not write target overlay')
    write_json(args.session / 'target_tcp.json', tcp_report)
    write_json(args.session / 'session.json', session)
    if prepared is not None:
        from cup_grasp_demo.calibration_debug.grasp import scene_from_points
        prepared['scene'] = scene_from_points(session, cfg, geom, points)
        if not args.replay:
            prepared['capture_proof'] = dict(session_sha256=digest(args.session / 'session.json'),
                                            started_monotonic_s=capture_started_monotonic_s)
    (args.session / CAPTURE_PENDING).unlink()
    print(json.dumps(dict(contact_base_mm=(contact * 1000).tolist(),
                          cup_height_mm=geom['height_m'] * 1000,
                          contact_height_mm=geom['contact_height_m'] * 1000,
                          contact_direction=direction_info,
                          tcp_flange_mm=(tcp[:3, 3] * 1000).tolist()), indent=2))
    show(args.session / 'target.png', args.show)


def plan(args):
    prepare_plan(args.output, 'flange_tcp_debug_plan')
    session, cfg = verify_session(args.session)
    run = new_run(args.session, 'plan_' + args.frame)
    if args.joints_json:
        raw = args.joints_json.read_text().splitlines()
        try:
            snapshot = json.loads('\n'.join(raw))
        except json.JSONDecodeError:
            snapshot = json.loads(raw[-1])
    else:
        snapshot = bridge('snapshot', run / 'snapshot.json', cfg)
    ready(snapshot)
    result = make_plan(session, snapshot['joints_rad'], cfg, args.frame, args.gap_mm, args.cup_removed)
    result.update(session_path=str(args.session.resolve()), session_sha256=digest(args.session / 'session.json'),
                  created_epoch_s=time.time(), offline_only=bool(args.joints_json) or session['replay_only'])
    write_json(args.output, result)
    summarize(result)
    print(f'计划：{args.output}')
    return 2 if result['blockers'] else 0


def summarize(plan):
    print(json.dumps({k: plan[k] for k in ('frame', 'gap_mm', 'comparison_point_base_m',
                                          'T_base_flange_target', 'blockers')}, indent=2))
    print(f"轨迹分段：{len(plan['stages'])}；screen_passed={plan['screen_passed']}；速度由会话配置决定")


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


def execute_plan(args):
    planned = read_json(args.plan)
    directory = Path(planned['session_path'])
    session, cfg = verify_session(directory)
    validate_plan(planned, directory, cfg)
    # Recompute rather than trusting mutable joint arrays in a JSON file.
    rebuilt = make_plan(session, planned['start_q_rad'], cfg, planned['frame'],
                        planned['gap_mm'], planned['cup_removed'])
    if (rebuilt['blockers'] or any(json.dumps(value, sort_keys=True)
                                  != json.dumps(planned.get(key), sort_keys=True)
                                  for key, value in rebuilt.items())):
        raise ValueError('计划内容或计算结果改变，请重新 plan')
    summarize(planned)
    if not args.execute:
        print('仅检查计划；加 --execute 才进入现场确认和运动')
        return 0
    if session['replay_only']:
        raise ValueError('Replay session cannot execute')
    run = new_run(directory, 'execute_' + planned['frame'])
    before = capture_with_feedback(run / 'rgbd', cfg)
    check_overlay_camera(run / 'rgbd/frame_000.png', session)
    if not planned['cup_removed']:
        # While the cup is still present, reject changed or occluded targets.
        meta, depth, image, _ = load_batch(run)
        camera_transform(meta, cfg)
        verify_cup(depth, image, meta, cfg, session['scene']['geometry'], run)
    target_overlay(run / ('rgbd/frame_000.png' if planned['cup_removed'] else 'cup_recheck.png'),
                   run / 'before_target.png', session, planned['comparison_point_base_m'], before)
    show(run / 'before_target.png', args.show)
    print('确认：相机/基座/桌面未移动；手指全张开；机械臂通路无障碍。')
    print('目标杯已经移开，保留原目标空间。' if planned['cup_removed']
          else '目标杯保持采样时的位置，程序仅移动到外侧偏置点。')
    print('运动期间不要操作 WEB 控制。异常时 Ctrl-C；程序将尝试保持当前关节位置。')
    validate_plan(planned, directory, cfg)
    if time.time() - (run / 'rgbd/frame_000.png').stat().st_mtime > 180:
        raise ValueError('图像已过期，请重新执行 move')
    request = dict(execution_authorized=True, authorized_epoch_s=time.time(), plan=planned, config=cfg)
    write_json(run / 'request.json', request)
    actual = bridge('run', run / 'actual.json', cfg, run / 'request.json')
    after = capture_with_feedback(run / 'after_rgbd', cfg)
    check_overlay_camera(run / 'after_rgbd/frame_000.png', session)
    flange = pose_matrix(actual['final_flange_pose_m_rad'])
    point = flange[:3, 3] if planned['frame'] == 'flange' else (flange @ matrix(session['T_flange_tcp']))[:3, 3]
    receipt = dict(frame=planned['frame'], gap_mm=planned['gap_mm'],
                   session_sha256=planned['session_sha256'], R_base_flange=planned['R_base_flange'],
                   actual_json=str((run / 'actual.json').resolve()),
                   model_feedback_error_mm=((point - planned['comparison_point_base_m']) * 1000).tolist(),
                   physical_error_measured=False, success=True)
    write_json(run / 'receipt.json', receipt)
    print(f'到位记录：{run / "receipt.json"}；这是模型反馈误差，不是实物测量误差。')
    target_overlay(run / 'after_rgbd/frame_000.png', run / 'after_target.png',
                   session, planned['comparison_point_base_m'], after)
    show(run / 'after_target.png', args.show)
    return 0


def home(args):
    cfg = load_config(args.config)
    run = new_run(args.output, 'home')
    start = bridge('snapshot', run / 'snapshot.json', cfg)
    ready(start)
    target = read_json(cfg['home'])['joints_rad']
    at_home = max(abs(a - b) for a, b in zip(start['joints_rad'], target)) < math.radians(.5)
    open_hand = cfg.get('home_open_hand', False)
    if at_home and not open_hand:
        print('已在保存的 HOME；没有发送运动命令')
        return 0
    capture_rgbd(run / 'rgbd', cfg)
    meta, depth, image, _ = load_batch(run)
    cam, quality = camera_transform(meta, cfg)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50))
    table, normal, _, _ = _plane(deproject(depth, red, meta['intrinsics'], meta['depth_scale_m']),
                                  Config(plane_tolerance_m=cfg['plane_tolerance_mm'] / 1000))
    scene = dict(cup_support_base_m=(cam[:3, :3] @ table + cam[:3, 3]).tolist(),
                 cup_normal_base=(cam[:3, :3] @ normal).tolist(), cup_envelope_radius_m=0,
                 geometry=dict(height_m=0))
    screen = Screen(table_only=True)
    checked = screen.arm.check_joint_path(start['joints_rad'], target)
    review = screen.check(checked.samples_rad, scene, True, cfg)
    blockers = review['blockers'] + ([] if checked.kinematic_checks_passed else ['关节路径未通过'])
    plan = dict(start_q_rad=start['joints_rad'], blockers=blockers,
                stages=[dict(name='saved_home', current_q_rad=start['joints_rad'], target_q_rad=target)],
                table_screen=review, calibration_quality_passed=quality)
    if open_hand:
        plan.update(kind='home_open_debug_plan', home_target_q_rad=target,
                    open_hand_target_0_100=[0] * 6, action_sequence=['OPEN_HAND', 'TO_HOME'])
        if at_home:
            plan['stages'] = []
    write_json(run / 'plan.json', plan)
    print(json.dumps(dict(home_deg=np.degrees(target).tolist(), screen=review), indent=2))
    show(run / 'rgbd/frame_000.png', args.show)
    if blockers:
        raise ValueError('HOME 路径未通过，请用已验证路径手动退出当前姿态')
    if not args.execute:
        print(f'HOME 仅预览；计划：{run / "plan.json"}')
        return 0
    if not args.cup_removed:
        raise ValueError('本 HOME 检查只检测桌面；执行前移开杯子并加 --cup-removed')
    if open_hand:
        print('本次动作：原位六路手指全 0 张开 → ' + ('保持 HOME。' if at_home else '回到 HOME。'))
    print('确认手中没有物体、杯子已移开、张手空间与通路无障碍，且没有其他控制器操作机械臂。')
    if not at_home and not open_hand:
        print('移动回 HOME 的检查采用全张开手形，请在运动前确认手指已张开。')
    if time.time() - (run / 'rgbd/frame_000.png').stat().st_mtime > 180:
        raise ValueError('图像过期，请重新 home')
    write_json(run / 'request.json', dict(execution_authorized=True, authorized_epoch_s=time.time(),
                                          plan=plan, config=cfg))
    actual = bridge('run', run / 'actual.json', cfg, run / 'request.json')
    print(f'HOME 已到位：{run / "actual.json"}')
    if open_hand:
        verified = actual.get('home_hand_open', {}).get('open_target_reached', False)
        print('六路全 0 张手已由位置反馈确认' if verified else
              '六路全 0 张手命令已发送；位置反馈不足，未验证到位。抓杯前请目视确认全部张开。')
    return 0


def record(args):
    receipt = read_json(args.receipt)
    if not receipt.get('success') or receipt.get('physical_error_measured'):
        raise ValueError('Expected an unmodified successful execution receipt')
    error = np.array(args.error_base_mm)
    if not np.isfinite(error).all():
        raise ValueError('Error must be finite')
    receipt.update(error_base_mm=error.tolist(), notes=args.notes, physical_error_measured=True,
                   measured_epoch_s=time.time(), convention='actual point minus frozen target, base XYZ mm')
    write_json(args.output, receipt)
    print(f'已记录独立测量：{args.output}')


def config_check(args):
    """Resolve and validate parameters without camera, CAN or movement."""
    from cup_grasp_demo.calibration_debug.grasp import options
    from cup_grasp_demo.calibration_debug.parameters import approach_options, closure_targets
    cfg = load_config(args.config)
    if cfg.get('pipeline_strategy') == 'green_open_cup':
        from cup_grasp_demo.calibration_debug.green_pipeline import validate
        green = validate(cfg)
        tcp = np.asarray(read_json(cfg['tcp_candidate'])['T_flange_contact_candidate'])
        tcp[:3, 3] += np.asarray(green['tcp_offset_flange_mm']) / 1000
        report = dict(config_path=str(args.config), strategy='green_open_cup',
                      green_cup=green, effective_tcp_flange_mm=(tcp[:3, 3]*1000).tolist(),
                      joint_test=read_json(ROOT/green['joint_test_config']), motion_attempted=False)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    from cup_grasp_demo.calibration_debug.joint_delivery import delivery_options
    report = dict(config_path=str(args.config), config=cfg,
                  joint_delivery=delivery_options(cfg.get('joint_delivery')),
                  effective_tcp_flange_mm=(configured_tcp(cfg)[:3, 3] * 1000).tolist(),
                  home_joints_deg=np.degrees(read_json(cfg['home'])['joints_rad']).tolist(),
                  motion_attempted=False)
    if 'side_grasp' in cfg:
        opts = options(cfg)
        report.update(approach=approach_options(opts), closure_targets_0_100=closure_targets(opts))
    if 'shake' in cfg:
        from cup_grasp_demo.calibration_debug.parameters import effective_shake_options
        report['shake'] = effective_shake_options(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('green-detect', help='只采集和定位绿色杯；不连接 CAN、不运动')
    p.add_argument('--config', type=Path, default=HERE / 'green_open_cup/config.json')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('pipeline', help='银杯抓取与摇晃；step 分步、auto 连续、fast 静默连续')
    p.add_argument('--config', type=Path, default=HERE / 'index_joint_center/config.json')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--mode', choices=('step', 'auto', 'fast', 'control'), default='step')
    p.add_argument('--until', choices=('ready', 'grip', 'shake-plan', 'shake', 'place'),
                   help='默认 step/auto 到 grip，fast 到 shake')
    p.add_argument('--execute', action='store_true', help='允许执行；省略时只显示流程，不访问硬件')
    p.add_argument('--resume', action='store_true', help='继续正常暂停的流程；失败或中断后不自动重放')
    p.add_argument('--status', action='store_true', help='只读查看最近一次流程状态')
    p.add_argument('--show', action='store_true', help='仅 STEP 模式弹出 X11 图像；auto/fast 只保存图像')
    p = commands.add_parser('config-check', help='只读检查配置、TCP 和手指目标，不访问硬件')
    p.add_argument('--config', type=Path, required=True)
    p = commands.add_parser('shake-plan', help='只读计算摇晃轨迹与限值；不执行运动')
    p.add_argument('--config', type=Path, default=HERE / 'index_joint_center/config.json')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--feedback-json', type=Path, help='离线回放关节和控制器限值；不访问 CAN')
    p.add_argument('--output', type=Path, help='默认覆盖 RUN/shake_plan.json')
    p = commands.add_parser('shake', help='复核/执行一次摇晃；结束保持闭手')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('home', help='预览/执行保存的 HOME；不使用 SDK 内置 HOME')
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cup-removed', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('capture', help='HOME 采集杯子接触点，覆盖同一 RUN 的上次采集')
    p.add_argument('--config', type=Path, default=HERE / 'config.json')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--replay', type=Path, help='离线回放；该会话不能运动')
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('tcp-view', help='只读拍摄当前 TCP 投影；不运动，不要求 HOME')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--session', type=Path, help='使用本次采集冻结的 TCP 和标定')
    source.add_argument('--config', type=Path, help='使用指定配置中的 TCP 和标定')
    p.add_argument('--output', type=Path, help='保存目录；每次新建运行子目录')
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('plan', help='用实时关节生成实验轨迹，不运动')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--frame', choices=('flange', 'tcp'), required=True)
    p.add_argument('--gap-mm', type=float, default=50)
    p.add_argument('--cup-removed', action='store_true')
    p.add_argument('--joints-json', type=Path, help='离线反馈输入；计划不能执行')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('move', help='复查已保存计划；--execute 才运动')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('grasp-plan', help='按会话配置规划杯侧停靠与两步闭手')
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--start', choices=('home', 'ready', 'pregrasp', 'contact'), default='home')
    p.add_argument('--until', choices=('ready', 'pregrasp', 'contact', 'grip'), default='grip')
    p.add_argument('--joints-json', type=Path, help='离线关节输入；计划不能执行')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('grasp', help='复核/执行抓取计划；默认不运动')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--show', action='store_true')
    p = commands.add_parser('record', help='记录人工独立测量的实物偏差')
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--error-base-mm', type=float, nargs=3, required=True)
    p.add_argument('--notes', default='')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('compare', help='比较两组实物测量，不自动修改标定')
    p.add_argument('--flange', type=Path, required=True)
    p.add_argument('--tcp', type=Path, required=True)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == 'green-detect':
        if load_config(args.config).get('pipeline_strategy') != 'green_open_cup':
            raise ValueError('green-detect requires green_open_cup configuration')
        from cup_grasp_demo.calibration_debug.green_pipeline import Workflow
        args.mode = 'step'
        def green_detect_only(args):
            flow = Workflow(args)
            flow.capture()
            print('绿色杯定位完成；未连接 CAN、未发送运动指令')
            return 0
        return dispatch(args, green_detect_only)
    if args.command == 'pipeline':
        if load_config(args.config).get('pipeline_strategy') == 'green_open_cup':
            if args.until is None:
                args.until = 'place'
            if args.mode == 'control':
                from cup_grasp_demo.calibration_debug.green_control import run
                return run(args)
            from cup_grasp_demo.calibration_debug.green_pipeline import run
            return run(args)
        if args.mode == 'control':
            raise ValueError('control 模式仅支持 green_open_cup 配置')
        if args.until == 'place':
            raise ValueError('place 当前用于 green_open_cup 配置')
        if args.until is None:
            args.until = 'shake' if args.mode == 'fast' else 'grip'
        from cup_grasp_demo.calibration_debug.pipeline_runner import run
        return run(args) if args.status else dispatch(args, run)
    if args.command == 'shake':
        from cup_grasp_demo.calibration_debug.shake_cli import execute
        return dispatch(args, execute)
    if args.command == 'shake-plan':
        from cup_grasp_demo.calibration_debug.shake_cli import plan as shake_plan
        return dispatch(args, shake_plan)
    if args.command in ('grasp-plan', 'grasp'):
        from cup_grasp_demo.calibration_debug import grasp_cli
        return dispatch(args, grasp_cli.plan if args.command == 'grasp-plan' else grasp_cli.execute)
    if args.command == 'compare':
        first, second = read_json(args.flange), read_json(args.tcp)
        if not first.get('physical_error_measured') or not second.get('physical_error_measured'):
            raise ValueError('需要两组人工独立测量，不能用机器人自己的 FK 验证自己')
        print(json.dumps(dict(tcp_minus_flange_error_in_flange_mm=measured_offset(first, second),
                              calibration_updated=False), indent=2))
        return 0
    handlers = dict(home=home, capture=capture, plan=plan, move=execute_plan, record=record)
    handlers['config-check'] = config_check
    handlers['tcp-view'] = tcp_view
    return dispatch(args, handlers[args.command]) or 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f'DEBUG STOP: {exc}', file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print('已中断；若正在运动，请检查 SDK 日志及实际保持状态。', file=sys.stderr)
        raise SystemExit(130)
