"""Release at the starting pose and return HOME before locating the cup."""

import math
import time

import cv2

from cup_grasp_demo.calibration_debug import debug as common
from cup_grasp_demo.calibration_debug.core import Screen, read_json, write_json
from cup_grasp_demo.side_grasp.preview_index import load_batch
from dice_cup_localization.geometry import Config, _plane, deproject


def home_scene(run, cfg):
    """Fit the red table without requiring a visible, unoccluded cup."""
    meta, depth, image, _ = load_batch(run)
    camera, quality = common.camera_transform(meta, cfg)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50))
    table, normal, fraction, rms = _plane(
        deproject(depth, red, meta['intrinsics'], meta['depth_scale_m']),
        Config(plane_tolerance_m=cfg['plane_tolerance_mm'] / 1000))
    scene = dict(cup_support_base_m=(camera[:3, :3] @ table + camera[:3, 3]).tolist(),
                 cup_normal_base=(camera[:3, :3] @ normal).tolist(),
                 cup_envelope_radius_m=0, geometry=dict(height_m=0),
                 table_fit=dict(inlier_fraction=fraction, rms_mm=rms * 1000))
    return scene, quality


def make_home_plan(start, target, scene, cfg, screen=None):
    screen = screen or Screen(table_only=True)
    checked = screen.arm.check_joint_path(start, target)
    # The cup can be occluded by the gripping hand. Operator checks the return
    # corridor; this stage screens the table and joints, not cup identity/shape.
    review = screen.check(checked.samples_rad, scene, True, cfg)
    blockers = review['blockers'] + ([] if checked.kinematic_checks_passed else ['HOME 关节路径未通过'])
    at_home = max(abs(a - b) for a, b in zip(start, target)) < math.radians(.5)
    return dict(kind='home_open_debug_plan', start_q_rad=list(start),
                home_target_q_rad=list(target), open_hand_target_0_100=[0] * 6,
                action_sequence=['OPEN_HAND', 'TO_HOME'], blockers=blockers,
                stages=[] if at_home else [dict(name='saved_home', current_q_rad=list(start),
                                                target_q_rad=list(target))],
                table_screen=review, cup_removed=False, cup_screened=False,
                screen_scope='table_and_joints', scene=scene)


def check_home_result(actual, *, require_position_feedback=False):
    """Keep joint arrival separate from optional hand-position measurements."""
    if (not actual.get('success') or not actual.get('home_joint_target_reached',
                                                   actual.get('home_ready_verified', False))):
        raise RuntimeError('HOME 关节未确认到位或执行失败；查看 actual.json')
    verified = actual.get('home_ready_verified') is True
    report = actual.get('home_hand_open', {})
    if not verified:
        completed_without_positions = (
            actual.get('finger_commands_sent') is True
            and report.get('command_wait_completed') is True
            and report.get('target_0_100') == [0] * 6
            and report.get('position_samples') == []
            and report.get('completion_basis') == 'command_duration_only_unverified')
        if require_position_feedback or not completed_without_positions:
            raise RuntimeError('HOME 关节已到位，但张手反馈或指令完成记录不足；查看 actual.json')
        print('HOME 关节已到位；张手指令与等待已完成、电流通信正常。'
              '本次无新手指位置反馈，按指令完成继续；手指位置未实测确认。', flush=True)
    return dict(home_joint_target_reached=True, home_ready_verified=verified,
                hand_position_verified=verified,
                hand_completion_basis=report.get('completion_basis', 'position_feedback'))


def cached_home_scene(directory, cfg):
    """Reuse only a verified fixed-table plane; never reuse the old cup target."""
    try:
        session, frozen = common.verify_session(directory)
        if session.get('replay_only') or frozen['serial'] != cfg['serial']:
            return None
        if (common.digest(frozen['calibration']) != common.digest(cfg['calibration'])
                or frozen['plane_tolerance_mm'] != cfg['plane_tolerance_mm']):
            return None
        source = session['scene']
        point, normal = source['cup_support_base_m'], source['cup_normal_base']
        if (len(point) != 3 or len(normal) != 3
                or not all(math.isfinite(v) for v in point + normal)
                or abs(math.sqrt(sum(v*v for v in normal)) - 1.) > .01):
            return None
        if not session['calibration_quality_passed'] and not cfg.get('allow_provisional_calibration'):
            return None
        return dict(cup_support_base_m=list(point), cup_normal_base=list(normal),
                    cup_envelope_radius_m=0, geometry=dict(height_m=0)), session['calibration_quality_passed']
    except (OSError, ValueError, KeyError, TypeError):
        return None


def execute_home(directory, cfg, show=False, *, fast=False):
    strict = cfg.get('home_require_position_feedback', False)
    if not isinstance(strict, bool):
        raise ValueError('home_require_position_feedback must be a boolean')
    run = common.new_run(directory, 'pipeline_home')
    cached = cached_home_scene(directory, cfg) if fast and cfg.get('fast_reuse_table', False) else None
    if cached is not None:
        snapshot = common.bridge('snapshot', run / 'snapshot.json', cfg)
        common.ready(snapshot)
        scene, quality = cached
    else:
        snapshot = common.capture_with_feedback(run / 'rgbd', cfg)
        scene, quality = home_scene(run, cfg)
    target = read_json(cfg['home'])['joints_rad']
    plan = make_home_plan(snapshot['joints_rad'], target, scene, cfg)
    plan['calibration_quality_passed'] = quality
    plan['table_source'] = 'verified_previous_capture' if cached is not None else 'fresh_depth_capture'
    write_json(run / 'plan.json', plan)
    if cached is None:
        common.show(run / 'rgbd/frame_000.png', show)
    if plan['blockers']:
        raise ValueError(f'HOME 桌面/关节路径检查未通过：{plan["blockers"]}；请先人工退离到已验证姿态')
    if cached is None and time.time() - (run / 'rgbd/frame_000.png').stat().st_mtime > 180:
        raise ValueError('HOME 图像过期，请重新开始 pipeline')
    request = dict(execution_authorized=True, authorized_epoch_s=time.time(), plan=plan, config=cfg)
    write_json(run / 'request.json', request)
    actual = common.bridge('run', run / 'actual.json', cfg, run / 'request.json')
    completed = check_home_result(actual, require_position_feedback=strict)
    return dict(actual_path=str(run / 'actual.json'), plan_path=str(run / 'plan.json'),
                **completed)
