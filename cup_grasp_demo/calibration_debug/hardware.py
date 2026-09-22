"""SDK-runtime bridge; invoked only by an explicit live CLI operation."""

import argparse
from contextlib import redirect_stdout, nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


class StartPositionChanged(RuntimeError):
    """Pre-motion mismatch that a planner can recover using fresh feedback."""


def hand_start(plan, current, status, enabled, cfg, result):
    """A stationary hand-only action needs no replay of an old arm start."""
    green = cfg.get('green_cup', {})
    if (cfg.get('pipeline_strategy') != 'green_open_cup'
            or green.get('precision_error_action') != 'record'
            or plan.get('kind') != 'green_hand_command' or plan.get('stages')):
        return plan
    # Retain the existing 0.5-degree abnormal-motion bound and healthy idle gate.
    try:
        validate_start(plan, current, status, enabled, .5)
    except StartPositionChanged:
        raise RuntimeError('Hand start exceeds 0.5 degree motion bound') from None
    error = math.degrees(max(abs(a-b) for a,b in zip(current, plan['start_q_rad'])))
    result['hand_start_reference'] = dict(error_deg=error, action='use_current_idle_pose',
                                         joints_rad=list(current))
    return dict(plan, start_q_rad=list(current))


def validate_start(plan, current, status, enabled, tolerance_deg):
    if (len(current) != 7 or any(not math.isfinite(x) for x in current)
            or len(plan['start_q_rad']) != 7
            or any(not math.isfinite(x) for x in plan['start_q_rad'])):
        raise ValueError('Invalid starting joints')
    if (status.arm_status != 0 or status.motion_status != 0
            or status.ctrl_mode not in (1, 3) or enabled != [True] * 7):
        raise RuntimeError('机械臂未停稳、未使能或控制状态异常')
    if max(abs(a - b) for a, b in zip(current, plan['start_q_rad'])) > math.radians(tolerance_deg):
        raise StartPositionChanged('起点已改变，请重新 plan，不能复用旧关节路径')



def arm_motion_args(demo, stage, speed, cfg):
    maximum = 100 if cfg.get('pipeline_strategy') == 'green_open_cup' else 10
    if type(speed) is not int or not 1 <= speed <= maximum:
        raise ValueError('Invalid pipeline arm speed')
    # Keep the standalone demo menu's 1..10 range; authorize this workflow separately.
    motion = demo.make_parser().parse_args([
        '--format', 'json', 'move-j', '--joints-rad',
        *map(str, stage['target_q_rad']), '--speed', str(min(speed, 10)),
        '--timeout', str(cfg['timeout_s']), '--execute'])
    motion.speed = speed
    required = cfg.get('green_cup', {}).get('require_arm_position', True)
    if type(required) is not bool:
        raise ValueError('require_arm_position must be boolean')
    motion.require_joint_position = required if cfg.get('pipeline_strategy') == 'green_open_cup' else True
    motion.fast_completion = cfg.get("pipeline_strategy") == "green_open_cup" and cfg.get("green_cup", {}).get("fast_completion", False)
    return motion


def main(argv=None, *, connected=None):
    began = time.perf_counter()
    timings = {}
    checkpoint = began
    def mark(name):
        nonlocal checkpoint
        now = time.perf_counter()
        timings[name] = now - checkpoint
        checkpoint = now
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('snapshot', 'run'))
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--request', type=Path)
    parser.add_argument('--sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cached-snapshot', action='store_true',
                        help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.cached_snapshot and (connected is None or args.command != 'snapshot'):
        raise ValueError('Cached snapshots require an owned persistent SDK connection')
    if args.output.exists():
        raise FileExistsError(args.output)
    request = None
    if args.command == 'run':
        raw = args.request.read_bytes()
        if hashlib.sha256(raw).hexdigest() != args.sha256:
            raise ValueError('Reviewed request changed')
        request = json.loads(raw)
        if request.get('execution_authorized') is not True or request['plan']['blockers']:
            raise ValueError('Unapproved or blocked request')
        if not 0 <= time.time() - request['authorized_epoch_s'] <= 60:
            raise ValueError('Execution authorization expired')
    from nero_revo2_control import nero_revo2_demo as demo
    fast_feedback = args.cached_snapshot or bool(request and request['config'].get('pipeline_strategy') == 'green_open_cup'
                         and request['config'].get('green_cup', {}).get('fast_completion')
                         and request['config']['green_cup'].get('fast_cached_feedback', False))
    def snapshot(robot):
        if fast_feedback:
            from cup_grasp_demo.calibration_debug.fast_feedback import arm_snapshot
            return arm_snapshot(robot, demo)
        return demo.arm_snapshot(robot)
    from visual_servo_probe import control_lock, evidence_blockers, host_control_evidence
    from can.interfaces.socketcan import SocketcanBus
    robot, tx = None, []
    result = dict(success=False, command=args.command, motion_attempted=False,
                  finger_commands_sent=False, stages=[])
    original = SocketcanBus.send

    def send(bus, message, *a, **kw):
        answer = original(bus, message, *a, **kw)
        tx.append(dict(id=hex(message.arbitration_id), data=bytes(message.data).hex(),
                       epoch_s=time.time()))
        return answer

    SocketcanBus.send = send
    mark('request_setup_s')
    try:
        with (control_lock('/tmp/nero_' + args.channel + '_control.lock') if connected is None else nullcontext()), redirect_stdout(sys.stderr):
            before = host_control_evidence(args.channel) if connected is None else connected[2]
            robot = demo.create_robot(args.channel) if connected is None else connected[0]
            is_grasp = bool(request and request['plan'].get('kind') == 'side_grasp_debug_plan')
            is_home_open = bool(request and request['plan'].get('kind') == 'home_open_debug_plan')
            is_green_hand = bool(request and request['plan'].get('kind') in ('green_hand_command', 'green_home_open', 'feedback_together'))
            hand = (robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2) if is_grasp or is_home_open or is_green_hand else None) if connected is None else connected[1]
            if connected is None:
                robot.connect()
            conflicts = evidence_blockers(before, host_control_evidence(args.channel))
            if conflicts:
                raise RuntimeError('; '.join(conflicts))
            mark('connection_and_ownership_s')
            joints, pose, status = snapshot(robot)
            enabled = robot.get_joints_enable_status_list()
            result.update(joints_rad=joints, flange_pose_m_rad=pose, joints_enabled=enabled,
                          arm_status=int(status.arm_status), motion_status=int(status.motion_status),
                          ctrl_mode=int(status.ctrl_mode), observed_epoch_s=time.time())
            mark('initial_feedback_s')
            if request:
                plan, cfg = request['plan'], request['config']
                plan = hand_start(plan, joints, status, enabled, cfg, result)
                validate_start(plan, joints, status, enabled, cfg['start_tolerance_deg'])
                if is_home_open and plan.get('open_hand_target_0_100') != [0] * 6:
                    raise ValueError('HOME 只允许六路全 0 张手')
                from joint_delivery import ServoJointRobot
                result['joint_delivery_events'] = []
                motion_robot = ServoJointRobot(robot, demo, result['joint_delivery_events'],
                                               batch_limits=bool(connected is not None and fast_feedback),
                                               options=cfg.get('joint_delivery'),
                                               repeat_final=not (cfg.get('pipeline_strategy') == 'green_open_cup'
                                                   and cfg.get('green_cup', {}).get('precision_error_action') == 'record'))
                def arm_step(stage, speed):
                    result['active_stage_name'] = stage['name']
                    result['active_target_q_rad'] = list(stage['target_q_rad'])
                    joints, _, status = snapshot(robot)
                    validate_start({'start_q_rad': stage['current_q_rad']}, joints, status,
                                   robot.get_joints_enable_status_list(), 1.0)
                    demo.validate_joints(robot, stage['target_q_rad'])
                    motion = arm_motion_args(demo, stage, speed, cfg)
                    result['motion_attempted'] = True
                    demo.run_arm_motion_connected(motion, motion_robot,
                        **({'snapshot_reader': snapshot} if fast_feedback else {}))
                    actual, actual_pose, _ = snapshot(robot)
                    result['stages'].append(dict(name=stage['name'], joints_rad=actual,
                                                 flange_pose_m_rad=actual_pose,
                                                 joint_error_deg=[math.degrees(t-a) for t,a in zip(stage['target_q_rad'], actual)],
                                                 position_required=motion.require_joint_position))
                mark('execution_setup_s')
                if plan.get('kind') == 'feedback_together':
                    from cup_grasp_demo.calibration_debug.feedback_execution import execute
                    execute(plan, cfg, robot, hand, demo, result, arm_step, motion_robot)
                elif is_green_hand:
                    from green_hand_execution import execute
                    execute(plan, cfg, robot, hand, demo, result, arm_step=arm_step)
                elif is_grasp:
                    from grasp_execution import execute
                    execute(plan, cfg, robot, hand, demo, arm_step, result)
                elif is_home_open:
                    from home_execution import execute_home
                    execute_home(plan, cfg, robot, hand, demo, arm_step, result)
                else:
                    for stage in plan['stages']:
                        arm_step(stage, cfg['speed_percent'])
                mark('action_s')
                joints, pose, status = snapshot(robot)
                result.update(final_joints_rad=joints, final_flange_pose_m_rad=pose)
                result['final_snapshot'] = dict(
                    success=True, joints_rad=joints, flange_pose_m_rad=pose,
                    joints_enabled=robot.get_joints_enable_status_list(),
                    arm_status=int(status.arm_status), motion_status=int(status.motion_status),
                    ctrl_mode=int(status.ctrl_mode), observed_epoch_s=time.time())
                mark('final_feedback_s')
            result['success'] = True
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        if isinstance(exc, StartPositionChanged) and not result['motion_attempted'] and not result['finger_commands_sent']:
            result['failure_code'] = 'start_position_changed'
        if request and request['plan'].get('kind') == 'side_grasp_debug_plan':
            result['failed_state'] = result.get('last_state', 'PRECHECK')
            result['failed_stage_name'] = result.get('active_stage_name')
            result['last_state'] = 'STOP'
            result.setdefault('state_events', []).append(dict(
                state='STOP', event='failed', error=result['error'], epoch_s=time.time()))
            if robot is not None and result.get('active_target_q_rad') is not None:
                try:
                    actual, pose, state = demo.arm_snapshot(robot)
                    result['failure_feedback'] = dict(
                        joints_rad=actual, flange_pose_m_rad=pose,
                        target_q_rad=result['active_target_q_rad'],
                        error_deg=[math.degrees(t - a) for t, a in
                                   zip(result['active_target_q_rad'], actual)],
                        arm_status=int(state.arm_status), ctrl_mode=int(state.ctrl_mode),
                        motion_status=int(state.motion_status), observed_epoch_s=time.time())
                except BaseException as feedback_error:
                    result['failure_feedback_error'] = str(feedback_error)
    finally:
        if robot is not None and connected is None:
            robot.disconnect()
        SocketcanBus.send = original
        result['tx_frames'] = tx
        result['timings_s'] = dict(timings, total_before_receipt_s=time.perf_counter()-began)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'tx_frames'}))
    return 0 if result['success'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
