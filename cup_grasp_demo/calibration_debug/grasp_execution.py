"""SDK-side grasp state execution. Uses only the Python standard library."""

import math
import time

from cup_grasp_demo.calibration_debug.parameters import DEFAULT_FINGER_TARGETS, closure_targets

TARGETS = tuple(map(tuple, DEFAULT_FINGER_TARGETS))


def validate_sequence(plan, cfg=None):
    """Reject unexpected hand recipes and any arm movement after closing starts."""
    states = {'TO_PREGRASP': 0, 'APPROACH': 1, 'THUMB_BASE': 2, 'CLOSE_FINGERS': 3}
    if plan.get('strategy') == 'direct_close':
        states = {'TO_CLOSE_READY': 0, 'APPROACH_CLOSE_READY': 1, 'THUMB_BASE': 2, 'CLOSE_FINGERS': 3}
    prior, hands, hand_states = -1, [], []
    for stage in plan['stages']:
        order = states.get(stage['state'], -1)
        if order < prior or order < 0:
            raise ValueError('Invalid grasp state order')
        prior = order
        if stage['kind'] == 'hand':
            if stage['state'] not in ('THUMB_BASE', 'CLOSE_FINGERS'):
                raise ValueError('Unexpected hand state')
            hands.append(tuple(stage['target_0_100']))
            hand_states.append(stage['state'])
            if not .5 <= stage['duration_s'] <= 3 or not .1 <= stage['settle_s'] <= 2:
                raise ValueError('Invalid finger timing')
        elif stage['kind'] != 'arm' or order > 1:
            raise ValueError('Unexpected arm state')
        elif not 1 <= stage['speed_percent'] <= 10:
            raise ValueError('Invalid arm speed')
    targets = closure_targets((cfg or {}).get('side_grasp', {}))
    expected = list(map(tuple, targets)) if plan['until_state'] == 'grip' else []
    expected_states = ['THUMB_BASE', 'CLOSE_FINGERS'] if expected else []
    if hands != expected or hand_states != expected_states:
        raise ValueError('Grasp must close thumb base first, then the other five channels')


def fresh_current(hand, demo, wallclock=time.time):
    value = demo.read_fresh(hand.get_finger_current, 1., 'Revo2 current')
    stamp = demo.feedback_stamp(value)
    if stamp is None or not 0 <= wallclock() - stamp <= 1:
        raise RuntimeError('No fresh Revo2 current feedback')
    return dict(received_epoch_s=stamp, values=demo.finger_values(value))


def send_closure(hand, demo, stage, monitor, *, monotonic=time.monotonic,
                 wallclock=time.time, sleep=time.sleep, read_feedback=True, during_action=None, maximum_speed=False):
    """Timed two-frame command; report unavailable position feedback explicitly."""
    if type(maximum_speed) is not bool:
        raise ValueError("maximum_speed must be boolean")
    before = fresh_current(hand, demo, wallclock) if read_feedback else None
    target = dict(zip(demo.FINGER_NAMES, stage['target_0_100']))
    duration = stage['duration_s']
    epoch, started = wallclock(), monotonic()
    hand.position_time_ctrl(mode='pos', **target)
    hand.position_time_ctrl(mode='time', **dict.fromkeys(demo.FINGER_NAMES, 0 if maximum_speed else round(duration * 100)))
    interval = monotonic() - started
    if interval > .05:
        raise RuntimeError('Revo2 position/time commands exceeded 50 ms')
    if during_action is not None:
        during_action()
    positions, last_stamp = [], None
    while monotonic() - started < duration + stage['settle_s']:
        monitor()
        feedback = hand.get_finger_pos() if read_feedback else None
        stamp = demo.feedback_stamp(feedback)
        if (stamp is not None and stamp > epoch and stamp != last_stamp
                and 0 <= wallclock() - stamp <= 1):
            last_stamp = stamp
            positions.append(dict(received_epoch_s=stamp, values=demo.position_values(feedback)))
        sleep(min(.05, max(0., duration + stage['settle_s'] - (monotonic() - started))))
    after = fresh_current(hand, demo, wallclock) if read_feedback else None
    error = (max(abs(positions[-1]['values'][k] - v) for k, v in target.items())
             if positions else None)
    if stage['state'] == 'THUMB_BASE' and error is not None and error > 3:
        raise RuntimeError('拇指根位置反馈未到目标，停止后续闭手')
    return dict(target_0_100=stage['target_0_100'], duration_s=duration,
                position_time_interval_s=interval, position_feedback_available=bool(positions),
                speed_mode='max' if maximum_speed else 'timed',
                command_time_ticks=0 if maximum_speed else round(duration * 100),
                position_target_reached=(error <= 3 if error is not None else None),
                completion_basis='position_feedback' if positions else 'elapsed_command_duration_only',
                position_samples=positions, current_before=before, current_after=after,
                physical_grip_verified=False)


def execute(plan, cfg, robot, hand, demo, arm_step, result):
    validate_sequence(plan, cfg)
    if any(stage['kind'] == 'hand' for stage in plan['stages']):
        demo.require_right_hand(hand, True)
        result['hand_communication'] = fresh_current(hand, demo)
    result['state_events'] = []
    for stage in plan['stages']:
        result['last_state'] = stage['state']
        print(f'GRASP_STATE {stage["state"]} {stage["name"]}', flush=True)
        result['state_events'].append(dict(state=stage['state'], event='enter', epoch_s=time.time()))
        if stage['kind'] == 'arm':
            arm_step(stage, stage['speed_percent'])
        else:
            def monitor():
                joints, _, status = demo.arm_snapshot(robot)
                if (status.arm_status != 0 or status.motion_status != 0
                        or status.ctrl_mode != 1 or robot.get_joints_enable_status_list() != [True] * 7
                        or max(abs(a - b) for a, b in zip(joints, stage['current_q_rad'])) > math.radians(.5)):
                    raise RuntimeError('闭手期间机械臂状态或位置改变')
            monitor()
            result['finger_commands_sent'] = True
            row = send_closure(hand, demo, stage, monitor)
            result['stages'].append(dict(name=stage['name'], state=stage['state'], **row))
        result['state_events'].append(dict(state=stage.get('state_completed', stage['state']),
                                          event='completed', epoch_s=time.time()))
    result['last_state'] = 'GRIP_COMMANDS_SENT' if plan['until_state'] == 'grip' else plan['until_state'].upper()
    result['physical_grip_verified'] = False
