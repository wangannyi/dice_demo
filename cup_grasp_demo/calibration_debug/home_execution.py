"""Open all six Revo2 channels at the starting pose, then return to HOME."""

import math
import time

from cup_grasp_demo.calibration_debug.grasp_execution import fresh_current


def open_before_home(robot, hand, demo, target, result, *, monotonic=time.monotonic,
                     wallclock=time.time, sleep=time.sleep):
    if len(target) != 7 or not all(math.isfinite(x) for x in target):
        raise ValueError('Invalid opening joint pose')
    result['home_ready_verified'] = False
    def monitor(allow_web=False):
        joints, _, status = demo.arm_snapshot(robot)
        if (len(joints) != 7 or not all(math.isfinite(x) for x in joints)
                or status.arm_status != 0 or status.motion_status != 0
                or status.ctrl_mode not in ((1, 3) if allow_web else (1,))
                or robot.get_joints_enable_status_list() != [True] * 7
                or max(abs(a - b) for a, b in zip(joints, target)) > math.radians(.5)):
            raise RuntimeError('归位前张手要求七轴保持起始姿态、停稳、使能且状态正常')
        return status
    status = monitor(allow_web=True)
    demo.require_right_hand(hand, True)
    before = fresh_current(hand, demo, wallclock)
    report = dict(target_0_100=[0] * 6, duration_s=1., current_before=before,
                  position_samples=[], open_target_reached=False, arm_target_sent=False,
                  command_wait_completed=False,
                  opening_q_rad=list(target), timing='before_home_motion',
                  position_tolerance_0_100=3, terminal_event_settle_s=.2,
                  terminal_event_max_age_s=1.5)
    result['home_hand_open'] = report
    if status.ctrl_mode == 3:
        # Change ownership only: do not issue a MoveJ/HOME target just to open.
        robot.set_motion_mode('js')
        deadline = monotonic() + 2.
        while monitor(allow_web=True).ctrl_mode != 1:
            if monotonic() >= deadline:
                raise TimeoutError('HOME 张手前未确认进入 CAN 控制')
            sleep(.02)
        report['can_mode_command_sent'] = True
    monitor()
    result['last_state'] = 'OPEN_HAND'
    result['finger_commands_sent'] = True
    epoch, started = wallclock(), monotonic()
    hand.position_time_ctrl(mode='pos', **dict.fromkeys(demo.FINGER_NAMES, 0))
    hand.position_time_ctrl(mode='time', **dict.fromkeys(demo.FINGER_NAMES, 100))
    report['position_time_interval_s'] = monotonic() - started
    if report['position_time_interval_s'] > .05:
        raise RuntimeError('HOME 张手位置/时间命令间隔超过 50 ms')
    previous, stable = None, 0
    while monotonic() - started < 2.:
        monitor()
        feedback = hand.get_finger_pos()
        stamp = demo.feedback_stamp(feedback)
        if (stamp is not None and stamp > epoch and (previous is None or stamp > previous)
                and 0 <= wallclock() - stamp <= .5):
            previous = stamp
            values = demo.position_values(feedback)
            error = max(abs(values[n]) for n in demo.FINGER_NAMES)
            stable = stable + 1 if error <= report['position_tolerance_0_100'] else 0
            report['position_samples'].append(dict(received_epoch_s=stamp, values=values, error=error))
        sleep(.05)
    report['current_after'] = fresh_current(hand, demo, wallclock)
    monitor()
    age = wallclock() - previous if previous is not None else None
    continuous = bool(stable >= 3 and age is not None and 0 <= age <= .5)
    # Some observed openings stop publishing positions after reaching the target.
    # Accept the last fresh-at-receipt event only after the commanded duration,
    # with time to observe a rebound and a bounded age; never count cache rereads.
    terminal = bool(stable > 0 and previous is not None
                    and previous - epoch >= report['duration_s']
                    and report['terminal_event_settle_s'] <= age <= report['terminal_event_max_age_s'])
    report['last_position_age_s'] = age
    report['open_target_reached'] = continuous or terminal
    report['completion_basis'] = ('position_feedback' if continuous else
                                 'terminal_position_event' if terminal else
                                 'command_duration_only_unverified')
    result['last_state'] = 'OPEN_HAND_VERIFIED' if report['open_target_reached'] else 'OPEN_HAND_SENT'
    if report['position_samples'] and not report['open_target_reached']:
        raise RuntimeError('手指位置反馈未确认六路全 0，停止 HOME 归位')
    report['command_wait_completed'] = True
    return report


def execute_home(plan, cfg, robot, hand, demo, arm_step, result, **timing):
    """An opening failure must prevent every HOME joint command."""
    result['home_joint_target_reached'] = False
    report = open_before_home(robot, hand, demo, plan['start_q_rad'], result, **timing)
    for stage in plan['stages']:
        result['last_state'] = 'TO_HOME'
        arm_step(stage, cfg['speed_percent'])
        report['arm_target_sent'] = True
    joints, _, status = demo.arm_snapshot(robot)
    target = plan['home_target_q_rad']
    if (len(joints) != 7 or not all(math.isfinite(x) for x in joints)
            or max(abs(a - b) for a, b in zip(joints, target)) > math.radians(.5)
            or status.arm_status != 0 or status.motion_status != 0 or status.ctrl_mode != 1
            or robot.get_joints_enable_status_list() != [True] * 7):
        raise RuntimeError('张手后未确认机械臂到达 HOME 并停稳')
    result['home_joint_target_reached'] = True
    result['home_ready_verified'] = report['open_target_reached']
    result['last_state'] = 'HOME_OPEN_VERIFIED' if report['open_target_reached'] else 'HOME_OPEN_SENT'
