"""One configured hand command while the arm remains stationary."""

import math
import time
from cup_grasp_demo.flow.grasp_execution import send_closure


def open_feedback_reference(options):
    if type(options.get("require_hand_position", True)) is not bool:
        raise ValueError("require_hand_position must be boolean")
    reference = options.get("open_feedback_reference_0_100")
    if reference is not None and (
        not isinstance(reference, list) or len(reference) != 6
        or any(type(v) is not int or not 0 <= v <= 100 for v in reference)
    ):
        raise ValueError("open_feedback_reference_0_100 must contain six integers 0..100")
    return reference


def apply_open_reference(report, reference, names):
    """A measured open-position reference changes verification, never commands."""
    if reference is None:
        return
    report["command_target_reached"] = report["position_target_reached"]
    report["open_feedback_reference_0_100"] = reference
    samples = report.get("position_samples", [])
    if not samples:
        return  # Preserve the existing explicit unverified-feedback behavior.
    tail = samples[-3:]
    errors = [max(abs(row["values"][key] - value)
                  for key, value in zip(names, reference)) for row in tail]
    stamps = [row["received_epoch_s"] for row in tail]
    reached = len(tail) == 3 and stamps[0] < stamps[1] < stamps[2] and max(errors) <= 3
    report["position_target_reached"] = reached
    report["open_reference_reached"] = reached
    report["completion_basis"] = "configured_open_position_feedback"
    report["open_reference_errors_0_100"] = errors


def execute(plan, cfg, robot, hand, demo, result, arm_step=None):
    options = cfg["green_cup"]
    arm_tolerance_deg = options.get('hand_start_tolerance_deg', .5)
    read_feedback = options.get('read_hand_feedback', True)
    if type(read_feedback) is not bool or (not read_feedback and options.get('require_hand_position', True)):
        raise ValueError('Cannot require hand position while disabling hand feedback')
    reference = open_feedback_reference(options)
    target = plan["target_0_100"]
    combined = plan.get('kind') == 'green_home_open'
    if combined and (target != [0] * 6 or arm_step is None
                     or not options.get('fast_completion') or options.get('require_hand_position', True)):
        raise ValueError('Combined HOME requires FAST open-hand command without position confirmation')
    if target not in (options["open_targets_0_100"], options["grip_targets_0_100"]):
        raise ValueError("Unexpected green-cup hand command")
    if len(target) != 6 or any(type(v) is not int or not 0 <= v <= 100 for v in target):
        raise ValueError("Invalid hand target")
    demo.require_right_hand(hand, True)

    def monitor(allow_web=False):
        if options.get('fast_completion') and options.get('fast_cached_feedback', False):
            from cup_grasp_demo.flow.fast_feedback import arm_snapshot
            q, _, status = arm_snapshot(robot, demo)
        else:
            q, _, status = demo.arm_snapshot(robot)
        if (
            status.arm_status != 0
            or status.ctrl_mode not in ((1, 3) if allow_web else (1,))
            or robot.get_joints_enable_status_list() != [True] * 7
        ):
            raise RuntimeError("Hand command requires healthy CAN control")
        if (not combined
                and max(abs(a - b) for a, b in zip(q, plan["start_q_rad"]))
                > math.radians(arm_tolerance_deg)):
            raise RuntimeError("Arm moved during hand command")
        return status

    status = monitor(allow_web=True)
    if status.ctrl_mode == 3:
        if status.motion_status != 0:
            raise RuntimeError("Cannot switch to CAN while arm is moving")
        robot.set_motion_mode("js")
        result["can_mode_command_sent"] = True
        deadline = time.monotonic() + 2.0
        while monitor(allow_web=True).ctrl_mode != 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("张手前未确认进入 CAN 控制")
            time.sleep(.02)
    monitor()
    result["finger_commands_sent"] = True
    result["hand_command"] = send_closure(
        hand,
        demo,
        dict(
            state="CLOSE_FINGERS",
            target_0_100=target,
            duration_s=options["finger_duration_s"],
            settle_s=options["finger_settle_s"],
        ),
        monitor,
        **({'read_feedback': False} if not read_feedback else {}),
        **({'maximum_speed': True} if options.get('feedback_hand_max_speed') is True else {}),
        **({'during_action': lambda: [arm_step(s, cfg['speed_percent']) for s in plan['stages']]}
           if combined else {}),
    )
    result["finger_commands_sent"] = True
    if target == options["open_targets_0_100"]:
        apply_open_reference(result["hand_command"], reference, demo.FINGER_NAMES)
    report = result["hand_command"]
    report["position_required"] = options.get("require_hand_position", True)
    if not report["position_required"]:
        report["completion_basis"] = "command_duration_only_position_not_required"
    if report["position_required"] and report["position_target_reached"] is False:
        raise RuntimeError("手指反馈未达到目标，停止后续动作")
