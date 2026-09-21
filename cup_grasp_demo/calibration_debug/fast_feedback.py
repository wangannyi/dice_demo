"""Reuse recent SDK feedback without waiting for three sequential packets."""
import math
import time


def arm_snapshot(robot, demo, *, wallclock=time.time):
    values = [robot.get_joint_angles(), robot.get_flange_pose(), robot.get_arm_status()]
    now = wallclock()
    stamps = [demo.feedback_stamp(v) if v is not None else None for v in values]
    if any(not isinstance(s, (int, float)) or not math.isfinite(s)
           or not 0 <= now - s <= .1 for s in stamps):
        return demo.arm_snapshot(robot)  # Existing bounded fresh-read fallback.
    joints, pose = list(values[0].msg), list(values[1].msg)
    if len(joints) != 7 or len(pose) != 6 or not all(math.isfinite(x) for x in joints + pose):
        raise ValueError('Invalid cached arm feedback')
    demo.check_comm(robot)
    return joints, pose, values[2].msg
