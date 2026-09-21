"""Take CAN control at a stationary user-designated HOME without moving joints."""

from __future__ import annotations

import argparse
import json
import math
import time

from nero_revo2_control.nero_revo2_demo import (
    arm_snapshot,
    create_robot,
    read_joints,
    require_arm_ready,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-joints-deg", nargs=7, type=float, required=True)
    args = parser.parse_args()

    robot = create_robot(args.channel)
    mode_frame_sent = False
    try:
        robot.connect()
        first = read_joints(robot)
        time.sleep(0.25)
        second = read_joints(robot)
        if first["ctrl_mode"] != second["ctrl_mode"]:
            raise RuntimeError("Control mode changed during stationary HOME check")
        if max(abs(a - b) for a, b in zip(first["joints_deg"], second["joints_deg"])) > 0.1:
            raise RuntimeError("Joint angles changed during stationary HOME check")
        if max(abs(a - b) for a, b in zip(second["joints_deg"], args.expected_joints_deg)) > 0.1:
            raise RuntimeError("Live joints differ from user-designated HOME")
        if second["within_sdk_limits"] != [True] * 7:
            raise RuntimeError("HOME joints are outside the SDK limits")
        joints, _, status = arm_snapshot(robot)
        require_arm_ready(robot, joints, status)
        if max(abs(math.degrees(a) - b) for a, b in zip(joints, args.expected_joints_deg)) > 0.1:
            raise RuntimeError("Final arm snapshot differs from designated HOME")
        if int(status.motion_status) != 0:
            raise RuntimeError("Arm has not reached a stationary target")
        if int(status.ctrl_mode) == 1:
            event = "can_control_already_active"
        elif not args.execute:
            event = "can_control_preview_only"
        else:
            robot.set_motion_mode("j")
            mode_frame_sent = True
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                time.sleep(0.1)
                after = read_joints(robot)
                if after["ctrl_mode"] == 1:
                    if after["arm_status"] != 0 or after["joints_enabled"] != [True] * 7:
                        raise RuntimeError("CAN handoff changed arm readiness")
                    if max(abs(a - b) for a, b in zip(after["joints_deg"], args.expected_joints_deg)) > 0.1:
                        raise RuntimeError("CAN handoff moved the designated HOME")
                    event = "can_control_handoff_verified"
                    break
            else:
                raise RuntimeError("Controller did not enter CAN mode within five seconds")
        print(json.dumps({"event": event, "mode_frame_sent": mode_frame_sent,
                          "joint_motion_sent": False,
                          "home_joints_deg": second["joints_deg"]},
                         ensure_ascii=False), flush=True)
        return 0
    except Exception as error:
        print(json.dumps({"event": "can_control_handoff_failed",
                          "mode_frame_may_have_been_sent": mode_frame_sent,
                          "error": str(error)}, ensure_ascii=False), flush=True)
        return 2
    finally:
        robot.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
