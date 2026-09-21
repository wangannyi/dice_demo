"""Read joint/flange limits and firmware; exact read-only CAN requests only."""

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2] / 'rgb_hand_tracking'
sys.path.insert(0, str(ROOT))
import visual_servo_probe as core


def is_read_only_request(can_id, data):
    """Allow no reset, parameter write, feedback change, payload or mode command."""
    data = bytes(data)
    joint_query = (
        can_id == 0x472
        and len(data) == 8
        and data[0] in range(1, 8)
        and data[1] in (1, 2)
        and not any(data[2:])
    )
    flange_query = can_id == 0x477 and data == bytes([1, 0, 0, 0, 3, 0, 0, 0])
    firmware_query = can_id == 0x4AF and data == bytes([1])
    return joint_query or flange_query or firmware_query


def query_joint_limit(getter, index):
    """Retry a missing reply once; never substitute cached/offline limits."""
    for timeout in (0.25, 1.0):
        reply = getter(index, timeout=timeout, min_interval=0)
        if reply is not None:
            return reply
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can0')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.channel.isalnum():
        raise ValueError('Invalid CAN channel')
    report = {
        "success": False,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "motion_commands_sent": 0,
        "parameter_write_commands_sent": 0,
        "limits": [],
        "speed_percentage_actual_readback": None,
        "speed_percentage_note": "SDK has setter but no controller speed-percent getter; local SDK cache is not readback.",
    }
    with core.control_lock(Path("/tmp/nero_" + args.channel + "_control.lock")):
        bus, factory = core.load_sdk_runtime(args.channel)
        guard = core.AuditedSendGuard(bus)
        guard.install()
        audited_send = bus.send

        def only_query(instance, message, *args, **kwargs):
            if not is_read_only_request(message.arbitration_id, message.data):
                raise RuntimeError("Read-only guard forbids this CAN transmission")
            return audited_send(instance, message, *args, **kwargs)

        bus.send = only_query
        session = core.PassivePoseSession(bus, factory)
        session.guard = guard
        try:
            session.start()
            rows, report["stationarity_before"] = core.stopped_window(session)
            assert all(row['status']['arm_status'] == 0 and row['status']['motion_status'] == 0
                       and row['enabled'] == [True] * 7 for row in rows)
            report["q_before_rad"] = rows[-1]["q_rad"]
            guard.permit()
            for index in range(1, 8):
                velocity = query_joint_limit(session.robot.get_joint_angle_vel_limits, index)
                acceleration = query_joint_limit(session.robot.get_joint_acc_limits, index)
                report["limits"].append(
                    {
                        "joint": index,
                        "max_velocity_rad_s": None
                        if velocity is None
                        else velocity.msg.max_joint_spd,
                        "min_angle_rad": None
                        if velocity is None
                        else velocity.msg.min_angle_limit,
                        "max_angle_rad": None
                        if velocity is None
                        else velocity.msg.max_angle_limit,
                        "max_acceleration_rad_s2": None
                        if acceleration is None
                        else acceleration.msg.max_joint_acc,
                        "velocity_feedback_timestamp": None
                        if velocity is None
                        else velocity.timestamp,
                        "acceleration_feedback_timestamp": None
                        if acceleration is None
                        else acceleration.timestamp,
                    }
                )
            flange = session.robot.get_flange_vel_acc_limits(timeout=1, min_interval=0)
            fields = (
                "end_max_linear_vel",
                "end_max_angular_vel",
                "end_max_linear_acc",
                "end_max_angular_acc",
            )
            report["flange_limits"] = (
                None
                if flange is None
                else {key: getattr(flange.msg, key) for key in fields}
            )
            report["flange_feedback_timestamp"] = (
                None if flange is None else flange.timestamp
            )
            report["firmware"] = session.robot.get_firmware(timeout=1, min_interval=0)
            guard.allowed = False
            rows, report["stationarity_after"] = core.stopped_window(session)
            assert all(row['status']['arm_status'] == 0 and row['status']['motion_status'] == 0
                       and row['enabled'] == [True] * 7 for row in rows)
            report["q_after_rad"] = rows[-1]["q_rad"]
            assert (
                max(
                    abs(a - b)
                    for a, b in zip(report["q_before_rad"], report["q_after_rad"])
                )
                < 0.001
            )
            missing = [f"J{row['joint']} {label}" for row in report["limits"]
                       for key, label in (("max_velocity_rad_s", "速度/角度限值"),
                                          ("max_acceleration_rad_s2", "加速度限值"))
                       if row[key] is None]
            if missing:
                raise TimeoutError("关节限值查询无回复（每项已等待 0.25s 并重试 1s）：" + ", ".join(missing))
            assert report["flange_limits"] is not None
            report["success"] = True
        except BaseException as error:
            report["error"] = type(error).__name__ + ": " + str(error)
        finally:
            guard.allowed = False
            session.close()
            report["sdk_disconnected"] = True
            report["tx"] = guard.report()
            bus.send = audited_send
            guard.restore()
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "success",
                    "error",
                    "limits",
                    "flange_limits",
                    "firmware",
                    "tx",
                )
            }
        )
    )
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
