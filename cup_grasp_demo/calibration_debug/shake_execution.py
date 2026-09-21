"""One bounded MoveJS shake. Separate SDK process; no finger, lift or HOME commands."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rgb_hand_tracking")]
import visual_servo_probe as core
from finger_feedback_probe import BroadcastCache, FINGERS, _copy_getter
from cup_grasp_demo.calibration_debug.joint_delivery import take_js_control, fresh_js_hold
from cup_grasp_demo.calibration_debug.shake_readback import is_read_only_request
from cup_grasp_demo.calibration_debug.parameters import shake_options
from cup_grasp_demo.calibration_debug.shake_tracking import (
    check_delivery,
    check_feedback,
    measured_wave,
    rotation,
    interpolate_displacement,
    state_ok,
    FeedbackMonitor,
)


class ShakeGuard(core.AuditedSendGuard):
    """Only limit reads, CAN/MoveJS speed and joint targets; no enable or hand writes."""

    motion_allowed = False
    max_speed_percent = 60

    def install(self):
        if self.installed:
            return
        super().install()
        audited = self.bus_class.send

        def send(bus, message, *args, **kwargs):
            ident, data = message.arbitration_id, bytes(message.data)
            mode = (
                ident == 0x151
                and len(data) == 8
                and data[0] == 1
                and data[1] in (1, 255)
                and 1 <= data[2] <= self.max_speed_percent
                and data[3] in (0, 0xad)
                and not any(data[4:])
            )
            joints = len(data) == 8 and (
                ident in (0x155, 0x156, 0x157) or ident == 0x170 and not any(data[4:])
            )
            if not is_read_only_request(ident, data) and not (
                self.motion_allowed and (mode or joints)
            ):
                raise RuntimeError(
                    "SHAKE CAN guard rejected unexpected transmission " + hex(ident)
                )
            return audited(bus, message, *args, **kwargs)

        self.bus_class.send = send


def hand_feedback(hand, cache):
    result = {
        k: _copy_getter(hand, k, cache, time.time)
        for k in ("position", "current", "status")
    }
    if not result["position"]["fresh"]:
        raise RuntimeError("闭手位置反馈不新鲜，停止摇晃")
    status = result["status"]
    if status["fresh"] and (
        status["left_or_right"] != 2 or any(x > 2 for x in status["values"].values())
    ):
        raise RuntimeError("灵巧手状态异常")
    result["positions_0_100"] = [
        round(result["position"]["values"][key] * 100 / 255) for key in FINGERS
    ]
    return result


def validate_request(request, now):
    plan = request["plan"]
    shake_options({"shake": plan["parameters"]})
    if (
        request.get("execution_authorized") is not True
        or not 0 <= now - request["authorized_epoch_s"] <= 60
        or plan.get("offline_only")
        or not plan.get("planning_passed")
        or plan["blockers"]
        or request["table_screen"]["blockers"]
        or not request.get("scene_observed")
    ):
        raise ValueError("Invalid, stale, offline or blocked shake request")
    if (
        plan["kind"] != "shake_design_plan"
        or not 0 < plan["parameters"]["duration_s"] <= 60
        or request["speed_percent"] != 60
        or not 1 <= request["restore_speed_percent"] <= 10
    ):
        raise ValueError("Invalid bounded trial profile")
    samples = plan["samples"]
    q = [s["q_rad"] for s in samples]
    t = [s["t_s"] for s in samples]
    if (
        len(samples) < 3
        or any(len(row) != 7 for row in q)
        or not all(math.isfinite(v) for row in q for v in row)
        or not all(math.isfinite(v) for v in t)
        or t[0] != 0
        or t[-1] != plan["parameters"]["duration_s"]
        or any(not 0 < b - a <= 0.011 for a, b in zip(t, t[1:]))
        or max(
            abs(v - start)
            for row in (q[0], q[-1])
            for v, start in zip(row, plan["start_q_rad"])
        )
        > 1e-8
    ):
        raise ValueError("Malformed shake samples")
    for filename, expected in request["input_hashes"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Reviewed input changed: " + filename)


def verify_stop(session, target, report, *, timeout=6.0, require_position=True,
                stable_samples=10, poll_s=.03):
    if not 3 <= stable_samples <= 10 or not 0 <= poll_s <= .03:
        raise ValueError("Invalid stop verification settings")
    previous = None
    stable = 0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = core.fresh_feedback(session, previous=previous)
        previous = row
        report.append(row)
        if not state_ok(row):
            raise RuntimeError("停止验证时控制状态异常")
        good = state_ok(row, True) and (not require_position or max(
            abs(a - b) for a, b in zip(row["q_rad"], target)
        ) <= math.radians(0.05))
        stable = stable + 1 if good else 0
        if stable >= stable_samples:
            return row
        time.sleep(poll_s)
    raise TimeoutError("未确认机械臂在目标处停稳")


def verified_log_reader(row, proc_root):
    """Recognize the real system tail following one project log, not a name match."""
    try:
        proc = proc_root / str(row["pid"])
        executable = (proc / "exe").resolve(strict=True)
        if executable != Path("/usr/bin/tail").resolve(strict=True):
            return None
        argv = [
            v.decode()
            for v in (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        ]
        if (
            len(argv) != 3
            or Path(argv[0]).name != "tail"
            or argv[1] not in (
                "-f", "-F", "--follow", "--follow=name", "--follow=descriptor"
            )
            or " ".join(argv) != row.get("args")
        ):
            return None
        log = Path(argv[2])
        if not log.is_absolute():
            return None
        log = log.resolve(strict=True)
        if (
            not log.is_relative_to((ROOT / "cup_grasp_demo/datasets").resolve())
            or log.suffix != ".log"
            or not log.is_file()
        ):
            return None
        return dict(
            row, executable=str(executable), log_path=str(log), kind="tail_log_reader"
        )
    except (OSError, ValueError, KeyError, UnicodeError, RuntimeError):
        return None


def control_evidence(channel, request, *, proc_root=Path("/proc")):
    evidence = core.host_control_evidence(channel)
    candidates, observers = [], []
    for row in evidence.get("candidate_control_processes", []):
        observer = verified_log_reader(row, proc_root)
        if observer is None:
            candidates.append(row)
        else:
            observers.append(observer)
    evidence["candidate_control_processes"] = candidates
    evidence["verified_log_readers"] = observers
    camera = request.get("camera_process")
    if not camera:
        return evidence
    argv = (
        (proc_root / str(camera["pid"]) / "cmdline")
        .read_bytes()
        .rstrip(b"\0")
        .split(b"\0")
    )
    argv = [v.decode() for v in argv]
    if (
        argv != camera["argv"]
        or len(argv) != 6
        or Path(argv[1]).resolve()
        != Path(__file__).with_name("shake_camera.py").resolve()
        or argv[2] != "--serial"
        or argv[4] != "--output"
    ):
        raise RuntimeError("本次只读录像进程身份发生变化")
    evidence["verified_camera_observer"] = camera
    evidence["candidate_control_processes"] = [
        row
        for row in evidence["candidate_control_processes"]
        if row["pid"] != camera["pid"]
    ]
    return evidence


def control_conflicts(before, current):
    conflicts = core.evidence_blockers(before, current)
    if conflicts:
        candidates = {
            row["pid"]: row
            for evidence in (before, current)
            for row in evidence.get("candidate_control_processes", [])
        }
        if candidates:
            conflicts.append(
                "候选控制进程："
                + "; ".join(
                    f"PID {pid}: {row.get('args', '(命令未记录)')}"
                    for pid, row in sorted(candidates.items())
                )
            )
    return conflicts


def run(request):
    validate_request(request, time.time())
    plan = request["plan"]
    utilization = plan["parameters"].get("limit_utilization", .8)
    channel = request["channel"]
    bus, factory = core.load_sdk_runtime(channel)
    guard = ShakeGuard(bus)
    session = core.PassivePoseSession(bus, factory, deadline_s=2)
    session.guard = guard
    report = dict(
        sdk_method="move_js",
        success=False,
        motion_attempted=False,
        finger_commands_sent=0,
        commands=[],
        feedback=[],
        started_epoch_s=time.time(),
        cup_retention_verified=False,
        dice_change_verified=False,
    )
    controlled = False
    limits = None
    try:
        before = control_evidence(channel, request)
        report["host_control_before_connect"] = before
        report["sdk_runtime"] = session.start()
        hand = session.robot.init_effector("revo2")
        cache = BroadcastCache()
        session.robot.get_context().register_parser_packet_fun(cache.receive)
        rows, report["stationarity"] = core.stopped_window(session)
        previous = rows[-1]
        limits = core.joint_limits(session.robot)
        for row in rows:
            if core.ready_blockers(row, take_can_control=True) or max(
                abs(a - b) for a, b in zip(row["q_rad"], plan["start_q_rad"])
            ) > math.radians(0.05):
                raise RuntimeError("机械臂当前姿态与摇晃计划不同，请重新 shake-plan")
        center = plan["T_base_flange_center"]
        direction = plan["direction_base"]
        for sample in plan["samples"]:
            q = core.validate_target(sample["q_rad"], limits)
            pose = session.robot.fk(q)
            expected = [
                center[i][3] + sample["displacement_m"] * direction[i] for i in range(3)
            ]
            r = rotation(pose)
            if (
                math.dist(pose[:3], expected) > 0.00002
                or max(abs(r[i][j] - center[i][j]) for i in range(3) for j in range(3))
                > 0.0001
            ):
                raise RuntimeError("SDK FK 与规划轨迹不一致")
        report["host_control_before_motion"] = control_evidence(channel, request)
        conflicts = control_conflicts(before, report["host_control_before_motion"])
        if conflicts:
            raise RuntimeError("; ".join(conflicts))
        report["hand_before"] = hand_feedback(hand, cache)
        baseline = report["hand_before"]["positions_0_100"]
        if baseline[1] < 90 or min(baseline[0], *baseline[2:]) < 5:
            raise RuntimeError("手指尚未处于当前两步闭手后的抓握姿态")
        guard.permit()  # Reads only until all live limits have been checked.
        for i in range(1, 8):
            v = session.robot.get_joint_angle_vel_limits(
                i, timeout=0.25, min_interval=0
            )
            a = session.robot.get_joint_acc_limits(i, timeout=0.25, min_interval=0)
            if (
                v is None
                or a is None
                or plan["joint_peak_velocity_rad_s"][i - 1] > utilization * v.msg.max_joint_spd
                or plan["joint_peak_acceleration_rad_s2"][i - 1]
                > utilization * a.msg.max_joint_acc
                or any(
                    not v.msg.min_angle_limit + math.radians(1)
                    <= s["q_rad"][i - 1]
                    <= v.msg.max_angle_limit - math.radians(1)
                    for s in plan["samples"]
                )
            ):
                raise RuntimeError("实时关节限值与计划不匹配")
        f = session.robot.get_flange_vel_acc_limits(timeout=1, min_interval=0)
        if (
            f is None
            or plan["cartesian_peak_velocity_m_s"] > utilization * f.msg.end_max_linear_vel
            or plan["cartesian_peak_acceleration_m_s2"] > utilization * f.msg.end_max_linear_acc
        ):
            raise RuntimeError("实时末端限值与计划不匹配")
        guard.motion_allowed = True
        controlled = True
        session.robot.set_joint_limits_enabled(True)
        session.robot.set_speed_percent(request["speed_percent"])
        report["can_mode"] = take_js_control(core, session, previous, timeout_s=3)
        previous = report["can_mode"]["samples"][-1]
        check_feedback(previous, plan, expected_s=0)
        check_hand = hand_feedback(hand, cache)
        if max(abs(a - b) for a, b in zip(check_hand["positions_0_100"], baseline)) > 3:
            raise RuntimeError("摇晃启动前手形已变化")
        if not 0 <= time.time() - request["authorized_epoch_s"] <= 60:
            raise RuntimeError("Shake preflight expired")
        start = time.monotonic()
        report["motion_started_epoch_s"] = time.time()
        last_index = -1
        last_q = plan["start_q_rad"]
        last_hand = -1
        count = len(plan["samples"])
        duration = plan["parameters"]["duration_s"]
        monitor = FeedbackMonitor(plan)
        while True:
            index = min(
                count - 1, int((time.monotonic() - start) * plan["sample_hz"])
            )
            if index <= last_index:
                time.sleep(0.001)
                continue
            row = core.fresh_feedback(session, previous=previous)
            previous = row
            elapsed = time.monotonic() - start
            # Feedback acquisition can take a control tick. Select the command
            # after acquisition so we do not send an already stale target.
            index = min(count - 1, int(elapsed * plan['sample_hz']))
            expected = interpolate_displacement(plan["samples"], min(elapsed, duration), monitor.times)
            report["last_observation"] = dict(
                elapsed_s=elapsed, feedback=row, expected_displacement_m=expected
            )
            tracking = monitor.check(row, elapsed)
            report["feedback"].append(
                dict(elapsed_s=elapsed, feedback=row, tracking=tracking)
            )
            if not session.robot.get_joint_limits_enabled():
                raise RuntimeError("SDK joint limits disabled")
            if elapsed - last_hand >= 0.1:
                h = hand_feedback(hand, cache)
                if max(abs(a - b) for a, b in zip(h["positions_0_100"], baseline)) > 8:
                    raise RuntimeError("摇晃中手指反馈改变超过 8，停止并保留闭手")
                last_hand = elapsed
            sample = plan["samples"][index]
            check_delivery(
                sample["q_rad"], last_q, time.monotonic() - start, sample["t_s"], plan['parameters']
            )
            report["motion_attempted"] = True
            session.robot.move_js(sample["q_rad"])
            report["commands"].append(
                dict(
                    index=index,
                    elapsed_s=time.monotonic() - start,
                    target_q_rad=sample["q_rad"],
                )
            )
            last_index = index
            last_q = sample["q_rad"]
            if index == count - 1:
                break
        report["reference_send_elapsed_s"] = time.monotonic() - start
        report['motion_elapsed_s'] = report['reference_send_elapsed_s']
        report['duration_completed'] = report['reference_send_elapsed_s'] >= duration
        report["center_settle"] = []
        verify_stop(session, plan["start_q_rad"], report["center_settle"])
        report["returned_center"] = True
        report["hand_after"] = hand_feedback(hand, cache)
        report["measured_wave"] = measured_wave(report["feedback"], plan)
        report["success"] = report["measured_wave"]["tracking_verified"]
        if not report["success"]:
            report["error"] = "轨迹发送结束，但实际摆幅/频率未达到验收范围；未自动重试"
    except BaseException as error:
        if 'motion_started_epoch_s' in report:
            report['motion_elapsed_s'] = time.monotonic() - start
        report["error"] = type(error).__name__ + ": " + str(error)
        if controlled and limits is not None:
            try:
                report["failure_hold"] = fresh_js_hold(core, session, limits)
                report["hold_feedback"] = []
                verify_stop(
                    session,
                    report["failure_hold"]["target_rad"],
                    report["hold_feedback"],
                )
                report["failure_hold"]["hold_verified"] = True
            except BaseException as hold_error:
                report["hold_error"] = str(hold_error)
    finally:
        report.setdefault('duration_completed', False)
        if controlled and (
            report.get("returned_center")
            or report.get("failure_hold", {}).get("hold_verified")
        ):
            try:
                session.robot.set_speed_percent(request["restore_speed_percent"])
                report["restored_config_speed_percent"] = request[
                    "restore_speed_percent"
                ]
            except BaseException as error:
                report["restore_error"] = str(error)
                report["success"] = False
        guard.allowed = False
        session.close()
        report["tx"] = guard.report()
        report["sdk_disconnected"] = True
        report["ended_epoch_s"] = time.time()
        if report["feedback"] and "measured_wave" not in report:
            report["measured_wave"] = measured_wave(report["feedback"], plan)
    return report


def interrupted(signum, frame):
    raise RuntimeError("Signal " + str(signum) + " requested stop")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.request.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Request hash changed")
    request = json.loads(raw)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, interrupted)
    with core.control_lock(Path("/tmp/nero_" + request["channel"] + "_control.lock")):
        report = run(request)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in (
                    "success",
                    "error",
                    "motion_attempted",
                    "measured_wave",
                    "hold_error",
                )
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
