"""Operator-authorized joint test using MoveJS; standard-library SDK process."""

import argparse
import hashlib
from itertools import chain
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "rgb_hand_tracking")]
from cup_grasp_demo.calibration_debug import shake_execution as shared
from cup_grasp_demo.calibration_debug.joint_delivery import take_js_control, fresh_js_hold
from cup_grasp_demo.calibration_debug.joint_profile import (
    KIND,
    options,
    make_plan,
    at,
    joint_values,
    feedback_check,
    reference_errors,
    tracking_exceedances,
    measurements,
)

core = shared.core


class JointGuard(shared.ShakeGuard):
    max_speed_percent = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scanned = 0
        self._pending = set()
        self._returned = 0
        self._uncertain = False

    def report(self, *, include_history=False):
        """Read counters incrementally; copy the full audit only for the receipt."""
        if include_history:
            return super().report()
        with self.lock:
            end = len(self.history)
            pending = set()
            for i in chain(self._pending, range(self._scanned, end)):
                row = self.history[i]
                if row["send_returned_successfully"]:
                    self._returned += 1
                elif row.get("error") or not row["allowed"]:
                    self._uncertain |= bool(row.get("transmission_outcome_uncertain"))
                else:
                    # A concurrent native send can finish after this snapshot.
                    pending.add(i)
            self._scanned, self._pending = end, pending
            return dict(
                tx_attempts=end,
                denied_tx_attempts=self.denied,
                actual_tx_count=self._returned,
                actual_tx_count_definition="native bus.send returned; not wire acknowledgement",
                transmission_outcome_uncertain=self._uncertain,
                guard_installed=self.installed,
                transmit_permitted=self.allowed,
                history_omitted=True,
            )


def start_tolerance(request):
    value = request.get('start_tolerance_deg', .05)
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not .05 <= value <= .5:
        raise ValueError('start_tolerance_deg must be 0.05..0.5')
    if request.get('load_context') != 'green_cup_held' and value != .05:
        raise ValueError('Expanded start tolerance is only for green held-cup pipeline')
    return value


def check_start_rows(request, rows, report):
    plan = request["plan"]
    tolerance = start_tolerance(request)
    report['start_feedback'] = rows
    report['start_tolerance_deg'] = tolerance
    report['start_max_error_deg'] = max(math.degrees(abs(a-b)) for row in rows
        for a,b in zip(row['q_rad'], plan['start_q_rad']))
    for row in rows:
        if core.ready_blockers(row, take_can_control=True):
            raise RuntimeError('摇晃起点控制状态异常')
        if max(
            abs(a - b) for a, b in zip(row["q_rad"], plan["start_q_rad"])
        ) > math.radians(tolerance):
            report['failure_code'] = 'start_position_changed'
            raise RuntimeError(f"当前姿态或控制状态与计划不同；最大起点误差 {report['start_max_error_deg']:.4f}°，容差 {tolerance}°；重新 plan，不自动移动到旧起点")


def prepare_control(request, session, baseline, report, *, persistent=False):
    """Reuse an owned CAN connection; a standalone test still performs handoff."""
    if persistent and baseline['status']['ctrl_mode'] == 1:
        fresh = core.fresh_feedback(session, previous=baseline)
        # Validate against the actual planned start, not a second stricter
        # handoff tolerance inherited from the standalone micro-move probe.
        recheck = {}
        try:
            check_start_rows(request, [fresh], recheck)
        finally:
            report['can_reuse_start_check'] = recheck
        if fresh['status']['ctrl_mode'] != 1 or fresh['status']['motion_status'] != 0:
            raise RuntimeError('Persistent shake requires fresh idle CAN control')
        return dict(mode_command_sent=False, joint_target_sent=False,
                    reused_can_control=True, samples=[fresh])
    return take_js_control(core, session, baseline, timeout_s=3)


def validate_request(request, now):
    start_tolerance(request)
    if type(request.get('require_center_position', True)) is not bool:
        raise ValueError('require_center_position must be boolean')
    if request.get('require_center_position') is False and request.get('load_context') != 'green_cup_held':
        raise ValueError('Only green pipeline may use idle-only completion')
    plan = request["plan"]
    options(plan["parameters"])
    load_ok = request.get("empty_hand_clear_path_confirmed") is True
    if request.get('load_context') == 'green_cup_held':
        from cup_grasp_demo.calibration_debug.green_shake_validation import validate_held_request
        validate_held_request(request)
        load_ok = True
    if (
        request.get("execution_authorized") is not True
        or not 0 <= now - request["authorized_epoch_s"] <= 60
        or not load_ok
        or plan.get("kind") != KIND
        or plan.get("offline_only")
        or not plan.get("planning_passed")
        or plan.get("blockers")
        or not 0 < plan["duration_s"] <= 60
        or request["table_screen"]["blockers"]
        or not 1 <= request["restore_speed_percent"] <= (100 if request.get("load_context") == "green_cup_held" else 10)
    ):
        raise ValueError("Invalid, stale, offline or blocked joint test")
    for filename, expected in request["input_hashes"].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise ValueError("Reviewed input changed: " + filename)


def live_plan(robot, plan):
    limits = []
    for i in range(1, 8):
        v = robot.get_joint_angle_vel_limits(i, timeout=0.25, min_interval=0)
        a = robot.get_joint_acc_limits(i, timeout=0.25, min_interval=0)
        if v is None or a is None:
            raise RuntimeError("缺少实时速度或加速度限值")
        limits.append(
            dict(
                joint=i,
                min_angle_rad=v.msg.min_angle_limit,
                max_angle_rad=v.msg.max_angle_limit,
                max_velocity_rad_s=v.msg.max_joint_spd,
                max_acceleration_rad_s2=a.msg.max_joint_acc,
            )
        )
    verified = make_plan(
        dict(success=True, limits=limits, q_after_rad=plan["start_q_rad"]),
        plan["parameters"],
        core.joint_limits(robot),
    )
    if (
        verified["blockers"]
        or verified["samples"] != plan["samples"]
        or verified["segments"] != plan["segments"]
        or verified["duration_s"] != plan["duration_s"]
        or verified["amplitude_rad"] != plan["amplitude_rad"]
    ):
        raise ValueError("轨迹或实时限值与计划不一致：" + str(verified["blockers"]))
    return limits


def run(request, *, connected=None, connection_evidence=None):
    validate_request(request, time.time())
    if connected is not None and request.get('load_context') != 'green_cup_held':
        raise ValueError('Persistent shake is restricted to the green pipeline')
    if connected is not None and connection_evidence is None:
        raise ValueError("Persistent shake requires original pre-connection evidence")
    plan = request["plan"]
    cfg = plan["parameters"]
    bus, factory = core.load_sdk_runtime(request["channel"])
    guard = JointGuard(bus)
    session = core.PassivePoseSession(bus, factory, deadline_s=2)
    session.guard = guard
    report = dict(
        sdk_method="move_js",
        success=False,
        duration_completed=False,
        motion_attempted=False,
        parameter_write_commands_sent=0,
        finger_commands_sent=0,
        feedback=[],
        commands=[],
        started_epoch_s=time.time(),
    )
    controlled = False
    limits = None
    try:
        before = (connection_evidence if connected is not None
                  else shared.control_evidence(request["channel"], request))
        report["host_control_before_connect"] = before
        if connected is None:
            report["sdk_runtime"] = session.start()
            rows, report["stationarity"] = core.stopped_window(session)
        else:
            guard.install()
            session.robot = connected
            session.started = True
            report["sdk_runtime"] = {"reused_connection": True}
            rows = [core.fresh_feedback(session)]
            rows.append(core.fresh_feedback(session, previous=rows[-1]))
            spans = [abs(a-b) for a,b in zip(rows[0]['q_rad'], rows[1]['q_rad'])]
            if max(spans) > core.TOLERANCE_RAD:
                raise RuntimeError('Persistent shake start is not stationary')
            report["stationarity"] = {"fresh_samples": 2, "joint_spans_deg": [math.degrees(v) for v in spans]}
        check_start_rows(request, rows, report)
        limits = core.joint_limits(session.robot)
        for sample in plan["samples"]:
            core.validate_target(sample["q_rad"], limits)
        for check in plan["model_flange_checks"]:
            pose = session.robot.fk(check["q_rad"])
            actual = shared.rotation(pose)
            target = check["T_base_flange"]
            if (
                math.dist(pose[:3], [row[3] for row in target[:3]]) > 0.00002
                or max(
                    abs(actual[i][j] - target[i][j]) for i in range(3) for j in range(3)
                )
                > 0.0001
            ):
                raise RuntimeError("SDK FK 与规划模型不一致")
        current = shared.control_evidence(request["channel"], request)
        report["host_control_before_motion"] = current
        conflicts = shared.control_conflicts(before, current)
        if conflicts:
            raise RuntimeError("; ".join(conflicts))
        guard.permit()
        report["live_limits"] = live_plan(session.robot, plan)
        guard.motion_allowed = True
        controlled = True
        session.robot.set_joint_limits_enabled(True)
        session.robot.set_speed_percent(cfg["controller_speed_percent"])
        report["can_mode"] = prepare_control(request, session, rows[-1], report,
                                             persistent=connected is not None)
        previous = report["can_mode"]["samples"][-1]
        feedback_check(previous, plan, 0.0)
        if not 0 <= time.time() - request["authorized_epoch_s"] <= 60:
            raise RuntimeError("Joint-test preflight expired")
        start = time.monotonic()
        report["motion_started_epoch_s"] = time.time()
        last_q = plan["start_q_rad"]
        last_time = 0.0
        while True:
            row = core.fresh_feedback(session, previous=previous)
            previous = row
            elapsed = time.monotonic() - start
            if elapsed - last_time > cfg["command_lag_s"]:
                raise RuntimeError("反馈/指令间隔超限；不能跳过大段运动")
            observation = dict(
                elapsed_s=elapsed,
                feedback=row,
                raw_error_deg=reference_errors(row, plan, elapsed),
                error_deg=reference_errors(
                    row, plan, elapsed, cfg.get("feedback_reference_delay_s", 0.0)
                ),
                tracking_check_passed=False,
            )
            observation["tracking_threshold_exceeded"] = tracking_exceedances(
                observation["error_deg"], cfg
            )
            report["last_observation"] = observation
            report["feedback"].append(observation)
            feedback_check(row, plan, elapsed)
            observation["tracking_check_passed"] = not observation["tracking_threshold_exceeded"]
            target = joint_values(plan, min(elapsed, plan['duration_s']))
            if elapsed >= plan["duration_s"]:
                target = list(plan["start_q_rad"])
            if max(abs(a - b) for a, b in zip(target, last_q)) > math.radians(
                cfg["command_step_deg"]
            ):
                raise RuntimeError("相邻指令角度变化超限")
            if not session.robot.get_joint_limits_enabled():
                raise RuntimeError("SDK 关节限位被关闭")
            report["motion_attempted"] = True
            session.robot.move_js(target)
            report["commands"].append(
                dict(elapsed_s=time.monotonic() - start, target_q_rad=target)
            )
            last_q, last_time = target, elapsed
            if elapsed >= plan["duration_s"]:
                break
            time.sleep(0.001)
        report["reference_send_elapsed_s"] = time.monotonic() - start
        report["duration_completed"] = (
            report["reference_send_elapsed_s"] >= plan["duration_s"]
        )
        report["center_settle"] = []
        require_center = request.get('require_center_position', True)
        shared.verify_stop(session, plan["start_q_rad"], report["center_settle"],
                           **({"require_position": False} if not require_center else {}),
                           **({"stable_samples": 3, "poll_s": 0.0} if connected is not None else {}))
        report["returned_center"] = require_center
        report["stopped_verified"] = True
        report["completion_basis"] = 'center_position' if require_center else 'fresh_normal_idle'
        report["success"] = True
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        if "motion_started_epoch_s" in report:
            report["motion_elapsed_s"] = time.monotonic() - start
        if controlled and limits is not None:
            try:
                report["failure_hold"] = fresh_js_hold(core, session, limits)
                report["hold_feedback"] = []
                shared.verify_stop(
                    session,
                    report["failure_hold"]["target_rad"],
                    report["hold_feedback"],
                )
                report["failure_hold"]["hold_verified"] = True
            except BaseException as hold_error:
                report["hold_error"] = str(hold_error)
    finally:
        if controlled and (
            report.get("returned_center")
            or report.get("stopped_verified")
            or report.get("failure_hold", {}).get("hold_verified")
        ):
            try:
                session.robot.set_speed_percent(request["restore_speed_percent"])
                report["restored_speed_percent"] = request["restore_speed_percent"]
            except BaseException as error:
                report["restore_error"] = str(error)
                report["success"] = False
        guard.allowed = False
        if connected is None:
            session.close()
        else:
            guard.restore()
        report["tx"] = guard.report(include_history=True)
        report["sdk_disconnected"] = connected is None
        report["measurement"] = measurements(report["feedback"], plan)
        violations = [
            row for row in report["feedback"] if row["tracking_threshold_exceeded"]
        ]
        report["tracking_summary"] = dict(
            action=cfg.get("tracking_error_action", "stop"),
            threshold_deg=cfg["tracking_error_deg"],
            exceeded_samples=len(violations),
            first_exceeded_s=violations[0]["elapsed_s"] if violations else None,
            max_abs_error_deg=[
                max((abs(row["error_deg"][i]) for row in report["feedback"]), default=None)
                for i in range(7)
            ],
        )
        report["physical_limit_verified"] = False
        report["ended_epoch_s"] = time.time()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.request.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Request hash changed")
    request = json.loads(raw)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, shared.interrupted)
    try:
        with core.control_lock(
            Path("/tmp/nero_" + request["channel"] + "_control.lock")
        ):
            report = run(request)
    except BaseException as error:
        report = dict(
            success=False,
            error=type(error).__name__ + ": " + str(error),
            motion_attempted=False,
        )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report.get(k)
                for k in ("success", "error", "duration_completed", "measurement")
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
