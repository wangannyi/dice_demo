"""Bounded planar JS trial, independent of the established pipeline executor."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
from nero_revo2_control.bridges.passive_pose_bridge import _packet_timestamps
from cup_grasp_demo.flow import shake_execution as legacy
from cup_grasp_demo.flow.parameters import number
from cup_grasp_demo.flow.phase_timing import PhaseTimer
from cup_grasp_demo.flow.joint_delivery import take_js_control, fresh_js_hold
from cup_grasp_demo.flow.shake_readback import is_read_only_request, query_joint_limit
from cup_grasp_demo.flow.shake_tracking import measured_wave, rotation, state_ok

core = legacy.core


class JSGuard(core.AuditedSendGuard):
    """Permit JS joint targets and handoff/holding, never MoveJ/torque/hand writes."""

    motion_allowed = False

    def install(self):
        if self.installed:
            return
        super().install()
        audited = self.bus_class.send

        def send(bus, message, *args, **kwargs):
            ident, data = message.arbitration_id, bytes(message.data)
            mode = (ident == 0x151 and len(data) == 8 and data[0] == 1
                    and 1 <= data[2] <= 97 and not any(data[4:])
                    and ((data[1] == 1 and data[3] == 0xAD)
                         # SDK speed-only updates preserve the motion mode (255).
                         or (data[1] == 255 and data[3] in (0, 0xAD))))
            joints = len(data) == 8 and (ident in (0x155, 0x156, 0x157)
                                        or ident == 0x170 and not any(data[4:]))
            if not is_read_only_request(ident, data) and not (self.motion_allowed and (mode or joints)):
                raise RuntimeError("JS guard rejected CAN frame " + hex(ident))
            return audited(bus, message, *args, **kwargs)

        self.bus_class.send = send


def poll_feedback(robot, previous_stamps=None, *, wallclock=time.time):
    """Copy already-received SDK data once. Never wait for the next CAN packet."""
    now = wallclock()
    status = robot.get_arm_status()
    if status is None or not 0 <= now - status.timestamp <= .5:
        raise RuntimeError("Arm status feedback lost for 500 ms")
    state = status.msg
    enabled = robot.get_joints_enable_status_list()
    row = dict(status=dict(arm_status=int(state.arm_status), ctrl_mode=int(state.ctrl_mode),
                           motion_status=int(state.motion_status)), enabled=list(enabled))
    if not state_ok(row):
        raise RuntimeError("控制器故障、控制模式变化或七轴使能异常")
    for i in range(1, 8):
        driver = robot.get_driver_states(joint_index=i)
        if driver is None or not 0 <= now - driver.timestamp <= .5:
            raise RuntimeError("Joint enable feedback lost for 500 ms")
    before = _packet_timestamps(robot)
    if before is None or any(not 0 <= now - s <= .25 for s in before.values()):
        raise RuntimeError("Joint feedback lost for 250 ms")
    if before == previous_stamps:
        return None, previous_stamps
    message = robot.get_joint_angles()
    if message is None:
        return None, previous_stamps
    q = list(message.msg)
    after = _packet_timestamps(robot)
    if before != after or max(before.values()) - min(before.values()) > .02:
        return None, previous_stamps
    if len(q) != 7 or not all(math.isfinite(v) for v in q):
        raise RuntimeError("Invalid joint feedback")
    row.update(q_rad=q, fk_flange_pose_m_rad=list(robot.fk(q)),
               observed_epoch_s=now, joint_packet_timestamps=before)
    return row, before


def tcp_metrics(row, plan):
    pose = row["fk_flange_pose_m_rad"]
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise RuntimeError("Invalid FK feedback")
    r = rotation(pose)
    tcp = plan["T_flange_virtual_tcp"]
    xyz = [pose[i] + sum(r[i][j] * tcp[j][3] for j in range(3)) for i in range(3)]
    rt = [[sum(r[i][k] * tcp[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    rpy = [math.atan2(rt[2][1], rt[2][2]),
           math.atan2(-rt[2][0], math.hypot(rt[0][0], rt[1][0])),
           math.atan2(rt[1][0], rt[0][0])]
    angles = [math.degrees((a-b+math.pi) % (2*math.pi)-math.pi)
              for a, b in zip(rpy, plan["tcp_rpy_center_rad"])]
    delta = [xyz[i] - plan["T_base_tcp_center"][i][3] for i in range(3)]
    direction = plan["direction_base"]
    return dict(along_m=sum(a*b for a, b in zip(delta, direction)),
                cross_m=-direction[1]*delta[0]+direction[0]*delta[1],
                height_m=delta[2], rotation_deg=abs(angles[2]),
                rx_error_deg=angles[0], ry_error_deg=angles[1])


def check_geometry(row, plan, bounds):
    """Reject loss of the constrained plane/envelope, not waveform phase error."""
    q = row["q_rad"]
    for i, (lo, hi) in enumerate(plan["joint_limits_rad"]):
        if not lo < q[i] < hi or not bounds[i][0] - math.radians(5) <= q[i] <= bounds[i][1] + math.radians(5):
            raise RuntimeError(f"J{i+1} 超出机械关节范围或摇晃空间包络")
    values = tcp_metrics(row, plan)
    if abs(values["height_m"]) > .005 or max(abs(values[k]) for k in ("rx_error_deg", "ry_error_deg")) > 5:
        raise RuntimeError("实际 TCP 脱离固定 Z/Rx/Ry 的运动平面")
    return values


def stream(robot, plan, report, *, clock=time.monotonic, sleep=time.sleep, poll=poll_feedback):
    """100 Hz schedule; polling never waits. Late slots are skipped, never replayed."""
    samples = plan["samples"]
    hz, duration = plan["sample_hz"], plan["parameters"]["duration_s"]
    bounds = [(min(s["q_rad"][i] for s in samples), max(s["q_rad"][i] for s in samples)) for i in range(7)]
    start = clock()
    last_index, stamps, last_q = -1, None, plan["start_q_rad"]
    last_good, last_sent = start, start
    report.update(motion_elapsed_s=0., duration_completed=False, commands=[], feedback=[])
    try:
        while True:
            now = clock()
            elapsed = now - start
            index = min(len(samples)-1, int(elapsed*hz))
            if index <= last_index:
                sleep(min(.002, max(0, (last_index+1)/hz-elapsed)))
                continue
            row, stamps = poll(robot, stamps)
            if row is not None:
                last_good = clock()
                metrics = check_geometry(row, plan, bounds)
                metrics["following_error_mm"] = (metrics["along_m"]-samples[index]["displacement_m"])*1000
                report["feedback"].append(dict(elapsed_s=last_good-start, feedback=row, tracking=metrics))
            if clock()-last_good > .25:
                raise RuntimeError("No coherent joint feedback for 250 ms")
            if clock()-last_sent > .10:
                raise RuntimeError("发送线程停顿超过 100 ms，停止补发跳变目标")
            if not robot.get_joint_limits_enabled():
                raise RuntimeError("SDK joint limits disabled")
            # Re-evaluate after the non-blocking copy/FK work.
            index = min(len(samples)-1, int((clock()-start)*hz))
            sample = samples[index]
            if max(abs(a-b) for a, b in zip(sample["q_rad"], last_q)) > math.radians(3):
                raise RuntimeError("相邻发送目标跳变超过 3°")
            report["motion_attempted"] = True
            robot.move_js(sample["q_rad"])
            last_sent = clock()
            report["commands"].append(dict(index=index, elapsed_s=last_sent-start, target_q_rad=sample["q_rad"]))
            last_q, last_index = sample["q_rad"], index
            if index == len(samples)-1:
                report["duration_completed"] = last_sent-start >= duration
                break
    finally:
        report["motion_elapsed_s"] = clock()-start
        report["measured_wave"] = measured_wave(report["feedback"], plan)
        report["send_rate_hz"] = len(report["commands"])/max(report["motion_elapsed_s"], 1e-9)


def settle(session, target, report):
    """JS has no MoveJ-arrival requirement: use fresh position and stationarity."""
    deadline, stamps, stable, previous = time.monotonic()+6, None, 0, None
    while time.monotonic() < deadline:
        row, stamps = poll_feedback(session.robot, stamps)
        if row is not None:
            report.append(row)
            close = max(abs(a-b) for a, b in zip(row["q_rad"], target)) < math.radians(.2)
            still = previous is not None and max(abs(a-b) for a, b in zip(row["q_rad"], previous)) < math.radians(.05)
            stable = stable+1 if close and still else 0
            if stable >= 10:
                return
            previous = row["q_rad"]
        session.robot.move_js(target)
        time.sleep(.01)
    raise TimeoutError("JS 已结束发送，但未确认回中心停稳")


def validate(request):
    p = request["plan"]
    number(p["parameters"].get("limit_utilization", .97), "limit_utilization", .1, 1.)
    if (not request.get("execution_authorized") or not 0 <= time.time()-request["authorized_epoch_s"] <= 60
            or p.get("offline_only") or p.get("kind") != "planar_js_shake_trial"
            or not p.get("planning_passed") or p["blockers"] or request["table_screen"]["blockers"]
            or request.get("load") not in ("empty", "cup") or not request.get("scene_observed")):
        raise ValueError("Invalid/stale/offline/blocked JS request")
    duration = p["parameters"]["duration_s"]
    if not .5 <= duration <= 60 or p["sample_hz"] != 100 or len(p["samples"]) != math.ceil(duration*100)+1:
        raise ValueError("Malformed JS duration/samples")
    for i, s in enumerate(p["samples"]):
        if abs(s["t_s"]-duration*i/(len(p["samples"])-1)) > 1e-8:
            raise ValueError("Malformed JS time base")
        core.validate_target(s["q_rad"], p["joint_limits_rad"])
    for row in (p["samples"][0], p["samples"][-1]):
        if max(abs(a-b) for a, b in zip(row["q_rad"], p["start_q_rad"])) > 1e-8:
            raise ValueError("JS endpoint differs from center")
    for path, expected in request["input_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("JS input changed: " + path)


def run(request):
    timer = PhaseTimer()
    with timer.phase("REQUEST_CHECK 请求校验"):
        validate(request)
    plan = request["plan"]
    with timer.phase('SDK_LOAD SDK 加载'):
        bus, factory = core.load_sdk_runtime(request["channel"])
        guard = JSGuard(bus)
        session = core.PassivePoseSession(bus, factory)
        session.guard = guard
    report = dict(success=False, motion_attempted=False, duration_completed=False,
                  backend="move_js", finger_commands_sent=0, load=request["load"],
                  performance_error_action="record", cup_retention_verified=False)
    report["phase_timings"] = timer.records
    controlled, limits = False, None
    try:
        with timer.phase('CONNECT_READY 连接与起始姿态复核'):
            before = legacy.control_evidence(request["channel"], request)
            report["sdk_runtime"] = session.start()
            rows, report["stationarity"] = core.stopped_window(session)
            previous = rows[-1]
            for row in rows:
                if core.ready_blockers(row, take_can_control=True) or max(abs(a-b) for a, b in zip(row["q_rad"], plan["start_q_rad"])) > math.radians(.05):
                    raise RuntimeError("当前姿态与计划不一致，重新规划")
        with timer.phase('FK_CHECK 全轨迹 SDK 正解复核'):
            limits = core.joint_limits(session.robot)
            for sample in plan["samples"]:
                core.validate_target(sample["q_rad"], limits)
                pose = session.robot.fk(sample["q_rad"])
                expected = sample["flange_pose_m_rad"]
                if math.dist(pose[:3], expected[:3]) > .00002 or max(abs((a-b+math.pi)%(2*math.pi)-math.pi) for a,b in zip(pose[3:], expected[3:])) > .0001:
                    raise RuntimeError("SDK FK 与 JS 规划不一致")
        with timer.phase('LIVE_LIMITS 控制器状态、固件与限值读取'):
            conflicts = legacy.control_conflicts(before, legacy.control_evidence(request["channel"], request))
            if conflicts:
                raise RuntimeError("; ".join(conflicts))
            guard.permit()
            firmware = session.robot.get_firmware(timeout=1, min_interval=0)
            report["firmware"] = firmware
            if firmware.get("software_version") != "1.20":
                raise RuntimeError("本 JS 实验仅核验过固件 1.20 的协议")
            utilization = plan["parameters"].get("limit_utilization", .97)
            for i in range(1, 8):
                v = query_joint_limit(session.robot.get_joint_angle_vel_limits, i)
                a = query_joint_limit(session.robot.get_joint_acc_limits, i)
                if v is None or a is None:
                    raise RuntimeError("无法读取实时关节限值")
                if (plan["joint_peak_velocity_rad_s"][i-1] > utilization*v.msg.max_joint_spd
                        or plan["joint_peak_acceleration_rad_s2"][i-1] > utilization*min(5., a.msg.max_joint_acc)
                        or any(not v.msg.min_angle_limit+math.radians(1) <= s["q_rad"][i-1] <= v.msg.max_angle_limit-math.radians(1) for s in plan["samples"])):
                    raise RuntimeError("实时关节限值低于 JS 轨迹需求")
        with timer.phase('TAKE_CONTROL 控制权与 JS 模式切换'):
            guard.motion_allowed = True
            controlled = True
            session.robot.set_joint_limits_enabled(True)
            session.robot.set_speed_percent(97)
            take_js_control(core, session, previous, timeout_s=3)
            session.robot.set_motion_mode("js")
            if time.time()-request["authorized_epoch_s"] > 60:
                raise RuntimeError("JS preflight expired")
        with timer.phase('SHAKE 摇晃发送与反馈统计'):
            stream(session.robot, plan, report)
        with timer.phase('CENTER_SETTLE 回中心停稳'):
            report["center_settle"] = []
            settle(session, plan["start_q_rad"], report["center_settle"])
            report["returned_center"] = True
        report["success"] = report["duration_completed"] and report["measured_wave"]["tracking_verified"]
        if not report["success"]:
            report["error"] = "已按指定时长发送；实际频率/幅度未通过验收，查看 measured_wave"
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
        if controlled and limits is not None:
            try:
                with timer.phase('FAILURE_HOLD 异常保持'):
                    report["failure_hold"] = fresh_js_hold(core, session, limits)
                    report["hold_feedback"] = []
                    settle(session, report["failure_hold"]["target_rad"], report["hold_feedback"])
                    report["failure_hold"]["hold_verified"] = True
            except BaseException as hold_error:
                report["hold_error"] = str(hold_error)
    finally:
        with timer.phase('CLEANUP 恢复速度与断开 SDK'):
            if controlled and (report.get("returned_center") or report.get("failure_hold", {}).get("hold_verified")):
                try:
                    session.robot.set_speed_percent(request["restore_speed_percent"])
                except BaseException as error:
                    report["restore_error"] = str(error)
                    report["success"] = False
            guard.allowed = False
            try:
                session.close()
                report["sdk_disconnected"] = True
            except BaseException as error:
                report["close_error"] = str(error)
                report["success"] = False
            report["tx"] = guard.report()
    report["phase_total_s"] = timer.summary("SDK 执行器")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.request.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256 or args.output.exists():
        raise ValueError("Changed request or existing receipt")
    request = json.loads(raw)
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, legacy.interrupted)
    with core.control_lock(Path("/tmp/nero_"+request["channel"]+"_control.lock")):
        report = run(request)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: report.get(k) for k in ("success", "error", "duration_completed", "motion_elapsed_s", "measured_wave")}, ensure_ascii=False))
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
