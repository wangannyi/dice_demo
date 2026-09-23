#!/usr/bin/env python3
"""Nero 7-axis arm and right Revo2 position-control demo for pyAgxArm."""

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
import json
import math
import re
import sys
import time

from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


FINGER_NAMES = (
    "thumb_tip",
    "thumb_base",
    "index_finger",
    "middle_finger",
    "ring_finger",
    "pinky_finger",
)

FINGER_ALIASES = {
    "thumb_tip": "thumb_tip",
    "thumb_base": "thumb_base",
    "index": "index_finger",
    "index_finger": "index_finger",
    "middle": "middle_finger",
    "middle_finger": "middle_finger",
    "ring": "ring_finger",
    "ring_finger": "ring_finger",
    "pinky": "pinky_finger",
    "pinky_finger": "pinky_finger",
}

# Six values always use FINGER_NAMES order.  half_close/fist are converted from
# AgileX's right-Revo2 SRDF.  This rig briefly reported 18/3 after a command
# toward the SRDF-derived 7/0, but subsequently drifted to 46/8.  The open
# endpoint is therefore an unverified candidate, not a calibrated hold target.
# The remaining named gestures are initial tuning values.
GESTURES = {
    "open": (18, 3, 0, 0, 0, 0),
    "half-close": (61, 50, 35, 35, 35, 35),
    "fist": (60, 5, 99, 99, 99, 99),
    "ok": (52, 68, 42, 0, 0, 0),
    "yeah": (55, 5, 0, 0, 90, 90),
    "thumbs-up": (7, 0, 90, 90, 90, 90),
}

TUNABLE_GESTURES = {"ok", "yeah", "thumbs-up"}
RIG_CANDIDATE_GESTURES = {"open"}
COMMAND_ALIASES = {
    "arm-enable": "enable",
    "read_joints": "read-joints",
    "move_j": "move-j",
    "move_p": "move-p",
}

# The vendor's ROS/MoveIt ``home`` state is seven zeros.  That puts J2 and J4
# at the singular position explicitly forbidden by the Nero user guide.  This
# rig-specific ready pose keeps J2 and J4 away from zero.  A valid endpoint
# does not certify that a path from an arbitrary current posture is clear.
READY_HOME_DEG = (55.0, -78.0, 80.0, -45.0, 130.0, -30.0, 40.0)
READY_HOME_RAD = tuple(math.radians(value) for value in READY_HOME_DEG)
# Manual maximum speeds are used only to size the home observation window.
# They do not predict the actual time taken by a controller-planned path.
NERO_MAX_JOINT_SPEED_DEG_S = (180.0, 180.0, 180.0, 225.0, 225.0, 225.0, 225.0)

OUTPUT_FORMAT = ContextVar("nero_revo2_output_format", default="json")

# Mechanical ranges from the Nero user manual.  pyAgxArm's built-in target
# limits are deliberately narrower and are still used for newly commanded axes.
NERO_DOCUMENTED_LIMITS_RAD = tuple(
    (math.radians(lower), math.radians(upper))
    for lower, upper in (
        (-157, 157),
        (-102, 102),
        (-160, 160),
        (-60, 125),
        (-160, 160),
        (-44, 57),
        (-97, 97),
    )
)


@contextmanager
def use_output_format(mode):
    """Select event rendering for this operation without changing other callers."""
    if mode not in ("json", "human"):
        raise ValueError(f"unknown output format: {mode}")
    token = OUTPUT_FORMAT.set(mode)
    try:
        yield
    finally:
        OUTPUT_FORMAT.reset(token)


def _joint_angles(joints_rad):
    """Show signed J1..J7 degree numbers in order on one copyable line."""
    if joints_rad is None:
        return "未提供"
    return "  ".join(f"{math.degrees(float(value)):+.3f}" for value in joints_rad)


def _pose_lines(label, pose):
    if pose is None:
        return [f"{label}：未提供"]
    x, y, z, roll, pitch, yaw = (float(value) for value in pose)
    return [
        f"{label}位置 (m)：{x:+.6f}  {y:+.6f}  {z:+.6f}",
        f"{label}姿态 (°)：{math.degrees(roll):+.3f}  "
        f"{math.degrees(pitch):+.3f}  {math.degrees(yaw):+.3f}",
    ]


def _limit_text(limits):
    if limits is None:
        return "未提供"
    lower, upper = (float(value) for value in limits)
    return f"{lower:+.1f}～{upper:+.1f}"


def _finger_text(values):
    if values is None:
        return "未收到反馈"
    return "  ".join(str(values[name]) for name in FINGER_NAMES)


def _status_field(status, name):
    match = re.search(rf"^\s*{re.escape(name)}:\s*(.+)$", status, re.MULTILINE)
    if not match:
        return "未提供"
    raw = match.group(1)
    value = raw.split("(", 1)[0]
    labels = {
        "CAN_CTRL": "CAN 控制",
        "ETHERNET_CONTROL_MODE": "以太网控制",
        "NORMAL": "正常",
        "JOINT_BRAKE_NOT_RELEASED": "抱闸未释放",
        "EMERGENCY_STOP": "急停",
        "MOVE_J": "关节运动",
        "MOVE_P": "笛卡尔点位运动",
        "REACH_TARGET_POS_SUCCESSFULLY": "已到目标",
    }
    return labels.get(value, raw)


def format_event(payload):
    """Render one SDK event as concise terminal text; JSON payloads stay intact."""
    event = payload.get("event", "unknown")
    if event == "read_joints":
        enabled = payload.get("joints_enabled") or []
        arm = {0: "正常", 1: "急停", 6: "抱闸未释放"}.get(
            payload.get("arm_status"), str(payload.get("arm_status"))
        )
        ctrl = {1: "CAN", 3: "以太网"}.get(
            payload.get("ctrl_mode"), str(payload.get("ctrl_mode"))
        )
        lines = [
            "Nero 七轴关节角度",
            f"状态：{arm}  |  控制：{ctrl}  |  使能：{sum(bool(v) for v in enabled)}/7",
            "关节    当前角度(°)    使能       SDK 软限位(°)            手册机械范围(°)",
        ]
        sdk_limits = payload.get("sdk_limits_deg")
        documented = payload.get("documented_limits_deg")
        within = payload.get("within_sdk_limits")
        for index, angle in enumerate(payload.get("joints_deg", [])):
            sdk = _limit_text(sdk_limits[index]) if sdk_limits else "未提供"
            manual = _limit_text(documented[index]) if documented else "未提供"
            state = "已使能" if index < len(enabled) and enabled[index] else "未使能"
            warning = "  [超软限位]" if within and not within[index] else ""
            lines.append(
                f"J{index + 1:<5} {float(angle):+8.3f}     {state:<6}    "
                f"{sdk:<22} {manual}{warning}"
            )
        return "\n".join(lines)

    if event == "arm_status":
        status = str(payload.get("status", ""))
        errors = re.findall(r"^\s*([a-zA-Z0-9_]+):\s*True\s*$", status, re.MULTILINE)
        checked = re.findall(
            r"^\s*([a-zA-Z0-9_]+):\s*(?:True|False)\s*$", status, re.MULTILINE
        )
        lines = [
            "Nero 机械臂状态",
            f"控制：{_status_field(status, 'ctrl_mode')}  |  机械臂：{_status_field(status, 'arm_status')}",
            f"运动模式：{_status_field(status, 'mode_feedback')}  |  "
            f"运动结果：{_status_field(status, 'motion_status')}",
            f"七轴使能：{sum(bool(v) for v in payload.get('joints_enabled', []))}/7",
            "关节角 (°)：" + _joint_angles(payload.get("joints_rad")),
            *_pose_lines("法兰", payload.get("flange_m_rad")),
        ]
        if "err_status:" in status:
            lines.append(
                "限位/通信异常："
                + (
                    ", ".join(errors)
                    if errors
                    else "未发现"
                    if checked
                    else "详细字段未提供"
                )
            )
        return "\n".join(lines)

    if event == "fk":
        lines = [
            "正运动学 FK（基座坐标系）",
            "关节角 (°)：" + _joint_angles(payload.get("joints_rad")),
            *_pose_lines("法兰", payload.get("flange_m_rad")),
        ]
        if payload.get("tcp_m_rad") is not None:
            lines.extend(_pose_lines("TCP", payload["tcp_m_rad"]))
            lines.extend(
                _pose_lines(
                    "法兰→TCP 偏移", payload.get("tcp_offset_flange_to_tcp_m_rad")
                )
            )
        if payload.get("position_difference_m") is not None:
            lines.append(
                f"与控制器反馈差：位置 {float(payload['position_difference_m']) * 1000:.3f} mm  "
                f"姿态 (°) {math.degrees(float(payload['orientation_difference_rad'])):.3f}"
            )
        return "\n".join(lines)

    if event == "hand_status":
        available = payload.get("available_feedback") or []
        lines = [
            "右 Revo2 灵巧手反馈",
            f"反馈类别：{', '.join(available) if available else '无'}  |  SDK 监测：{'正常' if payload.get('is_ok') else '异常'}",
            "六路位置 (0..100)：" + _finger_text(payload.get("positions")),
        ]
        for key, label in (
            ("currents", "六路电流"),
            ("speeds", "六路速度"),
            ("motor_status", "电机状态"),
        ):
            if payload.get(key) is not None:
                lines.append(f"{label}：" + _finger_text(payload[key]))
        return "\n".join(lines)

    if event == "arm_plan":
        command = payload.get("command", "运动")
        lines = [
            f"机械臂{'执行' if payload.get('execute') else '预览'}计划：{command}  "
            f"|  速度 {payload.get('speed_percent', '?')}%",
        ]
        if command == "home" and payload.get("wait_timeout_s") is not None:
            source = {"auto": "自动", "manual": "手动"}.get(
                payload.get("timeout_source"), "未标明"
            )
            lines.append(f"等待上限：{payload['wait_timeout_s']:g} 秒（{source}）")
        lines.extend(
            [
                "当前关节 (°)：" + _joint_angles(payload.get("current_joints_rad")),
                *_pose_lines("当前法兰", payload.get("current_flange_m_rad")),
            ]
        )
        if payload.get("target_frame") == "joint":
            lines.append("目标关节 (°)：" + _joint_angles(payload.get("flange_target")))
        else:
            lines.extend(_pose_lines("目标法兰", payload.get("flange_target")))
            if payload.get("target_frame") == "tcp":
                lines.extend(_pose_lines("目标 TCP", payload.get("requested_target")))
        return "\n".join(lines)

    if event == "hand_plan":
        return "\n".join(
            [
                f"灵巧手{'执行' if payload.get('execute') else '预览'}计划："
                f"{payload.get('gesture') or payload.get('control', '位置控制')}  "
                f"|  到位时间 {payload.get('duration_s', '?')} 秒",
                "当前六路：" + _finger_text(payload.get("current")),
                "目标六路：" + _finger_text(payload.get("target")),
            ]
        )

    if event == "arm_enable_plan":
        return (
            f"七轴使能{'执行' if payload.get('execute') else '预览'}计划\n"
            f"当前使能：{sum(bool(v) for v in payload.get('joints_enabled', []))}/7  "
            f"|  最多尝试 {payload.get('attempts', '?')} 次\n"
            "当前关节 (°)：" + _joint_angles(payload.get("current_joints_rad"))
        )

    if event == "system_prepare_plan":
        return (
            f"准备{'执行' if payload.get('execute') else '预览'}计划："
            "使能七轴 → ready_home → Revo2 open → 重连观察\n"
            "当前关节 (°)："
            + _joint_angles(
                [math.radians(value) for value in payload.get("current_joints_deg", [])]
            )
            + "\n"
            "当前手指：" + _finger_text(payload.get("current_hand_positions"))
        )

    if event == "failed":
        return f"操作失败：{payload.get('error', '原因未提供')}"
    if event == "preview_only":
        return "预览完成：未发送运动、使能或手部控制指令。"
    if event == "system_prepare_preview_complete":
        return "准备预览完成：未执行串联操作。"
    if event == "menu_cancelled":
        return "已取消本次操作。"
    if event == "menu_exit":
        return "已退出操作菜单。"
    if event == "arm_already_enabled":
        return "七轴已经使能，无需重新发送使能指令。"
    if event in ("arm_target_reached", "hand_target_reached"):
        target = "机械臂" if event.startswith("arm") else "灵巧手"
        return f"{target}目标反馈已到达；新反馈样本 {payload.get('fresh_samples', '?')} 次。"
    if event == "hold_requested":
        return "已发送 move_j(当前角度) 保持请求；实际保持仍需观察。"
    if event == "hold_unconfirmed":
        return f"软件保持请求未获确认：{payload.get('error', '原因未提供')}"

    details = "  ".join(
        f"{name}={value}" for name, value in payload.items() if name != "event"
    )
    return f"{event}：{details}" if details else event


def emit(event, **fields):
    """Write one JSON event for scripts or one readable event for a terminal."""
    payload = {"event": event, **fields}
    if OUTPUT_FORMAT.get() == "human":
        print(format_event(payload), flush=True)
    else:
        print(json.dumps(payload, ensure_ascii=False), flush=True)


def finite(values, label="value"):
    """Return a float list after rejecting NaN and infinity."""
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} contains NaN or infinity")
    return result


def feedback_stamp(message):
    """Return an SDK receive timestamp, or ``None`` when unavailable."""
    return getattr(message, "timestamp", None) if message is not None else None


def orientation_error(first, second):
    """Return geodesic rotation error for Rz(yaw) Ry(pitch) Rx(roll) RPY."""

    def matrix(pose):
        roll, pitch, yaw = pose[3:]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return (
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
            -sp,
            cp * sr,
            cp * cr,
        )

    trace = sum(a * b for a, b in zip(matrix(first), matrix(second)))
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def create_robot(channel):
    """Create the driver for the firmware installed on this Nero."""
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V120,
        interface="socketcan",
        channel=channel,
    )
    return AgxArmFactory.create_arm(config)


def parse_finger_updates(tokens):
    """Parse repeated NAME=POSITION arguments."""
    if not tokens:
        raise ValueError("at least one --set NAME=POSITION is required")
    updates = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"invalid finger assignment: {token!r}")
        raw_name, raw_value = token.split("=", 1)
        name = FINGER_ALIASES.get(raw_name.strip().lower())
        if name is None:
            raise ValueError(
                f"unknown finger {raw_name!r}; choices: {sorted(FINGER_ALIASES)}"
            )
        if name in updates:
            raise ValueError(f"finger specified more than once: {name}")
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{name} position must be an integer") from exc
        if not 0 <= value <= 100:
            raise ValueError(f"{name} position must be in [0, 100]")
        updates[name] = value
    return updates


def finger_values(message):
    """Convert an SDK finger feedback message to a plain dictionary."""
    payload = message.msg if hasattr(message, "msg") else message
    return {name: int(getattr(payload, name)) for name in FINGER_NAMES}


def position_values(message):
    """Convert this Nero/Revo2 bridge's raw 0..255 feedback to API 0..100."""
    raw = finger_values(message)
    invalid = [name for name, value in raw.items() if not 0 <= value <= 255]
    if invalid:
        raise ValueError(f"Revo2 raw position outside [0, 255]: {invalid}")
    return {name: round(value * 100 / 255) for name, value in raw.items()}


def merge_hand_target(current, updates):
    """Preserve unspecified fingers and always return a complete six-value target."""
    target = {}
    for name in FINGER_NAMES:
        if name in updates:
            value = updates[name]
        else:
            if name not in current:
                raise ValueError(f"missing current position for {name}")
            value = int(current[name])
            if not 0 <= value <= 100:
                raise ValueError(
                    f"current {name}={value} cannot be preserved by the 0..100 API; "
                    f"specify --set {name}=VALUE explicitly"
                )
        if not 0 <= int(value) <= 100:
            raise ValueError(f"{name} position must be in [0, 100]")
        target[name] = int(value)
    return target


def validate_joints(robot, values):
    """Reject, rather than silently clamp, a seven-axis target."""
    joints = finite(values, "joint target")
    if len(joints) != 7:
        raise ValueError("Nero requires exactly seven joint values")
    configured = robot.get_config().get("joint_limits", {})
    limits = [configured.get(f"joint{index}") for index in range(1, 8)]
    if any(limit is None or len(limit) != 2 for limit in limits):
        raise RuntimeError("SDK did not provide seven Nero joint limits")
    violations = [
        index
        for index, (value, limit) in enumerate(zip(joints, limits), start=1)
        if not float(limit[0]) <= value <= float(limit[1])
    ]
    if violations:
        raise ValueError(f"joint target exceeds SDK limits: {violations}")
    return joints


def validate_joint_vector(values):
    """Validate a seven-axis vector without imposing command limits."""
    joints = finite(values, "joint values")
    if len(joints) != 7:
        raise ValueError("Nero requires exactly seven joint values")
    return joints


def validate_documented_joints(values):
    """Reject feedback or a hold target outside the Nero mechanical ranges."""
    joints = validate_joint_vector(values)
    violations = [
        index
        for index, (value, limits) in enumerate(
            zip(joints, NERO_DOCUMENTED_LIMITS_RAD), start=1
        )
        if not limits[0] <= value <= limits[1]
    ]
    if violations:
        raise ValueError(f"joint values exceed documented Nero ranges: {violations}")
    return joints


def sdk_limit_violations(robot, values):
    """Return axes outside the SDK's conservative target ranges."""
    joints = validate_joint_vector(values)
    configured = robot.get_config().get("joint_limits", {})
    violations = []
    for index, value in enumerate(joints, start=1):
        limits = configured.get(f"joint{index}")
        if limits is None or len(limits) != 2:
            raise RuntimeError("SDK did not provide seven Nero joint limits")
        if not float(limits[0]) <= value <= float(limits[1]):
            violations.append(index)
    return violations


def validate_pose(values):
    """Validate the six-dimensional flange or TCP pose accepted by the SDK."""
    pose = finite(values, "pose")
    if len(pose) != 6:
        raise ValueError("pose must contain x y z roll pitch yaw")
    if abs(pose[3]) > math.pi or abs(pose[5]) > math.pi:
        raise ValueError("roll and yaw must be within [-pi, pi]")
    if abs(pose[4]) > math.pi / 2:
        raise ValueError("pitch must be within [-pi/2, pi/2]")
    return pose


def read_fresh(getter, timeout, label):
    """Wait for a new SDK feedback timestamp, avoiding the startup cache."""
    first = getter()
    first_stamp = getattr(first, "timestamp", None) if first is not None else None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = getter()
        if value is not None:
            stamp = getattr(value, "timestamp", None)
            if first is None or stamp is None or stamp != first_stamp:
                return value
        time.sleep(0.02)
    raise TimeoutError(f"no fresh {label} feedback within {timeout:g}s")


def check_comm(robot):
    if robot.has_comm_error():
        raise RuntimeError(f"CAN communication error: {robot.get_comm_error()}")


def arm_snapshot(robot):
    """Read one fresh, internally consistent-enough arm snapshot."""
    joints = list(read_fresh(robot.get_joint_angles, 2.0, "joint").msg)
    pose = list(read_fresh(robot.get_flange_pose, 2.0, "flange pose").msg)
    status = read_fresh(robot.get_arm_status, 2.0, "arm status").msg
    check_comm(robot)
    return joints, pose, status


def read_joints(robot):
    """Read seven fresh Nero joint angles and limits from a connected driver."""
    feedback = read_fresh(robot.get_joint_angles, 2.0, "joint")
    if feedback_stamp(feedback) is None:
        raise RuntimeError("joint feedback has no receive timestamp")
    joints_rad = validate_joint_vector(feedback.msg)
    status = read_fresh(robot.get_arm_status, 2.0, "arm status").msg
    enabled = list(robot.get_joints_enable_status_list())
    if len(enabled) != 7:
        raise RuntimeError(f"expected seven joint enable states, got {enabled}")
    check_comm(robot)

    documented_limits_deg = [
        [math.degrees(lower), math.degrees(upper)]
        for lower, upper in NERO_DOCUMENTED_LIMITS_RAD
    ]
    configured = robot.get_config().get("joint_limits", {})
    sdk_limits_rad = [configured.get(f"joint{index}") for index in range(1, 8)]
    if all(
        limit is not None
        and len(limit) == 2
        and all(math.isfinite(float(value)) for value in limit)
        for limit in sdk_limits_rad
    ):
        sdk_limits_deg = [
            [math.degrees(float(value)) for value in limit] for limit in sdk_limits_rad
        ]
        within_sdk_limits = [
            float(limit[0]) <= angle <= float(limit[1])
            for angle, limit in zip(joints_rad, sdk_limits_rad)
        ]
    else:
        sdk_limits_deg = None
        within_sdk_limits = None

    return {
        "joint_names": [f"J{index}" for index in range(1, 8)],
        "joints_rad": joints_rad,
        "joints_deg": [math.degrees(value) for value in joints_rad],
        "joints_enabled": enabled,
        "arm_status": int(status.arm_status),
        "ctrl_mode": int(status.ctrl_mode),
        "documented_limits_deg": documented_limits_deg,
        "sdk_limits_deg": sdk_limits_deg,
        "within_sdk_limits": within_sdk_limits,
    }


def run_read_joints(robot):
    """Expose read_joints as a command without sending any control frame."""
    robot.connect()
    try:
        emit("read_joints", **read_joints(robot))
    finally:
        robot.disconnect()


def require_arm_ready(robot, joints, status):
    """Require an already enabled, normal arm before sending a movement command."""
    firmware = robot.get_firmware(timeout=3)
    version = firmware.get("software_version") if firmware else None
    if version != "1.20":
        raise RuntimeError(f"this demo expects Nero firmware 1.20, got {firmware}")
    if int(status.arm_status) != 0:
        raise RuntimeError(f"arm is not NORMAL: arm_status={status.arm_status}")
    if int(status.ctrl_mode) not in (1, 3):
        raise RuntimeError(f"unsupported initial control mode: {status.ctrl_mode}")
    enabled = robot.get_joints_enable_status_list()
    if enabled != [True] * 7:
        raise RuntimeError(f"all seven joints must already be enabled: {enabled}")
    validate_documented_joints(joints)


def enable_arm_connected(args, robot, joints, pose, status):
    """Release all Nero brakes on an existing connection and verify new feedback."""
    enabled = list(robot.get_joints_enable_status_list())
    emit(
        "arm_enable_plan",
        current_joints_rad=joints,
        current_flange_m_rad=pose,
        arm_status=int(status.arm_status),
        ctrl_mode=int(status.ctrl_mode),
        joints_enabled=enabled,
        attempts=args.attempts,
        execute=args.execute,
    )
    if (
        int(status.arm_status) == 0
        and int(status.ctrl_mode) == 1
        and enabled == [True] * 7
    ):
        emit("arm_already_enabled")
        return
    if not args.execute:
        emit("preview_only", message="no arm enable frame was sent")
        return

    validate_documented_joints(joints)
    firmware = robot.get_firmware(timeout=3)
    version = firmware.get("software_version") if firmware else None
    if version != "1.20":
        raise RuntimeError(f"this demo expects Nero firmware 1.20, got {firmware}")
    if int(status.arm_status) not in (0, 6):
        raise RuntimeError(f"arm cannot be enabled from arm_status={status.arm_status}")
    if int(status.ctrl_mode) not in (1, 3):
        raise RuntimeError(f"unsupported initial control mode: {status.ctrl_mode}")

    before_enable = robot.get_arm_status()
    previous_status_stamp = feedback_stamp(before_enable)
    accepted = False
    for attempt in range(1, args.attempts + 1):
        accepted = bool(robot.enable(timeout=1.5))
        emit("arm_enable_attempt", attempt=attempt, accepted=accepted)
        if accepted:
            break
        check_comm(robot)
        time.sleep(0.05)
    if not accepted:
        raise TimeoutError(
            f"arm enable was not accepted after {args.attempts} attempts"
        )

    deadline = time.monotonic() + args.timeout
    last_state = None
    last_enabled = None
    while time.monotonic() < deadline:
        feedback = robot.get_arm_status()
        stamp = feedback_stamp(feedback)
        if feedback is None or stamp is None or stamp == previous_status_stamp:
            time.sleep(0.02)
            continue
        previous_status_stamp = stamp
        check_comm(robot)
        last_state = feedback.msg
        last_enabled = list(robot.get_joints_enable_status_list())
        if (
            int(last_state.arm_status) == 0
            and int(last_state.ctrl_mode) == 1
            and last_enabled == [True] * 7
        ):
            emit(
                "arm_enabled",
                arm_status=int(last_state.arm_status),
                ctrl_mode=int(last_state.ctrl_mode),
                joints_enabled=last_enabled,
                feedback_timestamp=stamp,
            )
            return
        if int(last_state.arm_status) not in (0, 6):
            raise RuntimeError(f"arm fault while enabling: {last_state}")
        time.sleep(0.05)
    raise TimeoutError(
        "enable returned but fresh NORMAL/all-joints-enabled feedback was not observed: "
        f"status={last_state}, joints_enabled={last_enabled}"
    )


def run_enable(args, robot):
    """Explicitly release all seven Nero brakes and verify fresh feedback."""
    connected = False
    try:
        robot.connect()
        connected = True
        joints, pose, status = arm_snapshot(robot)
        enable_arm_connected(args, robot, joints, pose, status)
    finally:
        if connected:
            robot.disconnect()


def request_move_j_hold(robot):
    """Request command-level holding at the latest valid joint feedback."""
    joints = list(read_fresh(robot.get_joint_angles, 0.5, "hold joint").msg)
    status = read_fresh(robot.get_arm_status, 0.5, "hold status").msg
    check_comm(robot)
    if int(status.arm_status) != 0 or int(status.ctrl_mode) != 1:
        raise RuntimeError("fresh NORMAL/CAN_CTRL feedback is required for move_j hold")
    if robot.get_joints_enable_status_list() != [True] * 7:
        raise RuntimeError("all joints must remain enabled for move_j hold")
    target = validate_documented_joints(joints)
    # SDK clamping a hold target would move axes that were meant to stay still.
    robot.set_joint_limits_enabled(False)
    robot.move_j(target)
    emit("hold_requested", joints_rad=target)


def capture_arm_feedback_stamps(robot):
    """Capture the last received frame stamps immediately before a command."""
    messages = {
        "joint": robot.get_joint_angles(),
        "pose": robot.get_flange_pose(),
        "status": robot.get_arm_status(),
    }
    stamps = {name: feedback_stamp(value) for name, value in messages.items()}
    missing = [name for name, stamp in stamps.items() if stamp is None]
    if missing:
        raise RuntimeError(f"missing timestamp before arm command: {missing}")
    return stamps


def wait_for_arm_target(
    robot,
    command,
    target,
    timeout,
    old_ik_stamp=None,
    previous_stamps=None,
    require_joint_position=True,
    completion_samples=10,
    completion_poll_s=.05,
):
    """Observe distinct post-command feedback frames until a target is stable."""
    if type(completion_samples) is not int or not 2 <= completion_samples <= 10 or not .01 <= completion_poll_s <= .05:
        raise ValueError("Invalid arm completion observation settings")
    deadline = time.monotonic() + timeout
    transition_deadline = time.monotonic() + 1.0
    stable = 0
    idle_anchor = None
    idle_since = None
    last_report = 0.0
    ik_reported = False
    joint_motion = command in ("move-j", "home")
    required_stamps = (
        ("joint", "status") if joint_motion else ("joint", "pose", "status")
    )
    last_stamps = dict(previous_stamps or {})
    while time.monotonic() < deadline:
        joints = robot.get_joint_angles()
        pose = robot.get_flange_pose()
        status = robot.get_arm_status()
        if joints is None or pose is None or status is None:
            time.sleep(0.02)
            continue
        stamps = {
            "joint": feedback_stamp(joints),
            "pose": feedback_stamp(pose),
            "status": feedback_stamp(status),
        }
        if any(
            stamps[name] is None or stamps[name] == last_stamps.get(name)
            for name in required_stamps
        ):
            time.sleep(0.02)
            continue
        for name in required_stamps:
            last_stamps[name] = stamps[name]
        check_comm(robot)
        state = status.msg
        if int(state.arm_status) != 0:
            raise RuntimeError(f"arm fault while moving: {state}")
        if robot.get_joints_enable_status_list() != [True] * 7:
            raise RuntimeError("joint enable was lost while moving")
        if int(state.ctrl_mode) != 1 and time.monotonic() >= transition_deadline:
            raise RuntimeError("controller did not enter CAN_CTRL")

        if joint_motion:
            error = max(abs(a - b) for a, b in zip(joints.msg, target))
            reached = not require_joint_position or error <= math.radians(0.5)
            error_fields = {"joint_error_deg": math.degrees(error)}
        else:
            position_error = math.dist(pose.msg[:3], target[:3])
            rotation_error = orientation_error(pose.msg, target)
            reached = position_error <= 0.002 and rotation_error <= math.radians(1.0)
            error_fields = {
                "position_error_m": position_error,
                "orientation_error_rad": rotation_error,
            }
            ik = robot.get_ik_joint_angles()
            if ik is not None and ik.timestamp != old_ik_stamp and not ik_reported:
                # This is controller feedback, not a new SDK joint command.  The
                # Nero controller may legally choose J2/J5 just beyond pyAgxArm's
                # conservative software limits while remaining inside the manual's
                # mechanical ranges.
                solution = validate_documented_joints(list(ik.msg))
                emit(
                    "ik_solution",
                    joints_rad=solution,
                    fk_flange_m_rad=robot.fk(solution),
                )
                ik_reported = True

        motion_status = int(state.motion_status)
        if motion_status not in (0, 1):
            raise RuntimeError(f"unknown motion_status={motion_status}")
        stable = stable + 1 if reached and motion_status == 0 else 0
        now = time.monotonic()
        if now - last_report >= 1.0:
            emit("arm_progress", joints_rad=list(joints.msg), **error_fields)
            last_report = now
        settled = True
        if joint_motion and not require_joint_position:
            # JS mode may report idle while the physical joints still coast.
            # Observe a bounded joint span before accepting this endpoint;
            # never rebase the next stream or weaken its 0.1-degree guard.
            if not reached or motion_status != 0:
                idle_anchor = idle_since = None
                settled = False
            else:
                if (idle_anchor is None or
                        max(abs(a-b) for a, b in zip(joints.msg, idle_anchor)) > math.radians(.05)):
                    idle_anchor = list(joints.msg)
                    idle_since = now
                settled = now - idle_since >= .06
        completed = stable >= completion_samples and settled and (joint_motion or ik_reported)
        if completed:
            emit(
                "arm_target_reached" if not joint_motion or require_joint_position else "arm_motion_idle",
                command=command,
                ik_feedback_received=ik_reported if not joint_motion else None,
                fresh_samples=stable,
                position_verified=bool(not joint_motion or require_joint_position),
            )
            return
        time.sleep(completion_poll_s)
    if command == "home":
        raise TimeoutError(f"home 观察超时：{timeout:g} 秒内未确认目标反馈")
    raise TimeoutError(f"{command} did not reach the target within {timeout:g}s")


def run_fk(args, robot):
    """Calculate current or supplied forward kinematics."""
    connected = False
    try:
        if args.joints_deg is not None:
            joints = [math.radians(value) for value in args.joints_deg]
            controller_pose = None
        elif args.joints_rad is not None:
            joints = list(args.joints_rad)
            controller_pose = None
        else:
            robot.connect()
            connected = True
            joints, controller_pose, _ = arm_snapshot(robot)
        joints = validate_joint_vector(joints)
        flange_pose = list(robot.fk(joints))
        output = {
            "joints_rad": joints,
            "flange_m_rad": flange_pose,
            "frame": "base_to_flange",
        }
        if controller_pose is not None:
            output["controller_flange_m_rad"] = controller_pose
            output["position_difference_m"] = math.dist(
                flange_pose[:3], controller_pose[:3]
            )
            output["orientation_difference_rad"] = orientation_error(
                flange_pose, controller_pose
            )
        if args.tcp_offset is not None:
            offset = validate_pose(args.tcp_offset)
            robot.set_tcp_offset(offset)
            output["tcp_offset_flange_to_tcp_m_rad"] = offset
            output["tcp_m_rad"] = list(robot.get_flange2tcp_pose(flange_pose))
        emit("fk", **output)
    finally:
        if connected:
            robot.disconnect()


def move_j_target(args, robot, current):
    if args.joints_deg is not None:
        target = [math.radians(value) for value in args.joints_deg]
        return validate_joints(robot, target)
    elif args.joints_rad is not None:
        target = list(args.joints_rad)
        return validate_joints(robot, target)
    if args.joint is None or args.delta_deg is None:
        raise ValueError(
            "move-j requires --joints-deg, --joints-rad, or --joint N --delta-deg D"
        )
    target = list(current)
    target[args.joint - 1] += math.radians(args.delta_deg)
    target = validate_documented_joints(target)
    changed_axis = args.joint
    if changed_axis in sdk_limit_violations(robot, target):
        raise ValueError(f"J{changed_axis} target exceeds its SDK conservative limit")
    return target


def move_p_target(args, robot):
    requested = validate_pose(args.pose)
    if args.target_frame == "flange":
        if args.tcp_offset is not None:
            raise ValueError("--tcp-offset is only used with --target-frame tcp")
        return requested, requested
    if args.tcp_offset is None:
        raise ValueError("--target-frame tcp requires --tcp-offset X Y Z RX RY RZ")
    offset = validate_pose(args.tcp_offset)
    robot.set_tcp_offset(offset)
    return requested, validate_pose(robot.get_tcp2flange_pose(requested))


def home_wait_timeout(current_joints, target_joints, speed_percent, requested_timeout):
    """Return home feedback wait limit and whether an operator supplied it."""
    if requested_timeout is not None:
        return requested_timeout, "manual"
    speed_fraction = speed_percent / 100.0
    theoretical_min_s = max(
        math.degrees(abs(current - target)) / (maximum * speed_fraction)
        for current, target, maximum in zip(
            current_joints, target_joints, NERO_MAX_JOINT_SPEED_DEG_S
        )
    )
    return max(30, math.ceil(2 * theoretical_min_s + 30)), "auto"


def run_arm_motion_connected(args, robot, *, snapshot_reader=None):
    """Plan and optionally execute one arm command on an open connection."""
    command_attempted = False
    try:
        current_joints, current_pose, status = (snapshot_reader or arm_snapshot)(robot)
        if args.command == "move-j":
            target = move_j_target(args, robot, current_joints)
            requested = target
        elif args.command == "home":
            target = validate_joints(robot, READY_HOME_RAD)
            requested = target
        else:
            requested, target = move_p_target(args, robot)
        wait_timeout_s = args.timeout
        timeout_source = None
        if args.command == "home":
            wait_timeout_s, timeout_source = home_wait_timeout(
                current_joints, target, args.speed, args.timeout
            )
        if getattr(args, "interactive_preview", False):
            if not args.execute:
                args.interactive_preview_current_joints = list(current_joints)
                args.interactive_preview_target = list(target)
            else:
                preview_joints = args.interactive_preview_current_joints
                preview_target = args.interactive_preview_target
                start_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(current_joints, preview_joints)
                )
                if start_error > math.radians(0.25):
                    raise RuntimeError(
                        "joint angles changed since interactive preview; preview again"
                    )
                if args.command == "move-j" and args.joint is not None:
                    # A relative target is recomputed from each fresh sample.
                    # Keep the exact seven-axis target the operator previewed;
                    # the separate start check above rejects meaningful drift.
                    preview_target = validate_documented_joints(preview_target)
                    if args.joint in sdk_limit_violations(robot, preview_target):
                        raise ValueError(
                            f"J{args.joint} preview target exceeds its SDK conservative limit"
                        )
                    target = list(preview_target)
                    requested = target
                target_error = max(
                    abs(actual - expected)
                    for actual, expected in zip(target, preview_target)
                )
                if target_error > 1e-6:
                    raise RuntimeError(
                        "motion target changed since interactive preview; preview again"
                    )
        emit(
            "arm_plan",
            command=args.command,
            current_joints_rad=current_joints,
            current_flange_m_rad=current_pose,
            requested_target=requested,
            flange_target=target,
            target_frame=getattr(args, "target_frame", "joint"),
            speed_percent=args.speed,
            execute=args.execute,
            **(
                {
                    "ready_home_deg": list(READY_HOME_DEG),
                    "ready_home_flange_m_rad": list(robot.fk(target)),
                    "j2_abs_deg": abs(READY_HOME_DEG[1]),
                    "j4_abs_deg": abs(READY_HOME_DEG[3]),
                    "wait_timeout_s": wait_timeout_s,
                    "timeout_source": timeout_source,
                }
                if args.command == "home"
                else {}
            ),
        )
        if not args.execute:
            emit("preview_only", message="no arm control frame was sent")
            return target

        require_arm_ready(robot, current_joints, status)
        if args.command == "home":
            home_error = max(abs(a - b) for a, b in zip(current_joints, target))
            if home_error <= math.radians(0.5) and int(status.ctrl_mode) == 1:
                emit(
                    "arm_target_reached",
                    command="home",
                    ik_feedback_received=None,
                    fresh_samples=1,
                    already_at_target=True,
                    joint_error_deg=math.degrees(home_error),
                )
                return target
        joint_motion = args.command in ("move-j", "home")
        limit_check_values = target if joint_motion else current_joints
        soft_limit_axes = sdk_limit_violations(robot, limit_check_values)
        if soft_limit_axes:
            emit(
                "sdk_soft_limit_warning",
                joints=soft_limit_axes,
                message=(
                    "these axes are within documented mechanical ranges but outside the "
                    "SDK's conservative ranges; preserving their exact values"
                ),
            )
        # Enabling SDK clipping while preserving an existing out-of-soft-limit
        # axis would silently move it.  Manual validation above remains active.
        robot.set_joint_limits_enabled(not soft_limit_axes)
        robot.set_speed_percent(args.speed)
        previous_stamps = capture_arm_feedback_stamps(robot)
        old_ik = robot.get_ik_joint_angles() if args.command == "move-p" else None
        old_ik_stamp = old_ik.timestamp if old_ik is not None else None
        command_attempted = True
        if joint_motion:
            robot.move_j(target)
        else:
            robot.move_p(target)
        wait_for_arm_target(
            robot,
            args.command,
            target,
            wait_timeout_s,
            old_ik_stamp,
            previous_stamps,
            **({"completion_samples": 2, "completion_poll_s": .01} if getattr(args, "fast_completion", False) else {}),
            **({"require_joint_position": False} if getattr(args, "require_joint_position", True) is False else {}),
        )
        return target
    except BaseException:
        if command_attempted:
            try:
                request_move_j_hold(robot)
            except BaseException as hold_error:
                emit("hold_unconfirmed", error=str(hold_error))
        raise


def run_arm_motion(args, robot):
    connected = False
    try:
        robot.connect()
        connected = True
        return run_arm_motion_connected(args, robot)
    finally:
        if connected:
            robot.disconnect()


def hand_side(hand):
    status = hand.get_hand_status()
    if status is None:
        return None
    return int(status.msg.left_or_right)


def require_right_hand(hand, execute):
    side = hand_side(hand)
    if side == 1 and execute:
        raise RuntimeError(
            "feedback identifies a left Revo2; this demo targets the right hand"
        )
    if side not in (1, 2):
        emit("hand_side", value="unavailable", configured="right")
    else:
        emit("hand_side", value="right" if side == 2 else "left", configured="right")


def wait_for_hand_target(hand, target, timeout, previous_stamp):
    """Require three distinct position frames received after the hand command."""
    deadline = time.monotonic() + timeout
    stable = 0
    last_report = 0.0
    last_stamp = previous_stamp
    while time.monotonic() < deadline:
        feedback = hand.get_finger_pos()
        stamp = feedback_stamp(feedback)
        if feedback is None or stamp is None or stamp == last_stamp:
            time.sleep(0.02)
            continue
        last_stamp = stamp
        actual = position_values(feedback)
        error = max(abs(actual[name] - target[name]) for name in FINGER_NAMES)
        stable = stable + 1 if error <= 3 else 0
        now = time.monotonic()
        if now - last_report >= 0.5:
            emit("hand_progress", positions=actual, max_error=error)
            last_report = now
        if stable >= 3:
            emit(
                "hand_target_reached",
                positions=actual,
                max_error=error,
                fresh_samples=stable,
            )
            return actual
        time.sleep(0.05)
    raise TimeoutError(f"hand did not reach the target within {timeout:g}s")


def read_hand_diagnostics(hand, initial_position, timeout=0.5):
    """Wait briefly for the SDK's first 100 ms FPS accounting interval."""
    deadline = time.monotonic() + timeout
    position = initial_position
    current = None
    speed = None
    status = None
    fps = 0.0
    while True:
        new_position = hand.get_finger_pos()
        new_current = hand.get_finger_current()
        new_speed = hand.get_finger_spd()
        new_status = hand.get_hand_status()
        position = new_position if new_position is not None else position
        current = new_current if new_current is not None else current
        speed = new_speed if new_speed is not None else speed
        status = new_status if new_status is not None else status
        fps = float(hand.get_fps())
        position_hz = float(getattr(position, "hz", 0.0))
        current_hz = float(getattr(current, "hz", 0.0)) if current is not None else 0.0
        if fps > 0.0 or position_hz > 0.0 or current_hz > 0.0:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    available = []
    if position is not None:
        available.append("position")
    if current is not None:
        available.append("current")
    if speed is not None:
        available.append("speed")
    if status is not None:
        available.append("status")
    return {
        "position": position,
        "current": current,
        "speed": speed,
        "status": status,
        "fps": fps,
        "position_hz": position_hz,
        "current_hz": current_hz if current is not None else None,
        "available": available,
    }


def run_hand_connected(args, robot, hand):
    """Plan and optionally execute one Revo2 command on an open connection."""
    position_feedback = read_fresh(hand.get_finger_pos, 2.0, "Revo2 position")
    current = position_values(position_feedback)
    check_comm(robot)
    require_right_hand(hand, getattr(args, "execute", False))

    if args.command == "hand-status":
        diagnostics = read_hand_diagnostics(hand, position_feedback)
        position_feedback = diagnostics["position"]
        current = position_values(position_feedback)
        current_feedback = diagnostics["current"]
        speed_feedback = diagnostics["speed"]
        status_feedback = diagnostics["status"]
        emit(
            "hand_status",
            is_ok=bool(hand.is_ok()),
            available_feedback=diagnostics["available"],
            feedback_complete=len(diagnostics["available"]) == 4,
            fps=diagnostics["fps"],
            position_hz=diagnostics["position_hz"],
            positions=current,
            raw_positions=finger_values(position_feedback),
            currents=(
                finger_values(current_feedback)
                if current_feedback is not None
                else None
            ),
            current_hz=diagnostics["current_hz"],
            speeds=finger_values(speed_feedback)
            if speed_feedback is not None
            else None,
            motor_status=(
                finger_values(status_feedback) if status_feedback is not None else None
            ),
        )
        return current

    if args.command == "gesture":
        target = dict(zip(FINGER_NAMES, GESTURES[args.name]))
        if args.name in RIG_CANDIDATE_GESTURES:
            source = "rig_unverified_candidate"
        elif args.name in TUNABLE_GESTURES:
            source = "tunable"
        else:
            source = "agilex_srdf"
        plan_kind = {"gesture": args.name, "preset_source": source}
    elif args.positions is not None:
        values = [int(value) for value in args.positions]
        target = merge_hand_target({}, dict(zip(FINGER_NAMES, values)))
        plan_kind = {"control": "all_positions"}
    else:
        updates = parse_finger_updates(args.set_values)
        target = merge_hand_target(current, updates)
        plan_kind = {"control": "partial_positions", "updated": updates}

    if getattr(args, "interactive_preview", False):
        if not args.execute:
            args.interactive_preview_hand_positions = dict(current)
            args.interactive_preview_target = dict(target)
        else:
            preview_current = args.interactive_preview_hand_positions
            if (
                max(abs(current[name] - preview_current[name]) for name in FINGER_NAMES)
                > 3
            ):
                raise RuntimeError(
                    "hand positions changed since interactive preview; preview again"
                )
            if target != args.interactive_preview_target:
                raise RuntimeError(
                    "hand target changed since interactive preview; preview again"
                )

    emit(
        "hand_plan",
        hand="right_revo2",
        current=current,
        target=target,
        duration_s=args.duration,
        execute=args.execute,
        **plan_kind,
    )
    if not args.execute:
        emit("preview_only", message="no Revo2 control frame was sent")
        return target
    if not hand.is_ok():
        raise RuntimeError("Revo2 feedback monitor is not healthy")

    previous_stamp = feedback_stamp(position_feedback)
    if previous_stamp is None:
        raise RuntimeError("missing Revo2 position timestamp before command")
    duration_ticks = round(args.duration * 100)
    position_kwargs = {name: target[name] for name in FINGER_NAMES}
    time_kwargs = {name: duration_ticks for name in FINGER_NAMES}
    started = time.monotonic()
    hand.position_time_ctrl(mode="pos", **position_kwargs)
    hand.position_time_ctrl(mode="time", **time_kwargs)
    interval = time.monotonic() - started
    emit("hand_command_sent", position_time_interval_s=interval)
    if interval > 0.05:
        raise RuntimeError("position/time frames exceeded the SDK 50ms interval")
    wait_for_hand_target(
        hand,
        target,
        max(args.timeout, args.duration + 1.0),
        previous_stamp,
    )
    return target


def run_hand(args, robot):
    hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
    connected = False
    try:
        robot.connect()
        connected = True
        return run_hand_connected(args, robot, hand)
    finally:
        if connected:
            robot.disconnect()


def check_prepare_feedback(robot, hand, home_target, hand_target, stage):
    """Require fresh, fault-free feedback from both devices at the ready pose."""
    hand_feedback = read_fresh(hand.get_finger_pos, 2.0, f"{stage} Revo2 position")
    stamp = feedback_stamp(hand_feedback)
    if stamp is None:
        raise RuntimeError(f"{stage} Revo2 position has no receive timestamp")
    hand_positions = position_values(hand_feedback)
    hand_error = max(
        abs(hand_positions[name] - hand_target[name]) for name in FINGER_NAMES
    )
    if hand_error > 3:
        raise RuntimeError(
            f"Revo2 left open target during {stage}: "
            f"positions={hand_positions}, error={hand_error}"
        )
    joints, _, status = arm_snapshot(robot)
    require_arm_ready(robot, joints, status)
    if int(status.ctrl_mode) != 1:
        raise RuntimeError(
            f"Nero did not remain in CAN_CTRL during {stage}: "
            f"ctrl_mode={status.ctrl_mode}"
        )
    enabled = list(robot.get_joints_enable_status_list())
    if enabled != [True] * 7:
        raise RuntimeError(f"Nero joint enable was lost during {stage}: {enabled}")
    arm_error = max(abs(a - b) for a, b in zip(joints, home_target))
    if arm_error > math.radians(0.5):
        raise RuntimeError(
            f"Nero left ready-home during {stage}: "
            f"error={math.degrees(arm_error):.3f}deg"
        )
    check_comm(robot)
    return {
        "arm_joints_deg": [math.degrees(value) for value in joints],
        "arm_max_error_deg": math.degrees(arm_error),
        "arm_status": int(status.arm_status),
        "ctrl_mode": int(status.ctrl_mode),
        "arm_joints_enabled": enabled,
        "hand_positions": hand_positions,
        "hand_max_error": hand_error,
    }, stamp


def observe_prepare_after_reconnect(robot, hand, home_target, hand_target, seconds):
    """Catch a target that rebounds after SDK disconnect or a short arrival."""
    started = time.monotonic()
    deadline = started + seconds
    result, last_stamp = check_prepare_feedback(
        robot, hand, home_target, hand_target, "post-reconnect"
    )
    fresh_samples = 1
    last_fresh = time.monotonic()
    next_arm_check = last_fresh + 1.0
    # A minimum of three separate RX frames is required even for a short
    # explicitly requested observation window.
    while time.monotonic() < deadline or fresh_samples < 3:
        now = time.monotonic()
        if now > deadline + 2.0 or now - last_fresh > 1.0:
            raise TimeoutError(
                "fresh Revo2 position feedback stopped during hold check"
            )
        feedback = hand.get_finger_pos()
        stamp = feedback_stamp(feedback)
        if feedback is not None and stamp is not None and stamp != last_stamp:
            last_stamp = stamp
            last_fresh = now
            fresh_samples += 1
            positions = position_values(feedback)
            error = max(
                abs(positions[name] - hand_target[name]) for name in FINGER_NAMES
            )
            if error > 3:
                raise RuntimeError(
                    "Revo2 open target drifted after SDK reconnect: "
                    f"positions={positions}, error={error}"
                )
        if now >= next_arm_check:
            result, last_stamp = check_prepare_feedback(
                robot, hand, home_target, hand_target, "hold observation"
            )
            fresh_samples += 1
            last_fresh = time.monotonic()
            next_arm_check = last_fresh + 1.0
        time.sleep(0.05)
    result, _ = check_prepare_feedback(
        robot, hand, home_target, hand_target, "final post-reconnect check"
    )
    return result, fresh_samples, time.monotonic() - started


def run_prepare(args, robot):
    """Enable Nero, move to ready-home, then command all Revo2 axes."""
    hand = robot.init_effector(robot.OPTIONS.EFFECTOR.REVO2)
    connected = False
    try:
        robot.connect()
        connected = True

        # Complete both feedback preflights before the first control frame.
        joints, pose, status = arm_snapshot(robot)
        hand_feedback = read_fresh(hand.get_finger_pos, 2.0, "Revo2 position")
        hand_positions = position_values(hand_feedback)
        check_comm(robot)
        if args.execute and not hand.is_ok():
            raise RuntimeError("Revo2 feedback monitor is not healthy")
        current_deg = [math.degrees(value) for value in joints]
        if getattr(args, "interactive_preview", False):
            if not args.execute:
                args.interactive_preview_current_joints = list(joints)
                args.interactive_preview_hand_positions = dict(hand_positions)
            else:
                preview_joints = args.interactive_preview_current_joints
                if max(
                    abs(actual - expected)
                    for actual, expected in zip(joints, preview_joints)
                ) > math.radians(0.25):
                    raise RuntimeError(
                        "joint angles changed since interactive preview; preview again"
                    )
                preview_hand = args.interactive_preview_hand_positions
                if (
                    max(
                        abs(hand_positions[name] - preview_hand[name])
                        for name in FINGER_NAMES
                    )
                    > 3
                ):
                    raise RuntimeError(
                        "hand positions changed since interactive preview; preview again"
                    )
        emit(
            "system_prepare_plan",
            sequence=[
                "enable_nero_j1_to_j7",
                "move_nero_to_ready_home",
                "command_revo2_all_six_open",
                "disconnect_reconnect_and_observe_feedback",
            ],
            current_joints_deg=current_deg,
            ready_home_deg=list(READY_HOME_DEG),
            current_hand_positions=hand_positions,
            execute=args.execute,
        )

        enable_args = argparse.Namespace(
            execute=args.execute,
            attempts=args.attempts,
            timeout=5.0 if args.timeout is None else min(args.timeout, 5.0),
        )
        enable_arm_connected(enable_args, robot, joints, pose, status)

        home_args = argparse.Namespace(
            command="home",
            execute=args.execute,
            speed=args.speed,
            timeout=args.timeout,
        )
        home_target = run_arm_motion_connected(home_args, robot)

        hand_args = argparse.Namespace(
            command="gesture",
            name="open",
            execute=args.execute,
            duration=args.duration,
            timeout=args.hand_timeout,
        )
        hand_target = run_hand_connected(hand_args, robot, hand)
        if not args.execute:
            emit("system_prepare_preview_complete")
            return

        # Arrival feedback alone was misleading on this rig: the hand drifted
        # after a previous successful prepare.  Reject a transient endpoint
        # before disconnecting, then watch fresh feedback after reconnecting.
        check_prepare_feedback(robot, hand, home_target, hand_target, "arrival")
        robot.disconnect()
        connected = False
        time.sleep(0.5)
        robot.connect()
        connected = True
        emit("system_hold_observation", seconds=args.post_settle_time)
        result, samples, elapsed = observe_prepare_after_reconnect(
            robot, hand, home_target, hand_target, args.post_settle_time
        )
        emit(
            "system_position_hold_observed",
            **result,
            hand="right_revo2_basic",
            hand_all_six_commanded=True,
            hand_enable_state="not_exposed_by_basic_sdk",
            verified_after_sdk_reconnect=True,
            hand_fresh_samples=samples,
            observed_duration_s=elapsed,
        )
    finally:
        if connected:
            robot.disconnect()


def add_execute_options(parser, *, hand=False):
    parser.add_argument(
        "--execute", action="store_true", help="send the planned command"
    )
    parser.add_argument("--timeout", type=float, default=5.0 if hand else 30.0)
    if hand:
        parser.add_argument(
            "--duration",
            type=float,
            default=1.0,
            help="finger arrival time in seconds (0.01..2.55)",
        )
    else:
        parser.add_argument("--speed", type=int, default=1, choices=range(1, 11))


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--confirm", action="store_true", help="require EXECUTE confirmation in the interactive menu")
    parser.add_argument(
        "--format",
        choices=("auto", "human", "json"),
        default="auto",
        help="human in a terminal, JSON Lines when piped; override explicitly",
    )
    subparsers = parser.add_subparsers(dest="command")
    parser.set_defaults(command="interactive")
    subparsers.add_parser(
        "interactive", aliases=["menu"], help="guided interactive menu"
    )

    subparsers.add_parser("status", help="read Nero status")
    subparsers.add_parser(
        "read-joints",
        aliases=["read_joints"],
        help="read seven joint angles and limits",
    )
    enable_parser = subparsers.add_parser(
        "enable", aliases=["arm-enable"], help="explicitly enable all seven Nero joints"
    )
    enable_parser.add_argument("--execute", action="store_true")
    enable_parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3)
    enable_parser.add_argument("--timeout", type=float, default=3.0)
    fk_parser = subparsers.add_parser("fk", help="calculate base-to-flange FK")
    fk_group = fk_parser.add_mutually_exclusive_group()
    fk_group.add_argument("--joints-deg", nargs=7, type=float)
    fk_group.add_argument("--joints-rad", nargs=7, type=float)
    fk_parser.add_argument("--tcp-offset", nargs=6, type=float)

    move_j_parser = subparsers.add_parser(
        "move-j", aliases=["move_j"], help="Nero seven-joint position control"
    )
    move_j_group = move_j_parser.add_mutually_exclusive_group()
    move_j_group.add_argument("--joints-deg", nargs=7, type=float)
    move_j_group.add_argument("--joints-rad", nargs=7, type=float)
    move_j_parser.add_argument("--joint", type=int, choices=range(1, 8))
    move_j_parser.add_argument("--delta-deg", type=float)
    add_execute_options(move_j_parser)

    move_p_parser = subparsers.add_parser(
        "move-p", aliases=["move_p"], help="controller-side IK and Cartesian motion"
    )
    move_p_parser.add_argument("--pose", nargs=6, type=float, required=True)
    move_p_parser.add_argument(
        "--target-frame", choices=("flange", "tcp"), default="flange"
    )
    move_p_parser.add_argument("--tcp-offset", nargs=6, type=float)
    add_execute_options(move_p_parser)

    home_parser = subparsers.add_parser(
        "home", help="move to the non-singular rig ready-home"
    )
    add_execute_options(home_parser)
    home_parser.set_defaults(timeout=None)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="enable Nero, move home, command all six Revo2 actuators, and verify",
    )
    prepare_parser.add_argument("--execute", action="store_true")
    prepare_parser.add_argument("--attempts", type=int, choices=range(1, 6), default=3)
    prepare_parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="home feedback wait limit in seconds; default sizes it from the current joints",
    )
    prepare_parser.add_argument("--speed", type=int, default=1, choices=range(1, 11))
    prepare_parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Revo2 open arrival time in seconds (0.01..2.55)",
    )
    prepare_parser.add_argument("--hand-timeout", type=float, default=5.0)
    prepare_parser.add_argument(
        "--post-settle-time",
        type=float,
        default=8.0,
        help="observe fresh Nero/Revo2 feedback for this many seconds after SDK reconnect",
    )

    subparsers.add_parser("hand-status", help="read right Revo2 feedback")
    hand_parser = subparsers.add_parser(
        "hand", help="control one or more hand actuators"
    )
    hand_group = hand_parser.add_mutually_exclusive_group(required=True)
    hand_group.add_argument(
        "--set",
        dest="set_values",
        action="append",
        help="repeat NAME=POSITION, e.g. --set index=30 --set middle=40",
    )
    hand_group.add_argument(
        "--positions",
        nargs=6,
        type=int,
        metavar=("THUMB_TIP", "THUMB_BASE", "INDEX", "MIDDLE", "RING", "PINKY"),
    )
    add_execute_options(hand_parser, hand=True)

    gesture_parser = subparsers.add_parser(
        "gesture", help="send a complete hand preset"
    )
    gesture_parser.add_argument("name", choices=tuple(GESTURES))
    add_execute_options(gesture_parser, hand=True)
    return parser


def validate_arguments(parser, args):
    args.command = COMMAND_ALIASES.get(args.command, args.command)
    if args.command == "menu":
        args.command = "interactive"
    if (
        hasattr(args, "timeout")
        and args.timeout is not None
        and (not math.isfinite(args.timeout) or args.timeout <= 0)
    ):
        parser.error("--timeout must be a positive finite number")
    if hasattr(args, "duration") and (
        not math.isfinite(args.duration) or not 0.01 <= args.duration <= 2.55
    ):
        parser.error("--duration must be within 0.01..2.55 seconds")
    if hasattr(args, "hand_timeout") and (
        not math.isfinite(args.hand_timeout) or args.hand_timeout <= 0
    ):
        parser.error("--hand-timeout must be a positive finite number")
    if hasattr(args, "post_settle_time") and (
        not math.isfinite(args.post_settle_time) or args.post_settle_time <= 0
    ):
        parser.error("--post-settle-time must be a positive finite number")
    if args.command == "move-j":
        has_vector = args.joints_deg is not None or args.joints_rad is not None
        has_delta = args.joint is not None or args.delta_deg is not None
        if has_vector and has_delta:
            parser.error("do not combine a seven-joint target with --joint/--delta-deg")
        if not has_vector and not (
            args.joint is not None and args.delta_deg is not None
        ):
            parser.error(
                "move-j requires a seven-joint target or --joint N --delta-deg D"
            )
        if args.delta_deg is not None and not math.isfinite(args.delta_deg):
            parser.error("--delta-deg must be finite")


INTERACTIVE_CHOICES = {
    "1": ("机械臂状态", "status"),
    "2": ("读取七轴角度", "read-joints"),
    "3": ("关节运动 move-j", "move-j"),
    "4": ("笛卡尔运动 move-p", "move-p"),
    "5": ("正运动学 FK", "fk"),
    "6": ("灵巧手状态", "hand-status"),
    "7": ("灵巧手单/多关节控制", "hand"),
    "8": ("灵巧手手势", "gesture"),
    "9": ("机械臂七轴使能", "enable"),
    "10": ("非奇异 ready-home", "home"),
    "11": ("机械臂与灵巧手准备", "prepare"),
}
INTERACTIVE_READ_ONLY = {"status", "read-joints", "fk", "hand-status"}


class MenuCancelled(Exception):
    """A user abandoned one menu action before any execution request."""


def menu_prompt(input_fn, label, default=None):
    """Read a menu field; q/cancel abandons only the current action."""
    value = input_fn(label).strip()
    if value.lower() in ("q", "quit", "cancel"):
        raise MenuCancelled()
    if not value:
        if default is None:
            raise ValueError("required menu input was empty")
        return default
    return value


def menu_values(input_fn, label, count):
    """Split a fixed-length numeric vector without invoking a shell."""
    values = menu_prompt(input_fn, label).replace(",", " ").split()
    if len(values) != count:
        raise ValueError(f"expected {count} values, got {len(values)}")
    return values


def interactive_command_argv(choice, input_fn):
    """Build ordinary CLI arguments from guided questions."""
    command = INTERACTIVE_CHOICES[choice][1]
    argv = [command]
    if command in ("status", "read-joints", "hand-status"):
        return argv
    if command == "fk":
        mode = menu_prompt(input_fn, "FK: 1当前角度 / 2输入七轴角度(度) [1]: ", "1")
        if mode == "2":
            argv.extend(
                ["--joints-deg", *menu_values(input_fn, "七轴角度 J1..J7: ", 7)]
            )
        elif mode != "1":
            raise ValueError("FK mode must be 1 or 2")
        tcp_offset = menu_prompt(
            input_fn, "法兰到TCP偏移 x y z rx ry rz (留空则不计算TCP): ", ""
        )
        if tcp_offset:
            values = tcp_offset.replace(",", " ").split()
            if len(values) != 6:
                raise ValueError("TCP offset requires six values")
            argv.extend(["--tcp-offset", *values])
        return argv
    if command == "move-j":
        mode = menu_prompt(input_fn, "move-j: 1单轴增量 / 2七轴绝对角度(度) [1]: ", "1")
        if mode == "1":
            raw_joint = menu_prompt(input_fn, "关节编号 1..7 或 J1..J7: ")
            joint = raw_joint[1:] if raw_joint[:1].lower() == "j" else raw_joint
            if joint not in {str(index) for index in range(1, 8)}:
                raise ValueError("关节编号请输入 1..7 或 J1..J7")
            argv.extend(
                [
                    "--joint",
                    joint,
                    "--delta-deg",
                    menu_prompt(input_fn, "相对当前角度的增量(度): "),
                ]
            )
        elif mode == "2":
            argv.extend(
                ["--joints-deg", *menu_values(input_fn, "目标七轴角度 J1..J7: ", 7)]
            )
        else:
            raise ValueError("move-j mode must be 1 or 2")
        argv.extend(["--speed", menu_prompt(input_fn, "速度百分比 1..10 [1]: ", "1")])
        return argv
    if command == "move-p":
        argv.extend(
            ["--pose", *menu_values(input_fn, "目标 x y z rx ry rz (m/rad): ", 6)]
        )
        frame = menu_prompt(input_fn, "目标坐标: 1法兰 / 2TCP [1]: ", "1")
        if frame == "2":
            argv.extend(
                [
                    "--target-frame",
                    "tcp",
                    "--tcp-offset",
                    *menu_values(input_fn, "法兰到TCP偏移 x y z rx ry rz: ", 6),
                ]
            )
        elif frame != "1":
            raise ValueError("target frame must be 1 or 2")
        argv.extend(["--speed", menu_prompt(input_fn, "速度百分比 1..10 [1]: ", "1")])
        return argv
    if command == "hand":
        mode = menu_prompt(input_fn, "灵巧手: 1指定关节 / 2六关节位置 [1]: ", "1")
        if mode == "1":
            updates = menu_prompt(
                input_fn, "NAME=0..100，可输入多个，如 index=30 middle=40: "
            ).split()
            for update in updates:
                argv.extend(["--set", update])
        elif mode == "2":
            argv.extend(
                [
                    "--positions",
                    *menu_values(input_fn, "六关节位置(拇指尖/根/食/中/环/小): ", 6),
                ]
            )
        else:
            raise ValueError("hand mode must be 1 or 2")
        argv.extend(
            ["--duration", menu_prompt(input_fn, "到位时间秒 0.01..2.55 [1]: ", "1")]
        )
        return argv
    if command == "gesture":
        argv.append(menu_prompt(input_fn, f"手势 {', '.join(GESTURES)}: "))
        argv.extend(
            ["--duration", menu_prompt(input_fn, "到位时间秒 0.01..2.55 [1]: ", "1")]
        )
        return argv
    if command == "enable":
        argv.extend(
            ["--attempts", menu_prompt(input_fn, "使能尝试次数 1..5 [3]: ", "3")]
        )
        return argv
    if command == "home":
        argv.extend(["--speed", menu_prompt(input_fn, "速度百分比 1..10 [1]: ", "1")])
        timeout = menu_prompt(input_fn, "等待上限秒 [自动]: ", "")
        if timeout:
            argv.extend(["--timeout", timeout])
        return argv
    if command == "prepare":
        argv.extend(
            ["--speed", menu_prompt(input_fn, "机械臂速度百分比 1..10 [1]: ", "1")]
        )
        timeout = menu_prompt(input_fn, "home 等待上限秒 [自动]: ", "")
        if timeout:
            argv.extend(["--timeout", timeout])
        argv.extend(
            [
                "--duration",
                menu_prompt(input_fn, "灵巧手到位时间秒 0.01..2.55 [1]: ", "1"),
                "--post-settle-time",
                menu_prompt(input_fn, "重连后观察时间秒 [8]: ", "8"),
            ]
        )
        return argv
    raise ValueError(f"unsupported menu command: {command}")


def _run_interactive_loop(args, input_fn=None):
    """Guide one action at a time through the existing preview/execute paths."""
    if input_fn is None:
        input_fn = input
    parser = make_parser()
    while True:
        print("\nNero + 右 Revo2 操作菜单", flush=True)
        for number, (label, _) in INTERACTIVE_CHOICES.items():
            print(f"{number:>2}. {label}", flush=True)
        print(" 0. 退出", flush=True)
        try:
            choice = input_fn("选择编号 (q也可退出): ").strip()
            if choice.lower() in ("0", "q", "quit"):
                emit("menu_exit")
                return
            if choice not in INTERACTIVE_CHOICES:
                raise ValueError(f"unknown menu choice: {choice}")
            argv = [
                "--channel",
                args.channel,
                *interactive_command_argv(choice, input_fn),
            ]
            action_args = parser.parse_args(argv)
            validate_arguments(parser, action_args)
            if action_args.command in INTERACTIVE_READ_ONLY:
                run(action_args)
                continue
            action_args.interactive_preview = True
            action_args.execute = False
            run(action_args)
            if getattr(args, "confirm", False):
                decision = input_fn("输入大写 EXECUTE 执行；其他输入取消: ").strip()
                if decision != "EXECUTE":
                    emit("menu_cancelled", command=action_args.command)
                    continue
            action_args.execute = True
            run(action_args)
        except MenuCancelled:
            emit("menu_cancelled", selected=choice)
        except (EOFError, KeyboardInterrupt):
            emit("menu_exit", reason="interrupted")
            return
        except SystemExit as error:
            emit("failed", error=f"invalid menu arguments: {error.code}")
        except Exception as error:
            emit("failed", error=str(error) or type(error).__name__)


def run_interactive(args, input_fn=None):
    """Show a readable menu while retaining an explicit JSON event override."""
    mode = "json" if getattr(args, "format", "auto") == "json" else "human"
    with use_output_format(mode):
        return _run_interactive_loop(args, input_fn)


def run(args):
    args.command = COMMAND_ALIASES.get(args.command, args.command)
    if args.command in ("interactive", "menu"):
        return run_interactive(args)
    robot = create_robot(args.channel)
    if args.command == "read-joints":
        run_read_joints(robot)
    elif args.command == "status":
        robot.connect()
        try:
            joints, pose, status = arm_snapshot(robot)
            emit(
                "arm_status",
                joints_rad=joints,
                flange_m_rad=pose,
                status=str(status),
                joints_enabled=robot.get_joints_enable_status_list(),
            )
        finally:
            robot.disconnect()
    elif args.command == "enable":
        run_enable(args, robot)
    elif args.command == "fk":
        run_fk(args, robot)
    elif args.command in ("move-j", "move-p", "home"):
        run_arm_motion(args, robot)
    elif args.command == "prepare":
        run_prepare(args, robot)
    else:
        run_hand(args, robot)


def main():
    parser = make_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    mode = args.format
    if mode == "auto":
        mode = "human" if sys.stdout.isatty() else "json"
    with use_output_format(mode):
        try:
            run(args)
        except (Exception, KeyboardInterrupt) as error:
            emit("failed", error=str(error) or "interrupted")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
