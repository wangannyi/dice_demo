"""Unit tests for ``nero_revo2_demo`` using an in-memory arm and hand."""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


# pyAgxArm is installed on the K3 but is intentionally not a host test dependency.
try:
    import pyAgxArm  # noqa: F401
except ModuleNotFoundError:
    sdk_stub = ModuleType("pyAgxArm")

    class _Factory:
        @staticmethod
        def create_arm(_config):
            raise AssertionError("unit tests must not create a real pyAgxArm driver")

    sdk_stub.AgxArmFactory = _Factory
    sdk_stub.ArmModel = SimpleNamespace(NERO="nero")
    sdk_stub.NeroFW = SimpleNamespace(V120="1.20")
    sdk_stub.create_agx_arm_config = lambda **values: values
    sys.modules["pyAgxArm"] = sdk_stub

from nero_revo2_control import nero_revo2_demo as demo


class FakeHand:
    """Small Revo2 double whose feedback advances on every read."""

    def __init__(self, positions=(10, 20, 30, 40, 50, 60), trace=None):
        self.positions = dict(zip(demo.FINGER_NAMES, positions))
        self.calls = []
        self.trace = trace if trace is not None else []
        self.pending_target = None
        self.timestamp = 0
        self.freeze_feedback = False

    def _message(self, values):
        if not self.freeze_feedback:
            self.timestamp += 1
        return SimpleNamespace(
            msg=SimpleNamespace(**values),
            timestamp=self.timestamp,
        )

    def get_finger_pos(self):
        # Revo2 feedback uses raw 0..255 although the command API uses 0..100.
        raw = {name: round(value * 255 / 100) for name, value in self.positions.items()}
        return self._message(raw)

    def get_finger_current(self):
        return self._message({name: 0 for name in demo.FINGER_NAMES})

    def get_finger_spd(self):
        return self._message({name: 0 for name in demo.FINGER_NAMES})

    def get_hand_status(self):
        self.timestamp += 1
        motor_status = {name: 0 for name in demo.FINGER_NAMES}
        return SimpleNamespace(
            msg=SimpleNamespace(left_or_right=2, **motor_status),
            timestamp=self.timestamp,
        )

    def get_fps(self):
        return 20.0

    def is_ok(self):
        return True

    def position_time_ctrl(self, **values):
        self.calls.append(dict(values))
        mode = values["mode"]
        self.trace.append(f"hand_{mode}")
        payload = {name: values[name] for name in demo.FINGER_NAMES}
        if mode == "pos":
            self.pending_target = payload
        elif mode == "time" and self.pending_target is not None:
            # The in-memory hand reaches the requested target immediately.
            self.positions = self.pending_target.copy()


class LoggedRawHand(FakeHand):
    """Replay exact settled 0..255 finger-position samples from the rig logs."""

    def __init__(self, initial_raw, settled_raw):
        super().__init__((0, 0, 0, 0, 0, 0))
        self.raw_positions = dict(zip(demo.FINGER_NAMES, initial_raw))
        self.settled_raw = dict(zip(demo.FINGER_NAMES, settled_raw))

    def get_finger_pos(self):
        return self._message(self.raw_positions)

    def position_time_ctrl(self, **values):
        super().position_time_ctrl(**values)
        if values["mode"] == "time":
            self.raw_positions = self.settled_raw.copy()


class FakeRobot:
    """Nero double; no constructor or method opens SocketCAN."""

    OPTIONS = SimpleNamespace(EFFECTOR=SimpleNamespace(REVO2="revo2"))

    def __init__(self, hand=None):
        self.trace = []
        self.hand = hand or FakeHand(trace=self.trace)
        self.hand.trace = self.trace
        self.calls = []
        self.joints = [0.0] * 7
        self.pose = [0.20, 0.10, 0.30, 0.0, 0.0, 0.0]
        self.fk_result = [0.21, 0.11, 0.31, 0.0, 0.0, 0.0]
        self.tcp_to_flange_result = None
        self.ik_solution = None
        self.timestamp = 0
        self.arm_status = 0
        self.ctrl_mode = 1
        self.motion_status = 0
        self.enabled = [True] * 7
        self.enable_result = True
        self.freeze_feedback = False

    def _message(self, payload):
        if not self.freeze_feedback:
            self.timestamp += 1
        return SimpleNamespace(msg=payload, timestamp=self.timestamp)

    def init_effector(self, effector):
        self.calls.append(("init_effector", effector))
        return self.hand

    def connect(self):
        self.calls.append(("connect",))

    def disconnect(self):
        self.calls.append(("disconnect",))

    def get_joint_angles(self):
        return self._message(self.joints.copy())

    def get_flange_pose(self):
        return self._message(self.pose.copy())

    def get_arm_status(self):
        status = SimpleNamespace(
            arm_status=self.arm_status,
            ctrl_mode=self.ctrl_mode,
            motion_status=self.motion_status,
        )
        return self._message(status)

    def get_config(self):
        return {
            "joint_limits": {
                f"joint{index}": [-math.pi, math.pi] for index in range(1, 8)
            }
        }

    def get_firmware(self, **_kwargs):
        return {"software_version": "1.20"}

    def get_joints_enable_status_list(self):
        return self.enabled.copy()

    def enable(self, **kwargs):
        self.calls.append(("enable", kwargs))
        self.trace.append("enable")
        if self.enable_result:
            self.arm_status = 0
            self.ctrl_mode = 1
            self.enabled = [True] * 7
        return self.enable_result

    def get_ik_joint_angles(self):
        if self.ik_solution is None:
            return None
        return SimpleNamespace(msg=self.ik_solution.copy(), timestamp=10_000)

    def has_comm_error(self):
        return False

    def get_comm_error(self):
        return None

    def fk(self, joints):
        self.calls.append(("fk", list(joints)))
        return self.fk_result.copy()

    def set_tcp_offset(self, offset):
        self.calls.append(("set_tcp_offset", list(offset)))

    def get_flange2tcp_pose(self, flange_pose):
        self.calls.append(("get_flange2tcp_pose", list(flange_pose)))
        return list(flange_pose)

    def get_tcp2flange_pose(self, tcp_pose):
        self.calls.append(("get_tcp2flange_pose", list(tcp_pose)))
        if self.tcp_to_flange_result is not None:
            return self.tcp_to_flange_result.copy()
        return list(tcp_pose)

    def set_joint_limits_enabled(self, enabled):
        self.calls.append(("set_joint_limits_enabled", enabled))

    def set_speed_percent(self, speed):
        self.calls.append(("set_speed_percent", speed))

    def move_j(self, joints):
        self.calls.append(("move_j", list(joints)))
        self.trace.append("move_j")
        self.joints = list(joints)

    def move_p(self, pose):
        self.calls.append(("move_p", list(pose)))
        self.trace.append("move_p")
        self.pose = list(pose)
        self.fk_result = list(pose)
        self.ik_solution = [0.2] * 7


class SimulatedClock:
    """Advance logical time without waiting through a real arm trajectory."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SlowRobot(FakeRobot):
    """Move toward a joint target over time instead of reaching it instantly."""

    def __init__(self, clock, duration):
        super().__init__()
        self.clock = clock
        self.duration = duration
        self.motion_start = None
        self.start_joints = None
        self.motion_target = None

    def move_j(self, joints):
        self.calls.append(("move_j", list(joints)))
        self.trace.append("move_j")
        self.motion_start = self.clock.monotonic()
        self.start_joints = self.joints.copy()
        self.motion_target = list(joints)
        self.motion_status = 1

    def get_joint_angles(self):
        if self.motion_start is not None:
            fraction = min(
                1.0, (self.clock.monotonic() - self.motion_start) / self.duration
            )
            self.joints = [
                start + fraction * (end - start)
                for start, end in zip(self.start_joints, self.motion_target)
            ]
            self.motion_status = 0 if fraction >= 1.0 else 1
        return super().get_joint_angles()


class NeroRevo2DemoTests(unittest.TestCase):
    def run_with_robot(self, args, robot):
        output = io.StringIO()
        with (
            patch.object(demo, "create_robot", return_value=robot),
            patch.object(demo.time, "sleep"),
            contextlib.redirect_stdout(output),
        ):
            demo.run(args)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def run_with_clock(self, args, robot, clock):
        output = io.StringIO()
        with (
            patch.object(demo, "create_robot", return_value=robot),
            patch.object(demo.time, "monotonic", side_effect=clock.monotonic),
            patch.object(demo.time, "sleep", side_effect=clock.sleep),
            contextlib.redirect_stdout(output),
        ):
            demo.run(args)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def run_menu_with_robot(
        self, responses, robot, *, before_response=None, return_output=False, confirm=True
    ):
        """Drive the real menu and SDK doubles through a scripted terminal."""
        answers = iter(responses)

        def input_fn(prompt):
            if before_response is not None:
                before_response(prompt)
            try:
                return next(answers)
            except StopIteration as error:
                raise EOFError() from error

        output = io.StringIO()
        stderr = io.StringIO()
        recorded_events = []
        original_emit = demo.emit

        def capture_emit(event, **fields):
            recorded_events.append({"event": event, **fields})
            original_emit(event, **fields)

        with (
            patch.object(demo, "create_robot", return_value=robot),
            patch.object(demo.time, "sleep"),
            patch.object(demo, "emit", side_effect=capture_emit),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(stderr),
        ):
            demo.run_interactive(
                argparse.Namespace(channel="can0", command="interactive", confirm=confirm), input_fn
            )
        if return_output:
            return recorded_events, output.getvalue()
        return recorded_events

    def test_human_read_joints_shows_each_axis_angle_limit_and_enable_state(self):
        payload = {
            "event": "read_joints",
            "joint_names": [f"J{axis}" for axis in range(1, 8)],
            "joints_rad": list(demo.READY_HOME_RAD),
            "joints_deg": list(demo.READY_HOME_DEG),
            "joints_enabled": [True, False, True, True, True, True, True],
            "arm_status": 0,
            "ctrl_mode": 1,
            "documented_limits_deg": [
                [-157, 157],
                [-102, 102],
                [-160, 160],
                [-60, 125],
                [-160, 160],
                [-44, 57],
                [-97, 97],
            ],
            "sdk_limits_deg": [
                [-155, 155],
                [-100, 100],
                [-158, 158],
                [-58, 123],
                [-158, 158],
                [-42, 55],
                [-90, 90],
            ],
            "within_sdk_limits": [True] * 7,
        }

        rendered = demo.format_event(payload)

        for axis in range(1, 8):
            self.assertIn(f"J{axis}", rendered)
        for angle in ("55", "-78", "80", "-45", "130", "-30", "40"):
            self.assertIn(angle, rendered)
        self.assertIn("SDK", rendered)
        self.assertIn("使能", rendered)
        self.assertIn("J2", rendered)
        self.assertRegex(rendered, r"(?m)^J1\s+\+55\.000\s")
        self.assertRegex(rendered, r"(?m)^J7\s+\+40\.000\s")
        self.assertNotIn("+55.000°", rendered)
        self.assertNotIn("+40.000°", rendered)
        self.assertIn("-155.0～+155.0", rendered)
        for line in rendered.splitlines():
            if line.startswith("J"):
                self.assertNotIn("°", line)
        self.assertNotIn('"joints_rad"', rendered)

    def test_human_arm_status_summarizes_without_raw_sdk_dump(self):
        payload = {
            "event": "arm_status",
            "joints_rad": list(demo.READY_HOME_RAD),
            "flange_m_rad": [0.114, 0.500, 0.218, -1.182, 0.420, 2.470],
            "status": (
                "ArmMsgFeedbackStatusV111:\n"
                "  ctrl_mode: CAN_CTRL(0x1)\n"
                "  arm_status: NORMAL(0x0)\n"
                "  mode_feedback: MOVE_J(0x1)\n"
                "  motion_status: REACH_TARGET_POS_SUCCESSFULLY(0x0)\n"
                "  err_status: ErrStatus:\n"
                "    joint_1_angle_limit: False\n"
            ),
            "joints_enabled": [True] * 7,
        }

        rendered = demo.format_event(payload)

        self.assertTrue("NORMAL" in rendered or "正常" in rendered)
        self.assertTrue("CAN_CTRL" in rendered or "CAN" in rendered)
        self.assertIn("关节角 (°)：", rendered)
        self.assertNotIn("ArmMsgFeedbackStatusV111", rendered)
        self.assertNotIn("ErrStatus", rendered)
        self.assertLess(len(rendered), 800)

    def test_human_status_fk_and_plans_show_numeric_vectors_on_one_line(self):
        angles_deg = [55.989, -77.998, 80.002, -45.0, 129.994, -29.998, 48.999]
        joints_rad = [math.radians(angle) for angle in angles_deg]
        flange = [
            0.105936,
            0.501986,
            0.217859,
            math.radians(-65.858),
            math.radians(32.360),
            math.radians(146.518),
        ]
        angle_values = "+55.989  -77.998  +80.002  -45.000  +129.994  -29.998  +48.999"
        position_values = "+0.105936  +0.501986  +0.217859"
        orientation_values = "-65.858  +32.360  +146.518"
        payloads = (
            (
                {
                    "event": "arm_status",
                    "joints_rad": joints_rad,
                    "flange_m_rad": flange,
                    "status": "  ctrl_mode: CAN_CTRL(0x1)\n  arm_status: NORMAL(0x0)",
                    "joints_enabled": [True] * 7,
                },
                "关节角 (°)：",
                "法兰",
            ),
            (
                {
                    "event": "fk",
                    "joints_rad": joints_rad,
                    "flange_m_rad": flange,
                },
                "关节角 (°)：",
                "法兰",
            ),
            (
                {
                    "event": "arm_plan",
                    "command": "move-j",
                    "current_joints_rad": joints_rad,
                    "current_flange_m_rad": flange,
                    "flange_target": joints_rad,
                    "target_frame": "joint",
                    "speed_percent": 1,
                    "execute": False,
                },
                "当前关节 (°)：",
                "当前法兰",
            ),
        )
        for payload, angle_label, pose_label in payloads:
            with self.subTest(event=payload["event"]):
                lines = demo.format_event(payload).splitlines()
                self.assertEqual(
                    [line for line in lines if line.startswith(angle_label)],
                    [angle_label + angle_values],
                )
                self.assertEqual(
                    [
                        line
                        for line in lines
                        if line.startswith(f"{pose_label}位置 (m)：")
                    ],
                    [f"{pose_label}位置 (m)：{position_values}"],
                )
                self.assertEqual(
                    [
                        line
                        for line in lines
                        if line.startswith(f"{pose_label}姿态 (°)：")
                    ],
                    [f"{pose_label}姿态 (°)：{orientation_values}"],
                )

        plan_lines = demo.format_event(payloads[-1][0]).splitlines()
        self.assertEqual(
            [line for line in plan_lines if line.startswith("目标关节 (°)：")],
            ["目标关节 (°)：" + angle_values],
        )

    def test_human_enable_and_prepare_plans_keep_copyable_joint_values(self):
        joints_deg = [55.989, -77.998, 80.002, -45.0, 129.994, -29.998, 48.999]
        joints_rad = [math.radians(angle) for angle in joints_deg]
        values = "+55.989  -77.998  +80.002  -45.000  +129.994  -29.998  +48.999"
        plans = (
            {
                "event": "arm_enable_plan",
                "execute": False,
                "joints_enabled": [False] * 7,
                "attempts": 1,
                "current_joints_rad": joints_rad,
            },
            {
                "event": "system_prepare_plan",
                "execute": False,
                "current_joints_deg": joints_deg,
                "current_hand_positions": None,
            },
        )
        for payload in plans:
            with self.subTest(event=payload["event"]):
                lines = demo.format_event(payload).splitlines()
                self.assertIn("当前关节 (°)：" + values, lines)
                self.assertNotIn("+55.989°", "\n".join(lines))

    def test_human_hand_status_shows_six_plain_values_in_actuator_order(self):
        positions = dict(reversed(list(zip(demo.FINGER_NAMES, [0, 69, 0, 0, 0, 0]))))
        currents = dict(reversed(list(zip(demo.FINGER_NAMES, [2, -10, -3, -3, 4, 5]))))
        payload = {
            "event": "hand_status",
            "available_feedback": ["position", "current"],
            "is_ok": True,
            "positions": positions,
            "currents": currents,
            "speeds": None,
            "motor_status": None,
        }

        rendered = demo.format_event(payload)
        lines = rendered.splitlines()

        self.assertIn("六路位置 (0..100)：0  69  0  0  0  0", lines)
        self.assertIn("六路电流：2  -10  -3  -3  4  5", lines)
        self.assertIn("反馈类别：position, current  |  SDK 监测：正常", lines)
        self.assertNotIn("六路速度：", rendered)
        self.assertNotIn("电机状态：", rendered)
        for name in demo.FINGER_NAMES:
            self.assertNotIn(name, rendered)
        self.assertNotIn('"positions"', rendered)

        output = io.StringIO()
        with contextlib.redirect_stdout(output), demo.use_output_format("json"):
            demo.emit(
                "hand_status",
                **{name: value for name, value in payload.items() if name != "event"},
            )
        self.assertEqual(json.loads(output.getvalue()), payload)

    def test_human_fk_distinguishes_flange_and_configured_tcp(self):
        payload = {
            "event": "fk",
            "frame": "base_to_flange",
            "joints_rad": list(demo.READY_HOME_RAD),
            "flange_m_rad": [0.114, 0.500, 0.218, 0.0, 0.0, 0.0],
            "tcp_offset_flange_to_tcp_m_rad": [0.0, 0.0, 0.12, 0.0, 0.0, 0.0],
            "tcp_m_rad": [0.114, 0.500, 0.338, 0.0, 0.0, 0.0],
        }

        rendered = demo.format_event(payload)

        self.assertTrue("法兰" in rendered or "flange" in rendered.lower())
        self.assertIn("TCP", rendered.upper())
        self.assertIn("0.338", rendered)
        self.assertNotIn('"tcp_m_rad"', rendered)

    def test_human_preview_and_error_are_clear(self):
        plan = {
            "event": "arm_plan",
            "command": "move-j",
            "current_joints_rad": list(demo.READY_HOME_RAD),
            "requested_target": list(demo.READY_HOME_RAD),
            "flange_target": list(demo.READY_HOME_RAD),
            "target_frame": "joint",
            "speed_percent": 1,
            "execute": False,
        }

        preview = demo.format_event(plan)
        failure = demo.format_event(
            {"event": "failed", "error": "missing fresh Revo2 finger position"}
        )

        self.assertIn("move-j", preview)
        self.assertTrue("预览" in preview or "preview" in preview.lower())
        self.assertIn("missing fresh Revo2 finger position", failure)
        self.assertTrue("失败" in failure or "error" in failure.lower())
        self.assertNotIn('"event"', preview)

    def test_emit_json_mode_remains_one_parseable_event_per_line(self):
        output = io.StringIO()
        status_fields = {
            "joints_rad": [math.radians(angle) for angle in demo.READY_HOME_DEG],
            "flange_m_rad": [0.105936, 0.501986, 0.217859, 0.0, 0.0, 0.0],
            "status": "  arm_status: NORMAL(0x0)",
            "joints_enabled": [True] * 7,
        }
        with contextlib.redirect_stdout(output), demo.use_output_format("json"):
            demo.emit("read_joints", joints_deg=[55, -78], arm_status=0)
            demo.emit("arm_status", **status_fields)
            demo.emit("failed", error="example")

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual(json.loads(lines[0])["joints_deg"], [55, -78])
        self.assertEqual(json.loads(lines[1]), {"event": "arm_status", **status_fields})
        self.assertEqual(json.loads(lines[2])["error"], "example")

    def test_main_auto_uses_human_on_terminal_and_json_in_pipe(self):
        class TerminalOutput(io.StringIO):
            def isatty(self):
                return True

        for output, argv, expect_json in (
            (TerminalOutput(), ["read-joints"], False),
            (io.StringIO(), ["read-joints"], True),
            (TerminalOutput(), ["--format", "json", "read-joints"], True),
            (io.StringIO(), ["--format", "human", "read-joints"], False),
        ):
            with (
                self.subTest(argv=argv, expect_json=expect_json),
                patch.object(demo, "create_robot", return_value=FakeRobot()),
                patch.object(sys, "argv", ["nero_revo2_demo.py", *argv]),
                contextlib.redirect_stdout(output),
            ):
                demo.main()
            rendered = output.getvalue()
            if expect_json:
                self.assertEqual(json.loads(rendered)["event"], "read_joints")
            else:
                self.assertIn("J1", rendered)
                self.assertNotIn('"joints_rad"', rendered)

    @staticmethod
    def hand_args(**overrides):
        values = {
            "channel": "can0",
            "command": "hand",
            "execute": False,
            "timeout": 0.2,
            "duration": 0.01,
            "positions": None,
            "set_values": ["index=35"],
            "name": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def enable_args(**overrides):
        values = {
            "channel": "can0",
            "command": "enable",
            "execute": False,
            "attempts": 3,
            "timeout": 0.2,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def move_j_args(**overrides):
        values = {
            "channel": "can0",
            "command": "move-j",
            "execute": False,
            "timeout": 0.2,
            "speed": 1,
            "joints_deg": None,
            "joints_rad": [0.1] * 7,
            "joint": None,
            "delta_deg": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def read_joints_args(**overrides):
        values = {"channel": "can0", "command": "read-joints"}
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def move_p_args(**overrides):
        values = {
            "channel": "can0",
            "command": "move-p",
            "execute": False,
            "timeout": 0.2,
            "speed": 1,
            "pose": [0.2, 0.1, 0.3, 0.0, 0.0, 0.0],
            "target_frame": "flange",
            "tcp_offset": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def home_args(**overrides):
        values = {
            "channel": "can0",
            "command": "home",
            "execute": False,
            "timeout": 0.2,
            "speed": 1,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def prepare_args(**overrides):
        values = {
            "channel": "can0",
            "command": "prepare",
            "execute": False,
            "attempts": 3,
            "timeout": 0.2,
            "speed": 1,
            "duration": 0.01,
            "hand_timeout": 0.2,
            "post_settle_time": 0.02,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    @staticmethod
    def valid_home_entry_rad():
        # A measured K3 posture distinct from READY_HOME, used by existing
        # arm and hand regressions; it is no longer a required entry posture.
        return [
            math.radians(value)
            for value in (56.32, -100.35, 82.12, -45.14, 159.40, -31.84, 38.04)
        ]

    @staticmethod
    def long_home_entry_rad():
        # First K3 attempt: J5 alone was 239.558 degrees from ready-home.
        return [
            math.radians(value)
            for value in (
                74.424,
                -100.347,
                -100.968,
                74.195,
                -109.558,
                -38.201,
                -21.240,
            )
        ]

    @staticmethod
    def formerly_blocked_home_entry_rad():
        # This posture was rejected solely because J7 was 9° from READY_HOME.
        return [math.radians(value) for value in (55, -78, 80, -45, 130, -30, 49)]

    def test_gestures_are_complete_and_within_api_range(self):
        for required in ("ok", "yeah", "thumbs-up"):
            self.assertIn(required, demo.GESTURES)
        for name, positions in demo.GESTURES.items():
            with self.subTest(name=name):
                self.assertEqual(len(positions), len(demo.FINGER_NAMES))
                self.assertTrue(all(isinstance(value, int) for value in positions))
                self.assertTrue(all(0 <= value <= 100 for value in positions))

    def test_ready_home_is_seven_axis_consistent_and_avoids_j2_j4_zero(self):
        self.assertEqual(len(demo.READY_HOME_DEG), 7)
        self.assertEqual(len(demo.READY_HOME_RAD), 7)
        for degrees, radians in zip(demo.READY_HOME_DEG, demo.READY_HOME_RAD):
            self.assertAlmostEqual(radians, math.radians(degrees))
        self.assertGreater(abs(demo.READY_HOME_DEG[1]), 10.0)
        self.assertGreater(abs(demo.READY_HOME_DEG[3]), 10.0)
        demo.validate_documented_joints(demo.READY_HOME_RAD)
        self.assertEqual(
            demo.sdk_limit_violations(FakeRobot(), demo.READY_HOME_RAD), []
        )

    def test_parse_and_merge_single_finger_preserves_other_positions(self):
        current = dict(zip(demo.FINGER_NAMES, (10, 20, 30, 40, 50, 60)))
        updates = demo.parse_finger_updates(["index=35"])
        target = demo.merge_hand_target(current, updates)
        self.assertEqual(
            target,
            dict(zip(demo.FINGER_NAMES, (10, 20, 35, 40, 50, 60))),
        )

    def test_parse_and_merge_multiple_fingers_preserves_unspecified_positions(self):
        current = dict(zip(demo.FINGER_NAMES, (10, 20, 30, 40, 50, 60)))
        updates = demo.parse_finger_updates(["thumb_tip=25", "ring=75"])
        target = demo.merge_hand_target(current, updates)
        self.assertEqual(
            target,
            dict(zip(demo.FINGER_NAMES, (25, 20, 30, 40, 75, 60))),
        )

    def test_out_of_range_feedback_requires_explicit_replacement(self):
        current = dict(zip(demo.FINGER_NAMES, (102, 102, 30, 40, 50, 60)))
        with self.assertRaisesRegex(ValueError, "specify --set"):
            demo.merge_hand_target(current, demo.parse_finger_updates(["index=35"]))

        updates = demo.parse_finger_updates(
            ["thumb_tip=100", "thumb_base=100", "index=35"]
        )
        target = demo.merge_hand_target(current, updates)
        self.assertEqual(
            target,
            dict(zip(demo.FINGER_NAMES, (100, 100, 35, 40, 50, 60))),
        )

    def test_position_feedback_scale_accepts_full_raw_range_and_rejects_invalid(self):
        raw = dict(zip(demo.FINGER_NAMES, (0, 46, 105, 171, 250, 255)))
        normalized = demo.position_values(SimpleNamespace(msg=SimpleNamespace(**raw)))
        self.assertEqual(
            [normalized[name] for name in demo.FINGER_NAMES],
            [0, 18, 41, 67, 98, 100],
        )
        for bad in (-1, 256):
            with (
                self.subTest(raw=bad),
                self.assertRaisesRegex(ValueError, "raw position outside"),
            ):
                invalid = {**raw, "thumb_tip": bad}
                demo.position_values(SimpleNamespace(msg=SimpleNamespace(**invalid)))

    def test_invalid_or_ambiguous_finger_updates_are_rejected(self):
        bad_tokens = (
            [],
            ["unknown=10"],
            ["index=101"],
            ["index=-1"],
            ["index=1.5"],
            ["index=1", "index_finger=2"],
        )
        for tokens in bad_tokens:
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                demo.parse_finger_updates(tokens)

    def test_validate_joints_rejects_partial_nonfinite_and_out_of_range_targets(self):
        robot = FakeRobot()
        self.assertEqual(demo.validate_joints(robot, [0.0] * 7), [0.0] * 7)
        for target in (
            [0.0] * 6,
            [0.0] * 6 + [float("nan")],
            [0.0] * 6 + [math.pi + 0.01],
        ):
            with self.subTest(target=target), self.assertRaises(ValueError):
                demo.validate_joints(robot, target)

    def test_orientation_error_handles_rpy_wraparound(self):
        positive_pi = [0.0, 0.0, 0.0, 0.0, 0.0, math.pi]
        negative_pi = [0.0, 0.0, 0.0, 0.0, 0.0, -math.pi]
        quarter_turn = [0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]
        self.assertAlmostEqual(demo.orientation_error(positive_pi, negative_pi), 0.0)
        self.assertAlmostEqual(
            demo.orientation_error([0.0] * 6, quarter_turn), math.pi / 2
        )

    def test_hand_and_gesture_preview_never_send_control_frames(self):
        for args in (
            self.hand_args(),
            self.hand_args(command="gesture", name="ok", set_values=None),
        ):
            hand = FakeHand()
            events = self.run_with_robot(args, FakeRobot(hand))
            with self.subTest(command=args.command):
                self.assertEqual(hand.calls, [])
                self.assertEqual(events[-1]["event"], "preview_only")

    def test_hand_status_is_read_only_and_reports_all_six_actuators(self):
        hand = FakeHand()
        args = argparse.Namespace(channel="can0", command="hand-status")
        events = self.run_with_robot(args, FakeRobot(hand))
        status = next(event for event in events if event["event"] == "hand_status")
        self.assertEqual(set(status["positions"]), set(demo.FINGER_NAMES))
        self.assertEqual(set(status["currents"]), set(demo.FINGER_NAMES))
        self.assertEqual(set(status["speeds"]), set(demo.FINGER_NAMES))
        self.assertEqual(set(status["motor_status"]), set(demo.FINGER_NAMES))
        self.assertEqual(
            status["available_feedback"],
            ["position", "current", "speed", "status"],
        )
        self.assertTrue(status["feedback_complete"])
        self.assertEqual(status["fps"], 20.0)
        self.assertEqual(hand.calls, [])

    def test_hand_status_reports_partial_basic_revo2_feedback(self):
        hand = FakeHand()
        hand.get_finger_spd = lambda: None
        hand.get_hand_status = lambda: None
        args = argparse.Namespace(channel="can0", command="hand-status")

        events = self.run_with_robot(args, FakeRobot(hand))

        status = next(event for event in events if event["event"] == "hand_status")
        self.assertEqual(status["available_feedback"], ["position", "current"])
        self.assertFalse(status["feedback_complete"])
        self.assertIsNone(status["speeds"])
        self.assertIsNone(status["motor_status"])

    def test_logged_raw_hand_gestures_reach_three_distinct_normalized_samples(self):
        settled_samples = {
            "open": (46, 8, 0, 0, 0, 0),
            "ok": (133, 171, 105, 0, 0, 0),
            "fist": (153, 13, 250, 250, 252, 250),
        }
        for gesture, raw in settled_samples.items():
            with self.subTest(gesture=gesture):
                hand = LoggedRawHand((0, 0, 0, 0, 0, 0), raw)
                args = self.hand_args(
                    command="gesture",
                    name=gesture,
                    set_values=None,
                    execute=True,
                    timeout=0.2,
                )
                events = self.run_with_robot(args, FakeRobot(hand))
                reached = next(
                    event for event in events if event["event"] == "hand_target_reached"
                )
                self.assertEqual(reached["fresh_samples"], 3)
                self.assertLessEqual(reached["max_error"], 3)
                self.assertEqual(
                    [hand.calls[0][name] for name in demo.FINGER_NAMES],
                    list(demo.GESTURES[gesture]),
                )
                self.assertEqual([call["mode"] for call in hand.calls], ["pos", "time"])

    def test_hand_status_reports_normalized_positions_and_exact_raw_feedback(self):
        raw = (133, 171, 105, 153, 178, 204)
        hand = LoggedRawHand(raw, raw)
        events = self.run_with_robot(
            argparse.Namespace(channel="can0", command="hand-status"),
            FakeRobot(hand),
        )
        status = next(event for event in events if event["event"] == "hand_status")
        self.assertEqual(
            [status["raw_positions"][name] for name in demo.FINGER_NAMES],
            list(raw),
        )
        self.assertEqual(
            [status["positions"][name] for name in demo.FINGER_NAMES],
            [52, 67, 41, 60, 70, 80],
        )
        self.assertEqual(hand.calls, [])

    def test_partial_hand_command_preserves_other_normalized_raw_channels(self):
        initial_raw = (133, 171, 105, 153, 178, 204)
        settled_raw = (133, 171, 89, 153, 178, 204)
        hand = LoggedRawHand(initial_raw, settled_raw)
        args = self.hand_args(set_values=["index=35"], execute=True)
        events = self.run_with_robot(args, FakeRobot(hand))
        plan = next(event for event in events if event["event"] == "hand_plan")
        self.assertEqual(
            [plan["target"][name] for name in demo.FINGER_NAMES],
            [52, 67, 35, 60, 70, 80],
        )
        self.assertEqual(
            [hand.calls[0][name] for name in demo.FINGER_NAMES],
            [52, 67, 35, 60, 70, 80],
        )
        self.assertIn("hand_target_reached", [event["event"] for event in events])

    def test_hand_execute_sends_two_complete_six_actuator_frames(self):
        hand = FakeHand((10, 20, 30, 40, 50, 60))
        robot = FakeRobot(hand)
        events = self.run_with_robot(self.hand_args(execute=True), robot)

        self.assertEqual(len(hand.calls), 2)
        position_frame, time_frame = hand.calls
        self.assertEqual(position_frame["mode"], "pos")
        self.assertEqual(time_frame["mode"], "time")
        self.assertEqual(set(position_frame) - {"mode"}, set(demo.FINGER_NAMES))
        self.assertEqual(set(time_frame) - {"mode"}, set(demo.FINGER_NAMES))
        self.assertEqual(
            [position_frame[name] for name in demo.FINGER_NAMES],
            [10, 20, 35, 40, 50, 60],
        )
        self.assertEqual(
            [time_frame[name] for name in demo.FINGER_NAMES],
            [1] * 6,
        )
        self.assertIn("hand_target_reached", [event["event"] for event in events])

    def test_hand_cached_target_feedback_cannot_count_as_three_samples(self):
        hand = FakeHand((10, 20, 35, 40, 50, 60))
        robot = FakeRobot(hand)
        original_control = hand.position_time_ctrl

        def send_then_freeze(**values):
            original_control(**values)
            if values["mode"] == "time":
                hand.freeze_feedback = True

        hand.position_time_ctrl = send_then_freeze

        with self.assertRaisesRegex(TimeoutError, "hand did not reach"):
            self.run_with_robot(
                self.hand_args(execute=True, timeout=0.01),
                robot,
            )

    def test_arm_motion_previews_never_send_move_commands(self):
        for args in (self.move_j_args(), self.move_p_args()):
            robot = FakeRobot()
            events = self.run_with_robot(args, robot)
            with self.subTest(command=args.command):
                call_names = [call[0] for call in robot.calls]
                self.assertNotIn("move_j", call_names)
                self.assertNotIn("move_p", call_names)
                self.assertNotIn("enable", call_names)
                self.assertNotIn("set_joint_limits_enabled", call_names)
                self.assertEqual(events[-1]["event"], "preview_only")

    def test_read_joints_reports_seven_axis_feedback_in_degrees_and_radians(self):
        robot = FakeRobot()
        robot.joints = list(demo.READY_HOME_RAD)
        robot.enabled = [True, False, True, True, False, True, True]
        robot.arm_status = 6

        events = self.run_with_robot(self.read_joints_args(), robot)

        self.assertEqual(len(events), 1)
        feedback = events[0]
        self.assertEqual(feedback["event"], "read_joints")
        self.assertEqual(feedback["joint_names"], [f"J{axis}" for axis in range(1, 8)])
        self.assertEqual(feedback["joints_rad"], list(demo.READY_HOME_RAD))
        for actual, expected in zip(feedback["joints_deg"], demo.READY_HOME_DEG):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(feedback["joints_enabled"], robot.enabled)
        self.assertEqual(feedback["arm_status"], 6)
        self.assertEqual(feedback["ctrl_mode"], 1)
        self.assertEqual(len(feedback["documented_limits_deg"]), 7)
        self.assertEqual(feedback["documented_limits_deg"][0], [-157, 157])
        self.assertAlmostEqual(feedback["documented_limits_deg"][3][0], -60.0)
        self.assertAlmostEqual(feedback["documented_limits_deg"][3][1], 125.0)
        self.assertEqual(len(feedback["sdk_limits_deg"]), 7)
        for limits in feedback["sdk_limits_deg"]:
            self.assertAlmostEqual(limits[0], -180.0)
            self.assertAlmostEqual(limits[1], 180.0)
        self.assertEqual(feedback["within_sdk_limits"], [True] * 7)

        call_names = [call[0] for call in robot.calls]
        self.assertEqual(call_names.count("connect"), 1)
        self.assertEqual(call_names.count("disconnect"), 1)
        for control in (
            "enable",
            "move_j",
            "move_p",
            "set_speed_percent",
            "set_joint_limits_enabled",
            "init_effector",
        ):
            self.assertNotIn(control, call_names)

    def test_read_joints_identifies_axes_outside_sdk_soft_limits_without_control(self):
        robot = FakeRobot()
        robot.joints = [0.0, -math.radians(101), 0.0, -0.4, math.radians(159), 0.0, 0.4]
        sdk_limits = {
            f"joint{index}": list(limits)
            for index, limits in enumerate(demo.NERO_DOCUMENTED_LIMITS_RAD, start=1)
        }
        sdk_limits["joint2"] = [-math.radians(100), math.radians(100)]
        sdk_limits["joint5"] = [-math.radians(158), math.radians(158)]
        robot.get_config = lambda: {"joint_limits": sdk_limits}

        feedback = self.run_with_robot(self.read_joints_args(), robot)[0]

        self.assertEqual(
            feedback["within_sdk_limits"],
            [True, False, True, True, False, True, True],
        )
        self.assertAlmostEqual(feedback["sdk_limits_deg"][1][0], -100.0)
        self.assertAlmostEqual(feedback["sdk_limits_deg"][4][1], 158.0)
        self.assertEqual([call[0] for call in robot.calls], ["connect", "disconnect"])

    def test_read_joints_keeps_feedback_available_without_sdk_limit_config(self):
        robot = FakeRobot()
        robot.joints = [0.1] * 7
        robot.get_config = lambda: {"joint_limits": {}}

        feedback = self.run_with_robot(self.read_joints_args(), robot)[0]

        self.assertEqual(feedback["joints_rad"], [0.1] * 7)
        self.assertIsNone(feedback["sdk_limits_deg"])
        self.assertIsNone(feedback["within_sdk_limits"])
        self.assertEqual(len(feedback["documented_limits_deg"]), 7)
        self.assertEqual([call[0] for call in robot.calls], ["connect", "disconnect"])

    def test_read_joints_function_uses_existing_connection_without_control(self):
        robot = FakeRobot()
        robot.joints = list(demo.READY_HOME_RAD)

        feedback = demo.read_joints(robot)

        self.assertEqual(feedback["joints_rad"], list(demo.READY_HOME_RAD))
        self.assertEqual(robot.calls, [])

    def test_read_joints_rejects_cached_joint_feedback_and_disconnects(self):
        robot = FakeRobot()
        robot.freeze_feedback = True

        with patch.object(demo.time, "monotonic", side_effect=[0.0, 0.1, 2.1]):
            with self.assertRaisesRegex(TimeoutError, "fresh joint feedback"):
                self.run_with_robot(self.read_joints_args(), robot)

        self.assertEqual([call[0] for call in robot.calls], ["connect", "disconnect"])

    def test_read_joints_result_can_be_used_for_existing_move_j_delta_preview(self):
        robot = FakeRobot()
        robot.joints = list(demo.READY_HOME_RAD)
        read_feedback = self.run_with_robot(self.read_joints_args(), robot)[0]

        args = self.move_j_args(
            joints_rad=None,
            joint=7,
            delta_deg=1.0,
        )
        plan = next(
            event
            for event in self.run_with_robot(args, robot)
            if event["event"] == "arm_plan"
        )

        expected = read_feedback["joints_rad"].copy()
        expected[6] += math.radians(1.0)
        for actual, target in zip(plan["requested_target"], expected):
            self.assertAlmostEqual(actual, target)
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_home_preview_sends_neither_enable_nor_motion_frames(self):
        robot = FakeRobot()
        robot.joints = self.formerly_blocked_home_entry_rad()

        events = self.run_with_robot(self.home_args(), robot)

        call_names = [call[0] for call in robot.calls]
        self.assertNotIn("enable", call_names)
        self.assertNotIn("move_j", call_names)
        self.assertNotIn("move_p", call_names)
        self.assertNotIn("set_joint_limits_enabled", call_names)
        plan = next(event for event in events if event["event"] == "arm_plan")
        self.assertEqual(plan["command"], "home")
        self.assertEqual(plan["requested_target"], list(demo.READY_HOME_RAD))
        self.assertEqual(events[-1]["event"], "preview_only")

    def test_home_execute_requires_arm_to_be_enabled_before_motion(self):
        robot = FakeRobot()
        robot.joints = self.valid_home_entry_rad()
        robot.arm_status = 6
        robot.enabled = [False] * 7

        with self.assertRaisesRegex(RuntimeError, "NORMAL|enabled"):
            self.run_with_robot(self.home_args(execute=True), robot)

        call_names = [call[0] for call in robot.calls]
        self.assertNotIn("enable", call_names)
        self.assertNotIn("move_j", call_names)

    def test_home_execute_moves_from_measured_entry_to_ready_home(self):
        robot = FakeRobot()
        robot.joints = self.valid_home_entry_rad()

        events = self.run_with_robot(self.home_args(execute=True), robot)

        move_calls = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(move_calls, [("move_j", list(demo.READY_HOME_RAD))])
        self.assertNotIn("enable", [call[0] for call in robot.calls])
        reached = next(
            event for event in events if event["event"] == "arm_target_reached"
        )
        self.assertEqual(reached["command"], "home")

    def test_long_home_preview_reports_effective_wait_without_sending_control(self):
        parser = demo.make_parser()
        robot = FakeRobot()
        robot.joints = self.long_home_entry_rad()

        events = self.run_with_robot(parser.parse_args(["home", "--speed", "1"]), robot)

        plan = next(event for event in events if event["event"] == "arm_plan")
        # J5's 239.558° at 1% of its 225°/s maximum dominates the seven axes.
        self.assertEqual(plan["wait_timeout_s"], 243)
        self.assertEqual(plan["timeout_source"], "auto")
        self.assertGreater(plan["wait_timeout_s"], 140)
        self.assertNotIn("move_j", [call[0] for call in robot.calls])
        self.assertNotIn("enable", [call[0] for call in robot.calls])

    def test_home_manual_timeout_overrides_distance_based_wait(self):
        parser = demo.make_parser()
        robot = FakeRobot()
        robot.joints = self.long_home_entry_rad()

        events = self.run_with_robot(
            parser.parse_args(["home", "--speed", "1", "--timeout", "40"]),
            robot,
        )

        plan = next(event for event in events if event["event"] == "arm_plan")
        self.assertEqual(plan["wait_timeout_s"], 40)
        self.assertEqual(plan["timeout_source"], "manual")
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_home_auto_wait_observes_a_slow_arrival_past_30s_without_hold(self):
        clock = SimulatedClock()
        robot = SlowRobot(clock, duration=110.0)
        robot.joints = self.long_home_entry_rad()
        args = demo.make_parser().parse_args(["home", "--execute", "--speed", "1"])

        events = self.run_with_clock(args, robot, clock)

        self.assertGreater(clock.monotonic(), 100.0)
        self.assertIn("arm_target_reached", [event["event"] for event in events])
        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_j"],
            [("move_j", list(demo.READY_HOME_RAD))],
        )
        self.assertNotIn("hold_requested", [event["event"] for event in events])

    def test_home_manual_timeout_still_requests_hold_on_nonarrival(self):
        clock = SimulatedClock()
        robot = SlowRobot(clock, duration=110.0)
        robot.joints = self.long_home_entry_rad()
        args = demo.make_parser().parse_args(
            ["home", "--execute", "--speed", "1", "--timeout", "30"]
        )

        with self.assertRaisesRegex(TimeoutError, "home.*30"):
            self.run_with_clock(args, robot, clock)

        self.assertGreaterEqual(clock.monotonic(), 30.0)
        move_calls = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(len(move_calls), 2)
        self.assertEqual(move_calls[0], ("move_j", list(demo.READY_HOME_RAD)))
        self.assertNotEqual(move_calls[1][1], list(demo.READY_HOME_RAD))
        self.assertEqual(move_calls[1][1], robot.joints)

    def test_home_already_at_target_does_not_resend_motion(self):
        robot = FakeRobot()
        robot.joints = list(demo.READY_HOME_RAD)

        events = self.run_with_robot(self.home_args(execute=True), robot)

        self.assertNotIn("move_j", [call[0] for call in robot.calls])
        reached = next(
            event for event in events if event["event"] == "arm_target_reached"
        )
        self.assertTrue(reached["already_at_target"])

    def test_home_execute_accepts_formerly_blocked_j7_entry(self):
        robot = FakeRobot()
        robot.joints = self.formerly_blocked_home_entry_rad()

        events = self.run_with_robot(self.home_args(execute=True), robot)

        plans = [event for event in events if event["event"] == "arm_plan"]
        self.assertEqual(len(plans), 1)
        self.assertTrue(plans[0]["execute"])
        self.assertEqual(
            plans[0]["current_joints_rad"], self.formerly_blocked_home_entry_rad()
        )
        self.assertEqual(plans[0]["flange_target"], list(demo.READY_HOME_RAD))
        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_j"],
            [("move_j", list(demo.READY_HOME_RAD))],
        )
        self.assertIn("arm_target_reached", [event["event"] for event in events])

    def test_home_execute_from_unreferenced_within_limits_posture(self):
        robot = FakeRobot()
        robot.joints = [
            math.radians(value) for value in (60, -70, 75, -40, 125, -25, 49)
        ]
        demo.validate_documented_joints(robot.joints)

        events = self.run_with_robot(self.home_args(execute=True), robot)

        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_j"],
            [("move_j", list(demo.READY_HOME_RAD))],
        )
        self.assertIn("arm_target_reached", [event["event"] for event in events])

    def test_home_execute_rejects_current_joint_outside_documented_limits(self):
        robot = FakeRobot()
        robot.joints = self.formerly_blocked_home_entry_rad()
        robot.joints[0] = math.radians(158)

        with self.assertRaisesRegex(ValueError, "documented Nero ranges"):
            self.run_with_robot(self.home_args(execute=True), robot)

        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_enable_preview_does_not_release_brakes(self):
        robot = FakeRobot()
        robot.arm_status = 6
        robot.enabled = [False] * 7

        events = self.run_with_robot(self.enable_args(), robot)

        self.assertNotIn("enable", [call[0] for call in robot.calls])
        self.assertEqual(events[-1]["event"], "preview_only")

    def test_enable_execute_releases_all_joints_and_verifies_feedback(self):
        robot = FakeRobot()
        robot.arm_status = 6
        robot.ctrl_mode = 1
        robot.enabled = [False] * 7

        events = self.run_with_robot(self.enable_args(execute=True), robot)

        self.assertIn(("enable", {"timeout": 1.5}), robot.calls)
        enabled = next(event for event in events if event["event"] == "arm_enabled")
        self.assertEqual(enabled["arm_status"], 0)
        self.assertEqual(enabled["ctrl_mode"], 1)
        self.assertEqual(enabled["joints_enabled"], [True] * 7)

    def test_enable_normal_web_control_still_switches_to_can(self):
        robot = FakeRobot()
        robot.arm_status = 0
        robot.ctrl_mode = 3
        robot.enabled = [True] * 7

        events = self.run_with_robot(self.enable_args(execute=True), robot)

        self.assertIn(("enable", {"timeout": 1.5}), robot.calls)
        self.assertEqual(events[-1]["event"], "arm_enabled")
        self.assertEqual(events[-1]["ctrl_mode"], 1)

    def test_enable_rejects_normal_values_from_a_cached_status_timestamp(self):
        robot = FakeRobot()
        robot.arm_status = 6
        robot.enabled = [False] * 7
        original_enable = robot.enable

        def enable_then_freeze(**kwargs):
            result = original_enable(**kwargs)
            robot.freeze_feedback = True
            return result

        robot.enable = enable_then_freeze

        with self.assertRaisesRegex(TimeoutError, "feedback|enable"):
            self.run_with_robot(
                self.enable_args(execute=True, timeout=0.01),
                robot,
            )

    def test_prepare_orders_enable_home_and_complete_hand_command(self):
        hand = FakeHand()
        robot = FakeRobot(hand)
        robot.joints = self.formerly_blocked_home_entry_rad()
        robot.arm_status = 6
        robot.enabled = [False] * 7

        events = self.run_with_robot(self.prepare_args(execute=True), robot)

        self.assertLess(robot.trace.index("enable"), robot.trace.index("move_j"))
        self.assertLess(robot.trace.index("move_j"), robot.trace.index("hand_pos"))
        self.assertLess(robot.trace.index("hand_pos"), robot.trace.index("hand_time"))
        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_j"],
            [("move_j", list(demo.READY_HOME_RAD))],
        )
        self.assertEqual(len(hand.calls), 2)
        position_frame, time_frame = hand.calls
        self.assertEqual(position_frame["mode"], "pos")
        self.assertEqual(time_frame["mode"], "time")
        self.assertEqual(set(position_frame) - {"mode"}, set(demo.FINGER_NAMES))
        self.assertEqual(set(time_frame) - {"mode"}, set(demo.FINGER_NAMES))
        self.assertEqual(
            [position_frame[name] for name in demo.FINGER_NAMES],
            list(demo.GESTURES["open"]),
        )
        self.assertEqual(events[-1]["event"], "system_position_hold_observed")
        self.assertNotIn("system_ready", [event["event"] for event in events])
        self.assertTrue(events[-1]["verified_after_sdk_reconnect"])
        self.assertGreaterEqual(events[-1]["hand_fresh_samples"], 3)
        self.assertEqual(events[-1]["hand_enable_state"], "not_exposed_by_basic_sdk")
        self.assertEqual(
            [call for call in robot.calls if call[0] == "connect"],
            [("connect",), ("connect",)],
        )

    def test_prepare_auto_sizes_only_home_wait_and_keeps_enable_hand_windows(self):
        robot = FakeRobot()
        robot.joints = self.long_home_entry_rad()
        robot.arm_status = 6
        robot.enabled = [False] * 7

        with (
            patch.object(
                demo, "enable_arm_connected", wraps=demo.enable_arm_connected
            ) as enable,
            patch.object(
                demo, "wait_for_arm_target", wraps=demo.wait_for_arm_target
            ) as wait,
            patch.object(
                demo, "run_hand_connected", wraps=demo.run_hand_connected
            ) as hand,
        ):
            events = self.run_with_robot(
                self.prepare_args(execute=True, timeout=None), robot
            )

        self.assertEqual(enable.call_args.args[0].timeout, 5.0)
        self.assertEqual(wait.call_args.args[3], 243)
        self.assertEqual(hand.call_args.args[0].timeout, 0.2)
        self.assertEqual(hand.call_args.args[0].duration, 0.01)
        self.assertLess(robot.trace.index("enable"), robot.trace.index("move_j"))
        self.assertLess(robot.trace.index("move_j"), robot.trace.index("hand_pos"))
        self.assertEqual(len(robot.hand.calls), 2)
        self.assertEqual(events[-1]["event"], "system_position_hold_observed")

    def test_prepare_manual_home_wait_does_not_extend_enable_preflight(self):
        robot = FakeRobot()
        robot.joints = self.long_home_entry_rad()
        robot.arm_status = 6
        robot.enabled = [False] * 7

        with (
            patch.object(
                demo, "enable_arm_connected", wraps=demo.enable_arm_connected
            ) as enable,
            patch.object(
                demo, "wait_for_arm_target", wraps=demo.wait_for_arm_target
            ) as wait,
        ):
            self.run_with_robot(self.prepare_args(execute=True, timeout=40.0), robot)

        self.assertEqual(enable.call_args.args[0].timeout, 5.0)
        self.assertEqual(wait.call_args.args[3], 40.0)
        self.assertEqual(len(robot.hand.calls), 2)

    def test_prepare_rejects_hand_rebound_after_sdk_disconnect(self):
        hand = FakeHand()
        robot = FakeRobot(hand)
        robot.joints = self.valid_home_entry_rad()
        original_disconnect = robot.disconnect
        disconnect_count = 0

        def disconnect_with_rebound():
            nonlocal disconnect_count
            original_disconnect()
            disconnect_count += 1
            if disconnect_count == 1:
                hand.positions.update(thumb_tip=46, thumb_base=8)

        robot.disconnect = disconnect_with_rebound

        with self.assertRaisesRegex(RuntimeError, "Revo2 left open target"):
            self.run_with_robot(self.prepare_args(execute=True), robot)

        self.assertEqual(len(hand.calls), 2)
        self.assertEqual(robot.joints, list(demo.READY_HOME_RAD))
        self.assertEqual(robot.enabled, [True] * 7)

    def test_prepare_rejects_delayed_hand_drift_during_hold_observation(self):
        hand = FakeHand()
        robot = FakeRobot(hand)
        robot.joints = self.valid_home_entry_rad()
        original_disconnect = robot.disconnect
        original_get_position = hand.get_finger_pos
        after_disconnect = False
        post_disconnect_reads = 0

        def disconnect_and_mark():
            nonlocal after_disconnect
            original_disconnect()
            after_disconnect = True

        def position_then_drift():
            nonlocal post_disconnect_reads
            if after_disconnect:
                post_disconnect_reads += 1
                if post_disconnect_reads == 8:
                    hand.positions.update(thumb_tip=46, thumb_base=8)
            return original_get_position()

        robot.disconnect = disconnect_and_mark
        hand.get_finger_pos = position_then_drift

        with self.assertRaisesRegex(RuntimeError, "drifted after SDK reconnect"):
            self.run_with_robot(self.prepare_args(execute=True), robot)

        self.assertGreaterEqual(post_disconnect_reads, 8)

    def test_prepare_rejects_lost_arm_enable_after_reconnect(self):
        hand = FakeHand()
        robot = FakeRobot(hand)
        robot.joints = self.valid_home_entry_rad()
        original_connect = robot.connect
        connect_count = 0

        def connect_with_lost_enable():
            nonlocal connect_count
            original_connect()
            connect_count += 1
            if connect_count == 2:
                robot.enabled[6] = False

        robot.connect = connect_with_lost_enable

        with self.assertRaisesRegex(
            RuntimeError, "seven joints must already be enabled"
        ):
            self.run_with_robot(self.prepare_args(execute=True), robot)

        self.assertEqual(len(hand.calls), 2)
        self.assertEqual(robot.joints, list(demo.READY_HOME_RAD))

    def test_move_j_execute_enables_limits_sets_speed_and_reaches_seven_joint_target(
        self,
    ):
        target = [0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70]
        robot = FakeRobot()
        events = self.run_with_robot(
            self.move_j_args(execute=True, joints_rad=target, speed=7),
            robot,
        )

        self.assertIn(("set_joint_limits_enabled", True), robot.calls)
        self.assertIn(("set_speed_percent", 7), robot.calls)
        move_calls = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(move_calls, [("move_j", target)])
        self.assertEqual(len(move_calls[0][1]), 7)
        reached = [event for event in events if event["event"] == "arm_target_reached"]
        self.assertEqual(len(reached), 1)
        self.assertEqual(reached[0]["command"], "move-j")

    def test_move_j_cached_target_feedback_cannot_count_as_stable_samples(self):
        target = [0.1] * 7
        robot = FakeRobot()
        original_move_j = robot.move_j

        def move_then_freeze(joints):
            original_move_j(joints)
            robot.freeze_feedback = True

        robot.move_j = move_then_freeze

        with self.assertRaisesRegex(TimeoutError, "did not reach"):
            self.run_with_robot(
                self.move_j_args(execute=True, joints_rad=target, timeout=0.01),
                robot,
            )

    def test_move_exception_after_command_attempt_still_requests_gated_hold(self):
        target = [0.1] * 7
        feedback_after_partial_send = [0.02] * 7
        robot = FakeRobot()
        send_count = 0

        def partially_send_then_raise(joints):
            nonlocal send_count
            send_count += 1
            robot.calls.append(("move_j", list(joints)))
            robot.trace.append("move_j")
            if send_count == 1:
                robot.joints = feedback_after_partial_send.copy()
                raise RuntimeError("SDK raised after writing a partial frame")
            robot.joints = list(joints)

        robot.move_j = partially_send_then_raise

        with self.assertRaisesRegex(RuntimeError, "partial frame"):
            self.run_with_robot(
                self.move_j_args(execute=True, joints_rad=target),
                robot,
            )

        move_calls = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(len(move_calls), 2)
        self.assertEqual(move_calls[0], ("move_j", target))
        self.assertEqual(move_calls[1], ("move_j", feedback_after_partial_send))

    def test_move_j_delta_preserves_other_axes_outside_sdk_soft_limits(self):
        robot = FakeRobot()
        robot.joints = [0.0, -1.76, 0.0, -0.4, 2.79, 0.0, 0.4]
        sdk_limits = {
            f"joint{index}": list(limits)
            for index, limits in enumerate(demo.NERO_DOCUMENTED_LIMITS_RAD, start=1)
        }
        sdk_limits["joint2"] = [-math.radians(100), math.radians(100)]
        sdk_limits["joint5"] = [-math.radians(158), math.radians(158)]
        robot.get_config = lambda: {"joint_limits": sdk_limits}
        args = self.move_j_args(
            execute=True,
            joints_rad=None,
            joint=7,
            delta_deg=1.0,
        )

        events = self.run_with_robot(args, robot)

        target = robot.joints
        self.assertEqual(target[1], -1.76)
        self.assertEqual(target[4], 2.79)
        self.assertAlmostEqual(target[6], 0.4 + math.radians(1.0))
        self.assertIn(("set_joint_limits_enabled", False), robot.calls)
        warning = next(
            event for event in events if event["event"] == "sdk_soft_limit_warning"
        )
        self.assertEqual(warning["joints"], [2, 5])

    def test_move_p_execute_sends_flange_target_and_reports_ik_fk_round_trip(self):
        target = [0.24, 0.12, 0.32, 0.10, -0.20, 0.30]
        robot = FakeRobot()
        events = self.run_with_robot(
            self.move_p_args(execute=True, pose=target, speed=6),
            robot,
        )

        self.assertIn(("set_joint_limits_enabled", True), robot.calls)
        self.assertIn(("set_speed_percent", 6), robot.calls)
        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_p"],
            [("move_p", target)],
        )
        ik_event = next(event for event in events if event["event"] == "ik_solution")
        self.assertEqual(ik_event["joints_rad"], [0.2] * 7)
        self.assertEqual(ik_event["fk_flange_m_rad"], target)
        reached = next(
            event for event in events if event["event"] == "arm_target_reached"
        )
        self.assertTrue(reached["ik_feedback_received"])

    def test_move_p_accepts_controller_ik_inside_documented_mechanical_limits(self):
        target = [0.24, 0.12, 0.32, 0.10, -0.20, 0.30]
        robot = FakeRobot()
        expected_ik = [0.0, -1.76, 0.0, -0.4, 2.79, 0.0, 0.4]
        robot.joints = expected_ik.copy()

        def move_p_without_replacing_ik(pose):
            robot.calls.append(("move_p", list(pose)))
            robot.pose = list(pose)
            robot.fk_result = list(pose)
            robot.ik_solution = expected_ik.copy()

        robot.move_p = move_p_without_replacing_ik
        sdk_limits = {
            f"joint{index}": list(limits)
            for index, limits in enumerate(demo.NERO_DOCUMENTED_LIMITS_RAD, start=1)
        }
        sdk_limits["joint2"] = [-math.radians(100), math.radians(100)]
        sdk_limits["joint5"] = [-math.radians(158), math.radians(158)]
        robot.get_config = lambda: {"joint_limits": sdk_limits}

        events = self.run_with_robot(
            self.move_p_args(execute=True, pose=target),
            robot,
        )

        self.assertIn(("set_joint_limits_enabled", False), robot.calls)
        ik_event = next(event for event in events if event["event"] == "ik_solution")
        self.assertEqual(ik_event["joints_rad"], expected_ik)

    def test_move_p_does_not_report_success_without_new_ik_feedback(self):
        target = [0.24, 0.12, 0.32, 0.10, -0.20, 0.30]
        robot = FakeRobot()

        def move_p_without_ik(pose):
            robot.calls.append(("move_p", list(pose)))
            robot.pose = list(pose)

        robot.move_p = move_p_without_ik
        with self.assertRaisesRegex(TimeoutError, "did not reach"):
            self.run_with_robot(
                self.move_p_args(execute=True, pose=target, timeout=0.01),
                robot,
            )

        move_j_calls = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(move_j_calls, [("move_j", [0.0] * 7)])

    def test_move_p_tcp_target_is_converted_to_flange_before_send(self):
        requested_tcp = [0.30, 0.20, 0.40, 0.0, 0.0, 0.0]
        flange_target = [0.30, 0.20, 0.28, 0.0, 0.0, 0.0]
        tcp_offset = [0.0, 0.0, 0.12, 0.0, 0.0, 0.0]
        robot = FakeRobot()
        robot.tcp_to_flange_result = flange_target
        events = self.run_with_robot(
            self.move_p_args(
                execute=True,
                pose=requested_tcp,
                target_frame="tcp",
                tcp_offset=tcp_offset,
            ),
            robot,
        )

        self.assertIn(("set_tcp_offset", tcp_offset), robot.calls)
        self.assertIn(("get_tcp2flange_pose", requested_tcp), robot.calls)
        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_p"],
            [("move_p", flange_target)],
        )
        plan = next(event for event in events if event["event"] == "arm_plan")
        self.assertEqual(plan["requested_target"], requested_tcp)
        self.assertEqual(plan["flange_target"], flange_target)
        conversion_index = robot.calls.index(("get_tcp2flange_pose", requested_tcp))
        send_index = robot.calls.index(("move_p", flange_target))
        self.assertLess(conversion_index, send_index)

    def test_fk_with_current_feedback_connects_and_compares_controller_pose(self):
        robot = FakeRobot()
        args = argparse.Namespace(
            channel="can0",
            command="fk",
            joints_deg=None,
            joints_rad=None,
            tcp_offset=None,
        )
        events = self.run_with_robot(args, robot)
        self.assertEqual(events[-1]["event"], "fk")
        self.assertIn("controller_flange_m_rad", events[-1])
        self.assertEqual([call[0] for call in robot.calls].count("connect"), 1)
        self.assertEqual([call[0] for call in robot.calls].count("disconnect"), 1)

    def test_fk_with_supplied_joints_is_offline(self):
        robot = FakeRobot()
        joints = [0.1] * 7
        args = argparse.Namespace(
            channel="can0",
            command="fk",
            joints_deg=None,
            joints_rad=joints,
            tcp_offset=None,
        )
        events = self.run_with_robot(args, robot)
        self.assertEqual(events[-1]["joints_rad"], joints)
        call_names = [call[0] for call in robot.calls]
        self.assertNotIn("connect", call_names)
        self.assertNotIn("disconnect", call_names)
        self.assertIn("fk", call_names)

    def test_cli_accepts_move_command_spellings(self):
        cases = (
            (["enable"], "enable"),
            (["arm-enable"], "enable"),
            (["read-joints"], "read-joints"),
            (["read_joints"], "read-joints"),
            (["home"], "home"),
            (["prepare"], "prepare"),
            (
                ["move-j", "--joints-rad", *("0" for _ in range(7))],
                "move-j",
            ),
            (
                ["move_j", "--joints-rad", *("0" for _ in range(7))],
                "move-j",
            ),
            (
                ["move-p", "--pose", *("0" for _ in range(6))],
                "move-p",
            ),
            (
                ["move_p", "--pose", *("0" for _ in range(6))],
                "move-p",
            ),
        )
        for argv, expected in cases:
            with self.subTest(argv=argv):
                parser = demo.make_parser()
                args = parser.parse_args(argv)
                demo.validate_arguments(parser, args)
                self.assertEqual(args.command, expected)

    def test_parser_defaults_to_menu_while_preserving_existing_cli(self):
        parser = demo.make_parser()
        for argv in ([], ["interactive"], ["menu"]):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                demo.validate_arguments(parser, args)
                self.assertEqual(args.command, "interactive")
        for argv, command in (
            (["read-joints"], "read-joints"),
            (["move-j", "--joint", "7", "--delta-deg", "1"], "move-j"),
            (["hand", "--set", "index=35"], "hand"),
        ):
            with self.subTest(argv=argv):
                args = parser.parse_args(argv)
                demo.validate_arguments(parser, args)
                self.assertEqual(args.command, command)
                self.assertFalse(getattr(args, "execute", False))

    def test_home_and_prepare_auto_defaults_preserve_other_command_timeouts(self):
        parser = demo.make_parser()
        cases = (
            (["home"], None),
            (["prepare"], None),
            (["move-j", "--joint", "7", "--delta-deg", "1"], 30.0),
            (["move-p", "--pose", "0.2", "0.1", "0.3", "0", "0", "0"], 30.0),
            (["hand", "--set", "index=35"], 5.0),
        )
        for argv, expected in cases:
            with self.subTest(command=argv[0]):
                args = parser.parse_args(argv)
                demo.validate_arguments(parser, args)
                self.assertEqual(args.timeout, expected)

    def test_parser_output_format_defaults_to_auto_and_accepts_overrides(self):
        parser = demo.make_parser()
        self.assertEqual(parser.parse_args(["read-joints"]).format, "auto")
        for mode in ("human", "json"):
            with self.subTest(mode=mode):
                args = parser.parse_args(["--format", mode, "read-joints"])
                self.assertEqual(args.format, mode)

    def test_menu_choices_build_valid_cli_arguments_for_all_operations(self):
        responses = {
            "1": [],
            "2": [],
            "3": ["1", "7", "1", "1"],
            "4": ["0.2 0.1 0.3 0 0 0", "1", "1"],
            "5": ["1", ""],
            "6": [],
            "7": ["1", "index=35 middle=40", "1"],
            "8": ["ok", "1"],
            "9": ["3"],
            "10": ["1", ""],
            "11": ["1", "", "1", "8"],
        }
        parser = demo.make_parser()
        self.assertEqual(set(responses), set(demo.INTERACTIVE_CHOICES))
        for choice, answers in responses.items():
            with self.subTest(choice=choice):
                iterator = iter(answers)
                argv = demo.interactive_command_argv(
                    choice, lambda _prompt: next(iterator)
                )
                args = parser.parse_args(["--channel", "can0", *argv])
                demo.validate_arguments(parser, args)
                self.assertEqual(args.command, demo.INTERACTIVE_CHOICES[choice][1])
                self.assertEqual(args.channel, "can0")
                self.assertFalse(getattr(args, "execute", False))

    def test_read_joints_from_menu_only_reads_feedback_then_exits(self):
        robot = FakeRobot()
        robot.joints = [
            math.radians(value) for value in (55, -78, 80, -45, 130, -30, 40)
        ]
        events = self.run_menu_with_robot(["2", "0"], robot)
        self.assertEqual(
            [event["event"] for event in events], ["read_joints", "menu_exit"]
        )
        self.assertEqual([call[0] for call in robot.calls], ["connect", "disconnect"])
        self.assertEqual(len(events[0]["joints_deg"]), 7)

    def test_menu_shows_read_joints_as_text_while_cli_stays_json(self):
        robot = FakeRobot()
        robot.joints = list(demo.READY_HOME_RAD)
        events, menu_output = self.run_menu_with_robot(
            ["2", "0"], robot, return_output=True
        )

        self.assertEqual(
            [event["event"] for event in events], ["read_joints", "menu_exit"]
        )
        self.assertIn("J1", menu_output)
        self.assertIn("J7", menu_output)
        self.assertNotIn('"joints_rad"', menu_output)
        self.assertNotIn('"event":', menu_output)

        cli_events = self.run_with_robot(self.read_joints_args(), FakeRobot())
        self.assertEqual(cli_events[0]["event"], "read_joints")

    def test_menu_hand_status_shows_copyable_numeric_feedback(self):
        hand = FakeHand((0, 69, 0, 0, 0, 0))
        hand.get_finger_current = lambda: hand._message(
            dict(zip(demo.FINGER_NAMES, [2, -10, -3, -3, 4, 5]))
        )
        hand.get_finger_spd = lambda: None
        hand.get_hand_status = lambda: None
        robot = FakeRobot(hand)

        events, output = self.run_menu_with_robot(["6", "0"], robot, return_output=True)

        self.assertIn("六路位置 (0..100)：0  69  0  0  0  0", output.splitlines())
        self.assertIn("六路电流：2  -10  -3  -3  4  5", output.splitlines())
        self.assertIn("hand_side：value=unavailable  configured=right", output)
        self.assertNotIn("thumb_tip", output)
        self.assertNotIn("六路速度：", output)
        self.assertNotIn("电机状态：", output)
        self.assertNotIn('"positions"', output)
        self.assertEqual(
            [event["event"] for event in events],
            ["hand_side", "hand_status", "menu_exit"],
        )
        self.assertEqual(hand.calls, [])

    def test_menu_default_executes_after_parameters_without_confirmation(self):
        self.assertFalse(demo.make_parser().parse_args([]).confirm)
        for answers, kind in ((["3", "1", "7", "1", "1", "0"], "arm"),
                              (["7", "1", "index=35 middle=40", "0.01", "0"], "hand")):
            with self.subTest(kind=kind):
                robot, prompts = FakeRobot(), []
                events = self.run_menu_with_robot(answers, robot, confirm=False,
                                                   before_response=prompts.append)
                self.assertFalse(any("EXECUTE" in p for p in prompts))
                self.assertEqual(events[-1]["event"], "menu_exit")
                if kind == "arm":
                    self.assertEqual(sum(c[0] == "move_j" for c in robot.calls), 1)
                else:
                    self.assertEqual(len(robot.hand.calls), 2)

    def test_menu_move_j_preview_and_lowercase_confirmation_send_no_motion(self):
        robot = FakeRobot()
        events = self.run_menu_with_robot(
            ["3", "1", "7", "1", "1", "execute", "0"], robot
        )
        plans = [event for event in events if event["event"] == "arm_plan"]
        self.assertEqual(len(plans), 1)
        self.assertFalse(plans[0]["execute"])
        self.assertEqual(plans[0]["command"], "move-j")
        self.assertEqual(
            [event["event"] for event in events][-2:], ["menu_cancelled", "menu_exit"]
        )
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_move_j_accepts_named_and_numeric_joint_input(self):
        for joint_input in ("J1", "1"):
            with self.subTest(joint_input=joint_input):
                robot = FakeRobot()
                robot.joints = list(demo.READY_HOME_RAD)
                events = self.run_menu_with_robot(
                    ["3", "1", joint_input, "1", "1", "cancel", "0"], robot
                )
                plans = [event for event in events if event["event"] == "arm_plan"]
                self.assertEqual(len(plans), 1)
                self.assertAlmostEqual(
                    plans[0]["flange_target"][0],
                    demo.READY_HOME_RAD[0] + math.radians(1),
                )
                self.assertEqual(events[-1]["event"], "menu_exit")
                self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_move_j_executes_only_after_exact_confirmation(self):
        robot = FakeRobot()
        events = self.run_menu_with_robot(
            ["3", "1", "7", "1", "1", "EXECUTE", "0"], robot
        )
        plans = [event for event in events if event["event"] == "arm_plan"]
        self.assertEqual([plan["execute"] for plan in plans], [False, True])
        moves = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(len(moves), 1)
        self.assertAlmostEqual(moves[0][1][6], math.radians(1))
        self.assertEqual(events[-1]["event"], "menu_exit")

    def test_menu_home_from_formerly_blocked_j7_entry_prompts_and_executes(self):
        robot = FakeRobot()
        robot.joints = self.formerly_blocked_home_entry_rad()
        prompts = []

        events = self.run_menu_with_robot(
            ["10", "1", "", "EXECUTE", "0"],
            robot,
            before_response=prompts.append,
        )

        plans = [event for event in events if event["event"] == "arm_plan"]
        self.assertEqual([plan["execute"] for plan in plans], [False, True])
        self.assertEqual([plan["command"] for plan in plans], ["home", "home"])
        self.assertTrue(any("EXECUTE" in prompt for prompt in prompts))
        self.assertIn("arm_target_reached", [event["event"] for event in events])
        self.assertEqual(events[-1]["event"], "menu_exit")
        self.assertEqual(
            [call for call in robot.calls if call[0] == "move_j"],
            [("move_j", list(demo.READY_HOME_RAD))],
        )

    def test_menu_home_auto_and_manual_waits_are_visible_before_confirmation(self):
        for entry, expected, source, label in (
            ("", 243, "auto", "自动"),
            ("40", 40, "manual", "手动"),
        ):
            with self.subTest(timeout_input=entry):
                robot = FakeRobot()
                robot.joints = self.long_home_entry_rad()
                prompts = []
                events, output = self.run_menu_with_robot(
                    ["10", "1", entry, "cancel", "0"],
                    robot,
                    before_response=prompts.append,
                    return_output=True,
                )

                plan = next(event for event in events if event["event"] == "arm_plan")
                self.assertEqual(plan["wait_timeout_s"], expected)
                self.assertEqual(plan["timeout_source"], source)
                self.assertIn(f"等待上限：{expected} 秒（{label}）", output)
                self.assertTrue(any("等待上限秒" in prompt for prompt in prompts))
                self.assertIn("menu_cancelled", [event["event"] for event in events])
                self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_prepare_auto_and_manual_home_waits_keep_hand_preview_only(self):
        for entry, expected, source in (("", 243, "auto"), ("40", 40, "manual")):
            with self.subTest(timeout_input=entry):
                robot = FakeRobot()
                robot.joints = self.long_home_entry_rad()
                events = self.run_menu_with_robot(
                    ["11", "1", entry, "1", "0.02", "cancel", "0"], robot
                )

                home_plan = next(
                    event
                    for event in events
                    if event["event"] == "arm_plan" and event["command"] == "home"
                )
                self.assertEqual(home_plan["wait_timeout_s"], expected)
                self.assertEqual(home_plan["timeout_source"], source)
                self.assertEqual(robot.hand.calls, [])
                self.assertNotIn("move_j", [call[0] for call in robot.calls])
                self.assertNotIn("enable", [call[0] for call in robot.calls])

    def test_menu_home_from_unreferenced_within_limits_posture_cancels_without_motion(
        self,
    ):
        robot = FakeRobot()
        robot.joints = [
            math.radians(value) for value in (60, -78, 80, -45, 130, -30, 49)
        ]
        prompts = []

        events = self.run_menu_with_robot(
            ["10", "1", "", "cancel", "0"],
            robot,
            before_response=prompts.append,
        )

        plans = [event for event in events if event["event"] == "arm_plan"]
        self.assertEqual(len(plans), 1)
        self.assertFalse(plans[0]["execute"])
        self.assertTrue(any("EXECUTE" in prompt for prompt in prompts))
        self.assertIn("menu_cancelled", [event["event"] for event in events])
        self.assertEqual(events[-1]["event"], "menu_exit")
        call_names = [call[0] for call in robot.calls]
        self.assertNotIn("move_j", call_names)
        self.assertNotIn("move_p", call_names)
        self.assertNotIn("enable", call_names)

    def test_menu_valid_home_still_requires_exact_confirmation(self):
        for confirmation, should_move in (("cancel", False), ("EXECUTE", True)):
            with self.subTest(confirmation=confirmation):
                robot = FakeRobot()
                robot.joints = self.valid_home_entry_rad()
                prompts = []

                events = self.run_menu_with_robot(
                    ["10", "1", "", confirmation, "0"],
                    robot,
                    before_response=prompts.append,
                )

                self.assertTrue(any("EXECUTE" in prompt for prompt in prompts))
                moves = [call for call in robot.calls if call[0] == "move_j"]
                self.assertEqual(len(moves), int(should_move))
                self.assertEqual(events[-1]["event"], "menu_exit")

    def test_menu_hand_partial_update_preview_and_confirmed_execution(self):
        for confirmation, expected_calls in (("cancel", 0), ("EXECUTE", 2)):
            with self.subTest(confirmation=confirmation):
                robot = FakeRobot()
                events = self.run_menu_with_robot(
                    ["7", "1", "index=35 middle=40", "0.01", confirmation, "0"],
                    robot,
                )
                plans = [event for event in events if event["event"] == "hand_plan"]
                self.assertEqual(plans[0]["target"]["index_finger"], 35)
                self.assertEqual(plans[0]["target"]["middle_finger"], 40)
                self.assertEqual(len(robot.hand.calls), expected_calls)
                self.assertEqual(
                    [call["mode"] for call in robot.hand.calls],
                    ["pos", "time"] if expected_calls else [],
                )
                self.assertEqual(events[-1]["event"], "menu_exit")

    def test_menu_cancelled_field_and_invalid_input_return_to_next_choice(self):
        robot = FakeRobot()
        events = self.run_menu_with_robot(["unknown", "3", "q", "2", "0"], robot)
        self.assertIn("failed", [event["event"] for event in events])
        self.assertIn("menu_cancelled", [event["event"] for event in events])
        self.assertIn("read_joints", [event["event"] for event in events])
        self.assertEqual(events[-1]["event"], "menu_exit")
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_argparse_error_does_not_terminate_session(self):
        robot = FakeRobot()
        events = self.run_menu_with_robot(["3", "1", "8", "1", "1", "2", "0"], robot)
        self.assertIn("failed", [event["event"] for event in events])
        self.assertIn("read_joints", [event["event"] for event in events])
        self.assertEqual(events[-1]["event"], "menu_exit")
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_rejects_arm_drift_between_preview_and_execution(self):
        robot = FakeRobot()

        def drift_before_confirmation(prompt):
            if "EXECUTE" in prompt:
                robot.joints[6] = math.radians(2)

        events = self.run_menu_with_robot(
            ["3", "1", "7", "1", "1", "EXECUTE", "0"],
            robot,
            before_response=drift_before_confirmation,
        )
        self.assertTrue(
            any(
                event["event"] == "failed"
                and "changed since interactive preview" in event["error"]
                for event in events
            )
        )
        self.assertEqual(events[-1]["event"], "menu_exit")
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_move_j_uses_preview_target_after_small_feedback_jitter(self):
        robot = FakeRobot()
        robot.joints = [
            math.radians(value) for value in (55, -78, 80, -45, 130, -30, 49)
        ]

        def jitter_before_confirmation(prompt):
            if "EXECUTE" in prompt:
                for axis, jitter_deg in (
                    (1, 0.006),
                    (2, -0.009),
                    (3, 0.008),
                    (4, -0.005),
                    (5, 0.007),
                    (6, -0.006),
                ):
                    robot.joints[axis] += math.radians(jitter_deg)

        events = self.run_menu_with_robot(
            ["3", "1", "J1", "20", "1", "EXECUTE", "0"],
            robot,
            before_response=jitter_before_confirmation,
        )
        plans = [event for event in events if event["event"] == "arm_plan"]
        self.assertEqual([plan["execute"] for plan in plans], [False, True])
        moves = [call for call in robot.calls if call[0] == "move_j"]
        self.assertEqual(len(moves), 1)
        self.assertEqual(moves[0][1], plans[0]["flange_target"])
        self.assertAlmostEqual(moves[0][1][0], math.radians(75))
        self.assertEqual(events[-1]["event"], "menu_exit")

    def test_menu_move_j_rejects_meaningful_start_drift_without_sending_motion(self):
        robot = FakeRobot()
        robot.joints = [
            math.radians(value) for value in (55, -78, 80, -45, 130, -30, 49)
        ]

        def drift_before_confirmation(prompt):
            if "EXECUTE" in prompt:
                robot.joints[1] += math.radians(0.30)

        events = self.run_menu_with_robot(
            ["3", "1", "J1", "20", "1", "EXECUTE", "0"],
            robot,
            before_response=drift_before_confirmation,
        )
        self.assertTrue(
            any(
                event["event"] == "failed"
                and "changed since interactive preview" in event["error"]
                for event in events
            )
        )
        self.assertEqual(events[-1]["event"], "menu_exit")
        self.assertNotIn("move_j", [call[0] for call in robot.calls])

    def test_menu_rejects_hand_drift_between_preview_and_execution(self):
        robot = FakeRobot()

        def drift_before_confirmation(prompt):
            if "EXECUTE" in prompt:
                robot.hand.positions["index_finger"] = 50

        events = self.run_menu_with_robot(
            ["7", "1", "middle=45", "0.01", "EXECUTE", "0"],
            robot,
            before_response=drift_before_confirmation,
        )
        self.assertTrue(
            any(
                event["event"] == "failed"
                and "changed since interactive preview" in event["error"]
                for event in events
            )
        )
        self.assertEqual(events[-1]["event"], "menu_exit")
        self.assertEqual(robot.hand.calls, [])

    def test_menu_eof_and_keyboard_interrupt_exit_without_control(self):
        for error_type in (EOFError, KeyboardInterrupt):
            with self.subTest(error_type=error_type):
                robot = FakeRobot()
                events = self.run_menu_with_robot(
                    [],
                    robot,
                    before_response=lambda _prompt: (_ for _ in ()).throw(error_type()),
                )
                self.assertEqual(events[-1]["event"], "menu_exit")
                self.assertEqual(robot.calls, [])

    def test_run_directly_accepts_underscore_move_aliases(self):
        for args, expected in (
            (self.move_j_args(command="move_j"), "move-j"),
            (self.move_p_args(command="move_p"), "move-p"),
        ):
            robot = FakeRobot()
            events = self.run_with_robot(args, robot)
            with self.subTest(command=expected):
                plan = next(event for event in events if event["event"] == "arm_plan")
                self.assertEqual(args.command, expected)
                self.assertEqual(plan["command"], expected)
                self.assertNotIn("init_effector", [call[0] for call in robot.calls])

        robot = FakeRobot()
        events = self.run_with_robot(
            self.read_joints_args(command="read_joints"), robot
        )
        self.assertEqual(events[0]["event"], "read_joints")
        self.assertEqual([call[0] for call in robot.calls], ["connect", "disconnect"])


if __name__ == "__main__":
    unittest.main()
