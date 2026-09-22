"""Actual SDK V1.20 encoding into a memory bus; physical CAN send is forbidden."""

import math
import struct
import unittest
from types import SimpleNamespace as NS
from cup_grasp_demo.calibration_debug.joint_delivery import take_js_control, fresh_js_hold
from unittest.mock import patch

from cup_grasp_demo.calibration_debug.planar_shake_execution import JSGuard


class SDKTest(unittest.TestCase):
    def test_real_js_seven_axis_packet_encoding_and_guard(self):
        from pyAgxArm import AgxArmFactory, create_agx_arm_config
        from can.interfaces.socketcan import SocketcanBus

        frames = []

        class MemoryBus:
            def send(self, frame):
                frames.append(frame)

            def get_channel(self):
                return "memory"

        with patch.object(SocketcanBus, "send", side_effect=AssertionError("Real CAN forbidden")) as physical:
            robot = AgxArmFactory.create_arm(create_agx_arm_config(
                robot="nero", firmeware_version="v120", channel="can0"))
            robot._ctx.get_comm = lambda: MemoryBus()
            guard = JSGuard(MemoryBus)
            guard.install()
            guard.permit()
            guard.motion_allowed = True
            try:
                robot.set_speed_percent(97)
                q = [.1, -.2, .3, .4, -.5, .1, -.1]
                robot.move_js(q)
                modes = [f for f in frames if f.arbitration_id == 0x151]
                self.assertEqual(list(modes[-1].data[:4]), [1, 1, 97, 0xAD])
                joints = [f for f in frames if f.arbitration_id in (0x155, 0x156, 0x157, 0x170)]
                self.assertEqual([f.arbitration_id for f in joints], [0x155, 0x156, 0x157, 0x170])
                values = [v for f in joints for v in struct.unpack(">ii", bytes(f.data))][:7]
                for raw, radians in zip(values, q):
                    self.assertAlmostEqual(raw, math.degrees(radians)*1000, delta=1)
                self.assertEqual(len([f for f in frames if f.arbitration_id in (0x475, 0x1B1)]), 0)
                session = NS(robot=robot)
                core = NS(take_can_control=lambda s, *a, **k: s.robot.set_motion_mode('j'),
                          fresh_feedback=lambda _: dict(q_rad=q, enabled=[True]*7,
                                                        status=dict(arm_status=0, ctrl_mode=1)),
                          validate_target=lambda q, limits: q)
                take_js_control(core, session, {}, timeout_s=1)
                fresh_js_hold(core, session, [])
                robot.set_speed_percent(5)
                self.assertTrue(all(f.data[3] == 0xAD for f in frames
                                    if f.arbitration_id == 0x151 and f.data[1] != 255))
                with self.assertRaisesRegex(RuntimeError, 'JS guard rejected'):
                    robot.set_motion_mode('j')
                physical.assert_not_called()
            finally:
                guard.restore()


if __name__ == "__main__":
    unittest.main()
