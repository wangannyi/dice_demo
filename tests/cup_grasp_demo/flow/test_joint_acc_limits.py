"""Acceleration writes: protocol encoding, no-motion guard and partial failures."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from cup_grasp_demo.flow import joint_acc_limits as acc
from cup_grasp_demo.flow.joint_profile import options


class AccelerationTest(unittest.TestCase):
    def test_seven_targets_are_optional_and_bounded(self):
        self.assertIsNone(options({})['controller_acceleration_rad_s2'])
        self.assertEqual(options({'controller_acceleration_rad_s2':[5]*7})['controller_acceleration_rad_s2'], [5]*7)
        for bad in ([5]*6, [5.01]*7, [True]*7, [float('nan')]*7, [.015]*7, [0]*7):
            with self.assertRaises(ValueError):
                options({'controller_acceleration_rad_s2':bad})

    def test_real_sdk_encoding_is_500_not_50000_without_can(self):
        from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config
        robot = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.NERO, firmeware_version=NeroFW.V120,
            interface='socketcan', channel='can0'))
        messages = []
        original = robot._MSG_JointConfig
        with patch.object(robot, '_send_msg', side_effect=messages.append), \
             patch.object(robot, '_check_set_by_readback', side_effect=lambda **k: (k['request'](), True)[1]):
            result = acc.set_acceleration_v120(robot, 7, 5.)
        self.assertTrue(result['sdk_return'])
        self.assertEqual(len(messages), 1)
        msg = messages[0]
        self.assertEqual(msg.max_joint_acc, 500)
        self.assertEqual(msg.joint_index, 7)
        self.assertEqual(msg.set_motor_current_pos_as_zero, 0)
        self.assertEqual(msg.clear_joint_err, 0)
        self.assertEqual(robot._MSG_JointConfig, original)
        self.assertEqual(acc.acceleration_frame(7,5).hex(), '0700ae01f4000000')

    def test_guard_allows_only_exact_selected_axis_write(self):
        class Bus:
            def send(self, message):
                return None
        guard = acc.AccelerationGuard(Bus)
        try:
            guard.install(); guard.permit()
            frame = acc.acceleration_frame(1,5)
            guard.expected_write = frame
            Bus().send(SimpleNamespace(arbitration_id=0x475,data=frame,is_extended_id=False))
            for ident, data in ((0x475,acc.acceleration_frame(2,5)),(0x155,bytes(8)),
                                (0x151,bytes(8)),(0x475,bytes([1,0xAE,0xAE,1,244,0,0,0]))):
                with self.assertRaises(RuntimeError):
                    Bus().send(SimpleNamespace(arbitration_id=ident,data=data,is_extended_id=False))
            self.assertEqual(guard.report()['actual_tx_count'],1)
        finally:
            guard.restore()

    def test_readback_mismatch_stops_before_next_joint(self):
        robot = Mock()
        robot.get_joint_acc_limits.return_value = SimpleNamespace(msg=SimpleNamespace(max_joint_acc=.5))
        report = {'joints':[]}
        guard = SimpleNamespace(expected_write=None)
        saved=[]
        def save(): saved.append(len(report['joints']))
        with patch.object(acc,'set_acceleration_v120',return_value={'sdk_return':True}) as setter:
            with self.assertRaisesRegex(RuntimeError,'读回'):
                acc.apply_targets(robot,guard,[5]*7,[2]*7,report,save,lambda:None)
            setter.assert_called_once_with(robot,1,5)
        self.assertIsNone(guard.expected_write)
        self.assertEqual(len(report['joints']),1)
        self.assertFalse(report['joints'][0]['verified'])
        self.assertGreaterEqual(len(saved),2)

    def test_success_reads_each_joint_even_when_already_set(self):
        robot = Mock()
        robot.get_joint_acc_limits.return_value = SimpleNamespace(msg=SimpleNamespace(max_joint_acc=5))
        report={'joints':[]}
        with patch.object(acc,'set_acceleration_v120') as setter:
            acc.apply_targets(robot,SimpleNamespace(expected_write=None),[5]*7,[5]*7,
                              report,lambda:None,lambda:None)
            setter.assert_not_called()
        self.assertEqual(len(report['joints']),7)
        self.assertTrue(all(r['verified'] and not r['write_attempted'] for r in report['joints']))

    def test_missing_backup_prevents_settings(self):
        robot=Mock()
        robot.get_joint_acc_limits.return_value=None
        with self.assertRaisesRegex(RuntimeError,'原值备份'):
            acc.read_accelerations(robot)
        robot.set_joint_acc_limits.assert_not_called()


if __name__ == '__main__':
    unittest.main()
