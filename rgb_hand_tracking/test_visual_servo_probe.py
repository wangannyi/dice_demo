import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import visual_servo_probe as probe


def feedback(q=None, mode=1):
    return {'q_rad': list(q or [0.]*7), 'fk_flange_pose_m_rad': [0.]*6,
            'status': {'arm_status': 0, 'ctrl_mode': mode, 'motion_status': 0},
            'enabled': [True]*7, 'observed_monotonic_s': 12.,
            'sdk_snapshot': {'request_start_monotonic_s': 11.}}


def receiver_evidence():
    return {'errors': [], 'channel': 'can0', 'candidate_control_processes': [],
            'receiver_rows': [{'list': 'all', 'line': 'can0 000 00000000 fn data 10 raw'},
                              {'list': 'err', 'line': 'can0 000 1fffffff fn data 0 raw'}]}


class FakeBus:
    def send(self, message):
        return None


class FakeRobot:
    def __init__(self):
        self.bus = FakeBus()
        self.q = [0.]*7
        self.limits_enabled = False
        self.commands = []
        self.disconnect_error = False

    def connect(self):
        pass

    def disconnect(self):
        if self.disconnect_error:
            raise RuntimeError('disconnect failed')

    def get_config(self):
        return {'joint_limits': {'joint'+str(i): [-2., 2.] for i in range(1, 8)}}

    def fk(self, target):
        return [target[3], 0., 0., 0., 0., 0.]

    def set_joint_limits_enabled(self, value):
        self.limits_enabled = value

    def get_joint_limits_enabled(self):
        return self.limits_enabled

    def set_speed_percent(self, value):
        self.commands.append(('speed', value))
        self.bus.send(SimpleNamespace(arbitration_id=1, data=b'1'))

    def set_motion_mode(self, value):
        self.commands.append(('mode', value))
        self.bus.send(SimpleNamespace(arbitration_id=4, data=b'4'))

    def move_j(self, target):
        self.commands.append(('move', target.copy()))
        self.bus.send(SimpleNamespace(arbitration_id=2, data=b'2'))
        self.q = target.copy()


class FakeSession:
    def __init__(self, bus_class, factory, **kwargs):
        self.factory = factory
        self.robot = None

    def start(self):
        self.guard.install()
        self.robot = self.factory()
        self.robot.connect()
        return {'event': 'ready'}

    def close(self):
        if self.robot is not None:
            self.robot.disconnect()


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.original_send = FakeBus.send

    def tearDown(self):
        FakeBus.send = self.original_send

    def run_fake(self, *, execute=False, robot=None, mode=1, evidence=None, wait=None,
                 take_control=False):
        robot = robot or FakeRobot()
        base = feedback(mode=mode)
        evidence = evidence or iter([
            {'errors': [], 'receiver_rows': [], 'candidate_control_processes': []},
            receiver_evidence()])
        def reached(session, current, target, **kwargs):
            return {'samples': [feedback(target)], 'fresh_stable_samples': 10}
        with patch.object(probe, 'stopped_window', return_value=([base]*10, {})), \
                patch.object(probe, 'fresh_feedback', side_effect=lambda session, **kw: feedback(robot.q)), \
                patch.object(probe, 'wait_target', side_effect=wait or reached):
            report = probe.run_probe(FakeBus, lambda: robot, execute=execute,
                                     take_control=take_control,
                                     evidence_provider=lambda channel: next(evidence),
                                     session_factory=FakeSession)
        return robot, report

    def test_inspect_never_transmits(self):
        robot, report = self.run_fake()
        self.assertTrue(report['success'])
        self.assertEqual(report['tx']['tx_attempts'], 0)
        self.assertEqual(robot.commands, [])

    def test_execute_changes_only_fixed_joint_with_limits(self):
        robot, report = self.run_fake(execute=True)
        self.assertTrue(report['success'])
        self.assertEqual(robot.commands[0], ('speed', 1))
        target = robot.commands[1][1]
        self.assertEqual(target[:3]+target[4:], [0.]*6)
        self.assertAlmostEqual(math.degrees(target[3]), .2)
        self.assertTrue(report['sdk_limits_enabled_at_completion'])
        self.assertEqual(report['tx']['actual_tx_count'], 2)

    def test_mode_three_without_handoff_does_not_send(self):
        robot, report = self.run_fake(execute=True, mode=3)
        self.assertFalse(report['success'])
        self.assertEqual(robot.commands, [])
        self.assertEqual(report['tx']['actual_tx_count'], 0)

    def test_explicit_handoff_precedes_speed_and_joint_target(self):
        robot, report = self.run_fake(execute=True, mode=3, take_control=True)
        self.assertTrue(report['success'])
        self.assertEqual([c[0] for c in robot.commands], ['mode', 'speed', 'move'])
        self.assertFalse(report['can_handoff']['joint_target_sent'])
        self.assertEqual(report['can_handoff']['samples'][-1]['status']['ctrl_mode'], 1)

    def test_handoff_drift_refuses_joint_target(self):
        robot = FakeRobot()
        session = SimpleNamespace(robot=robot)
        shifted = feedback([math.radians(.1)]+[0.]*6)
        with patch.object(probe, 'fresh_feedback', return_value=shifted):
            with self.assertRaisesRegex(RuntimeError, 'CAN handoff changed'):
                probe.take_can_control(session, feedback(mode=3), timeout_s=1)
        self.assertEqual([c[0] for c in robot.commands], ['mode'])

    def test_foreign_receiver_blocks_before_first_send(self):
        evidence = iter([{'errors': [], 'receiver_rows': [{}], 'candidate_control_processes': []},
                         {'errors': [], 'receiver_rows': [{}, {}], 'candidate_control_processes': []}])
        robot, report = self.run_fake(execute=True, evidence=evidence)
        self.assertFalse(report['success'])
        self.assertEqual(robot.commands, [])

    def test_two_filters_of_sdk_socket_are_accepted_but_extra_filter_is_not(self):
        before = {'errors': [], 'receiver_rows': [], 'candidate_control_processes': []}
        current = receiver_evidence()
        self.assertEqual(probe.evidence_blockers(before, current), [])
        current['receiver_rows'].append(current['receiver_rows'][0])
        self.assertTrue(probe.evidence_blockers(before, current))

    def test_large_fk_translation_blocks(self):
        robot = FakeRobot()
        robot.fk = lambda q: [.01, 0., 0., 0., 0., 0.]
        _, report = self.run_fake(execute=True, robot=robot)
        self.assertFalse(report['success'])
        self.assertEqual(robot.commands, [])

    def test_outside_limit_on_any_axis_blocks(self):
        robot = FakeRobot()
        robot.get_config = lambda: {'joint_limits': {'joint'+str(i): [.1, 2.] for i in range(1, 8)}}
        _, report = self.run_fake(execute=True, robot=robot)
        self.assertFalse(report['success'])
        self.assertEqual(robot.commands, [])

    def test_timeout_hold_preserves_enabled_limits(self):
        _, report = self.run_fake(execute=True, wait=TimeoutError('not reached'))
        self.assertFalse(report['success'])
        self.assertTrue(report['hold']['requested'])
        self.assertTrue(report['hold']['sdk_limits_enabled'])
        self.assertFalse(report['hold']['hold_verified'])

    def test_disconnect_failure_keeps_denying_guard(self):
        robot = FakeRobot()
        robot.disconnect_error = True
        _, report = self.run_fake(execute=True, robot=robot)
        self.assertFalse(report['success'])
        self.assertTrue(report['tx']['guard_installed'])
        self.assertFalse(report['tx']['transmit_permitted'])
        with self.assertRaises(probe.PassiveTransmitForbidden):
            robot.bus.send(SimpleNamespace(arbitration_id=3, data=b'3'))

    def test_factory_send_is_refused(self):
        def factory():
            FakeBus().send(SimpleNamespace(arbitration_id=3, data=b'3'))
        report = probe.run_probe(FakeBus, factory, session_factory=FakeSession,
                                 evidence_provider=lambda channel: {})
        self.assertFalse(report['success'])
        self.assertEqual(report['tx']['actual_tx_count'], 0)
        self.assertEqual(report['tx']['denied_tx_attempts'], 1)

    def test_static_feedback_does_not_prove_tiny_motion(self):
        current = [0.]*7
        target = current.copy()
        target[3] = probe.DELTA_RAD
        self.assertFalse(probe.strict_target(feedback(current), current, target))
        self.assertTrue(probe.strict_target(feedback(target), current, target))
        wrong = target.copy()
        wrong[0] = math.radians(.051)
        self.assertFalse(probe.strict_target(feedback(wrong), current, target))

    def test_ten_consecutive_new_target_samples_required(self):
        target = [0.]*7
        target[3] = probe.DELTA_RAD
        rows = iter([feedback(target)]*9+[feedback()]+[feedback(target)]*10)
        with patch.object(probe, 'fresh_feedback', side_effect=lambda *a, **kw: next(rows)):
            result = probe.wait_target(None, [0.]*7, target, timeout_s=5, sleep=lambda t: None)
        self.assertEqual(len(result['samples']), 20)

    def test_waits_for_first_enable_packets_then_refreshes_joint_snapshot(self):
        clock = SimpleNamespace(now=1.)
        counts = []
        def snapshot():
            counts.append(clock.now)
            return {'q_rad': [0.]*7, 'fk_flange_pose_m_rad': [0.]*6,
                    'packet_ages_s': {k: 0. for k in probe.PACKETS},
                    'packet_timestamps_after_epoch_s': {k: clock.now for k in probe.PACKETS}}
        robot = SimpleNamespace(
            get_arm_status=lambda: SimpleNamespace(timestamp=clock.now, msg=SimpleNamespace(
                arm_status=0, ctrl_mode=1, motion_status=0)),
            get_driver_states=lambda joint_index: SimpleNamespace(
                timestamp=clock.now if clock.now >= 1.15 else None),
            get_joints_enable_status_list=lambda: [True]*7)
        guard = SimpleNamespace(report=lambda: {'tx_attempts': 0, 'actual_tx_count': 0}, allowed=False)
        session = SimpleNamespace(snapshot=snapshot, robot=robot, guard=guard)
        def sleep(t):
            clock.now += t
        row = probe.fresh_feedback(session, monotonic=lambda: clock.now,
                                   wallclock=lambda: clock.now, sleep=sleep)
        self.assertEqual(len(counts), 2)
        self.assertTrue(all(v >= 1.15 for v in row['enable_feedback_timestamps_epoch_s']))
        self.assertGreaterEqual(row['sdk_snapshot']['packet_timestamps_after_epoch_s']['joint_7'], 1.15)

    def test_missing_enable_packet_has_bounded_wait(self):
        clock = SimpleNamespace(now=1.)
        robot = SimpleNamespace(
            get_arm_status=lambda: SimpleNamespace(timestamp=clock.now, msg=SimpleNamespace()),
            get_driver_states=lambda joint_index: SimpleNamespace(timestamp=None))
        session = SimpleNamespace(robot=robot, snapshot=lambda: {
            'q_rad': [0.]*7, 'fk_flange_pose_m_rad': [0.]*6,
            'packet_ages_s': {k: 0. for k in probe.PACKETS}})
        def sleep(t):
            clock.now += t
        with self.assertRaisesRegex(RuntimeError, 'No fresh joint enable'):
            probe.fresh_feedback(session, monotonic=lambda: clock.now,
                                 wallclock=lambda: clock.now, sleep=sleep)
        self.assertLess(clock.now, 3.1)


if __name__ == '__main__':
    unittest.main()
