from contextlib import contextmanager
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cartesian_microprobe as probe
from passive_pose_bridge import PassiveTransmitForbidden


def fk(q):
    return [.25*q[0]+.1*q[6], .25*q[1], .25*q[2],
            q[3]+.2*q[0], q[4]-.1*q[1], q[5]+.5*q[0]]


def feedback(q=None, mode=1, status=0, stationary=True, enabled=True):
    q = list([0.]*7 if q is None else q)
    return {'q_rad': q, 'fk_flange_pose_m_rad': fk(q),
            'status': {'arm_status': status, 'ctrl_mode': mode,
                       'motion_status': 0 if stationary else 1},
            'enabled': [enabled]*7, 'observed_monotonic_s': 10.,
            'status_timestamp_epoch_s': 10.,
            'sdk_snapshot': {'request_start_monotonic_s': 9.}}


def evidence():
    return {'errors': [], 'channel': 'can0', 'candidate_control_processes': [],
            'receiver_rows': [{'list': 'all', 'line': 'can0 000 00000000 fn data 10 raw'},
                              {'list': 'err', 'line': 'can0 000 1fffffff fn data 0 raw'}]}


class Clock:
    def __init__(self):
        self.t = 10.

    def now(self):
        return self.t

    def sleep(self, amount):
        self.t += amount


class FakeBus:
    def send(self, message):
        return None


class FakeRobot:
    def __init__(self, *, mode=1, apply_move=True):
        self.bus, self.q, self.mode = FakeBus(), [0.]*7, mode
        self.commands, self.apply_move, self.limits = [], apply_move, False
        self.disconnect_tx = False
        self.fk = fk

    def connect(self):
        pass

    def disconnect(self):
        if self.disconnect_tx:
            self.send(99)

    def send(self, arbitration_id):
        self.bus.send(SimpleNamespace(arbitration_id=arbitration_id, data=b'1'))

    def get_config(self):
        return {'joint_limits': {'joint'+str(i): [-2., 2.] for i in range(1, 8)}}

    def set_joint_limits_enabled(self, value):
        self.commands.append(('limits', value))
        self.limits = value

    def get_joint_limits_enabled(self):
        return self.limits

    def set_speed_percent(self, value):
        self.commands.append(('speed', value))
        self.send(2)

    def set_motion_mode(self, value):
        self.commands.append(('mode', value))
        self.send(3)
        self.mode = 1

    def move_j(self, target):
        self.commands.append(('move', list(target)))
        self.send(4)
        if callable(self.apply_move):
            self.q = self.apply_move(list(target))
        elif self.apply_move:
            self.q = list(target)


class FakeSession:
    def __init__(self, bus_class, factory, **kwargs):
        self.factory, self.robot = factory, None

    def start(self):
        self.guard.install()
        self.robot = self.factory()
        self.robot.connect()
        return {'event': 'ready'}

    def close(self):
        if self.robot:
            self.robot.disconnect()


class CartesianMicroprobeTests(unittest.TestCase):
    def setUp(self):
        self.send_before = FakeBus.send

    def tearDown(self):
        FakeBus.send = self.send_before

    def run_fake(self, *, execute=False, robot=None, factory=None, take_control=False,
                 host=None, fresh=None, baseline=None, **kwargs):
        robot, clock = robot or FakeRobot(), Clock()
        baseline = baseline or feedback(mode=robot.mode)
        serial = [0]

        def read(session, previous=None, **unused):
            serial[0] += 1
            result = feedback(robot.q, mode=robot.mode)
            result['observed_monotonic_s'] = clock.now()
            result['status_timestamp_epoch_s'] = 10.+serial[0]*.01
            if fresh:
                result = fresh(result, serial[0], previous)
            return result

        host = host or iter([{'errors': [], 'receiver_rows': [],
                             'candidate_control_processes': []}, evidence()])
        with patch.object(probe, 'stopped_window', return_value=([baseline]*10, {'fresh_samples': 10})), \
                patch.object(probe, 'fresh_feedback', side_effect=read), \
                patch('visual_servo_probe.fresh_feedback', side_effect=read):
            result = probe.run_probe(FakeBus, factory or (lambda: robot), axis='x', distance_m=.001,
                                     execute=execute, take_control=take_control, timeout_s=1,
                                     evidence_provider=lambda channel: next(host),
                                     monotonic=clock.now, wallclock=clock.now, sleep=clock.sleep,
                                     session_factory=FakeSession, **kwargs)
        return robot, result

    def test_inspect_zero_tx_and_keeps_hardware_settings(self):
        robot, result = self.run_fake()
        self.assertTrue(result['success'])
        self.assertTrue(result['motion_target_valid'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])
        self.assertIs(FakeBus.send, self.send_before)

    def test_factory_transmission_is_denied_before_connect(self):
        def factory():
            FakeBus().send(SimpleNamespace(arbitration_id=1, data=b'1'))
            return FakeRobot()

        _, result = self.run_fake(execute=True, factory=factory)
        self.assertFalse(result['success'])
        self.assertEqual(result['tx']['denied_tx_attempts'], 1)
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertFalse(result['joint_motion_attempted'])

    def test_foreign_receiver_blocks_before_first_tx(self):
        foreign = evidence()
        foreign['receiver_rows'].append({'list': 'all', 'line': 'can0 000 00000000 foreign'})
        robot, result = self.run_fake(execute=True, host=iter([
            {'errors': [], 'receiver_rows': [], 'candidate_control_processes': []}, foreign]))
        self.assertFalse(result['success'])
        self.assertEqual(result['event'], 'cartesian_microprobe_blocked')
        self.assertEqual(robot.commands, [])
        self.assertEqual(result['tx']['actual_tx_count'], 0)

    def test_invalid_plan_cannot_enable_execution(self):
        robot = FakeRobot()
        robot.fk = lambda q: [.25*q[0], 0., 0., 0., 0., 0.]
        _, result = self.run_fake(execute=True, robot=robot)
        self.assertFalse(result['proposal']['valid'])
        self.assertFalse(result['motion_target_valid'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])

    def test_execute_has_ten_strict_all_axis_fk_samples(self):
        robot, result = self.run_fake(execute=True)
        self.assertTrue(result['success'], result)
        verification = result['target_verification']
        self.assertEqual(verification['fresh_stable_samples'], 10)
        self.assertTrue(all(s['verification']['reached'] for s in verification['samples']))
        self.assertEqual(result['tx']['actual_tx_count'], 2)
        self.assertEqual([c[0] for c in robot.commands], ['limits', 'speed', 'move'])
        self.assertTrue(result['sdk_fk_is_model_not_visual_displacement'])
        self.assertFalse(result['finger_command_sent'])

    def test_one_off_target_joint_cannot_pass_then_fresh_hold_is_used(self):
        def off_axis(target):
            target[6] += math.radians(.025)
            return target

        _, result = self.run_fake(execute=True, robot=FakeRobot(apply_move=off_axis))
        self.assertFalse(result['success'])
        self.assertIn('TimeoutError', result['error'])
        self.assertTrue(result['hold']['requested'])
        self.assertFalse(result['hold']['hold_verified'])

    def test_zero_actual_change_never_reaches_and_timeout_holds_fresh_q(self):
        robot, result = self.run_fake(execute=True, robot=FakeRobot(apply_move=False))
        self.assertFalse(result['success'])
        self.assertTrue(result['hold']['requested'])
        self.assertEqual(result['hold']['target_rad'], [0.]*7)
        self.assertEqual(robot.commands[-1], ('move', [0.]*7))
        self.assertEqual(result['tx']['actual_tx_count'], 3)

    def test_baseline_drift_prevents_proposed_move(self):
        def drifting(row, serial, previous):
            if serial == 1:
                row['q_rad'][2] += math.radians(.06)
            return row

        robot, result = self.run_fake(execute=True, fresh=drifting)
        self.assertFalse(result['joint_motion_attempted'])
        self.assertIn('Baseline changed', result['error'])
        self.assertTrue(result['hold']['requested'])
        self.assertEqual([c[0] for c in robot.commands], ['limits', 'speed', 'limits', 'move'])

    def test_web_mode_requires_explicit_handoff(self):
        robot, blocked = self.run_fake(execute=True, robot=FakeRobot(mode=3))
        self.assertFalse(blocked['success'])
        self.assertEqual(robot.commands, [])
        _, reached = self.run_fake(execute=True, robot=FakeRobot(mode=3), take_control=True)
        self.assertTrue(reached['success'], reached)
        self.assertTrue(reached['can_handoff']['mode_command_sent'])

    def test_guard_denies_disconnect_transmission_and_stays_installed_on_failure(self):
        robot = FakeRobot()
        robot.disconnect_tx = True
        _, result = self.run_fake(execute=True, robot=robot)
        self.assertFalse(result['success'])
        self.assertFalse(result['sdk_disconnected'])
        self.assertEqual(result['tx']['denied_tx_attempts'], 1)
        self.assertTrue(result['tx']['guard_installed'])
        self.assertFalse(result['tx']['transmit_permitted'])
        with self.assertRaises(PassiveTransmitForbidden):
            FakeBus().send(SimpleNamespace(arbitration_id=1, data=b''))

    def test_target_checks_reject_wrong_sign_z_rotation_and_stationarity(self):
        before = feedback()
        plan = {'axis': 'x', 'requested_distance_m': .001, 'target_position_m': [.001, 0., 0.]}
        target = [.004, 0., 0., -.0008, 0., -.002, 0.]
        ideal = feedback(target)
        self.assertTrue(probe.target_checks(ideal, before, target, plan)['reached'])
        for key, value in [('x', -.001), ('z', .00011), ('roll', math.radians(.06))]:
            candidate = feedback(target)
            index = {'x': 0, 'z': 2, 'roll': 3}[key]
            candidate['fk_flange_pose_m_rad'][index] = value
            self.assertFalse(probe.target_checks(candidate, before, target, plan)['reached'])
        moving = feedback(target, stationary=False)
        self.assertFalse(probe.target_checks(moving, before, target, plan)['reached'])

    def test_wait_requires_consecutive_ready_samples(self):
        def intermittent(row, serial, previous):
            if serial == 6:
                row['status']['motion_status'] = 1
            return row

        _, result = self.run_fake(execute=True, fresh=intermittent)
        self.assertTrue(result['success'], result)
        self.assertEqual(len(result['target_verification']['samples']), 15)

    def test_cli_defaults_inspect_with_required_axis_distance_and_atomic_json(self):
        @contextmanager
        def lock(path):
            self.assertEqual(path, Path('/tmp/nero_can0_control.lock'))
            yield

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/'result.json'
            with patch.object(probe, 'control_lock', side_effect=lock), \
                    patch.object(probe, 'load_sdk_runtime', return_value=(FakeBus, lambda: None)), \
                    patch.object(probe, 'run_probe', return_value={'success': True}) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                code = probe.main(['--axis', 'y', '--distance-mm', '-1', '--output', str(destination)])
            self.assertEqual(code, 0)
            self.assertFalse(run.call_args.kwargs['execute'])
            self.assertEqual(run.call_args.kwargs['distance_m'], -.001)
            report = json.loads(destination.read_text())
            self.assertEqual(len(report['tool_source_sha256']), 64)
            self.assertEqual(len(report['planner_source_sha256']), 64)
            self.assertEqual(list(Path(temporary).glob('*.tmp')), [])

    def test_invalid_cli_values_do_not_load_sdk(self):
        for argv in [[], ['--axis', 'x', '--distance-mm', '0'],
                     ['--axis', 'x', '--distance-mm', '1.1'],
                     ['--axis', 'x', '--distance-mm', '1', '--timeout-s', '.5']]:
            with patch.object(probe, 'load_sdk_runtime') as runtime, \
                    patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                probe.main(argv)
            runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
