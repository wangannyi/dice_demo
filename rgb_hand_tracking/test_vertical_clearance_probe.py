from contextlib import contextmanager
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import vertical_clearance_probe as probe
from passive_pose_bridge import PassiveTransmitForbidden
from test_cartesian_microprobe import Clock, FakeBus, FakeRobot, FakeSession, evidence, feedback


LIMITS = [[-2., 2.]]*7


class VerticalClearanceProbeTests(unittest.TestCase):
    def setUp(self):
        self.send_before = FakeBus.send

    def tearDown(self):
        FakeBus.send = self.send_before

    def run_fake(self, *, execute=False, robot=None, factory=None, host=None,
                 free_space=True, take_control=False, fresh=None, **options):
        robot, clock, serial = robot or FakeRobot(), Clock(), [0]
        baseline = feedback(robot.q, mode=robot.mode)

        def read(session, previous=None, **unused):
            serial[0] += 1
            result = feedback(robot.q, mode=robot.mode)
            result['observed_monotonic_s'] = clock.now()
            result['status_timestamp_epoch_s'] = 10.+serial[0]*.01
            return fresh(result, serial[0]) if fresh else result

        host = host or iter([{'errors': [], 'receiver_rows': [],
                             'candidate_control_processes': []}, evidence()])
        with patch.object(probe, 'stopped_window', return_value=([baseline]*10, {'fresh_samples': 10})), \
                patch.object(probe, 'fresh_feedback', side_effect=read), \
                patch('visual_servo_probe.fresh_feedback', side_effect=read):
            result = probe.run_probe(FakeBus, factory or (lambda: robot), execute=execute,
                take_control=take_control, noncontact_free_space_confirmed=free_space,
                timeout_s=1, evidence_provider=lambda channel: next(host),
                monotonic=clock.now, wallclock=clock.now, sleep=clock.sleep,
                session_factory=FakeSession, **options)
        json.dumps(result, allow_nan=False)
        return robot, result

    def test_positive_pure_plan_checks_five_model_points_without_physical_claim(self):
        robot = FakeRobot()
        plan = probe.plan_vertical_step(robot.fk, [0.]*7, LIMITS)
        self.assertTrue(plan['valid'], plan)
        self.assertAlmostEqual(plan['predicted_displacement_m'][2], .001, delta=.00002)
        self.assertLessEqual(max(abs(v) for v in plan['delta_q_rad']), math.radians(.35))
        self.assertEqual([row['fraction'] for row in plan['sampled_model_path']], [0., .25, .5, .75, 1.])
        self.assertIsNone(plan['current_fingertip_extent_m'])
        self.assertFalse(plan['whole_hand_clearance_validated'])
        self.assertFalse(plan['physical_registration_valid'])
        self.assertFalse(plan['actual_trajectory_validated'])

    def test_bad_inputs_limits_and_rank_are_rejected_without_target(self):
        robot = FakeRobot()
        for lift in [0., -.001, .0005, .0011, True, float('nan')]:
            plan = probe.plan_vertical_step(robot.fk, [0.]*7, LIMITS, lift_m=lift)
            self.assertFalse(plan['valid'])
            self.assertIsNone(plan['target_q_rad'])
        for q, limits in [([1.999]+[0.]*6, LIMITS), ([0.]*7, [[2., -2.]]*7)]:
            self.assertFalse(probe.plan_vertical_step(robot.fk, q, limits)['valid'])
        rank = probe.plan_vertical_step(lambda q: [0., 0., .25*q[2], 0., 0., 0.], [0.]*7, LIMITS)
        self.assertFalse(rank['valid'])

    def test_midpath_dip_xy_or_rotation_is_rejected_despite_good_endpoint(self):
        candidate = [0., 0., .004, 0., 0., 0., 0.]
        model = {'valid': True, 'candidate_q_rad': candidate}
        for kind in ['dip', 'xy', 'rotation']:
            def curved(q):
                middle = math.sin(math.pi*q[2]/.004)
                return [.0002*middle if kind == 'xy' else 0., 0.,
                        .25*q[2]-(.0011*middle if kind == 'dip' else 0.),
                        .0018*middle if kind == 'rotation' else 0., 0., 0.]
            with patch.object(probe, 'plan_taught_pregrasp', return_value=model):
                result = probe.plan_vertical_step(curved, [0.]*7, LIMITS)
            self.assertFalse(result['valid'], kind)
            self.assertIsNone(result['target_q_rad'])
            self.assertIn('Sampled vertical path', result['error'])

    def test_inspection_zero_tx_and_no_hardware_setting_changes(self):
        robot, result = self.run_fake()
        self.assertTrue(result['success'])
        self.assertTrue(result['motion_target_valid'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])

    def test_factory_send_denied_before_connect(self):
        def factory():
            FakeBus().send(SimpleNamespace(arbitration_id=1, data=b'1'))
            return FakeRobot()

        _, result = self.run_fake(execute=True, factory=factory)
        self.assertFalse(result['success'])
        self.assertEqual(result['tx']['denied_tx_attempts'], 1)
        self.assertEqual(result['tx']['actual_tx_count'], 0)

    def test_free_space_assertion_required_for_execution_not_inspection(self):
        _, inspect = self.run_fake(free_space=False)
        self.assertTrue(inspect['success'])
        self.assertFalse(inspect['motion_target_valid'])
        robot, execute = self.run_fake(execute=True, free_space=False)
        self.assertFalse(execute['success'])
        self.assertEqual(execute['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])
        self.assertIsNone(execute['current_fingertip_extent_m'])

    def test_execution_ten_fresh_all_axis_model_samples_and_no_fingers(self):
        robot, result = self.run_fake(execute=True)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['target_verification']['fresh_stable_samples'], 10)
        self.assertEqual(result['tx']['actual_tx_count'], 2)
        self.assertEqual([v[0] for v in robot.commands], ['limits', 'speed', 'move'])
        self.assertTrue(all(row['verification']['reached'] for row in result['target_verification']['samples']))
        self.assertFalse(result['finger_command_sent'])
        self.assertFalse(result['physical_contact_verified'])
        self.assertFalse(result['whole_hand_clearance_validated'])

    def test_off_target_and_stalled_motion_timeout_then_hold_fresh_q(self):
        def wrong_joint(q):
            q[6] += math.radians(.025)
            return q

        for applied in [False, wrong_joint]:
            robot, result = self.run_fake(execute=True, robot=FakeRobot(apply_move=applied))
            self.assertFalse(result['success'])
            self.assertIn('TimeoutError', result['error'])
            self.assertTrue(result['hold']['requested'])
            self.assertFalse(result['hold']['hold_verified'])
            self.assertEqual(result['tx']['actual_tx_count'], 3)
            if applied is False:
                self.assertEqual(robot.commands[-1], ('move', [0.]*7))

    def test_actual_model_checks_reject_wrong_z_xy_rotation_and_moving_state(self):
        robot, inspected = self.run_fake()
        plan, before = inspected['proposal'], feedback()
        settled = feedback(plan['target_q_rad'])
        self.assertTrue(probe.target_checks(settled, before, plan)['reached'])
        for kind in ['z', 'xy', 'rotation', 'stationary']:
            row = feedback(plan['target_q_rad'])
            if kind == 'z':
                row['fk_flange_pose_m_rad'][2] = .00049
            elif kind == 'xy':
                row['fk_flange_pose_m_rad'][0] = .00011
            elif kind == 'rotation':
                row['fk_flange_pose_m_rad'][3] += math.radians(.06)
            else:
                row['status']['motion_status'] = 1
            self.assertFalse(probe.target_checks(row, before, plan)['reached'], kind)

    def test_pre_tx_drift_aborts_then_hold(self):
        def drift(row, serial):
            if serial == 1:
                row['q_rad'][0] += math.radians(.06)
            return row

        _, result = self.run_fake(execute=True, fresh=drift)
        self.assertFalse(result['joint_motion_attempted'])
        self.assertIn('Baseline changed', result['error'])
        self.assertTrue(result['hold']['requested'])

    def test_foreign_receivers_and_web_without_handoff_block(self):
        foreign = evidence()
        foreign['receiver_rows'].append({'list': 'all', 'line': 'can0 000 00000000 foreign'})
        _, result = self.run_fake(execute=True, host=iter([
            {'errors': [], 'receiver_rows': [], 'candidate_control_processes': []}, foreign]))
        self.assertFalse(result['success'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        _, web = self.run_fake(execute=True, robot=FakeRobot(mode=3))
        self.assertFalse(web['success'])
        self.assertEqual(web['tx']['actual_tx_count'], 0)
        _, handoff = self.run_fake(execute=True, robot=FakeRobot(mode=3), take_control=True)
        self.assertTrue(handoff['success'], handoff)
        self.assertTrue(handoff['can_handoff']['mode_command_sent'])

    def test_cleanup_denies_tx_and_keeps_hook_if_disconnect_fails(self):
        robot = FakeRobot()
        robot.disconnect_tx = True
        _, result = self.run_fake(execute=True, robot=robot)
        self.assertFalse(result['success'])
        self.assertFalse(result['sdk_disconnected'])
        self.assertFalse(result['tx']['transmit_permitted'])
        self.assertTrue(result['tx']['guard_installed'])
        with self.assertRaises(PassiveTransmitForbidden):
            FakeBus().send(SimpleNamespace(arbitration_id=1, data=b''))

    def test_cli_explicit_lift_default_inspection_common_lock_atomic_json(self):
        @contextmanager
        def lock(path):
            self.assertEqual(path, Path('/tmp/nero_can0_control.lock'))
            yield

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result.json'
            with patch.object(probe, 'control_lock', side_effect=lock), \
                    patch.object(probe, 'load_sdk_runtime', return_value=(FakeBus, lambda: None)), \
                    patch.object(probe, 'run_probe', return_value={'success': True}) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(probe.main(['--lift-mm', '1', '--output', str(output)]), 0)
            self.assertFalse(run.call_args.kwargs['execute'])
            self.assertEqual(run.call_args.kwargs['lift_m'], .001)
            self.assertFalse(run.call_args.kwargs['noncontact_free_space_confirmed'])
            self.assertEqual(len(json.loads(output.read_text())['tool_source_sha256']), 64)
            self.assertFalse(list(Path(directory).glob('*.tmp')))

    def test_invalid_cli_lift_never_loads_sdk(self):
        for argv in [[], ['--lift-mm', '-1'], ['--lift-mm', '0'], ['--lift-mm', '.5'],
                     ['--lift-mm', '1.1'], ['--lift-mm', 'nan']]:
            with patch.object(probe, 'load_sdk_runtime') as runtime, \
                    patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                probe.main(argv)
            runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
