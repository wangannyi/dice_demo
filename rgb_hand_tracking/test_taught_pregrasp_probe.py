from contextlib import contextmanager
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import taught_pregrasp_probe as probe
from passive_pose_bridge import PassiveTransmitForbidden
from test_cartesian_microprobe import Clock, FakeBus, FakeRobot, FakeSession, evidence, feedback


CURRENT = [-.03, 0., .12, 0., 0., 0., 0.]
LIMITS = [[-2., 2.]]*7


class TaughtPregraspProbeTests(unittest.TestCase):
    def setUp(self):
        self.original_send = FakeBus.send

    def tearDown(self):
        FakeBus.send = self.original_send

    def run_fake(self, *, execute=False, robot=None, factory=None, take_control=False,
                 fresh=None, host=None, candidate=None, **probe_options):
        robot = robot or FakeRobot()
        if robot.q == [0.]*7:
            robot.q = CURRENT.copy()
        clock, serial = Clock(), [0]
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
            if candidate is not None:
                with patch.object(probe, 'plan_taught_pregrasp', return_value=candidate):
                    result = self.call_run(robot, factory, host, clock, execute, take_control, **probe_options)
            else:
                result = self.call_run(robot, factory, host, clock, execute, take_control, **probe_options)
        return robot, result

    def call_run(self, robot, factory, host, clock, execute, take_control, **probe_options):
        return probe.run_probe(FakeBus, factory or (lambda: robot),
            reference={'taught_q_rad': [0.]*7, 'physical_registration_valid': False},
            execute=execute, take_control=take_control, operator_same_scene_confirmed=True,
            timeout_s=1, evidence_provider=lambda channel: next(host), monotonic=clock.now,
            wallclock=clock.now, sleep=clock.sleep, session_factory=FakeSession, **probe_options)

    def test_inspect_zero_tx_and_keeps_inactive_reference(self):
        robot, result = self.run_fake()
        self.assertTrue(result['success'], result)
        self.assertTrue(result['motion_target_valid'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])
        self.assertFalse(result['physical_registration_valid'])
        self.assertFalse(result['model_candidate']['execution_enabled'])

    def test_factory_send_is_denied_before_connect(self):
        def factory():
            FakeBus().send(SimpleNamespace(arbitration_id=1, data=b'1'))
            return FakeRobot()

        _, result = self.run_fake(execute=True, factory=factory)
        self.assertFalse(result['success'])
        self.assertEqual(result['tx']['denied_tx_attempts'], 1)
        self.assertEqual(result['tx']['actual_tx_count'], 0)

    def test_first_sent_step_is_bounded_and_not_whole_candidate(self):
        robot, result = self.run_fake(execute=True)
        self.assertTrue(result['success'], result)
        segment = result['proposal']
        self.assertGreater(segment['remaining_interpolated_segments'], 1)
        self.assertLessEqual(segment['requested_max_joint_delta_rad'], math.radians(.2))
        self.assertFalse(segment['next_equals_candidate'])
        self.assertEqual(robot.commands[-1], ('move', segment['next_q_rad']))
        self.assertNotEqual(robot.commands[-1][1], segment['candidate_q_rad'])
        self.assertEqual(result['target_verification']['fresh_stable_samples'], 10)
        self.assertFalse(result['trajectory_collision_validated'])
        self.assertFalse(result['finger_command_sent'])

    def test_margin_translation_rotation_and_lift_gates(self):
        candidate = [0., 0., .08, 0., 0., 0., 0.]
        taught = [0.]*6
        for kind in ['margin', 'translation', 'rotation', 'height']:
            current, goal = CURRENT.copy(), candidate.copy()
            def fk(q):
                if kind == 'translation':
                    return [10*q[0], 0., .03, 0., 0., 0.]
                if kind == 'rotation':
                    return [0., 0., .03, 3*q[0], 0., 0.]
                if kind == 'height':
                    return [0., 0., .01997, 0., 0., 0.]
                return [0., 0., .03, 0., 0., 0.]
            if kind == 'margin':
                goal[0] = 2.-.0005
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                probe.first_segment(fk, current, goal, taught, LIMITS)

    def test_gate_failure_and_invalid_model_never_send_target(self):
        bad_models = [
            {'valid': False, 'error': {'code': 'unreachable'}},
            {'valid': True, 'candidate_q_rad': [1.9999]+[0.]*6, 'taught_fk_flange_m_rad': [0.]*6}]
        for candidate in bad_models:
            robot, result = self.run_fake(execute=True, candidate=candidate)
            self.assertFalse(result['success'])
            self.assertEqual(result['tx']['actual_tx_count'], 0)
            self.assertEqual(robot.commands, [])

    def test_off_target_cannot_advance_and_holds(self):
        def off_target(target):
            target[6] += math.radians(.025)
            return target

        _, result = self.run_fake(execute=True, robot=FakeRobot(apply_move=off_target))
        self.assertFalse(result['success'])
        self.assertIn('TimeoutError', result['error'])
        self.assertTrue(result['hold']['requested'])
        self.assertFalse(result['hold']['hold_verified'])

    def test_stall_fails_and_holds_only_fresh_current(self):
        robot, result = self.run_fake(execute=True, robot=FakeRobot(apply_move=False))
        self.assertFalse(result['success'])
        self.assertEqual(result['hold']['target_rad'], CURRENT)
        self.assertEqual(robot.commands[-1], ('move', CURRENT))
        self.assertEqual(result['tx']['actual_tx_count'], 3)

    def test_near_final_reports_already_pregrasp_without_any_tx(self):
        robot = FakeRobot()
        robot.q = [0., 0., .08, 0., 0., 0., 0.]
        _, result = self.run_fake(execute=True, robot=robot)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['event'], 'already_pregrasp')
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])

    def test_foreign_receiver_and_web_without_handoff_block(self):
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

    def test_pre_tx_drift_aborts_then_holds(self):
        def drift(row, serial):
            if serial == 1:
                row['q_rad'][0] += math.radians(.06)
            return row

        _, result = self.run_fake(execute=True, fresh=drift)
        self.assertFalse(result['joint_motion_attempted'])
        self.assertIn('Baseline changed', result['error'])
        self.assertTrue(result['hold']['requested'])

    def test_cleanup_deny_stays_installed_when_disconnect_sends(self):
        robot = FakeRobot()
        robot.disconnect_tx = True
        _, result = self.run_fake(execute=True, robot=robot)
        self.assertFalse(result['success'])
        self.assertFalse(result['sdk_disconnected'])
        self.assertEqual(result['tx']['denied_tx_attempts'], 1)
        self.assertFalse(result['tx']['transmit_permitted'])
        with self.assertRaises(PassiveTransmitForbidden):
            FakeBus().send(SimpleNamespace(arbitration_id=1, data=b''))

    def test_sdk_target_pose_and_stationarity_must_match(self):
        robot, result = self.run_fake()
        segment, before = result['proposal'], feedback(CURRENT)
        final = feedback(segment['next_q_rad'])
        self.assertTrue(probe.target_checks(final, before, segment)['reached'])
        for kind in ['position', 'rotation', 'moving', 'nochange']:
            candidate = feedback(segment['next_q_rad'])
            if kind == 'position':
                candidate['fk_flange_pose_m_rad'][0] += .00026
            elif kind == 'rotation':
                candidate['fk_flange_pose_m_rad'][3] += math.radians(.06)
            elif kind == 'moving':
                candidate['status']['motion_status'] = 1
            else:
                candidate = feedback(CURRENT)
            self.assertFalse(probe.target_checks(candidate, before, segment)['reached'], kind)

    def write_reference(self, root):
        config = root/'config'
        config.mkdir()
        source = root/'output/pose.json'
        source.parent.mkdir()
        dataset = {'intrinsics_sha256': 'a'*64, 'camera_configuration_epoch': 'camera',
                   'marker_attachment_epoch': 'attachment', 'pose_record_valid': True,
                   'reference_q_rad': [0.]*7}
        for key in ['execution_enabled', 'motion_target_valid', 'reference_activation_valid',
                    'physical_contact_verified', 'physical_branch_verified', 'physical_palm_transform_valid']:
            dataset[key] = False
        source.write_text(json.dumps(dataset))
        descriptor = {'schema': 1, 'kind': 'usb_rgb_operator_middle_root_pregrasp_reference',
                      'camera_configuration_epoch': 'camera', 'marker_attachment_epoch': 'attachment',
                      'marker': {'T_marker_contact': None, 'physical_branch_verified': False},
                      'contact_definition': {'physical_point_hand_frame_m': None},
                      'arm': {'joints_rad': [0.]*7, 'joints_deg': [0.]*7},
                      'sources': {'pose_dataset': {'path': 'output/pose.json',
                                                  'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}}
        for key in ['execution_enabled', 'motion_target_valid', 'reference_activation_valid', 'physical_contact_verified']:
            descriptor[key] = False
        path = config/probe.REFERENCE_NAME
        path.write_text(json.dumps(descriptor))
        return path, source

    def test_reference_bound_provenance_epochs_flags_and_joint_units(self):
        with tempfile.TemporaryDirectory() as directory:
            path, source = self.write_reference(Path(directory))
            arguments = {'camera_epoch': 'camera', 'intrinsics_sha256': 'a'*64,
                         'marker_attachment_epoch': 'attachment'}
            result = probe.load_reference(path, **arguments)
            self.assertEqual(result['taught_q_rad'], [0.]*7)
            self.assertEqual(len(result['reference_sha256']), 64)
            for field, value in [('camera_epoch', 'moved'), ('intrinsics_sha256', 'b'*64),
                                 ('marker_attachment_epoch', 'new')]:
                with self.assertRaises(ValueError):
                    probe.load_reference(path, **(arguments | {field: value}))
            for field, value in [('execution_enabled', True), ('kind', 'unrelated'),
                                 ('arm', {'joints_rad': [0.]*7, 'joints_deg': [1.]*7})]:
                original = path.read_text()
                descriptor = json.loads(original)
                descriptor[field] = value
                path.write_text(json.dumps(descriptor))
                with self.assertRaises(ValueError):
                    probe.load_reference(path, **arguments)
                path.write_text(original)
            source.write_text(source.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                probe.load_reference(path, **arguments)

    def test_cli_default_inspect_atomic_json_and_fixed_step_cap(self):
        @contextmanager
        def lock(path):
            self.assertEqual(path, Path('/tmp/nero_can0_control.lock'))
            yield

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'report.json'
            argv = ['--camera-epoch', 'camera', '--intrinsics-sha256', 'a'*64,
                    '--marker-attachment-epoch', 'attachment', '--operator-same-scene-confirmed',
                    '--output', str(output)]
            with patch.object(probe, 'load_reference', return_value={'taught_q_rad': [0.]*7}), \
                    patch.object(probe, 'control_lock', side_effect=lock), \
                    patch.object(probe, 'load_sdk_runtime', return_value=(FakeBus, lambda: None)), \
                    patch.object(probe, 'run_probe', return_value={'success': True}) as run, \
                    patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(probe.main(argv), 0)
            self.assertFalse(run.call_args.kwargs['execute'])
            self.assertEqual(run.call_args.kwargs['lift_m'], .020)
            self.assertFalse(run.call_args.kwargs['allow_extended_lift'])
            self.assertEqual(run.call_args.kwargs['rotation_length_m'], .1)
            self.assertEqual(len(json.loads(output.read_text())['tool_source_sha256']), 64)
            self.assertFalse(list(Path(directory).glob('*.tmp')))
            with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                probe.main(argv+['--max-step-deg', '1'])

    def test_extended_ninety_mm_inspects_candidate_without_claiming_clearance(self):
        robot, result = self.run_fake(lift_m=.090, allow_extended_lift=True)
        self.assertTrue(result['success'], result)
        self.assertTrue(result['model_candidate']['valid'])
        self.assertAlmostEqual(result['model_candidate']['predicted_fk_flange_m_rad'][2], .090, delta=.00002)
        self.assertFalse(result['motion_target_valid'])
        self.assertAlmostEqual(result['proposal']['minimum_flange_z_m'], .090-.00002)
        self.assertFalse(result['proposal']['current_flange_floor_satisfied'])
        self.assertFalse(result['whole_hand_clearance_validated'])
        self.assertFalse(result['trajectory_validated'])
        self.assertFalse(result['physical_contact_verified'])
        self.assertEqual(result['per_joint_delta_caps_rad'], list(probe.EXTENDED_JOINT_CAPS_RAD))
        self.assertEqual(robot.commands, [])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertTrue(any('below requested' in value for value in result['blockers']))

    def test_current_below_extended_floor_cannot_execute_prospective_rise(self):
        robot, result = self.run_fake(execute=True, lift_m=.090, allow_extended_lift=True)
        self.assertFalse(result['success'])
        self.assertEqual(result['event'], 'taught_pregrasp_blocked')
        self.assertFalse(result['joint_motion_attempted'])
        self.assertGreater(result['proposal']['first_model_z_change_m'], 0.)
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])

    def test_below_extended_floor_downward_first_segment_is_explicitly_blocked(self):
        def curved_fk(q):
            return [0., 0., .030-.3*q[0], 0., 0., 0.]

        current, candidate = [0.]*7, [.030]+[0.]*6
        result = probe.first_segment(curved_fk, current, candidate, [0.]*6, LIMITS,
                                     lift_m=.090, allow_extended_lift=True)
        self.assertLess(result['first_model_z_change_m'], 0.)
        self.assertTrue(any('downward' in value for value in result['blocking_reasons']))
        self.assertFalse(result['whole_hand_clearance_validated'])

    def test_explicit_high_pose_preserves_single_small_segment_gates(self):
        robot = FakeRobot()
        robot.q = [0., 0., .4, 0., 0., 0., 0.]
        _, result = self.run_fake(execute=True, robot=robot, lift_m=.090, allow_extended_lift=True)
        self.assertTrue(result['success'], result)
        segment = result['proposal']
        self.assertTrue(segment['current_flange_floor_satisfied'])
        self.assertTrue(segment['first_endpoint_floor_satisfied'])
        self.assertLessEqual(segment['requested_max_joint_delta_rad'], math.radians(.2))
        self.assertFalse(segment['next_equals_candidate'])
        self.assertFalse(result['whole_hand_clearance_validated'])

    def test_extended_flag_caps_and_lift_bounds_reject_before_factory(self):
        for options in [{'lift_m': .090}, {'lift_m': .121, 'allow_extended_lift': True},
                        {'lift_m': 0., 'allow_extended_lift': True},
                        {'lift_m': .090, 'allow_extended_lift': 1}]:
            factory = unittest.mock.Mock(return_value=FakeRobot())
            _, result = self.run_fake(execute=True, factory=factory, **options)
            self.assertFalse(result['success'])
            factory.assert_not_called()
            self.assertEqual(result['tx']['actual_tx_count'], 0)
        for caps in [(.523,)*6, (0.,)*7, (math.radians(31),)*7, (False,)*7]:
            with patch.object(probe, 'EXTENDED_JOINT_CAPS_RAD', caps):
                factory = unittest.mock.Mock(return_value=FakeRobot())
                _, result = self.run_fake(factory=factory, lift_m=.090, allow_extended_lift=True)
            self.assertFalse(result['success'])
            factory.assert_not_called()

    def test_extended_lift_does_not_bypass_supplied_joint_limits(self):
        robot = FakeRobot()
        robot.get_config = lambda: {'joint_limits': {
            'joint'+str(i): ([-.15, .15] if i == 3 else [-2., 2.]) for i in range(1, 8)}}
        _, result = self.run_fake(execute=True, robot=robot, lift_m=.090, allow_extended_lift=True)
        self.assertFalse(result['success'])
        self.assertFalse(result['model_candidate']['valid'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)
        self.assertEqual(robot.commands, [])

    def test_explicit_extended_lift_accepts_bounded_120_mm_endpoint(self):
        _, result = self.run_fake(lift_m=.120, allow_extended_lift=True)
        self.assertTrue(result['success'], result)
        self.assertTrue(result['model_candidate']['valid'])
        self.assertAlmostEqual(result['model_candidate']['predicted_fk_flange_m_rad'][2], .120, delta=.00002)
        self.assertFalse(result['motion_target_valid'])
        self.assertEqual(result['tx']['actual_tx_count'], 0)

    def test_cli_extended_ninety_mm_is_explicit_and_default_stays_twenty(self):
        argv = ['--camera-epoch', 'camera', '--intrinsics-sha256', 'a'*64,
                '--marker-attachment-epoch', 'attachment', '--operator-same-scene-confirmed',
                '--lift-mm', '90']
        with patch.object(probe, 'load_sdk_runtime') as runtime, \
                patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
            probe.main(argv)
        runtime.assert_not_called()
        @contextmanager
        def lock(path):
            yield

        with patch.object(probe, 'load_reference', return_value={'taught_q_rad': [0.]*7}), \
                patch.object(probe, 'control_lock', side_effect=lock), \
                patch.object(probe, 'load_sdk_runtime', return_value=(FakeBus, lambda: None)), \
                patch.object(probe, 'run_probe', return_value={'success': True}) as run, \
                patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(probe.main(argv+['--allow-extended-lift']), 0)
        self.assertEqual(run.call_args.kwargs['lift_m'], .090)
        self.assertTrue(run.call_args.kwargs['allow_extended_lift'])
        self.assertFalse(run.call_args.kwargs['execute'])

    def test_rotation_merit_default_and_explicit_forwarding_preserve_endpoint_gates(self):
        for merit in [.1, .08, .05]:
            options = {} if merit == .1 else {'rotation_length_m': merit}
            with patch.object(probe, 'plan_taught_pregrasp', wraps=probe.plan_taught_pregrasp) as plan:
                _, result = self.run_fake(lift_m=.090, allow_extended_lift=True, **options)
            self.assertTrue(result['success'], result)
            self.assertTrue(result['model_candidate']['valid'])
            self.assertEqual(plan.call_args.kwargs['rotation_length_m'], merit)
            self.assertEqual(result['rotation_length_m'], merit)
            self.assertTrue(result['rotation_weight_is_merit_only'])
            self.assertEqual(result['model_candidate']['planner_configuration']['rotation_length_m'], merit)
            self.assertAlmostEqual(result['model_candidate']['tolerances']['rotation_rad'], math.radians(.05))
            self.assertEqual(result['model_candidate']['tolerances']['position_m'], .00002)
            self.assertFalse(result['motion_target_valid'])
            self.assertTrue(result['blockers'])
            self.assertFalse(result['whole_hand_clearance_validated'])
            self.assertEqual(result['tx']['actual_tx_count'], 0)

    def test_invalid_rotation_merit_rejected_before_factory(self):
        for merit in [True, False, float('nan'), float('inf'), .049, .101]:
            factory = unittest.mock.Mock(return_value=FakeRobot())
            _, result = self.run_fake(factory=factory, rotation_length_m=merit)
            self.assertFalse(result['success'], merit)
            factory.assert_not_called()
            self.assertEqual(result['tx']['actual_tx_count'], 0)

    def test_cli_explicit_eighty_mm_forwarding_and_invalid_numeric_values(self):
        argv = ['--camera-epoch', 'camera', '--intrinsics-sha256', 'a'*64,
                '--marker-attachment-epoch', 'attachment', '--operator-same-scene-confirmed',
                '--lift-mm', '90', '--allow-extended-lift']
        @contextmanager
        def lock(path):
            yield

        with patch.object(probe, 'load_reference', return_value={'taught_q_rad': [0.]*7}), \
                patch.object(probe, 'control_lock', side_effect=lock), \
                patch.object(probe, 'load_sdk_runtime', return_value=(FakeBus, lambda: None)), \
                patch.object(probe, 'run_probe', return_value={'success': True}) as run, \
                patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(probe.main(argv+['--rotation-length-mm', '80']), 0)
        self.assertEqual(run.call_args.kwargs['rotation_length_m'], .08)
        self.assertFalse(run.call_args.kwargs['execute'])
        for value in ['49', '101', 'nan', 'inf']:
            with patch.object(probe, 'load_sdk_runtime') as runtime, \
                    patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit):
                probe.main(argv+['--rotation-length-mm', value])
            runtime.assert_not_called()


if __name__ == '__main__':
    unittest.main()
