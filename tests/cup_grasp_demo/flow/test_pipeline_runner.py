"""Shared phase order, pause/resume and release-before-localization HOME."""

from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.flow import debug, pipeline_home, pipeline_runner as runner
from cup_grasp_demo.flow.core import ROOT, load_config, read_json, write_json
from cup_grasp_demo.flow.session_storage import dispatch, session_lock


class FakeBackend:
    def __init__(self, _args):
        self.calls, self.version, self.outputs = [], 1, {}
        self.failure = None

    def fingerprint(self):
        return {'version': self.version}

    def artifacts(self):
        return dict(self.outputs)

    def perform(self, phase):
        self.calls.append(phase)
        if phase == self.failure:
            raise RuntimeError('simulated failure')
        self.outputs[phase] = 'completed'
        return {'phase': phase}


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = SimpleNamespace(command='pipeline', session=self.directory, config=Path('/unused'),
                                    status=False, mode='auto', until='grip', execute=True,
                                    resume=False, show=False)
        self.backend = FakeBackend(self.args)

    def run_flow(self, answers=('RUN',)):
        prompts = []
        it = iter(answers)
        def ask(prompt):
            prompts.append(prompt)
            return next(it)
        result = runner.run(self.args, backend_factory=lambda _: self.backend, ask=ask)
        return result, prompts

    def state(self):
        return read_json(self.directory / runner.STATE_FILE)

    def test_preview_and_status_never_construct_hardware_backend(self):
        self.args.execute = False
        with patch.object(runner, 'Backend', side_effect=AssertionError('hardware')) as backend:
            runner.run(self.args, backend_factory=backend)
            self.args.status = True
            runner.run(self.args, backend_factory=backend)
            backend.assert_not_called()
        self.assertFalse((self.directory / runner.STATE_FILE).exists())

    def test_auto_execute_runs_all_phases_without_confirmation(self):
        _, prompts = self.run_flow()
        self.assertEqual(len(prompts), 0)
        self.assertEqual(self.backend.calls, list(runner.PHASES))
        self.assertEqual(self.state()['completed_state'], 'GRIP_COMMANDS_SENT')
        self.assertFalse(self.state()['physical_grip_verified'])

    def test_phase_timing_excludes_step_prompts_and_prints_summary(self):
        self.args.mode = 'step'
        now = [0.]
        original = self.backend.perform
        def perform(phase):
            now[0] += 2
            return original(phase)
        def ask(_):
            now[0] += 100  # Human thinking time must not inflate execution time.
            return ''
        self.backend.perform = perform
        output = io.StringIO()
        with patch.object(runner.time, 'perf_counter', side_effect=lambda: now[0]), redirect_stdout(output):
            runner.run(self.args, backend_factory=lambda _: self.backend, ask=ask)
        self.assertEqual(self.state()['phase_timings_s'], dict.fromkeys(runner.PHASES, 2.))
        self.assertIn('[耗时] HOME：2.000 s', output.getvalue())
        self.assertIn('本次 PIPELINE：10.000 s', output.getvalue())

    def test_step_pause_resume_never_repeats_completed_motion(self):
        self.args.mode = 'step'
        self.run_flow(('', '', '', '', 'q'))
        self.assertEqual(self.state()['status'], 'PAUSED')
        self.assertEqual(self.state()['next_index'], 4)
        self.args.resume = True
        self.run_flow(('',))
        self.assertEqual(self.backend.calls, list(runner.PHASES))
        self.assertEqual(self.state()['status'], 'COMPLETED')

    def test_ready_endpoint_never_closes(self):
        self.args.until = 'ready'
        self.run_flow()
        self.assertEqual(self.backend.calls, list(runner.PHASES[:4]))
        self.assertEqual(self.state()['completed_state'], 'READY')

    def test_preview_preserves_old_state_and_does_no_work(self):
        old = dict(status='COMPLETED', notes='preserve')
        (self.directory / runner.STATE_FILE).write_text(json.dumps(old))
        self.args.execute = False
        self.run_flow(())
        self.assertEqual(self.state(), old)
        self.assertEqual(self.backend.calls, [])

    def test_failure_stops_before_later_phases_and_cannot_resume(self):
        self.backend.failure = 'APPROACH'
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            self.run_flow()
        self.assertEqual(self.backend.calls, list(runner.PHASES[:4]))
        self.assertEqual(self.state()['status'], 'FAILED')
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'PAUSED'):
            self.run_flow()

    def test_changed_config_or_outputs_reject_resume(self):
        self.args.mode = 'step'
        self.run_flow(('', 'q'))
        self.args.resume = True
        self.backend.version += 1
        with self.assertRaisesRegex(ValueError, '已改变'):
            self.run_flow()
        self.backend.version -= 1
        self.backend.outputs['HOME'] = 'changed'
        with self.assertRaisesRegex(ValueError, '已改变'):
            self.run_flow()

    def test_edit_during_step_pause_is_not_silently_applied(self):
        self.args.mode = 'step'
        def ask(_):
            self.backend.version += 1
            return ''
        with self.assertRaisesRegex(ValueError, '已改变'):
            runner.run(self.args, backend_factory=lambda _: self.backend, ask=ask)
        self.assertEqual(self.backend.calls, [])

    def test_inflight_marker_precedes_action_and_keyboard_interrupt_is_not_success(self):
        def perform(_):
            self.assertEqual(self.state()['status'], 'RUNNING')
            raise KeyboardInterrupt()
        self.backend.perform = perform
        with self.assertRaises(KeyboardInterrupt):
            self.run_flow()
        self.assertEqual(self.state()['status'], 'INTERRUPTED')

    def test_crashed_running_state_cannot_be_replayed(self):
        (self.directory / runner.STATE_FILE).write_text(json.dumps({'status': 'RUNNING'}))
        with self.assertRaisesRegex(ValueError, '中断'):
            self.run_flow()
        self.assertEqual(self.backend.calls, [])

    def test_pipeline_holds_existing_run_lock(self):
        with session_lock(self.directory), patch.object(runner, 'Backend') as backend:
            with self.assertRaisesRegex(RuntimeError, '此 RUN 正在'):
                dispatch(self.args, runner.run)
            backend.assert_not_called()

    def test_status_can_be_read_while_pipeline_holds_run_lock(self):
        (self.directory / runner.STATE_FILE).write_text(json.dumps({'status': 'RUNNING'}))
        with session_lock(self.directory), patch.object(debug, 'bridge') as sdk:
            self.assertEqual(debug.main(['pipeline', '--session', str(self.directory), '--status']), 0)
        sdk.assert_not_called()

    def test_cli_preview_preserves_existing_session_without_camera_or_can(self):
        with patch.object(debug, 'bridge') as sdk, patch.object(debug, 'capture_rgbd') as camera:
            self.assertEqual(debug.main(['pipeline', '--session', str(self.directory), '--mode', 'auto']), 0)
        sdk.assert_not_called()
        camera.assert_not_called()


class BackendTest(unittest.TestCase):
    def test_new_round_capture_uses_live_config_file_not_old_session(self):
        args = SimpleNamespace(config=ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json',
                               session=Path('/unused'), show=True, mode='auto')
        backend = runner.Backend(args)
        self.assertFalse(backend.show)
        with patch.object(debug, 'capture') as capture, \
             patch.object(debug, 'verify_session', return_value=({'capture_id': 'new'}, {})):
            backend.perform('CAPTURE')
        self.assertEqual(capture.call_args.args[0].config, args.config)
        self.assertIsNone(capture.call_args.args[0].replay)

    def test_plan_blockage_never_calls_motion(self):
        backend = runner.Backend.__new__(runner.Backend)
        backend.args = SimpleNamespace(session=Path('/unused'))
        with patch.object(runner.grasp_cli, 'plan', return_value=2), \
             patch.object(runner.grasp_cli, 'execute') as execute:
            with self.assertRaisesRegex(RuntimeError, '规划被阻塞'):
                backend.perform('GRIP')
        execute.assert_not_called()

    def test_success_receipt_for_wrong_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / 'ready_plan.json'
            write_json(path, {'until_state': 'ready'})
            backend = runner.Backend.__new__(runner.Backend)
            backend.args = SimpleNamespace(session=directory)
            backend.show = False
            def execute(*_, **__):
                write_json(directory / 'runs/new/receipt.json',
                           {'success': True, 'completed_state': 'OTHER'})
            with patch.object(runner.grasp_cli, 'execute', side_effect=execute):
                with self.assertRaisesRegex(RuntimeError, '所选终点'):
                    backend.move(path)


class HomeTest(unittest.TestCase):
    @staticmethod
    def missing_position_receipt():
        return dict(success=True, home_joint_target_reached=True, home_ready_verified=False,
                    finger_commands_sent=True,
                    home_hand_open=dict(command_wait_completed=True, target_0_100=[0] * 6,
                                        position_samples=[],
                                        completion_basis='command_duration_only_unverified'))

    def test_missing_position_after_completed_command_is_not_joint_home_failure(self):
        actual = self.missing_position_receipt()
        result = pipeline_home.check_home_result(actual)
        self.assertTrue(result['home_joint_target_reached'])
        self.assertFalse(result['home_ready_verified'])
        self.assertFalse(result['hand_position_verified'])
        self.assertEqual(result['hand_completion_basis'], 'command_duration_only_unverified')
        with self.assertRaisesRegex(RuntimeError, '张手'):
            pipeline_home.check_home_result(actual, require_position_feedback=True)

    def test_incomplete_open_or_failed_joint_motion_cannot_continue(self):
        updates = (
            {'success': False}, {'home_joint_target_reached': False},
            {'finger_commands_sent': False},
            {'home_hand_open': {'command_wait_completed': False}},
            {'home_hand_open': {'position_samples': [{'values': [100] * 6}]}},
            {'home_hand_open': {'target_0_100': [100] * 6}},
        )
        for update in updates:
            actual = self.missing_position_receipt()
            if 'home_hand_open' in update:
                actual['home_hand_open'].update(update['home_hand_open'])
            else:
                actual.update(update)
            with self.subTest(update=update), self.assertRaises(RuntimeError):
                pipeline_home.check_home_result(actual)

    def test_invalid_feedback_policy_fails_before_hardware(self):
        with patch.object(debug, 'new_run') as run, patch.object(debug, 'bridge') as sdk:
            with self.assertRaisesRegex(ValueError, 'boolean'):
                pipeline_home.execute_home(Path('/unused'), {'home_require_position_feedback': 'false'})
        run.assert_not_called()
        sdk.assert_not_called()

    def test_home_checks_table_and_joints_without_cup_geometry(self):
        arm = SimpleNamespace(check_joint_path=lambda *_: SimpleNamespace(samples_rad=[[0.] * 7],
                                                                          kinematic_checks_passed=True))
        with patch.object(pipeline_home, 'Screen') as factory:
            screen = factory.return_value
            screen.arm = arm
            screen.check.return_value = {'blockers': ['right_pinky: table clearance']}
            plan = pipeline_home.make_home_plan([0.] * 7, [0.] * 7, {},
                                                {'side_grasp': {'allow_hand_cup_contact': True}})
        self.assertTrue(plan['blockers'])
        self.assertEqual(screen.check.call_args.args[2], True)
        self.assertNotIn('allow_hand_cup_contact', screen.check.call_args.kwargs)
        self.assertEqual(plan['stages'], [])
        self.assertFalse(plan['cup_screened'])
        self.assertFalse(plan['cup_removed'])

    def test_invalid_joint_path_still_blocks_home(self):
        screen = SimpleNamespace(
            arm=SimpleNamespace(check_joint_path=lambda *_: SimpleNamespace(
                samples_rad=[[0.] * 7], kinematic_checks_passed=False)),
            check=lambda *_: {'blockers': []})
        plan = pipeline_home.make_home_plan([0.] * 7, [0.] * 7, {}, {}, screen)
        self.assertEqual(plan['blockers'], ['HOME 关节路径未通过'])

    def test_home_blocks_bad_path_or_incomplete_execution_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            cfg = {'home': str(directory / 'home.json')}
            Path(cfg['home']).write_text(json.dumps({'joints_rad': [0.] * 7}))
            for blockers, verified in ((['table'], True), ([], False), ([], True)):
                run = directory / str(len(list(directory.iterdir())))
                (run / 'rgbd').mkdir(parents=True)
                (run / 'rgbd/frame_000.png').touch()
                with patch.object(debug, 'new_run', return_value=run), \
                     patch.object(debug, 'capture_with_feedback', return_value={'joints_rad': [0.] * 7}), \
                     patch.object(pipeline_home, 'home_scene', return_value=({}, True)), \
                     patch.object(pipeline_home, 'make_home_plan', return_value={'blockers': blockers}), \
                     patch.object(debug, 'show'), \
                     patch.object(debug, 'bridge', return_value={'success': True, 'home_joint_target_reached': True,
                                                                 'home_ready_verified': verified}) as sdk:
                    if blockers:
                        with self.assertRaisesRegex(ValueError, 'HOME'):
                            pipeline_home.execute_home(directory, cfg)
                        sdk.assert_not_called()
                    elif not verified:
                        with self.assertRaisesRegex(RuntimeError, '张手'):
                            pipeline_home.execute_home(directory, cfg)
                    else:
                        self.assertTrue(pipeline_home.execute_home(directory, cfg)['home_ready_verified'])

    def test_recorded_cup_scene_supports_home_screen_without_camera(self):
        source = ROOT / 'cup_grasp_demo/datasets/contact_yaw20_20260917_01/current'
        session = read_json(source / 'session.json')
        cfg = load_config(ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json')
        cfg['contact_height_fraction'] = session['config']['contact_height_fraction']
        cfg['side_grasp'] = deepcopy(session['config']['side_grasp'])
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            shutil.copytree(source / 'rgbd', run / 'rgbd')
            with patch.object(debug, 'bridge') as sdk, patch.object(debug, 'capture_rgbd') as camera:
                scene, _ = pipeline_home.home_scene(run, cfg)
                q = read_json(cfg['home'])['joints_rad']
                plan = pipeline_home.make_home_plan(q, q, scene, cfg)
        self.assertEqual(plan['blockers'], [])
        self.assertFalse(plan['cup_removed'])
        self.assertFalse(plan['cup_screened'])
        self.assertEqual(plan['open_hand_target_0_100'], [0] * 6)
        sdk.assert_not_called()
        camera.assert_not_called()

    def test_recorded_gripping_scene_can_plan_home_without_cup_detection(self):
        source = ROOT / 'cup_grasp_demo/datasets/pipeline_release_home_20260918_01/failed_home'
        cfg = load_config(source / 'config.json')
        q = read_json(source / 'rgbd_before.json')['joints_rad']
        target = read_json(cfg['home'])['joints_rad']
        with patch.object(debug, 'bridge') as sdk, \
             patch.object(debug, 'capture_rgbd') as camera, \
             patch.object(debug, 'select_cup', side_effect=AssertionError('Cup detection before HOME')):
            scene, _ = pipeline_home.home_scene(source, cfg)
            plan = pipeline_home.make_home_plan(q, target, scene, cfg)
        self.assertEqual(plan['blockers'], [])
        self.assertGreaterEqual(plan['table_screen']['table_min_mm'], cfg['table_margin_mm'])
        self.assertEqual(plan['action_sequence'], ['OPEN_HAND', 'TO_HOME'])
        self.assertFalse(plan['cup_screened'])
        self.assertFalse(plan['cup_removed'])
        sdk.assert_not_called()
        camera.assert_not_called()


class RecordedPipelineTest(unittest.TestCase):
    def test_real_geometry_and_receipts_flow_with_simulated_hardware(self):
        """Exercise the real backend end to end; all I/O stays in a temporary RUN."""
        source = ROOT / 'cup_grasp_demo/datasets/contact_yaw20_20260917_01/current'
        raw = read_json(ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json')
        raw.pop('cup_perception', None)  # Preserve the known-good geometric pipeline replay.
        raw['contact_height_fraction'] = 9 / 11
        raw['side_grasp'].update(close_gap_mm=40, allow_hand_cup_contact=True,
                                 approach={'enabled': False})
        q = read_json(ROOT / raw['home'])['joints_rad']
        sent, hands = [], []
        def bridge(command, output, cfg, request=None):
            nonlocal q
            result = dict(success=True, joints_rad=list(q), arm_status=0, motion_status=0,
                          ctrl_mode=1, joints_enabled=[True] * 7, observed_epoch_s=1.)
            if command == 'run':
                plan = read_json(request)['plan']
                sent.append(plan['kind'])
                for stage in plan['stages']:
                    if stage.get('kind') == 'hand':
                        hands.append(stage['target_0_100'])
                    elif 'target_q_rad' in stage:
                        q = stage['target_q_rad']
                result.update(home_ready_verified=False, home_joint_target_reached=True,
                              finger_commands_sent=True,
                              home_hand_open=HomeTest.missing_position_receipt()['home_hand_open'],
                              last_state='GRIP_COMMANDS_SENT' if hands else 'READY',
                              final_joints_rad=list(q))
            write_json(output, result)
            return result
        def capture(output, _cfg):
            shutil.copytree(source / 'rgbd', output)
            # Production freshness checks use capture time, not old fixture mtimes.
            for file in output.glob('*.png'):
                file.touch()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = directory / 'config.json'
            write_json(config, raw)
            args = SimpleNamespace(command='pipeline', session=directory, config=config, execute=True,
                                    status=False, mode='auto', until='grip', resume=False, show=False)
            with patch.object(debug, 'bridge', side_effect=bridge), \
                 patch.object(debug, 'capture_rgbd', side_effect=capture), \
                 patch('builtins.input', side_effect=AssertionError('unexpected nested prompt')):
                self.assertEqual(dispatch(args, lambda a: runner.run(a, ask=lambda _: 'RUN')), 0)
            state = read_json(directory / runner.STATE_FILE)
            self.assertEqual(state['completed_state'], 'GRIP_COMMANDS_SENT')
            home_result = next(e['result'] for e in state['events']
                               if e['phase'] == 'HOME' and e['event'] == 'completed')
            self.assertTrue(home_result['home_joint_target_reached'])
            self.assertFalse(home_result['hand_position_verified'])
            self.assertEqual([e['phase'] for e in state['events'] if e['event'] == 'completed'],
                             list(runner.PHASES))
            self.assertTrue(all(read_json(directory / name)['screen_passed']
                                for name in ('ready_plan.json', 'grip_plan.json')))
        self.assertEqual(sent, ['home_open_debug_plan', 'side_grasp_debug_plan', 'side_grasp_debug_plan'])
        self.assertEqual(hands, [[0, 100, 0, 0, 0, 0], [100] * 6])


if __name__ == '__main__':
    unittest.main()
