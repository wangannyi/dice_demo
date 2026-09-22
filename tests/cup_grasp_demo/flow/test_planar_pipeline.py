"""Main configuration -> planar planner -> JS receipt, without hardware access."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.flow import debug, planar_shake_cli as planar, pipeline_runner, shake_cli
from cup_grasp_demo.flow.core import ROOT, load_config, read_json, write_json
from cup_grasp_demo.flow.parameters import effective_shake_options, shake_options

DATA = ROOT / 'cup_grasp_demo/datasets/move_js_assessment_20260920'
CONFIG = ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json'


class PlanarPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.cfg = load_config(CONFIG)
        cls.config = cls.directory / 'config.json'
        write_json(cls.config, cls.cfg)
        reference = read_json(DATA / 'ready_plan.json')
        cls.session = {k: reference[k] for k in ('T_flange_tcp', 'scene')}
        write_json(cls.directory / 'session.json', cls.session)
        cls.feedback = read_json(DATA / 'controller_limits_live.json')
        cls.feedback['q_after_rad'] = read_json(DATA / 'grasp_actual.json')['final_joints_rad']
        cls.feedback_path = cls.directory / 'feedback.json'
        write_json(cls.feedback_path, cls.feedback)
        cls.output = cls.directory / 'shake_plan.json'
        args = SimpleNamespace(config=cls.config, session=cls.directory, output=cls.output,
                               feedback_json=cls.feedback_path)
        with redirect_stdout(io.StringIO()), patch.object(planar.subprocess, 'run', side_effect=AssertionError('CAN')):
            assert shake_cli.plan(args) == 0
        cls.plan = read_json(cls.output)

    def test_main_config_check_reports_same_parameters_as_plan(self):
        text = io.StringIO()
        with redirect_stdout(text), patch.object(debug, 'bridge', side_effect=AssertionError('CAN')), \
             patch.object(debug, 'capture_rgbd', side_effect=AssertionError('camera')):
            debug.config_check(SimpleNamespace(config=self.config))
        effective = json.loads(text.getvalue())['shake']
        self.assertEqual(effective.pop('strategy'), 'planar_js')
        self.assertEqual(effective, self.plan['parameters'])
        self.assertEqual(effective['frequency_hz'], 1.44)
        self.assertEqual(effective['amplitude_mm'], 50)
        self.assertEqual(effective['limit_utilization'], 1.)
        self.assertEqual(effective['joint_acceleration_cap_rad_s2'], 5.)

    def test_main_plan_uses_main_file_not_standalone_trial_config(self):
        self.assertEqual(self.plan['kind'], 'planar_js_shake_trial')
        self.assertEqual(self.plan['configuration_source'], 'main_config.shake')
        self.assertIsNone(self.plan['trial_config_path'])
        self.assertIn(str(self.config.resolve()), self.plan['input_hashes'])
        self.assertFalse(any(Path(p).name == 'planar_shake.json' for p in self.plan['input_hashes']))
        self.assertTrue(self.plan['planning_passed'])
        self.assertLess(max(self.plan['joint_peak_acceleration_rad_s2']), 5.)
        self.assertAlmostEqual(max(self.plan['joint_peak_acceleration_rad_s2']), 4.9729026554, places=6)

    def test_offline_plan_cannot_dispatch_any_executor(self):
        with redirect_stdout(io.StringIO()), patch.object(planar.subprocess, 'Popen') as sdk:
            with self.assertRaisesRegex(ValueError, '不可执行'):
                shake_cli.execute(SimpleNamespace(plan=self.output, execute=True))
            sdk.assert_not_called()

    def test_old_strategy_validation_and_invalid_strategy_stay_separate(self):
        old = dict(frequency_hz=.45, amplitude_mm=50, azimuth_deg=10, duration_s=20, limit_utilization=.97)
        self.assertEqual(effective_shake_options({'shake': old}), shake_options({'shake': old}))
        with self.assertRaises(ValueError):
            effective_shake_options({'shake': dict(old, limit_utilization=1.)})
        with self.assertRaises(ValueError):
            effective_shake_options({'shake': dict(old, strategy='typo')})
        for field, value in [('limit_utilization', 1.01), ('joint_acceleration_cap_rad_s2', 6)]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                effective_shake_options({'shake': dict(self.cfg['shake'], **{field: value})})

    def test_strategy_switch_replaces_old_plan_and_failure_invalidates_previous(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            output = Path(temp) / 'shake_plan.json'
            write_json(output, {'kind': shake_cli.KIND, 'planning_passed': True})
            args = SimpleNamespace(config=self.config, session=self.directory, output=output,
                                   feedback_json=self.feedback_path)
            self.assertEqual(shake_cli.plan(args), 0)
            self.assertEqual(read_json(output)['kind'], 'planar_js_shake_trial')
            with patch.object(shake_cli, 'load_config', return_value={'shake': {'invalid': True}}):
                with self.assertRaises(ValueError):
                    shake_cli.plan(args)
            self.assertFalse(output.exists())
            write_json(output, {'kind': 'unrelated_artifact'})
            with self.assertRaisesRegex(ValueError, '不能覆盖'):
                shake_cli.plan(args)
            self.assertEqual(read_json(output)['kind'], 'unrelated_artifact')

    def test_pipeline_plan_and_execute_route_to_js_and_cache_without_replanning(self):
        for success in (True, False):
            with self.subTest(success=success), ExitStack() as stack:
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(patch.object(planar.common, 'verify_session', return_value=(self.session, self.cfg)))
                def readback(command, **kwargs):
                    write_json(Path(command[command.index('--output')+1]), self.feedback)
                    return SimpleNamespace(returncode=0)
                stack.enter_context(patch.object(planar.subprocess, 'run', side_effect=readback))
                args = SimpleNamespace(config=self.config, session=self.directory,
                                       mode='fast', until='shake', show=False)
                backend = pipeline_runner.Backend(args)
                backend.perform('SHAKE_PLAN')
                planned = read_json(self.output)
                self.assertFalse(planned['offline_only'])
                self.assertEqual(planned['parameters'], self.plan['parameters'])
                self.assertTrue(backend.shake_prepared)
                requests = []
                def launch(command, **kwargs):
                    self.assertEqual(Path(command[1]).name, 'planar_shake_execution.py')
                    request = read_json(Path(command[command.index('--request')+1]))
                    requests.append(request)
                    actual = Path(command[command.index('--output')+1])
                    write_json(actual, dict(success=success, returned_center=success,
                        duration_completed=success, motion_elapsed_s=20 if success else 1,
                        measured_wave={'tracking_verified': success}, error=None if success else 'simulated fault'))
                    return SimpleNamespace(wait=lambda **kwargs: 0)
                child = stack.enter_context(patch.object(planar.subprocess, 'Popen', side_effect=launch))
                replan = stack.enter_context(patch.object(planar, 'make_plan', side_effect=AssertionError('redundant IK')))
                table = stack.enter_context(patch.object(planar, 'Screen', side_effect=AssertionError('redundant geometry')))
                if success:
                    receipt = backend.perform('SHAKE')
                    self.assertEqual(receipt['completed_state'], 'SHAKE_COMPLETED')
                else:
                    with self.assertRaisesRegex(RuntimeError, 'simulated fault'):
                        backend.perform('SHAKE')
                child.assert_called_once()
                replan.assert_not_called()
                table.assert_not_called()
                self.assertEqual(requests[0]['load'], 'cup')
                self.assertEqual(requests[0]['plan']['parameters']['frequency_hz'], 1.44)
                # Neither path sends a finger command from the shaking stage.
                self.assertNotIn('hand', requests[0])
                saved = self.config.read_text()
                try:
                    self.config.write_text(saved + '\n')
                    with self.assertRaisesRegex(ValueError, '依赖改变'):
                        backend.perform('SHAKE')
                    self.assertEqual(child.call_count, 1)
                finally:
                    self.config.write_text(saved)
        # Restore the class's read-only plan for other tests.
        self.output.write_text(json.dumps(self.plan))


if __name__ == '__main__':
    unittest.main()
