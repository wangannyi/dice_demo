"""FAST logging, shake phase ordering and receipt-based completion without hardware."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.flow import debug, pipeline_runner as runner, shake_cli
from cup_grasp_demo.flow.core import read_json
from test_pipeline_runner import FakeBackend


class FastTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.args = SimpleNamespace(session=self.path, config=Path('/unused'), status=False,
                                    until='shake', mode='fast', execute=True, resume=False, show=True)
        self.backend = FakeBackend(self.args)

    def run_flow(self):
        return runner.run(self.args, backend_factory=lambda _: self.backend,
                          ask=lambda _: self.fail('FAST must not prompt'))

    def test_cli_defaults_and_explicit_endpoint(self):
        for mode, endpoint in [('step', 'grip'), ('auto', 'grip'), ('fast', 'shake')]:
            with self.subTest(mode=mode), patch.object(debug, 'dispatch') as dispatch:
                debug.main(['pipeline', '--session', str(self.path), '--mode', mode])
                self.assertEqual(dispatch.call_args.args[0].until, endpoint)
        with patch.object(debug, 'dispatch') as dispatch:
            debug.main(['pipeline', '--session', str(self.path), '--mode', 'fast', '--until', 'grip'])
            self.assertEqual(dispatch.call_args.args[0].until, 'grip')

    def test_quiet_terminal_keeps_python_native_and_child_output_in_log(self):
        original = self.backend.perform
        def noisy(phase):
            print('python phase', phase)
            print('python stderr', file=sys.stderr)
            if phase == 'HOME':
                os.write(1, b'native stdout\n')
                os.write(2, b'native stderr\n')
                subprocess.run([sys.executable, '-c', 'print("camera frame path")'], check=True)
            return original(phase)
        self.backend.perform = noisy
        stdout, stderr = io.StringIO(), io.StringIO()
        fd = os.fstat(1)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(self.run_flow(), 0)
        self.assertEqual(os.fstat(1), fd)
        self.assertEqual(len(stdout.getvalue().splitlines()), 10)
        for phase in list(runner.PHASES) + ['SHAKE_PLAN', 'SHAKE']:
            self.assertIn(f'[耗时] {phase}：', stdout.getvalue())
        self.assertIn('[耗时汇总]', stdout.getvalue())
        self.assertEqual(stderr.getvalue(), '')
        state = read_json(self.path / runner.STATE_FILE)
        self.assertEqual(state['completed_state'], 'SHAKE_COMPLETED')
        self.assertEqual(self.backend.calls, list(runner.PHASES[:4]) + ['SHAKE_PLAN', 'GRIP', 'SHAKE'])
        self.assertFalse(state['physical_grip_verified'])
        content = Path(state['log_path']).read_text()
        for value in ('python phase', 'python stderr', 'native stdout', 'native stderr', 'camera frame path'):
            self.assertIn(value, content)
            self.assertNotIn(value, stdout.getvalue())

    def test_shake_plan_failure_stops_and_error_output_is_restored(self):
        self.backend.failure = 'SHAKE_PLAN'
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
                self.run_flow()
            print('stderr restored', file=sys.stderr)
        self.assertEqual(self.backend.calls[-1], 'SHAKE_PLAN')
        self.assertNotIn('SHAKE', self.backend.calls)
        state = read_json(self.path / runner.STATE_FILE)
        self.assertEqual(state['status'], 'FAILED')
        self.assertEqual(state['active_phase'], 'SHAKE_PLAN')
        self.assertIn('RuntimeError: simulated failure', Path(state['log_path']).read_text())
        self.assertIn('[耗时] SHAKE_PLAN：', stdout.getvalue())
        self.assertIn('（未完成）', stdout.getvalue())
        self.assertNotIn('[耗时] SHAKE：', stdout.getvalue())
        self.assertGreaterEqual(state['phase_timings_s']['SHAKE_PLAN'], 0)
        self.assertIn('FAST 停止', stderr.getvalue())
        self.assertIn('stderr restored', stderr.getvalue())

    def test_fast_timing_reaches_real_stdout_while_native_details_remain_logged(self):
        script = f'''\nimport sys\nsys.path.insert(0, r'{Path(__file__).resolve().parent}')\nfrom pathlib import Path\nfrom types import SimpleNamespace
from cup_grasp_demo.flow import pipeline_runner as r
from test_pipeline_runner import FakeBackend
a = SimpleNamespace(session=Path(sys.argv[1]), config=Path('/unused'),
    status=False, until='ready', mode='fast', execute=True, resume=False, show=False)
r.run(a, backend_factory=FakeBackend)
'''
        result = subprocess.run([sys.executable, '-c', script, str(self.path)],
                                capture_output=True, text=True, check=True)
        self.assertIn('[耗时] HOME：', result.stdout)
        self.assertIn('[耗时汇总]', result.stdout)
        self.assertNotIn('开始条件', result.stdout)

    def test_preview_has_no_backend_and_early_stop_has_no_shake(self):
        self.args.execute = False
        with patch.object(runner, 'Backend', side_effect=AssertionError('hardware')) as backend:
            runner.run(self.args, backend_factory=backend)
        self.assertFalse((self.path / 'runs').exists())
        self.args.execute = True
        self.args.until = 'grip'
        self.run_flow()
        self.assertEqual(self.backend.calls, list(runner.PHASES))

    def test_step_can_pause_before_shake_then_resume_without_repeating_grip(self):
        self.args.mode = 'step'
        answers = iter(['']*6 + ['q'])
        runner.run(self.args, backend_factory=lambda _: self.backend, ask=lambda _: next(answers))
        self.assertNotIn('SHAKE', self.backend.calls)
        self.args.resume = True
        runner.run(self.args, backend_factory=lambda _: self.backend, ask=lambda _: '')
        self.assertEqual(self.backend.calls, list(runner.PHASES) + ['SHAKE_PLAN', 'SHAKE'])

    def test_shake_backend_requires_success_tracking_and_center_receipt(self):
        backend = runner.Backend.__new__(runner.Backend)
        backend.args = self.args
        backend.show = False
        good = dict(success=True, returned_center=True, measured_wave={'tracking_verified': True},
                    actual_path='/unused/actual.json')
        with patch.object(shake_cli, 'execute', return_value=good) as execute:
            result = backend.perform('SHAKE')
            self.assertEqual(result['completed_state'], 'SHAKE_COMPLETED')
            self.assertFalse(result['cup_retention_verified'])
            self.assertTrue(execute.call_args.kwargs['return_receipt'])
            self.assertEqual(execute.call_args.kwargs['confirm']('unused'), 'SHAKE')
            self.assertFalse(execute.call_args.args[0].show)
        for change in ({'success': False}, {'returned_center': False}, {'measured_wave': {}}):
            bad = dict(deepcopy(good), **change)
            with self.subTest(change=change), patch.object(shake_cli, 'execute', return_value=bad):
                with self.assertRaisesRegex(RuntimeError, '摇晃未确认'):
                    backend.perform('SHAKE')


if __name__ == '__main__':
    unittest.main()
