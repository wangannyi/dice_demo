"""Timing excludes operator waits and preserves SDK cleanup and failure paths."""

from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from cup_grasp_demo.calibration_debug.phase_timing import PhaseTimer
from cup_grasp_demo.calibration_debug import planar_shake_execution as sdk
from cup_grasp_demo.calibration_debug import planar_shake_cli as cli
from cup_grasp_demo.calibration_debug.core import read_json


class TimingTest(unittest.TestCase):
    def test_failed_phase_is_recorded_and_exception_propagates(self):
        now, lines = [0.], []
        timer = PhaseTimer(emit=lines.append, clock=lambda: now[0])
        with timer.phase('first'):
            now[0] += 2
        now[0] += 60  # Outside any phase: operator confirmation.
        with self.assertRaises(KeyboardInterrupt):
            with timer.phase('second'):
                now[0] += 3
                raise KeyboardInterrupt()
        self.assertEqual(timer.summary('test'), 5.)
        self.assertEqual([r['status'] for r in timer.records], ['completed', 'failed'])
        self.assertIn('（未完成）', lines[1])

    def test_cli_failure_still_saves_timing(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'timings.json'
            def operation(_, timer):
                timer.output = output
                with timer.phase('TABLE_CHECK'):
                    raise ValueError('bad table')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'bad table'):
                cli.timed_command(operation, None, 'JS_PLAN')
            saved = read_json(output)
            self.assertEqual(saved['phases'][0]['status'], 'failed')
            self.assertTrue(saved['excludes_confirmation_wait'])

    def test_sdk_timing_preserves_motion_sequence_and_failure_hold(self):
        for failure in (False, True):
            with self.subTest(failure=failure), ExitStack() as stack:
                now = [0.]
                timer = PhaseTimer(emit=lambda _: None, clock=lambda: now[0])
                robot = MagicMock()
                robot.fk.return_value = [0.] * 6
                robot.get_firmware.return_value = {'software_version': '1.20'}
                session = MagicMock(robot=robot)
                guard = MagicMock()
                plan = dict(start_q_rad=[0.]*7,
                            samples=[dict(q_rad=[0.]*7, flange_pose_m_rad=[0.]*6)],
                            parameters={'limit_utilization': 1.},
                            joint_peak_velocity_rad_s=[1.]*7,
                            joint_peak_acceleration_rad_s2=[1.]*7)
                request = dict(plan=plan, channel='memory', load='empty',
                               restore_speed_percent=5, authorized_epoch_s=sdk.time.time())
                def replace(obj, name, **kw):
                    return stack.enter_context(patch.object(obj, name, **kw))
                replace(sdk, 'PhaseTimer', return_value=timer)
                replace(sdk, 'validate')
                replace(sdk.core, 'load_sdk_runtime', return_value=(object, object))
                replace(sdk.core, 'PassivePoseSession', return_value=session)
                replace(sdk, 'JSGuard', return_value=guard)
                replace(sdk.core, 'stopped_window', return_value=([dict(q_rad=[0.]*7)], {}))
                replace(sdk.core, 'ready_blockers', return_value=[])
                replace(sdk.core, 'joint_limits', return_value=[[-2., 2.]]*7)
                replace(sdk.core, 'validate_target')
                replace(sdk.legacy, 'control_evidence', return_value={})
                replace(sdk.legacy, 'control_conflicts', return_value=[])
                replace(sdk, 'query_joint_limit', return_value=SimpleNamespace(msg=SimpleNamespace(
                    max_joint_spd=5., max_joint_acc=5., min_angle_limit=-2., max_angle_limit=2.)))
                replace(sdk, 'take_js_control')
                def stream(_, __, report):
                    now[0] += 1 if failure else 20
                    if failure:
                        raise RuntimeError('lost feedback')
                    report.update(duration_completed=True, measured_wave={'tracking_verified': True})
                motion = replace(sdk, 'stream', side_effect=stream)
                settle = replace(sdk, 'settle', side_effect=lambda *args: now.__setitem__(0, now[0]+.2))
                hold = replace(sdk, 'fresh_js_hold', return_value={'target_rad': [0.]*7})
                replace(sdk.legacy, 'verify_stop')
                report = sdk.run(request)
                motion.assert_called_once()
                session.close.assert_called_once()
                self.assertEqual([c.args[0] for c in robot.set_motion_mode.call_args_list], ['js'])
                phases = {r['phase'].split()[0]: r for r in report['phase_timings']}
                self.assertEqual(phases['SHAKE']['duration_s'], 1. if failure else 20.)
                self.assertEqual(phases['SHAKE']['status'], 'failed' if failure else 'completed')
                self.assertIn('CLEANUP', phases)
                self.assertEqual(report['success'], not failure)
                if failure:
                    hold.assert_called_once()
                    settle.assert_called_once()
                    self.assertTrue(report['failure_hold']['hold_verified'])
                else:
                    hold.assert_not_called()
                    settle.assert_called_once()
                    self.assertAlmostEqual(report['phase_total_s'], 20.2)


if __name__ == '__main__':
    unittest.main()
