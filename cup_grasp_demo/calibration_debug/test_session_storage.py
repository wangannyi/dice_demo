"""Same RUN refresh/replan must not silently reuse stale or mixed inputs."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug import debug, grasp_cli
from cup_grasp_demo.calibration_debug.core import read_json, write_json
from cup_grasp_demo.calibration_debug.session_storage import (
    CAPTURE_PENDING, dispatch, prepare_plan, reset_capture, session_lock,
)


class StorageTest(unittest.TestCase):
    def test_capture_cannot_overwrite_its_own_replay_source(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'rgbd').mkdir()
            (path / 'rgbd/frame.png').write_text('saved')
            with self.assertRaisesRegex(ValueError, '回放源'):
                reset_capture(path, path)
            self.assertEqual((path / 'rgbd/frame.png').read_text(), 'saved')
            self.assertFalse((path / CAPTURE_PENDING).exists())

    def test_capture_does_not_delete_files_behind_rgbd_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source, output = path / 'source', path / 'output'
            source.mkdir()
            output.mkdir()
            (source / 'data').write_text('preserve')
            (output / 'rgbd').symlink_to(source, target_is_directory=True)
            reset_capture(output)
            self.assertEqual((source / 'data').read_text(), 'preserve')
            self.assertFalse((output / 'rgbd').exists())

    def test_invalid_new_config_leaves_session_unusable_not_old_success(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / 'session.json').write_text('{}')
            args = SimpleNamespace(session=path, config=path / 'bad.json', replay=None)
            with patch.object(debug, 'load_config', side_effect=ValueError('invalid config')), \
                 patch.object(debug, 'bridge') as bridge, patch.object(debug, 'capture_rgbd') as camera:
                with self.assertRaisesRegex(ValueError, 'invalid config'):
                    debug.capture(args)
                bridge.assert_not_called()
                camera.assert_not_called()
            with self.assertRaisesRegex(ValueError, '最新采集尚未成功'):
                debug.verify_session(path)

    def test_cli_rejects_concurrent_refresh_or_execution_then_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_json(path / 'plan.json', {'session_path': temp})
            args_list = [SimpleNamespace(command='capture', session=path),
                         SimpleNamespace(command='grasp', plan=path / 'plan.json')]
            with session_lock(path), patch.object(debug, 'capture') as handler:
                for args in args_list:
                    with self.assertRaisesRegex(RuntimeError, '此 RUN 正在'):
                        dispatch(args, handler)
                handler.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'simulated error'):
                dispatch(args_list[0], lambda _: (_ for _ in ()).throw(ValueError('simulated error')))
            self.assertEqual(dispatch(args_list[0], lambda _: 'released'), 'released')

    def test_plan_overwrite_refuses_unrelated_json(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'session.json'
            write_json(output, {'kind': 'frozen_flange_tcp_comparison'})
            with self.assertRaisesRegex(ValueError, '不是同类型计划'):
                prepare_plan(output, 'flange_tcp_debug_plan')
            self.assertEqual(read_json(output)['kind'], 'frozen_flange_tcp_comparison')

    def exercise_plan(self, module, builder, kind):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            write_json(path / 'session.json', {})
            write_json(path / 'joints.json', dict(joints_rad=[0] * 7, arm_status=0, motion_status=0,
                                               joints_enabled=[True] * 7, ctrl_mode=1))
            output = path / 'plan.json'
            args = SimpleNamespace(session=path, output=output, joints_json=path / 'joints.json',
                                   frame='tcp', gap_mm=0, cup_removed=True, start='home', until='ready')
            def result(value):
                return dict(kind=kind, blockers=[], value=value)
            with patch.object(debug, 'verify_session', return_value=({'replay_only': False}, {})), \
                 patch.object(module, builder, side_effect=[result(1), result(2), ValueError('failed replan')]), \
                 patch.object(module, 'summarize'), patch.object(debug, 'bridge') as bridge, \
                 patch.object(grasp_cli, 'observed_scene', return_value={}):
                module.plan(args)
                self.assertEqual(read_json(output)['value'], 1)
                module.plan(args)
                self.assertEqual(read_json(output)['value'], 2)
                with self.assertRaisesRegex(ValueError, 'failed replan'):
                    module.plan(args)
                self.assertFalse(output.exists())
                bridge.assert_not_called()

    def test_same_name_comparison_plan_is_replaced_and_failure_removes_old_plan(self):
        self.exercise_plan(debug, 'make_plan', 'flange_tcp_debug_plan')

    def test_same_name_grasp_plan_is_replaced_and_failure_removes_old_plan(self):
        self.exercise_plan(grasp_cli, 'make_grasp_plan', 'side_grasp_debug_plan')


if __name__ == '__main__':
    unittest.main()
