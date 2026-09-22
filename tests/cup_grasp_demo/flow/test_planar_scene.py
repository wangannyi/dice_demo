"""A failed cup capture cannot invalidate an explicitly captured table scene."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from cup_grasp_demo.flow import planar_scene as scene


class TableSceneTest(unittest.TestCase):
    def test_independent_capture_preserves_cup_failure_and_checks_changes(self):
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            config = root / 'config.json'; config.write_text('{}')
            calibration = root / 'calibration.json'; calibration.write_text('{}')
            tcp = root / 'tcp.json'; tcp.write_text('{}')
            marker = root / '.capture_pending'; marker.write_text('cup failed')
            cfg = dict(calibration=str(calibration), tcp_candidate=str(tcp))
            stack.enter_context(patch.object(scene, 'load_config', return_value=cfg))
            stack.enter_context(patch.object(scene, 'configured_tcp', return_value=np.eye(4)))
            stack.enter_context(patch.object(scene.common, 'new_run', return_value=root / 'run'))
            def camera(output, cfg):
                output.mkdir(parents=True); (output/'depth.bin').write_bytes(b'depth')
            stack.enter_context(patch.object(scene.common, 'capture_rgbd', side_effect=camera))
            stack.enter_context(patch.object(scene, 'home_scene', return_value=(dict(cup_normal_base=[0,0,1]), True)))
            args = SimpleNamespace(session=root, config=config)
            scene.capture(args)
            output = root / 'planar_table_scene.json'
            self.assertEqual(scene.verify(output, config)[0]['kind'], 'planar_table_scene')
            self.assertEqual(marker.read_text(), 'cup failed')
            self.assertFalse((root / 'session.json').exists())
            calibration.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, '依赖已改变'):
                scene.verify(output, config)
            with patch.object(scene.common, 'capture_rgbd', side_effect=RuntimeError('camera failed')):
                with self.assertRaises(RuntimeError):scene.capture(args)
            with self.assertRaisesRegex(ValueError, '最新桌面采集未成功'):
                scene.verify(output, config)


if __name__ == '__main__':unittest.main()
