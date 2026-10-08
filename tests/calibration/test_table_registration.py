"""Table registration checks use saved files and never access camera or CAN."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cup_grasp_demo.flow import core
from tools import register_home_table, table_capture


class TableRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root/'pipeline.json'
        self.calibration = self.root/'calibration.json'
        self.calibration.write_text('installed hand-eye result')
        self.cfg = json.loads((core.ROOT/'configs/green_cup.json').read_text())
        self.cfg['calibration'] = 'calibration.json'
        self.cfg['green_cup'].update(home_table_scene='table/home.json',
                                     installation_requires_calibration=True)
        self.config.write_text(json.dumps(self.cfg))
        self.table = self.root/'capture.json'
        self.scene = {'cup_support_base_m': [0, 0, .02], 'cup_normal_base': [0, 0, 1]}
        self.table.write_text(json.dumps({
            'kind': 'planar_table_scene', 'config_path': str(self.config), 'scene': self.scene,
            'input_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (self.config, self.calibration)}}))
        self.addCleanup(patch.stopall)
        patch.object(core, 'ROOT', self.root).start()
        patch.object(register_home_table, 'ROOT', self.root).start()

    def test_checked_table_binds_calibration_and_preserves_config_paths(self):
        output = register_home_table.register(self.config, self.table)
        record = json.loads(output.read_text())
        self.assertEqual(record['scene'], self.scene)
        self.assertEqual(record['calibration_sha256'],
                         hashlib.sha256(self.calibration.read_bytes()).hexdigest())
        expected = copy.deepcopy(self.cfg)
        expected['green_cup']['installation_requires_calibration'] = False
        self.assertEqual(json.loads(self.config.read_text()), expected)

    def test_changed_dependency_is_rejected_before_binding(self):
        self.calibration.write_text('different calibration')
        with self.assertRaisesRegex(ValueError, '桌面依赖已改变'):
            register_home_table.register(self.config, self.table)
        self.assertEqual(json.loads(self.config.read_text()), self.cfg)
        self.assertFalse((self.root/'table/home.json').exists())

    def test_incomplete_table_capture_cannot_enable_pipeline(self):
        self.table.with_suffix('.pending').write_text('incomplete')
        with self.assertRaisesRegex(ValueError, '最新桌面采集未成功'):
            register_home_table.register(self.config, self.table)
        self.assertEqual(json.loads(self.config.read_text()), self.cfg)

    def test_failed_table_write_keeps_installation_gate_closed(self):
        with patch.object(register_home_table, 'atomic_bytes', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                register_home_table.register(self.config, self.table)
        self.assertEqual(json.loads(self.config.read_text()), self.cfg)

    def test_capture_entry_creates_locked_session_before_read_only_capture(self):
        session = self.root/'new-capture'
        def capture(args):
            self.assertEqual(args.config, self.config)
            self.assertEqual(args.session, session)
            self.assertTrue((session/'.debug.lock').exists())
            return 0
        with patch.object(table_capture.planar_scene, 'capture', side_effect=capture):
            self.assertEqual(table_capture.main(['--config', str(self.config),
                                                  '--session', str(session)]), 0)


if __name__ == '__main__':
    unittest.main()
