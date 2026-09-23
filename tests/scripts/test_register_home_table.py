import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import register_home_table


class RegisterHomeTableTests(unittest.TestCase):
    def test_success_writes_bound_scene_and_clears_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration = root / 'calibration.json'; calibration.write_text('{"ok":true}')
            config = root / 'config.json'
            config.write_text(json.dumps({'calibration': str(calibration), 'green_cup': {
                'home_table_scene': 'home_table.json',
                'installation_requires_calibration': True}}))
            scene = root / 'scene.json'; scene.write_text('{}')
            cfg = json.loads(config.read_text())
            record = {'scene': {'table': 'test'}}
            with patch.object(register_home_table, 'ROOT', root), \
                 patch.object(register_home_table, 'verify', return_value=(record, cfg)):
                output = register_home_table.register(config, scene)
            self.assertEqual(output, root / 'home_table.json')
            saved = json.loads(output.read_text())
            self.assertEqual(saved['scene'], record['scene'])
            self.assertEqual(saved['calibration_sha256'], hashlib.sha256(calibration.read_bytes()).hexdigest())
            self.assertFalse(json.loads(config.read_text())['green_cup']['installation_requires_calibration'])

    def test_verify_failure_leaves_config_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); config = root / 'config.json'; config.write_text('{"unchanged":true}')
            before = config.read_bytes()
            with patch.object(register_home_table, 'verify', side_effect=ValueError('bad scene')):
                with self.assertRaisesRegex(ValueError, 'bad scene'):
                    register_home_table.register(config, root / 'scene.json')
            self.assertEqual(config.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
