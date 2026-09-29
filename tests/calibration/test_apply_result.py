import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import apply_result


class ApplyResultTests(unittest.TestCase):
    def test_install_copies_result_and_sets_registration_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'
            source.write_text(json.dumps({'mode': 'eye_to_hand', 'quality_passed': False,
                                          'T_base_camera': [[1, 0, 0, 0], [0, 1, 0, 0],
                                                            [0, 0, 1, 0], [0, 0, 0, 1]]}))
            config = root / 'green.json'
            config.write_text(json.dumps({'calibration': 'old.json', 'green_cup': {
                'installation_requires_calibration': False}}))
            destination = root / 'installed.json'
            with patch.object(apply_result, 'ROOT', root):
                report = apply_result.install(source, config, destination)
            self.assertFalse(report['quality_passed'])
            self.assertTrue(report['table_registration_required'])
            self.assertEqual(json.loads(destination.read_text()), json.loads(source.read_text()))
            installed = json.loads(config.read_text())
            self.assertEqual(installed['calibration'], 'installed.json')
            self.assertTrue(installed['green_cup']['installation_requires_calibration'])

    def test_invalid_result_is_rejected_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.json'; source.write_text('{}')
            config = root / 'green.json'; config.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'eye_to_hand'):
                apply_result.install(source, config, root / 'installed.json')
            self.assertFalse((root / 'installed.json').exists())


if __name__ == '__main__':
    unittest.main()
