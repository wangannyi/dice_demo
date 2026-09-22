"""Camera profile switching keeps calibration geometry."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts.set_camera_profile import GREEN_FILES, ROOT, apply_profile


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        for name in GREEN_FILES:
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text((ROOT / name).read_text())

    def tearDown(self):
        self._tmp.cleanup()

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def test_unsupported_usb2_profile_is_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, 'USB 2 profile'):
            apply_profile(self.root, 'usb2', fps=15)
        self.assertEqual(self.read(GREEN_FILES[0])['green_cup']['camera']['fps'], 6)


if __name__ == '__main__':
    unittest.main()
