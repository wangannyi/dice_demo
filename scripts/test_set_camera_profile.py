"""Profile switching updates every active green and red-cloth camera entry."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.set_camera_profile import (BOARD_FILES, GREEN_FILES, ROOT,
                                        apply_profile)


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in GREEN_FILES + BOARD_FILES:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def test_usb2_and_usb3_keep_calibration_geometry(self):
        for link, fps in [('usb2', 6), ('usb3', 15), ('usb2', 6)]:
            report = apply_profile(self.root, link)
            self.assertFalse(report['recalibration_required'])
            for name in GREEN_FILES:
                camera = self.read(name)['green_cup']['camera']
                self.assertEqual(camera['fps'], fps)
                self.assertEqual(camera['color_resolution'], [1280, 720])
                self.assertEqual(camera['depth_resolution'], [1280, 720])
                self.assertEqual(camera['crop_xywh'], [220, 0, 960, 720])
            for name in BOARD_FILES:
                self.assertEqual(self.read(name)['image_profile']['fps'], fps)

    def test_resolution_change_requires_new_board_roi_and_recalibration(self):
        with self.assertRaisesRegex(ValueError, 'hand-roi'):
            apply_profile(self.root, 'usb2', color=[640, 480], depth=[640, 480], fps=15)
        self.assertEqual(self.read(GREEN_FILES[0])['green_cup']['camera']['fps'], 6)
        report = apply_profile(self.root, 'usb2', color=[640, 480], depth=[640, 480],
            fps=15, hand_roi=[0, 0, 640, 480], reference_roi=[300, 240, 500, 450])
        self.assertTrue(report['recalibration_required'])
        self.assertTrue(self.read(GREEN_FILES[0])['green_cup']['installation_requires_calibration'])
        self.assertEqual(self.read(BOARD_FILES[0])['image_exclude_rois_xyxy'], [])
        self.assertEqual(self.read(BOARD_FILES[1])['image_roi_xyxy'], [300, 240, 500, 450])

    def test_unsupported_usb2_profile_is_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, 'USB 2 profile'):
            apply_profile(self.root, 'usb2', fps=15)
        self.assertEqual(self.read(GREEN_FILES[0])['green_cup']['camera']['fps'], 6)


if __name__ == '__main__':
    unittest.main()
