"""Camera switching keeps Pipeline and calibration board geometry aligned."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts.set_camera_profile import (BOARD_FILES, CAMERA_FILE, GREEN_FILES,
                                        PIPELINE_FILE, ROOT, apply_profile)


class CameraProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in GREEN_FILES + BOARD_FILES + (PIPELINE_FILE,):
            target = self.root / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def test_usb2_and_usb3_update_pipeline_and_board_profiles(self):
        for link, fps in [('usb2', 6), ('usb3', 15), ('usb2', 6)]:
            report = apply_profile(self.root, link)
            self.assertFalse(report['recalibration_required'])
            camera = self.read(CAMERA_FILE)
            self.assertEqual(camera['fps'], fps)
            self.assertEqual(camera['color_resolution'], [1280, 720])
            self.assertEqual(camera['depth_resolution'], [1280, 720])
            self.assertEqual(camera['crop_xywh'], [220, 0, 960, 720])
            for name in BOARD_FILES:
                self.assertEqual(self.read(name)['image_profile']['fps'], fps)

    def test_resolution_change_requires_rois_and_sets_installation_gate(self):
        with self.assertRaisesRegex(ValueError, 'hand-roi'):
            apply_profile(self.root, 'usb2', color=[640, 480], depth=[640, 480], fps=15)
        report = apply_profile(self.root, 'usb2', color=[640, 480], depth=[640, 480],
            fps=15, hand_roi=[0, 0, 640, 480], reference_roi=[300, 240, 500, 450])
        self.assertTrue(report['recalibration_required'])
        self.assertTrue(self.read(PIPELINE_FILE)['green_cup']['installation_requires_calibration'])
        self.assertEqual(self.read(BOARD_FILES[0])['image_exclude_rois_xyxy'], [])
        self.assertEqual(self.read(BOARD_FILES[1])['image_exclude_rois_xyxy'], [])
        self.assertEqual(self.read(BOARD_FILES[2])['image_roi_xyxy'], [300, 240, 500, 450])

    def test_unsupported_usb2_profile_is_rejected_before_writes(self):
        with self.assertRaisesRegex(ValueError, 'USB 2 profile'):
            apply_profile(self.root, 'usb2', fps=15)
        self.assertEqual(self.read(CAMERA_FILE)['fps'], 6)


if __name__ == '__main__':
    unittest.main()
