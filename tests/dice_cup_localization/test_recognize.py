"""Same-frame executor snapshot bridge and green-ROI proposal tests."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from dice_cup_localization.recognize import _green_roi, geometry_config, load_source


class SnapshotBridgeTests(unittest.TestCase):
    def test_side_section_gate_requires_explicit_narrow_opt_in(self):
        self.assertEqual(geometry_config().max_center_spread_m, .005)
        self.assertEqual(geometry_config(6).max_center_spread_m, .006)
        for value in (4.9, 6.1, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                geometry_config(value)

    def test_snapshot_hash_and_identity_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.zeros((20, 30, 3), np.uint8)
            depth = np.full((20, 30), 600, np.uint16)
            cv2.imwrite(str(root/'color.png'), image)
            np.savez_compressed(root/'depth.npz', depth_raw=depth)
            metadata = {'schema': 1, 'depth_registered_to': 'color_optical',
                        'frame_id': 'serial:21', 'timestamp_ms': 123.4,
                        'timestamp_domain': 'hardware_clock',
                        'intrinsics': {'frame': 'color_optical'},
                        'sha256_color': hashlib.sha256((root/'color.png').read_bytes()).hexdigest(),
                        'sha256_depth': hashlib.sha256((root/'depth.npz').read_bytes()).hexdigest()}
            (root/'metadata.json').write_text(json.dumps(metadata))
            loaded, aligned, meta, source = load_source(snapshot=root)
            np.testing.assert_array_equal(loaded, image)
            np.testing.assert_array_equal(aligned, depth)
            self.assertEqual(meta['frame_id'], 'serial:21')
            self.assertEqual(source['sha256_depth'], metadata['sha256_depth'])
            (root/'depth.npz').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Snapshot hash mismatch'):
                load_source(snapshot=root)

    def test_auto_roi_is_proposal_only(self):
        image = np.zeros((480, 640, 3), np.uint8)
        image[128:200, 332:401] = [0, 170, 0]
        roi = _green_roi(image)
        self.assertTrue(roi[0] < 332 and roi[1] < 128)
        self.assertTrue(roi[2] > 401 and roi[3] > 200)
        with self.assertRaisesRegex(ValueError, 'No green area'):
            _green_roi(np.zeros_like(image))


if __name__ == '__main__':
    unittest.main()
