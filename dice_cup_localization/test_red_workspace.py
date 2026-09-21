"""Red-mat selection checks independent of YOLO's ground category."""

from pathlib import Path
import unittest

import numpy as np

from recognize import _select_red_cap
from red_workspace import RedWorkspace


class RedWorkspaceTests(unittest.TestCase):
    def test_inside_cap_has_red_context_outside_cap_does_not(self):
        config = Path(__file__).with_name('config')/'red_mat_640x480.json'
        image = np.full((480, 640, 3), [40, 80, 190], np.uint8)
        image[:110] = [190, 100, 40]
        inside = np.zeros((480, 640), bool)
        inside[180:220, 330:380] = True
        image[inside] = [0, 170, 0]
        outside = np.zeros_like(inside)
        outside[20:65, 60:110] = True
        image[outside] = [0, 170, 0]
        work = RedWorkspace(config, image.shape)
        self.assertTrue(work.evaluate(image, inside, [outside])['valid'])
        self.assertFalse(work.evaluate(image, outside, [inside])['valid'])
        instances = [{'class_id': 0, 'mask': outside}, {'class_id': 0, 'mask': inside}]
        selected, reason = _select_red_cap(image, instances, work)
        self.assertEqual(selected, 1)
        self.assertIsNone(reason)

    def test_support_center_must_project_inside_polygon(self):
        config = Path(__file__).with_name('config')/'red_mat_640x480.json'
        work = RedWorkspace(config, (480, 640, 3))
        intr = {'fx': 600, 'fy': 600, 'cx': 320, 'cy': 240}
        self.assertTrue(work.contains_projected([0, 0, .6], intr))
        self.assertFalse(work.contains_projected([0, -.25, .6], intr))


if __name__ == '__main__':
    unittest.main()
