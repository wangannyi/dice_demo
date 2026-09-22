"""Known-geometry synthetic and archived RGB-D tests, with no hardware commands."""

import unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
import cv2
from cup_grasp_demo.flow.core import ROOT, load_config
from vision.geometry.cup_height import detect
from cup_grasp_demo.flow.green_image_rim import options


class ImageRimTest(unittest.TestCase):
    def setUp(self):
        self.opts = deepcopy(
            load_config(
                ROOT / "cup_grasp_demo/flow/green_open_cup/config.json"
            )["green_cup"]["perception"]
        )
        y, x = np.indices((480, 640))
        self.rad = np.hypot(x - 320, y - 240)
        self.ring = (self.rad >= 21) & (self.rad <= 24)
        self.image = np.zeros((480, 640, 3), np.uint8)
        self.image[:] = [0, 0, 200]
        self.image[self.rad <= 24] = [0, 90, 0]
        self.depth = np.full((480, 640), 1000, np.uint16)
        self.depth[self.rad <= 24] = 990
        self.depth[self.ring] = 935
        # Cup interior contains high dice whose depth must not set the rim height.
        self.depth[self.rad < 8] = 920
        self.image[self.rad < 8] = [240, 240, 240]
        self.meta = dict(
            intrinsics=dict(
                fx=600,
                fy=600,
                cx=320,
                cy=240,
                width=640,
                height=480,
                frame="color_optical",
                dist_coeffs=[0] * 5,
            ),
            depth_scale_m=0.001,
        )
        self.instances = [dict(mask=self.rad <= 24)]

    def run_detect(self, diagnostic):
        return detect(
            self.depth,
            self.image,
            self.meta,
            self.opts,
            5,
            instances=self.instances,
            diagnostics=diagnostic,
        )

    def test_image_rim_center_and_metric_scale(self):
        diagnostic = {}
        geo, _, _ = self.run_detect(diagnostic)
        self.assertTrue(diagnostic["valid"])
        self.assertAlmostEqual(geo["height_m"], 0.065, places=5)
        self.assertAlmostEqual(geo["radius_m"] * 2000, 75, delta=4)
        np.testing.assert_allclose(
            geo["rim_center_camera_m"], [0, 0, 0.935], atol=0.001
        )
        self.assertEqual(geo["section_height_fraction"], 1.0)
        self.assertGreaterEqual(geo["depth_sector_fraction"], 0.5)

    def test_tilted_table_recovers_rim_center(self):
        normal = np.array([0.0, -0.15, -1.0])
        normal /= np.linalg.norm(normal)
        table = np.array([0.0, 0.0, 1.0])
        center = table + 0.065 * normal
        u = np.array([1.0, 0.0, 0.0])
        v = np.cross(normal, u)
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        ring = center + 0.0375 * (
            np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v
        )
        uv = np.c_[
            600 * ring[:, 0] / ring[:, 2] + 320, 600 * ring[:, 1] / ring[:, 2] + 240
        ]
        mask = np.zeros((480, 640), np.uint8)
        cv2.fillPoly(mask, [np.rint(uv).astype("int32")], 1)
        y, x = np.indices(mask.shape)
        rays = np.stack(((x - 320) / 600, (y - 240) / 600, np.ones_like(x)), axis=-1)
        depth = (table @ normal) / (rays @ normal)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
        depth[inner] = (table @ normal + 0.01) / (rays[inner] @ normal)
        rim = (mask > 0) & ~inner
        depth[rim] = (table @ normal + 0.065) / (rays[rim] @ normal)
        self.depth = depth * 1000
        self.image[:] = [0, 0, 200]
        self.image[mask > 0] = [0, 90, 0]
        self.instances = [dict(mask=mask > 0)]
        geo, _, _ = self.run_detect({})
        self.assertAlmostEqual(geo["height_m"], 0.065, delta=0.002)
        self.assertAlmostEqual(geo["radius_m"] * 2000, 75, delta=4)
        np.testing.assert_allclose(geo["rim_center_camera_m"], center, atol=0.003)

    def test_two_depth_patches_do_not_authorize_circle(self):
        self.depth[self.ring] = 0
        self.depth[240:247, 340:345] = 935
        self.depth[260:265, 318:325] = 935
        diagnostic = {}
        with self.assertRaisesRegex(ValueError, "深度覆盖不足"):
            self.run_detect(diagnostic)
        self.assertFalse(diagnostic["valid"])
        self.assertIn("image_ellipse", diagnostic["candidates"][0])

    def test_no_depth_cannot_use_manual_dimensions_as_fallback(self):
        self.depth[self.rad < 28] = 0
        diagnostic = {}
        with self.assertRaisesRegex(ValueError, "可靠深度"):
            self.run_detect(diagnostic)
        self.assertFalse(diagnostic["valid"])

    def test_reference_dimensions_check_measured_result(self):
        self.opts["reference_dimensions_mm"]["height"] = 100
        with self.assertRaisesRegex(ValueError, "实测参考不符"):
            self.run_detect({})

    def test_green_detect_entry_never_requests_robot_snapshot(self):
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from cup_grasp_demo.flow import debug, green_pipeline
        from types import SimpleNamespace

        calls = []
        fake = SimpleNamespace(capture=lambda: calls.append("capture"))
        with (
            TemporaryDirectory() as d,
            patch.object(green_pipeline, "Workflow", return_value=fake),
            patch.object(
                debug, "bridge", side_effect=AssertionError("CAN must not be used")
            ),
        ):
            self.assertEqual(
                debug.main(
                    [
                        "green-detect",
                        "--config",
                        str(
                            ROOT
                            / "cup_grasp_demo/flow/green_open_cup/config.json"
                        ),
                        "--session",
                        d,
                    ]
                ),
                0,
            )
        self.assertEqual(calls, ["capture"])

    def test_repeated_capture_replaces_latest_diagnostics(self):
        import json
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        from cup_grasp_demo.flow import green_pipeline as flowmod

        with TemporaryDirectory() as directory:
            flow = object.__new__(flowmod.Workflow)
            flow.root = Path(directory)
            flow.cfg = load_config(
                ROOT / "cup_grasp_demo/flow/green_open_cup/config.json"
            )
            flow.g = flow.cfg["green_cup"]
            flow.tcp = np.eye(4)
            flow.args = SimpleNamespace(show=False, mode="fast")
            with (
                patch.object(flowmod.common, "capture_rgbd"),
                patch.object(
                    flowmod,
                    "load_batch",
                    side_effect=lambda p: (
                        self.meta,
                        self.depth,
                        self.image.copy(),
                        [],
                    ),
                ),
                patch.object(
                    flowmod.common, "camera_transform", return_value=(np.eye(4), True)
                ),
                patch.object(flowmod, "infer", return_value=(self.instances, {})),
            ):
                flow.capture()
                self.assertTrue(
                    json.loads((flow.root / "green_scene.json").read_text())["valid"]
                )
                flow.capture()
                with patch.object(
                    flowmod, "detect", side_effect=ValueError("invalid rim")
                ):
                    with self.assertRaises(ValueError):
                        flow.capture()
                self.assertFalse(
                    json.loads((flow.root / "green_scene.json").read_text())["valid"]
                )
                self.assertIn(
                    "invalid rim",
                    json.loads((flow.root / "green_rim_diagnostics.json").read_text())[
                        "error"
                    ],
                )

    def test_camera_failure_invalidates_previous_target(self):
        import json
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        from cup_grasp_demo.flow import green_pipeline

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "green_scene.json").write_text('{"valid":true}')
            (root / "green_rim_debug.png").write_bytes(b"old image")
            flow = object.__new__(green_pipeline.Workflow)
            flow.root = root
            flow.cfg = {}
            flow.g = {"table_plane_source": "live_depth", "perception": {"frame_count": 1}}
            from types import SimpleNamespace
            flow.args = SimpleNamespace(mode="fast", show=False)
            with patch.object(
                green_pipeline.common,
                "capture_rgbd",
                side_effect=RuntimeError("camera unavailable"),
            ):
                with self.assertRaises(RuntimeError):
                    flow.capture()
            self.assertFalse(
                json.loads((root / "green_scene.json").read_text())["valid"]
            )
            self.assertFalse((root / "green_rim_debug.png").exists())
            self.assertIn(
                "camera unavailable",
                json.loads((root / "green_rim_diagnostics.json").read_text())["error"],
            )

    def test_invalid_configuration(self):
        for key, value in [
            ("depth_sectors", 36.5),
            ("depth_tolerance_mm", float("nan")),
            ("min_depth_sector_fraction", 0),
        ]:
            with self.subTest(key=key):
                opts = deepcopy(self.opts)
                opts["image_rim"][key] = value
                with self.assertRaises(ValueError):
                    options(opts)



class OpeningContourTest(unittest.TestCase):
    def test_opening_excludes_offset_lower_wall(self):
        from cup_grasp_demo.flow.green_image_rim import image_contour
        image = np.zeros((160, 160, 3), np.uint8)
        image[:] = [0, 0, 200]
        cv2.ellipse(image, (85, 85), (42, 48), 0, 0, 360, (0, 100, 0), -1)
        cv2.circle(image, (75, 70), 30, (0, 35, 0), -1)
        cv2.rectangle(image, (85, 68), (94, 77), (240, 240, 240), -1)
        opts = {'min_area_px': 100, 'image_rim': {'contour_source': 'opening', 'opening_value_quantile': 0.3}}
        diagnostic = {}
        ellipse, _, _ = image_contour(image, np.ones((160, 160), bool), opts, diagnostic)
        np.testing.assert_allclose(ellipse[0], [75, 70], atol=1.5)
        self.assertEqual(diagnostic['contour_source'], 'opening')

    def test_uniform_green_does_not_invent_opening(self):
        from cup_grasp_demo.flow.green_image_rim import image_contour
        image = np.zeros((160, 160, 3), np.uint8)
        image[:] = [0, 0, 200]
        cv2.circle(image, (80, 80), 35, (0, 90, 0), -1)
        with self.assertRaisesRegex(ValueError, '无法可靠分离'):
            image_contour(image, np.ones((160, 160), bool), {
                'min_area_px': 100, 'image_rim': {'contour_source': 'opening'}}, {})

if __name__ == "__main__":
    unittest.main()
