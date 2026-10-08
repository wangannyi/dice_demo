"""A runtime error may be rechecked; missing/ambiguous cups never authorize motion."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow import green_pipeline as flow
from cup_grasp_demo.flow.core import ROOT, load_config
from vision.inference.detector import YOLOOutputError, cap_outputs


MISSING = "Stereo rim requires one YOLO cup in the red workspace"


class DetectionRecheckTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        self.run.mkdir()
        self.w = object.__new__(flow.Workflow)
        self.w.root = self.root
        self.w.cfg = load_config(ROOT / "configs/green_cup.json")
        self.w.g = self.w.cfg["green_cup"]
        self.w.g["table_plane_source"] = "live_depth"
        self.w.g["contact_offset_base_mm"] = [0, 0, 0]
        self.w.g["perception"].update(
            geometry_method="stereo_rim", frame_count=1,
            inference_provider="spacemit", inference_cpu_ids=[12, 13],
            cpu_recheck_on_detection_failure=True)
        self.w.tcp = np.eye(4)
        self.image = np.full((8, 8, 3), 100, dtype=np.uint8)
        self.geo = dict(rim_center_camera_m=[0, 0, .8],
                        table_normal_camera=[0, 0, 1], height_m=.065, radius_m=.0375)
        # A previous valid target must become unusable on any new failed capture.
        (self.root / "green_scene.json").write_text(json.dumps(dict(valid=True)))
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(flow.common, "new_run", return_value=self.run))
        stack.enter_context(patch.object(flow.common, "capture_rgbd"))
        stack.enter_context(patch.object(flow, "load_batch", return_value=(
            {}, np.ones((8, 8)), self.image, None)))
        stack.enter_context(patch.object(flow.common, "camera_transform",
                                        return_value=(np.eye(4), True)))
        stack.enter_context(patch.object(flow.cv2, "imwrite"))
        stack.enter_context(patch.object(self.w, "bridge",
                                        side_effect=AssertionError("No hardware")))
        self.infer = stack.enter_context(patch.object(flow, "infer"))
        self.locate = stack.enter_context(patch(
            "vision.geometry.circle_rim.detect_stereo", side_effect=self.geometry))

    def geometry(self, run, depth, image, meta, opts, tolerance, instances, diagnostic,
                 fixed_table=None):
        if len(instances) != 1:
            raise ValueError(MISSING)
        diagnostic["valid"] = True
        return self.geo, np.zeros((8, 8), np.uint8), None

    @staticmethod
    def prediction(count, provider="SpaceMITExecutionProvider"):
        return [{}] * count, dict(providers=[provider], sha256="same-model")

    def read(self, name, run=False):
        return json.loads(((self.run if run else self.root) / name).read_text())

    def test_missing_or_ambiguous_ai_rechecked_on_identical_image_and_thresholds(self):
        for count in (0, 7):
            with self.subTest(count=count):
                # Each capture has immutable evidence files, so use a fresh run.
                self.run = self.root / str(count)
                self.run.mkdir()
                with patch.object(flow.common, "new_run", return_value=self.run):
                    self.infer.side_effect = [self.prediction(count),
                                             self.prediction(1, "CPUExecutionProvider")]
                    self.w._capture_once()
                initial, cpu = self.infer.call_args_list[-2:]
                self.assertIs(initial.args[0], cpu.args[0])
                opts = dict(self.w.g["perception"], inference_provider="cpu",
                            inference_cpu_ids=[])
                self.assertEqual(cpu.args[1], opts)
                self.assertEqual(self.read("yolo_seg.json", True)["candidates"], count)
                report = self.read("yolo_cpu_recheck.json", True)
                self.assertEqual(report["candidates"], 1)
                self.assertEqual(report["reason"], MISSING)
                self.assertEqual(len(report["input_sha256"]), 64)
                self.assertTrue(self.read("green_scene.json")["valid"])
                self.assertEqual(self.read("green_scene.json")["model"]["providers"],
                                 ["CPUExecutionProvider"])

    def test_nonfinite_or_invalid_scores_rechecked_before_geometry(self):
        self.infer.side_effect = [YOLOOutputError("Nonfinite YOLO outputs"),
                                 self.prediction(1, "CPUExecutionProvider")]
        self.w._capture_once()
        self.assertEqual(self.locate.call_count, 1)
        self.assertIn("Nonfinite", self.read("yolo_seg.json", True)["error"])
        self.assertTrue(self.read("green_scene.json")["valid"])

    def test_successful_ai_needs_no_cpu_and_default_or_cpu_mode_never_falls_back(self):
        self.infer.return_value = self.prediction(1)
        self.w._capture_once()
        self.infer.assert_called_once()
        self.assertFalse((self.run / "yolo_cpu_recheck.json").exists())
        for provider, enabled in (("spacemit", None), ("cpu", True)):
            with self.subTest(provider=provider), tempfile.TemporaryDirectory() as d:
                self.run = Path(d)
                self.w.g["perception"]["inference_provider"] = provider
                self.w.g["perception"].pop("cpu_recheck_on_detection_failure", None)
                if enabled is not None:
                    self.w.g["perception"]["cpu_recheck_on_detection_failure"] = enabled
                self.infer.reset_mock()
                self.infer.return_value = self.prediction(0)
                with patch.object(flow.common, "new_run", return_value=self.run):
                    with self.assertRaisesRegex(ValueError, MISSING):
                        self.w._capture_once()
                self.infer.assert_called_once()
                self.assertFalse(self.read("green_scene.json")["valid"])

    def test_bad_cpu_predictions_still_stop_and_recheck_only_once(self):
        self.infer.side_effect = [YOLOOutputError("Nonfinite YOLO outputs"),
                                 self.prediction(0, "CPUExecutionProvider")]
        with self.assertRaisesRegex(ValueError, MISSING):
            self.w._capture_once()
        self.assertEqual(self.infer.call_count, 2)
        self.assertFalse(self.read("green_scene.json")["valid"])
        self.assertEqual(self.read("rim_diagnostics.json", True)["error"], MISSING)

    def test_cpu_runtime_failure_is_recorded_and_never_uses_old_target(self):
        self.infer.side_effect = [self.prediction(0), RuntimeError("CPU failed")]
        with self.assertRaisesRegex(RuntimeError, "CPU failed"):
            self.w._capture_once()
        self.assertFalse(self.read("green_scene.json")["valid"])
        self.assertEqual(self.read("yolo_cpu_recheck.json", True)["status"], "failed")

    def test_geometry_or_configuration_errors_do_not_trigger_recheck(self):
        self.infer.return_value = self.prediction(1)
        self.locate.side_effect = ValueError("bad stereo edges")
        with self.assertRaisesRegex(ValueError, "bad stereo edges"):
            self.w._capture_once()
        self.infer.assert_called_once()
        self.assertFalse(self.read("green_scene.json")["valid"])

    def test_cpu_still_checks_geometry_quality(self):
        self.infer.side_effect = [self.prediction(0), self.prediction(1)]
        self.locate.side_effect = [ValueError(MISSING), ValueError("bad stereo edges")]
        with self.assertRaisesRegex(ValueError, "bad stereo edges"):
            self.w._capture_once()
        self.assertFalse(self.read("green_scene.json")["valid"])

    def test_recheck_flag_requires_boolean(self):
        for bad in (0, 1, "true", None):
            cfg = load_config(ROOT / "configs/green_cup.json")
            cfg["green_cup"]["perception"]["cpu_recheck_on_detection_failure"] = bad
            with self.assertRaisesRegex(ValueError, "cpu_recheck_on_detection_failure"):
                flow.validate(cfg)

    def test_min_workspace_fraction_requires_fraction(self):
        for bad in (1.5, 0, -1, True, "0.1", None):
            cfg = load_config(ROOT / "configs/green_cup.json")
            cfg["green_cup"]["perception"]["min_workspace_fraction"] = bad
            with self.assertRaisesRegex(ValueError, "min_workspace_fraction"):
                flow.validate(cfg)

    def test_invalid_output_has_specific_error_type(self):
        detection = np.zeros((1, 38, 8400), np.float32)
        proto = np.zeros((1, 32, 160, 160), np.float32)
        for value in (float("nan"), float("inf"), 2):
            detection[0, 4, 0] = value
            with self.assertRaises(YOLOOutputError):
                cap_outputs([detection, proto])
