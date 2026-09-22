"""Green-cup geometry, FK/IK, load authorization and failure ordering; no hardware."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import cv2
from cup_grasp_demo.flow import green_pipeline as flow
from cup_grasp_demo.flow.green_cup_geometry import rim_geometry, detect
from cup_grasp_demo.flow.green_cup_planning import (
    solve,
    held_cup_clearance,
    vertical_targets,
)
from cup_grasp_demo.flow.green_shake_validation import (
    validate_held_request,
)
from cup_grasp_demo.flow.core import ROOT, load_config, digest
from cup_grasp_demo.flow.shake import Kinematics

CFG = ROOT / "cup_grasp_demo/flow/green_open_cup/config.json"


class GreenTest(unittest.TestCase):
    def test_analytic_ik_jacobian_matches_finite_difference(self):
        from cup_grasp_demo.flow import green_cup_planning as planning
        from scipy.optimize import least_squares
        kin = Kinematics()
        seed = np.radians([0, -70, -90, 100, -10, -5, 5])
        target, _ = kin.forward(seed + np.radians([5, 3, -2, 4, 2, 1, -3]))
        def checked(fun, q, **kwargs):
            for probe in (q, q + .015):
                finite = np.column_stack([(fun(probe + np.eye(7)[j]*1e-6) -
                                           fun(probe - np.eye(7)[j]*1e-6))/2e-6 for j in range(7)])
                np.testing.assert_allclose(kwargs['jac'](probe), finite, atol=2e-8)
            return least_squares(fun, q, **kwargs)
        with patch.object(planning, 'least_squares', side_effect=checked):
            q = planning.solve(target, seed, [0, -13, 5])
        pose, _ = kin.forward(q)
        self.assertLess(np.linalg.norm(pose[:3, 3]-target[:3, 3]), .0015)

    def test_hand_recipe_and_feedback_failure(self):
        from cup_grasp_demo.flow import green_hand_execution as handmod
        from unittest.mock import Mock

        robot = Mock()
        robot.get_joints_enable_status_list.return_value = [True] * 7
        demo = Mock()
        demo.arm_snapshot.return_value = (
            [0] * 7,
            None,
            SimpleNamespace(arm_status=0, ctrl_mode=1),
        )
        cfg = load_config(CFG)
        cfg["green_cup"]["require_hand_position"] = True
        cfg["green_cup"]["read_hand_feedback"] = True
        target = [0, 100, 100, 40, 40, 100]
        plan = dict(start_q_rad=[0] * 7, target_0_100=target)
        result = {}
        with patch.object(
            handmod,
            "send_closure",
            return_value=dict(target_0_100=target, position_target_reached=None),
        ) as send:
            handmod.execute(plan, cfg, robot, Mock(), demo, result)
            self.assertEqual(send.call_args.args[2]["target_0_100"], target)
            self.assertTrue(result["finger_commands_sent"])
        with patch.object(
            handmod, "send_closure", return_value=dict(position_target_reached=False)
        ):
            with self.assertRaises(RuntimeError):
                handmod.execute(plan, cfg, robot, Mock(), demo, {})
        cfg["green_cup"]["require_hand_position"] = False
        with patch.object(handmod, "send_closure", return_value=dict(position_target_reached=False)):
            r = {}
            handmod.execute(plan, cfg, robot, Mock(), demo, r)
            self.assertFalse(r["hand_command"]["position_target_reached"])
            self.assertFalse(r["hand_command"]["position_required"])
        plan["target_0_100"] = [100] * 6
        with self.assertRaises(ValueError):
            handmod.execute(plan, cfg, robot, Mock(), demo, {})

    def test_home_uses_saved_table_without_camera(self):
        from unittest.mock import Mock

        wf = object.__new__(flow.Workflow)
        wf.cfg = load_config(CFG)
        wf.args = SimpleNamespace(mode="fast")
        wf.g = wf.cfg["green_cup"]
        wf.prepare_vision = Mock()
        wf.snapshot = Mock(return_value=[0.1] * 7)  # 不在 HOME → arm_plan 路径
        wf.issue = Mock()
        real_read_json = flow.read_json

        def home_only(path):
            if str(path) == str(wf.cfg["home"]):
                return {"joints_deg": [0] * 7}
            return real_read_json(path)

        with patch.object(
            flow.common,
            "capture_rgbd",
            side_effect=AssertionError("HOME camera forbidden"),
        ), patch.object(flow, "read_json", side_effect=home_only), patch.object(
            flow, "arm_plan", return_value=dict(blockers=[])
        ) as plan:
            wf.perform("HOME")
        plan.assert_called_once()
        self.assertEqual(wf.issue.call_args.args[0]["kind"], "green_home_open")
        self.assertIn("cup_normal_base", wf.scene)
        with patch.object(flow, "digest", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "标定已改变"):
                wf.table()

    def test_open_rim_height_ignores_cavity(self):
        opts = dict(load_config(CFG)["green_cup"]["perception"], geometry_method="depth_band")
        theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        rim = np.c_[0.045 * np.cos(theta), 0.045 * np.sin(theta), np.full(360, 0.89)]
        lower = np.c_[0.035 * np.cos(theta), 0.035 * np.sin(theta), np.full(360, 0.97)]
        result = rim_geometry(np.r_[rim, lower], [0, 0, 1], [0, 0, -1], opts)
        self.assertAlmostEqual(result["height_m"], 0.11)
        np.testing.assert_allclose(
            result["rim_center_camera_m"], [0, 0, 0.89], atol=1e-12
        )
        self.assertAlmostEqual(result["radius_m"], 0.045)
        with self.assertRaises(ValueError):
            rim_geometry(rim[:20], [0, 0, 1], [0, 0, -1], opts)

    def test_color_depth_detect_excludes_dice_and_multiple_cups(self):
        opts = dict(load_config(CFG)["green_cup"]["perception"], geometry_method="depth_band")
        image = np.zeros((480, 640, 3), np.uint8)
        image[:] = [0, 0, 200]
        depth = np.full((480, 640), 1000, np.uint16)
        yy, xx = np.indices(depth.shape)
        for cx in (240,):
            radius = np.hypot(xx - cx, yy - 240)
            ring = (radius >= 29) & (radius <= 32)
            image[radius < 29] = [240, 240, 240]
            depth[radius < 29] = 980
            image[ring] = [0, 130, 0]
            depth[ring] = 890
        meta = dict(
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
        instances = [dict(mask=np.hypot(xx - 240, yy - 240) <= 32)]
        result, mask, _ = detect(depth, image, meta, opts, 5, instances=instances)
        self.assertAlmostEqual(result["height_m"], 0.11, places=4)
        self.assertEqual(
            mask[240, 240], 255
        )  # Rim fitting, not mask holes, excludes the cavity.
        radius = np.hypot(xx - 400, yy - 240)
        ring = (radius >= 29) & (radius <= 32)
        image[ring] = [0, 130, 0]
        depth[ring] = 890
        instances.append(dict(mask=radius <= 32))
        with self.assertRaisesRegex(ValueError, "候选数：2"):
            detect(depth, image, meta, opts, 5, instances=instances)

    def test_two_class_decoder_drops_ground_without_shifting_mask_coefficients(self):
        from cup_grasp_demo.flow.green_yolo import cap_outputs

        detections = np.zeros((1, 38, 8400), np.float32)
        proto = np.zeros((1, 32, 160, 160), np.float32)
        detections[0, 4, :] = 0.8
        detections[0, 5, 0] = 0.9
        detections[0, 6, :] = 7
        converted = cap_outputs([detections, proto])[0]
        self.assertEqual(converted.shape, (1, 37, 8400))
        self.assertEqual(converted[0, 4, 0], 0)
        self.assertAlmostEqual(float(converted[0, 4, 1]), 0.8)
        self.assertEqual(converted[0, 5, 1], 7)
        with self.assertRaises(ValueError):
            cap_outputs([detections[:, :37], proto])

    def test_reference_ik_and_lift_hold_geometry(self):
        cfg = load_config(CFG)
        g = flow.validate(cfg)
        reference = flow.read_json(ROOT / g["reference"])
        q = np.array(reference["joints_rad"])
        kin = Kinematics()
        target = kin.forward(q)[0]
        solved = solve(target, q, g["wrist_reference_deg"])
        self.assertLess(
            np.linalg.norm(kin.forward(solved)[0][:3, 3] - target[:3, 3]), 0.0015
        )
        tcp = np.array(
            flow.read_json(cfg["tcp_candidate"])["T_flange_contact_candidate"]
        )
        targets = vertical_targets(q, tcp, 0.05, g["wrist_reference_deg"])
        direct = vertical_targets(q, tcp, .05, g["wrist_reference_deg"], single_target=True)
        self.assertEqual(len(direct), 1)
        self.assertEqual(len(targets), 5)
        direct_end = kin.forward(direct[0])[0] @ tcp
        np.testing.assert_allclose(direct_end[:3, 3] - (target @ tcp)[:3, 3],
                                   [0, 0, .05], atol=.0015)
        start = target @ tcp
        end = kin.forward(targets[-1])[0] @ tcp
        np.testing.assert_allclose(end[:3, 3] - start[:3, 3], [0, 0, 0.05], atol=0.0015)
        # The cup minimum includes radius when tilted, not just its center height.
        cup = np.eye(4)
        cup[:3, 3] = [0, 0, 0.055]
        rel = np.linalg.inv(target) @ cup
        scene = dict(cup_support_base_m=[0, 0, 0], cup_normal_base=[0, 0, 1])
        self.assertAlmostEqual(
            held_cup_clearance([q], rel, 0.045, 0.11, scene), 0, places=8
        )
        self.assertGreater(
            held_cup_clearance([targets[-1]], rel, 0.045, 0.11, scene), 48
        )

    def test_cycle_order_and_failure_never_release(self):
        for fail in (None, "LIFT", "SHAKE", "LOWER"):
            with tempfile.TemporaryDirectory() as d, self.subTest(fail=fail):
                calls = []

                class Fake:
                    receipts = {}

                    def __init__(self, args):
                        pass

                    def close(self):
                        pass

                    def unchanged(self):
                        pass

                    def perform(self, phase):
                        calls.append(phase)
                        if phase == fail:
                            raise RuntimeError("injected fault")

                args = SimpleNamespace(
                    session=Path(d),
                    config=CFG,
                    execute=True,
                    status=False,
                    resume=False,
                    until="place",
                    mode="fast",
                    show=False,
                )
                with patch.object(flow, "Workflow", Fake):
                    if fail:
                        with self.assertRaises(RuntimeError):
                            flow.run(args)
                        self.assertNotIn("OPEN", calls)
                        self.assertNotIn("RETURN_HOME", calls)
                    else:
                        self.assertEqual(flow.run(args), 0)
                        self.assertEqual(tuple(calls), flow.PHASES)

    def test_held_shake_requires_real_receipt_hash_and_cup_clearance(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "grip.json"
            target = [0, 100, 100, 40, 40, 100]
            path.write_text(
                json.dumps(
                    dict(
                        success=True,
                        hand_command=dict(
                            target_0_100=target, position_target_reached=None
                        ),
                    )
                )
            )
            req = dict(
                grasp_receipt_path=str(path),
                input_hashes={str(path): digest(path)},
                grip_targets_0_100=target,
                table_screen=dict(held_cup_min_mm=40),
                held_cup_margin_mm=5,
            )
            validate_held_request(req)
            receipt = json.loads(path.read_text())
            receipt["hand_command"].update(position_target_reached=False)
            path.write_text(json.dumps(receipt))
            req["input_hashes"][str(path)] = digest(path)
            with self.assertRaises(ValueError):
                validate_held_request(req)
            receipt["hand_command"].update(position_required=False, completion_basis="command_duration_only_position_not_required")
            path.write_text(json.dumps(receipt))
            req["input_hashes"][str(path)] = digest(path)
            validate_held_request(req)
            req["table_screen"]["held_cup_min_mm"] = -1
            with self.assertRaises(ValueError):
                validate_held_request(req)
            req["table_screen"]["held_cup_min_mm"] = 40
            path.write_text("{}")
            with self.assertRaises(ValueError):
                validate_held_request(req)

    def test_preview_never_constructs_hardware_workflow(self):
        with patch.object(flow, "Workflow", side_effect=AssertionError("hardware")):
            self.assertEqual(
                flow.run(SimpleNamespace(status=False, resume=False, execute=False)), 0
            )


if __name__ == "__main__":
    unittest.main()
