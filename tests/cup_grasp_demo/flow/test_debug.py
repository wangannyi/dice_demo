"""Coordinate-contract, execution-boundary and recorded-scene regression tests."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow.core import (
    ROOT, digest, flange_target, load_config, make_plan, measured_offset,
)
from cup_grasp_demo.flow.debug import validate_plan
from cup_grasp_demo.flow.hardware import validate_start
from nero_calibration.core import pose_matrix
from nero_revo2_control.kinematics import load_model

HERE = Path(__file__).resolve().parents[3] / 'cup_grasp_demo/flow'


class DebugTest(unittest.TestCase):
    def test_flange_and_tcp_have_same_comparison_point_and_flange_orientation(self):
        tcp = pose_matrix([.14, .006, .007, .1, .2, .3])
        rot = pose_matrix([0, 0, 0, .2, -.3, 1.1])[:3, :3]
        point = np.array([.12, .40, .10])
        flange = flange_target(point, rot, tcp, 'flange')
        contact = flange_target(point, rot, tcp, 'tcp')
        np.testing.assert_allclose(flange[:3, 3], point)
        np.testing.assert_allclose((contact @ tcp)[:3, 3], point)
        np.testing.assert_allclose(contact[:3, :3], flange[:3, :3])
        np.testing.assert_allclose(flange[:3, 3] - contact[:3, 3], rot @ tcp[:3, 3])

    def test_j7_rotation_is_in_fk_without_mutating_local_tcp(self):
        arm = load_model()
        q = np.array(json.loads((Path(HERE).parents[2] / 'configs/actions/home.json').read_text())['joints_rad'])
        tcp = pose_matrix([.14, .006, .007, 0, 0, 0])
        original = tcp.copy()
        first = np.array(arm.fk(q)) @ tcp
        q[6] += .3
        flange = np.array(arm.fk(q))
        second = flange @ tcp
        self.assertGreater(np.linalg.norm(second[:3, 3] - first[:3, 3]), .02)
        np.testing.assert_allclose(tcp, original)
        np.testing.assert_allclose(second[:3, 3], flange[:3, 3] + flange[:3, :3] @ tcp[:3, 3])

    def test_target_point_requires_clear_space(self):
        with self.assertRaisesRegex(ValueError, 'removing the cup'):
            make_plan({}, [0] * 7, {}, 'flange', 0, False)
        with self.assertRaisesRegex(ValueError, 'gap_mm'):
            make_plan({}, [0] * 7, {}, 'tcp', float('nan'), True)

    def test_stale_altered_replay_and_blocked_plans_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'session.json').write_text('{}')
            plan = dict(kind='flange_tcp_debug_plan', offline_only=False, blockers=[],
                        screen_passed=True, session_sha256=digest(root / 'session.json'),
                        created_epoch_s=100)
            cfg = {'plan_max_age_s': 600}
            validate_plan(plan, root, cfg, now=101)
            for change in ({'offline_only': True}, {'blockers': ['table']},
                           {'created_epoch_s': -1000}, {'created_epoch_s': 200},
                           {'session_sha256': 'changed'}):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_plan({**plan, **change}, root, cfg, now=101)

    def test_motion_start_requires_fresh_matching_idle_feedback(self):
        status = SimpleNamespace(arm_status=0, motion_status=0, ctrl_mode=3)
        plan = {'start_q_rad': [0] * 7}
        validate_start(plan, [0] * 7, status, [True] * 7, .25)
        for joints, state, enabled in [([.1] * 7, status, [True] * 7),
                                       ([0] * 7, SimpleNamespace(arm_status=0, motion_status=1, ctrl_mode=1), [True] * 7),
                                       ([0] * 7, status, [False] * 7)]:
            with self.assertRaises(RuntimeError):
                validate_start(plan, joints, state, enabled, .25)

    def test_error_comparison_is_in_flange_frame_and_requires_same_target(self):
        rot = pose_matrix([0, 0, 0, 0, 0, np.pi / 2])[:3, :3].tolist()
        first = dict(frame='flange', session_sha256='same', gap_mm=0,
                     R_base_flange=rot, error_base_mm=[1, 2, 3])
        second = {**first, 'frame': 'tcp', 'error_base_mm': [1, 12, 3]}
        np.testing.assert_allclose(measured_offset(first, second), [10, 0, 0], atol=1e-9)
        with self.assertRaises(ValueError):
            measured_offset(first, {**second, 'gap_mm': 50})

    def test_recorded_target_tcp_plan_passes_and_flange_ik_blocks(self):
        # A real saved capture; it is never opened as a live execution session.
        source = ROOT / 'cup_grasp_demo/datasets/standard_mount_20260917_01/preview/preview.json'
        saved = json.loads(source.read_text())
        contact = saved['stages'][-1]
        session = dict(scene=saved, T_flange_tcp=saved['tcp_model']['T_flange_contact_candidate'],
                       R_base_flange=np.array(contact['T_base_flange_target'])[:3, :3].tolist(),
                       contact_base_m=contact['tcp_target_base_m'], calibration_quality_passed=False)
        cfg = load_config(HERE / 'config.json')
        current = json.loads(Path(cfg['home']).read_text())['joints_rad']
        flange = make_plan(session, current, cfg, 'flange', 0, True)
        self.assertTrue(flange['blockers'])
        self.assertEqual(flange['stages'], [])
        tcp = make_plan(session, current, cfg, 'tcp', 0, True)
        self.assertEqual(tcp['blockers'], [])
        self.assertFalse(tcp['physical_tcp_verified'])
        np.testing.assert_allclose(tcp['comparison_point_base_m'], flange['comparison_point_base_m'])

    def test_preview_move_has_no_hardware_calls(self):
        from cup_grasp_demo.flow.debug import execute_plan
        plan = dict(session_path='/not_used', start_q_rad=[0] * 7, frame='tcp',
                    gap_mm=0, cup_removed=True, stages=[], blockers=[], screen_passed=True,
                    comparison_point_base_m=[0, 0, 0], T_base_flange_target=np.eye(4).tolist())
        with patch('cup_grasp_demo.flow.debug.read_json', return_value=plan), \
             patch('cup_grasp_demo.flow.debug.verify_session', return_value=({}, {})), \
             patch('cup_grasp_demo.flow.debug.validate_plan'), \
             patch('cup_grasp_demo.flow.debug.make_plan', return_value=copy.deepcopy(plan)), \
             patch('cup_grasp_demo.flow.debug.bridge') as hardware, \
             patch('cup_grasp_demo.flow.debug.capture_rgbd') as camera:
            execute_plan(SimpleNamespace(plan=Path('/not_used'), execute=False))
            hardware.assert_not_called()
            camera.assert_not_called()


if __name__ == '__main__':
    unittest.main()
