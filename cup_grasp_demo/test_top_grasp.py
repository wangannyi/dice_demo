"""Behavioral checks for the opt-in read-only Revo2 cup-top proposal."""

import copy
import io
import json
import math
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from cup_grasp_demo.top_grasp import compute_top_grasp, main
from nero_calibration.core import PALM, matrix_pose
from nero_revo2_control.kinematics import load_model


HAND_OPEN = {'thumb_tip': 40., 'thumb_base': 40., 'index_finger': 0.,
             'middle_finger': 0., 'ring_finger': 0., 'pinky_finger': 0.}
MANUAL_Q = np.radians((48.705, -100.345, -101.039, 77.192,
                       -91.351, -41.289, -24.416))
SDK = [[-157., 157.], [-100., 100.], [-160., 160.], [-60., 125.],
       [-160., 160.], [-44., 57.], [-97., 97.]]


def source_and_feedback():
    """Measured-size cup fixture with explicit offline joint feedback."""
    flange = np.asarray(load_model().fk(MANUAL_Q))
    source = {
        'schema': 1, 'kind': 'read_only_side_grasp_proposal', 'units': 'm_rad',
        'localization': {'valid': True}, 'snapshot_frame_id': 'fixture-color-optical:63',
        'calibration_sha256': 'b'*64,
        'cup_base': {
            'frame': 'base', 'source': {'frame_id': 'fixture-color-optical:63'},
            'support_center_m': [-.1140344, .4741704, -.0140432],
            'axis': [.0286169, .0148116, .9994807],
            'dimensions': {'observed_height_m': .0704270,
                           'observed_side_diameter_m': .0757828}},
        'current_flange_pose_base_m_rad': matrix_pose(flange),
        'current_palm_pose_base_m_rad': matrix_pose(flange @ PALM),
        'pose_source': {'kind': 'can_feedback_read_only', 'channel': 'can0',
                        'host_read_time_ns': time.time_ns()},
        'checks': {'snapshot_age_s': 2., 'provisional_opt_in': True,
                   'calibration_quality_passed': False},
    }
    feedback = {'joints_rad': MANUAL_Q.tolist(), 'sdk_limits_deg': SDK,
                'joints_enabled': [False]*7, 'arm_status': 6, 'ctrl_mode': 3}
    return source, feedback


class TopGraspTest(unittest.TestCase):
    def test_provisional_steps_have_explicit_contact_gate_and_no_motion(self):
        source, feedback = source_and_feedback()
        before = copy.deepcopy(source)
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64, live_joint_feedback=feedback,
            hand_open_positions=HAND_OPEN,
            hand_status_source={'kind': 'hand_feedback_read_only'},
            orientation_reviewed=True, provisional_opt_in=True)
        self.assertEqual(source, before)
        self.assertEqual(result['kind'], 'read_only_top_grasp_proposal')
        self.assertEqual(result['stage_sequence'], [
            'clearance', 'pretop', 'guarded_hover', 'probe_1', 'probe_2',
            'contact_candidate', 'tiny_shake_a', 'tiny_shake_b'])
        axis = np.asarray(result['cup_base']['axis'])
        top = np.asarray(result['cup_top_center_base_m'])
        offsets = [float((np.asarray(result['waypoints'][name]['T_base_palm'])[:3, 3]-top) @ axis)
                   for name in ('guarded_hover', 'probe_1', 'probe_2', 'contact_candidate')]
        self.assertTrue(all(a-b <= .00501 for a, b in zip(offsets, offsets[1:])))
        np.testing.assert_allclose(offsets, [.028, .023, .018, .013], atol=1e-5)
        self.assertGreater(offsets[-1], 0.)
        self.assertFalse(result['checks']['physical_contact_verified'])
        self.assertFalse(result['checks']['cup_held_verified'])
        self.assertFalse(result['motion_sent'])
        self.assertFalse(result['checks']['execute_ready'])
        self.assertFalse(result['checks']['current_joints_within_sdk_urdf_documented_limits'])
        self.assertFalse(result['checks']['current_all_joints_enabled'])
        self.assertEqual(result['hand_close_updates']['thumb_base'], 43.)
        self.assertEqual(result['hand_close_updates']['thumb_tip'], 45.)
        self.assertEqual(result['hand_close_updates']['index_finger'], 18.)
        self.assertEqual(result['provenance']['palm_contact_axis_local'], '+Z')
        self.assertFalse(result['checks']['measured_dice_exit_lane_reviewed'])
        contact = np.asarray(result['waypoints']['contact_candidate']['T_base_palm'])[:3, 3]
        shake = np.asarray(result['waypoints']['tiny_shake_a']['T_base_palm'])[:3, 3]
        self.assertAlmostEqual(float((shake-contact) @ axis), 0., places=5)
        self.assertNotIn('lift', result['waypoints'])

    def test_old_flange_or_wrong_palm_orientation_is_rejected(self):
        source, feedback = source_and_feedback()
        stale = np.asarray(load_model().fk(np.radians((55, -78, 80, -45, 130, -30, 40))))
        source['current_flange_pose_base_m_rad'] = matrix_pose(stale)
        source['current_palm_pose_base_m_rad'] = matrix_pose(stale @ PALM)
        with self.assertRaisesRegex(ValueError, 'recapture cup and flange'):
            compute_top_grasp(
                source, source_side_plan_sha256='a'*64,
                live_joint_feedback=feedback, hand_open_positions=HAND_OPEN)
        source, feedback = source_and_feedback()
        with self.assertRaisesRegex(ValueError, 'Palm \\+Z must face down'):
            compute_top_grasp(
                source, source_side_plan_sha256='a'*64,
                live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
                palm_orientation_base=np.eye(3))

    def test_read_only_quality_and_hand_feedback_gates(self):
        source, feedback = source_and_feedback()
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64,
            live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
            orientation_reviewed=True, provisional_opt_in=False)
        self.assertFalse(result['checks']['provisional_allowed'])
        self.assertFalse(result['checks']['execute_ready'])
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64,
            live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
            orientation_reviewed=True, provisional_opt_in=True,
            now_ns=source['pose_source']['host_read_time_ns']+int(20e9))
        self.assertFalse(result['checks']['source_snapshot_fresh'])
        with self.assertRaisesRegex(ValueError, 'all six'):
            compute_top_grasp(
                source, source_side_plan_sha256='a'*64,
                live_joint_feedback=feedback,
                hand_open_positions={'index_finger': 0})

    def _positive_fixture(self):
        source, feedback = source_and_feedback()
        q = np.radians((48.705, -80., -101.039, 77.192,
                        -91.351, -35., -24.416))
        flange = np.asarray(load_model().fk(q))
        source['current_flange_pose_base_m_rad'] = matrix_pose(flange)
        source['current_palm_pose_base_m_rad'] = matrix_pose(flange @ PALM)
        source['cup_base']['support_center_m'] = [-.1640344, .4741704, .04]
        source['snapshot_sha256_color'] = 'c'*64
        source['snapshot_sha256_depth'] = 'd'*64
        axis = np.asarray(source['cup_base']['axis'])
        axis /= np.linalg.norm(axis)
        support = np.asarray(source['cup_base']['support_center_m'])
        height = source['cup_base']['dimensions']['observed_height_m']
        nominal_top = support+axis*height
        measured_top = nominal_top+axis*.002
        source['cup_base']['measured_top_surface'] = {
            'valid': True, 'source_frame_id': source['snapshot_frame_id'],
            'snapshot_sha256_color': source['snapshot_sha256_color'],
            'snapshot_sha256_depth': source['snapshot_sha256_depth'],
            'model_sha256': 'e'*64, 'center_base_m': measured_top.tolist(),
            'normal_base': axis.tolist(),
            'center_definition': 'side_axis_intersection_with_visible_top_plane',
            'quality': {'rim_center_independently_measured': False,
                        'coaxial_cup_assumption': True},
        }
        source['recognition'] = {
            'selected_instance': 0, 'model_sha256': 'e'*64,
            'red_workspace': {
                'selected_green_cap_in_red_mat': True,
                'table_support_center_inside_red_mat': True,
                'config_sha256': 'f'*64, 'selected_instance': 0,
                'source_frame_id': source['snapshot_frame_id'],
                'snapshot_sha256_color': source['snapshot_sha256_color'],
            },
        }
        feedback.update(joints_rad=q.tolist(), joints_enabled=[True]*7,
                        arm_status=0, ctrl_mode=1)
        finger_dir = (flange @ PALM)[:3, 0]
        finger_dir -= axis*float(finger_dir @ axis)
        finger_dir /= np.linalg.norm(finger_dir)
        exit_palm = measured_top+axis*(.013+.08)+finger_dir*.14
        lane = {
            'frame': 'base', 'source': 'measured_dice_roi_plus_reviewed_exit_lane',
            'reviewed': True, 'source_frame_id': source['snapshot_frame_id'],
            'snapshot_sha256_color': source['snapshot_sha256_color'],
            'snapshot_sha256_depth': source['snapshot_sha256_depth'],
            'dice_roi_center_base_m': support.tolist(),
            'dice_roi_radius_m': .04, 'minimum_roi_clearance_m': .02,
            'destination_palm_base_m': exit_palm.tolist(),
        }
        hand_source = {'kind': 'hand_feedback_read_only',
                       'host_read_time_ns': time.time_ns()}
        return source, feedback, hand_source, lane

    def test_same_frame_measured_top_and_inverted_reveal_positive_plan(self):
        source, feedback, hand_source, lane = self._positive_fixture()
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64, live_joint_feedback=feedback,
            hand_open_positions=HAND_OPEN, hand_status_source=hand_source,
            orientation_reviewed=True, provisional_opt_in=True,
            pretop_height_m=.11, reveal_exit_lane=lane)
        self.assertTrue(result['checks']['execute_ready'], result['checks'])
        self.assertEqual(result['stage_sequence'][-4:], [
            'tiny_shake_a', 'tiny_shake_b', 'reveal_lift', 'reveal_exit'])
        self.assertTrue(result['checks']['shake_mouth_support_geometry_passed'])
        self.assertTrue(result['checks']['reveal_held_cup_volume_path_screen_passed'])
        self.assertTrue(result['checks']['visible_top_measured_same_frame'])
        self.assertTrue(result['checks']['green_cap_in_red_mat_verified'])
        self.assertGreaterEqual(result['reveal_exit_lane'][
            'held_cup_mouth_altitude_from_table_m'], .075)
        self.assertGreater(result['reveal_exit_lane'][
            'held_cup_projected_distance_from_dice_roi_m'], result['reveal_exit_lane'][
            'held_cup_required_projected_distance_m'])
        self.assertFalse(result['checks']['physical_contact_verified'])
        self.assertFalse(result['checks']['cup_held_verified'])
        self.assertFalse(result['motion_sent'])

    def test_retracted_thumb_gate_rejects_the_contact_posture_that_pushed_cup(self):
        source, feedback, hand_source, lane = self._positive_fixture()
        kwargs = dict(source_side_plan_sha256='a'*64,
                      live_joint_feedback=feedback,
                      hand_status_source=hand_source,
                      orientation_reviewed=True, provisional_opt_in=True,
                      pretop_height_m=.11, reveal_exit_lane=lane,
                      require_retracted_thumb=True)
        unsafe = compute_top_grasp(source, hand_open_positions=HAND_OPEN, **kwargs)
        self.assertFalse(unsafe['checks']['top_approach_thumb_retracted'])
        self.assertFalse(unsafe['checks']['execute_ready'])
        opened = dict(HAND_OPEN, thumb_tip=0., thumb_base=0.)
        safe = compute_top_grasp(source, hand_open_positions=opened, **kwargs)
        self.assertTrue(safe['checks']['top_approach_thumb_retracted'])
        self.assertGreater(safe['checks']['top_approach_thumb_tip_clearance_min_m'],
                           safe['checks']['top_approach_thumb_clearance_threshold_m'])
        self.assertTrue(safe['checks']['execute_ready'], safe['checks'])

    def test_hidden_dice_uses_same_frame_conservative_cup_footprint(self):
        source, feedback, hand_source, lane = self._positive_fixture()
        axis = np.asarray(source['cup_base']['axis'], dtype=float)
        axis /= np.linalg.norm(axis)
        flange = np.asarray(load_model().fk(np.asarray(feedback['joints_rad'])))
        finger_dir = (flange @ PALM)[:3, 0]
        finger_dir -= axis*float(finger_dir @ axis)
        finger_dir /= np.linalg.norm(finger_dir)
        measured_top = np.asarray(source['cup_base']['measured_top_surface']['center_base_m'])
        lane.update(source='measured_cup_footprint_conservative_dice_roi',
                    dice_visibility='hidden_under_inverted_cup',
                    dice_directly_observed=False, dice_roi_radius_m=.09,
                    destination_palm_base_m=(measured_top+axis*.093+finger_dir*.18).tolist())
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64, live_joint_feedback=feedback,
            hand_open_positions=HAND_OPEN, hand_status_source=hand_source,
            orientation_reviewed=True, provisional_opt_in=True,
            pretop_height_m=.11, reveal_exit_lane=lane)
        self.assertTrue(result['checks']['execute_ready'], result['checks'])
        self.assertTrue(result['reveal_exit_lane']['cup_footprint_proxy'])
        self.assertFalse(result['reveal_exit_lane']['dice_directly_observed'])
        self.assertEqual(result['reveal_exit_lane']['dice_roi_source_kind'],
                         'measured_cup_footprint_conservative_dice_roi')
        self.assertGreater(result['reveal_exit_lane'][
            'held_cup_projected_distance_from_dice_roi_m'], result['reveal_exit_lane'][
            'held_cup_required_projected_distance_m'])
        for field, value, message in (
                ('dice_roi_center_base_m', [0., 0., 0.], 'cup mouth footprint'),
                ('dice_roi_radius_m', .05, 'cover cup radius'),
                ('dice_visibility', 'directly_visible', 'hidden, unobserved dice'),
                ('snapshot_sha256_depth', '0'*64, 'source RGB-D frame')):
            bad = copy.deepcopy(lane)
            bad[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                compute_top_grasp(
                    source, source_side_plan_sha256='a'*64,
                    live_joint_feedback=feedback,
                    hand_open_positions=HAND_OPEN,
                    hand_status_source=hand_source, orientation_reviewed=True,
                    provisional_opt_in=True, pretop_height_m=.11,
                    reveal_exit_lane=bad)

    def test_prep_clearance_keeps_prep_rotation_and_stale_hand_blocks(self):
        source, feedback, hand_source, lane = self._positive_fixture()
        q = np.asarray(feedback['joints_rad'])
        prep = q.copy()
        prep[4] += math.radians(5.)
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64,
            live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
            hand_status_source=hand_source, orientation_reviewed=True,
            provisional_opt_in=True, pretop_height_m=.11,
            prep_joints_rad=prep, reveal_exit_lane=lane)
        prep_rotation = np.asarray(result['waypoints']['prep']['T_base_palm'])[:3, :3]
        clearance_rotation = np.asarray(result['waypoints']['clearance']['T_base_palm'])[:3, :3]
        pretop_rotation = np.asarray(result['waypoints']['pretop']['T_base_palm'])[:3, :3]
        np.testing.assert_allclose(clearance_rotation, prep_rotation, atol=1e-9)
        np.testing.assert_allclose(pretop_rotation, prep_rotation, atol=1e-9)
        old_rotation = np.asarray(load_model().fk(q)) @ PALM
        self.assertGreater(np.linalg.norm(clearance_rotation-old_rotation[:3, :3]), .01)
        stale = {'kind': 'hand_feedback_read_only',
                 'host_read_time_ns': hand_source['host_read_time_ns']-int(30e9)}
        result = compute_top_grasp(
            source, source_side_plan_sha256='a'*64,
            live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
            hand_status_source=stale, orientation_reviewed=True,
            provisional_opt_in=True, pretop_height_m=.11,
            reveal_exit_lane=lane)
        self.assertFalse(result['checks']['current_hand_status_live'])
        self.assertFalse(result['checks']['execute_ready'])

    def test_visible_top_hash_and_quality_must_match_source(self):
        source, feedback, hand_source, lane = self._positive_fixture()
        source['cup_base']['measured_top_surface']['snapshot_sha256_depth'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'differs from side RGB-D source'):
            compute_top_grasp(
                source, source_side_plan_sha256='a'*64,
                live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
                hand_status_source=hand_source, orientation_reviewed=True,
                provisional_opt_in=True, pretop_height_m=.11,
                reveal_exit_lane=lane)
        source, feedback, hand_source, lane = self._positive_fixture()
        source['recognition']['red_workspace']['selected_instance'] = 1
        with self.assertRaisesRegex(ValueError, 'red mat workspace evidence disagrees'):
            compute_top_grasp(
                source, source_side_plan_sha256='a'*64,
                live_joint_feedback=feedback, hand_open_positions=HAND_OPEN,
                hand_status_source=hand_source, orientation_reviewed=True,
                provisional_opt_in=True, pretop_height_m=.11,
                reveal_exit_lane=lane)

    def test_cli_pipeline_hand_timestamp_or_saved_feedback_gate(self):
        source, feedback, hand_source, lane = self._positive_fixture()
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            side, joints, hand, exit_lane = (
                base/'side.json', base/'joints.json', base/'hand.json', base/'lane.json')
            side.write_text(json.dumps(source), encoding='utf-8')
            joints.write_text(json.dumps({'event': 'read_joints', **feedback}), encoding='utf-8')
            status = {'event': 'hand_status', 'is_ok': True,
                      'feedback_complete': False,
                      'available_feedback': ['position', 'current'],
                      'positions': HAND_OPEN,
                      'host_read_time_ns': hand_source['host_read_time_ns']}
            hand.write_text(json.dumps(status), encoding='utf-8')
            exit_lane.write_text(json.dumps(lane), encoding='utf-8')
            args = ['--side-plan', str(side), '--joint-feedback', str(joints),
                    '--hand-feedback', str(hand), '--orientation-reviewed',
                    '--reveal-exit-lane', str(exit_lane), '--pretop-height-m', '.11',
                    '--allow-provisional']
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([*args, '--output', str(base/'ready.json')]), 0)
            ready = json.loads((base/'ready.json').read_text())
            self.assertTrue(ready['checks']['current_hand_status_live'])
            self.assertTrue(ready['checks']['execute_ready'], ready['checks'])
            self.assertGreater(ready['checks']['planning_compute_duration_s'], 0.)
            del status['host_read_time_ns']
            hand.write_text(json.dumps(status), encoding='utf-8')
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([*args, '--output', str(base/'saved.json')]), 0)
            saved = json.loads((base/'saved.json').read_text())
            self.assertFalse(saved['checks']['current_hand_status_live'])
            self.assertFalse(saved['checks']['execute_ready'])


if __name__ == '__main__':
    unittest.main()
