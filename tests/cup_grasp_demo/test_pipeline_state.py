"""Regressions for scene invalidation and physical-observation gates."""

import unittest

from cup_grasp_demo.pipeline_state import (
    PipelineTransitionError, apply_event, new_state,
)


class PipelineStateTest(unittest.TestCase):
    def test_user_current_home_waits_for_can_without_repeating_old_motion(self):
        state = new_state('a'*64)
        state = apply_event(state, 'scene_clear', {
            'scene_id': 'new-red-scene', 'observed_at_ns': 1,
            'scene_image_sha256': 'b'*64, 'path_scene_sha256': 'b'*64,
        })
        designated = [0.52, -1.22, -1.76, 1.35, -1.59, -.65, -.43]
        state = apply_event(state, 'user_home_designated', {
            'user_instruction': 'current_pose_as_home',
            'first_joints_rad': designated, 'second_joints_rad': designated,
            'observed_at_ns': 2, 'scene_photo_sha256': 'c'*64,
            'arm_status': 0, 'ctrl_mode': 3, 'joints_enabled': [True]*7,
        })
        self.assertEqual(state['phase'], 'WAIT_CONTROL_HANDOFF')
        self.assertEqual(state['home_move_count'], 0)
        self.assertIsNone(state['home_arrival_time_ns'])
        self.assertEqual(state['user_designated_home_joints_rad'], designated)
        with self.assertRaisesRegex(PipelineTransitionError, 'cannot advance'):
            apply_event(state, 'rgbd_captured', {'frame_id': 'old'})
        changed = list(designated)
        changed[0] += .01
        with self.assertRaisesRegex(PipelineTransitionError, 'changed'):
            apply_event(state, 'home_control_verified', {
                'first_joints_rad': changed, 'second_joints_rad': changed,
                'observed_at_ns': 3, 'scene_photo_sha256': 'd'*64,
                'arm_status': 0, 'ctrl_mode': 1, 'joints_enabled': [True]*7,
            })
        state = apply_event(state, 'home_control_verified', {
            'first_joints_rad': designated, 'second_joints_rad': designated,
            'observed_at_ns': 3, 'scene_photo_sha256': 'd'*64,
            'arm_status': 0, 'ctrl_mode': 1, 'joints_enabled': [True]*7,
        })
        self.assertEqual(state['phase'], 'CAPTURE')
        self.assertEqual(state['home_arrival_time_ns'], 3)

    def test_scene_change_discards_bound_snapshot_plan_and_hold(self):
        original = new_state('a' * 64)
        scene = apply_event(original, 'scene_clear', {
            'scene_id': 'scene-1', 'observed_at_ns': 1,
            'scene_image_sha256': 'b' * 64,
            'path_scene_sha256': 'b' * 64,
        })
        scene = apply_event(scene, 'home_move_reached', {
            'target_joints_rad': [0] * 7, 'arrival_joints_rad': [0] * 7,
            'feedback_event': 'arm_target_reached', 'observed_at_ns': 2,
            'final_home_target': True, 'reviewed_final_home_joints_rad': [0] * 7,
            'arm_status': 0, 'ctrl_mode': 1, 'joints_enabled': [True] * 7,
        })
        scene = apply_event(scene, 'rgbd_captured', {
            'frame_id': 'rgbd-1', 'metadata_sha256': 'c' * 64,
            'color_sha256': 'd' * 64, 'depth_sha256': 'e' * 64,
            'captured_at_ns': 3,
            'target_red_roi_xyxy': [1, 2, 100, 101],
            'target_red_roi_sha256': 'r' * 64,
            'path_scene_sha256': 'b' * 64,
        })
        self.assertEqual(scene['phase'], 'LOCALIZE')
        self.assertEqual(original['phase'], 'WAIT_SCENE_CLEAR')
        paused = apply_event(scene, 'pause', {'reason': 'cup was moved',
                                                'command_may_have_been_sent': False})
        with self.assertRaisesRegex(PipelineTransitionError, 'new scene'):
            apply_event(paused, 'rgbd_captured', {'frame_id': 'rgbd-2'})
        with self.assertRaisesRegex(PipelineTransitionError, 'different scene'):
            apply_event(paused, 'reset_after_scene_change', {
                'new_scene_id': 'scene-1', 'live_arm_status': {},
                'live_camera_frame_id': 'rgbd-2',
            })
        fresh = apply_event(paused, 'reset_after_scene_change', {
            'new_scene_id': 'scene-2', 'live_arm_status': {'arm_status': 0},
            'live_camera_frame_id': 'rgbd-2',
        })
        self.assertEqual(fresh['phase'], 'WAIT_SCENE_CLEAR')
        self.assertIsNone(fresh['snapshot_frame_id'])
        self.assertIsNone(fresh['plan_sha256'])
        self.assertIsNone(fresh['target_red_roi_sha256'])
        self.assertFalse(fresh['hold_observed'])
        self.assertEqual(fresh['home_move_count'], 0)

    def test_motion_feedback_cannot_skip_contact_or_hold_observation(self):
        state = new_state('a' * 64)
        state = apply_event(state, 'scene_clear', {
            'scene_id': 's1', 'observed_at_ns': 1,
            'scene_image_sha256': 'b' * 64,
            'path_scene_sha256': 'b' * 64,
        })
        state = apply_event(state, 'home_move_reached', {
            'target_joints_rad': [0] * 7, 'arrival_joints_rad': [0] * 7,
            'feedback_event': 'arm_target_reached', 'observed_at_ns': 2,
            'final_home_target': True, 'reviewed_final_home_joints_rad': [0] * 7,
            'arm_status': 0, 'ctrl_mode': 1, 'joints_enabled': [True] * 7,
        })
        state = apply_event(state, 'rgbd_captured', {
            'frame_id': 'f1', 'metadata_sha256': 'b' * 64,
            'color_sha256': 'c' * 64, 'depth_sha256': 'd' * 64,
            'captured_at_ns': 3,
            'target_red_roi_xyxy': [1, 2, 100, 101],
            'target_red_roi_sha256': 'r' * 64,
            'path_scene_sha256': 'b' * 64,
        })
        with self.assertRaisesRegex(PipelineTransitionError, 'current RGB-D'):
            apply_event(state, 'visible_top_localized', {
                'frame_id': 'old-frame', 'top_center_camera_m': [0] * 3,
                'top_center_base_m': [0] * 3, 'top_quality': {'valid': True},
            })
        state = apply_event(state, 'visible_top_localized', {
            'frame_id': 'f1', 'top_center_camera_m': [0] * 3,
            'top_center_base_m': [0] * 3, 'top_quality': {'valid': True},
        })
        state = apply_event(state, 'top_plan_ready', {
            'frame_id': 'f1', 'plan_sha256': 'p' * 64, 'execute_ready': True,
            'reviewed_approach_sequence': ['clearance', 'pretop', 'guarded_hover'],
        })
        for stage in ('clearance', 'pretop', 'guarded_hover'):
            state = apply_event(state, 'approach_stage_reached', {
                'stage': stage, 'plan_sha256': 'p' * 64,
                'feedback_event': 'arm_target_reached',
                'reviewed_stage_sequence': ['clearance', 'pretop', 'guarded_hover'],
            })
        self.assertEqual(state['phase'], 'CONTACT')
        with self.assertRaisesRegex(PipelineTransitionError, 'cannot advance'):
            apply_event(state, 'hand_close_reached', {
                'plan_sha256': 'p' * 64, 'feedback_event': 'hand_target_reached',
                'six_hand_positions': {str(i): 18 for i in range(6)},
            })
        state = apply_event(state, 'probe_stage_reached', {
            'stage': 'probe_1', 'plan_sha256': 'p' * 64,
            'feedback_event': 'arm_target_reached',
        })
        with self.assertRaisesRegex(PipelineTransitionError, 'current verified'):
            apply_event(state, 'physical_contact_observed', {
                'observed_at_ns': 3, 'scene_image_sha256': 'b' * 64,
                'at_stage': 'guarded_hover',
            })
        state = apply_event(state, 'physical_contact_observed', {
            'observed_at_ns': 3, 'scene_image_sha256': 'b' * 64,
            'at_stage': 'probe_1',
        })
        state = apply_event(state, 'hand_close_reached', {
            'plan_sha256': 'p' * 64, 'feedback_event': 'hand_target_reached',
            'six_hand_positions': {str(i): 18 for i in range(6)},
        })
        with self.assertRaisesRegex(PipelineTransitionError, 'does not prove'):
            apply_event(state, 'cup_hold_observed', {
                'observed_at_ns': 4, 'scene_image_sha256': 'b' * 64,
                'cup_pose_before_m': [0] * 3,
                'cup_pose_after_m': [0] * 3, 'held': False,
            })
        state = apply_event(state, 'cup_hold_observed', {
            'observed_at_ns': 4, 'scene_image_sha256': 'b' * 64,
            'cup_pose_before_m': [0] * 3,
            'cup_pose_after_m': [0] * 3, 'held': True,
        })
        self.assertEqual(state['phase'], 'SHAKE')
        with self.assertRaisesRegex(PipelineTransitionError, 'held cup'):
            apply_event(state, 'tiny_shake_stage_reached', {
                'stage': 'tiny_shake_a', 'plan_sha256': 'p' * 64,
                'feedback_event': 'arm_target_reached',
                'hold_observed_at_ns': 2,
            })
        for stage in ('tiny_shake_a', 'tiny_shake_b'):
            state = apply_event(state, 'tiny_shake_stage_reached', {
                'stage': stage, 'plan_sha256': 'p' * 64,
                'feedback_event': 'arm_target_reached',
                'hold_observed_at_ns': 4,
            })
        self.assertEqual(state['phase'], 'REVEAL')
        with self.assertRaisesRegex(PipelineTransitionError, 'cannot advance'):
            apply_event(state, 'cup_revealed', {
                'plan_sha256': 'p' * 64,
                'scene_image_sha256': 'z' * 64,
                'dice_region_clear': True,
            })
        with self.assertRaisesRegex(PipelineTransitionError, 'next verified'):
            apply_event(state, 'reveal_stage_reached', {
                'stage': 'reveal_exit', 'plan_sha256': 'p' * 64,
                'feedback_event': 'arm_target_reached',
                'hold_observed_at_ns': 4, 'observed_at_ns': 6,
            })
        for observed_at, stage in ((6, 'reveal_lift'), (7, 'reveal_exit')):
            state = apply_event(state, 'reveal_stage_reached', {
                'stage': stage, 'plan_sha256': 'p' * 64,
                'feedback_event': 'arm_target_reached',
                'hold_observed_at_ns': 4, 'observed_at_ns': observed_at,
            })
        self.assertEqual(state['phase'], 'VERIFY_REVEAL')
        with self.assertRaisesRegex(PipelineTransitionError, 'post-exit observation'):
            apply_event(state, 'cup_revealed', {
                'plan_sha256': 'p' * 64,
                'scene_image_sha256': 'z' * 64,
                'dice_region_clear': True, 'hold_observed_at_ns': 4,
                'observed_at_ns': 7,
            })
        state = apply_event(state, 'cup_revealed', {
            'plan_sha256': 'p' * 64,
            'scene_image_sha256': 'z' * 64,
            'dice_region_clear': True, 'hold_observed_at_ns': 4,
            'observed_at_ns': 8,
        })
        self.assertEqual(state['phase'], 'DONE')


if __name__ == '__main__':
    unittest.main()
