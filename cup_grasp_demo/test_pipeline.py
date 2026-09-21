"""Offline regressions for opt-in single-step persistence and command gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from cup_grasp_demo import pipeline, pipeline_state


class PipelineCliTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root/'green_pipeline.json'
        self.config.write_bytes(pipeline.DEFAULT_CONFIG.read_bytes())
        self.state = self.root/'state.json'
        _, identity, _ = pipeline._load_config(self.config)
        pipeline._write_atomic(self.state, pipeline_state.new_state(identity), first=True)
        self.image = self.root/'scene.jpg'
        self.image.write_bytes(b'fresh camera observation')
        self.now = time.time_ns()

    def scene(self, scene_id='scene-a'):
        return {'scene_id': scene_id, 'observed_at_ns': self.now,
                'scene_image_path': str(self.image),
                'scene_image_sha256': hashlib.sha256(self.image.read_bytes()).hexdigest(),
                'people_clear': True, 'cup_still': True}

    def loaded(self):
        return json.loads(self.state.read_bytes())

    def test_dry_run_never_calls_api_or_advances_scene_or_home(self):
        runner = mock.Mock(side_effect=AssertionError('would call hardware'))
        preview = pipeline.run_step(self.state, self.config,
                                    evidence=self.scene(), runner=runner,
                                    now_ns=self.now)
        self.assertFalse(preview['state_advanced'])
        self.assertEqual(self.loaded()['phase'], 'WAIT_SCENE_CLEAR')
        pipeline.run_step(self.state, self.config, evidence=self.scene(),
                          execute_motion=True, runner=runner, now_ns=self.now)
        home_preview = pipeline.run_step(self.state, self.config,
                                         runner=runner, now_ns=self.now)
        self.assertEqual(home_preview['action']['home_stage_index'], 0)
        self.assertEqual(self.loaded()['home_move_count'], 0)
        runner.assert_not_called()

    def test_failed_home_latches_uncertain_command_and_cannot_retry(self):
        pipeline.run_step(self.state, self.config, evidence=self.scene(),
                          execute_motion=True, now_ns=self.now)
        runner = mock.Mock(side_effect=AssertionError('API should stay closed'))
        with self.assertRaisesRegex(ValueError, 'positive host ns'):
            pipeline.run_step(self.state, self.config, evidence={},
                              execute_motion=True, runner=runner, now_ns=self.now)
        paused = self.loaded()
        self.assertEqual(paused['phase'], 'PAUSED')
        self.assertTrue(paused['command_may_have_been_sent'])
        with self.assertRaisesRegex(ValueError, 'PAUSED cannot step'):
            pipeline.run_step(self.state, self.config,
                              execute_motion=True, runner=runner, now_ns=self.now)
        runner.assert_not_called()
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_scene_image_mismatch_blocks_formal_start(self):
        observation = self.scene()
        observation['scene_image_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'match'):
            pipeline.run_step(self.state, self.config, evidence=observation,
                              execute_motion=True, now_ns=self.now)
        self.assertEqual(self.loaded()['phase'], 'PAUSED')
        self.assertIsNone(self.loaded()['scene_id'])

    def test_final_home_arrival_gate_precedes_formal_capture(self):
        pipeline.run_step(self.state, self.config, evidence=self.scene(),
                          execute_motion=True, now_ns=self.now)
        state = self.loaded()
        config, _, _ = pipeline._load_config(self.config)
        waypoints = config['home_candidate']['waypoints_deg']
        for target in waypoints[:-1]:
            radians = [v*3.141592653589793/180 for v in target]
            state = pipeline_state.apply_event(state, 'home_move_reached', {
                'target_joints_rad': radians, 'arrival_joints_rad': radians,
                'feedback_event': 'arm_target_reached',
                'observed_at_ns': self.now+1})
        pipeline._write_atomic(self.state, state)
        with mock.patch.object(pipeline, '_home_motion') as motion:
            radians = [v*3.141592653589793/180 for v in waypoints[-1]]
            motion.return_value = {
                'target_joints_rad': radians,
                'arrival_joints_rad': radians,
                'feedback_event': 'arm_target_reached',
                'observed_at_ns': self.now+2,
                'final_home_target': True,
                'reviewed_final_home_joints_rad': radians,
                'arm_status': 0, 'ctrl_mode': 1,
                'joints_enabled': [True]*7,
            }
            result = pipeline.run_step(self.state, self.config,
                                       execute_motion=True, now_ns=self.now+3)
        self.assertEqual(result['phase_after'], 'CAPTURE')
        self.assertEqual(self.loaded()['home_arrival_joints_rad'], radians)
        with mock.patch.object(pipeline, '_snapshot') as capture:
            capture.return_value = {'frame_id': 'diagnostic-before-home',
                                    'metadata_sha256': 'a'*64,
                                    'color_sha256': 'b'*64,
                                    'depth_sha256': 'c'*64,
                                    'captured_at_ns': self.now+1,
                                    'target_red_roi_xyxy': [10, 10, 80, 80],
                                    'target_red_roi_sha256': 'd'*64,
                                    'path_scene_sha256': 'e'*64}
            with self.assertRaisesRegex(ValueError, 'Formal RGB-D capture'):
                pipeline.run_step(self.state, self.config,
                                  execute_motion=True, now_ns=self.now+4)
        self.assertEqual(self.loaded()['phase'], 'PAUSED')
        self.assertIsNone(self.loaded()['snapshot_frame_id'])

    def test_top_command_interruption_persists_pause_before_next_attempt(self):
        _, identity, _ = pipeline._load_config(self.config)
        state = pipeline_state.new_state(identity)
        state['phase'] = 'APPROACH'
        state['scene_id'] = 'scene-a'
        state['snapshot_frame_id'] = 'current-rgbd'
        state['plan_sha256'] = 'a'*64
        state['reviewed_approach_sequence'] = ['clearance']
        pipeline._write_atomic(self.state, state)
        evidence = {**self.scene(), 'frame_id': 'current-rgbd',
                    'plan_sha256': 'a'*64, 'scene_observed': True,
                    'path_obstacles_clear': True}
        with mock.patch.object(pipeline, '_top_motion',
                               side_effect=RuntimeError('interrupted after dispatch')) as motion:
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                pipeline.run_step(self.state, self.config, evidence=evidence,
                                  execute_motion=True, now_ns=self.now)
            self.assertEqual(motion.call_count, 1)
            with self.assertRaisesRegex(ValueError, 'PAUSED cannot step'):
                pipeline.run_step(self.state, self.config, evidence=evidence,
                                  execute_motion=True, now_ns=self.now)
            self.assertEqual(motion.call_count, 1)
        self.assertTrue(self.loaded()['command_may_have_been_sent'])

    def test_stale_top_photo_rejects_before_inflight_marker(self):
        _, identity, _ = pipeline._load_config(self.config)
        state = pipeline_state.new_state(identity)
        state.update(phase='APPROACH', scene_id='scene-a',
                     snapshot_frame_id='current-rgbd', plan_sha256='a'*64,
                     reviewed_approach_sequence=['clearance'])
        pipeline._write_atomic(self.state, state)
        evidence = {**self.scene(), 'frame_id': 'current-rgbd',
                    'plan_sha256': 'a'*64, 'scene_observed': True,
                    'path_obstacles_clear': True,
                    'observed_at_ns': self.now-30_000_000_000}
        with mock.patch.object(pipeline, '_top_motion') as motion:
            with self.assertRaisesRegex(ValueError, 'stale'):
                pipeline.run_step(self.state, self.config, evidence=evidence,
                                  execute_motion=True, now_ns=self.now)
            motion.assert_not_called()
        self.assertEqual(self.loaded()['phase'], 'APPROACH')
        self.assertFalse(self.loaded()['command_may_have_been_sent'])

    def test_stale_read_only_review_keeps_localize_and_plan_retryable(self):
        _, identity, _ = pipeline._load_config(self.config)
        for phase in ('LOCALIZE', 'PLAN'):
            state = pipeline_state.new_state(identity)
            state.update(phase=phase, scene_id='scene-a')
            pipeline._write_atomic(self.state, state)
            evidence = {**self.scene(),
                        'observed_at_ns': self.now-30_000_000_000}
            runner = mock.Mock(side_effect=AssertionError('adapter must stay closed'))
            with self.assertRaisesRegex(ValueError, 'stale'):
                pipeline.run_step(self.state, self.config, evidence=evidence,
                                  execute_motion=True, runner=runner, now_ns=self.now)
            runner.assert_not_called()
            self.assertEqual(self.loaded()['phase'], phase)
            self.assertFalse(self.loaded()['command_may_have_been_sent'])

    def test_top_executor_uses_vision_python_and_keeps_sdk_python_for_can(self):
        config, _, _ = pipeline._load_config(self.config)
        state = {'phase': 'APPROACH', 'approach_completed': [],
                 'reviewed_approach_sequence': ['clearance'],
                 'scene_id': 'scene-a', 'snapshot_frame_id': 'frame-a',
                 'plan_sha256': 'a'*64}
        evidence = {'scene_id': 'scene-a', 'frame_id': 'frame-a',
                    'plan_sha256': 'a'*64, 'scene_observed': True,
                    'path_obstacles_clear': True}
        calls = []

        def refused(args, **_):
            calls.append(args)
            return subprocess.CompletedProcess(args, 2, '', 'mocked preflight failure')

        with mock.patch.object(pipeline, '_scene_evidence',
                               return_value={'scene_id': 'scene-a'}):
            with self.assertRaisesRegex(RuntimeError, 'mocked preflight failure'):
                pipeline._top_motion(config, self.root, state, evidence,
                                     python='/sdk/python',
                                     camera_python='/vision/python',
                                     runner=refused, now_ns=self.now)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:3], ['/vision/python', '-m',
                                       'cup_grasp_demo.top_execute'])
        self.assertEqual(calls[0][calls[0].index('--python')+1], '/sdk/python')

    def test_red_target_identity_ignores_blue_mat_change(self):
        import cv2
        import numpy as np

        color = np.zeros((480, 640, 3), dtype=np.uint8)
        color[150:400, 300:600] = (0, 0, 255)
        color[80:350, 20:250] = (255, 0, 0)
        first = self.root/'first.png'
        second = self.root/'second.png'
        self.assertTrue(cv2.imwrite(str(first), color))
        roi, red_identity = pipeline._red_mat_roi(first)
        color[100:250, 50:200] = (0, 255, 0)
        self.assertTrue(cv2.imwrite(str(second), color))
        _, unchanged_red = pipeline._red_mat_roi(second, roi)
        self.assertEqual(red_identity, unchanged_red)
        self.assertEqual(roi, [300, 150, 600, 400])
        color[200:260, 400:450] = (0, 0, 100)
        self.assertTrue(cv2.imwrite(str(second), color))
        _, changed_red = pipeline._red_mat_roi(second, roi)
        self.assertNotEqual(red_identity, changed_red)

    def test_polygon_red_workspace_hash_ignores_blue_pixels_inside_polygon(self):
        import cv2
        import numpy as np

        workspace = (pipeline.DEFAULT_CONFIG.parent.parent.parent/
                     'dice_cup_localization/config/red_mat_640x480.json')
        color = np.zeros((480, 640, 3), dtype=np.uint8)
        color[140:400, 280:520] = (0, 0, 255)
        color[140:400, 50:220] = (255, 0, 0)
        first = self.root/'workspace_first.png'
        second = self.root/'workspace_second.png'
        self.assertTrue(cv2.imwrite(str(first), color))
        bbox, identity = pipeline._red_mat_roi(first,
                        workspace_config_path=workspace)
        color[180:300, 80:180] = (0, 255, 0)
        self.assertTrue(cv2.imwrite(str(second), color))
        second_bbox, second_identity = pipeline._red_mat_roi(second,
                        workspace_config_path=workspace)
        self.assertEqual(bbox, second_bbox)
        self.assertEqual(identity, second_identity)

    def test_current_frame_plan_review_can_be_supplied_without_config_change(self):
        config, identity, _ = pipeline._load_config(self.config)
        original_config_sha = pipeline._sha256(self.config)
        state = pipeline_state.new_state(identity)
        state['scene_id'] = 'reviewed-scene'
        state['snapshot_frame_id'] = 'formal-rgbd-after-home'
        base = self.root
        snapshot = pipeline._relative(base, config['snapshot_dir'], state['scene_id'])
        snapshot.mkdir(parents=True)
        (snapshot/'color.png').write_bytes(b'formal color')
        (snapshot/'depth.npz').write_bytes(b'formal depth')
        orientation = self.root/'palm_orientation.json'
        orientation.write_text(json.dumps([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        lane = self.root/'reviewed_exit_lane.json'
        lane.write_text(json.dumps({
            'source': 'measured_cup_footprint_conservative_dice_roi',
            'reviewed': True, 'source_frame_id': state['snapshot_frame_id'],
            'snapshot_sha256_color': pipeline._sha256(snapshot/'color.png'),
            'snapshot_sha256_depth': pipeline._sha256(snapshot/'depth.npz'),
            'dice_visibility': 'hidden_under_inverted_cup',
            'dice_directly_observed': False,
        }))
        review = {'snapshot_frame_id': state['snapshot_frame_id'],
                  'palm_orientation_path': str(orientation),
                  'palm_orientation_sha256': pipeline._sha256(orientation),
                  'orientation_reviewed': True,
                  'reveal_exit_lane_path': str(lane),
                  'reveal_exit_lane_sha256': pipeline._sha256(lane),
                  'exit_lane_reviewed': True,
                  'pretop_height_m': .11,
                  'pretop_height_reviewed': True}
        self.assertEqual(pipeline._reviewed_plan_inputs(
            config, base, state, review), (orientation, lane, .11, .08))
        self.assertEqual(pipeline._sha256(self.config), original_config_sha)
        changed_frame = dict(review, snapshot_frame_id='old-diagnostic')
        with self.assertRaisesRegex(ValueError, 'Current-frame'):
            pipeline._reviewed_plan_inputs(config, base, state, changed_frame)
        lane.write_text(lane.read_text().replace('false', 'true'))
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            pipeline._reviewed_plan_inputs(config, base, state, review)

    def test_current_pose_home_waits_for_web_can_handoff_without_motion(self):
        q = [math.radians(v) for v in
             [30, -70, -101, 77, -91, -37, -24]]
        calls = []
        current_mode = [3]

        def reads(command, **_):
            action = command[-1]
            self.assertIn(action, ('status', 'read-joints'))
            calls.append(action)
            if action == 'status':
                mode_label = ('CAN_CTRL' if current_mode[0] == 1
                              else 'ETHERNET_CONTROL_MODE')
                event = {'event': 'arm_status',
                         'status': f'ctrl_mode: {mode_label}(0x{current_mode[0]:x})\n'
                                   'arm_status: NORMAL(0x0)',
                         'joints_enabled': [True]*7}
            else:
                event = {'event': 'read_joints', 'joints_rad': q,
                         'arm_status': 0, 'ctrl_mode': current_mode[0],
                         'joints_enabled': [True]*7}
            return subprocess.CompletedProcess(command, 0, json.dumps(event)+'\n', '')

        external = self.scene('current-user-home')
        external.update(red_cup_on_red_mat_observed=True,
                        path_obstacles_clear=True)
        first = pipeline.adopt_current_home(
            self.state, self.config, external, runner=reads, now_ns=self.now,
            sleep_fn=lambda _: None, clock_fn=lambda: self.now+1)
        self.assertEqual(first['phase'], 'WAIT_CONTROL_HANDOFF')
        waiting = self.loaded()
        self.assertIsNone(waiting['home_arrival_time_ns'])
        self.assertEqual(waiting['home_move_count'], 0)
        self.assertEqual(waiting['user_designated_home_joints_rad'], q)
        self.assertFalse(first['CAN_command_sent'])
        self.assertEqual(calls, ['status', 'read-joints']*2)
        with self.assertRaisesRegex(ValueError, 'WEB handoff'):
            pipeline.run_step(self.state, self.config, execute_motion=True,
                              runner=mock.Mock(side_effect=AssertionError('CAN TX')),
                              now_ns=self.now)
        current_mode[0] = 1
        external['observed_at_ns'] = self.now+2
        second = pipeline.adopt_current_home(
            self.state, self.config, external, runner=reads, now_ns=self.now+2,
            sleep_fn=lambda _: None, clock_fn=lambda: self.now+3)
        self.assertEqual(second['phase'], 'CAPTURE')
        self.assertFalse(second['CAN_command_sent'])
        self.assertEqual(self.loaded()['home_arrival_joints_rad'], q)
        self.assertEqual(self.loaded()['home_source'],
                         'user_designated_current_pose_read_only')
        self.assertEqual(calls, ['status', 'read-joints']*4)

    def test_current_pose_home_rejects_missing_red_mat_or_moving_joints(self):
        evidence = self.scene('user-home')
        evidence['path_obstacles_clear'] = True
        with self.assertRaisesRegex(ValueError, 'red cup'):
            pipeline.adopt_current_home(self.state, self.config, evidence,
                                        runner=mock.Mock(side_effect=AssertionError('read')),
                                        now_ns=self.now)
        self.assertEqual(self.loaded()['phase'], 'WAIT_SCENE_CLEAR')
        evidence['red_cup_on_red_mat_observed'] = True
        read_count = [0]

        def moving_read(command, **_):
            if command[-1] == 'status':
                event = {'event': 'arm_status', 'joints_enabled': [True]*7,
                         'status': 'ctrl_mode: CAN_CTRL(0x1)\narm_status: NORMAL(0x0)'}
            elif command[-1] == 'read-joints':
                read_count[0] += 1
                event = {'event': 'read_joints',
                         'joints_rad': [math.radians(read_count[0]*.3), 0, 0, 0, 0, 0, 0],
                         'ctrl_mode': 1, 'arm_status': 0,
                         'joints_enabled': [True]*7}
            else:
                raise AssertionError('Adoption must not send motion CAN')
            return subprocess.CompletedProcess(command, 0, json.dumps(event)+'\n', '')

        with self.assertRaisesRegex(ValueError, 'double feedback changed'):
            pipeline.adopt_current_home(
                self.state, self.config, evidence, runner=moving_read,
                now_ns=self.now, sleep_fn=lambda _: None,
                clock_fn=lambda: self.now+1)
        self.assertEqual(self.loaded()['phase'], 'WAIT_SCENE_CLEAR')

    def test_adopt_existing_home_and_formal_capture_uses_only_live_reads(self):
        import cv2
        import numpy as np

        config_folder = self.root/'config'
        config_folder.mkdir()
        config_path = config_folder/'green_pipeline.json'
        config = json.loads(pipeline.DEFAULT_CONFIG.read_bytes())
        workspace = (pipeline.DEFAULT_CONFIG.parent.parent.parent/
                     'dice_cup_localization/config/red_mat_640x480.json')
        config['red_workspace'] = str(workspace)
        config_path.write_text(json.dumps(config))
        _, identity, _ = pipeline._load_config(config_path)
        state_path = self.root/'existing_state.json'
        pipeline._write_atomic(state_path, pipeline_state.new_state(identity), first=True)
        source = self.root/'formal_snapshot'
        source.mkdir()
        image = np.zeros((480, 640, 3), np.uint8)
        image[145:440, 250:550] = (0, 0, 255)
        self.assertTrue(cv2.imwrite(str(source/'color.png'), image))
        (source/'depth.npz').write_bytes(b'formal depth evidence')
        t0 = self.now-15_000_000_000
        ways = config['home_candidate']['waypoints_deg']
        scene_photo = self.root/'before.jpg'
        scene_photo.write_bytes(b'clear scene before actual motions')
        os.utime(scene_photo, ns=(t0, t0))
        scene = {'scene_id': 'adopted-red-scene', 'observed_at_ns': t0,
                 'photo_path': str(scene_photo),
                 'photo_sha256': pipeline._sha256(scene_photo),
                 'people_clear': True, 'cup_still': True,
                 'path_obstacles_clear': True}
        steps = []
        for i, joints in enumerate(ways, 1):
            stdout = self.root/f'home_{i:02d}_stdout.txt'
            stdout.write_text('\n'.join(json.dumps(e) for e in (
                {'event': 'arm_plan', 'command': 'move-j',
                 'requested_target_deg_transcribed': joints,
                 'speed_percent': 1, 'execute': True},
                {'event': 'arm_target_reached', 'command': 'move-j',
                 'fresh_samples': 10}))+'\n')
            photo = self.root/f'home_{i:02d}.jpg'
            photo.write_bytes(f'actual onsite stage {i}'.encode())
            when = t0+i*1_000_000_000
            os.utime(photo, ns=(when, when))
            steps.append({'index': i, 'command_stdout_path': str(stdout),
                          'stdout_sha256': pipeline._sha256(stdout),
                          'stdout_provenance':
                          'post_run_transcription_from_Codex_tool_result',
                          'arrival_joints_deg': joints,
                          'observed_at_ns': when, 'photo_path': str(photo),
                          'photo_sha256': pipeline._sha256(photo)})
        metadata = {'schema': 1, 'serial': config['camera_serial'],
                    'frame_id': 'formal-after-home',
                    'host_capture_time_ns': steps[-1]['observed_at_ns']+1_000_000_000,
                    'sha256_color': pipeline._sha256(source/'color.png'),
                    'sha256_depth': pipeline._sha256(source/'depth.npz')}
        (source/'metadata.json').write_text(json.dumps(metadata))
        manifest = {'schema': 1, 'kind': 'read_only_existing_home_evidence',
                    'config_sha256': identity, 'source_thread_id': 'existing-task',
                    'scene': scene, 'steps': steps,
                    'formal_snapshot_dir': str(source),
                    'capture_scene': {'photo_path': steps[-1]['photo_path'],
                                      'photo_sha256': steps[-1]['photo_sha256'],
                                      'people_clear': True,
                                      'path_obstacles_clear': True}}
        manifest_path = self.root/'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        q_final = [math.radians(v) for v in ways[-1]]
        calls = []

        def read_only_control(command, **_):
            calls.append(command)
            if command[-1] == 'status':
                event = {'event': 'arm_status',
                         'status': 'ctrl_mode: CAN_CTRL(0x1)\narm_status: NORMAL(0x0)',
                         'joints_enabled': [True]*7,
                         'joints_rad': q_final,
                         'flange_m_rad': [0]*6}
            elif command[-1] == 'read-joints':
                event = {'event': 'read_joints', 'joints_rad': q_final,
                         'sdk_limits_deg': [[-180, 180]]*7,
                         'arm_status': 0, 'ctrl_mode': 1,
                         'joints_enabled': [True]*7}
            else:
                raise AssertionError('Existing HOME import must not dispatch CAN TX')
            return subprocess.CompletedProcess(command, 0, json.dumps(event)+'\n', '')

        adopted = pipeline.record_existing_home(state_path, config_path,
                    manifest_path, None, runner=read_only_control)
        final = json.loads(state_path.read_bytes())
        self.assertEqual(adopted['phase'], 'LOCALIZE')
        self.assertFalse(adopted['CAN_command_sent'])
        self.assertEqual(final['home_move_count'], 9)
        self.assertEqual(final['snapshot_frame_id'], 'formal-after-home')
        self.assertEqual([command[-1] for command in calls], ['status', 'read-joints'])
        destination = pipeline._relative(self.root, config['snapshot_dir'],
                                         scene['scene_id'])
        self.assertEqual((destination/'color.png').read_bytes(),
                         (source/'color.png').read_bytes())


if __name__ == '__main__':
    unittest.main()
