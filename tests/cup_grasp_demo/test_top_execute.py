"""Offline fake-driver regressions for the opt-in top-grasp stage executor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from cup_grasp_demo import execute, hand_geometry, top_execute, top_grasp
from nero_calibration.core import PALM, matrix_pose
from nero_revo2_control.kinematics import load_model


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FakeControl:
    """No CAN device is opened; return the existing demo JSON event contract."""

    def __init__(self, q_before, q_after, sdk, hand, *, ctrl=1, arm=0,
                 enabled=None, omit_arrival=False):
        self.model = load_model()
        self.q = list(q_before)
        self.q_after = list(q_after)
        self.sdk = sdk
        self.hand = dict(hand)
        self.ctrl, self.arm = ctrl, arm
        self.enabled = [True]*7 if enabled is None else enabled
        self.omit_arrival = omit_arrival
        self.calls = []

    def __call__(self, command, **_):
        self.calls.append(command)
        kind = 'status' if command[-1] == 'status' else (
            'read-joints' if command[-1] == 'read-joints' else (
                'hand-status' if command[-1] == 'hand-status' else (
                    'move-j' if 'move-j' in command else 'hand')))
        if kind == 'status':
            mode = 'CAN_CTRL' if self.ctrl == 1 else 'ETHERNET_CONTROL_MODE'
            health = 'NORMAL' if self.arm == 0 else 'JOINT_BRAKE_NOT_RELEASED'
            event = {'event': 'arm_status',
                     'status': f'ctrl_mode: {mode}(0x{self.ctrl:x})\narm_status: {health}(0x{self.arm:x})',
                     'joints_enabled': self.enabled,
                     'joints_rad': self.q,
                     'flange_m_rad': matrix_pose(self.model.fk(self.q))}
        elif kind == 'read-joints':
            event = {'event': 'read_joints', 'joints_rad': self.q,
                     'sdk_limits_deg': self.sdk, 'ctrl_mode': self.ctrl,
                     'arm_status': self.arm, 'joints_enabled': self.enabled}
        elif kind == 'hand-status':
            event = {'event': 'hand_status', 'is_ok': True,
                     'positions': self.hand}
        elif kind == 'move-j':
            self.q = self.q_after
            event = ({'event': 'arm_progress'} if self.omit_arrival else
                     {'event': 'arm_target_reached', 'command': 'move-j',
                      'fresh_samples': 10})
        else:
            if '--positions' in command:
                start = command.index('--positions')+1
                self.hand = {name: int(value) for name, value in zip(
                    top_execute.HAND_KEYS, command[start:start+6])}
            else:
                for index, token in enumerate(command):
                    if token != '--set':
                        continue
                    key, value = command[index+1].split('=', 1)
                    key = {'index': 'index_finger', 'middle': 'middle_finger',
                           'ring': 'ring_finger', 'pinky': 'pinky_finger'}.get(key, key)
                    self.hand[key] = int(value)
            event = {'event': 'hand_target_reached', 'fresh_samples': 3,
                     'positions': self.hand}
        return subprocess.CompletedProcess(command, 0, json.dumps(event)+'\n', '')


class TopExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now_ns = time.time_ns()
        self.snapshot = self.root/'snapshot'
        self.snapshot.mkdir()
        (self.snapshot/'color.png').write_bytes(b'color')
        (self.snapshot/'depth.npz').write_bytes(b'depth')
        self.metadata = {'schema': 1, 'camera_backend': 'realsense',
                         'frame': 'color_optical', 'depth_registered_to': 'color_optical',
                         'serial': 'camera-1', 'frame_id': 'camera-1:42',
                         'host_capture_time_ns': self.now_ns,
                         'timestamp_ms': 1000., 'depth_timestamp_ms': 1001.,
                         'timestamp_domain': 'hardware_clock',
                         'depth_timestamp_domain': 'hardware_clock',
                         'sha256_color': sha(b'color'), 'sha256_depth': sha(b'depth')}
        self._write_metadata()
        self.calibration = self.root/'calibration.json'
        self.calibration.write_text(json.dumps({'schema': 1, 'mode': 'eye_to_hand',
                                                'quality_passed': False, 'tcp': 'palm',
                                                'T_flange_tcp': np.eye(4).tolist()}))
        self.source = self.root/'side_plan.json'
        self.source.write_text(json.dumps({'schema': 1,
                                           'kind': 'read_only_side_grasp_proposal',
                                           'snapshot_frame_id': 'camera-1:42',
                                           'localization': {'valid': True,
                                                            'camera_serial': 'camera-1',
                                                            'frame_id': 'camera-1:42'},
                                           'calibration_sha256': sha(self.calibration.read_bytes())}))
        self.plan_path = self.root/'top_plan.json'
        self.receipt = self.root/'receipt.json'
        self.model = load_model()
        self.q0 = np.radians([55., -78., 80., -45., 130., -30., 40.])
        self.sdk = [[-155., 155.], [-100., 100.], [-158., 158.],
                    [-58., 123.], [-158., 158.], [-42., 55.], [-90., 90.]]
        self.hand = {'thumb_tip': 40, 'thumb_base': 40,
                     'index_finger': 0, 'middle_finger': 0,
                     'ring_finger': 0, 'pinky_finger': 0}
        self.deltas = {'clearance': .1, 'pretop': .2, 'guarded_hover': .4,
                       'probe_1': 1.0, 'probe_2': 1.6, 'contact_candidate': 2.2,
                       'tiny_shake_a': .2, 'tiny_shake_b': -.2}
        waypoints = {}
        for name, delta in self.deltas.items():
            q = self.target_q(name)
            goal = np.asarray(self.model.fk(q))
            waypoints[name] = {
                'T_base_flange': goal.tolist(), 'T_base_palm': goal.tolist(),
                'flange_pose_base_m_rad': matrix_pose(goal),
                'target_joints_rad': list(q),
                'ik': {'success': True}, 'joint_path': {'kinematic_checks_passed': True},
                'hand_geometry': {'screen_passed': True,
                                  'actual_hand_posture_verified': False},
            }
        hover = np.asarray(waypoints['guarded_hover']['T_base_palm'])[:3, 3]
        first_probe = np.asarray(waypoints['probe_1']['T_base_palm'])[:3, 3]
        axis = (hover-first_probe)/np.linalg.norm(hover-first_probe)
        candidate = np.asarray(waypoints['contact_candidate']['T_base_palm'])[:3, 3]
        top = candidate-axis*.013
        support = top-axis*.072
        self.plan = {
            'schema': 1, 'kind': 'read_only_top_grasp_proposal', 'units': 'm_rad',
            'source_side_plan_sha256': sha(self.source.read_bytes()),
            'calibration_sha256': sha(self.calibration.read_bytes()),
            'snapshot_frame_id': 'camera-1:42',
            'cup_base': {'frame': 'base', 'support_center_m': support.tolist(),
                         'axis': axis.tolist(),
                         'dimensions': {'observed_side_diameter_m': .075,
                                        'observed_height_m': .072},
                         'source': {'frame_id': 'camera-1:42'}},
            'cup_top_center_base_m': top.tolist(),
            'cup_top_provenance': {
                'visible_top_measured': True,
                'source_frame_id': 'camera-1:42',
                'snapshot_sha256_color': self.metadata['sha256_color'],
                'snapshot_sha256_depth': self.metadata['sha256_depth'],
                'model_sha256': 'a'*64,
            },
            'T_flange_palm': np.eye(4).tolist(),
            'stage_sequence': list(self.deltas),
            'current_joints_rad': list(self.q0),
            'current_flange_pose_base_m_rad': matrix_pose(self.model.fk(self.q0)),
            'pose_source': {'kind': 'can_feedback_read_only', 'channel': 'can0',
                            'host_read_time_ns': self.now_ns,
                            'sdk_limits_deg': self.sdk},
            'checks': {'execute_ready': True, 'source_pose_live_can': True,
                       'calibration_quality_passed': False,
                       'provisional_opt_in': True,
                       'source_snapshot_fresh': True,
                       'current_arm_status_normal': True,
                       'current_control_mode_can': True,
                       'current_all_joints_enabled': True,
                       'current_hand_status_live': True,
                       'visible_top_measured_same_frame': True,
                       'green_cap_in_red_mat_verified': True,
                       'all_stage_hand_geometry_screen_passed': True,
                       'measured_dice_exit_lane_reviewed': True,
                       'reveal_held_cup_volume_path_screen_passed': True,
                       'palm_top_facing': True,
                       'orientation_reviewed': True,
                       'all_ik_passed': True,
                       'all_joint_paths_kinematically_passed': True,
                       'all_stage_joint_limits_margin_1deg_passed': True,
                       'approach_hand_geometry_screen_passed': True,
                       'shake_mouth_support_geometry_passed': True,
                       'physical_contact_verified': False},
            'hand_open_positions': self.hand,
            'hand_close_updates': {**self.hand, 'thumb_tip': 45,
                                   'thumb_base': 43, 'index_finger': 20,
                                   'middle_finger': 20, 'ring_finger': 15,
                                   'pinky_finger': 15},
            'provenance': {'uncertainty_envelope_m': .013,
                           'nominal_palm_STL_front_of_TCP_m': .012,
                           'hand_collision_model': hand_geometry.RightRevo2Model().provenance},
            'waypoints': waypoints,
        }
        self.plan['cup_base']['measured_top_surface'] = {
            'valid': True, 'source_frame_id': 'camera-1:42',
            'snapshot_sha256_color': self.metadata['sha256_color'],
            'snapshot_sha256_depth': self.metadata['sha256_depth'],
            'model_sha256': 'a'*64,
            'center_definition': 'side_axis_intersection_with_visible_top_plane',
            'center_base_m': top.tolist(), 'normal_base': axis.tolist(),
            'quality': {'rim_center_independently_measured': False,
                        'coaxial_cup_assumption': True},
        }
        for name in ('tiny_shake_a', 'tiny_shake_b'):
            self.plan['waypoints'][name]['held_inverted_cup'] = {
                'screen_passed': True, 'actual_mouth_support_verified': False}
        source_content = json.loads(self.source.read_text())
        source_content['snapshot_sha256_color'] = self.metadata['sha256_color']
        source_content['snapshot_sha256_depth'] = self.metadata['sha256_depth']
        source_content['recognition'] = {
            'selected_instance': 0, 'model_sha256': 'a'*64,
            'red_workspace': {
                'selected_green_cap_in_red_mat': True,
                'table_support_center_inside_red_mat': True,
                'config_sha256': 'f'*64, 'selected_instance': 0,
                'source_frame_id': 'camera-1:42',
                'snapshot_sha256_color': self.metadata['sha256_color']}}
        self.plan['target_workspace_provenance'] = {
            'verified': True,
            **source_content['recognition']['red_workspace']}
        source_content['cup_base'] = self.plan['cup_base']
        source_content['current_flange_pose_base_m_rad'] = self.plan['current_flange_pose_base_m_rad']
        self.source.write_text(json.dumps(source_content))
        self.source_original_bytes = self.source.read_bytes()
        self.plan['source_side_plan_sha256'] = sha(self.source_original_bytes)
        self._write_plan()
        self.real_joint_screen = top_execute._geometry_screen
        geometry = mock.patch.object(top_execute, '_geometry_screen',
                                     return_value={'table_margin_passed': True,
                                                   'source': 'offline_fake_geometry'})
        geometry.start()
        self.addCleanup(geometry.stop)
        self.real_close_screen = top_execute._close_geometry_screen
        close_geometry = mock.patch.object(top_execute, '_close_geometry_screen',
                                           return_value={'table_margin_passed': True,
                                                         'source': 'offline_fake_close_screen'})
        close_geometry.start()
        self.addCleanup(close_geometry.stop)

    def _write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def _write_metadata(self):
        (self.snapshot/'metadata.json').write_text(json.dumps(self.metadata))

    def target_q(self, stage):
        q = self.q0.copy()
        if stage in ('tiny_shake_a', 'tiny_shake_b'):
            q[1] -= math.radians(self.deltas['contact_candidate'])
            q[0] += math.radians(self.deltas[stage])
        else:
            q[1] -= math.radians(self.deltas[stage])
        return tuple(float(x) for x in q)

    def args(self, stage='clearance', **values):
        fields = dict(stage=stage, plan=self.plan_path, source_side_plan=self.source,
                      snapshot_dir=self.snapshot, calibration=self.calibration,
                      receipt=self.receipt, python='/fake/python', channel='can0',
                      allow_provisional=True, scene_observed=False,
                      contact_observed=False, hold_observed=False,
                      mouth_supported_observed=False,
                      execute=False, move_timeout=90.)
        fields.update(values)
        return argparse.Namespace(**fields)

    def _fake(self, stage, prior):
        before = (self.q0 if prior is None else
                  self.target_q((json.loads(self.receipt.read_bytes()).get('contact_observed_at_stage')
                                 or 'contact_candidate') if prior == 'close' else prior))
        after = before if stage == 'close' else self.target_q(stage)
        hand = (self.closed_hand() if prior == 'close' or prior in
                ('tiny_shake_a', 'tiny_shake_b', 'reveal_lift') else self.hand)
        return FakeControl(before, after, self.sdk, hand)

    def closed_hand(self):
        return {**self.hand, 'thumb_tip': 45, 'thumb_base': 43,
                'index_finger': 20, 'middle_finger': 20,
                'ring_finger': 15, 'pinky_finger': 15}

    def _advance(self, stage, prior, **flags):
        defaults = {'execute': True, 'scene_observed': True}
        defaults.update(flags)
        fake = self._fake(stage, prior)
        result = top_execute.run_stage(self.args(stage, **defaults),
                                       runner=fake, now_ns=time.time_ns())
        self.assertEqual(result['event'], 'top_stage_verified')
        return result, fake

    def _to_hover(self):
        previous = None
        for name in ('clearance', 'pretop', 'guarded_hover'):
            self._advance(name, previous)
            previous = name

    def _to_candidate(self):
        self._to_hover()
        previous = 'guarded_hover'
        for name in ('probe_1', 'probe_2', 'contact_candidate'):
            self._advance(name, previous)
            previous = name

    def test_speed_opt_in_keeps_default_and_contact_probe_at_one_percent(self):
        default = top_execute._command(['/fake/python'], 'clearance',
                                       {'clearance': self.q0}, None, 90.)
        faster = top_execute._command(['/fake/python'], 'clearance',
                                      {'clearance': self.q0}, None, 90., 3)
        self.assertEqual(default[default.index('--speed')+1], '1')
        self.assertEqual(faster[faster.index('--speed')+1], '3')
        self._to_hover()
        fake = self._fake('probe_1', 'guarded_hover')
        with self.assertRaisesRegex(ValueError, 'contact approach stays at 1%'):
            top_execute.run_stage(self.args('probe_1', execute=True,
                                            scene_observed=True, speed_percent=3),
                                  runner=fake, now_ns=time.time_ns())
        self.assertFalse(any('move-j' in command for command in fake.calls))

    def test_dry_run_does_not_connect_or_create_receipt(self):
        fake = self._fake('clearance', None)
        result = top_execute.run_stage(self.args(), runner=fake, now_ns=self.now_ns)
        self.assertEqual(fake.calls, [])
        self.assertFalse(result['execute'])
        self.assertFalse(self.receipt.exists())
        self.assertAlmostEqual(result['expected_target_joints_deg'][1], -78.1)

    def test_live_web_brake_disabled_or_sdk_overflow_refuses_command(self):
        for ctrl, arm, enabled in ((3, 6, [False]*7), (1, 6, [True]*7),
                                   (1, 0, [False]*7)):
            fake = self._fake('clearance', None)
            fake.ctrl, fake.arm, fake.enabled = ctrl, arm, enabled
            with self.assertRaisesRegex(RuntimeError, 'NORMAL'):
                top_execute.run_stage(self.args(execute=True, scene_observed=True),
                                      runner=fake, now_ns=self.now_ns)
            self.assertEqual(len(fake.calls), 1)
            self.assertFalse(self.receipt.exists())
        fake = self._fake('clearance', None)
        bad_q = list(self.q0)
        bad_q[1] = math.radians(-100.345)
        fake.q = bad_q
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            top_execute.run_stage(self.args(execute=True, scene_observed=True),
                                  runner=fake, now_ns=self.now_ns)
        self.assertFalse(any('move-j' in call for call in fake.calls))

    def test_hash_frame_quality_and_provisional_provenance(self):
        with self.assertRaisesRegex(ValueError, 'allow-provisional'):
            top_execute.run_stage(self.args(allow_provisional=False), now_ns=self.now_ns)
        self.calibration.write_bytes(b'altered')
        with self.assertRaisesRegex(json.JSONDecodeError, ''):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)
        self.calibration.write_text(json.dumps({'quality_passed': False,
                                                'tcp': 'palm',
                                                'T_flange_tcp': np.eye(4).tolist()}))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)
        self.calibration.write_text(json.dumps({'schema': 1, 'mode': 'eye_to_hand',
                                                'quality_passed': False, 'tcp': 'palm',
                                                'T_flange_tcp': np.eye(4).tolist()}))
        self.source.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'Source side plan SHA256'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)
        self.source.write_bytes(self.source_original_bytes)
        self.metadata['frame_id'] = 'camera-1:43'
        self._write_metadata()
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)

    def test_independent_fk_rejects_forged_ik_success(self):
        q = list(self.plan['waypoints']['probe_1']['target_joints_rad'])
        q[0] += math.radians(2.)
        self.plan['waypoints']['probe_1']['target_joints_rad'] = q
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'independent URDF FK'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)

    def test_probe_and_shake_amplitude_gate(self):
        original = json.loads(self.plan_path.read_text())
        self.plan['waypoints']['probe_2']['T_base_palm'][0][3] += .02
        self.plan['waypoints']['probe_2']['T_base_flange'][0][3] += .02
        self.plan['waypoints']['probe_2']['flange_pose_base_m_rad'][0] += .02
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'probe must advance'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)
        self.plan = original
        self.plan['waypoints']['tiny_shake_b']['target_joints_rad'][2] += math.radians(12.)
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'tiny-shake joint amplitude'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)

    def test_first_stage_staleness_rejected(self):
        self.metadata['host_capture_time_ns'] -= 11_000_000_000
        self._write_metadata()
        with self.assertRaisesRegex(ValueError, 'stale'):
            top_execute.run_stage(self.args(), now_ns=self.now_ns)

    def test_close_requires_contact_then_supported_shake_hold(self):
        self._to_candidate()
        with self.assertRaisesRegex(ValueError, 'physical palm/cup contact'):
            top_execute.run_stage(self.args('close', execute=True,
                                             scene_observed=True), now_ns=time.time_ns())
        fake = self._fake('close', 'contact_candidate')
        fake.hand['index_finger'] = 45
        with self.assertRaisesRegex(RuntimeError, 'hand feedback'):
            top_execute.run_stage(self.args('close', execute=True,
                                             scene_observed=True,
                                             contact_observed=True),
                                  runner=fake, now_ns=time.time_ns())
        result, fake = self._advance('close', 'contact_candidate', contact_observed=True)
        self.assertIs(result['grip_confirmed'], False)
        self.assertTrue(any('hand' in call for call in fake.calls))
        self.assertGreater(fake.hand['thumb_tip'], self.hand['thumb_tip'])
        self.assertGreater(fake.hand['thumb_base'], self.hand['thumb_base'])
        receipt = json.loads(self.receipt.read_bytes())
        self.assertEqual(receipt['contact_observed_at_stage'], 'contact_candidate')
        self.assertEqual(tuple(receipt['stages']),
                         ('clearance', 'pretop', 'guarded_hover', 'probe_1',
                          'probe_2', 'contact_candidate', 'close'))
        with self.assertRaisesRegex(ValueError, 'observed cup hold'):
            top_execute.run_stage(self.args('tiny_shake_a', execute=True,
                                             scene_observed=True), now_ns=time.time_ns())
        with self.assertRaisesRegex(ValueError, 'mouth support'):
            top_execute.run_stage(self.args('tiny_shake_a', execute=True,
                                             scene_observed=True,
                                             hold_observed=True), now_ns=time.time_ns())
        self._advance('tiny_shake_a', 'close', hold_observed=True,
                      mouth_supported_observed=True)
        with self.assertRaisesRegex(ValueError, 'observed cup hold'):
            top_execute.run_stage(self.args('tiny_shake_b', execute=True,
                                             scene_observed=True), now_ns=time.time_ns())
        self._advance('tiny_shake_b', 'tiny_shake_a', hold_observed=True,
                      mouth_supported_observed=True)
        receipt = json.loads(self.receipt.read_bytes())
        self.assertEqual(tuple(receipt['stages']),
                         ('clearance', 'pretop', 'guarded_hover', 'probe_1',
                          'probe_2', 'contact_candidate', 'close',
                          'tiny_shake_a', 'tiny_shake_b'))

    def test_early_contact_close_never_drives_to_lower_shake_height(self):
        self._to_hover()
        self._advance('close', 'guarded_hover', contact_observed=True)
        with self.assertRaisesRegex(ValueError, 'different height'):
            top_execute.run_stage(self.args('tiny_shake_a', execute=True,
                                             scene_observed=True,
                                             hold_observed=True,
                                             mouth_supported_observed=True),
                                  now_ns=time.time_ns())

    def test_probe_progression_and_receipt_source_binding(self):
        self._to_hover()
        self._advance('probe_1', 'guarded_hover')
        with self.assertRaisesRegex(ValueError, 'next reviewed'):
            top_execute.run_stage(self.args('contact_candidate'), now_ns=time.time_ns())
        self._advance('probe_2', 'probe_1')
        self._advance('contact_candidate', 'probe_2')
        receipt = json.loads(self.receipt.read_bytes())
        receipt['plan_sha256'] = '0'*64
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'different plan'):
            top_execute.run_stage(self.args('close', contact_observed=True),
                                  now_ns=time.time_ns())

    def test_prior_receipt_must_bind_exact_target_and_arrival(self):
        self._to_hover()
        receipt = json.loads(self.receipt.read_bytes())
        receipt['stages']['guarded_hover']['target_joints_rad'][0] += math.radians(1.)
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'target joints differ from exact plan'):
            top_execute.run_stage(self.args('probe_1'), now_ns=time.time_ns())
        receipt['stages']['guarded_hover']['target_joints_rad'][0] -= math.radians(1.)
        receipt['stages']['guarded_hover']['arrival_flange_m_rad'][2] += .02
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'arrival_flange_m_rad differs'):
            top_execute.run_stage(self.args('probe_1'), now_ns=time.time_ns())

    def test_missing_arrival_does_not_record_stage_and_marks_uncertain(self):
        fake = self._fake('clearance', None)
        fake.omit_arrival = True
        with self.assertRaisesRegex(execute.StageCommandError, 'lacks fresh') as error:
            top_execute.run_stage(self.args(execute=True, scene_observed=True),
                                  runner=fake, now_ns=self.now_ns)
        self.assertTrue(error.exception.command_may_have_been_sent)
        self.assertFalse(self.receipt.exists())

    def test_ros_hand_mapping_current_thumb_is_nonzero(self):
        angles = top_execute.hand_urdf_angles(self.hand)
        self.assertAlmostEqual(angles['right_thumb_metacarpal_joint'], .628, places=3)
        self.assertAlmostEqual(angles['right_thumb_proximal_joint'], 40*1.03/79.8, places=3)
        self.assertEqual(angles['right_index_proximal_joint'], 0.)
        self.assertAlmostEqual(top_execute.hand_urdf_angles({**self.hand, 'thumb_tip': 100})
                               ['right_thumb_proximal_joint'], 1.03, places=6)

    def test_actual_close_screen_checks_sampled_stl_table_clearance(self):
        plan = json.loads(self.plan_path.read_text())
        plan['cup_base']['support_center_m'] = [2., 2., -.2]
        plan['cup_base']['axis'] = [0., 0., 1.]
        plan['cup_top_center_base_m'] = [2., 2., -.128]
        pose = matrix_pose(self.model.fk(self.q0))
        clear = self.real_close_screen(plan, pose, self.hand, self.closed_hand())
        self.assertGreaterEqual(clear['sample_count'], 5)
        self.assertGreater(clear['min_exact_hand_mesh_table_m'], .005)
        plan['cup_base']['support_center_m'] = [2., 2., .3]
        plan['cup_top_center_base_m'] = [2., 2., .372]
        with self.assertRaisesRegex(ValueError, 'tabletop margin'):
            self.real_close_screen(plan, pose, self.hand, self.closed_hand())

    def test_actual_joint_screen_keeps_inverted_cup_mouth_supported_for_shake(self):
        plan = json.loads(self.plan_path.read_text())
        plan['cup_base']['support_center_m'] = [2., 2., -.2]
        axis = np.asarray(plan['cup_base']['axis'])
        plan['cup_top_center_base_m'] = (np.asarray([2., 2., -.2]) + axis*.072).tolist()
        safe = self.real_joint_screen(
            plan, self.model, self.target_q('contact_candidate'),
            self.target_q('tiny_shake_a'), self.hand, 'tiny_shake_a')
        self.assertLess(abs(safe['mouth_support_projection']['max_mouth_altitude_m']), .002)
        lower = list(self.target_q('tiny_shake_a'))
        lower[1] -= math.radians(1.)
        with self.assertRaisesRegex(ValueError, 'cup mouth lifts or presses table'):
            self.real_joint_screen(
                plan, self.model, self.target_q('contact_candidate'),
                lower, self.hand, 'tiny_shake_a')

    def test_optional_reveal_lift_then_exit_requires_measured_roi_and_cup_clearance(self):
        plan = json.loads(self.plan_path.read_text())
        plan['stage_sequence'] += ['reveal_lift', 'reveal_exit']
        axis = np.asarray(plan['cup_base']['axis'])
        contact = np.asarray(plan['waypoints']['contact_candidate']['T_base_palm'])
        lift = contact.copy()
        lift[:3, 3] += axis*.08
        horizontal = np.array([1., 0., 0.])
        horizontal -= horizontal.dot(axis)*axis
        horizontal /= np.linalg.norm(horizontal)
        exit_pose = lift.copy()
        exit_pose[:3, 3] += horizontal*.25
        for name, goal in (('reveal_lift', lift), ('reveal_exit', exit_pose)):
            plan['waypoints'][name] = {'T_base_palm': goal.tolist(),
                                        'T_base_flange': goal.tolist(),
                                        'flange_pose_base_m_rad': matrix_pose(goal),
                                        'target_joints_rad': list(self.q0),
                                        'ik': {'success': True},
                                        'hand_geometry': {'screen_passed': True,
                                                          'actual_hand_posture_verified': False}}
        plan['reveal_exit_lane'] = {
            'frame': 'base', 'source': 'measured_dice_roi_plus_reviewed_exit_lane',
            'dice_roi_source_kind': 'measured_dice_roi_plus_reviewed_exit_lane',
            'dice_directly_observed': True, 'cup_footprint_proxy': False,
            'reviewed': True, 'source_frame_id': 'camera-1:42',
            'snapshot_sha256_color': self.metadata['sha256_color'],
            'snapshot_sha256_depth': self.metadata['sha256_depth'],
            'dice_roi_center_base_m': list(plan['cup_base']['support_center_m']),
            'dice_roi_radius_m': .04, 'minimum_roi_clearance_m': .02,
            'destination_palm_base_m': exit_pose[:3, 3].tolist(),
        }
        self.assertEqual(top_execute._motion_sequence(plan)[-2:],
                         ('reveal_lift', 'reveal_exit'))
        self.assertEqual(top_execute._execution_sequence(plan)[-2:],
                         ('reveal_lift', 'reveal_exit'))
        top_execute._check_reveal(plan, self.metadata)
        plan['reveal_exit_lane']['snapshot_sha256_depth'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'exact RGB-D frame'):
            top_execute._check_reveal(plan, self.metadata)
        plan['reveal_exit_lane']['snapshot_sha256_depth'] = self.metadata['sha256_depth']
        proxy = plan['reveal_exit_lane']
        proxy.update(source='measured_cup_footprint_conservative_dice_roi',
                     dice_roi_source_kind='measured_cup_footprint_conservative_dice_roi',
                     dice_visibility='hidden_under_inverted_cup',
                     dice_directly_observed=False, cup_footprint_proxy=True,
                     dice_roi_radius_m=.071, minimum_roi_clearance_m=.01)
        top_execute._check_reveal(plan, self.metadata)
        proxy['dice_roi_radius_m'] = .05
        with self.assertRaisesRegex(ValueError, 'cover cup radius'):
            top_execute._check_reveal(plan, self.metadata)
        proxy['dice_roi_radius_m'] = .071
        proxy['dice_roi_center_base_m'][0] += .04
        with self.assertRaisesRegex(ValueError, 'measured cup mouth footprint'):
            top_execute._check_reveal(plan, self.metadata)
        proxy['dice_roi_center_base_m'][0] -= .04
        proxy['dice_directly_observed'] = True
        with self.assertRaisesRegex(ValueError, 'hidden, unobserved dice'):
            top_execute._check_reveal(plan, self.metadata)
        proxy['dice_directly_observed'] = False
        plan['waypoints']['reveal_lift']['T_base_palm'][0][3] += .03
        plan['waypoints']['reveal_lift']['T_base_flange'][0][3] += .03
        plan['waypoints']['reveal_lift']['flange_pose_base_m_rad'][0] += .03
        with self.assertRaisesRegex(ValueError, 'along cup axis'):
            top_execute._check_reveal(plan, self.metadata)

    def test_real_top_planner_output_is_consumable_in_positive_dry_run(self):
        """Exercise the actual planner output contract, with no mocked plan."""
        q = np.radians([48.705, -80., -101.039, 77.192,
                        -91.351, -35., -24.416])
        flange = np.asarray(self.model.fk(q))
        calibration = self.root/'real_palm_calibration.json'
        calibration.write_text(json.dumps({'schema': 1, 'mode': 'eye_to_hand',
                                           'quality_passed': False, 'tcp': 'palm',
                                           'T_flange_tcp': PALM.tolist()}))
        source = json.loads(self.source_original_bytes)
        source['units'] = 'm_rad'
        source['calibration_sha256'] = sha(calibration.read_bytes())
        source['snapshot_sha256_color'] = self.metadata['sha256_color']
        source['snapshot_sha256_depth'] = self.metadata['sha256_depth']
        source['cup_base'] = {'frame': 'base',
                              'support_center_m': [-.1640344, .4741704, .04],
                              'axis': [.0286169, .0148116, .9994807],
                              'dimensions': {'observed_height_m': .070427,
                                             'observed_side_diameter_m': .0757828},
                              'source': {'frame_id': 'camera-1:42'}}
        support = np.asarray(source['cup_base']['support_center_m'])
        axis = np.asarray(source['cup_base']['axis'])
        axis /= np.linalg.norm(axis)
        top_point = support+axis*(.070427+.002)
        source['cup_base']['measured_top_surface'] = {
            'valid': True, 'source_frame_id': 'camera-1:42',
            'snapshot_sha256_color': self.metadata['sha256_color'],
            'snapshot_sha256_depth': self.metadata['sha256_depth'],
            'model_sha256': 'a'*64,
            'center_definition': 'side_axis_intersection_with_visible_top_plane',
            'center_base_m': top_point.tolist(), 'normal_base': axis.tolist(),
            'quality': {'rim_center_independently_measured': False,
                        'coaxial_cup_assumption': True},
        }
        source['recognition'] = {
            'selected_instance': 0, 'model_sha256': 'a'*64,
            'red_workspace': {
                'selected_green_cap_in_red_mat': True,
                'table_support_center_inside_red_mat': True,
                'config_sha256': 'f'*64, 'selected_instance': 0,
                'source_frame_id': 'camera-1:42',
                'snapshot_sha256_color': self.metadata['sha256_color']}}
        source['current_flange_pose_base_m_rad'] = matrix_pose(flange)
        source['current_palm_pose_base_m_rad'] = matrix_pose(flange @ PALM)
        source['pose_source'] = {'kind': 'can_feedback_read_only',
                                 'channel': 'can0', 'host_read_time_ns': time.time_ns()}
        source['checks'] = {'snapshot_age_s': 2., 'provisional_opt_in': True,
                            'calibration_quality_passed': False}
        source_path = self.root/'real_side_plan.json'
        source_path.write_text(json.dumps(source))
        feedback = {'joints_rad': q.tolist(), 'sdk_limits_deg': self.sdk,
                    'joints_enabled': [True]*7, 'arm_status': 0, 'ctrl_mode': 1}
        horizontal = np.asarray(flange @ PALM)[:3, 0].copy()
        horizontal -= float(horizontal @ axis)*axis
        horizontal /= np.linalg.norm(horizontal)
        lane = {
            'frame': 'base', 'source': 'measured_cup_footprint_conservative_dice_roi',
            'reviewed': True, 'source_frame_id': 'camera-1:42',
            'snapshot_sha256_color': self.metadata['sha256_color'],
            'snapshot_sha256_depth': self.metadata['sha256_depth'],
            'dice_visibility': 'hidden_under_inverted_cup',
            'dice_directly_observed': False,
            'dice_roi_center_base_m': support.tolist(),
            'dice_roi_radius_m': .071, 'minimum_roi_clearance_m': .01,
            'destination_palm_base_m': (top_point+axis*(.013+.08)+horizontal*.14).tolist(),
        }
        top = top_grasp.compute_top_grasp(
            source, source_side_plan_sha256=sha(source_path.read_bytes()),
            live_joint_feedback=feedback, hand_open_positions=self.hand,
            hand_status_source={'kind': 'hand_feedback_read_only',
                                'host_read_time_ns': time.time_ns()},
            orientation_reviewed=True, provisional_opt_in=True,
            pretop_height_m=.11, reveal_exit_lane=lane)
        self.assertTrue(top['checks']['execute_ready'],
                        {'checks': top['checks'],
                         'stages': {name: {'ik': item['ik'],
                                           'target_joints_deg': item['target_joints_deg']}
                                    for name, item in top['waypoints'].items()}})
        self.assertFalse(top['checks']['physical_contact_verified'])
        self.assertEqual(top['reveal_exit_lane']['dice_roi_source_kind'],
                         'measured_cup_footprint_conservative_dice_roi')
        self.assertFalse(top['reveal_exit_lane']['dice_directly_observed'])
        self.assertGreater(top['hand_close_updates']['thumb_tip'],
                           self.hand['thumb_tip'])
        top_path = self.root/'real_top_plan.json'
        top_path.write_text(json.dumps(top))
        fake = FakeControl(q, top['waypoints']['clearance']['target_joints_rad'],
                           self.sdk, self.hand)
        preview = top_execute.run_stage(
            self.args(plan=top_path, source_side_plan=source_path,
                      calibration=calibration, receipt=self.root/'real_receipt.json'),
            runner=fake, now_ns=time.time_ns())
        self.assertEqual(preview['event'], 'top_stage_plan')
        self.assertFalse(preview['execute'])
        self.assertFalse(preview['dice_directly_observed'])
        self.assertEqual(fake.calls, [])
        self.assertFalse((self.root/'real_receipt.json').exists())


if __name__ == '__main__':
    unittest.main()
