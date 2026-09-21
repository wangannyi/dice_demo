"""Fake-control checks for one-stage grasp execution and fail-closed receipts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from cup_grasp_demo import execute
from cup_grasp_demo import ik_review
from nero_calibration.core import matrix_pose
from nero_revo2_control.kinematics import load_model


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


class FakeControl:
    def __init__(self, before, after, *, fault=False, omit_arrival=False,
                 before_q=None, after_q=None, sdk_limits=None,
                 arrival_command=None):
        self.status_poses = [before, after]
        self.status_joints = [before_q, after_q]
        self.before_q = before_q
        self.sdk_limits = sdk_limits
        self.fault = fault
        self.omit_arrival = omit_arrival
        self.arrival_command = arrival_command
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if 'read-joints' in command:
            event = {'event': 'read_joints', 'joints_rad': self.before_q,
                     'sdk_limits_deg': self.sdk_limits, 'arm_status': 0,
                     'ctrl_mode': 1, 'joints_enabled': [True] * 7}
            return subprocess.CompletedProcess(command, 0,
                                               json.dumps(event) + '\n', '')
        if 'hand-status' in command:
            event = {'event': 'hand_status', 'is_ok': True,
                     'positions': {'thumb_tip': 18, 'thumb_base': 3,
                                   'index_finger': 0, 'middle_finger': 0,
                                   'ring_finger': 0, 'pinky_finger': 0}}
            return subprocess.CompletedProcess(command, 0,
                                               json.dumps(event) + '\n', '')
        if 'status' in command:
            pose = self.status_poses.pop(0)
            joints = self.status_joints.pop(0)
            status = ('  ctrl_mode: CAN_CTRL(0x1)\n  arm_status: EMERGENCY_STOP(0x1)'
                      if self.fault else
                      '  ctrl_mode: CAN_CTRL(0x1)\n  arm_status: NORMAL(0x0)')
            event = {'event': 'arm_status', 'status': status,
                     'joints_enabled': [True] * 7, 'flange_m_rad': pose,
                     'joints_rad': joints}
        elif self.omit_arrival:
            event = {'event': 'arm_progress'}
        elif 'hand' in command:
            event = {'event': 'hand_target_reached', 'fresh_samples': 3}
        else:
            event = {'event': 'arm_target_reached',
                     'command': (self.arrival_command or
                                 ('move-j' if 'move-j' in command else 'move-p')),
                     'fresh_samples': 10}
        return subprocess.CompletedProcess(command, 0, json.dumps(event) + '\n', '')


class ExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.snapshot = self.root / 'snap'
        self.snapshot.mkdir()
        (self.snapshot / 'color.png').write_bytes(b'color')
        (self.snapshot / 'depth.npz').write_bytes(b'depth')
        self.now_ns = time.time_ns()
        self.metadata = {
            'schema': 1, 'camera_backend': 'realsense', 'serial': 'camera-1',
            'frame': 'color_optical', 'depth_registered_to': 'color_optical',
            'frame_id': 'camera-1:42', 'host_capture_time_ns': self.now_ns,
            'timestamp_ms': 1000., 'depth_timestamp_ms': 1001.,
            'timestamp_domain': 'hardware_clock',
            'depth_timestamp_domain': 'hardware_clock',
            'sha256_color': _sha(b'color'), 'sha256_depth': _sha(b'depth'),
        }
        self._write_metadata()
        self.calibration = self.root / 'calibration.json'
        self.calibration.write_bytes(b'{"quality_passed": false}')
        self.plan_path = self.root / 'plan.json'
        self.receipt = self.root / 'stages.json'
        self.plan = {
            'schema': 1, 'kind': 'read_only_side_grasp_proposal',
            'units': 'm_rad', 'checks': {
                'execute_ready': True, 'pose_source_live_can': True,
                'calibration_quality_passed': False, 'provisional_opt_in': True,
            },
            'pose_source': {'kind': 'can_feedback_read_only',
                            'host_read_time_ns': self.now_ns},
            'calibration_sha256': _sha(self.calibration.read_bytes()),
            'snapshot_frame_id': 'camera-1:42',
            'localization': {'valid': True, 'frame': 'color_optical',
                             'frame_id': 'camera-1:42', 'camera_serial': 'camera-1'},
            'cup_base': {'source': {'frame_id': 'camera-1:42',
                                    'camera_serial': 'camera-1'}},
            'current_flange_pose_base_m_rad': [.1, .2, .3, 0, 0, 0],
            'waypoints': {
                'pregrasp': {'flange_pose_base_m_rad': [.2, .2, .3, 0, 0, 0]},
                'contact': {'flange_pose_base_m_rad': [.25, .2, .3, 0, 0, 0]},
                'lift': {'flange_pose_base_m_rad': [.25, .2, .35, 0, 0, 0]},
            },
        }
        self._write_plan()

    def _write_metadata(self):
        (self.snapshot / 'metadata.json').write_text(json.dumps(self.metadata))

    def _write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def _require_clearance(self):
        self.plan['checks']['clearance_required'] = True
        self.plan['waypoints']['clearance'] = {
            'flange_pose_base_m_rad': [.1, .2, .38, 0, 0, 0]}
        self._write_plan()

    def args(self, stage='pregrasp', **overrides):
        fields = dict(stage=stage, plan=self.plan_path, snapshot_dir=self.snapshot,
                      calibration=self.calibration, receipt=self.receipt,
                      channel='can0', python='/fake/python',
                      allow_provisional=True, scene_observed=False,
                      grip_observed=False, execute=False,
                      set_values=None, positions=None, duration=1.5,
                      move_timeout=90., arm_mode='cartesian', ik_review=None)
        fields.update(overrides)
        return argparse.Namespace(**fields)

    def _joint_fixture(self):
        model = load_model()
        start = tuple(math.radians(v) for v in
                      (55., -78., 80., -45., 130., -30., 40.))
        targets = {}
        self.plan['current_flange_pose_base_m_rad'] = matrix_pose(model.fk(start))
        for name, delta in (('pregrasp', 1.), ('contact', 2.), ('lift', 3.)):
            q = list(start)
            q[0] += math.radians(delta)
            goal = model.fk(q)
            targets[name] = tuple(q)
            self.plan['waypoints'][name] = {
                'T_base_flange': [list(row) for row in goal],
                'flange_pose_base_m_rad': matrix_pose(goal),
            }
        self._write_plan()
        sdk = [[-155., 155.], [-100., 100.], [-158., 158.],
               [-58., 123.], [-158., 158.], [-42., 55.], [-90., 90.]]
        report = ik_review.review_plan(
            self.plan, start, sdk_limits_deg=sdk,
            joint_source='nero_revo2_control_read_joints:can0',
            source_plan_sha256=_sha(self.plan_path.read_bytes()))
        self.assertTrue(report['kinematic_checks_passed'], report['blockers'])
        self.ik_path = self.root / 'ik_review.json'
        self.ik_path.write_text(json.dumps(report))
        return start, targets, sdk, report

    def test_preview_never_connects_or_creates_receipt(self):
        control = FakeControl(self.plan['current_flange_pose_base_m_rad'],
                              self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'])
        preview = execute.run_stage(self.args(), runner=control)
        self.assertFalse(preview['execute'])
        self.assertEqual(control.commands, [])
        self.assertFalse(self.receipt.exists())

    def test_unsynced_rgbd_snapshot_cannot_execute(self):
        self.metadata['depth_timestamp_ms'] = 1170.
        self._write_metadata()
        with self.assertRaisesRegex(ValueError, 'timestamp gap'):
            execute.run_stage(self.args(execute=True))
        self.assertFalse(self.receipt.exists())

    def test_provisional_requires_double_opt_in_and_real_file_hash(self):
        with self.assertRaisesRegex(ValueError, 'allow-provisional'):
            execute.run_stage(self.args(allow_provisional=False))
        self.calibration.write_bytes(b'{"quality_passed": true}')
        with self.assertRaisesRegex(ValueError, 'differs'):
            execute.run_stage(self.args())

    def test_pregrasp_rejects_stale_snapshot_and_stale_pose(self):
        old = self.now_ns - 11_000_000_000
        self.metadata['host_capture_time_ns'] = old
        self._write_metadata()
        with self.assertRaisesRegex(ValueError, 'old'):
            execute.run_stage(self.args(), now_ns=self.now_ns)
        self.metadata['host_capture_time_ns'] = self.now_ns
        self._write_metadata()
        self.plan['pose_source']['host_read_time_ns'] = old
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'stale'):
            execute.run_stage(self.args(), now_ns=self.now_ns)

    def test_optional_clearance_stage_requires_fresh_first_action(self):
        self._require_clearance()
        with self.assertRaisesRegex(ValueError, 'requires existing receipt'):
            execute.run_stage(self.args('pregrasp'))
        self.metadata['host_capture_time_ns'] = self.now_ns - 11_000_000_000
        self._write_metadata()
        with self.assertRaisesRegex(ValueError, 'clearance snapshot is .* old'):
            execute.run_stage(self.args('clearance'), now_ns=self.now_ns)
        self.metadata['host_capture_time_ns'] = self.now_ns
        self._write_metadata()
        self.plan['pose_source']['host_read_time_ns'] = self.now_ns - 11_000_000_000
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'clearance planned arm feedback is stale'):
            execute.run_stage(self.args('clearance'), now_ns=self.now_ns)

    def test_clearance_waypoint_lifts_in_base_z_without_rotation(self):
        self._require_clearance()
        self.plan['waypoints']['clearance']['flange_pose_base_m_rad'][0] += .02
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'lift 80 mm in base Z'):
            execute.run_stage(self.args('clearance', execute=True))
        self.assertFalse(self.receipt.exists())

    def test_aligned_clearance_pregrasp_chain_and_receipt(self):
        self._require_clearance()
        start = self.plan['current_flange_pose_base_m_rad']
        clearance = self.plan['waypoints']['clearance']['flange_pose_base_m_rad']
        pregrasp = self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad']
        with self.assertRaisesRegex(ValueError, 'not a stage for this plan'):
            self.plan['checks']['clearance_required'] = False
            self._write_plan()
            execute.run_stage(self.args('clearance'))
        self.plan['checks']['clearance_required'] = True
        self._write_plan()
        control = FakeControl(start, clearance)
        with self.assertRaisesRegex(ValueError, 'scene-observed'):
            execute.run_stage(self.args('clearance', execute=True), runner=control)
        self.assertEqual(control.commands, [])
        result = execute.run_stage(self.args('clearance', execute=True, scene_observed=True),
                                   runner=control)
        self.assertEqual(result['stage'], 'clearance')
        self.assertIn('--speed', control.commands[1])
        self.assertEqual(control.commands[1][control.commands[1].index('--speed') + 1], '1')
        self.assertIn('0.38', control.commands[1])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            execute.run_stage(self.args('clearance'))
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(tuple(receipt['stages']), ('clearance',))
        control = FakeControl(clearance, pregrasp)
        preview = execute.run_stage(self.args('pregrasp'), runner=control)
        self.assertEqual(preview['previous_verified_stage'], 'clearance')
        self.assertEqual(preview['expected_start_flange_m_rad'], clearance)
        self.assertEqual(control.commands, [])
        with self.assertRaisesRegex(ValueError, 'scene-observed'):
            execute.run_stage(self.args('pregrasp', execute=True),
                              runner=FakeControl(clearance, pregrasp))
        execute.run_stage(self.args('pregrasp', execute=True, scene_observed=True),
                          runner=FakeControl(clearance, pregrasp))
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(tuple(receipt['stages']), ('clearance', 'pregrasp'))
        contact = self.plan['waypoints']['contact']['flange_pose_base_m_rad']
        lift = self.plan['waypoints']['lift']['flange_pose_base_m_rad']
        execute.run_stage(self.args('contact', execute=True, scene_observed=True),
                          runner=FakeControl(pregrasp, contact))
        execute.run_stage(self.args('close', execute=True), runner=FakeControl(contact, contact))
        execute.run_stage(self.args('lift', execute=True, scene_observed=True,
                                    grip_observed=True), runner=FakeControl(contact, lift))
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(tuple(receipt['stages']), execute.ALL_STAGES)

    def test_clearance_receipt_must_match_exact_plan_and_verified_stage(self):
        self._require_clearance()
        start = self.plan['current_flange_pose_base_m_rad']
        clearance = self.plan['waypoints']['clearance']['flange_pose_base_m_rad']
        execute.run_stage(self.args('clearance', execute=True, scene_observed=True),
                          runner=FakeControl(start, clearance))
        self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'][0] += .01
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'different plan'):
            execute.run_stage(self.args('pregrasp', execute=True))
        self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'][0] -= .01
        self._write_plan()
        receipt = json.loads(self.receipt.read_text())
        receipt['stages']['clearance']['verified'] = False
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'exact verified prior-stage sequence'):
            execute.run_stage(self.args('pregrasp', execute=True))

    def test_asset_hash_or_frame_id_change_is_rejected(self):
        (self.snapshot / 'color.png').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'differs'):
            execute.run_stage(self.args())
        (self.snapshot / 'color.png').write_bytes(b'color')
        self.plan['snapshot_frame_id'] = 'camera-1:43'
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'frame IDs differ'):
            execute.run_stage(self.args())

    def test_arm_fault_prevents_command_and_receipt(self):
        control = FakeControl(self.plan['current_flange_pose_base_m_rad'],
                              self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'],
                              fault=True)
        with self.assertRaisesRegex(RuntimeError, 'NORMAL'):
            execute.run_stage(self.args(execute=True), runner=control)
        self.assertEqual(len(control.commands), 1)
        self.assertFalse(self.receipt.exists())

    def test_unexpected_start_or_missing_arrival_never_records_completion(self):
        far = [.12, .2, .3, 0, 0, 0]
        control = FakeControl(far, self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'])
        with self.assertRaisesRegex(RuntimeError, 'differs'):
            execute.run_stage(self.args(execute=True), runner=control)
        self.assertEqual(len(control.commands), 1)
        control = FakeControl(self.plan['current_flange_pose_base_m_rad'],
                              self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'],
                              omit_arrival=True)
        with self.assertRaisesRegex(execute.StageCommandError, 'lacks fresh') as failure:
            execute.run_stage(self.args(execute=True), runner=control)
        self.assertTrue(failure.exception.command_may_have_been_sent)
        self.assertEqual(len(control.commands), 2)
        self.assertFalse(self.receipt.exists())

    def test_strict_four_stage_chain_and_modest_partial_hand_command(self):
        start = self.plan['current_flange_pose_base_m_rad']
        pre = self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad']
        contact = self.plan['waypoints']['contact']['flange_pose_base_m_rad']
        lift = self.plan['waypoints']['lift']['flange_pose_base_m_rad']
        with self.assertRaisesRegex(ValueError, 'receipt'):
            execute.run_stage(self.args('contact', execute=True, scene_observed=True))
        execute.run_stage(self.args('pregrasp', execute=True),
                          runner=FakeControl(start, pre))
        self.assertTrue(self.receipt.exists())
        with self.assertRaisesRegex(ValueError, 'scene-observed'):
            execute.run_stage(self.args('contact', execute=True),
                              runner=FakeControl(pre, contact))
        execute.run_stage(self.args('contact', execute=True, scene_observed=True),
                          runner=FakeControl(pre, contact))
        hand = FakeControl(contact, contact)
        close = execute.run_stage(self.args('close', execute=True), runner=hand)
        self.assertIs(close['grip_confirmed'], False)
        hand_command = hand.commands[2]
        for update in execute.DEFAULT_FINGER_UPDATES:
            self.assertIn(update, hand_command)
        self.assertNotIn('--positions', hand_command)
        with self.assertRaisesRegex(ValueError, 'scene-observed'):
            execute.run_stage(self.args('lift', execute=True),
                              runner=FakeControl(contact, lift))
        with self.assertRaisesRegex(ValueError, 'grip-observed'):
            execute.run_stage(self.args('lift', execute=True,
                                        scene_observed=True),
                              runner=FakeControl(contact, lift))
        execute.run_stage(self.args('lift', execute=True, scene_observed=True,
                                    grip_observed=True),
                          runner=FakeControl(contact, lift))
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(tuple(receipt['stages']), execute.STAGES)
        with self.assertRaisesRegex(ValueError, 'already completed'):
            execute.run_stage(self.args('close', execute=True))

    def test_plan_change_or_old_receipt_blocks_next_stage(self):
        start = self.plan['current_flange_pose_base_m_rad']
        pre = self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad']
        execute.run_stage(self.args(execute=True), runner=FakeControl(start, pre))
        original_plan_bytes = self.plan_path.read_bytes()
        self.plan['waypoints']['lift']['flange_pose_base_m_rad'][0] += .01
        self._write_plan()
        with self.assertRaisesRegex(ValueError, 'different plan'):
            execute.run_stage(self.args('contact', scene_observed=True, execute=True))
        self.plan_path.write_bytes(original_plan_bytes)
        receipt = json.loads(self.receipt.read_text())
        receipt['stages']['pregrasp']['completed_at_ns'] -= 181_000_000_000
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'stale'):
            execute.run_stage(self.args('contact', scene_observed=True,
                                        execute=True))

    def test_default_arm_mode_still_emits_move_p(self):
        start = self.plan['current_flange_pose_base_m_rad']
        target = self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad']
        control = FakeControl(start, target)
        execute.run_stage(self.args(execute=True), runner=control)
        self.assertIn('move-p', control.commands[1])
        self.assertNotIn('move-j', control.commands[1])
        self.assertNotIn('arm_mode', json.loads(self.receipt.read_text()))

    def test_joint_mode_binds_plan_review_and_arrival_to_move_j(self):
        start, targets, sdk, report = self._joint_fixture()
        args = self.args(arm_mode='joint', ik_review=self.ik_path)
        dry = execute.run_stage(args)
        self.assertEqual(dry['arm_mode'], 'joint')
        self.assertEqual(dry['target_joints_deg'],
                         [math.degrees(v) for v in report['stages'][0]['ik']['joints_rad']])
        self.assertFalse(self.receipt.exists())
        target = tuple(report['stages'][0]['ik']['joints_rad'])
        control = FakeControl(self.plan['current_flange_pose_base_m_rad'],
                              self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'],
                              before_q=start, after_q=target, sdk_limits=sdk)
        execute.run_stage(self.args(arm_mode='joint', ik_review=self.ik_path,
                                    execute=True), runner=control)
        self.assertEqual([next(name for name in ('status', 'read-joints', 'move-j')
                               if name in command) for command in control.commands],
                         ['status', 'read-joints', 'move-j', 'status'])
        move = control.commands[2]
        self.assertIn('--joints-deg', move)
        self.assertEqual(move[move.index('--speed') + 1], '1')
        self.assertNotIn('move-p', move)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['arm_mode'], 'joint')
        self.assertEqual(receipt['stages']['pregrasp']['arm_mode'], 'joint')
        self.assertEqual(receipt['stages']['pregrasp']['target_joints_deg'],
                         [math.degrees(v) for v in target])

    def test_joint_mode_rejects_plan_hash_frame_and_ik_target_tampering(self):
        start, targets, sdk, original = self._joint_fixture()
        args = self.args(arm_mode='joint', ik_review=self.ik_path)
        invalid = copy.deepcopy(original)
        invalid['source_plan_sha256'] = '0' * 64
        self.ik_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'exact plan SHA256'):
            execute.run_stage(args)
        invalid = copy.deepcopy(original)
        invalid['source_plan_frame_id'] = 'camera-1:wrong'
        self.ik_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'RGB-D frame'):
            execute.run_stage(args)
        invalid = copy.deepcopy(original)
        invalid['stages'][0]['ik']['joints_rad'][0] += math.radians(15.)
        invalid['stages'][0]['ik']['joints_deg'][0] += 15.
        self.ik_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'URDF FK differs'):
            execute.run_stage(args)
        self.assertFalse(self.receipt.exists())

    def test_joint_mode_rejects_failed_ik_live_sdk_and_wrong_arrival(self):
        start, targets, sdk, original = self._joint_fixture()
        args = self.args(arm_mode='joint', ik_review=self.ik_path, execute=True)
        invalid = copy.deepcopy(original)
        invalid['kinematic_checks_passed'] = False
        self.ik_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'kinematic checks failed'):
            execute.run_stage(args)
        invalid = copy.deepcopy(original)
        invalid['stages'][0]['ik']['success'] = False
        self.ik_path.write_text(json.dumps(invalid))
        with self.assertRaisesRegex(ValueError, 'IK did not succeed'):
            execute.run_stage(args)
        self.ik_path.write_text(json.dumps(original))
        target = tuple(original['stages'][0]['ik']['joints_rad'])
        current_pose = self.plan['current_flange_pose_base_m_rad']
        target_pose = self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad']
        altered_sdk = copy.deepcopy(sdk)
        altered_sdk[6][1] -= 1.
        control = FakeControl(current_pose, target_pose, before_q=start,
                              after_q=target, sdk_limits=altered_sdk)
        with self.assertRaisesRegex(ValueError, 'Live SDK joint limits differ'):
            execute.run_stage(args, runner=control)
        self.assertEqual(len(control.commands), 2)
        control = FakeControl(current_pose, target_pose, before_q=start,
                              after_q=target, sdk_limits=sdk,
                              arrival_command='move-p')
        with self.assertRaisesRegex(execute.StageCommandError, 'not from move-j'):
            execute.run_stage(args, runner=control)
        self.assertEqual(len(control.commands), 3)
        self.assertFalse(self.receipt.exists())

    def test_joint_receipt_requires_same_review_for_next_stage(self):
        start, targets, sdk, report = self._joint_fixture()
        pre_q = tuple(report['stages'][0]['ik']['joints_rad'])
        pre_control = FakeControl(self.plan['current_flange_pose_base_m_rad'],
                                  self.plan['waypoints']['pregrasp']['flange_pose_base_m_rad'],
                                  before_q=start, after_q=pre_q, sdk_limits=sdk)
        execute.run_stage(self.args(arm_mode='joint', ik_review=self.ik_path,
                                    execute=True), runner=pre_control)
        with self.assertRaisesRegex(ValueError, 'joint-mode grasp sequence'):
            execute.run_stage(self.args('contact', execute=True,
                                        scene_observed=True))
        altered = copy.deepcopy(report)
        altered['review_note'] = 'unreviewed edit after pregrasp'
        self.ik_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, 'different arm mode or IK review'):
            execute.run_stage(self.args('contact', arm_mode='joint',
                                        ik_review=self.ik_path, execute=True,
                                        scene_observed=True))
        self.assertEqual(tuple(json.loads(self.receipt.read_text())['stages']),
                         ('pregrasp',))


if __name__ == '__main__':
    unittest.main()
