"""Exercise persistent SDK failure receipts through the workflow retry boundary."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from cup_grasp_demo.calibration_debug.green_pipeline import Workflow
from cup_grasp_demo.calibration_debug.green_runtime import SDKClient


class ShakeRecoveryTest(unittest.TestCase):
    def scenario(self, failures):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        grip = root / 'grip.json'
        lift = root / 'lift.json'
        grip.write_text('{}')
        lift.write_text(json.dumps({'joint_delivery_events': [{'live_limits': {'source': 'live'}}]}))
        w = object.__new__(Workflow)
        w.root = root
        w.args = SimpleNamespace(mode='fast')
        w.g = dict(persistent_runtime=True, recovery_attempts=1,
                   grip_targets_0_100=[0,100,40,40,40,100], held_cup_margin_mm=0,
                   require_arm_position=False)
        w.cfg = dict(timeout_s=40, channel='can0', speed_percent=30)
        w.receipts = dict(grip=str(grip), lift=str(lift))
        w.hashes = {}
        w.snapshot = Mock(side_effect=[[.01]*7, [.02]*7])
        w.build_shake = Mock(side_effect=lambda feedback: (
            dict(blockers=[], duration_s=5, start_q_rad=feedback['q_after_rad']), {}))
        w.record_handoff = Mock()
        client = object.__new__(SDKClient)
        client.cfg = w.cfg
        client.process = Mock()
        client.close = Mock()
        messages = []
        client.process.stdin.write.side_effect = lambda line: messages.append(json.loads(line))
        outcomes = iter(failures)
        def receive(timeout):
            message = messages[-1]
            report = next(outcomes)
            Path(message['output']).write_text(json.dumps(report))
            return dict(output=message['output'], returncode=0 if report.get('success') else 2)
        client.receive = receive
        w._sdk = client
        return w, client, messages

    def test_pre_motion_mismatch_rebuilds_from_new_snapshot(self):
        mismatch = dict(success=False, duration_completed=False,
                        failure_code='start_position_changed', motion_attempted=False)
        w, client, messages = self.scenario([mismatch, dict(success=True, duration_completed=True)])
        w.shake()
        self.assertEqual(w.build_shake.call_count, 2)
        self.assertEqual(w.build_shake.call_args.args[0]['q_after_rad'], [.02]*7)
        self.assertEqual(len(messages), 2)
        self.assertEqual(len(w.recovery_events), 1)
        self.assertIn('shake', w.receipts)
        client.close.assert_not_called()

    def test_second_mismatch_exhausts_retry_without_third_command(self):
        mismatch = dict(success=False, duration_completed=False,
                        failure_code='start_position_changed', motion_attempted=False)
        w, _, messages = self.scenario([mismatch, mismatch])
        with self.assertRaisesRegex(RuntimeError, '摇晃未完成'):
            w.shake()
        self.assertEqual(len(messages), 2)
        self.assertNotIn('shake', w.receipts)

    def test_started_motion_and_communication_fault_never_retry(self):
        for attempted, code in [(True, 'start_position_changed'), (False, 'communication')]:
            with self.subTest(attempted=attempted, code=code):
                w, client, messages = self.scenario([dict(success=False, error='fault',
                    motion_attempted=attempted, failure_code=code)])
                with self.assertRaisesRegex(RuntimeError, 'fault'):
                    w.shake()
                self.assertEqual(len(messages), 1)
                self.assertNotIn('shake', w.receipts)
                client.close.assert_called_once()
