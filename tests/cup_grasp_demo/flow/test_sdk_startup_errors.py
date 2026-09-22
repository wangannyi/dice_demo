"""Startup failures retain process evidence and never dispatch robot motion."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from cup_grasp_demo.flow import green_runtime as runtime
from cup_grasp_demo.flow import green_sdk_worker as worker
from cup_grasp_demo.flow.shake_execution import control_conflicts


class StartupErrorsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.before = dict(channel='can0', receiver_rows=[], errors=[], candidate_control_processes=[])
        self.after = dict(self.before, receiver_rows=[
            dict(list='all', line='can0 000 00000000'),
            dict(list='err', line='can0 000 1fffffff')])
        self.robot = Mock()
        self.factory = Mock(return_value=self.robot)
        self.evidence = self.root/'sdk_startup.json'

    def connect(self, before=None, after=None):
        return worker.connect_sdk('can0', self.factory,
                                  Mock(side_effect=[before or self.before, after or self.after]),
                                  control_conflicts, self.evidence)

    def test_conflict_names_pid_and_keeps_evidence_before_disconnecting(self):
        conflict = dict(self.before, candidate_control_processes=[
            dict(pid=123, ppid=1, args='python -m unittest cup_grasp_demo.test')])
        with self.assertRaisesRegex(RuntimeError, 'PID 123: python -m unittest'):
            self.connect(before=conflict)
        record = json.loads(self.evidence.read_text())
        self.assertFalse(record['ready'])
        self.assertFalse(record['motion_attempted'])
        self.assertEqual(record['before_connect'], conflict)
        self.assertEqual(record['after_connect'], self.after)
        self.robot.disconnect.assert_called_once()
        self.assertEqual([c[0] for c in self.robot.method_calls],
                         ['init_effector', 'connect', 'disconnect'])

    def test_can_receiver_conflict_stays_blocked(self):
        with self.assertRaisesRegex(RuntimeError, 'CAN receivers existed'):
            self.connect(before=dict(self.before, receiver_rows=[dict(line='other socket')]))
        self.robot.disconnect.assert_called_once()

    def test_success_preserves_connection_and_original_baseline(self):
        robot, hand, baseline = self.connect()
        self.assertIs(robot, self.robot)
        self.assertEqual(baseline, self.before)
        self.assertTrue(json.loads(self.evidence.read_text())['ready'])
        self.assertEqual([c[0] for c in self.robot.method_calls], ['init_effector', 'connect'])

    def test_connect_exception_is_recorded_and_disconnected(self):
        self.robot.connect.side_effect = OSError('CAN unavailable')
        with self.assertRaisesRegex(OSError, 'CAN unavailable'):
            self.connect()
        self.assertIn('CAN unavailable', json.loads(self.evidence.read_text())['error'])
        self.robot.disconnect.assert_called_once()

    def test_failed_ready_response_reaches_user_and_closes_client(self):
        proc = Mock()
        proc.poll.return_value = 1
        with patch.object(runtime.subprocess, 'Popen', return_value=proc), \
             patch.object(runtime.SDKClient, 'receive', return_value=dict(ready=False, error='PID 123 conflict')):
            with self.assertRaisesRegex(RuntimeError, 'PID 123 conflict.*sdk_worker.log'):
                runtime.SDKClient(dict(channel='can0'), self.root)
        proc.stdin.write.assert_not_called()
        proc.stdin.close.assert_called_once()
        proc.stdout.close.assert_called_once()

    def test_early_exit_reports_cause_and_full_log_path(self):
        client = object.__new__(runtime.SDKClient)
        client.log_path = self.root/'sdk_worker.log'
        client.log_path.write_text('Traceback:\nRuntimeError: lock held by another process\n')
        client.process = Mock(stdout=io.StringIO(''))
        with patch.object(runtime.select, 'select', return_value=([client.process.stdout], [], [])):
            with self.assertRaisesRegex(RuntimeError, 'lock held by another process.*sdk_worker.log'):
                client.receive(1)


if __name__ == '__main__':
    unittest.main()
