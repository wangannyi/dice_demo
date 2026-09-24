"""SDK worker restart: only a dead worker with a zero-transmission receipt retries."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

from cup_grasp_demo.flow.green_runtime import SDKClient

FAKE_WORKER = r'''
import json, os, sys
behavior = {}
try:
    with open(sys.argv[1]) as f:
        behavior = json.load(f)
except OSError:
    pass
print(json.dumps({'ready': True}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message.get('command') == 'close':
        break
    output = message['output']
    # die_once: the first command dies after (optionally) writing the receipt;
    # every later command on this process completes normally.
    marker = behavior.get('marker_path')
    should_die = behavior.get('die_once') and marker and not os.path.exists(marker)
    receipt = behavior.get('receipt') if should_die else {'success': True}
    if should_die:
        with open(marker, 'w') as m:
            m.write('died')
        if not behavior.get('skip_receipt'):
            with open(output, 'w') as f:
                json.dump(receipt, f)
        sys.exit(3)
    with open(output, 'w') as f:
        json.dump(receipt, f)
    print(json.dumps({'output': output, 'returncode': behavior.get('returncode', 0)}), flush=True)
'''

ZERO_TX_FAILURE = {'success': False, 'motion_attempted': False, 'finger_commands_sent': 0,
                   'tx': {'actual_tx_count': 0, 'transmission_outcome_uncertain': False},
                   'error': 'simulated pre-motion death'}


class WorkerRestartTests(unittest.TestCase):
    def client(self, behavior, tmp):
        script = Path(tmp) / 'fake_worker.py'
        script.write_text(FAKE_WORKER)
        behavior_file = Path(tmp) / 'behavior.json'
        behavior_file.write_text(json.dumps(behavior))
        cfg = dict(timeout_s=5, green_cup={})
        argv = [sys.executable, str(script), str(behavior_file)]
        return SDKClient(cfg, Path(tmp), worker_argv=argv)

    def test_zero_tx_death_restarts_and_retries_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            behavior = dict(die_once=True, marker_path=str(Path(tmp) / 'died.marker'),
                            receipt=ZERO_TX_FAILURE)
            client = self.client(behavior, tmp)
            try:
                output = Path(tmp) / 'receipt.json'
                report = client.call('snapshot', output)
                self.assertTrue(report.get('worker_restarted'))
                # After the retry the connection is a fresh worker.
                self.assertEqual(client._log_index, 2)
            finally:
                client.close()

    def test_completed_receipt_is_returned_without_resend(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt = {'success': True, 'motion_attempted': True,
                       'tx': {'actual_tx_count': 12, 'transmission_outcome_uncertain': False}}
            behavior = dict(die_once=True, marker_path=str(Path(tmp) / 'died.marker'),
                            receipt=receipt)
            client = self.client(behavior, tmp)
            try:
                output = Path(tmp) / 'receipt.json'
                report = client.call('run', output)
                # Real completion: the receipt is returned as-is, never resent,
                # but the connection is still replaced for later commands.
                self.assertTrue(report.get('success'))
                self.assertIsNone(report.get('worker_restarted'))
                self.assertEqual(client._log_index, 2)
                marker = Path(tmp) / 'died.marker'
                stamp = marker.stat().st_mtime_ns
                import time as _t; _t.sleep(.05)
                # A follow-up call runs on the restarted worker and succeeds.
                follow = client.call('snapshot', Path(tmp) / 'receipt2.json')
                self.assertTrue(follow.get('success'))
                self.assertIsNone(follow.get('worker_restarted'))
            finally:
                client.close()

    def test_death_without_receipt_still_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            behavior = dict(die_once=True, marker_path=str(Path(tmp) / 'died.marker'),
                            skip_receipt=True)
            client = self.client(behavior, tmp)
            try:
                output = Path(tmp) / 'never.json'
                with self.assertRaises(RuntimeError):
                    client.call('snapshot', output)
                self.assertFalse(output.exists())
            finally:
                client.close()

    def test_zero_tx_evaluation_requires_full_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client({}, tmp)
            try:
                output = Path(tmp) / 'receipt.json'
                for missing in (
                    {'success': True, 'motion_attempted': True,
                     'finger_commands_sent': 0,
                     'tx': {'actual_tx_count': 0, 'transmission_outcome_uncertain': False}},
                    {'success': False, 'motion_attempted': True, 'finger_commands_sent': 0,
                     'tx': {'actual_tx_count': 0, 'transmission_outcome_uncertain': False}},
                    {'success': False, 'motion_attempted': False, 'finger_commands_sent': 1,
                     'tx': {'actual_tx_count': 0, 'transmission_outcome_uncertain': False}},
                    {'success': False, 'motion_attempted': False, 'finger_commands_sent': 0,
                     'tx': {'actual_tx_count': 2, 'transmission_outcome_uncertain': False}},
                    {'success': False, 'motion_attempted': False, 'finger_commands_sent': 0,
                     'tx': {'actual_tx_count': 0, 'transmission_outcome_uncertain': True}},
                ):
                    output.write_text(json.dumps(missing))
                    self.assertFalse(client._zero_tx_failure(missing), missing)
                self.assertTrue(client._zero_tx_failure(ZERO_TX_FAILURE))
                self.assertIsNone(client._dead_worker_receipt(Path(tmp) / 'absent.json'))
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
