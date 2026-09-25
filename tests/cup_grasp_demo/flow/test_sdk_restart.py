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
        print('[worker] exit: close command', file=sys.stderr, flush=True)
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
    # fail_first: the first command answers a failure receipt and exits the
    # loop cleanly (mirrors the real worker: failed motion never leaves a
    # reusable executor), like hardware.main returning a failed receipt.
    if behavior.get('fail_first') and marker and not os.path.exists(marker):
        with open(marker, 'w') as m:
            m.write('failed')
        with open(output, 'w') as f:
            json.dump(behavior.get('fail_receipt',
                                   {'success': False, 'error': 'simulated motion failure'}), f)
        print(json.dumps({'output': output, 'returncode': 2}), flush=True)
        print('[worker] exit: failed motion code=2', file=sys.stderr, flush=True)
        break
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

    def test_closed_pipe_write_is_classified_as_worker_exit(self):
        """P0-1：管道已关后的写入失败归类为 SDK worker exited，不再以裸
        ValueError 漏出（曾把 'I/O operation on closed file' 误判成
        unknown_action，形成秒拒僵尸态）。"""
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client({}, tmp)
            client.close()
            with self.assertRaisesRegex(RuntimeError, 'SDK worker exited \\(write failed\\)'):
                client.call('snapshot', Path(tmp) / 'receipt.json')

    def test_receipt_failure_does_not_terminate_the_worker(self):
        """P1-3：回执失败（worker 正常上报动作失败）不触发父进程收线——
        worker 按自身安全设计自然退出，父进程不打断它的 disconnect。"""
        import subprocess
        import time as _t
        with tempfile.TemporaryDirectory() as tmp:
            behavior = dict(fail_first=True, marker_path=str(Path(tmp) / 'failed.marker'))
            client = self.client(behavior, tmp)
            try:
                output = Path(tmp) / 'receipt.json'
                with self.assertRaisesRegex(RuntimeError, 'simulated motion failure'):
                    client.call('run', output)
                # The worker exits by itself (no parent SIGINT needed to make
                # room); the receipt it wrote stays on disk.
                self.assertEqual(client.process.wait(timeout=5), 0)
                self.assertEqual(json.loads(output.read_text())['success'], False)
            finally:
                client.close()

    def test_worker_clean_exits_are_logged(self):
        """P1-4：三个干净出口（close/failed/eof）各留一行 stderr 归因痕迹。"""
        import cup_grasp_demo.flow.green_sdk_worker as worker_module
        source = Path(worker_module.__file__).read_text(encoding='utf-8')
        self.assertEqual(source.count("'[worker] exit: "), 3)
        for kind in ('close command', 'failed motion code=', 'eof'):
            self.assertIn(f'[worker] exit: {kind}', source)

    def test_restart_terminates_a_still_alive_worker(self):
        """P0-2 配套：restart 对活进程先收掉，避免泄漏持有非阻塞控制锁的
        旧 worker（否则后续每个新 worker 都会启动失败）。"""
        with tempfile.TemporaryDirectory() as tmp:
            client = self.client({}, tmp)
            try:
                self.assertIsNone(client.process.poll())  # alive
                client.restart()
                self.assertEqual(client._log_index, 2)
                # The replaced worker answers on the fresh pipes.
                report = client.call('snapshot', Path(tmp) / 'receipt.json')
                self.assertTrue(report.get('success'))
            finally:
                client.close()


if __name__ == '__main__':
    unittest.main()
