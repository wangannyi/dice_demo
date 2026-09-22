"""Cached feedback is explicitly scoped to the owned FAST worker."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from cup_grasp_demo.calibration_debug import hardware
from cup_grasp_demo.calibration_debug.green_runtime import SDKClient


class CachedSnapshotTests(unittest.TestCase):
    def test_background_callback_runs_after_flush_before_receipt_wait(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'out.json';output.write_text('{"success": true}')
            events=[]
            client=object.__new__(SDKClient)
            client.cfg=dict(timeout_s=2)
            client.process=SimpleNamespace(stdin=Mock())
            client.process.stdin.write.side_effect=lambda _:events.append('write')
            client.process.stdin.flush.side_effect=lambda:events.append('flush')
            client.receive=lambda _: (events.append('wait') or dict(output=str(output),returncode=0))
            client.call('snapshot',output,on_dispatched=lambda:events.append('background'))
            self.assertEqual(events,['write','flush','background','wait'])

    def test_standalone_cannot_enable_cached_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'owned persistent'):
                hardware.main(['snapshot','--cached-snapshot','--output',str(Path(d)/'out.json')])

    def test_rpc_flag_respects_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'out.json';output.write_text('{"success": true}')
            for enabled in (False,True):
                client=object.__new__(SDKClient)
                client.cfg=dict(timeout_s=2,green_cup=dict(fast_cached_feedback=enabled))
                client.process=SimpleNamespace(stdin=Mock())
                client.receive=Mock(return_value=dict(output=str(output),returncode=0))
                result=client.call('snapshot',output)
                message=json.loads(client.process.stdin.write.call_args.args[0])
                self.assertEqual(message.get('cached_snapshot',False),enabled)
                self.assertTrue(result['success'])
