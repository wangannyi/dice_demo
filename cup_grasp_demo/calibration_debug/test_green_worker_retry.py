"""Worker reuse requires a certified transmission-free shake start rejection."""
import unittest
from copy import deepcopy
from cup_grasp_demo.calibration_debug.green_sdk_worker import retryable_shake_start

class WorkerRetryTest(unittest.TestCase):
    def report(self):
        return dict(failure_code='start_position_changed', motion_attempted=False,
                    finger_commands_sent=0, parameter_write_commands_sent=0,
                    tx=dict(actual_tx_count=0,transmission_outcome_uncertain=False))

    def test_only_certified_pre_motion_failure_is_reusable(self):
        r=self.report()
        self.assertTrue(retryable_shake_start('shake',r))
        self.assertFalse(retryable_shake_start('run',r))
        for key,value in [('failure_code','communication'),('motion_attempted',True),
                          ('finger_commands_sent',1),('parameter_write_commands_sent',1)]:
            a=deepcopy(r);a[key]=value
            self.assertFalse(retryable_shake_start('shake',a))
        for key in r:
            a=deepcopy(r);del a[key]
            self.assertFalse(retryable_shake_start('shake',a))
        for key,value in [('actual_tx_count',1),('transmission_outcome_uncertain',True)]:
            a=deepcopy(r);a['tx'][key]=value
            self.assertFalse(retryable_shake_start('shake',a))

    def test_worker_processes_replanned_request_on_same_connection(self):
        import hashlib
        import io
        import json
        from pathlib import Path
        import sys
        import tempfile
        from contextlib import nullcontext, redirect_stdout
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from cup_grasp_demo.calibration_debug import green_sdk_worker as worker
        from cup_grasp_demo.calibration_debug import hardware, joint_execution
        robot=Mock();demo=SimpleNamespace(create_robot=Mock(return_value=robot))
        probe=SimpleNamespace(control_lock=lambda path:nullcontext(),host_control_evidence=lambda channel:{},evidence_blockers=lambda *args:[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request=root/'request.json';request.write_text('{}')
            messages=[dict(command='shake',request=str(request),sha256=hashlib.sha256(b'{}').hexdigest(),output=str(root/f'out{i}.json')) for i in range(2)]
            messages.append(dict(command='close'))
            out=io.StringIO()
            with patch.dict(sys.modules,{'nero_revo2_demo':demo,'visual_servo_probe':probe}),patch.object(sys,'argv',['worker','--channel','can0']),patch.object(sys,'stdin',io.StringIO(''.join(json.dumps(m)+'\n' for m in messages))),patch.object(worker.signal,'signal'),patch.object(joint_execution,'run',side_effect=[dict(self.report(),success=False),dict(success=True)]) as run,redirect_stdout(out):
                worker.main()
            self.assertEqual(run.call_count,2)
            demo.create_robot.assert_called_once()
            robot.connect.assert_called_once();robot.disconnect.assert_called_once()
            replies=[json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual([x.get('returncode') for x in replies],[None,2,0])
