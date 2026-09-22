"""Worker reuse requires a certified transmission-free shake start rejection."""
import unittest
from copy import deepcopy
from cup_grasp_demo.flow.green_sdk_worker import retryable_shake_start

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
        from cup_grasp_demo.flow import green_sdk_worker as worker
        from cup_grasp_demo.flow import hardware, joint_execution
        robot=Mock();demo=SimpleNamespace(create_robot=Mock(return_value=robot))
        # shake_execution 持有真实的 visual_servo_probe 引用（包属性不随
        # sys.modules patch 变化），evidence 必须满足 evidence_blockers 的
        # 真实校验：连接前无 CAN 接收器，连接后恰为 SDK 默认两条注册行。
        observe_calls=[]
        def fake_evidence(channel):
            observe_calls.append(channel)
            if len(observe_calls)==1:
                return {'errors':[], 'channel':channel, 'receiver_rows':[],
                        'candidate_control_processes':[]}
            line='00000000 00000000'
            return {'errors':[], 'channel':channel, 'candidate_control_processes':[],
                    'receiver_rows':[{'list':'all','line':f'{channel} 000 00000000 {line}'},
                                     {'list':'err','line':f'{channel} 000 1fffffff {line}'}]}
        probe=SimpleNamespace(control_lock=lambda path:nullcontext(),host_control_evidence=fake_evidence,evidence_blockers=lambda *args:[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);request=root/'request.json';request.write_text('{}')
            messages=[dict(command='shake',request=str(request),sha256=hashlib.sha256(b'{}').hexdigest(),output=str(root/f'out{i}.json')) for i in range(2)]
            messages.append(dict(command='close'))
            out=io.StringIO()
            # sys.modules patch 在包属性已被先前测试设置时会被
            # `from 包 import 子模块` 的 getattr 绕过，需同时 patch 包属性。
            import nero_revo2_control
            import nero_revo2_control.bridges
            with patch.dict(sys.modules,{'nero_revo2_control.nero_revo2_demo':demo,'nero_revo2_control.bridges.visual_servo_probe':probe}),patch.object(nero_revo2_control,'nero_revo2_demo',demo,create=True),patch.object(nero_revo2_control.bridges,'visual_servo_probe',probe,create=True),patch.object(sys,'argv',['worker','--channel','can0']),patch.object(sys,'stdin',io.StringIO(''.join(json.dumps(m)+'\n' for m in messages))),patch.object(worker.signal,'signal'),patch.object(joint_execution,'run',side_effect=[dict(self.report(),success=False),dict(success=True)]) as run,redirect_stdout(out):
                worker.main()
            self.assertEqual(run.call_count,2)
            demo.create_robot.assert_called_once()
            robot.connect.assert_called_once();robot.disconnect.assert_called_once()
            replies=[json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual([x.get('returncode') for x in replies],[None,2,0])
