import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from cup_grasp_demo.flow import green_pipeline as g

class FastOverheadTest(unittest.TestCase):
    def test_fast_finger_override_preserves_step_duration(self):
        cfg=g.ROOT/'cup_grasp_demo/flow/green_open_cup/stereo_config.json'
        with tempfile.TemporaryDirectory() as d, patch.object(g,'digest',return_value='test'), patch.object(g.Workflow,'file_stamp',return_value=(1,)):
            fast=g.Workflow(SimpleNamespace(config=cfg,session=Path(d),mode='fast'))
            step=g.Workflow(SimpleNamespace(config=cfg,session=Path(d),mode='step'))
        self.assertEqual(fast.cfg['speed_percent'],fast.g.get('fast_speed_percent',100))
        self.assertEqual(fast.g['finger_settle_s'],0)
        self.assertGreater(step.g['finger_settle_s'],0)
        self.assertEqual(fast.g['finger_duration_s'],step.g.get('fast_finger_duration_s', step.g['finger_duration_s']))
        self.assertEqual(step.g['finger_duration_s'], g.read_json(cfg)['green_cup']['finger_duration_s'])

    def test_fresh_receipt_is_reused(self):
        for age,reads in [(0,0),(2,1)]:
            wf=object.__new__(g.Workflow);wf.args=SimpleNamespace(mode='fast')
            wf.root=Path('/tmp');wf.cfg={}
            wf._snapshot_cache=dict(observed_epoch_s=time.time()-age,joints_rad=[.1]*7)
            with patch.object(g.common,'ready'),patch.object(g.common,'new_run',return_value=Path('/tmp/run')),patch.object(g.common,'bridge',return_value=dict(joints_rad=[.2]*7)) as bridge:
                q=wf.snapshot()
            self.assertEqual(bridge.call_count,reads)
            self.assertEqual(q,[.2 if reads else .1]*7)

    def test_changed_file_still_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config';p.write_text('old')
            wf=object.__new__(g.Workflow);wf.args=SimpleNamespace(mode='fast')
            wf.hashes={str(p):g.digest(p)};wf._file_stats={str(p):wf.file_stamp(p)}
            with patch.object(g,'digest',wraps=g.digest) as digest, patch.object(g.time, 'time_ns', return_value=time.time_ns()+3_000_000_000):
                wf.unchanged();digest.assert_not_called()
            p.write_text('new contents')
            with self.assertRaises(ValueError):wf.unchanged()

    def test_return_has_two_endpoints_one_execution(self):
        wf=object.__new__(g.Workflow);wf.args=SimpleNamespace(mode='fast')
        wf.snapshot=Mock(return_value=[0]*7);wf.tcp=object();wf.home=[.2]*7
        wf.g=dict(retreat_clearance_mm=50,wrist_reference_deg=[0,-13,5]);wf.move=Mock()
        with patch.object(g,'vertical_targets',return_value=[[.1]*7]) as vertical:
            wf.perform('RETURN_HOME')
        self.assertTrue(vertical.call_args.kwargs['single_target'])
        wf.move.assert_called_once_with([[.1]*7,wf.home],'return_home')


class ShakeStartTest(unittest.TestCase):
    def test_small_settling_residual_and_real_start_change(self):
        import math
        from cup_grasp_demo.flow import joint_execution as execution
        req=dict(load_context='green_cup_held',start_tolerance_deg=.5,plan=dict(start_q_rad=[0]*7))
        row=dict(q_rad=[math.radians(.16)]*7)
        with patch.object(execution.core,'ready_blockers',return_value=[]):
            report={};execution.check_start_rows(req,[row],report)
            self.assertAlmostEqual(report['start_max_error_deg'],.16)
            with self.assertRaises(RuntimeError):
                execution.check_start_rows(dict(plan=req['plan']),[row],{})
            with self.assertRaises(RuntimeError):
                execution.check_start_rows(req,[dict(q_rad=[math.radians(.51)]*7)],{})
        with patch.object(execution.core,'ready_blockers',return_value=['fault']):
            with self.assertRaises(RuntimeError):execution.check_start_rows(req,[row],{})
        for value in (True,0,1,float('nan')):
            with self.assertRaises(ValueError):execution.start_tolerance(dict(req,start_tolerance_deg=value))
