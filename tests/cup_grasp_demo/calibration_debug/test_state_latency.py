"""FAST reuses only the current run's recent capture; standalone checks remain."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug import debug, grasp_cli
from cup_grasp_demo.calibration_debug.core import write_json
from test_fast_motion import EVIDENCE
from cup_grasp_demo.calibration_debug.core import read_json


class CaptureReuseTest(unittest.TestCase):
    def test_home_feedback_reused_only_while_fresh(self):
        for observed, expected_reads in ((1000, 0), (998, 1)):
            with self.subTest(observed=observed), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                home = directory/'home.json'; write_json(home, {'joints_rad':[0]*7})
                snapshot = dict(observed_epoch_s=observed, joints_rad=[0]*7, arm_status=0,
                                motion_status=0, joints_enabled=[True]*7, ctrl_mode=1)
                cfg = {'home':str(home)}
                args = SimpleNamespace(config=directory/'cfg.json', session=directory, replay=None)
                with patch.object(debug,'reset_capture'), \
                     patch.object(debug,'load_config',return_value=cfg), \
                     patch.object(debug.time,'time',return_value=1001), \
                     patch.object(debug,'bridge',return_value=snapshot) as bridge, \
                     patch.object(debug,'capture_rgbd',side_effect=RuntimeError('camera starts')):
                    with self.assertRaisesRegex(RuntimeError,'camera starts'):
                        debug.capture(args, snapshot=snapshot)
                self.assertEqual(bridge.call_count, expected_reads)

    def test_capture_proof_must_be_recent_same_session_and_home(self):
        plan = dict(start_state='home', session_sha256='same')
        proof = dict(session_sha256='same', started_monotonic_s=100)
        cfg = dict(fast_capture_reuse_max_age_s=15)
        self.assertTrue(grasp_cli.reusable_capture(plan, cfg, proof, now=114))
        self.assertFalse(grasp_cli.reusable_capture(plan, {}, proof, now=114))
        self.assertFalse(grasp_cli.reusable_capture(plan, cfg, None, now=114))
        for changed, changed_proof, now in (
            (plan, proof, 116), (plan, proof, 99),
            (dict(plan, start_state='ready'), proof, 114),
            (plan, dict(proof, session_sha256='different'), 114),
        ):
            self.assertFalse(grasp_cli.reusable_capture(changed, cfg, changed_proof, now=now))

    def test_recent_capture_skips_camera_and_detection_but_still_executes_sdk(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            cfg = dict(fast_capture_reuse_max_age_s=15)
            plan = dict(read_json(EVIDENCE / 'ready_plan.json'), session_path=str(directory))
            path = directory / 'ready_plan.json'
            write_json(path, plan)
            prepared = dict(plan_sha256=grasp_cli.digest(path))
            proof = dict(session_sha256=plan['session_sha256'], started_monotonic_s=100)
            with patch.object(grasp_cli.time, 'monotonic', return_value=105), \
                 patch.object(debug, 'verify_session', return_value=({'replay_only':False}, cfg)), \
                 patch.object(grasp_cli, 'validate'), \
                 patch.object(debug, 'capture_rgbd', side_effect=AssertionError('duplicate camera')), \
                 patch.object(grasp_cli, 'load_batch', side_effect=AssertionError('duplicate detection')), \
                 patch.object(debug, 'bridge', return_value={'last_state':'READY'}) as sdk:
                grasp_cli.execute(SimpleNamespace(plan=path, execute=True, fast=True, show=False),
                                  confirm=lambda _: 'GRASP', prepared=prepared, capture_proof=proof)
            self.assertEqual(sdk.call_args.args[0], 'run')
            evidence = list((directory/'runs').glob('*/cup_recheck.json'))
            self.assertEqual(len(evidence), 1)
            self.assertFalse(read_json(evidence[0])['independent_recapture'])

    def test_expired_or_missing_capture_reverts_to_live_camera(self):
        for fast, age, supplied in ((True,16,True), (True,1,False), (False,1,True)):
            with self.subTest(fast=fast, age=age, supplied=supplied), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                plan = dict(read_json(EVIDENCE/'ready_plan.json'), session_path=str(directory))
                path = directory/'ready_plan.json'; write_json(path, plan)
                prepared = dict(plan_sha256=grasp_cli.digest(path)) if fast else None
                proof = dict(session_sha256=plan['session_sha256'], started_monotonic_s=100) if supplied else None
                with patch.object(grasp_cli.time,'monotonic', return_value=100+age), \
                     patch.object(debug,'verify_session',return_value=({'replay_only':False},{'fast_capture_reuse_max_age_s':15})), \
                     patch.object(grasp_cli,'validate'), \
                     patch.object(grasp_cli,'observed_scene'), \
                     patch.object(grasp_cli,'make_grasp_plan',return_value={}), \
                     patch.object(debug,'capture_rgbd',side_effect=RuntimeError('fresh camera requested')), \
                     patch.object(debug,'capture_with_feedback',side_effect=RuntimeError('fresh camera requested')):
                    with self.assertRaisesRegex(RuntimeError,'fresh camera requested'):
                        grasp_cli.execute(SimpleNamespace(plan=path,execute=True,fast=fast,show=False),
                                          prepared=prepared,capture_proof=proof)


if __name__ == '__main__':
    unittest.main()
