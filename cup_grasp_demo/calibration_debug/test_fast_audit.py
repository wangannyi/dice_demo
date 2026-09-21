"""FAST capture contract, immutable table reuse and changing-file invalidation."""

from pathlib import Path
import os
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from cup_grasp_demo.calibration_debug import core, debug, grasp_cli, pipeline_home
from cup_grasp_demo.calibration_debug.test_fast_motion import EVIDENCE


class FastAuditTest(unittest.TestCase):
    def test_digest_cache_checks_changes_replacement_and_context_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'large-model'
            path.write_bytes(b'aaaa')
            original = Path.read_bytes
            reads = []
            def read(p):
                reads.append(p)
                return original(p)
            with patch.object(Path, 'read_bytes', read), core.cached_file_digests():
                with patch.object(core.time, 'time_ns', return_value=core.time.time_ns()+3_000_000_000):
                    first = core.digest(path)
                    self.assertEqual(first, core.digest(path))
                    self.assertEqual(len(reads), 1)
                stamp = path.stat().st_mtime_ns
                path.write_bytes(b'bbbb')
                os.utime(path, ns=(stamp, stamp))
                self.assertNotEqual(first, core.digest(path))
                replacement = path.with_suffix('.new')
                replacement.write_bytes(b'cccc'); replacement.replace(path)
                self.assertNotEqual(first, core.digest(path))
                self.assertEqual(len(reads), 3)
            with patch.object(Path, 'read_bytes', read):
                core.digest(path)
            self.assertEqual(len(reads), 4)

    def test_table_reuse_is_bound_to_camera_calibration_and_valid_live_session(self):
        cfg = dict(serial='camera', calibration='calibration', plane_tolerance_mm=5,
                   allow_provisional_calibration=True)
        scene = dict(cup_support_base_m=[0., .5, 0.], cup_normal_base=[0., 0., 1.],
                     cup_envelope_radius_m=.04, geometry=dict(height_m=.1))
        session = dict(replay_only=False, scene=scene, calibration_quality_passed=False)
        with patch.object(debug, 'verify_session', return_value=(session, cfg)), \
             patch.object(debug, 'digest', side_effect=lambda p: str(p)):
            cached = pipeline_home.cached_home_scene(Path('/not-read'), cfg)
            self.assertIsNotNone(cached)
            self.assertEqual(cached[0]['geometry']['height_m'], 0)
            for changed in (dict(serial='different'), dict(calibration='changed'),
                            dict(plane_tolerance_mm=3), dict(allow_provisional_calibration=False)):
                self.assertIsNone(pipeline_home.cached_home_scene(Path('/not-read'), dict(cfg, **changed)))
            session['replay_only'] = True
            self.assertIsNone(pipeline_home.cached_home_scene(Path('/not-read'), cfg))
        with patch.object(debug, 'verify_session', side_effect=ValueError('changed sources')):
            self.assertIsNone(pipeline_home.cached_home_scene(Path('/not-read'), cfg))

    def test_fast_home_reads_joints_without_camera_using_valid_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            home = directory / 'home.json'; core.write_json(home, dict(joints_rad=[0.]*7))
            cfg = dict(home=str(home), fast_reuse_table=True)
            scene = dict(cup_support_base_m=[0., .5, 0.], cup_normal_base=[0., 0., 1.])
            snapshot = dict(joints_rad=[0.]*7)
            actual = dict(success=True, home_joint_target_reached=True, home_ready_verified=True)
            def bridge(command, output, *_):
                data = snapshot if command == 'snapshot' else actual
                core.write_json(output, data)
                return data
            with patch.object(pipeline_home, 'cached_home_scene', return_value=(scene, True)), \
                 patch.object(debug, 'capture_with_feedback', side_effect=AssertionError('extra camera')), \
                 patch.object(debug, 'ready'), patch.object(debug, 'bridge', side_effect=bridge) as sdk, \
                 patch.object(pipeline_home, 'make_home_plan', return_value={'blockers': []}):
                result = pipeline_home.execute_home(directory, cfg, fast=True)
            self.assertEqual([c.args[0] for c in sdk.call_args_list], ['snapshot', 'run'])
            self.assertEqual(core.read_json(result['plan_path'])['table_source'], 'verified_previous_capture')

    def test_single_capture_expiry_stops_without_second_detection_or_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            plan = dict(core.read_json(EVIDENCE / 'ready_plan.json'), session_path=str(directory))
            path = directory / 'ready_plan.json'; core.write_json(path, plan)
            cfg = dict(fast_single_capture=True, fast_capture_reuse_max_age_s=120)
            prepared = dict(plan_sha256=core.digest(path))
            proof = dict(session_sha256=plan['session_sha256'], started_monotonic_s=0)
            with patch.object(grasp_cli.time, 'monotonic', return_value=121), \
                 patch.object(debug, 'verify_session', return_value=({'replay_only': False}, cfg)), \
                 patch.object(grasp_cli, 'validate'), \
                 patch.object(debug, 'capture_rgbd', side_effect=AssertionError('extra camera')), \
                 patch.object(grasp_cli, 'verify_cup', side_effect=AssertionError('extra YOLO')), \
                 patch.object(debug, 'bridge', side_effect=AssertionError('motion before valid capture')):
                with self.assertRaisesRegex(ValueError, '本模式不重复'):
                    grasp_cli.execute(NS(plan=path, execute=True, fast=True, show=False),
                                      prepared=prepared, capture_proof=proof)


if __name__ == '__main__':
    unittest.main()
