"""Workflow orchestration tests: no camera, CAN or robot motion."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = copy.deepcopy(workflow.read(workflow.ROOT/'configs/calibration_workflow.json'))
        self.cfg.update(output_root=str(self.root/'runs'), state_file=str(self.root/'state.json'),
                        allow_provisional=False, dataset=None, result=None, plan=None, registration=None)
        self.path = self.root/'config.json'
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.cfg))

    def test_invalid_motion_fails_before_subprocess(self):
        self.cfg['motion']['speed_deg_s'] = float('nan')
        self.write()
        with patch('workflow.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError, 'speed_deg_s'):
                workflow.Workflow(self.path)
            run.assert_not_called()

    def test_dry_run_auto_never_changes_files_or_calls_hardware(self):
        self.cfg['plan'] = 'some_plan.json'; self.write()
        w = workflow.Workflow(self.path, True)
        with patch('workflow.subprocess.run') as run:
            w.run('auto')
            run.assert_not_called()
        self.assertFalse(w.state_path.exists())
        self.assertFalse((self.root/'runs').exists())

    def test_auto_requires_explicit_execute(self):
        with patch('workflow.require_idle'), patch('workflow.subprocess.run') as run:
            with self.assertRaisesRegex(ValueError, '--execute'):
                workflow.Workflow(self.path).run('auto')
            run.assert_not_called()

    def test_failed_collection_does_not_solve_or_advance_state(self):
        self.cfg.update(plan=str(self.root/'plan.json'), allow_provisional=True); self.write()
        result = self.root/'result.json'; result.write_text('{"quality_passed":true}')
        (self.root/'plan.json').write_text(json.dumps({'calibration': str(result)}))
        w = workflow.Workflow(self.path)
        with patch('workflow.require_idle'), patch('workflow.subprocess.run') as run:
            run.return_value.returncode = 1
            with self.assertRaises(RuntimeError):
                w.run('auto', True)
            self.assertEqual(run.call_count, 1)
            self.assertIn('--start-from', run.call_args.args[0])
        self.assertFalse(w.state_path.exists())

    def test_generated_state_is_used_and_explicit_config_overrides(self):
        w = workflow.Workflow(self.path)
        w.save(plan='/generated/plan.json', dataset='/generated/data')
        self.assertEqual(w.selected('plan'), Path('/generated/plan.json'))
        self.cfg['plan'] = '/explicit/plan.json'; self.write()
        self.assertEqual(workflow.Workflow(self.path).selected('plan'), Path('/explicit/plan.json'))

    def test_restore_routes_home_and_all_motion_config(self):
        self.cfg['registration'] = '/registered.json'; self.write()
        w = workflow.Workflow(self.path, True)
        with patch.object(w, 'call') as run:
            w.run('restore')
        args = run.call_args.args
        self.assertEqual(args[:2], ('calibration/reference_board.py', 'restore-auto'))
        self.assertIn('--home', args)
        self.assertIn('--execute', args)
        self.assertEqual(args[args.index('--speed-percent')+1], 40)

    def test_apply_rejects_bad_quality_without_mutations(self):
        r = self.root/'bad.json'; r.write_text('{"quality_passed":false}')
        self.cfg['result'] = str(r); self.write()
        w = workflow.Workflow(self.path)
        with patch('workflow.require_idle'), patch.object(w, 'call') as call:
            with self.assertRaisesRegex(ValueError, '质量未通过'):
                w.run('apply')
            call.assert_not_called()
        self.assertFalse((self.root/'runs').exists())

    def test_apply_backs_up_before_install_and_stops_on_table_failure(self):
        result = self.root/'result.json'; result.write_text('{"quality_passed":true}')
        old = self.root/'old.json'; old.write_text('old calibration')
        table = self.root/'table.json'; table.write_text('old table')
        pipeline = self.root/'pipeline.json'
        pipeline.write_text(json.dumps({'calibration':str(old), 'green_cup':{'home_table_scene':str(table)}}))
        self.cfg.update(result=str(result), pipeline_config=str(pipeline)); self.write()
        w = workflow.Workflow(self.path)
        calls = []
        def call(script, *args, **kwargs):
            calls.append(script)
            backup = Path(w.state['backup'])
            self.assertEqual((backup/'handeye_result.json').read_text(), 'old calibration')
            self.assertEqual((backup/'home_table_scene.json').read_text(), 'old table')
            if script == 'scripts/table_capture.py':
                raise RuntimeError('camera busy')
        with patch('workflow.require_idle'), patch.object(w, 'call', side_effect=call):
            with self.assertRaisesRegex(RuntimeError, 'camera busy'):
                w.run('apply')
        self.assertEqual(calls, ['calibration/apply_result.py', 'scripts/table_capture.py'])
        self.assertNotIn('table', w.state)

    def test_first_solves_and_tracks_new_teaching_calibration(self):
        w = workflow.Workflow(self.path)
        def call(script, *args, **kwargs):
            if script.endswith('reprocess_dimensions.py'):
                out = Path(args[args.index('--output')+1]); out.mkdir(parents=True)
                (out/'result.json').write_text('{"quality_passed":true}')
        with patch('workflow.require_idle'), patch.object(w, 'call', side_effect=call):
            w.run('first')
        self.assertEqual(w.state['result'], w.state['teaching_calibration'])
        self.assertEqual(w.state['dataset'], w.state['teaching_dataset'])


if __name__ == '__main__':
    unittest.main()
