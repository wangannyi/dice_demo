"""The application protocol advances phases without reopening hardware."""

from contextlib import nullcontext
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cup_grasp_demo.flow import green_control as control


def commands(*items):
    return io.StringIO(''.join(json.dumps(item) + '\n' for item in items))


def events(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class ControlSessionTest(unittest.TestCase):
    def fake_flow(self):
        flow = Mock()
        flow.receipts = {}
        flow.recovery_events = []
        flow.held = None
        flow._snapshot_cache = None
        return flow

    def test_waits_after_target_and_advances_only_on_next_command(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(id='one', command='advance', until='GRIP'),
                    dict(id='check', command='status'),
                    dict(id='two', command='advance'),
                    dict(id='end', command='close')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 0)
            self.assertEqual([call.args[0] for call in flow.perform.call_args_list],
                             list(control.PHASES[:6]))
            reports = events(outgoing)
            self.assertEqual(reports[0]['event'], 'ready')
            check = next(item for item in reports if item.get('id') == 'check')
            self.assertEqual(check['next_phase'], 'LIFT')
            self.assertEqual(check['status'], 'WAITING')
            self.assertEqual(reports[-1]['event'], 'closed')
            self.assertEqual(json.loads((Path(directory) / 'state.json').read_text())['status'], 'PAUSED')

    def test_no_skip_replay_or_unexpected_new_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(id=1, command='advance', until='GRIP'),
                    dict(id=1, command='advance'),
                    dict(id=2, command='advance', until='HOME'),
                    dict(id=3, command='new_cycle'),
                    dict(id=4, command='close')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 0)
            self.assertEqual(flow.perform.call_count, 5)
            self.assertEqual([x['code'] for x in events(outgoing) if x['event'] == 'rejected'],
                             ['duplicate_id', 'already_completed', 'cycle_in_progress'])

    def test_refresh_only_before_approach_and_new_cycle_reuses_flow(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', until='PLAN'),
                    dict(command='refresh_perception'),
                    dict(command='advance', until='RETURN_HOME'),
                    dict(command='new_cycle'),
                    dict(command='advance'),
                    dict(command='close')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 0)
            names = [call.args[0] for call in flow.perform.call_args_list]
            self.assertEqual(names[:3], ['HOME', 'CAPTURE', 'PLAN'])
            self.assertEqual(names[3:5], ['CAPTURE', 'PLAN'])
            self.assertEqual(names[-1], 'HOME')
            self.assertEqual(names.count('RETURN_HOME'), 1)
            self.assertEqual(server.cycle, 2)

    def test_failure_stops_following_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            flow.perform.side_effect = [None, RuntimeError('camera failed')]
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(id='go', command='advance', until='GRIP')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 2)
            self.assertEqual(flow.perform.call_count, 2)
            self.assertEqual(events(outgoing)[-1]['event'], 'failed')
            self.assertEqual(json.loads((Path(directory) / 'state.json').read_text())['status'], 'FAILED')

    def test_phase_logs_do_not_pollute_json_events(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            flow.perform.side_effect = lambda phase: print('hardware log for ' + phase)
            outgoing, diagnostic = io.StringIO(), io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json',
                commands(dict(command='advance'), dict(command='close')),
                outgoing, diagnostic)
            self.assertEqual(server.serve(), 0)
            self.assertIn('hardware log for HOME', diagnostic.getvalue())
            self.assertNotIn('hardware log', outgoing.getvalue())
            self.assertEqual(events(outgoing)[-1]['event'], 'closed')

    def test_run_opens_resources_once_before_ready_and_closes_on_command(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(status=False, resume=False, show=False,
                                   until='place', execute=True, mode='control',
                                   session=Path(directory), config=Path(directory) / 'config.json')
            flow = self.fake_flow()
            flow.g = {'installation_requires_calibration': False}
            output = io.StringIO()
            with patch.object(control, 'Workflow', return_value=flow) as factory, \
                 patch.object(control, 'cached_screen_geometry', return_value=nullcontext()):
                self.assertEqual(control.run(args, incoming=commands(
                    dict(id='stop', command='close')),
                    outgoing=output, diagnostic=io.StringIO()), 0)
            factory.assert_called_once()
            flow.prepare_step_runtime.assert_called_once()
            flow.perform.assert_not_called()
            flow.close.assert_called_once()
            self.assertEqual([item['event'] for item in events(output)], ['ready', 'closed'])


if __name__ == '__main__':
    unittest.main()
