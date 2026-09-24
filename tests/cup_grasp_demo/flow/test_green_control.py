"""The application protocol advances phases without reopening hardware."""

from contextlib import nullcontext
import io
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cup_grasp_demo.flow import green_control as control


def commands(*items):
    return io.StringIO(''.join(json.dumps(item) + '\n' for item in items))


def events(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


class ControlSessionTest(unittest.TestCase):
    def test_home_action_uses_the_registry_recipe(self):
        """home 不再有内建 recipe（旧分支硬编码 30%/timed，把 9be7d7a 的
        100% 现场调参静默遮蔽了）：统一走注册表，names 不再硬编码塞 home。"""
        with tempfile.TemporaryDirectory() as tmp:
            green = dict(home_table_scene='table.json', open_targets_0_100=[0] * 6)
            flow = SimpleNamespace(
                cfg=dict(green_cup=green, home='home.json', calibration='cal.json'),
                root=tmp, _sdk=Mock())
            recipe = dict(gesture='home', joints_deg=[0] * 7, hand_0_100=[0] * 6,
                          speed_percent=100, finger_duration_s=0.5,
                          execution=dict(mode='together', delay_s=0.0),
                          finger_speed_mode='max', finger_max_wait_s=0.65)
            registry = Mock(errors=[])
            registry.names.return_value = ['home', 'yeah']
            registry.recipe.return_value = recipe
            with patch('scripts.action_registry.load_registry', return_value=registry), \
                 patch('cup_grasp_demo.flow.core.read_json', side_effect=[
                     dict(calibration_sha256='hash', scene={}),
                     dict(joints_deg=[0] * 7)]), \
                 patch('cup_grasp_demo.flow.core.digest', return_value='hash'), \
                 patch('scripts.result_feedback.execute_recipe') as execute:
                names, run = control.build_action_runtime(flow, io.StringIO())
                self.assertEqual(names, ['home', 'yeah'])
                run('home')
            execute.assert_called_once()
            self.assertIs(execute.call_args.args[0], recipe)

    def test_home_action_warns_when_registry_lacks_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            green = dict(home_table_scene='table.json')
            flow = SimpleNamespace(
                cfg=dict(green_cup=green, home='home.json', calibration='cal.json'),
                root=tmp, _sdk=Mock())
            registry = Mock(errors=[])
            registry.names.return_value = ['yeah']
            diagnostic = io.StringIO()
            with patch('scripts.action_registry.load_registry', return_value=registry), \
                 patch('cup_grasp_demo.flow.core.read_json', side_effect=[
                     dict(calibration_sha256='hash', scene={})]), \
                 patch('cup_grasp_demo.flow.core.digest', return_value='hash'):
                names, _ = control.build_action_runtime(flow, diagnostic)
            self.assertEqual(names, ['yeah'])
            self.assertIn('缺少 home', diagnostic.getvalue())

    def test_home_action_warns_when_home_json_drifts_from_registry(self):
        """home.json 是阶段机 HOME 姿态来源，注册表 home 是 action 姿态来源——
        关节角分叉只警告不拒绝（分叉 = 两条路去到不同姿态）。"""
        with tempfile.TemporaryDirectory() as tmp:
            green = dict(home_table_scene='table.json')
            flow = SimpleNamespace(
                cfg=dict(green_cup=green, home='home.json', calibration='cal.json'),
                root=tmp, _sdk=Mock())
            recipe = dict(gesture='home', joints_deg=[1] * 7)
            registry = Mock(errors=[])
            registry.names.return_value = ['home']
            registry.recipe.return_value = recipe
            diagnostic = io.StringIO()
            with patch('scripts.action_registry.load_registry', return_value=registry), \
                 patch('cup_grasp_demo.flow.core.read_json', side_effect=[
                     dict(calibration_sha256='hash', scene={}),
                     dict(joints_deg=[0] * 7)]), \
                 patch('cup_grasp_demo.flow.core.digest', return_value='hash'):
                _, _ = control.build_action_runtime(flow, diagnostic)
            self.assertIn('joints_deg 不一致', diagnostic.getvalue())

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
                             ['duplicate_id', 'already_completed', 'removed'])

    def test_query_pose_reports_at_home_and_failures_as_rejected(self):
        """只读姿态探针（game 侧巡检归位的判定依据）：pose 事件带 at_home；
        snapshot 失败必须 rejected（探针绝不杀会话、不进 failed）。"""
        home_deg = [0, -70, -90, 100, -10, -5, 5]
        at_home_rad = [math.radians(x) for x in home_deg]
        away_rad = [math.radians(x + 30) for x in home_deg]
        for joints, expect_home in ((at_home_rad, True), (away_rad, False)):
            with self.subTest(expect_home=expect_home), \
                    tempfile.TemporaryDirectory() as tmp:
                flow = SimpleNamespace(cfg=dict(home='home.json'), root=tmp, _sdk=Mock())
                flow._sdk.call.return_value = dict(success=True, joints_rad=joints)
                outgoing = io.StringIO()
                server = control.ControlSession(
                    flow, Path(tmp) / 'state.json',
                    commands(dict(id='p', command='query_pose'),
                             dict(command='close')),
                    outgoing, io.StringIO())
                with patch('cup_grasp_demo.flow.core.read_json',
                           return_value=dict(joints_deg=home_deg)), \
                     patch('cup_grasp_demo.flow.debug.new_run',
                           return_value=Path(tmp) / 'pose_run'):
                    self.assertEqual(server.serve(), 0)
                pose = next(x for x in events(outgoing) if x['event'] == 'pose')
                self.assertEqual(pose['at_home'], expect_home)
                self.assertEqual(len(pose['delta_deg']), 7)
        # CAN/worker 断 → rejected + 会话存活
        with tempfile.TemporaryDirectory() as tmp:
            flow = SimpleNamespace(cfg=dict(home='home.json'), root=tmp, _sdk=Mock())
            flow._sdk.call.side_effect = RuntimeError('SDK worker exited')
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(tmp) / 'state.json',
                commands(dict(id='q', command='query_pose'),
                         dict(command='close')),
                outgoing, io.StringIO())
            with patch('cup_grasp_demo.flow.core.read_json',
                       return_value=dict(joints_deg=home_deg)), \
                 patch('cup_grasp_demo.flow.debug.new_run',
                       return_value=Path(tmp) / 'pose_run'):
                self.assertEqual(server.serve(), 0)
            rejected = next(x for x in events(outgoing) if x['event'] == 'rejected')
            self.assertEqual(rejected['code'], 'pose_unavailable')
            self.assertNotIn('pose', [x['event'] for x in events(outgoing)])

    def test_return_home_auto_resets_without_new_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', until='PLAN'),
                    dict(command='refresh_perception'),
                    dict(command='advance', until='RETURN_HOME'),
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
            reports = events(outgoing)
            completed = [x for x in reports if x['event'] == 'run_completed']
            self.assertEqual(len(completed), 1)
            self.assertEqual(completed[0]['runs'], 1)
            after = [x for x in reports if x['event'] == 'command_completed'
                     and x.get('through') == 'RETURN_HOME']
            self.assertEqual(after[0]['next_phase'], 'HOME')

    def test_action_dispatches_in_idle_and_records_history(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            seen = []
            def fake_run_action(name):
                seen.append(name)
                if name == 'nope':
                    raise ValueError('未知动作：nope；可选：yeah')
                return dict(name=name, receipt='/tmp/receipt.json', elapsed_s=1.5)
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='actions'),
                    dict(command='action', name='win'),
                    dict(command='action', name='nope'),
                    dict(command='close')),
                outgoing, io.StringIO(),
                actions=['home', 'yeah', 'win'], run_action=fake_run_action)
            self.assertEqual(server.serve(), 0)
            reports = events(outgoing)
            listed = next(x for x in reports if x['event'] == 'actions')
            self.assertEqual(listed['names'], ['home', 'win', 'yeah'])
            self.assertEqual(seen, ['win', 'nope'])
            started = next(x for x in reports if x['event'] == 'action_started')
            self.assertEqual(started['name'], 'win')
            done = next(x for x in reports if x['event'] == 'action_completed')
            self.assertEqual((done['name'], done['receipt']), ('win', '/tmp/receipt.json'))
            rejected = [x for x in reports if x['event'] == 'rejected']
            self.assertEqual(rejected[0]['code'], 'unknown_action')
            self.assertIn('yeah', rejected[0]['message'])
            state = json.loads((Path(directory) / 'state.json').read_text())
            self.assertEqual(state['actions'][0]['name'], 'win')

    def test_action_rejected_while_flow_in_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            def fake_run_action(name):
                raise AssertionError('must not run mid-flow')
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', until='GRIP'),
                    dict(command='action', name='home'),
                    dict(command='close')),
                outgoing, io.StringIO(),
                actions=['home'], run_action=fake_run_action)
            self.assertEqual(server.serve(), 0)
            rejected = [x for x in events(outgoing) if x['event'] == 'rejected']
            self.assertEqual(rejected[0]['code'], 'flow_in_progress')
            self.assertIn('LIFT', rejected[0]['message'])
            ready = events(outgoing)[0]
            self.assertEqual(ready['actions'], ['home'])

    def test_action_without_runtime_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='action', name='yeah'),
                    dict(command='close')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 0)
            rejected = [x for x in events(outgoing) if x['event'] == 'rejected']
            self.assertEqual(rejected[0]['code'], 'no_actions')

    def test_reload_swaps_action_table_only_when_idle(self):
        def fresh_actions():
            return ['home', 'wave'], Mock()

        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='reload'),
                    dict(command='actions'),
                    dict(command='close')),
                outgoing, io.StringIO(),
                actions=['home', 'yeah'], run_action=Mock(),
                reload_actions=fresh_actions)
            self.assertEqual(server.serve(), 0)
            reports = events(outgoing)
            reloaded = next(x for x in reports if x['event'] == 'actions_reloaded')
            self.assertEqual(reloaded['names'], ['home', 'wave'])
            listed = next(x for x in reports if x['event'] == 'actions')
            self.assertEqual(listed['names'], ['home', 'wave'])

    def test_reload_rejected_while_flow_in_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', until='GRIP'),
                    dict(command='reload'),
                    dict(command='close')),
                outgoing, io.StringIO(),
                reload_actions=lambda: (['home'], Mock()))
            self.assertEqual(server.serve(), 0)
            rejected = [x for x in events(outgoing)
                        if x['event'] == 'rejected']
            self.assertEqual(rejected[0]['code'], 'flow_in_progress')

    def test_reload_failure_keeps_previous_table(self):
        def broken():
            raise RuntimeError('gesture directory missing')

        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='reload'),
                    dict(command='actions'),
                    dict(command='close')),
                outgoing, io.StringIO(),
                actions=['home', 'yeah'], run_action=Mock(),
                reload_actions=broken)
            self.assertEqual(server.serve(), 0)
            reports = events(outgoing)
            rejected = next(x for x in reports if x['event'] == 'rejected')
            self.assertEqual(rejected['code'], 'reload_failed')
            listed = next(x for x in reports if x['event'] == 'actions')
            self.assertEqual(listed['names'], ['home', 'yeah'])

    def test_multi_round_runs_every_round_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', until='RETURN_HOME', rounds=3),
                    dict(command='close')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 0)
            self.assertEqual(flow.perform.call_count, 3 * len(control.PHASES))
            reports = events(outgoing)
            starts = [x for x in reports if x['event'] == 'round_started']
            self.assertEqual([x['round'] for x in starts], [1, 2, 3])
            self.assertEqual(len([x for x in reports if x['event'] == 'run_completed']), 3)
            done = next(x for x in reports if x['event'] == 'rounds_completed')
            self.assertEqual(done['rounds'], 3)
            command = next(x for x in reports
                           if x['event'] == 'command_completed' and 'rounds_completed' in x)
            self.assertEqual(command['rounds_completed'], 3)
            # StringIO has no fileno: between-round peek is a no-op, all rounds run.

    def test_multi_round_rejects_invalid_rounds_and_idle_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', rounds=0),
                    dict(command='advance', rounds=True),
                    dict(command='advance', rounds=100),
                    dict(command='stop'),
                    dict(command='close')),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 0)
            codes = [x['code'] for x in events(outgoing) if x['event'] == 'rejected']
            self.assertEqual(codes, ['invalid_rounds', 'invalid_rounds', 'invalid_rounds',
                                     'not_in_multi_round'])
            self.assertEqual(flow.perform.call_count, 0)

    def test_multi_round_failure_ends_rounds_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            # Round 1 fine; round 2 fails on its first phase.
            flow.perform.side_effect = [None] * len(control.PHASES) + [RuntimeError('boom')]
            outgoing = io.StringIO()
            server = control.ControlSession(
                flow, Path(directory) / 'state.json', commands(
                    dict(command='advance', until='RETURN_HOME', rounds=3)),
                outgoing, io.StringIO())
            self.assertEqual(server.serve(), 2)
            self.assertEqual(flow.perform.call_count, len(control.PHASES) + 1)
            reports = events(outgoing)
            self.assertEqual(len([x for x in reports if x['event'] == 'round_started']), 2)
            self.assertEqual(reports[-1]['event'], 'failed')

    def _pipe_session(self, lines, flow, directory):
        """ControlSession over a real pipe so select-based peeking works.

        Both commands are written up front, then the writer closes: the pipe
        delivers them in order and EOF lets serve() exit once it runs dry.
        The advance loop consumes the queued stop/close between rounds.
        """
        import os
        read_fd, write_fd = os.pipe()
        incoming = os.fdopen(read_fd, 'r', buffering=1)
        writer = os.fdopen(write_fd, 'w', buffering=1)
        for line in lines:
            writer.write(json.dumps(line) + '\n')
        writer.close()
        outgoing = io.StringIO()
        server = control.ControlSession(
            flow, Path(directory) / 'state.json', incoming, outgoing, io.StringIO())
        return server, outgoing

    def test_multi_round_stop_between_rounds_parks_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            server, outgoing = self._pipe_session(
                [dict(command='advance', until='RETURN_HOME', rounds=3),
                 dict(command='stop')],
                flow, directory)
            self.assertEqual(server.serve(), 0)
            reports = events(outgoing)
            stopped = next(x for x in reports if x['event'] == 'rounds_stopped')
            self.assertEqual(stopped['status'], 'WAITING')
            self.assertLess(flow.perform.call_count, 3 * len(control.PHASES))
            self.assertGreaterEqual(flow.perform.call_count, len(control.PHASES))

    def test_multi_round_close_between_rounds_exits(self):
        with tempfile.TemporaryDirectory() as directory:
            flow = self.fake_flow()
            server, outgoing = self._pipe_session(
                [dict(command='advance', until='RETURN_HOME', rounds=3),
                 dict(command='close')],
                flow, directory)
            self.assertEqual(server.serve(), 0)
            reports = events(outgoing)
            self.assertEqual(reports[-1]['event'], 'closed')
            self.assertLess(flow.perform.call_count, 3 * len(control.PHASES))

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
