"""Gesture recipes and execution ordering, without hardware."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from scripts import result_feedback as feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((feedback.ROOT / 'configs/actions/gestures/result_feedback.json').read_text())

    def test_user_recipes_and_robot_perspective(self):
        win = feedback.recipe_for(self.config, 'win')
        lose = feedback.recipe_for(self.config, 'lose')
        self.assertEqual(win['joints_deg'], [0.116,-90.329,-20.379,100.034,-9.853,-5.002,5.068])
        self.assertEqual(lose['joints_deg'], [0.110,-80.329,-90.385,100.029,80.136,-4.995,5.064])
        self.assertEqual(win['hand_0_100'], [100,100,0,0,100,100])
        self.assertEqual(lose['hand_0_100'], [0,0,100,100,100,100])
        self.assertEqual(win['speed_percent'], 100)
        self.assertEqual(lose['speed_percent'], 100)
        self.assertEqual(win['finger_speed_mode'], 'max')
        self.assertEqual(lose['finger_speed_mode'], 'max')

    def test_tie_user_pose_and_sequence(self):
        recipe=feedback.recipe_for(self.config,'draw')
        self.assertEqual(recipe['gesture'],'tie')
        self.assertEqual(recipe['joints_deg'],[.122,-80.481,-90.400,110.059,155.191,-4.993,5.265])
        self.assertEqual(recipe['hand_sequence']['poses'],[[0]*6,[0,0,40,40,40,40]])
        self.assertEqual(recipe['finger_speed_mode'],'max')
        self.assertEqual(recipe['execution'],dict(mode='together',delay_s=0))
        self.assertEqual(recipe['speed_percent'], 100)
        self.assertEqual(recipe['hand_sequence']['interval_s'], .5)

    def test_rock_paper_scissors_recipes(self):
        joints = [40.117, -90.324, -90.391, 79.804, -9.524, -5.203, 4.942]
        # RPS gestures live in the rps.json group since the split; load them
        # through the registry exactly as the runtime does.
        from scripts.action_registry import load_registry
        registry = load_registry()
        rock = registry.recipe('rock')
        paper = registry.recipe('paper')
        scissors = registry.recipe('scissors')
        for recipe in (rock, paper, scissors):
            self.assertEqual(recipe['joints_deg'], joints)
            self.assertEqual(recipe['speed_percent'], 100)
            self.assertEqual(recipe['finger_speed_mode'], 'max')
            self.assertEqual(recipe['execution'], dict(mode='together', delay_s=0))
        self.assertEqual(rock['hand_0_100'], [0, 0, 100, 100, 100, 100])
        self.assertFalse(rock['hand_sequence']['return_to_initial'])
        self.assertEqual(rock['hand_sequence']['interval_s'], .1)
        self.assertEqual(paper['hand_0_100'], [0] * 6)
        self.assertEqual(scissors['hand_0_100'], [100, 100, 0, 0, 100, 100])

    def test_home_recipe_opens_hand_and_returns_to_saved_pose(self):
        home = feedback.recipe_for(self.config, 'home')
        self.assertEqual(home['joints_deg'], [0, -70, -90, 100, -10, -5, 5])
        self.assertEqual(home['hand_0_100'], [0] * 6)
        self.assertEqual(home['execution'], dict(mode='together', delay_s=0))
        self.assertEqual(home['finger_speed_mode'], 'max')

    def test_preview_without_runtime(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(feedback.main(['yeah', '--config', '/nonexistent']), 0)

    def test_interactive_menu_selects_action_for_preview(self):
        with patch.object(feedback, 'choose_action', return_value='paper') as choose:
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(feedback.main([]), 0)
        choose.assert_called_once()
        self.assertIn('"gesture": "paper"', output.getvalue())
        self.assertIn('仅预览', output.getvalue())

    def test_interactive_menu_number_name_and_quit(self):
        from scripts.action_registry import load_registry
        registry = load_registry(feedback.ROOT / 'configs/actions/gestures')
        for entered, expected in [('1', 'win'), ('paper', 'paper'), ('q', None)]:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(feedback.choose_action(registry, lambda _: entered), expected)

    def test_interactive_execute_reuses_connection_and_returns_to_menu(self):
        from scripts.action_registry import load_registry
        registry = load_registry(feedback.ROOT / 'configs/actions/gestures')
        client, planner = Mock(), Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            created = []
            def new_run(session, gesture):
                directory = root / f'{len(created)}_{gesture}'
                directory.mkdir()
                created.append((session, gesture, directory))
                return directory
            with patch.object(feedback, 'choose_action', side_effect=['rock', 'paper', None]), \
                    patch.object(feedback, 'execute_recipe') as execute, \
                    redirect_stdout(io.StringIO()) as output:
                completed = feedback.interactive_execute(
                    registry, {'green_cup': {}}, {}, root, client, planner, new_run)
        self.assertEqual(completed, 2)
        self.assertEqual(execute.call_count, 2)
        self.assertEqual([call.args[0]['gesture'] for call in execute.call_args_list],
                         ['rock', 'paper'])
        self.assertTrue(all(call.args[4] is client for call in execute.call_args_list))
        self.assertEqual([entry[1] for entry in created], ['rock', 'paper'])
        self.assertIn('常驻动作会话已退出', output.getvalue())

    def test_invalid_recipes(self):
        for key, value in [('speed_percent', 101), ('finger_duration_s', float('nan'))]:
            cfg = deepcopy(self.config)
            cfg['gestures']['yeah'][key] = value
            with self.assertRaises(ValueError):
                feedback.recipe_for(cfg, 'yeah')
        for key, value in [('joints_deg', [0]*6), ('hand_0_100', [101]*6)]:
            cfg = deepcopy(self.config)
            cfg['gestures']['yeah'][key] = value
            with self.assertRaises(ValueError):
                feedback.recipe_for(cfg, 'yeah')

    def perform(self, *, blocked=False, fail_arm=False, mode='arm_then_hand'):
        self.config['gestures']['yeah']['execution'] = dict(mode=mode, delay_s=0)
        calls = []
        client = Mock()
        def call(command, output, request=None):
            if command == 'snapshot':
                return dict(arm_status=0, motion_status=0, joints_enabled=[True]*7,
                            ctrl_mode=1, joints_rad=[0]*7)
            payload = json.loads(request.read_text())
            calls.append(payload)
            return dict(success=not fail_arm)
        client.call.side_effect = call
        planner = Mock(return_value=dict(kind='green_arm_plan', blockers=['collision'] if blocked else [], stages=[]))
        with tempfile.TemporaryDirectory() as tmp:
            cfg = dict(green_cup={})
            if blocked or fail_arm:
                with self.assertRaises((ValueError, RuntimeError)):
                    feedback.execute_recipe(feedback.recipe_for(self.config,'yeah'), cfg, {}, Path(tmp), client, planner)
            else:
                feedback.execute_recipe(feedback.recipe_for(self.config,'yeah'), cfg, {}, Path(tmp), client, planner)
                self.assertTrue((Path(tmp)/'receipt.json').exists())
            self.assertEqual(cfg, dict(green_cup={}))
        return calls, planner

    def test_arm_then_hand(self):
        calls, planner = self.perform()
        self.assertEqual([c['plan']['kind'] for c in calls], ['green_arm_plan', 'green_hand_command'])
        self.assertEqual(calls[-1]['plan']['target_0_100'], [100,100,0,0,100,100])
        self.assertAlmostEqual(planner.call_args.args[1][0][1], -90.329 * feedback.math.pi/180)
        self.assertEqual(calls[-1]['config']['green_cup']['finger_duration_s'], .65)
        self.assertTrue(calls[-1]['config']['green_cup']['feedback_hand_max_speed'])
        self.assertEqual(calls[-1]['config']['green_cup']['grip_targets_0_100'], calls[-1]['plan']['target_0_100'])

    def test_legacy_timed_mode(self):
        self.config.pop('finger_speed_mode')
        for g in self.config['gestures'].values():
            g.pop('finger_speed_mode')
        self.assertEqual(feedback.recipe_for(self.config, 'yeah')['finger_speed_mode'], 'timed')

    def test_custom_action_and_delete(self):
        self.config['gestures']['hello'] = self.config['gestures'].pop('yeah')
        self.config['aliases']['win'] = 'hello'
        self.assertEqual(feedback.recipe_for(self.config, 'win')['gesture'], 'hello')
        with self.assertRaises(ValueError):
            feedback.recipe_for(self.config, 'yeah')

    def test_hand_first(self):
        calls, _ = self.perform(mode='hand_then_arm')
        self.assertEqual([c['plan']['kind'] for c in calls], ['green_hand_command', 'green_arm_plan'])

    def test_together_one_request(self):
        calls, _ = self.perform(mode='together')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['plan']['kind'], 'feedback_together')

    def test_invalid_timing(self):
        for execution in [dict(mode='oops'), dict(delay_s=-1), dict(delay_s=float('nan'))]:
            self.config['gestures']['yeah']['execution'] = execution
            with self.assertRaises(ValueError):
                feedback.recipe_for(self.config, 'yeah')

    def test_blocked_path_sends_nothing(self):
        calls, _ = self.perform(blocked=True)
        self.assertEqual(calls, [])

    def test_failed_arm_does_not_send_hand(self):
        calls, _ = self.perform(fail_arm=True)
        self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
