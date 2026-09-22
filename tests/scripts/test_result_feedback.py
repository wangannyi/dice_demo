"""Gesture recipes and execution ordering, without hardware."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from scripts import result_feedback as feedback


class FeedbackTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((feedback.ROOT / 'configs/result_feedback.json').read_text())

    def test_user_recipes_and_robot_perspective(self):
        win = feedback.recipe_for(self.config, 'win')
        lose = feedback.recipe_for(self.config, 'lose')
        self.assertEqual(win['joints_deg'], [0.116,-90.329,-20.379,100.034,-9.853,-5.002,5.068])
        self.assertEqual(lose['joints_deg'], [0.110,-80.329,-90.385,100.029,80.136,-4.995,5.064])
        self.assertEqual(win['hand_0_100'], [100,100,0,0,100,100])
        self.assertEqual(lose['hand_0_100'], [0,0,100,100,100,100])

    def test_tie_user_pose_and_sequence(self):
        recipe=feedback.recipe_for(self.config,'draw')
        self.assertEqual(recipe['gesture'],'tie')
        self.assertEqual(recipe['joints_deg'],[.122,-80.481,-90.400,110.059,155.191,-4.993,5.265])
        self.assertEqual(recipe['hand_sequence']['poses'],[[0]*6,[0,0,40,40,40,40]])
        self.assertEqual(recipe['finger_speed_mode'],'max')
        self.assertEqual(recipe['execution'],dict(mode='together',delay_s=0))

    def test_preview_without_runtime(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(feedback.main(['yeah', '--config', '/nonexistent']), 0)

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
