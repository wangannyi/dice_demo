"""HOME must open the hand even when the arm is already at HOME."""

from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cup_grasp_demo.flow import debug
from cup_grasp_demo.flow.core import ROOT, load_config, read_json
from cup_grasp_demo.flow.home_execution import execute_home, open_before_home


class OpeningTest(unittest.TestCase):
    def setUp(self):
        self.clock, self.sent, self.modes = 1000., [], []
        self.q = [0.] * 7
        self.status = SimpleNamespace(arm_status=0, motion_status=0, ctrl_mode=1)
        self.names = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger')
        self.position = 0
        self.robot = SimpleNamespace(get_joints_enable_status_list=lambda: [True] * 7,
                                    set_motion_mode=self.mode, move_j=self.arm_forbidden)
        self.hand = SimpleNamespace(position_time_ctrl=lambda **kw: self.sent.append(kw),
                                    get_finger_pos=self.feedback, get_finger_current=self.feedback)
        self.demo = SimpleNamespace(
            FINGER_NAMES=self.names, arm_snapshot=lambda _: (self.q, None, self.status),
            require_right_hand=lambda *_: None, read_fresh=lambda getter, *_: getter(),
            feedback_stamp=lambda f: None if f is None else f.timestamp,
            position_values=lambda f: dict.fromkeys(self.names, self.position),
            finger_values=lambda f: dict.fromkeys(self.names, 0))
        self.result = {}

    def feedback(self):
        return SimpleNamespace(timestamp=self.clock)

    def sleep(self, seconds):
        self.clock += seconds

    def mode(self, value):
        self.modes.append(value)
        self.status.ctrl_mode = 1

    def arm_forbidden(self, *_):
        self.fail('Opening at HOME must not issue an arm target')

    def run_open(self):
        return open_before_home(self.robot, self.hand, self.demo, [0.] * 7, self.result,
                            monotonic=lambda: self.clock, wallclock=lambda: self.clock, sleep=self.sleep)

    def test_exact_zero_target_and_duration_without_arm_target(self):
        report = self.run_open()
        self.assertEqual(self.sent, [dict(mode='pos', **dict.fromkeys(self.names, 0)),
                                    dict(mode='time', **dict.fromkeys(self.names, 100))])
        self.assertTrue(report['open_target_reached'])
        self.assertEqual(report['completion_basis'], 'position_feedback')
        self.assertFalse(self.result['home_ready_verified'])
        self.assertEqual(self.result['last_state'], 'OPEN_HAND_VERIFIED')

    def test_web_mode_handoff_precedes_open_without_joint_targets(self):
        self.status.ctrl_mode = 3
        self.run_open()
        self.assertEqual(self.modes, ['js'])
        self.assertEqual(len(self.sent), 2)

    def test_already_home_with_no_positions_sends_open_but_does_not_claim_verified(self):
        self.hand.get_finger_pos = lambda: None
        report = self.run_open()
        self.assertEqual(len(self.sent), 2)
        self.assertFalse(report['open_target_reached'])
        self.assertFalse(self.result['home_ready_verified'])
        self.assertEqual(self.result['last_state'], 'OPEN_HAND_SENT')

    def test_wrong_home_or_lost_can_ownership_prevents_open(self):
        for field in ('joints', 'control', 'fault'):
            self.setUp()
            if field == 'joints':
                self.q[0] = .02
            elif field == 'control':
                self.status.ctrl_mode = 0
            else:
                self.status.arm_status = 7
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.run_open()
            self.assertEqual(self.sent, [])

    def test_reported_closed_fingers_are_a_failure(self):
        self.position = 100
        with self.assertRaisesRegex(RuntimeError, '手指位置反馈'):
            self.run_open()
        self.assertFalse(self.result['home_ready_verified'])

    def test_finger_rebound_is_not_success(self):
        def sleep(seconds):
            self.clock += seconds
            if self.clock > 1001.5:
                self.position = 40
        self.sleep = sleep
        with self.assertRaisesRegex(RuntimeError, '手指位置反馈'):
            self.run_open()

    def test_arm_drift_during_open_aborts(self):
        def sleep(seconds):
            self.clock += seconds
            self.q[0] = .02
        self.sleep = sleep
        with self.assertRaisesRegex(RuntimeError, '停稳'):
            self.run_open()
        self.assertEqual(len(self.sent), 2)

    def replay_positions(self, samples):
        """Replay timestamped events; repeated reads retain the cached timestamp."""
        epoch = self.clock
        def feedback():
            available = [(delay, values) for delay, values in samples
                         if epoch + delay <= self.clock]
            if not available:
                return None
            delay, values = available[-1]
            return SimpleNamespace(timestamp=epoch + delay, values=values)
        self.hand.get_finger_pos = feedback
        self.demo.position_values = lambda f: dict(zip(self.names, f.values))

    def test_recorded_grasp_opening_with_one_terminal_event_then_home(self):
        # 20260918_105101_home_444f3f: relative to the first position TX frame.
        self.replay_positions([
            (.020007, [18, 99, 18, 24, 24, 17]),
            (.169549, [16, 88, 16, 22, 21, 15]),
            (.269640, [14, 78, 14, 19, 18, 13]),
            (.369864, [12, 68, 12, 17, 16, 11]),
            (.469703, [10, 58, 10, 14, 14, 10]),
            (.569930, [9, 53, 9, 13, 12, 9]),
            (.722745, [5, 33, 6, 8, 7, 5]),
            (.822622, [4, 23, 4, 5, 5, 4]),
            (.924360, [2, 13, 2, 3, 3, 2]),
            (1.025265, [0, 3, 0, 0, 0, 0]),
        ])
        self.run_home()
        report = self.result['home_hand_open']
        self.assertEqual(len(self.moves), 1)
        self.assertTrue(self.result['home_ready_verified'])
        self.assertEqual(report['completion_basis'], 'terminal_position_event')
        self.assertEqual(len(report['position_samples']), 10)
        self.assertEqual(report['position_samples'][-1]['error'], 3)

    def test_terminal_feedback_stall_or_rebound_blocks_home(self):
        for samples in (
            [(1.05, [0, 13, 0, 0, 0, 0])],
            [(1.05, [0, 3, 0, 0, 0, 0]), (1.5, [0, 13, 0, 0, 0, 0])],
            [(.2, [0] * 6)],  # No terminal event after the commanded duration.
            [(1.95, [0] * 6)],  # Not yet observed through a settling interval.
        ):
            with self.subTest(samples=samples):
                self.setUp()
                self.replay_positions(samples)
                with self.assertRaisesRegex(RuntimeError, '停止 HOME'):
                    self.run_home()
                self.assertFalse(self.result['home_ready_verified'])
                self.assertEqual(self.moves, [])

    def test_precommand_cached_open_is_not_new_completion_evidence(self):
        self.replay_positions([(-.01, [0] * 6)])
        report = self.run_open()
        self.assertFalse(report['open_target_reached'])
        self.assertEqual(report['position_samples'], [])

    def test_terminal_position_does_not_replace_current_communication(self):
        self.replay_positions([(1.05, [0] * 6)])
        self.hand.get_finger_current = lambda: SimpleNamespace(timestamp=1000.)
        with self.assertRaisesRegex(RuntimeError, 'current feedback'):
            self.run_home()
        self.assertEqual(self.moves, [])
        self.assertFalse(self.result['home_ready_verified'])

    def run_home(self, at_home=False):
        target = [0.] * 7 if at_home else [.2] * 7
        stages = [] if at_home else [dict(name='saved_home', target_q_rad=target)]
        plan = dict(start_q_rad=[0.] * 7, home_target_q_rad=target, stages=stages)
        def move(stage, speed):
            self.assertEqual(len(self.sent), 2)
            self.assertGreaterEqual(self.clock, 1002.)
            self.assertFalse(self.result['home_ready_verified'])
            self.assertEqual(self.result['last_state'], 'TO_HOME')
            self.assertEqual(speed, 5)
            self.q = list(stage['target_q_rad'])
            self.moves.append(stage)
        self.moves = []
        execute_home(plan, {'speed_percent': 5}, self.robot, self.hand, self.demo, move, self.result,
                     monotonic=lambda: self.clock, wallclock=lambda: self.clock, sleep=self.sleep)

    def test_open_wait_then_home_motion_in_that_order(self):
        self.run_home()
        self.assertEqual(len(self.moves), 1)
        self.assertEqual(self.q, [.2] * 7)
        self.assertTrue(self.result['home_joint_target_reached'])
        self.assertTrue(self.result['home_ready_verified'])
        self.assertEqual(self.result['last_state'], 'HOME_OPEN_VERIFIED')
        self.assertEqual(self.result['home_hand_open']['opening_q_rad'], [0.] * 7)

    def test_failed_open_prevents_home_motion(self):
        self.position = 100
        with self.assertRaisesRegex(RuntimeError, '停止 HOME'):
            self.run_home()
        self.assertEqual(self.moves, [])
        self.assertEqual(self.q, [0.] * 7)
        self.assertFalse(self.result['home_joint_target_reached'])
        self.assertFalse(self.result['home_hand_open']['command_wait_completed'])

    def test_home_with_missing_positions_keeps_unverified_result(self):
        self.hand.get_finger_pos = lambda: None
        self.run_home()
        self.assertEqual(len(self.moves), 1)
        self.assertFalse(self.result['home_ready_verified'])
        self.assertTrue(self.result['home_joint_target_reached'])
        self.assertTrue(self.result['home_hand_open']['command_wait_completed'])
        self.assertTrue(self.result['home_hand_open']['arm_target_sent'])
        self.assertEqual(self.result['last_state'], 'HOME_OPEN_SENT')

    def test_missing_positions_do_not_hide_failed_joint_arrival(self):
        self.hand.get_finger_pos = lambda: None
        plan = dict(start_q_rad=[0.] * 7, home_target_q_rad=[.2] * 7, stages=[{}])
        with self.assertRaisesRegex(RuntimeError, '到达 HOME'):
            execute_home(plan, {'speed_percent': 5}, self.robot, self.hand, self.demo,
                         lambda *_: None, self.result, monotonic=lambda: self.clock,
                         wallclock=lambda: self.clock, sleep=self.sleep)
        self.assertFalse(self.result['home_joint_target_reached'])
        self.assertFalse(self.result['home_ready_verified'])

    def test_already_home_opens_without_any_joint_motion(self):
        self.run_home(at_home=True)
        self.assertEqual(len(self.sent), 2)
        self.assertEqual(self.moves, [])
        self.assertTrue(self.result['home_ready_verified'])


class HomeCLITest(unittest.TestCase):
    def exercise(self, *, execute=False, at_home=True, opening=True):
        cfg = load_config(ROOT / 'cup_grasp_demo/flow/index_joint_center/config.json')
        cfg['home_open_hand'] = opening
        target = read_json(cfg['home'])['joints_rad']
        start = deepcopy(target)
        if not at_home:
            start[0] += .02
        snapshot = dict(joints_rad=start, arm_status=0, motion_status=0,
                        joints_enabled=[True] * 7, ctrl_mode=1)
        screen = SimpleNamespace(
            arm=SimpleNamespace(check_joint_path=lambda *_: SimpleNamespace(samples_rad=[start, target], kinematic_checks_passed=True)),
            check=lambda *_: dict(blockers=[]))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            calls = []
            def bridge(command, *args):
                calls.append(command)
                return snapshot if command == 'snapshot' else {'home_hand_open': {'open_target_reached': True}}
            def capture(path, cfg):
                path.mkdir()
                (path / 'frame_000.png').touch()
            meta = dict(intrinsics={'fx': 1, 'fy': 1, 'cx': 0, 'cy': 0, 'frame': 'color_optical',
                                    'height': 2, 'width': 2, 'dist_coeffs': [0] * 5}, depth_scale_m=.001)
            with patch.object(debug, 'load_config', return_value=cfg), \
                 patch.object(debug, 'new_run', return_value=run), \
                 patch.object(debug, 'bridge', side_effect=bridge), \
                 patch.object(debug, 'capture_rgbd', side_effect=capture), \
                 patch.object(debug, 'load_batch', return_value=(meta, np.ones((2, 2)), np.zeros((2, 2, 3), np.uint8), None)), \
                 patch.object(debug, 'camera_transform', return_value=(np.eye(4), True)), \
                 patch.object(debug, '_plane', return_value=(np.zeros(3), np.array([0, 0, 1]), None, None)), \
                 patch.object(debug, 'Screen', return_value=screen), \
                 patch('builtins.input', side_effect=AssertionError('Unexpected confirmation')):
                debug.home(SimpleNamespace(config=Path('unused'), output=run, execute=execute,
                                           cup_removed=True, show=False))
            plan = read_json(run / 'plan.json') if (run / 'plan.json').exists() else None
            request = read_json(run / 'request.json') if (run / 'request.json').exists() else None
            return calls, plan, request

    def test_at_home_still_dispatches_open_with_no_arm_stage(self):
        calls, plan, request = self.exercise(execute=True)
        self.assertEqual(calls, ['snapshot', 'run'])
        self.assertEqual(plan['stages'], [])
        self.assertEqual(request['plan']['open_hand_target_0_100'], [0] * 6)
        self.assertEqual(plan['kind'], 'home_open_debug_plan')
        self.assertEqual(plan['action_sequence'], ['OPEN_HAND', 'TO_HOME'])

    def test_not_at_home_preserves_joint_target_and_adds_open(self):
        _, plan, request = self.exercise(execute=True, at_home=False)
        self.assertEqual(len(plan['stages']), 1)
        self.assertEqual(plan['stages'][0]['target_q_rad'], plan['home_target_q_rad'])
        self.assertTrue(request['execution_authorized'])

    def test_preview_never_dispatches_hand_or_arm_commands(self):
        calls, _, request = self.exercise()
        self.assertEqual(calls, ['snapshot'])
        self.assertIsNone(request)

    def test_disabled_option_preserves_legacy_home_short_circuit(self):
        calls, plan, request = self.exercise(execute=True, opening=False)
        self.assertEqual(calls, ['snapshot'])
        self.assertIsNone(plan)
        self.assertIsNone(request)


if __name__ == '__main__':
    unittest.main()
