import unittest
from types import SimpleNamespace
from cup_grasp_demo.calibration_debug.batched_limits import read_limits


class BatchedLimitsTests(unittest.TestCase):
    def run_reader(self, missing=False):
        now = [0.]
        calls = {}
        def getter(kind):
            def read(joint, *, timeout, min_interval):
                self.assertEqual(timeout, 0.)
                key = kind, joint
                calls[key] = calls.get(key, 0) + 1
                if calls[key] == 1:
                    self.assertEqual(min_interval, 0.)
                    return None
                self.assertEqual(min_interval, 1.)
                return None if missing and key == ('acc', 7) else key
            return read
        robot = SimpleNamespace(get_joint_angle_vel_limits=getter('vel'), get_joint_acc_limits=getter('acc'))
        result = read_limits(robot, sleep=lambda dt: now.__setitem__(0, now[0]+dt), monotonic=lambda: now[0])
        return result, calls

    def test_collects_delayed_replies_without_requerying_completed_axes(self):
        pairs, calls = self.run_reader()
        self.assertEqual(pairs, [(('vel', j), ('acc', j)) for j in range(1, 8)])
        self.assertEqual(set(calls.values()), {2})

    def test_missing_axis_never_returns_partial_limits(self):
        with self.assertRaisesRegex(TimeoutError, 'J7 acceleration'):
            self.run_reader(missing=True)
