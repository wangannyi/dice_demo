"""Limits reads survive a transiently busy CAN bus: one retry, configurable budget.

Field evidence (game_20260924_165900, run 20260924_170554_green_lower_36ed0a):
the pre-MoveJS limits poll missed five replies inside the hard-coded 0.5 s
deadline and killed the whole LOWER action with zero commands sent, while the
failure hold right after re-read every limit in 85 ms — the bus was only busy
for a moment.
"""

import unittest

from cup_grasp_demo.flow.batched_limits import read_limits
from cup_grasp_demo.flow.joint_delivery import delivery_options


class Clock:
    """A fake monotonic clock that only advances through the injected sleep."""

    def __init__(self):
        self.now = 1000.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeRobot:
    """Getters return values only once the clock passes ready_after."""

    def __init__(self, clock, ready_after=None):
        self.clock = clock
        self.ready_after = ready_after

    def _value(self, joint):
        if self.ready_after is None or self.clock.now >= self.ready_after:
            return (joint, self.clock.now)
        return None

    def get_joint_angle_vel_limits(self, joint, **kwargs):
        return self._value(joint)

    def get_joint_acc_limits(self, joint, **kwargs):
        return self._value(joint)


class DeliveryOptionsTests(unittest.TestCase):
    def test_new_key_defaults_to_half_second(self):
        self.assertEqual(delivery_options()['limits_read_timeout_s'], .5)

    def test_new_key_accepts_override(self):
        options = delivery_options({'limits_read_timeout_s': 1.})
        self.assertEqual(options['limits_read_timeout_s'], 1.)

    def test_new_key_rejects_out_of_range(self):
        for value in (.05, 5.1, True, 'fast'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                delivery_options({'limits_read_timeout_s': value})


class ReadLimitsRetryTests(unittest.TestCase):
    def test_ready_bus_never_enters_retry(self):
        clock = Clock()
        robot = FakeRobot(clock)
        pairs = read_limits(robot, timeout_s=.5,
                            sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(len(pairs), 7)
        self.assertLess(clock.now, 1000.1)

    def test_busy_bus_is_retried_once_and_recovers(self):
        clock = Clock()
        robot = FakeRobot(clock, ready_after=1000.6)
        pairs = read_limits(robot, timeout_s=.5,
                            sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(len(pairs), 7)
        # The first 0.5 s budget expired once; the values landed inside the retry.
        self.assertGreaterEqual(clock.now, 1000.5)

    def test_dead_bus_raises_after_exactly_two_budgets(self):
        clock = Clock()
        robot = FakeRobot(clock, ready_after=1e9)
        with self.assertRaisesRegex(TimeoutError, 'after one retry'):
            read_limits(robot, timeout_s=.5,
                        sleep=clock.sleep, monotonic=clock.monotonic)
        # Burst (~0.014 s) plus two full 0.5 s budgets, never a third.
        self.assertGreaterEqual(clock.now, 1001.)
        self.assertLess(clock.now, 1001.1)

    def test_configured_budget_is_honoured(self):
        clock = Clock()
        robot = FakeRobot(clock, ready_after=1e9)
        with self.assertRaisesRegex(TimeoutError, 'after one retry'):
            read_limits(robot, timeout_s=.1,
                        sleep=clock.sleep, monotonic=clock.monotonic)
        # Two 0.1 s budgets, not two 0.5 s ones.
        self.assertLess(clock.now, 1000.21 + .05)
