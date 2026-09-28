"""Limits reads survive a transiently busy CAN bus: one retry, configurable budget.

Field evidence (game_20260924_165900, run 20260924_170554_green_lower_36ed0a):
the pre-MoveJS limits poll missed five replies inside the hard-coded 0.5 s
deadline and killed the whole LOWER action with zero commands sent, while the
failure hold right after re-read every limit in 85 ms — the bus was only busy
for a moment.
"""

import unittest

from cup_grasp_demo.flow.batched_limits import read_limits
from cup_grasp_demo.flow.joint_delivery import delivery_options, read_delivery_limits


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

    def test_retry_and_session_cache_options_preserve_legacy_defaults(self):
        options = delivery_options()
        self.assertEqual(options['limits_read_retries'], 1)
        self.assertIs(options['cache_live_limits'], False)
        configured = delivery_options({
            'limits_read_retries': 3,
            'cache_live_limits': True,
        })
        self.assertEqual(configured['limits_read_retries'], 3)
        self.assertIs(configured['cache_live_limits'], True)

    def test_retry_and_session_cache_options_are_strict(self):
        for value in (-1, 6, 1.5, True, '3'):
            with self.subTest(retries=value), self.assertRaises(ValueError):
                delivery_options({'limits_read_retries': value})
        for value in (0, 1, None, 'yes'):
            with self.subTest(cache=value), self.assertRaises(ValueError):
                delivery_options({'cache_live_limits': value})


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

    def test_configured_retries_recover_from_a_longer_transient(self):
        clock = Clock()
        robot = FakeRobot(clock, ready_after=1001.2)
        pairs = read_limits(robot, timeout_s=.5, retries=3,
                            sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(len(pairs), 7)
        self.assertGreaterEqual(clock.now, 1001.2)

    def test_retry_argument_is_strict(self):
        clock = Clock()
        robot = FakeRobot(clock)
        for value in (-1, 6, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                read_limits(robot, retries=value,
                            sleep=clock.sleep, monotonic=clock.monotonic)


class SessionCacheTests(unittest.TestCase):
    def test_second_action_reuses_complete_limit_table(self):
        clock = Clock()
        robot = FakeRobot(clock)
        options = delivery_options({'cache_live_limits': True})
        first, first_source = read_delivery_limits(
            robot, options, sleep=clock.sleep, monotonic=clock.monotonic)
        robot.ready_after = 1e9
        second, second_source = read_delivery_limits(
            robot, options, sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(first, second)
        self.assertEqual(first_source, 'live')
        self.assertEqual(second_source, 'session_cache')

    def test_incomplete_cache_is_ignored(self):
        clock = Clock()
        robot = FakeRobot(clock)
        robot._dice_demo_movejs_limit_pairs = [(1, 2)]
        pairs, source = read_delivery_limits(
            robot, delivery_options({'cache_live_limits': True}),
            sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(len(pairs), 7)
        self.assertEqual(source, 'live')


class LostQueryRobot(FakeRobot):
    """Simulate SDK throttling and a dropped first J7 acceleration response."""

    def __init__(self, clock, drop_all=False):
        super().__init__(clock)
        self.sent = {}
        self.last = {}
        self.drop_all = drop_all

    def _read(self, kind, joint, min_interval):
        key = (kind, joint)
        if key not in self.last or self.clock.now - self.last[key] >= min_interval:
            self.last[key] = self.clock.now
            self.sent[key] = self.sent.get(key, 0) + 1
        if key != ('acc', 7):
            return (joint, self.clock.now)
        if self.drop_all or self.sent[key] < 2 or self.clock.now - self.last[key] < .04:
            return None
        return (joint, self.clock.now)

    def get_joint_angle_vel_limits(self, joint, *, timeout, min_interval):
        return self._read('angle', joint, min_interval)

    def get_joint_acc_limits(self, joint, *, timeout, min_interval):
        return self._read('acc', joint, min_interval)


class LostReplyRetryTests(unittest.TestCase):
    def test_missing_query_is_actually_resent_before_retry_budget(self):
        clock = Clock()
        robot = LostQueryRobot(clock)
        pairs = read_limits(robot, sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(len(pairs), 7)
        self.assertEqual(robot.sent[('acc', 7)], 2)
        self.assertTrue(all(count == 1 for key, count in robot.sent.items() if key != ('acc', 7)))
        self.assertLess(clock.now, 1000.7)

    def test_second_lost_reply_still_stops_with_missing_joint(self):
        clock = Clock()
        robot = LostQueryRobot(clock, drop_all=True)
        with self.assertRaisesRegex(TimeoutError, 'J7 acceleration'):
            read_limits(robot, sleep=clock.sleep, monotonic=clock.monotonic)
        self.assertEqual(robot.sent[('acc', 7)], 2)
        self.assertLess(clock.now, 1001.1)
