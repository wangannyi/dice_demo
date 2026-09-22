"""Bounded read-only retry; no stale limit fallback."""
import unittest
from unittest.mock import Mock, call
from cup_grasp_demo.calibration_debug.shake_readback import query_joint_limit


class ReadbackRetryTest(unittest.TestCase):
    def test_success_does_not_retry(self):
        reply = object()
        getter = Mock(return_value=reply)
        self.assertIs(query_joint_limit(getter, 4), reply)
        getter.assert_called_once_with(4, timeout=.25, min_interval=0)

    def test_missing_reply_retries_once(self):
        reply = object()
        getter = Mock(side_effect=[None, reply])
        self.assertIs(query_joint_limit(getter, 7), reply)
        self.assertEqual(getter.call_args_list, [call(7, timeout=.25, min_interval=0),
                                                call(7, timeout=1., min_interval=0)])

    def test_missing_limits_remain_missing(self):
        getter = Mock(return_value=None)
        self.assertIsNone(query_joint_limit(getter, 1))
        self.assertEqual(getter.call_count, 2)

    def test_transport_error_is_not_suppressed(self):
        getter = Mock(side_effect=RuntimeError('bus error'))
        with self.assertRaisesRegex(RuntimeError, 'bus error'):
            query_joint_limit(getter, 1)
        self.assertEqual(getter.call_count, 1)
