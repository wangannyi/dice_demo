"""Feedback age policy stays strict by default and is configurable for held-cup shake."""

import unittest

from cup_grasp_demo.flow import joint_execution
from cup_grasp_demo.flow.joint_stream import FeedbackReader


def feedback(stamp):
    return {
        'sdk_snapshot': {
            'packet_timestamps_after_epoch_s': {
                'joint_12': stamp,
                'joint_34': stamp,
                'joint_56': stamp,
                'joint_7': stamp,
            },
        },
        'status_timestamp_epoch_s': stamp,
        'enable_feedback_timestamps_epoch_s': [stamp] * 7,
    }


class FeedbackFreshnessTests(unittest.TestCase):
    def test_expanded_limit_is_restricted_to_held_cup_pipeline(self):
        self.assertEqual(
            joint_execution.feedback_freshness_limit({
                'load_context': 'green_cup_held',
                'feedback_freshness_limit_s': .3,
            }),
            .3,
        )
        with self.assertRaisesRegex(ValueError, 'only for green held-cup'):
            joint_execution.feedback_freshness_limit({
                'feedback_freshness_limit_s': .3,
            })

    def test_default_still_rejects_feedback_older_than_100_ms(self):
        reader = FeedbackReader(lambda **_: None, feedback(9.85))
        with self.assertRaisesRegex(RuntimeError, '100 ms'):
            reader.latest(10.0)

    def test_held_cup_limit_accepts_short_scheduling_gap(self):
        reader = FeedbackReader(
            lambda **_: None,
            feedback(9.85),
            joint_max_age_s=.3,
            state_max_age_s=.3,
        )
        self.assertIs(reader.latest(10.0), reader.row)

    def test_configured_limit_still_rejects_a_real_feedback_outage(self):
        reader = FeedbackReader(
            lambda **_: None,
            feedback(9.6),
            joint_max_age_s=.3,
            state_max_age_s=.3,
        )
        with self.assertRaisesRegex(RuntimeError, '300 ms'):
            reader.latest(10.0)


if __name__ == '__main__':
    unittest.main()
