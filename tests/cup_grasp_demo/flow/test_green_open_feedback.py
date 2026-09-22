"""Measured opening reference must not weaken unrelated hand verification."""
import unittest
from cup_grasp_demo.flow.green_hand_execution import apply_open_reference, open_feedback_reference

NAMES = ('thumb_tip', 'thumb_base', 'index_finger', 'middle_finger', 'ring_finger', 'pinky_finger')
REF = [0, 19, 0, 0, 0, 0]


def report(values=REF, stamps=(1, 2, 3)):
    return dict(position_target_reached=False, position_samples=[
        dict(received_epoch_s=s, values=dict(zip(NAMES, values))) for s in stamps])


class OpeningReferenceTest(unittest.TestCase):
    def test_measured_open_keeps_original_result(self):
        r = report()
        apply_open_reference(r, REF, NAMES)
        self.assertTrue(r['position_target_reached'])
        self.assertFalse(r['command_target_reached'])

    def test_wrong_finger_and_stale_feedback_still_fail(self):
        for r in (report([0, 30, 0, 0, 0, 0]), report([0, 19, 20, 0, 0, 0]),
                  report(stamps=(1, 1, 1)), report(stamps=(1, 2))):
            apply_open_reference(r, REF, NAMES)
            self.assertFalse(r['position_target_reached'])

    def test_opt_in_only(self):
        r = report()
        apply_open_reference(r, None, NAMES)
        self.assertFalse(r['position_target_reached'])
        self.assertNotIn('open_reference_reached', r)

    def test_missing_feedback_not_claimed_verified(self):
        r = dict(position_target_reached=None, position_samples=[])
        apply_open_reference(r, REF, NAMES)
        self.assertIsNone(r['position_target_reached'])

    def test_invalid_config(self):
        for v in ([0], [False] * 6, [101] * 6, 'zero'):
            with self.assertRaises(ValueError):
                open_feedback_reference(dict(open_feedback_reference_0_100=v))

class NoHandReadbackTest(unittest.TestCase):
    def test_command_only_still_waits_for_physical_duration(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from cup_grasp_demo.flow.grasp_execution import send_closure
        t=[100.]
        hand=Mock()
        hand.get_finger_pos.side_effect=AssertionError('unexpected position read')
        hand.get_finger_current.side_effect=AssertionError('unexpected current read')
        demo=SimpleNamespace(FINGER_NAMES=NAMES,feedback_stamp=lambda x:None)
        stage=dict(state='CLOSE_FINGERS',target_0_100=[0]*6,duration_s=1.,settle_s=0.)
        report=send_closure(hand,demo,stage,Mock(),monotonic=lambda:t[0],wallclock=lambda:t[0],
                            sleep=lambda dt:t.__setitem__(0,t[0]+dt),read_feedback=False)
        self.assertEqual(hand.position_time_ctrl.call_count,2)
        self.assertGreaterEqual(t[0]-100,1.)
        self.assertIsNone(report['position_target_reached'])
        self.assertIsNone(report['current_after'])
