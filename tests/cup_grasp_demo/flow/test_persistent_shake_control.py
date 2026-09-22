"""A reused CAN connection must not repeat the standalone handoff."""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from cup_grasp_demo.flow import joint_execution as execution


class PersistentControlTests(unittest.TestCase):
    def setUp(self):
        self.request = dict(load_context='green_cup_held', start_tolerance_deg=.5,
                            plan=dict(start_q_rad=[0.] * 7))
        self.baseline = dict(q_rad=[0.] * 7, enabled=[True] * 7,
                             status=dict(arm_status=0, ctrl_mode=1, motion_status=0))
        self.session = SimpleNamespace(robot=object())

    def prepare(self, fresh, persistent=True):
        with patch.object(execution.core, 'fresh_feedback', return_value=fresh), \
             patch.object(execution, 'take_js_control') as handoff:
            result = execution.prepare_control(self.request, self.session, self.baseline, {},
                                               persistent=persistent)
            if persistent:
                handoff.assert_not_called()
            return result, handoff

    def test_settling_within_pipeline_tolerance_reuses_can(self):
        fresh = dict(self.baseline, q_rad=[math.radians(.28)] + [0.] * 6)
        result, _ = self.prepare(fresh)
        self.assertFalse(result['mode_command_sent'])
        self.assertTrue(result['reused_can_control'])

    def test_control_fault_motion_or_large_drift_stops(self):
        for fresh in [dict(self.baseline, q_rad=[math.radians(.6)] + [0.] * 6),
                      dict(self.baseline, enabled=[False] * 7),
                      *[dict(self.baseline, status=dict(self.baseline['status'], **change))
                        for change in [dict(arm_status=1), dict(ctrl_mode=3), dict(motion_status=1)]]]:
            with self.subTest(fresh=fresh), self.assertRaises(RuntimeError):
                self.prepare(fresh)

    def test_standalone_preserves_handoff(self):
        _, handoff = self.prepare(self.baseline, persistent=False)
        handoff.assert_called_once()


if __name__ == '__main__':
    unittest.main()
