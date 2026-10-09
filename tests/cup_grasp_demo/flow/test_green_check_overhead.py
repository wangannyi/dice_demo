import unittest
from unittest.mock import Mock, patch
from cup_grasp_demo.flow import green_pipeline as flow

class FileScanTest(unittest.TestCase):
    def workflow(self, interval):
        w = object.__new__(flow.Workflow)
        w.g = {'fast_file_check_interval_s': interval}
        w.hashes = {'config': 'old'}
        w._file_stats = {'config': (1, 2, 3, 4, 5)}
        w.file_stamp = Mock(return_value=(1, 2, 3, 4, 5))
        return w

    def test_duplicate_scan_is_coalesced_then_change_detected(self):
        w = self.workflow(.25)
        with patch.object(flow.time, 'monotonic', side_effect=[10.,10.01,10.3]), patch.object(flow, 'digest', return_value='new'):
            w.unchanged()
            w.file_stamp.return_value = (1, 2, 6, 7, 8)
            w.unchanged()
            self.assertEqual(w.file_stamp.call_count, 1)
            with self.assertRaisesRegex(ValueError, '变化'):
                w.unchanged()
        self.assertEqual(w.file_stamp.call_count, 2)

    def test_default_checks_every_call(self):
        w = self.workflow(0)
        w.unchanged(); w.unchanged()
        self.assertEqual(w.file_stamp.call_count, 2)

    def test_round_boundary_forces_scan_inside_interval(self):
        w = self.workflow(.25)
        with patch.object(flow.time, 'monotonic', side_effect=[10.,10.01]), patch.object(flow, 'digest', return_value='new'):
            w.unchanged()
            w.file_stamp.return_value = (1, 2, 6, 7, 8)
            with self.assertRaisesRegex(ValueError, '变化'):
                w.unchanged(force=True)
