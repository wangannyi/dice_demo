import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from rgb_hand_tracking.marker_hand import MarkerGuidedObserver, run


class MarkerObserverTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((400, 600, 3), 255, np.uint8)
        self.corners = [[100, 100], [150, 100], [150, 150], [100, 150]]
        self.points = np.full((21, 2), 125., dtype=float)
        self.points[[0, 5, 9, 13, 17]] = np.array([
            [-.8, .5], [1.5, 0], [1.6, .4], [1.5, .8], [1.3, 1.2]])*50+100
        self.calls = []
        self.closed = []
        outer = self

        class FakeBridge:
            def __init__(self, *args):
                pass

            def infer(self, image, rects):
                outer.calls.append((image, len(rects)))
                return [{'presence': .99, 'landmarks_px': outer.points.tolist()} for _ in rects]

            def close(self):
                outer.closed.append(True)

        self.patch = patch('rgb_hand_tracking.marker_hand.Bridge', FakeBridge)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.observer = MarkerGuidedObserver(Path('worker'), search_mode='omni')
        self.addCleanup(self.observer.close)

    def marker(self, timestamp, *, valid=True, epoch='initial'):
        return {'timestamp_s': timestamp, 'camera_epoch': epoch, 'marker_id': 40,
                'observation_valid': valid,
                'marker_corners_px': copy.deepcopy(self.corners) if valid else None}

    def observe(self, timestamp, *, valid=True, epoch='initial', generation=0):
        return self.observer.observe(self.frame, Path(f'frame_{timestamp}.png'), timestamp,
                                     epoch, marker_observation=self.marker(timestamp, valid=valid, epoch=epoch),
                                     tracking_generation=generation)

    def test_external_confirmed_source_does_not_detect_again_or_reset_at_slow_inference_rate(self):
        with patch.object(self.observer.tracker, 'update', side_effect=AssertionError('Repeated detection')):
            rows = [self.observe(t) for t in (1., 3., 5.)]
        self.assertEqual([r['reason'] for r in rows],
                         ['template_confirming', 'template_confirming', 'visual_template_confirmed'])
        self.assertEqual([count for _, count in self.calls], [36, 1, 1])
        self.assertEqual([r['landmark_calls'] for r in rows], [36, 1, 1])
        self.assertEqual([r['detector_calls'] for r in rows], [0, 0, 0])
        self.assertEqual([r['from_tracking'] for r in rows], [False, True, True])
        self.assertTrue(all(r['processing_ms'] >= 0 for r in rows))
        self.assertEqual(rows[-1]['source_timestamp_s'], 5.)
        self.assertEqual(rows[-1]['marker']['timestamp_s'], 5.)
        self.assertIsNone(rows[-1]['physical_palm_m'])
        self.assertFalse(rows[-1]['motion_target_valid'])
        self.assertEqual(len(rows[-1]['selected_roi_px']), 5)

    def test_external_observation_rejects_other_capture_id_and_epoch_before_inference(self):
        for change in ({'timestamp_s': 2.}, {'marker_id': 7}, {'camera_epoch': 'previous'}):
            marker = self.marker(1.)
            marker.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.observer.observe(self.frame, 'frame.png', 1., 'initial', marker_observation=marker)
        self.assertEqual(self.calls, [])

    def test_hidden_loss_generation_discards_old_roi_and_template_before_reacquisition(self):
        first = [self.observe(float(i)) for i in (1, 2, 3)][-1]
        self.assertTrue(first['valid'])
        self.points += [50, 0]
        rows = [self.observe(float(i), generation=2) for i in (4, 5, 6)]
        self.assertEqual([r['valid'] for r in rows], [False, False, True])
        self.assertEqual([count for _, count in self.calls], [36, 1, 1, 36, 1, 1])
        np.testing.assert_allclose(np.asarray(rows[-1]['stable_visual_palm_px'])-
                                   first['stable_visual_palm_px'], [50, 0], atol=1e-4)

    def test_epoch_change_discards_template_even_if_external_generation_is_unchanged(self):
        self.assertTrue([self.observe(float(i)) for i in (1, 2, 3)][-1]['valid'])
        self.points += [50, 0]
        rows = [self.observe(float(i), epoch='adjusted') for i in (4, 5, 6)]
        self.assertEqual([r['valid'] for r in rows], [False, False, True])
        self.assertEqual(self.calls[3][1], 36)

    def test_missing_marker_immediately_invalidates_and_requires_new_template(self):
        self.assertTrue([self.observe(float(i)) for i in (1, 2, 3)][-1]['valid'])
        invalid = self.observe(4., valid=False)
        self.assertEqual(invalid['reason'], 'marker_unconfirmed')
        self.assertFalse(invalid['valid'])
        self.assertNotIn('stable_visual_palm_px', invalid)
        self.assertEqual(invalid['landmark_calls'], 0)
        self.assertEqual(len(self.calls), 3)
        rows = [self.observe(float(i)) for i in (5, 6, 7)]
        self.assertEqual([r['valid'] for r in rows], [False, False, True])
        self.assertEqual(self.calls[3][1], 36)

    def test_output_copies_marker_snapshot_and_does_not_annotate_source_frame(self):
        marker = self.marker(1.)
        original = self.frame.copy()
        row = self.observer.observe(self.frame, 'frame.png', 1., marker_observation=marker)
        marker['marker_corners_px'][0][0] = 999
        self.assertEqual(row['marker']['marker_corners_px'][0][0], 100)
        np.testing.assert_array_equal(self.frame, original)

    def test_landmark_worker_exception_releases_process_once(self):
        with patch.object(self.observer.worker, 'infer', side_effect=RuntimeError('Worker exited')):
            with self.assertRaisesRegex(RuntimeError, 'Worker exited'):
                self.observe(1.)
        self.observer.close()
        self.assertEqual(self.closed, [True])
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            self.observe(2.)

    def test_existing_output_directory_does_not_leak_worker_or_overwrite_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root/'observations.json'
            results.write_text('original')
            with self.assertRaises(FileExistsError):
                run(root/'missing.json', root, Path('worker'))
            self.assertEqual(results.read_text(), 'original')
        self.assertEqual(self.closed, [True])


if __name__ == '__main__':
    unittest.main()
