import unittest
import json
import tempfile
import math
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from marker_hand import candidate_rects, acceptable, normalized_palm


class Tests(unittest.TestCase):
    @staticmethod
    def synthetic_frames(root, marker_id, count, epochs=None):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, 'generateImageMarker'):
            code = cv2.aruco.generateImageMarker(dictionary, marker_id, 50)
        else:
            code = cv2.aruco.drawMarker(dictionary, marker_id, 50)
        image = np.full((400, 600, 3), 255, np.uint8)
        image[100:150, 100:150] = code[:, :, None]
        entries = []
        for i in range(count):
            cv2.imwrite(str(root/f'frame_{i}.png'), image)
            entries.append({'image_path': f'frame_{i}.png', 'timestamp_s': i*.4,
                            'camera_epoch': epochs[i] if epochs else 'first'})
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps(entries))
        return manifest

    def synthetic_run(self, root, manifest, *, shift_after=None, fail_selected_at=None, **options):
        from marker_hand import run
        points = np.full((21, 2), 125., dtype=float)
        points[[0, 5, 9, 13, 17]] = np.array([
            [-.8, .5], [1.5, 0], [1.6, .4], [1.5, .8], [1.3, 1.2]])*50+100
        calls = []
        class FakeBridge:
            def __init__(self, *args):
                pass
            def infer(self, image, rects):
                index = int(image.stem.split('_')[-1])
                calls.append((index, len(rects)))
                pts = points.copy()
                if shift_after is not None and index >= shift_after:
                    pts += [50, 0]
                presence = .0 if index == fail_selected_at and len(rects) == 1 else .99
                return [{'presence': presence, 'landmarks_px': pts.tolist()} for _ in rects]
            def close(self):
                pass
        with patch('marker_hand.Bridge', FakeBridge), patch(
                'marker_hand.cv2.aruco.drawDetectedMarkers', wraps=cv2.aruco.drawDetectedMarkers) as draw:
            run(manifest, root/'out', root/'unused', **options)
        rows = json.loads((root/'out/observations.json').read_text())
        return rows, calls, [int(call.args[2][0, 0]) for call in draw.call_args_list]

    def test_default_search_preserves_all_legacy_candidates(self):
        corners = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], float)
        params, rects = candidate_rects(corners)
        expected_params = [(u, scale, delta) for u in (.5, 1.25, 2.)
                           for scale in (6., 8., 10.) for delta in (-.35, 0., .35)]
        self.assertEqual(params, expected_params)
        self.assertEqual(len(rects), 27)
        np.testing.assert_allclose(np.asarray(rects)[:, :2],
                                   np.repeat([[125, 125], [162.5, 125], [200, 125]], 9, axis=0))
        np.testing.assert_allclose(np.asarray(rects)[:, 2:4],
                                   np.tile(np.repeat([[300, 300], [400, 400], [500, 500]], 3, axis=0), (3, 1)))
        np.testing.assert_allclose(np.asarray(rects)[:, 4],
                                   np.tile([math.pi/2-.35, math.pi/2, math.pi/2+.35], 9))

    def test_omni_search_spans_full_rotation_at_observed_marker_center(self):
        corners = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], float)
        params, rects = candidate_rects(corners, search_mode='omni')
        self.assertEqual(len(rects), 36)
        self.assertEqual(set(p[0] for p in params), {.5})
        self.assertEqual(set(p[1] for p in params), {6., 8., 10.})
        np.testing.assert_allclose(np.asarray(rects)[:, :2], np.tile([125, 125], (36, 1)))
        for scale in (6., 8., 10.):
            angles = sorted(rect[4] for param, rect in zip(params, rects) if param[1] == scale)
            self.assertEqual(len(angles), 12)
            np.testing.assert_allclose(np.diff(angles), math.pi/6)
            self.assertAlmostEqual(angles[-1]-angles[0], 11*math.pi/6)

    def test_omni_selected_candidate_follows_marker_translation_scale_rotation(self):
        corners = np.array([[100, 100], [150, 100], [150, 150], [100, 150]], float)
        params, _ = candidate_rects(corners, search_mode='omni')
        selected = params[19]
        _, initial = candidate_rects(corners, selected, search_mode='omni')
        angle = .4
        rotation = np.array([[math.cos(angle), -math.sin(angle)],
                             [math.sin(angle), math.cos(angle)]])
        changed = corners@rotation.T*1.7+[80, 40]
        parameters, updated = candidate_rects(changed, selected, search_mode='omni')
        self.assertEqual(parameters, [selected])
        self.assertEqual(len(updated), 1)
        np.testing.assert_allclose(updated[0][:2], np.asarray(initial[0][:2])@rotation.T*1.7+[80, 40])
        np.testing.assert_allclose(updated[0][2:4], np.asarray(initial[0][2:4])*1.7)
        self.assertAlmostEqual(updated[0][4], initial[0][4]+angle)

    def test_low_capture_rate_needs_explicit_gap_for_real_marker_acquisition(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            default = root/'default'
            default.mkdir()
            manifest = self.synthetic_frames(default, 40, 6)
            rows, calls, _ = self.synthetic_run(default, manifest)
            self.assertTrue(all(not r['valid'] for r in rows))
            self.assertTrue(all(r['reason'] == 'marker_unconfirmed' for r in rows))
            self.assertEqual(calls, [])
            slower = root/'slower'
            slower.mkdir()
            manifest = self.synthetic_frames(slower, 40, 6)
            rows, calls, _ = self.synthetic_run(slower, manifest, max_gap_s=1., search_mode='omni')
            self.assertFalse(rows[3]['valid'])
            self.assertTrue(rows[4]['valid'])
            self.assertEqual(calls, [(2, 36), (3, 1), (4, 1), (5, 1)])
            self.assertTrue(all(r['physical_palm_m'] is None and not r['motion_target_valid'] for r in rows))

    def test_nondefault_marker_id_reaches_tracker_and_drawing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.synthetic_frames(root, 7, 6)
            rows, _, drawn_ids = self.synthetic_run(root, manifest, marker_id=7, max_gap_s=1.)
            self.assertTrue(rows[-1]['valid'])
            self.assertTrue(all(r['marker']['marker_id'] == 7 for r in rows))
            self.assertEqual(drawn_ids, [7]*4)

    def test_epoch_change_clears_confirmed_template_and_reacquires(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.synthetic_frames(root, 40, 10, ['first']*5+['second']*5)
            rows, calls, _ = self.synthetic_run(root, manifest, shift_after=5,
                                              max_gap_s=1., search_mode='omni')
            self.assertTrue(rows[4]['valid'])
            self.assertTrue(all(not rows[i]['valid'] for i in range(5, 9)))
            self.assertEqual(rows[7]['reason'], 'template_confirming')
            self.assertTrue(rows[9]['valid'])
            np.testing.assert_allclose(np.asarray(rows[9]['stable_visual_palm_px'])-
                                       rows[4]['stable_visual_palm_px'], [50, 0], atol=1e-4)
            self.assertIn((7, 36), calls)

    def test_failed_selected_candidate_triggers_bounded_full_search(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = self.synthetic_frames(root, 40, 6)
            rows, calls, _ = self.synthetic_run(root, manifest, fail_selected_at=3,
                                              max_gap_s=1., search_mode='omni')
            self.assertEqual(calls, [(2, 36), (3, 1), (3, 36), (4, 1), (5, 1)])
            self.assertTrue(rows[4]['valid'])

    def test_invalid_options_fail_before_starting_bridge_or_creating_output(self):
        from marker_hand import run
        invalid = ([{'marker_id': x} for x in (True, False, -1, 50, 4.5, '7', None)]
                   + [{'max_gap_s': x} for x in (True, 0, -1, float('nan'), float('inf'), '1', None)]
                   + [{'search_mode': x} for x in ('all', '', None)])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with patch('marker_hand.Bridge') as bridge:
                for options in invalid:
                    with self.subTest(options=options), self.assertRaises(ValueError):
                        run(root/'missing.json', root/'out', root/'unused', **options)
                    self.assertFalse((root/'out').exists())
                bridge.assert_not_called()

    def test_outlier_does_not_redefine_palm(self):
        from marker_hand import run
        corners = [[100,100],[150,100],[150,150],[100,150]]
        points = np.full((21,2),125.,dtype=float)
        points[[0,5,9,13,17]] = np.array([[-.8,.5],[1.5,0],[1.6,.4],[1.5,.8],[1.3,1.2]])*50+100
        class FakeBridge:
            calls = 0
            def __init__(self, *args):
                pass
            def infer(self, image, rects):
                pts = points.copy()
                if self.calls == 3:
                    pts += [60,0]
                self.calls += 1
                return [{'presence':.99,'landmarks_px':pts.tolist()} for _ in rects]
            def close(self):
                pass
        marker = {'observation_valid':True,'marker_corners_px':corners}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cv2.imwrite(str(root/'frame.png'),np.full((400,600,3),255,np.uint8))
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps([{'image_path':'frame.png','timestamp_s':i/30} for i in range(7)]))
            with patch('marker_hand.Bridge',FakeBridge), patch('marker_hand.MarkerTracker.update',return_value=marker):
                run(manifest,root/'out',root/'unused')
            rows=json.loads((root/'out/observations.json').read_text())
            self.assertTrue(rows[2]['valid'])
            self.assertEqual(rows[3]['reason'],'palm_geometry_disagreement')
            self.assertFalse(rows[4]['valid'])
            self.assertFalse(rows[5]['valid'])
            self.assertTrue(rows[6]['valid'])
            np.testing.assert_allclose(rows[2]['stable_visual_palm_px'],rows[6]['stable_visual_palm_px'])

    def test_roi_moves_scales_with_marker(self):
        c=np.array([[100,100],[150,100],[150,150],[100,150]],float)
        _,a=candidate_rects(c,(1.25,8.,0.))
        _,b=candidate_rects(c*2+20,(1.25,8.,0.))
        np.testing.assert_allclose(np.array(b[0][:2]),np.array(a[0][:2])*2+20)
        self.assertAlmostEqual(b[0][2],a[0][2]*2)
        self.assertAlmostEqual(b[0][4],a[0][4])

    def test_collapsed_skeleton_rejected_despite_confidence(self):
        h={'presence':.99,'landmarks_px':np.full((21,2),130).tolist()}
        self.assertFalse(acceptable(h,[[100,100],[150,100],[150,150],[100,150]],(400,600,3)))

    def test_unrelated_hand_rejected(self):
        h={'presence':.99,'landmarks_px':np.full((21,2),400).tolist()}
        self.assertFalse(acceptable(h,[[100,100],[150,100],[150,150],[100,150]],(600,600,3)))

    def test_marker_local_coordinates_invariant_to_translation(self):
        c=np.array([[100,100],[150,100],[150,150],[100,150]],float)
        points=np.arange(42,dtype=float).reshape(21,2)+110
        np.testing.assert_allclose(normalized_palm(points,c),normalized_palm(points+50,c+50),atol=1e-6)

    def test_nonfinite_landmark_rejected(self):
        h={'presence':.99,'landmarks_px':np.full((21,2),np.nan).tolist()}
        self.assertFalse(acceptable(h,[[100,100],[150,100],[150,150],[100,150]],(400,600,3)))


if __name__=='__main__':
    unittest.main()
