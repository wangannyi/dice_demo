"""HOME validation and fixed-board workflow isolation, without hardware."""
import json
import tempfile
import unittest

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from home_start import joint_route, load_home, move_home
from reference_board import DIRECTION, restore_auto, main


class HomeStartTests(unittest.TestCase):
    def test_home_units_and_limits_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'home.json'
            p.write_text(json.dumps({'joints_rad': [0.]*7, 'joints_deg': [1.]*7}))
            with self.assertRaisesRegex(ValueError, 'disagree'):
                load_home(p)
        model = SimpleNamespace(joints=[SimpleNamespace(lower_rad=-1., upper_rad=1.)]*7)
        with self.assertRaisesRegex(ValueError, 'joint limits'):
            joint_route([0.]*7, [2.]*7, model)
        with self.assertRaises(ValueError):
            joint_route([float('nan')]*7, [0.]*7, model)
        route = joint_route([0.]*7, [.1]*7, model)
        self.assertEqual(route[-1], [.1]*7)

    def test_home_requires_real_arrival(self):
        arm = Mock()
        arm.read_joints.return_value = {'joints_rad': [.1]*7}
        with patch('auto_collect.require_arm_ready'), patch('auto_collect.stream_smooth_route'), \
             patch('auto_collect.move_and_watch'):
            with self.assertRaisesRegex(RuntimeError, 'confirm arrival'):
                move_home(arm, [[.1]*7, [0.]*7], 4., 6.)

    def test_auto_restore_home_then_observation_and_calculation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = dict(type='charuco', dictionary='4x4_50', squares_x=4, squares_y=5,
                         square_length_m=.02, marker_length_m=.014, legacy_pattern=False,
                         image_roi_xyxy=[0, 0, 640, 480])
            reg = dict(schema=1, mode='fixed_reference_registration', reference_id='table',
                       board=board, source_calibration={'schema': 1, 'mode': 'eye_to_hand', 'direction': DIRECTION,
                                           'quality_passed': True, 'T_base_camera': np.eye(4).tolist(),
                                           'T_flange_tcp': np.eye(4).tolist()},
                       camera_at_registration={'serial': 'test'})
            rp, bp = root/'reg.json', root/'board.json'
            rp.write_text(json.dumps(reg)); bp.write_text(json.dumps(board))
            events = []
            def home(*args):
                events.append('HOME'); return [0.]*7
            def observe(*args):
                events.append('observe'); args[3].mkdir(); return {}
            def restore(*args):
                events.append('restore'); return {'quality_passed': True}
            with patch('home_start.return_home', side_effect=home), \
                 patch('reference_board.observe', side_effect=observe), \
                 patch('reference_board.restore', side_effect=restore):
                restore_auto(rp, bp, 'test', 'table', root/'output')
            self.assertEqual(events, ['HOME', 'observe', 'restore'])
            self.assertTrue((root/'output/restored_calibration.json').exists())
            with patch('home_start.return_home', side_effect=RuntimeError('HOME failed')), \
                 patch('reference_board.observe') as capture:
                with self.assertRaisesRegex(RuntimeError, 'HOME failed'):
                    restore_auto(rp, bp, 'test', 'table', root/'failed')
                capture.assert_not_called()
            with patch('home_start.return_home') as move:
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    restore_auto(rp, bp, 'test', 'table', root/'output')
                move.assert_not_called()

    def test_observe_cli_remains_read_only(self):
        with patch('reference_board.observe', return_value={'quality': {}}) as observe, \
             patch('home_start.return_home') as move:
            main(['observe', '--board', 'board.json', '--serial', 'test',
                  '--reference-id', 'table', '--output', 'output'])
            observe.assert_called_once()
            move.assert_not_called()


if __name__ == '__main__':
    unittest.main()
