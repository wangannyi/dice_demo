"""Single-frame shortcut preserves depth semantics and multi-frame behavior."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from cup_grasp_demo.side_grasp.preview_index import load_batch


class SingleFrameDepthTests(unittest.TestCase):
    def dataset(self, root, depths):
        folder=root/'rgbd';folder.mkdir()
        meta=dict(serial='test',intrinsics={},depth_scale_m=.001,depth_registered_to='color_optical')
        for i, depth in enumerate(depths):
            prefix=folder/f'frame_{i:03d}'
            prefix.with_suffix('.json').write_text(json.dumps(meta))
            np.savez(prefix.with_suffix('.npz'),depth=depth,
                     color=np.zeros((*depth.shape,3),np.uint8))

    def test_single_frame_matches_nanmedian_without_computing_it(self):
        depth=np.array([[0,123,65535],[np.nan,np.inf,-np.inf]],dtype=float)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.dataset(root,[depth])
            stack=depth[None].copy();stack[stack==0]=np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');expected=np.nan_to_num(np.nanmedian(stack,axis=0))
            with patch('numpy.nanmedian',side_effect=AssertionError('single frame needs no median')):
                _,actual,_,_=load_batch(root,min_frames=1)
            np.testing.assert_array_equal(actual,expected)
            with self.assertRaises(ValueError):load_batch(root)

    def test_multiple_frames_still_ignore_missing_depth(self):
        depths=[np.array([[0,10],[0,100]]),np.array([[0,20],[30,200]]),np.array([[0,0],[50,300]])]
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.dataset(root,depths)
            _,actual,_,_=load_batch(root)
            np.testing.assert_array_equal(actual,[[0,15],[40,200]])

    def test_changed_geometry_still_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.dataset(root,[np.ones((2,2))]*3)
            path=root/'rgbd/frame_002.json';meta=json.loads(path.read_text());meta['serial']='different';path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'changed'):load_batch(root)
