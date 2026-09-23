import unittest
import numpy as np
from image_profile import profile_options, crop_intrinsics, crop_image

class CropTest(unittest.TestCase):
    def test_projection_invariant_and_no_mutation(self):
        k=dict(width=1280,height=720,fx=900.,fy=901.,cx=640.,cy=360.)
        c=crop_intrinsics(k,[220,0,960,720])
        self.assertEqual((c['width'],c['height'],c['cx'],c['cy']),(960,720,420,360))
        self.assertEqual(k['cx'],640)
        for xyz in ([.1,.2,1],[-.1,.1,2]):
            x,y,z=xyz
            self.assertAlmostEqual(k['fx']*x/z+k['cx']-220,c['fx']*x/z+c['cx'])
            self.assertAlmostEqual(k['fy']*y/z+k['cy'],c['fy']*y/z+c['cy'])
    def test_rgb_aligned_depth_same_crop(self):
        d=np.arange(1280*720).reshape(720,1280)
        c=np.stack([d]*3,axis=-1)
        np.testing.assert_array_equal(crop_image(c,[220,0,960,720])[:,:,0],crop_image(d,[220,0,960,720]))
    def test_legacy_default_and_invalid_crop(self):
        self.assertEqual(profile_options(),([640,480],15,None))
        for crop in ([220,0,1280,720],[-1,0,960,720],[220,0,0,720],[220.,0,960,720]):
            with self.assertRaises(ValueError):profile_options(dict(color_resolution=[1280,720],crop_xywh=crop))
