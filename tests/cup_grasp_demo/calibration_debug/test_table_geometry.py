"""Table-only acceleration retains every sample and the full mesh plane minima."""
import math
import unittest
from unittest.mock import patch
import numpy as np
from cup_grasp_demo.calibration_debug.core import Screen, cached_screen_geometry


class TableGeometryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = Screen()
        cls.fast = Screen(table_only=True)

    def test_all_plane_directions_have_same_extrema_on_original_meshes(self):
        rng = np.random.default_rng(401)
        directions = rng.normal(size=(3, 64))
        directions /= np.linalg.norm(directions, axis=0)
        for name, (vertices, _, _) in self.full.meshes.items():
            with self.subTest(link=name):
                reduced = self.fast.meshes[name][0]
                np.testing.assert_allclose(np.min(vertices @ directions, axis=0),
                                           np.min(reduced @ directions, axis=0),atol=1e-12,rtol=0)

    def test_all_path_samples_keep_table_result_and_blockers(self):
        center = np.radians([30,-70,-90,100,-10,-5,5])
        qs = [center + np.radians([2*u,0,0,3*u,0,0,5*u]) for u in np.linspace(-1,1,93)]
        for normal in ([0,0,1],[.02,.03,math.sqrt(1-.02**2-.03**2)]):
            scene=dict(cup_support_base_m=[0,0,0],cup_normal_base=normal)
            full=self.full.check_table_batch(qs,scene,{'table_margin_mm':5})
            for margin in (5,full['table_min_mm']-1e-6,full['table_min_mm']+1e-6):
                cfg={'table_margin_mm':margin}
                a=self.full.check_table_batch(qs,scene,cfg)
                b=self.fast.check_table_batch(qs,scene,cfg)
                self.assertAlmostEqual(a['table_min_mm'],b['table_min_mm'],places=9)
                self.assertEqual(a['table_link'],b['table_link'])
                self.assertEqual(a['blockers'],b['blockers'])

    def test_fast_path_does_not_construct_cup_spheres_and_cannot_screen_a_cup(self):
        with patch('cup_grasp_demo.calibration_debug.core.cover',side_effect=AssertionError('unneeded sphere cover')):
            fast=Screen(table_only=True)
        with self.assertRaisesRegex(ValueError,'cannot check cup'):
            fast.check([[0]*7],{},False,{})

    def test_fast_and_full_caches_are_separate(self):
        with cached_screen_geometry():
            fast=Screen(table_only=True)
            with patch("cup_grasp_demo.calibration_debug.core._stl_bounds",side_effect=AssertionError("reloading geometry")):
                full=Screen()
            self.assertIs(Screen(table_only=True).meshes,fast.meshes)
            self.assertIs(Screen().meshes,full.meshes)
            self.assertIsNot(fast.meshes,full.meshes)
            self.assertTrue(all(v[1] is not None for v in full.meshes.values()))

    def test_degenerate_mesh_keeps_all_points(self):
        screen=Screen.__new__(Screen);screen.table_only=True;screen.meshes={}
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        screen.add('flat',vertices)
        self.assertEqual(len(screen.meshes['flat'][0]),3)

    def test_repeated_vertices_need_no_sort_for_table_hull(self):
        screen=Screen.__new__(Screen);screen.table_only=True;screen.meshes={};screen._sources={}
        vertices=np.repeat(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),100,axis=0)
        real_unique=np.unique
        def checked_unique(points,*args,**kwargs):
            self.assertIsNot(points,vertices,'unneeded full mesh sort')
            return real_unique(points,*args,**kwargs)
        with patch('cup_grasp_demo.calibration_debug.core.np.unique',side_effect=checked_unique):
            screen.add('tetrahedron',vertices)
        directions=np.random.default_rng(5).normal(size=(3,100))
        np.testing.assert_allclose((vertices@directions).min(axis=0),
                                   (screen.meshes['tetrahedron'][0]@directions).min(axis=0),atol=1e-12)
        self.assertIsNone(screen._sources['tetrahedron'][1])


if __name__=='__main__': unittest.main()
