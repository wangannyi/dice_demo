"""Analytic rim derivative must match independent finite differences."""
import unittest
import numpy as np
from scipy.ndimage import map_coordinates
from vision.geometry.circle_rim import fixed_rim_residual,project,fit_circle
import test_circle_rim as fixtures

class JacobianTest(unittest.TestCase):
    def test_residual_and_derivative_match(self):
        rng=np.random.default_rng(17)
        basis=np.array([[1.,0],[0,1.],[0,0.]])
        theta=np.linspace(0,2*np.pi,21,endpoint=False)
        unit=np.c_[np.cos(theta),np.sin(theta),np.zeros(len(theta))]
        origin=np.array([0,0,.6]);views=[]
        for angle,baseline in [(0,0),(.07,.05)]:
            rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
            views.append(dict(R=rotation,t=np.array([baseline,0,0]),
                k=dict(fx=600,fy=590,cx=320,cy=240),distance=rng.uniform(0,10,(480,640))))
        fun,jac=fixed_rim_residual(views,origin,basis,unit)
        for q in [np.array([.00123,.00321,.03752]),np.array([1.,0,.0375])]:
            points=origin+basis@q[:2]+q[2]*unit
            expected=[]
            for v in views:
                uv=project(points,v);expected.extend(map_coordinates(v['distance'],[uv[:,1],uv[:,0]],order=1,mode='constant',cval=50))
            np.testing.assert_allclose(fun(q),expected,atol=1e-9)
            finite=np.column_stack([(fun(q+np.eye(3)[i]*1e-9)-fun(q-np.eye(3)[i]*1e-9))/2e-9 for i in range(3)])
            np.testing.assert_allclose(jac(q),finite,rtol=2e-5,atol=.001)

    def test_synthetic_circle_still_recovers_size_and_center(self):
        fixture=fixtures.StereoRimTest();fixture.setUp()
        fit=fit_circle(fixture.views,fixture.table,fixture.normal,[[.003,.002,.59,.034],[-.002,.001,.62,.04]],
            [.025,.075],[.04,.18],fixed_height=.065,analytic_jacobian=True)
        np.testing.assert_allclose(fit['center'],[0,0,.6],atol=.002)
        self.assertAlmostEqual(fit['radius_m']*2,.075,delta=.002)
