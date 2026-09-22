"""Phase-delayed shake stays within the original per-axis kinematic budgets."""
import math
import unittest
from cup_grasp_demo.flow import joint_profile as p

class PhaseTest(unittest.TestCase):
    def plan(self, phases=None):
        cfg=dict(joints=[1,4,6,7],amplitude_deg=[4]*4,velocity_deg_s=[170,170,200,200],
                 acceleration_deg_s2=[277.8845]*4,cycles=6,tracking_error_action='record')
        if phases is not None: cfg['phase_delay_deg']=phases
        limits=[dict(joint=j,min_angle_rad=-3,max_angle_rad=3,max_velocity_rad_s=4,max_acceleration_rad_s2=5) for j in range(1,8)]
        return p.make_plan(dict(success=True,q_after_rad=[0]*7,limits=limits),cfg,[(-3,3)]*7)

    def test_omitted_or_zero_phase_preserves_old_trajectory(self):
        a,b=self.plan(),self.plan([0]*4)
        self.assertEqual(a['samples'],b['samples'])
        self.assertEqual(a['duration_s'],b['duration_s'])
        for sample in a['samples']:
            u=p.at(a['segments'],sample['t_s'])[0]
            for q,amp in zip(sample['q_rad'],a['amplitude_rad']):
                self.assertAlmostEqual(q,amp*u)

    def test_quarter_period_delay_and_return_to_center(self):
        plan=self.plan([0,0,0,90]);base=self.plan();delay=plan['cycle_period_s']/4
        self.assertAlmostEqual(plan['duration_s'],base['duration_s']+delay)
        self.assertTrue(plan['planning_passed'])
        for sample in plan['samples']:
            t=sample['t_s'];q=sample['q_rad']
            self.assertAlmostEqual(q[6],p.joint_values(base,t-delay)[6])
            self.assertAlmostEqual(q[5],p.joint_values(base,t)[5])
            self.assertLessEqual(max(map(abs,q)),math.radians(4)+1e-12)
            self.assertTrue(all(abs(e)<1e-10 for e in p.reference_errors({'q_rad':q},plan,t)))
        self.assertEqual(plan['samples'][0]['q_rad'],[0]*7)
        self.assertEqual(plan['samples'][-1]['q_rad'],[0]*7)
        self.assertEqual(plan['joint_peak_acceleration_rad_s2'],base['joint_peak_acceleration_rad_s2'])
        self.assertEqual(plan['joint_peak_velocity_rad_s'],base['joint_peak_velocity_rad_s'])
        # Continuous start/stop; shifted velocity and acceleration obey same bounds.
        dt=1e-4
        for k in range(1,1000):
            t=k*plan['duration_s']/1000
            a,b,c=[p.joint_values(plan,t+x*dt) for x in (-1,0,1)]
            for j in range(7):
                self.assertLessEqual(abs((c[j]-a[j])/(2*dt)),plan['joint_peak_velocity_rad_s'][j]+1e-6)
                self.assertLessEqual(abs((c[j]-2*b[j]+a[j])/dt**2),plan['joint_peak_acceleration_rad_s2'][j]+1e-5)

    def test_invalid_phase_rejected(self):
        for phase in ([0], [0,0,0,-1], [0,0,0,361], [0,0,0,float('nan')], [0,0,0,True]):
            with self.subTest(phase=phase),self.assertRaises(ValueError): self.plan(phase)

    def test_null_phase_accepts_changed_joint_count(self):
        for count in (1,4,5,7):
            raw=dict(joints=list(range(1,count+1)),amplitude_deg=[5]*count,
                     velocity_deg_s=[170]*count,acceleration_deg_s2=[277.8845]*count,
                     phase_delay_deg=None,cycles=6)
            cfg,segments,duration=p.trajectory(raw)
            self.assertIsNone(cfg['phase_delay_deg'])
            plan=dict(parameters=cfg,segments=segments,start_q_rad=[0]*7,
                      amplitude_rad=[math.radians(5)]*count+[0]*(7-count))
            at_peak=p.joint_values(plan,segments[0]['duration_s'])
            self.assertEqual(at_peak[:count],[math.radians(5)]*count)
            self.assertEqual(p.joint_values(plan,duration),[0]*7)

    def test_bad_recipe_fails_before_any_workflow_resources(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from cup_grasp_demo.flow import green_pipeline as flow
        bad=dict(joints=[1,4,5,6,7],amplitude_deg=[5]*5,velocity_deg_s=[170]*5,
                 acceleration_deg_s2=[277.8845]*5,phase_delay_deg=[0]*4)
        with patch.object(flow,'load_config',return_value={}),patch.object(flow,'validate',return_value={'joint_test_config':'unused.json'}),patch.object(flow,'read_json',return_value=bad),patch.object(flow,'Kinematics') as kin,patch.object(flow.Workflow,'bridge') as bridge:
            with self.assertRaisesRegex(ValueError,'phase_delay_deg'):
                flow.Workflow(SimpleNamespace(config='unused',mode='fast'))
            kin.assert_not_called();bridge.assert_not_called()
