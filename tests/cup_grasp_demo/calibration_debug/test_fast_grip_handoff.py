"""FAST-only overlap and feedback reuse, with no CAN or camera access."""
import time
import unittest
from threading import Event
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cup_grasp_demo.calibration_debug.fast_feedback import arm_snapshot
import test_green_fast_reuse as reuse
from cup_grasp_demo.calibration_debug import green_pipeline as flow
from cup_grasp_demo.calibration_debug.grasp_execution import send_closure


class HandoffTest(unittest.TestCase):
    def test_recent_feedback_does_not_wait_for_three_new_packets(self):
        state = NS(arm_status=0, motion_status=0)
        robot = NS(get_joint_angles=lambda:NS(timestamp=99.98,msg=[0]*7),
                   get_flange_pose=lambda:NS(timestamp=99.97,msg=[0]*6),
                   get_arm_status=lambda:NS(timestamp=99.99,msg=state))
        demo=Mock();demo.feedback_stamp.side_effect=lambda x:x.timestamp
        self.assertEqual(arm_snapshot(robot,demo,wallclock=lambda:100),([0]*7,[0]*6,state))
        demo.arm_snapshot.assert_not_called();demo.check_comm.assert_called_once()
        robot.get_joint_angles=lambda:NS(timestamp=99.8,msg=[0]*7)
        arm_snapshot(robot,demo,wallclock=lambda:100)
        demo.arm_snapshot.assert_called_once()

    def test_cached_fault_is_not_hidden(self):
        robot = NS(get_joint_angles=lambda:NS(timestamp=100,msg=[0]*7),
                   get_flange_pose=lambda:NS(timestamp=100,msg=[0]*6),
                   get_arm_status=lambda:NS(timestamp=100,msg=NS(arm_status=2)))
        demo=Mock();demo.feedback_stamp.side_effect=lambda x:x.timestamp
        self.assertEqual(arm_snapshot(robot,demo,wallclock=lambda:100)[2].arm_status,2)
        demo.check_comm.side_effect=RuntimeError('CAN error')
        with self.assertRaises(RuntimeError):arm_snapshot(robot,demo,wallclock=lambda:100)

    def test_grip_not_blocked_by_following_planner(self):
        w=reuse.FastReuseTest().workflow()
        w.g['fast_overlap_grip_preparation']=True
        w._snapshot_cache=dict(success=True,joints_rad=[.1]*7,joints_enabled=[True]*7,
            arm_status=0,motion_status=0,ctrl_mode=1,observed_epoch_s=time.time())
        release=Event(); entered=Event()
        def blocked(*args,**kwargs):
            entered.set()
            if not release.wait(2):raise RuntimeError('test timeout')
            raise ValueError('blocked route')
        w.build_following=blocked
        try:
            w.prepare_following()
            self.assertFalse(entered.is_set())
            w.launch_following()
            self.assertTrue(entered.wait(1))
            w.snapshot.assert_not_called()
            self.assertFalse(w._route_future.done())
            w.start_shake_prepare.assert_not_called()
            release.set()
            with self.assertRaises(ValueError):w.finish_following()
        finally:
            release.set();w._route_pool.shutdown()

    def test_hand_monitor_does_not_add_a_full_poll_after_deadline(self):
        clock=[0.]
        def monitor():clock[0]+=.021
        def sleep(s):clock[0]+=s
        send_closure(Mock(),NS(FINGER_NAMES=list('abcdef'), feedback_stamp=lambda value: None),
            dict(state='CLOSE_FINGERS',target_0_100=[0]*6,duration_s=.5,settle_s=0),
            monitor,read_feedback=False,monotonic=lambda:clock[0],wallclock=lambda:100+clock[0],sleep=sleep)
        self.assertLessEqual(clock[0],.521)

class SettlingTest(unittest.TestCase):
    def test_hand_0281_degree_change_is_recorded_without_reconnect(self):
        import math
        from cup_grasp_demo.calibration_debug.hardware import hand_start, validate_start, StartPositionChanged
        cfg=dict(pipeline_strategy='green_open_cup',green_cup={'precision_error_action':'record'})
        plan=dict(kind='green_hand_command',stages=[],start_q_rad=[0]*7)
        actual=[0]*6+[math.radians(.281)]
        status=NS(arm_status=0,motion_status=0,ctrl_mode=1);report={}
        updated=hand_start(plan,actual,status,[True]*7,cfg,report)
        validate_start(updated,actual,status,[True]*7,.25)
        self.assertAlmostEqual(report['hand_start_reference']['error_deg'],.281)
        self.assertEqual(plan['start_q_rad'],[0]*7)
        for joints,state in [(actual,NS(arm_status=0,motion_status=1,ctrl_mode=1)),
                             ([0]*6+[math.radians(.501)],status)]:
            with self.assertRaises(RuntimeError) as error:hand_start(plan,joints,state,[True]*7,cfg,{})
            self.assertNotIsInstance(error.exception,StartPositionChanged)
        self.assertIs(hand_start(dict(plan,kind='green_arm_plan'),actual,status,[True]*7,cfg,{}).get('stages'),plan['stages'])

    def test_lift_endpoint_bounds_include_tcp_and_orientation(self):
        import numpy as np
        from cup_grasp_demo.calibration_debug.green_prepared import lift_seed_close
        kin=Mock();before=np.eye(4);after=np.eye(4)
        kin.forward.side_effect=lambda q:(before if q[0]==0 else after,None)
        old=[0]*7;current=[.001]*7
        after[0,3]=.0019
        self.assertTrue(lift_seed_close(old,current,kin,np.eye(4)))
        after[0,3]=.0021
        self.assertFalse(lift_seed_close(old,current,kin,np.eye(4)))
        after[:]=np.eye(4);a=np.radians(1.1)
        after[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
        self.assertFalse(lift_seed_close(old,current,kin,np.eye(4)))
        self.assertFalse(lift_seed_close(old,[.02]*7,kin,np.eye(4)))

    def test_lift_reuses_ik_but_never_old_path(self):
        import numpy as np
        from concurrent.futures import Future
        from cup_grasp_demo.calibration_debug.green_prepared import PreparedRoutes
        w=reuse.FastReuseTest().workflow();w.g['fast_overlap_grip_preparation']=True
        w.place_q=[.1]*7;w._snapshot_cache={};w.snapshot.return_value=[.101]*7
        w.kin=Mock();w.kin.forward.return_value=(np.eye(4),None)
        prepared=PreparedRoutes();prepared.put_vertical('lift',[.1]*7,[[.2]*7])
        prepared.routes['lift']='obsolete path'
        w._route_future=Future();w._route_future.set_result(prepared)
        w.finish_following()
        self.assertEqual(w.place_q,[.101]*7)
        self.assertEqual(w._prepared.routes,{})
        self.assertEqual(w._prepared.vertical['lift'],([.101]*7,[[.2]*7]))
        self.assertTrue(w.lift_endpoint_reused)
