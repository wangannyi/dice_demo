"""FAST measured-pose preparation, coordinated HOME, and diagnostic isolation."""
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from cup_grasp_demo.calibration_debug import green_pipeline as f
from cup_grasp_demo.calibration_debug import green_hand_execution as hand
from cup_grasp_demo.calibration_debug.grasp_execution import send_closure
from cup_grasp_demo.calibration_debug.green_prepared import PreparedRoutes


class FastReuseTest(unittest.TestCase):
    def workflow(self):
        w=object.__new__(f.Workflow)
        w.args=SimpleNamespace(mode='fast',until='place')
        w.cfg={'timeout_s':5,'start_tolerance_deg':.5}
        w.g=dict(lift_mm=50,wrist_reference_deg=[0,-13,5],direct_return_home=True,place_tolerance_mm=3)
        w._prepared=PreparedRoutes();w._route_future=None
        w.snapshot=Mock(return_value=[.1]*7);w.held_geometry=Mock(return_value={'radius_m':.04})
        w.scene={};w.tcp=np.eye(4);w.home=[0]*7;w.start_shake_prepare=Mock()
        return w

    def test_actual_grasp_routes_reused_for_all_following_moves(self):
        w=self.workflow()
        def route(start,targets,*a,**kw):
            return dict(start_q_rad=list(start),stages=[dict(target_q_rad=t) for t in targets],blockers=[])
        with patch.object(f,'vertical_targets',return_value=[[.2]*7]) as ik, patch.object(f,'arm_plan',side_effect=route) as plan:
            try:
                w.prepare_following();w.finish_following()
                w._route_pool.shutdown()
                for label, future in w._following_futures.items():
                    w._prepared.routes.update(future.result().routes)
                self.assertEqual(plan.call_count,3)
                for label,start,target,held,margin in [('lift',[.1]*7,[[.2]*7],w.held,-3),('lower',[.201]*7,[[.1]*7],w.held,-3),('return_home',[.101]*7,[[0]*7],None,0)]:
                    self.assertIsNotNone(w._prepared.take(label,start,target,w.scene,held,margin,start_tolerance_deg=.5))
                ik.assert_called_once()
            finally:w._route_pool.shutdown()

    def test_background_routes_inherit_geometry_cache(self):
        from cup_grasp_demo.calibration_debug.core import cached_screen_geometry, _screen_cache
        w=self.workflow()
        seen=[]
        def route(start, targets, *args, **kwargs):
            seen.append(_screen_cache.get())
            return dict(start_q_rad=list(start),stages=[dict(target_q_rad=t) for t in targets],blockers=[])
        with cached_screen_geometry(), patch.object(f,'vertical_targets',return_value=[[.2]*7]), patch.object(f,'arm_plan',side_effect=route):
            expected=_screen_cache.get()
            try:
                w.prepare_following()
                w._route_pool.shutdown()
                self.assertEqual(len(seen),3)
                self.assertTrue(all(item is expected for item in seen))
            finally:w._route_pool.shutdown()

    def test_phase_speed_override_does_not_change_other_phases(self):
        w=self.workflow();w.cfg['speed_percent']=30
        w.g['fast_phase_speed_percent']={'approach':60,'return_home':60}
        w.root=Path('/unused');w.receipts={};w.unchanged=Mock();w.bridge=Mock(return_value={'success':True})
        with patch.object(f.common,'new_run',return_value=Path('/unused/run')), patch.object(f,'write_json') as write:
            for label, speed in [('approach',60),('return_home',60),('lower',30)]:
                w.issue({'blockers':[]},label)
                request=write.call_args_list[-1].args[1]
                self.assertEqual(request['config']['speed_percent'],speed)
                self.assertEqual(w.cfg['speed_percent'],30)

    def test_lift_reads_fresh_pose_instead_of_grip_cache(self):
        w=self.workflow();w.place_q=[.1]*7
        w._route_future=Mock();w._route_future.result.return_value=PreparedRoutes()
        w._snapshot_cache={'stale': True}
        def fresh():
            self.assertIsNone(w._snapshot_cache)
            return [.1]*7
        w.snapshot=Mock(side_effect=fresh)
        w.finish_following()
        w.snapshot.assert_called_once()

    def test_near_tolerance_cached_route_is_replanned_from_current_pose(self):
        w=self.workflow();w._following_futures={}
        old=[.1]*7;current=[.1]*7;current[5]+=np.radians(.20)
        target=[[.2]*7]
        route=dict(start_q_rad=old,stages=[dict(target_q_rad=target[0])],blockers=[])
        w._prepared.put('lift',route,w.scene)
        w.snapshot=Mock(return_value=current);w.issue=Mock()
        fresh=dict(start_q_rad=current,stages=route['stages'],blockers=[])
        with patch.object(f,'arm_plan',return_value=fresh) as replan:
            w.move(target,'lift')
        replan.assert_called_once()
        self.assertEqual(replan.call_args.args[0],current)
        self.assertEqual(w.issue.call_args.args[0]['start_q_rad'],current)

    def test_actual_start_change_discards_background_results(self):
        w=self.workflow();w.place_q=[.1]*7
        w._route_future=Mock();w._route_future.result.return_value=PreparedRoutes()
        w.snapshot.return_value=[.2]*7
        w.finish_following()
        self.assertEqual(w.place_q,[.2]*7)
        self.assertEqual(w._prepared.routes,{})
        self.assertEqual(len(w.recovery_events),1)

    def test_home_issues_single_request(self):
        w=self.workflow();w.args.mode='fast'
        w.prepare_vision=Mock();w.table=Mock();w.hand=Mock();w.move=Mock();w.issue=Mock()
        w.g['open_targets_0_100']=[0]*6;w.cfg['home']='unused'
        with patch.object(f,'read_json',return_value={'joints_deg':[0]*7}),patch.object(f,'arm_plan',return_value={'blockers':[]}):
            w.perform('HOME')
        w.hand.assert_not_called();w.move.assert_not_called()
        self.assertEqual(w.issue.call_args.args[0]['kind'],'green_home_open')

    def test_hand_and_home_overlap_without_extra_duration_wait(self):
        clock=[0.];events=[]
        h=Mock();h.position_time_ctrl.side_effect=lambda **kw:events.append(kw['mode'])
        demo=SimpleNamespace(FINGER_NAMES=list('abcdef'))
        def move():events.append('arm');clock[0]+=1.8
        report=send_closure(h,demo,dict(state="CLOSE_FINGERS",target_0_100=[0]*6,duration_s=1,settle_s=0),Mock(),
            monotonic=lambda:clock[0],wallclock=lambda:100+clock[0],sleep=lambda s:clock.__setitem__(0,clock[0]+s),read_feedback=False,during_action=move)
        self.assertEqual(events,['pos','time','arm'])
        self.assertEqual(clock[0],1.8)
        self.assertFalse(report['position_feedback_available'])

    def test_combined_home_fault_prevents_hand_and_arm_commands(self):
        d=Mock();d.arm_snapshot.return_value=([0]*7,None,SimpleNamespace(arm_status=1,ctrl_mode=1,motion_status=0))
        robot=Mock();robot.get_joints_enable_status_list.return_value=[True]*7
        options=dict(fast_completion=True,require_hand_position=False,read_hand_feedback=False,open_targets_0_100=[0]*6,grip_targets_0_100=[100]*6)
        arm=Mock()
        with patch.object(hand,'send_closure') as send:
            with self.assertRaises(RuntimeError):hand.execute(dict(kind='green_home_open',target_0_100=[0]*6,start_q_rad=[0]*7),{'green_cup':options},robot,Mock(),d,{},arm)
            send.assert_not_called();arm.assert_not_called()

    def test_fast_capture_keeps_result_without_debug_images(self):
        w=self.workflow();w.args.show=False
        w.g.update(contact_offset_base_mm=[40,0,0],perception={'frame_count':1})
        w.cfg['plane_tolerance_mm']=6
        w._vision=Mock()
        image=np.zeros((20,20,3),dtype=np.uint8)
        geo=dict(rim_center_camera_m=[0,0,1],table_normal_camera=[0,0,1],height_m=.065,radius_m=.0375)
        with tempfile.TemporaryDirectory() as d:
            w.root=Path(d)
            with patch.object(f.common,'new_run',return_value=w.root),patch.object(f,'load_batch',return_value=({},None,image,None)),patch.object(f.common,'camera_transform',return_value=(np.eye(4),True)),patch.object(f,'infer',return_value=([],{})),patch.object(f,'detect',return_value=(geo,None,None)),patch.object(f.cv2,'imwrite') as write:
                w.capture()
            write.assert_not_called()
            self.assertTrue(f.read_json(w.root/'green_scene.json')['valid'])

    def test_lift_published_before_later_routes_complete(self):
        from threading import Event
        w=self.workflow();release=Event();entered=Event()
        def route(start,targets,*a,**kw):
            if start == [.2]*7:
                entered.set()
                if not release.wait(3):raise RuntimeError('test planning stalled')
            return dict(start_q_rad=list(start),stages=[dict(target_q_rad=t) for t in targets],blockers=[])
        with patch.object(f,'vertical_targets',return_value=[[.2]*7]),patch.object(f,'arm_plan',side_effect=route):
            try:
                w.prepare_following()
                self.assertTrue(entered.wait(3))
                self.assertTrue(w._route_future.done())
                self.assertFalse(w._following_futures['lower'].done())
                w.finish_following()
                self.assertIn('lift',w._prepared.routes)
            finally:
                release.set();w._route_pool.shutdown()

    def test_failed_background_route_never_sends_motion(self):
        from concurrent.futures import Future
        w=self.workflow();failed=Future();failed.set_exception(ValueError('collision'))
        w._following_futures={'lower':failed};w.issue=Mock()
        with self.assertRaisesRegex(ValueError,'collision'):w.move([[.1]*7],'lower')
        w.issue.assert_not_called()
