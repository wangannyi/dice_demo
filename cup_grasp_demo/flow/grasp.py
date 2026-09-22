"""Side approach planning, isolated from the flange/TCP comparison workflow."""

from copy import deepcopy
from dataclasses import asdict
import math

import numpy as np

from cup_grasp_demo.flow.core import Screen, flange_target, make_plan, read_json
from cup_grasp_demo.flow.cup_selection import select_cup
from cup_grasp_demo.side_grasp.preview_index import load_batch
from cup_grasp_demo.flow.transforms import matrix
from nero_revo2_control.kinematics import load_model

from cup_grasp_demo.flow.parameters import (
    DEFAULT_FINGER_TARGETS, approach_options, closure_targets, hand_contact_allowed, number,
)

FINGER_TARGETS = DEFAULT_FINGER_TARGETS
STATES = ('home', 'pregrasp', 'contact', 'grip')


def options(cfg):
    if 'side_grasp' not in cfg:
        raise ValueError('此会话未选择抓杯配置；用 index_joint_center/config.json 重新 capture')
    opts = dict(cfg['side_grasp'])
    opts.setdefault('strategy', 'side_approach')
    if opts['strategy'] not in ('side_approach', 'direct_close'):
        raise ValueError('Invalid side_grasp strategy')
    bounds = [('close_gap_mm', 5, 50)] if opts['strategy'] == 'direct_close' else [
        ('pregrasp_gap_mm', 5, 100), ('approach_step_mm', 1, 5),
        ('approach_speed_percent', 1, 5)]
    for key, lo, hi in bounds + [
        ('finger_duration_s', .5, 3),
        ('finger_settle_s', .1, 2), ('cup_radius_percentile', 98, 99.5),
        ('cup_model_padding_mm', 3, 10),
    ]:
        value = number(opts[key], key, lo, hi)
        opts[key] = value
    if opts['strategy'] == 'side_approach':
        if int(opts['approach_speed_percent']) != opts['approach_speed_percent']:
            raise ValueError('approach_speed_percent must be an integer')
        opts['approach_speed_percent'] = int(opts['approach_speed_percent'])
    closure_targets(opts)
    approach_options(opts)
    hand_contact_allowed(opts)
    if opts.get('transfer_mode', 'waypoints') not in ('waypoints', 'direct_checked'):
        raise ValueError('side_grasp.transfer_mode must be waypoints or direct_checked')
    return opts


def target_points(session, gap_m=.05):
    """The approach is perpendicular to the horizontal cross-section tangent."""
    scene = session['scene']
    contact = np.asarray(session['contact_base_m'], dtype=float)
    normal = np.asarray(scene['cup_normal_base'], dtype=float)
    normal /= np.linalg.norm(normal)
    radial = contact - np.asarray(scene['cup_support_base_m'])
    radial -= (radial @ normal) * normal
    if np.linalg.norm(radial) < .005:
        raise ValueError('Cannot determine cup-side outward normal')
    outward = radial / np.linalg.norm(radial)
    if np.dot(outward, scene['outward_base']) < .99:
        raise ValueError('Contact point and frozen approach side disagree')
    return contact, contact + gap_m * outward, outward, np.cross(normal, outward)


def observed_scene(directory, session, cfg):
    """Robust observed cup cylinder; retain the raw bound as a diagnostic."""
    meta, depth, image, _ = load_batch(directory)
    geom, _, points = select_cup(depth, image, meta, cfg)
    return scene_from_points(session, cfg, geom, points)


def scene_from_points(session, cfg, geom, points):
    """Build the same collision cylinder from an already verified capture."""
    opts = options(cfg)
    old = session['scene']['geometry']
    if np.linalg.norm(np.array(geom['center_camera_m']) - old['center_camera_m']) > .001:
        raise ValueError('Saved cup geometry no longer reproduces')
    normal = np.array(geom['normal_camera'])
    delta = points - np.array(geom['center_camera_m'])
    z = delta @ normal
    radial = np.linalg.norm(delta - z[:, None] * normal, axis=1)
    selected = (z + geom['contact_height_m'] > 0) & (z + geom['contact_height_m'] <= geom['height_m'])
    if selected.sum() < 100:
        raise ValueError('Insufficient cup points for side approach')
    radius = max(geom['radius_m'], float(np.percentile(radial[selected], opts['cup_radius_percentile'])))
    radius += opts['cup_model_padding_mm'] / 1000
    if not .01 <= radius <= .1:
        raise ValueError('Observed cup collision radius outside model range')
    scene = deepcopy(session['scene'])
    scene['cup_model'] = dict(method='observed_radial_percentile_plus_padding',
                             percentile=opts['cup_radius_percentile'],
                             padding_mm=opts['cup_model_padding_mm'],
                             raw_max_radius_mm=float(radial.max() * 1000),
                             radius_mm=radius * 1000,
                             observed_coverage_fraction=float(np.mean(radial[selected] <= radius)),
                             hidden_surface_verified=False)
    scene['cup_envelope_radius_m'] = radius
    return scene


class GraspScreen:
    """Allow intended hand/cup contact only on the final approach."""

    def __init__(self, screen=None):
        self.full = screen or Screen()
        self.non_hand = Screen.__new__(Screen)
        self.non_hand.arm = self.full.arm
        self.non_hand.meshes = {k: v for k, v in self.full.meshes.items() if not k.startswith('right_')}

    def approach(self, samples, scene, cfg):
        table = self.full.check(samples, scene, True, cfg)
        rigid = self.non_hand.check(samples, scene, False, cfg)
        return dict(table_min_mm=table['table_min_mm'], table_link=table['table_link'],
                    non_hand_cup_lower_bound_mm=rigid['cup_lower_bound_mm'],
                    blockers=list(dict.fromkeys(table['blockers'] + rigid['blockers'])),
                    hand_cup_contact='intentional during final approach only',
                    closed_hand_geometry_verified=False)


def require_at_point(arm, current, target, tcp):
    actual = np.array(arm.fk(current))
    point_error = np.linalg.norm((actual @ tcp)[:3, 3] - (target @ tcp)[:3, 3])
    angle = math.acos(float(np.clip((np.trace(actual[:3, :3].T @ target[:3, :3]) - 1) / 2, -1, 1)))
    if point_error > .001 or angle > math.radians(.5):
        raise ValueError('当前姿态不在所选阶段起点；先完成上一阶段或重新从 HOME 规划')


def make_grasp_plan(session, current, cfg, scene, start='home', until='grip'):
    opts = options(cfg)
    if opts['strategy'] == 'direct_close':
        from cup_grasp_demo.flow.direct_grasp import make_direct_plan
        return make_direct_plan(session, current, cfg, scene, opts, start, until)
    if start not in STATES[:-1] or until not in STATES[1:] or STATES.index(start) >= STATES.index(until):
        raise ValueError('Invalid grasp stage interval')
    arm = load_model()
    tcp = matrix(session['T_flange_tcp'])
    current = np.asarray(current, dtype=float)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError('Expected seven finite joints')
    contact, pregrasp, outward, tangent = target_points(session, opts['pregrasp_gap_mm'] / 1000)
    rotation = np.array(session['R_base_flange'])
    pre_pose = flange_target(pregrasp, rotation, tcp, 'tcp')
    contact_pose = flange_target(contact, rotation, tcp, 'tcp')
    screen = GraspScreen()
    stages, blockers = [], []
    if start == 'home':
        home = read_json(cfg['home'])['joints_rad']
        if np.max(np.abs(current - home)) > math.radians(1):
            raise ValueError('先回新的 HOME，再从 home 阶段规划')
        shaped_session = {**session, 'scene': scene}
        transfer = make_plan(shaped_session, current, cfg, 'tcp', opts['pregrasp_gap_mm'], False,
                             screen=screen.full)
        blockers.extend(transfer['blockers'])
        for stage in transfer['stages']:
            stages.append({**stage, 'kind': 'arm', 'state': 'TO_PREGRASP',
                           'speed_percent': cfg['speed_percent']})
        if not blockers and stages:
            stages[-1]['state_completed'] = 'PREGRASP'
    else:
        require_at_point(arm, current, pre_pose if start == 'pregrasp' else contact_pose, tcp)
        review = (screen.full.check([current], scene, False, cfg) if start == 'pregrasp'
                  else screen.approach([current], scene, cfg))
        blockers.extend(review['blockers'])
    seed = np.array(stages[-1]['target_q_rad']) if stages else current
    if not blockers and until in ('contact', 'grip') and start != 'contact':
        steps = math.ceil(opts['pregrasp_gap_mm'] / opts['approach_step_mm'])
        for i in range(1, steps + 1):
            point = pregrasp + (contact - pregrasp) * i / steps
            target = flange_target(point, rotation, tcp, 'tcp')
            ik = arm.ik(target, seed, max_iterations=600, position_tolerance_m=1e-5,
                        orientation_tolerance_rad=1e-4)
            if not ik.success:
                blockers.append(f'approach/{i}: IK {ik.reason}')
                break
            path = arm.check_joint_path(seed, ik.joints_rad)
            if not path.kinematic_checks_passed or np.max(np.abs(np.array(ik.joints_rad) - seed)) > math.radians(20):
                blockers.append(f'approach/{i}: joint limit/discontinuity')
                break
            # Bound the sampled joint-interpolation bow away from the intended radial line.
            errors = []
            for q in path.samples_rad:
                delta = (np.array(arm.fk(q)) @ tcp)[:3, 3] - contact
                errors.append(np.linalg.norm(delta - (delta @ outward) * outward))
            review = screen.approach(path.samples_rad, scene, cfg)
            if max(errors) > .001:
                review['blockers'].append(f'approach/{i}: sampled line deviation exceeds 1 mm')
            stage = dict(name=f'approach_{i}', state='APPROACH', kind='arm',
                         current_q_rad=seed.tolist(), target_q_rad=ik.joints_rad,
                         T_base_flange=target.tolist(), tcp_target_base_m=point.tolist(),
                         ik=asdict(ik), screen=review, speed_percent=opts['approach_speed_percent'])
            stages.append(stage)
            blockers.extend(review['blockers'])
            seed = np.array(ik.joints_rad)
            if blockers:
                break
        if not blockers:
            stages[-1]['state_completed'] = 'CONTACT'
    if not blockers and until == 'grip':
        for name, target in zip(('THUMB_BASE', 'CLOSE_FINGERS'), closure_targets(opts)):
            stages.append(dict(name=name.lower(), state=name, kind='hand',
                               current_q_rad=seed.tolist(), target_0_100=target,
                               duration_s=opts['finger_duration_s'], settle_s=opts['finger_settle_s']))
    return dict(kind='side_grasp_debug_plan', start_state=start, until_state=until,
                start_q_rad=current.tolist(), stages=stages, blockers=blockers,
                screen_passed=not blockers, cup_removed=False,
                contact_base_m=contact.tolist(), pregrasp_base_m=pregrasp.tolist(),
                outward_base=outward.tolist(), tangent_base=tangent.tolist(),
                T_flange_tcp=tcp.tolist(), T_base_flange_contact=contact_pose.tolist(),
                scene=scene, physical_grip_verified=False,
                limitations=['Sampled geometry; actual interpolation, self collision and cables not verified',
                             'Cup cylinder uses robust observed radii; hidden surface not measured',
                             'Hand contact is intentional only during approach; closed-finger geometry not verified',
                             'Grip commands do not establish secure holding; no lift or shake'])
