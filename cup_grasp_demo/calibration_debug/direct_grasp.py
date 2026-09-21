"""Move the open-hand TCP to the specified cup-side gap, then close in place."""

from dataclasses import asdict
import math

import numpy as np

from cup_grasp_demo.calibration_debug.core import Screen, flange_target, make_plan, read_json
from cup_grasp_demo.calibration_debug.grasp import require_at_point, target_points
from cup_grasp_demo.calibration_debug.parameters import approach_options, closure_targets, hand_contact_allowed
from nero_calibration.core import matrix
from nero_revo2_control.kinematics import load_model


def make_direct_plan(session, current, cfg, scene, opts, start, until):
    if (start, until) not in (('home', 'ready'), ('home', 'grip'), ('ready', 'grip')):
        raise ValueError('direct_close 使用 home → ready → grip；单测到位用 --until ready')
    current = np.asarray(current, dtype=float)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError('Expected seven finite joints')
    tcp = matrix(session['T_flange_tcp'])
    contact, ready, outward, tangent = target_points(session, opts['close_gap_mm'] / 1000)
    pose = flange_target(ready, np.asarray(session['R_base_flange']), tcp, 'tcp')
    screen, stages, blockers = Screen(), [], []
    approach = approach_options(opts)
    allow_contact = hand_contact_allowed(opts)
    if start == 'home':
        home = read_json(cfg['home'])['joints_rad']
        if np.max(np.abs(current - home)) > math.radians(1):
            raise ValueError('先回新的 HOME，再从 home 阶段规划')
        initial_gap = approach['start_gap_mm'] if approach['enabled'] else opts['close_gap_mm']
        transfer = None
        if opts.get('transfer_mode') == 'direct_checked':
            transfer = direct_transfer(session, current, cfg, scene, screen, initial_gap)
        if transfer is None or transfer['blockers']:
            direct_blockers = transfer['blockers'] if transfer else []
            transfer = make_plan({**session, 'scene': scene}, current, cfg, 'tcp',
                                 initial_gap, False, screen=screen, allow_hand_cup_contact=allow_contact)
            transfer['direct_fallback_reasons'] = direct_blockers
        blockers.extend(transfer['blockers'])
        stages = [{**s, 'kind': 'arm', 'state': 'TO_CLOSE_READY',
                   'speed_percent': cfg['speed_percent']} for s in transfer['stages']]
        if not blockers and stages:
            if approach['enabled']:
                stages[-1]['state_completed'] = 'APPROACH_START'
                radial_stages, radial_blockers = approach_stages(session, cfg, scene, screen,
                                                                 stages[-1]['target_q_rad'], approach,
                                                                 opts['close_gap_mm'])
                stages.extend(radial_stages)
                blockers.extend(radial_blockers)
            if not blockers:
                stages[-1]['state_completed'] = 'READY'
    else:
        require_at_point(load_model(), current, pose, tcp)
        ready_review = screen.check([current], scene, False, cfg, allow_hand_cup_contact=allow_contact)
        blockers.extend(ready_review['blockers'])
    seed = stages[-1]['target_q_rad'] if stages else current.tolist()
    if not blockers and until == 'grip':
        for name, target in zip(('THUMB_BASE', 'CLOSE_FINGERS'), closure_targets(opts)):
            stages.append(dict(name=name.lower(), state=name, kind='hand',
                               current_q_rad=list(seed), target_0_100=list(target),
                               duration_s=opts['finger_duration_s'], settle_s=opts['finger_settle_s']))
    result = dict(kind='side_grasp_debug_plan', strategy='direct_close',
                start_state=start, until_state=until, start_q_rad=current.tolist(),
                stages=stages, blockers=blockers, screen_passed=not blockers, cup_removed=False,
                contact_base_m=contact.tolist(), close_ready_base_m=ready.tolist(),
                closing_gap_mm=opts['close_gap_mm'], outward_base=outward.tolist(),
                tangent_base=tangent.tolist(), T_flange_tcp=tcp.tolist(),
                T_base_flange_close_ready=pose.tolist(), scene=scene, physical_grip_verified=False,
                limitations=['Full open-hand table/cup checks apply until the closing point',
                             'Closed-finger geometry and physical grip are not verified',
                             'Sampled geometry; actual interpolation, self collision and cables not verified',
                             'Cup hidden surface is not measured; no lift or shake'])
    if start == 'home' and opts.get('transfer_mode') == 'direct_checked':
        result['transfer_mode'] = 'direct_checked'
        result['direct_fallback_reasons'] = transfer.get('direct_fallback_reasons', [])
    if approach['enabled']:
        result['approach'] = approach
    if allow_contact:
        result['allow_hand_cup_contact'] = True
        result['contact_policy'] = 'hand/target-cup contact allowed during lower, radial approach and closure'
        result['limitations'][0] = 'Full table and arm/adapter cup checks; hand cup margin is advisory in contact stages'
        reviews = [stage['screen'] for stage in stages if 'screen' in stage]
        if start == 'ready':
            result['ready_screen'] = ready_review
            reviews.append(ready_review)
        result['contact_warnings'] = list(dict.fromkeys(
            warning for review in reviews for warning in review.get('contact_warnings', [])))
    return result


def direct_transfer(session, current, cfg, scene, screen, gap_mm):
    """One MoveJ, only after screening the complete joint-linear swept path."""
    _, point, _, _ = target_points(session, gap_mm / 1000)
    pose = flange_target(point, np.asarray(session['R_base_flange']),
                         session['T_flange_tcp'], 'tcp')
    reference = np.asarray(read_json(cfg['orientation_reference'])['joints_rad'])
    errors = []
    for seed in (current, np.clip(reference, *np.asarray(screen.arm.limits_rad).T)):
        ik = screen.arm.ik(pose, seed, max_iterations=600,
                           position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4)
        if not ik.success:
            errors.append('direct: endpoint IK ' + ik.reason)
            continue
        path = screen.arm.check_joint_path(current, ik.joints_rad,
                                           max_sample_step_rad=math.radians(1))
        if not path.kinematic_checks_passed:
            errors.append('direct: ' + path.reason)
            continue
        review = screen.check(path.samples_rad, scene, False, cfg,
                              allow_hand_cup_contact=hand_contact_allowed(cfg['side_grasp']))
        if review['blockers']:
            errors.extend(review['blockers'])
            continue
        return dict(blockers=[], stages=[dict(name='direct_to_ready',
                    current_q_rad=list(current), target_q_rad=list(ik.joints_rad),
                    T_base_flange=pose.tolist(), ik=asdict(ik), screen=review,
                    joint_path_samples=len(path.samples_rad))])
    return dict(blockers=errors, stages=[])


def approach_stages(session, cfg, scene, screen, initial_q, opts, final_gap_mm):
    """Sample a table-parallel straight approach with unchanged hand orientation."""
    tcp = matrix(session['T_flange_tcp'])
    rotation = np.asarray(session['R_base_flange'])
    contact, beginning, outward, _ = target_points(session, opts['start_gap_mm'] / 1000)
    endpoint = contact + final_gap_mm / 1000 * outward
    steps = math.ceil((opts['start_gap_mm'] - final_gap_mm) / opts['step_mm'])
    seed = np.asarray(initial_q)
    stages, blockers = [], []
    if cfg['side_grasp'].get('transfer_mode') == 'direct_checked':
        pose = flange_target(endpoint, rotation, tcp, 'tcp')
        ik = screen.arm.ik(pose, seed, max_iterations=600,
                           position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4)
        if ik.success:
            # Solve once, but retain dense checks along the entire approach.
            delta = float(np.max(np.abs(np.asarray(ik.joints_rad) - seed)))
            path = screen.arm.check_joint_path(seed, ik.joints_rad,
                max_sample_step_rad=max(1e-8, min(math.radians(.25), delta / steps)))
            review = screen.check(path.samples_rad, scene, False, cfg,
                                  allow_hand_cup_contact=hand_contact_allowed(cfg['side_grasp']))
            for q in path.samples_rad:
                offset = (np.asarray(screen.arm.fk(q)) @ tcp)[:3, 3] - contact
                along = float(offset @ outward)
                if (np.linalg.norm(offset - along * outward) > .001
                        or not final_gap_mm / 1000 - .0001 <= along <= opts['start_gap_mm'] / 1000 + .0001):
                    review['blockers'].append('sampled approach leaves radial corridor')
                    break
            if path.kinematic_checks_passed and not review['blockers'] and delta <= math.radians(20):
                return [dict(name='approach_continuous', kind='arm', state='APPROACH_CLOSE_READY',
                             current_q_rad=seed.tolist(), target_q_rad=list(ik.joints_rad),
                             T_base_flange=pose.tolist(), tcp_target_base_m=endpoint.tolist(),
                             ik=asdict(ik), screen=review, speed_percent=opts['speed_percent'],
                             merged_waypoints=steps, joint_path_samples=len(path.samples_rad))], []
    for index in range(1, steps + 1):
        point = beginning + (endpoint - beginning) * index / steps
        pose = flange_target(point, rotation, tcp, 'tcp')
        ik = screen.arm.ik(pose, seed, max_iterations=600,
                           position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4)
        if not ik.success:
            blockers.append(f'approach/{index}: IK {ik.reason}')
            break
        path = screen.arm.check_joint_path(seed, ik.joints_rad)
        if (not path.kinematic_checks_passed
                or np.max(np.abs(np.array(ik.joints_rad) - seed)) > math.radians(20)):
            blockers.append(f'approach/{index}: joint limit/discontinuity')
            break
        review = screen.check(path.samples_rad, scene, False, cfg,
                              allow_hand_cup_contact=hand_contact_allowed(cfg['side_grasp']))
        for q in path.samples_rad:
            offset = (np.asarray(screen.arm.fk(q)) @ tcp)[:3, 3] - contact
            if np.linalg.norm(offset - (offset @ outward) * outward) > .001:
                review['blockers'].append('sampled line deviation exceeds 1 mm')
                break
        stages.append(dict(name=f'approach_{index}', kind='arm', state='APPROACH_CLOSE_READY',
                           current_q_rad=seed.tolist(), target_q_rad=ik.joints_rad,
                           T_base_flange=pose.tolist(), tcp_target_base_m=point.tolist(),
                           ik=asdict(ik), screen=review, speed_percent=opts['speed_percent']))
        blockers.extend(f'approach/{index}: {b}' for b in review['blockers'])
        seed = np.asarray(ik.joints_rad)
        if blockers:
            break
    return stages, blockers
