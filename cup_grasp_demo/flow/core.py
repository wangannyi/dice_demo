"""Offline comparison geometry. No camera, CAN, SDK, or control side effects."""

from dataclasses import asdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np

from cup_grasp_demo.hand_geometry import (
    DESCRIPTION, FLANGE_XACRO, RightRevo2Model, _fixed_joint, _stl_bounds,
)
from nero_calibration.core import matrix
from nero_revo2_control.kinematics import _axis_rotation, _rpy_transform, load_model

ROOT = Path(__file__).resolve().parents[2]
_screen_cache = ContextVar('pipeline_screen_geometry', default=None)
_digest_cache = ContextVar('pipeline_file_digests', default=None)


@contextmanager
def cached_file_digests():
    """Reuse hashes within FAST while inode/size/mtime/ctime stay unchanged."""
    token = _digest_cache.set({})
    try:
        yield
    finally:
        _digest_cache.reset(token)


@contextmanager
def cached_screen_geometry():
    """Reuse immutable model geometry only for one fingerprint-checked FAST run."""
    token = _screen_cache.set({})
    try:
        yield
    finally:
        _screen_cache.reset(token)


def digest(path):
    path = Path(path)
    cache = _digest_cache.get()
    if cache is None:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    path = path.resolve()
    def signature():
        s = path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    before = signature()
    # Recently changed files can share one coarse filesystem timestamp tick.
    # Always read them; cache only files whose metadata has settled for 2 s.
    settled = all(time.time_ns() - stamp >= 2_000_000_000 for stamp in before[-2:])
    if settled and path in cache and cache[path][0] == before:
        return cache[path][1]
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    if signature() != before:
        raise ValueError(f'File changed while hashing: {path}')
    if settled:
        cache[path] = before, value
    else:
        cache.pop(path, None)
    return value


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_config(path):
    from cup_grasp_demo.flow.cup_selection import selection_options
    from cup_grasp_demo.flow.contact_geometry import contact_azimuth
    cfg = read_json(path)
    from cup_grasp_demo.flow.cup_perception import perception_options
    if 'cup_perception' in cfg:
        cfg['cup_perception'] = perception_options(cfg)
    selection_options(cfg)
    contact_azimuth(cfg.get('contact_direction'))
    from cup_grasp_demo.flow.parameters import tcp_offset
    tcp_offset(cfg)
    if 'side_grasp' in cfg:
        from cup_grasp_demo.flow.grasp import options
        options(cfg)
    if 'home_open_hand' in cfg and not isinstance(cfg['home_open_hand'], bool):
        raise ValueError('home_open_hand must be a boolean')
    if 'fast_record_shake_video' in cfg and not isinstance(cfg['fast_record_shake_video'], bool):
        raise ValueError('fast_record_shake_video must be a boolean')
    if 'fast_capture_reuse_max_age_s' in cfg:
        from cup_grasp_demo.flow.parameters import number
        number(cfg['fast_capture_reuse_max_age_s'], 'fast_capture_reuse_max_age_s', 0, 300)
    for key in ('fast_single_capture', 'fast_reuse_table'):
        if key in cfg and not isinstance(cfg[key], bool):
            raise ValueError(f'{key} must be a boolean')
    from cup_grasp_demo.flow.joint_delivery import delivery_options
    delivery_options(cfg.get('joint_delivery'))
    for key in ('home', 'calibration', 'orientation_reference', 'tcp_candidate', 'grasp_config'):
        source = Path(cfg[key])
        cfg[key] = str(source if source.is_absolute() else ROOT / source)
    for key, low, high in (
        ('speed_percent', 1, 100 if cfg.get('pipeline_strategy') == 'green_open_cup' else 10), ('timeout_s', 1, 120),
        ('table_margin_mm', 5, 30), ('cup_margin_mm', 3, 30),
        ('clearance_height_mm', 100, 250), ('plane_tolerance_mm', 1, 6 if cfg.get('pipeline_strategy') == 'green_open_cup' else 5),
        ('start_tolerance_deg', .01, .5), ('plan_max_age_s', 30, 1800),
        ('contact_height_fraction', .1, .95),
    ):
        value = float(cfg[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid configuration: {key}')
    if int(cfg['speed_percent']) != cfg['speed_percent']:
        raise ValueError('speed_percent must be an integer')
    cfg['speed_percent'] = int(cfg['speed_percent'])
    return cfg


def configured_tcp(cfg):
    """Offset the candidate in its named open-hand link axes, in millimetres."""
    from cup_grasp_demo.flow.parameters import tcp_offset
    candidate = read_json(cfg['tcp_candidate'])
    transform = matrix(candidate['T_flange_contact_candidate']).copy()
    links = RightRevo2Model().link_transforms_from_flange(np.eye(4))
    link = links[candidate['link']]
    transform[:3, 3] += link[:3, :3] @ (np.array(tcp_offset(cfg)) / 1000)
    return transform


def flange_target(point_base, rotation_base_flange, tcp, frame):
    """Keep the same flange orientation in both experiments, including J7."""
    target = np.eye(4)
    target[:3, :3] = rotation_base_flange
    target[:3, 3] = point_base
    if frame == 'tcp':
        target[:3, 3] -= target[:3, :3] @ matrix(tcp)[:3, 3]
    elif frame != 'flange':
        raise ValueError('frame must be flange or tcp')
    return matrix(target)


def cover(vertices, maximum=.006):
    """Cover each triangle with spheres, each radius <= maximum."""
    todo = vertices.reshape(-1, 3, 3)
    centers, radii = [], []
    while len(todo):
        center = todo.mean(axis=1)
        radius = np.linalg.norm(todo - center[:, None, :], axis=2).max(axis=1)
        done = radius <= maximum
        centers.append(center[done])
        radii.append(radius[done])
        tri = todo[~done]
        if not len(tri):
            break
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        todo = np.concatenate([np.stack(x, axis=1) for x in
                               [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]])
    return np.concatenate(centers), np.concatenate(radii)


class Screen:
    """Sampled arm/adapter/open-hand vs table/cup; not a collision planner."""

    def __init__(self, *, table_only=False):
        self.table_only = table_only
        key = "table_geometry" if table_only else "geometry"
        cache = _screen_cache.get()
        if cache and key in cache:
            self.arm, self.meshes = cache[key]
            return
        self.meshes = {}
        self._sources = {} if cache is not None else None
        if cache and 'source_geometry' in cache:
            self.arm, sources = cache['source_geometry']
            for name, (vertices, unique) in sources.items():
                self.add(name, vertices, unique=unique)
            cache[key] = (self.arm, self.meshes)
            return
        self.arm = load_model()
        hand = RightRevo2Model()
        for i in range(1, 8):
            name = f'link{i}'
            self.add(name, _stl_bounds(DESCRIPTION / f'nero/meshes/{name}.stl')[2])
        mount = _fixed_joint(FLANGE_XACRO, 'revo2_flange_joint', 'link7', 'revo2_flange')
        vertices = _stl_bounds(DESCRIPTION / 'nero/meshes/revo2_flange.stl')[2]
        self.add('revo2_flange', vertices @ mount[:3, :3].T + mount[:3, 3])
        links = hand.link_transforms_from_flange(np.eye(4))
        for name, (_, _, vertices, origin, _) in hand.collision.items():
            transform = links[name] @ origin
            self.add(name, vertices @ transform[:3, :3].T + transform[:3, 3])
        if cache is not None:
            cache[key] = (self.arm, self.meshes)
            cache['source_geometry'] = (self.arm, self._sources)

    def add(self, name, vertices, *, unique=None):
        # Qhull accepts repeated triangle vertices. Sorting the entire STL first
        # is unnecessary for plane extrema; retain deduplication for cup covers.
        vertices_unique = (vertices if self.table_only else np.unique(vertices, axis=0)) if unique is None else unique
        if getattr(self, '_sources', None) is not None:
            self._sources[name] = (vertices, unique if self.table_only else vertices_unique)
        if self.table_only:
            # A plane's minimum over a mesh occurs on its convex hull. Keep every
            # trajectory sample; omit triangle-sphere covers used only for cups.
            from scipy.spatial import ConvexHull, QhullError
            try:
                hull = ConvexHull(vertices_unique)
                extrema = vertices_unique[hull.vertices]
            except QhullError:
                extrema = vertices_unique  # Degenerate meshes retain every point.
            self.meshes[name] = (extrema, None, None)
        else:
            self.meshes[name] = (vertices_unique, *cover(vertices))

    def check_table_batch(self, samples, scene, cfg):
        """Every sample and the mesh plane minimum, evaluated in bounded batches."""
        support = np.asarray(scene['cup_support_base_m'])
        normal = np.asarray(scene['cup_normal_base'])
        qs = np.asarray(samples, dtype=float)
        if qs.ndim != 2 or qs.shape[1] != 7 or not len(qs) or not np.isfinite(qs).all():
            raise ValueError('Invalid table trajectory samples')
        minimum, link = math.inf, None
        origins = [_rpy_transform(j.origin_xyz, j.origin_rpy) for j in self.arm.joints]
        for offset in range(0, len(qs), 64):
            chunk = qs[offset:offset + 64]
            pose = np.broadcast_to(np.eye(4), (len(chunk), 4, 4)).copy()
            transforms = {}
            for i, joint in enumerate(self.arm.joints):
                # Use the identical axis helper; batch the expensive mesh projections.
                rotations = np.asarray([_axis_rotation(joint.axis_xyz, q[i]) for q in chunk])
                pose = pose @ origins[i] @ rotations
                transforms[joint.child] = pose
            for name, (vertices, _, _) in self.meshes.items():
                transform = transforms.get(name, pose)
                local_normal = np.einsum('nji,j->ni', transform[:, :3, :3], normal)
                height = (transform[:, :3, 3] - support) @ normal
                value = float(np.min(vertices @ local_normal.T + height))
                if value < minimum:
                    minimum, link = value, name
        blockers = ([f'{link}: table clearance {minimum * 1000:.1f} mm']
                    if minimum < cfg['table_margin_mm'] / 1000 else [])
        return dict(table_min_mm=minimum * 1000, table_link=link,
                    cup_lower_bound_mm=None, cup_link=None, blockers=blockers)

    def check(self, samples, scene, cup_removed, cfg, *, allow_hand_cup_contact=False):
        if getattr(self, 'table_only', False) and not cup_removed:
            raise ValueError('Table-only geometry cannot check cup clearance')
        support = np.array(scene['cup_support_base_m'])
        normal = np.array(scene['cup_normal_base'])
        radius, height = scene['cup_envelope_radius_m'], scene['geometry']['height_m']
        table_min, cup_min = math.inf, math.inf
        table_link, cup_link = None, None
        rigid_min, hand_min = math.inf, math.inf
        rigid_link, hand_link = None, None
        for q in samples:
            transforms, pose = {}, np.eye(4)
            for joint, angle in zip(self.arm.joints, q):
                pose = pose @ _rpy_transform(joint.origin_xyz, joint.origin_rpy)
                pose = pose @ _axis_rotation(joint.axis_xyz, angle)
                transforms[joint.child] = pose
            for name, (vertices, centers, radii) in self.meshes.items():
                transform = transforms.get(name, pose)
                rot, pos = transform[:3, :3], transform[:3, 3]
                distance = float((vertices @ (rot.T @ normal)).min() + (pos - support) @ normal)
                if distance < table_min:
                    table_min, table_link = distance, name
                if cup_removed:
                    continue
                delta = centers @ rot.T + pos - support
                z = delta @ normal
                radial = np.linalg.norm(delta - z[:, None] * normal, axis=1) - radius
                vertical = np.maximum(-z, z - height)
                sdf = (np.hypot(np.maximum(radial, 0), np.maximum(vertical, 0))
                       + np.minimum(np.maximum(radial, vertical), 0))
                distance = float((sdf - radii).min())
                if distance < cup_min:
                    cup_min, cup_link = distance, name
                if allow_hand_cup_contact:
                    if name.startswith('right_'):
                        if distance < hand_min:
                            hand_min, hand_link = distance, name
                    elif distance < rigid_min:
                        rigid_min, rigid_link = distance, name
        blockers = []
        if table_min < cfg['table_margin_mm'] / 1000:
            blockers.append(f'{table_link}: table clearance {table_min * 1000:.1f} mm')
        enforced_min, enforced_link = (rigid_min, rigid_link) if allow_hand_cup_contact else (cup_min, cup_link)
        if not cup_removed and enforced_min < cfg['cup_margin_mm'] / 1000:
            blockers.append(f'{enforced_link}: cup clearance lower bound {enforced_min * 1000:.1f} mm')
        result = dict(table_min_mm=table_min * 1000, table_link=table_link,
                    cup_lower_bound_mm=None if cup_removed else cup_min * 1000,
                    cup_link=cup_link, blockers=blockers)
        if allow_hand_cup_contact and not cup_removed:
            result.update(hand_cup_contact_allowed=True,
                          non_hand_cup_lower_bound_mm=rigid_min * 1000 if math.isfinite(rigid_min) else None,
                          non_hand_cup_link=rigid_link,
                          hand_cup_lower_bound_mm=hand_min * 1000 if math.isfinite(hand_min) else None,
                          hand_cup_link=hand_link,
                          contact_warnings=([f'{hand_link}: allowed hand/cup contact; '
                                             f'clearance lower bound {hand_min * 1000:.1f} mm']
                                            if hand_min < cfg['cup_margin_mm'] / 1000 else []))
        return result


def make_plan(session, current, cfg, frame, gap_mm, cup_removed, screen=None, *, allow_hand_cup_contact=False):
    if not math.isfinite(gap_mm) or not 0 <= gap_mm <= 200:
        raise ValueError('gap_mm must be 0..200')
    if gap_mm == 0 and not cup_removed:
        raise ValueError('Exact-point experiments require removing the cup after capture')
    arm = load_model()
    current = np.asarray(current, dtype=float)
    if current.shape != (7,) or not np.isfinite(current).all():
        raise ValueError('Expected seven finite current joint angles')
    scene, tcp = session['scene'], matrix(session['T_flange_tcp'])
    rotation = np.array(session['R_base_flange'])
    point = np.array(session['contact_base_m']) + gap_mm / 1000 * np.array(scene['outward_base'])
    normal = np.array(scene['cup_normal_base'])
    fk = np.array(arm.fk(current))
    # First raise the current flange, then transfer above the target, then lower.
    final = flange_target(point, rotation, tcp, frame)
    reference = np.clip(read_json(cfg['orientation_reference'])['joints_rad'],
                        *np.array(arm.limits_rad).T)
    endpoint_trials = [arm.ik(final, np.clip(seed_value, *np.array(arm.limits_rad).T),
                              max_iterations=600, position_tolerance_m=1e-5,
                              orientation_tolerance_rad=1e-4)
                       for seed_value in (current, reference)]
    if not any(trial.success for trial in endpoint_trials):
        return dict(kind='flange_tcp_debug_plan', frame=frame, gap_mm=gap_mm,
                    cup_removed=cup_removed, start_q_rad=current.tolist(),
                    comparison_point_base_m=point.tolist(), T_base_flange_target=final.tolist(),
                    R_base_flange=rotation.tolist(), T_flange_tcp=tcp.tolist(), stages=[],
                    endpoint_trials=[asdict(trial) for trial in endpoint_trials],
                    blockers=['终点 IK 未找到解；可将杯子移近基座后重新 capture。'
                              '这是局部求解结果，不是整个工作空间不可达的证明'],
                    screen_passed=False, calibration_quality_passed=session['calibration_quality_passed'],
                    physical_tcp_verified=False, collision_verified=False)
    support = np.array(scene['cup_support_base_m'])
    high = max(float((fk[:3, 3] - support) @ normal),
               float((final[:3, 3] - support) @ normal) + cfg['clearance_height_mm'] / 1000)
    raised = fk.copy()
    raised[:3, 3] += (high - (fk[:3, 3] - support) @ normal) * normal
    over = final.copy()
    over[:3, 3] += (high - (final[:3, 3] - support) @ normal) * normal
    targets = [('raise', raised), ('transfer', over), ('lower', final)]
    screen = screen or Screen(table_only=cup_removed)
    previous, seed, stages, blockers = current, current, [], []
    # Short Cartesian waypoints limit the unchecked bow of each joint move.
    for name, target in targets:
        beginning = np.array(arm.fk(previous))
        steps = max(1, math.ceil(np.linalg.norm(target[:3, 3] - beginning[:3, 3]) / .02))
        # Translation is interpolated; orientation changes only on the high transfer.
        from scipy.spatial.transform import Rotation, Slerp
        interpolate = Slerp([0, 1], Rotation.from_matrix(np.stack([beginning[:3, :3], target[:3, :3]])))
        for index in range(1, steps + 1):
            ratio = index / steps
            waypoint = np.eye(4)
            waypoint[:3, :3] = interpolate(ratio).as_matrix()
            waypoint[:3, 3] = (1 - ratio) * beginning[:3, 3] + ratio * target[:3, 3]
            ik = arm.ik(waypoint, seed, max_iterations=600,
                        position_tolerance_m=1e-5, orientation_tolerance_rad=1e-4)
            if not ik.success:
                blockers.append(f'{name}/{index}: IK {ik.reason}')
                break
            checked = arm.check_joint_path(previous, ik.joints_rad)
            maximum_delta = float(np.degrees(np.max(np.abs(np.array(ik.joints_rad) - previous))))
            if not checked.kinematic_checks_passed or maximum_delta > 20:
                blockers.append(f'{name}/{index}: joint limit/discontinuity')
                break
            # Contact permission is limited to the target descent. Transit keeps
            # the original full-hand cup checks, as do comparison experiments.
            if allow_hand_cup_contact and name == 'lower':
                review = screen.check(checked.samples_rad, scene, cup_removed, cfg, allow_hand_cup_contact=True)
            else:
                review = screen.check(checked.samples_rad, scene, cup_removed, cfg)
            stages.append(dict(name=f'{name}_{index}', current_q_rad=list(previous),
                               target_q_rad=list(ik.joints_rad), ik=asdict(ik),
                               T_base_flange=waypoint.tolist(), screen=review))
            blockers.extend(f'{name}/{index}: {b}' for b in review['blockers'])
            previous = seed = np.array(ik.joints_rad)
        if blockers:
            break
    return dict(kind='flange_tcp_debug_plan', frame=frame, gap_mm=gap_mm,
                cup_removed=cup_removed, start_q_rad=current.tolist(),
                comparison_point_base_m=point.tolist(), T_base_flange_target=final.tolist(),
                R_base_flange=rotation.tolist(), T_flange_tcp=tcp.tolist(),
                stages=stages, blockers=blockers, screen_passed=not blockers,
                calibration_quality_passed=session['calibration_quality_passed'],
                physical_tcp_verified=False, collision_verified=False,
                limitations=['Sampled model table/cup checks; actual controller interpolation, '
                             'self-collisions, cables and unmodeled obstacles require operator observation',
                             'Open-hand geometry and provisional calibration are under test'])


def measured_offset(first, second):
    """Only an approximate differential estimate; never modify calibration."""
    for key in ('session_sha256', 'gap_mm'):
        if first[key] != second[key]:
            raise ValueError('Measurements must use the same frozen target and gap')
    if first['frame'] != 'flange' or second['frame'] != 'tcp':
        raise ValueError('Expected flange measurement then TCP measurement')
    if not np.allclose(first['R_base_flange'], second['R_base_flange'], atol=1e-8):
        raise ValueError('Flange orientations differ')
    delta = np.array(second['error_base_mm']) - first['error_base_mm']
    return (np.array(first['R_base_flange']).T @ delta).tolist()
