"""Read-only NERO/right-Revo2 hand geometry against the observed cup and table.

Frames are column-vector, metres, radians. The tabletop is the observed
support-center plane with its measured upright normal; the cup is an expanded
upright cylinder. This is a *screen*: joint-linear samples and model mesh
bounds cannot establish the actual controller trajectory, mesh-to-mesh
clearance, calibration accuracy, cable clearance, or the real hand posture.
No camera, CAN, arm SDK, or hand SDK is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from cup_grasp_demo.flow.transforms import inverse, matrix, pose_matrix


REPO = Path(__file__).resolve().parents[1]
DESCRIPTION = REPO / 'nero_revo2_control/models/hand_geometry'
RIGHT_XACRO = DESCRIPTION / 'nero/urdf/nero_with_right_revo2_description.xacro'
FLANGE_XACRO = DESCRIPTION / 'nero/urdf/nero_with_revo2_flange_description.xacro'
RIGHT_URDF = DESCRIPTION / 'revo2/urdf/revo2_right_hand.urdf'
MESH_ROOT = DESCRIPTION / 'revo2/meshes/revo2_right_hand'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
DEFAULT_MARGIN_M = 0.005


def _xyz(text: str) -> tuple[float, float, float]:
    values = tuple(float(x) for x in text.split())
    if len(values) != 3 or any(not math.isfinite(x) for x in values):
        raise ValueError(f'Expected three finite URDF values: {text!r}')
    return values


def _origin(joint) -> np.ndarray:
    node = joint.find('origin')
    return pose_matrix([*_xyz(node.get('xyz', '0 0 0')),
                        *_xyz(node.get('rpy', '0 0 0'))])


def _fixed_joint(path: Path, name: str, parent: str, child: str) -> np.ndarray:
    root = ElementTree.parse(path).getroot()
    matches = [j for j in root.findall('joint') if j.get('name') == name]
    if len(matches) != 1 or matches[0].get('type') != 'fixed':
        raise ValueError(f'Expected one fixed joint {name} in {path}')
    joint = matches[0]
    if joint.find('parent').get('link') != parent or joint.find('child').get('link') != child:
        raise ValueError(f'Wrong URDF chain at {name}')
    return _origin(joint)


def _axis_rotation(axis, theta: float) -> np.ndarray:
    x, y, z = axis
    c, s, v = math.cos(theta), math.sin(theta), 1-math.cos(theta)
    r = np.array([[c+x*x*v, x*y*v-z*s, x*z*v+y*s],
                  [y*x*v+z*s, c+y*y*v, y*z*v-x*s],
                  [z*x*v-y*s, z*y*v+x*s, c+z*z*v]])
    out = np.eye(4)
    out[:3, :3] = r
    return out


def _stl_bounds(path: Path) -> tuple[np.ndarray, float, np.ndarray]:
    """Exact triangle vertices and their conservative enclosing sphere."""
    raw = path.read_bytes()
    if len(raw) >= 84 and 84 + 50*struct.unpack_from('<I', raw, 80)[0] == len(raw):
        triangles = struct.unpack_from('<I', raw, 80)[0]
        records = np.frombuffer(raw, dtype=np.dtype([('normal', '<f4', (3,)),
                         ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]),
                         count=triangles, offset=84)
        vertices = np.asarray(records['vertices'], dtype=float).reshape(-1, 3)
    else:
        lines = raw.decode('ascii').splitlines()
        vertices = np.array([_xyz(line.strip().removeprefix('vertex ').strip())
                             for line in lines if line.strip().startswith('vertex ')])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f'Invalid STL vertices: {path}')
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo+hi)/2
    radius = float(np.linalg.norm(vertices-center, axis=1).max())
    if radius <= 0 or radius > 0.5:
        raise ValueError(f'Unexpected hand collision mesh scale: {path} radius {radius}')
    return center, radius, vertices


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: tuple[float, float, float]
    lower: float
    upper: float
    mimic: tuple[str, float, float] | None


class RightRevo2Model:
    """Official NERO mounting joints plus the five URDF finger chains."""

    def __init__(self):
        self.T_flange_hand_base = matrix(
            _fixed_joint(FLANGE_XACRO, 'revo2_flange_joint', 'link7', 'revo2_flange')
            @ _fixed_joint(RIGHT_XACRO, 'hand_base_joint',
                           'revo2_flange', 'right_base_link'))
        root = ElementTree.parse(RIGHT_URDF).getroot()
        joints = {}
        for j in root.findall('joint'):
            parent, child = j.find('parent').get('link'), j.find('child').get('link')
            if not child.startswith('right_'):
                continue
            kind = j.get('type')
            if kind not in ('fixed', 'revolute'):
                raise ValueError(f'Unsupported Revo2 joint type {kind}')
            limit = j.find('limit')
            mimic = j.find('mimic')
            joints[child] = Joint(
                j.get('name'), parent, child, _origin(j),
                _xyz(j.find('axis').get('xyz')) if kind == 'revolute' else (0., 0., 0.),
                float(limit.get('lower')) if limit is not None and kind == 'revolute' else 0.,
                float(limit.get('upper')) if limit is not None and kind == 'revolute' else 0.,
                ((mimic.get('joint'), float(mimic.get('multiplier', '1')),
                  float(mimic.get('offset', '0'))) if mimic is not None else None),
            )
        self.joints = joints
        self.by_name = {joint.name: joint for joint in joints.values()}
        expected = {f'right_{f}_tip_link' for f in FINGERS}
        if not expected <= joints.keys():
            raise ValueError('Official right-hand URDF lacks fingertip frames')
        self.collision = {}
        for link in root.findall('link'):
            name = link.get('name')
            collision = link.find('collision')
            if collision is None:
                continue
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                raise ValueError(f'Unexpected non-mesh Revo2 collision at {name}')
            filename = mesh.get('filename').split('/')[-1]
            source = MESH_ROOT / filename
            center, radius, vertices = _stl_bounds(source)
            origin = collision.find('origin')
            mesh_transform = (pose_matrix([*_xyz(origin.get('xyz', '0 0 0')),
                                           *_xyz(origin.get('rpy', '0 0 0'))])
                              if origin is not None else np.eye(4))
            self.collision[name] = (center, radius, vertices, mesh_transform, source)
        self.provenance = {
            'description': 'official NERO right-Revo2 xacro + URDF collision STLs',
            'flange_frame': 'link7', 'hand_base_frame': 'right_base_link',
            'model_sources': [{
                'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()
            } for p in (FLANGE_XACRO, RIGHT_XACRO, RIGHT_URDF)],
            'mesh_count': len(self.collision),
            'collision_mesh_sources': [{
                'link': name, 'filename': source.name,
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest()
            } for name, (_, _, _, _, source) in self.collision.items()],
            'T_flange_hand_base': self.T_flange_hand_base.tolist(),
        }

    def _angles(self, joint_positions_rad: dict[str, float] | None) -> dict[str, float]:
        supplied = {} if joint_positions_rad is None else dict(joint_positions_rad)
        if any(name not in self.by_name for name in supplied):
            raise ValueError('Unknown right-Revo2 URDF joint angle')
        angles = {}
        for name, joint in self.by_name.items():
            if joint.axis == (0., 0., 0.):
                if name in supplied and abs(float(supplied[name])) > 1e-12:
                    raise ValueError(f'Fixed joint {name} cannot move')
                angles[name] = 0.
                continue
            if joint.mimic is not None:
                if name in supplied:
                    raise ValueError(f'Mimic joint {name} must follow its parent')
                master, factor, offset = joint.mimic
                value = factor*float(supplied.get(master, 0.))+offset
            else:
                value = float(supplied.get(name, 0.))
            if not math.isfinite(value) or not joint.lower-1e-8 <= value <= joint.upper+1e-8:
                raise ValueError(f'{name} angle outside official URDF range')
            angles[name] = value
        return angles

    def link_transforms_from_flange(
        self, T_base_flange, *, joint_positions_rad: dict[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        flange = matrix(T_base_flange)
        angles = self._angles(joint_positions_rad)
        links = {'right_base_link': matrix(flange @ self.T_flange_hand_base)}
        remaining = dict(self.joints)
        while remaining:
            advanced = False
            for child, joint in list(remaining.items()):
                if joint.parent not in links:
                    continue
                transform = links[joint.parent] @ joint.origin
                if joint.axis != (0., 0., 0.):
                    transform = transform @ _axis_rotation(joint.axis, angles[joint.name])
                links[child] = matrix(transform)
                del remaining[child]
                advanced = True
            if not advanced:
                raise ValueError('Right-Revo2 URDF contains disconnected/cyclic finger joints')
        return links

    def tip_transforms_from_flange(
        self, T_base_flange, *, joint_positions_rad: dict[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        links = self.link_transforms_from_flange(
            T_base_flange, joint_positions_rad=joint_positions_rad)
        return {finger: links[f'right_{finger}_tip_link'] for finger in FINGERS}

    def tip_transforms_from_palm(
        self, T_base_palm, T_flange_palm, *,
        joint_positions_rad: dict[str, float] | None = None
    ) -> dict[str, np.ndarray]:
        flange = matrix(matrix(T_base_palm) @ inverse(T_flange_palm))
        return self.tip_transforms_from_flange(
            flange, joint_positions_rad=joint_positions_rad)


def _unit(value, name):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f'{name} must be three finite metres/components')
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f'{name} cannot be zero')
    return v/n


def _cup_point_clearance(point, support, axis, radius, height):
    """Positive signed distance outside capped cylinder, negative inside."""
    v = point-support
    h = float(v @ axis)
    radial = float(np.linalg.norm(v-h*axis))
    dr = radial-radius
    dh = max(-h, h-height, 0.)
    if dr <= 0 and dh == 0:
        return -min(-dr, h, height-h), radial, h
    return math.hypot(max(dr, 0.), dh), radial, h


def check_hand_pose(
    T_base_flange, *, support_center_m, axis, cup_diameter_m, cup_height_m,
    joint_positions_rad: dict[str, float] | None = None,
    margin_m: float = DEFAULT_MARGIN_M, model: RightRevo2Model | None = None
) -> dict:
    """Screen fingertip origins and enclosing URDF collision-mesh spheres.

    Sphere overlap is a conservative *possible* collision, not exact mesh
    intersection. A passing sphere screen proves only clearance for these
    model meshes at this sampled pose, within supplied geometry assumptions.
    """
    model = model or RightRevo2Model()
    if not (math.isfinite(margin_m) and 0 <= margin_m <= .03):
        raise ValueError('margin_m must be 0..0.03 metres')
    diameter, height = float(cup_diameter_m), float(cup_height_m)
    if not (math.isfinite(diameter) and .02 <= diameter <= .20
            and math.isfinite(height) and .03 <= height <= .30):
        raise ValueError('Cup diameter/height outside measured plausible range')
    support = np.asarray(support_center_m, dtype=float)
    if support.shape != (3,) or not np.isfinite(support).all():
        raise ValueError('support_center_m must be three finite metres')
    cup_axis = _unit(axis, 'cup/table axis')
    links = model.link_transforms_from_flange(
        T_base_flange, joint_positions_rad=joint_positions_rad)
    tips = {}
    blockers = []
    for finger in FINGERS:
        point = links[f'right_{finger}_tip_link'][:3, 3]
        altitude = float((point-support) @ cup_axis)
        clearance, radial, cup_h = _cup_point_clearance(
            point, support, cup_axis, diameter/2, height)
        table_hit, cup_hit = altitude <= margin_m, clearance <= margin_m
        tips[finger] = {
            'link': f'right_{finger}_tip_link', 'position_base_m': point.tolist(),
            'table_altitude_m': altitude, 'cup_signed_clearance_m': clearance,
            'cup_radial_distance_m': radial, 'cup_axis_height_m': cup_h,
            'table_margin_passed': not table_hit, 'cup_margin_passed': not cup_hit,
        }
        if table_hit:
            blockers.append(f'{finger} tip within {margin_m*1000:g} mm of table plane')
        if cup_hit:
            blockers.append(f'{finger} tip within {margin_m*1000:g} mm of cup cylinder')
    collision_mesh_bounds = {}
    exact_table_minimum = math.inf
    exact_table_blockers = []
    for name, (local_center, radius, vertices, mesh_origin, source) in model.collision.items():
        transform = links[name] @ mesh_origin
        center = transform[:3, :3] @ local_center + transform[:3, 3]
        altitude = float((center-support) @ cup_axis)
        # Signed distance to a plane is affine, so minimum over every triangle
        # occurs at one of its STL vertices. This is exact for the URDF mesh
        # tabletop screen, apart from supplied model/plane uncertainty.
        local_plane_axis = transform[:3, :3].T @ cup_axis
        exact_mesh_table_minimum = float(np.min(vertices @ local_plane_axis)
                                         + (transform[:3, 3]-support) @ cup_axis)
        exact_table_minimum = min(exact_table_minimum, exact_mesh_table_minimum)
        table_vertex_hit = exact_mesh_table_minimum <= margin_m
        if table_vertex_hit:
            exact_table_blockers.append(f'{name} collision mesh within table margin')
        clearance, _, _ = _cup_point_clearance(
            center, support, cup_axis, diameter/2, height)
        table_overlap = altitude-radius <= margin_m
        cup_overlap = clearance-radius <= margin_m
        collision_mesh_bounds[name] = {
            'mesh': source.name, 'enclosing_sphere_center_base_m': center.tolist(),
            'enclosing_sphere_radius_m': radius,
            'lower_table_bound_m': altitude-radius,
            'exact_stl_triangle_table_min_m': exact_mesh_table_minimum,
            'exact_stl_triangle_table_margin_passed': not table_vertex_hit,
            'lower_cup_bound_m': clearance-radius,
            'possible_table_overlap': table_overlap,
            'possible_cup_overlap': cup_overlap,
        }
        if cup_overlap:
            blockers.append(f'{name} mesh sphere may overlap cup')
    blockers.extend(exact_table_blockers)
    table_passed = (all(t['table_margin_passed'] for t in tips.values())
                    and not exact_table_blockers)
    cup_passed = (all(t['cup_margin_passed'] for t in tips.values())
                  and not any(item['possible_cup_overlap']
                              for item in collision_mesh_bounds.values()))
    return {
        'schema': 1, 'kind': 'read_only_open_hand_geometry_screen',
        'frame': 'base_link', 'units': 'm_rad',
        'hand_joint_positions_rad': model._angles(joint_positions_rad),
        'hand_posture_source': ('caller_supplied_urdf_angles' if joint_positions_rad is not None
                                else 'nominal_urdf_zero_open_assumption'),
        'table_plane': {'point_base_m': support.tolist(), 'normal_base': cup_axis.tolist()},
        'cup_cylinder': {'support_center_base_m': support.tolist(),
                         'axis_base': cup_axis.tolist(), 'radius_m': diameter/2,
                         'height_m': height},
        'margin_m': margin_m, 'tips': tips,
        'collision_mesh_bounds': collision_mesh_bounds,
        'exact_collision_mesh_table_min_m': exact_table_minimum,
        'table_margin_passed': table_passed,
        'cup_margin_passed': cup_passed,
        'possible_collision': bool(blockers), 'blockers': blockers,
        'model_provenance': model.provenance,
        'limits': ['Tip link origins are kinematic frames, not full fingertip surfaces.',
                   'Table distance is exact for the URDF STL collision triangles at this pose.',
                   'Cup uses enclosing spheres, so overlap can be a false positive.',
                   'Cup is a complete cylinder inferred from visible side, with hidden surfaces unmeasured.',
                   'Table plane uses observed support center and assumed upright axis.',
                   'Actual hand joints, soft pads, cables, arm links, self collision, and controller path are not verified.'],
    }


def check_plan_geometry(
    plan: dict, *, ik_review: dict | None = None,
    joint_positions_rad: dict[str, float] | None = None,
    margin_m: float = DEFAULT_MARGIN_M, plan_sha256: str | None = None
) -> dict:
    """Screen planned flange poses and, when supplied, joint-linear IK samples."""
    if plan.get('schema') != 1 or plan.get('kind') != 'read_only_side_grasp_proposal':
        raise ValueError('Expected read-only side-grasp proposal')
    if plan_sha256 is not None:
        if (not isinstance(plan_sha256, str) or len(plan_sha256) != 64
                or any(c not in '0123456789abcdef' for c in plan_sha256)):
            raise ValueError('plan_sha256 must be a lowercase SHA-256 hex digest')
        if (ik_review is not None
                and ik_review.get('source_plan_sha256') != plan_sha256):
            raise ValueError('IK review SHA-256 does not match this plan file')
    cup = plan['cup_base']
    support = cup['support_center_m']
    axis = cup['axis']  # measured support-plane normal, never assumed base +Z
    dimensions = cup['dimensions']
    diameter = dimensions['observed_side_diameter_m']
    height = plan.get('center_provenance', {}).get('nominal_height_m')
    if height is None:
        height = dimensions['observed_height_m']
    model = RightRevo2Model()
    stage_names = (('clearance',) if plan.get('checks', {}).get('clearance_required') else ()) + (
        'pregrasp', 'contact', 'lift')
    stages = {}
    for name in stage_names:
        stage = plan['waypoints'][name]
        screen = check_hand_pose(
            stage['T_base_flange'], support_center_m=support, axis=axis,
            cup_diameter_m=diameter, cup_height_m=height,
            joint_positions_rad=joint_positions_rad, margin_m=margin_m, model=model)
        stages[name] = {
            'flange_target_pose': screen,
            'tip_table_min_m': min(v['table_altitude_m'] for v in screen['tips'].values()),
            'tip_cup_min_m': min(v['cup_signed_clearance_m'] for v in screen['tips'].values()),
            'exact_mesh_table_min_m': screen['exact_collision_mesh_table_min_m'],
            'table_margin_passed': screen['table_margin_passed'],
            'cup_margin_passed': screen['cup_margin_passed'],
            'cup_contact_policy': ('required_clearance' if name in ('clearance', 'pregrasp')
                                   else 'intentional_contact_not_granted_as_clearance'),
            'stage_geometry_passed': (
                screen['table_margin_passed'] and
                (screen['cup_margin_passed'] if name in ('clearance', 'pregrasp') else True)),
            'possible_collision': screen['possible_collision'],
        }
    path = {}
    if ik_review is not None:
        if (ik_review.get('kind') != 'read_only_urdf_kinematics_review'
                or ik_review.get('source_plan_frame_id') != plan.get('snapshot_frame_id')
                or ik_review.get('sequence') != ['current', *stage_names]):
            raise ValueError('IK review does not describe this plan frame and stage sequence')
        from nero_revo2_control.kinematics import load_model
        arm = load_model()
        prior = ik_review['current_joints_rad']
        for item in ik_review['stages']:
            name = item['name']
            if name not in stages or not item['ik']['success']:
                raise ValueError('IK review stage missing or failed')
            target = item['ik']['joints_rad']
            flange_target = matrix(plan['waypoints'][name]['T_base_flange'])
            flange_fk = matrix(arm.fk(target))
            position_error = float(np.linalg.norm(flange_fk[:3, 3]-flange_target[:3, 3]))
            trace = float(np.trace(flange_fk[:3, :3].T @ flange_target[:3, :3]))
            orientation_error = math.acos(max(-1., min(1., (trace-1.)/2.)))
            if position_error > .001 or orientation_error > .01:
                raise ValueError(f'IK {name} joints do not reproduce this flange waypoint')
            checked = arm.check_joint_path(prior, target)
            if not checked.kinematic_checks_passed:
                raise ValueError(f'IK {name} joint-linear path failed kinematic limits')
            samples = checked.samples_rad
            reports = [check_hand_pose(
                arm.fk(q), support_center_m=support, axis=axis,
                cup_diameter_m=diameter, cup_height_m=height,
                joint_positions_rad=joint_positions_rad, margin_m=margin_m, model=model)
                for q in samples]
            path[name] = {
                'sample_type': 'URDF joint_linear_interpolation_only',
                'sample_count': len(reports),
                'tip_table_min_m': min(min(v['table_altitude_m'] for v in r['tips'].values())
                                   for r in reports),
                'tip_cup_min_m': min(min(v['cup_signed_clearance_m'] for v in r['tips'].values())
                                 for r in reports),
                'exact_mesh_table_min_m': min(r['exact_collision_mesh_table_min_m']
                                              for r in reports),
                'table_margin_passed': all(r['table_margin_passed'] for r in reports),
                'cup_margin_passed': all(r['cup_margin_passed'] for r in reports),
                'path_geometry_passed': (
                    all(r['table_margin_passed'] for r in reports)
                    and (all(r['cup_margin_passed'] for r in reports)
                         if name in ('clearance', 'pregrasp') else True)),
                'first_possible_collision_sample': next(
                    (i for i, r in enumerate(reports) if r['possible_collision']), None),
                'possible_collision': any(r['possible_collision'] for r in reports),
            }
            prior = target
    approach_names = [name for name in stage_names if name in ('clearance', 'pregrasp')]
    approach_passed = all(stages[n]['stage_geometry_passed'] for n in approach_names)
    if path:
        approach_passed = approach_passed and all(
            path[n]['path_geometry_passed'] for n in approach_names)
    contact_table_passed = stages['contact']['table_margin_passed']
    if path:
        contact_table_passed = contact_table_passed and path['contact']['table_margin_passed']
    return {
        'schema': 1, 'kind': 'read_only_plan_hand_geometry_screen',
        'plan_frame_id': plan.get('snapshot_frame_id'),
        'plan_sha256': plan_sha256,
        'ik_review_source_plan_sha256': (
            ik_review.get('source_plan_sha256') if ik_review is not None else None),
        'stage_sequence': list(stage_names), 'stages': stages, 'joint_linear_paths': path,
        'approach_geometry_passed': approach_passed,
        'contact_table_margin_passed': contact_table_passed,
        'contact_geometry_passed': approach_passed and contact_table_passed,
        'actual_hand_posture_verified': False,
        'any_possible_collision': any(s['possible_collision'] for s in stages.values())
                                  or any(s['possible_collision'] for s in path.values()),
        'motion_sent': False,
        'limitations': ['Only right-Revo2 hand model; arm/cables/scene self collision omitted.',
                        'IK interpolation does not identify real controller trajectory.',
                        'An open hand has to be confirmed from current actuator feedback; URDF zero is an assumption.',
                        'Cup-side contact may be intentional, but this screen does not grant contact clearance.'],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--ik-review', type=Path)
    parser.add_argument('--output', type=Path, required=True,
                        help='Create a new JSON report; existing files are never overwritten')
    parser.add_argument('--margin-mm', type=float, default=DEFAULT_MARGIN_M*1000)
    args = parser.parse_args(argv)
    try:
        plan_raw = args.plan.read_bytes()
        plan_sha = hashlib.sha256(plan_raw).hexdigest()
        plan = json.loads(plan_raw)
        ik = json.loads(args.ik_review.read_bytes()) if args.ik_review else None
        result = check_plan_geometry(
            plan, ik_review=ik, plan_sha256=plan_sha,
            margin_m=args.margin_mm/1000)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        print(json.dumps({
            'output': str(args.output), 'plan_sha256': plan_sha,
            'approach_geometry_passed': result['approach_geometry_passed'],
            'contact_table_margin_passed': result['contact_table_margin_passed'],
            'contact_geometry_passed': result['contact_geometry_passed'],
            'actual_hand_posture_verified': False,
            'stages': {name: {
                'tip_table_min_mm': round(stage['tip_table_min_m']*1000, 2),
                'exact_mesh_table_min_mm': round(stage['exact_mesh_table_min_m']*1000, 2),
                'table_margin_passed': stage['table_margin_passed'],
                'cup_margin_passed': stage['cup_margin_passed'],
            } for name, stage in result['stages'].items()},
            'motion_sent': False,
        }, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f'hand geometry screen: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
