"""Replay an RGB-D batch and preview the annotated index contact; no hardware IO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import warnings

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from cup_grasp_demo.hand_geometry import RightRevo2Model, check_hand_pose  # noqa: E402
from cup_grasp_demo.planning import load_calibration  # noqa: E402
from dice_cup_localization.geometry import Config, _circle, _plane, deproject  # noqa: E402
from nero_calibration.core import inverse  # noqa: E402
from nero_revo2_control.kinematics import load_model  # noqa: E402


def section(points, table, normal, fraction, cfg, *, half_band_m=.004):
    """Use table-normal height, not optical Z, to select a cup cross-section."""
    if not 0 < fraction < 1:
        raise ValueError('Contact fraction must be inside (0, 1)')
    if not np.isfinite(half_band_m) or not 0 < half_band_m <= .004:
        raise ValueError('Section half-band must be inside (0, 0.004] m')
    heights = (points - table) @ normal
    height = float(np.quantile(heights, .99))
    if not cfg.min_height_m <= height <= cfg.max_height_m:
        raise ValueError('Observed height outside cup range')
    tangent = np.cross(normal, np.eye(3)[np.argmin(abs(normal))])
    tangent /= np.linalg.norm(tangent)
    basis = np.column_stack((tangent, np.cross(normal, tangent)))
    selected = abs(heights - fraction * height) < half_band_m
    center, radius, rms, arc = _circle((points[selected] - table) @ basis, cfg)
    center3 = table + basis @ center + fraction * height * normal
    return dict(height_m=height, contact_height_m=fraction * height,
                radius_m=radius, circle_rms_m=rms, visible_arc_deg=arc,
                section_points=int(selected.sum()), center_camera_m=center3.tolist(),
                normal_camera=normal.tolist(), basis_camera=basis.tolist())


def ray_surface_x(triangles, y, z):
    """Outermost +X mesh hit through a specified YZ location, in mesh metres."""
    triangles = np.asarray(triangles).reshape(-1, 3, 3)
    origin = np.array([-.1, y, z])
    direction = np.array([1., 0., 0.])
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    cross = np.cross(direction, edge2)
    determinant = np.einsum('ij,ij->i', edge1, cross)
    valid = abs(determinant) > 1e-12
    inv = np.zeros_like(determinant)
    inv[valid] = 1 / determinant[valid]
    offset = origin - triangles[:, 0]
    u = inv * np.einsum('ij,ij->i', offset, cross)
    q = np.cross(offset, edge1)
    v = inv * (q @ direction)
    t = inv * np.einsum('ij,ij->i', edge2, q)
    valid &= (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8) & (t >= 0)
    if not np.any(valid):
        raise ValueError('No mesh surface at selected finger location')
    return float(origin[0] + np.max(t[valid]))


def model_tcp(hand, finger_fraction):
    """Photo red mark lies between the two visible index hinge centers."""
    if not .1 <= finger_fraction <= .9:
        raise ValueError('Photo finger fraction must be in 0.1..0.9')
    transforms = hand.link_transforms_from_flange(np.eye(4))
    link = 'right_index_proximal_link'
    span = hand.joints['right_index_distal_link'].origin[2, 3]
    _, _, vertices, origin, _ = hand.collision[link]
    # Express the actual collision triangles in the proximal link frame.
    vertices = vertices @ origin[:3, :3].T + origin[:3, 3]
    z = float(span * finger_fraction)
    x = ray_surface_x(vertices, 0., z)
    local = np.array([x, 0., z, 1.])
    transform = np.eye(4)  # Contact frame axes intentionally parallel to flange.
    transform[:3, 3] = (transforms[link] @ local)[:3]
    return transform, transforms[link][:3, 0], dict(
        link=link, local_surface_point_m=local[:3].tolist(),
        finger_hinge_span_m=float(span), photo_fraction_estimate=finger_fraction,
        T_flange_contact_candidate=transform.tolist(),
        point_selection='Red-marked surface between visible hinges; not URDF joint origin',
        axes='Parallel to flange; local +X palmar normal stored separately',
        physical_tcp_verified=False, sdk_zero_matches_urdf_zero_verified=False,
        point_uncertainty_m=None, executable=False,
        source=hand.provenance)


def load_batch(dataset, *, min_frames=3):
    frames = sorted((dataset / 'rgbd').glob('frame_*.json'))
    if type(min_frames) is not int or min_frames < 1 or len(frames) < min_frames:
        raise ValueError(f'At least {min_frames} RGB-D frames are required')
    metas = [json.loads(p.read_text()) for p in frames]
    first = metas[0]
    for meta in metas:
        if any(meta[key] != first[key] for key in ('serial', 'intrinsics', 'depth_scale_m')):
            raise ValueError('Batch camera or geometry changed')
        if meta['depth_registered_to'] != 'color_optical':
            raise ValueError('Depth must be registered to color')
    stack = np.array([np.load(p.with_suffix('.npz'))['depth'].astype(float) for p in frames])
    stack[stack == 0] = np.nan
    if len(frames) == 1:
        # The median of one sample is that sample; avoid per-pixel masked
        # sorting and all-NaN warnings at missing-depth pixels in FAST mode.
        depth = np.nan_to_num(stack[0])
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            depth = np.nan_to_num(np.nanmedian(stack, axis=0))
    image = cv2.imread(str(frames[0].with_suffix('.png')))
    if image is None or image.shape[:2] != depth.shape:
        raise ValueError('Missing or mismatched RGB image')
    return first, depth, image, frames


def depth_candidates(depth, image, meta, cfg, fraction):
    """Diagnostic depth components in the red workspace; no semantic identity."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165))
           & (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50))
    intr, scale = meta['intrinsics'], meta['depth_scale_m']
    table, normal, _, _ = _plane(deproject(depth, red, intr, scale), cfg)
    yy, xx = np.indices(depth.shape)
    zz = depth * scale
    xyz = np.stack(((xx - intr['cx']) * zz / intr['fx'],
                    (yy - intr['cy']) * zz / intr['fy'], zz), axis=-1)
    contours, _ = cv2.findContours(red.astype('uint8'), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Red workspace not visible')
    workspace = np.zeros(depth.shape, 'uint8')
    cv2.fillConvexPoly(workspace, cv2.convexHull(max(contours, key=cv2.contourArea)), 1)
    altitude = (xyz - table) @ normal
    foreground = ((altitude > .015) & (altitude < .30) & (depth > 0)
                  & (workspace > 0)).astype('uint8')
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), 'uint8'))
    _, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    proposals, rejected = [], []
    for label, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if area < 300 or w > 150 or h > 160:
            continue
        try:
            mask = labels == label
            ring = ((cv2.dilate(mask.astype('uint8'), np.ones((101, 101), 'uint8')) > 0)
                    & ~(cv2.dilate(mask.astype('uint8'), np.ones((21, 21), 'uint8')) > 0)
                    & red)
            table, normal, fraction_table, rms = _plane(deproject(depth, ring, intr, scale), cfg)
            mask = cv2.erode(mask.astype('uint8'), np.ones((3, 3), 'uint8')) > 0
            points = deproject(depth, mask, intr, scale)
            result = section(points, table, normal, fraction, cfg)
            result.update(table_fraction=fraction_table, table_rms_m=rms,
                          table_point_camera_m=table.tolist(),
                          bbox_xywh=[int(x), int(y), int(w), int(h)])
            proposals.append((result, mask, points))
        except ValueError as exc:
            rejected.append(dict(component=int(label), reason=str(exc)))
    return proposals, rejected


def depth_proposals(depth, image, meta, cfg, fraction):
    """Preserve the original single-candidate contract for initial selection."""
    proposals, rejected = depth_candidates(depth, image, meta, cfg, fraction)
    if len(proposals) != 1:
        raise ValueError(f'Expected exactly one geometric proposal, got {len(proposals)}; {rejected}')
    return proposals[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--orientation-reference', type=Path, required=True)
    parser.add_argument('--grasp-config', type=Path, default=Path(__file__).with_name('CURRENT_GRASP.json'))
    parser.add_argument('--allow-provisional', action='store_true')
    parser.add_argument('--plane-tolerance-mm', type=float, default=3.)
    parser.add_argument('--finger-photo-fraction', type=float, required=True)
    args = parser.parse_args()
    if not 0 < args.plane_tolerance_mm <= 5:
        parser.error('Plane tolerance must be in (0, 5] mm')
    args.output.mkdir(parents=True, exist_ok=False)
    profile = json.loads(args.grasp_config.read_text())
    if profile['definition_id'] != 'silver_side_index_height_9_of_11' or profile['preform'] != [0] * 6:
        raise ValueError('This diagnostic supports only the selected open index contact')
    definition = profile['height_definition']
    fraction = definition['contact_fraction_numerator'] / definition['contact_fraction_denominator']
    cfg = Config(plane_tolerance_m=args.plane_tolerance_mm / 1000)
    meta, depth, image, frames = load_batch(args.dataset)
    calibration = load_calibration(args.calibration, expected_camera_serial=meta['serial'],
                                   allow_provisional=args.allow_provisional)
    intr = meta['intrinsics']
    live_k = np.array([[intr['fx'], 0, intr['cx']],
                       [0, intr['fy'], intr['cy']], [0, 0, 1.]])
    calibrated = calibration['camera']
    if ((intr['width'], intr['height']) != (calibrated['width'], calibrated['height'])
            or not np.allclose(live_k, calibrated['camera_matrix'], atol=1e-4, rtol=0)):
        raise ValueError('Calibration intrinsics do not match the recorded RGB-D mode')
    cam = np.array(calibration['T_base_camera'])
    geom, mask, points = depth_proposals(depth, image, meta, cfg, fraction)
    hand, arm = RightRevo2Model(), load_model()
    tcp, palm_normal, tcp_info = model_tcp(hand, args.finger_photo_fraction)
    reference = json.loads(args.orientation_reference.read_text())
    ref_q = np.array(reference['joints_rad'])
    rotation = np.array(arm.fk(ref_q))[:3, :3]
    normal = cam[:3, :3] @ geom['normal_camera']
    center = cam[:3, :3] @ geom['center_camera_m'] + cam[:3, 3]
    support = center - geom['contact_height_m'] * normal
    outward = -(rotation @ palm_normal)
    outward -= (outward @ normal) * normal
    outward /= np.linalg.norm(outward)
    contact = center + geom['radius_m'] * outward
    pregrasp = contact + profile['pregrasp_surface_gap_m'] * outward
    stages = []
    seed = np.clip(ref_q, *np.array(arm.limits_rad).T)
    for name, point in [('clearance', pregrasp + .14 * normal),
                        ('pregrasp', pregrasp), ('contact', contact)]:
        flange = np.eye(4)
        flange[:3, :3] = rotation
        flange[:3, 3] = point - rotation @ tcp[:3, 3]
        ik = arm.ik(flange, seed, max_iterations=600, position_tolerance_m=1e-5,
                    orientation_tolerance_rad=1e-4)
        screen = check_hand_pose(flange, support_center_m=support, axis=normal,
                                 cup_diameter_m=2 * geom['radius_m'],
                                 cup_height_m=geom['height_m'], model=hand)
        stages.append(dict(name=name, tcp_target_base_m=point.tolist(),
                           T_base_flange_target=flange.tolist(), ik=asdict(ik),
                           joints_deg=np.degrees(ik.joints_rad).tolist(),
                           hand_table_min_m=screen['exact_collision_mesh_table_min_m'],
                           hand_table_screen_passed=screen['table_margin_passed'],
                           hand_cup_sphere_screen_passed=screen['cup_margin_passed']))
        if ik.success:
            seed = ik.joints_rad
    # Offline path sampling retains a separate start state; it never promotes
    # a diagnostic to an executable plan. The full cup shape is not certified.
    start = json.loads((args.dataset / 'joints_after.jsonl').read_text().splitlines()[-1])
    previous = start['joints_rad']
    paths = []
    model_links = hand.link_transforms_from_flange(np.eye(4))
    centers, radii, clouds = [], [], []
    for name, (c, radius, vertices, mesh_origin, _) in hand.collision.items():
        t = model_links[name] @ mesh_origin
        centers.append(t[:3, :3] @ c + t[:3, 3])
        radii.append(radius)
        clouds.append(vertices @ t[:3, :3].T + t[:3, 3])
    cloud, centers, radii = np.vstack(clouds), np.array(centers), np.array(radii)
    # Maximum observed radial extent (including possible depth outliers), used
    # only as a conservative cylinder screen; hidden geometry remains unknown.
    camera_delta = points - np.array(geom['center_camera_m'])
    camera_normal = np.array(geom['normal_camera'])
    radial = camera_delta - (camera_delta @ camera_normal)[:, None] * camera_normal
    envelope_radius = float(np.linalg.norm(radial, axis=1).max())
    for stage in stages:
        if not stage['ik']['success']:
            break
        q = stage['ik']['joints_rad']
        checked = arm.check_joint_path(previous, q)
        minimum_table, minimum_cup_bound = float('inf'), float('inf')
        for sample in checked.samples_rad:
            fk = np.array(arm.fk(sample))
            minimum_table = min(minimum_table, float(np.min(cloud @ (fk[:3, :3].T @ normal))
                                                     + (fk[:3, 3] - support) @ normal))
            delta = centers @ fk[:3, :3].T + fk[:3, 3] - support
            heights = delta @ normal
            dr = np.linalg.norm(delta - heights[:, None] * normal, axis=1) - envelope_radius
            dh = np.maximum(-heights, heights - geom['height_m'])
            signed = np.hypot(np.maximum(dr, 0), np.maximum(dh, 0)) + np.minimum(np.maximum(dr, dh), 0)
            minimum_cup_bound = min(minimum_cup_bound, float(np.min(signed - radii)))
        paths.append(dict(stage=stage['name'], samples=len(checked.samples_rad),
                          joint_limits_passed=checked.joint_limits_passed,
                          model_table_min_m=minimum_table,
                          conservative_cup_sphere_bound_m=minimum_cup_bound,
                          controller_interpolation_verified=False,
                          collision_verified=False))
        previous = q
    report = dict(schema=1, kind='offline_index_contact_preview', motion_ready=False,
                  motion_sent=False, identity_verified=False,
                  segmentation='depth foreground component in red workspace; not YOLO detection',
                  definition_id=profile['definition_id'], contact_height_fraction=fraction,
                  table_tolerance_m=cfg.plane_tolerance_m,
                  calibration_quality_passed=calibration['quality_passed'],
                  geometry=geom, tcp_model=tcp_info, cup_support_base_m=support.tolist(),
                  cup_normal_base=normal.tolist(), outward_base=outward.tolist(), stages=stages,
                  orientation_source=str(args.orientation_reference),
                  orientation_reference_use='historical orientation/solver seed only, not a new contact sample',
                  sampled_joint_paths=paths, cup_envelope_radius_m=envelope_radius,
                  trajectory_checked=False, self_collision_checked=False,
                  limitations=['Finger surface fraction is estimated from user photos, not measured',
                               'URDF open hand and actual mounting require physical verification',
                               'Geometric component is not semantically verified as a cup',
                               'Circle diameter is only at contact height, not a full cup collision model',
                               'Endpoint IK and hand screens do not validate a motion path'],
                  inputs_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [args.grasp_config, args.calibration,
                                           args.orientation_reference, *frames,
                                           *[p.with_suffix('.npz') for p in frames],
                                           frames[0].with_suffix('.png')]})
    (args.output / 'preview.json').write_text(json.dumps(report, indent=2) + '\n')
    overlay = image.copy()
    overlay[mask] = (overlay[mask] * .6 + np.array([0, 180, 0]) * .4).astype('uint8')
    intr = meta['intrinsics']
    def project(point):
        return tuple(np.rint([intr['fx'] * point[0] / point[2] + intr['cx'],
                              intr['fy'] * point[1] / point[2] + intr['cy']]).astype(int))
    basis = np.array(geom['basis_camera'])
    c = np.array(geom['center_camera_m'])
    ring = np.array([project(c + geom['radius_m'] * basis @ [np.cos(t), np.sin(t)])
                     for t in np.linspace(0, 2 * np.pi, 80)])
    cv2.polylines(overlay, [ring], True, (0, 220, 255), 1)
    invcam = inverse(cam)
    for label, point, color in [('contact', contact, (0, 0, 255)),
                                 ('pregrasp50', pregrasp, (255, 230, 0))]:
        pixel = project((invcam @ np.r_[point, 1])[:3])
        cv2.circle(overlay, pixel, 4, color, -1)
        cv2.putText(overlay, label, (pixel[0] + 6, pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
    cv2.putText(overlay, '9/11H  MODEL PREVIEW ONLY', (15, 465), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
    cv2.imwrite(str(args.output / 'preview.png'), overlay)
    np.savez_compressed(args.output / 'geometry_inputs.npz', depth=depth, mask=mask)
    print(json.dumps(dict(output=str(args.output), height_mm=geom['height_m'] * 1000,
                          contact_height_mm=geom['contact_height_m'] * 1000,
                          diameter_mm=geom['radius_m'] * 2000,
                          tcp_flange_mm=(tcp[:3, 3] * 1000).tolist(),
                          stages=[{k: s[k] for k in ('name', 'ik', 'hand_table_min_m')} for s in stages]), indent=2))


if __name__ == '__main__':
    main()
