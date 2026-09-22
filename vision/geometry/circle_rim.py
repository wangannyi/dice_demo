"""Fit one upright circular rim jointly in the two calibrated IR images.

RGB selects the object and supplies initial guesses only. Metric scale comes from
stereo extrinsics, not the manually measured cup dimensions. No motion commands.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from vision.geometry.table_plane import Config, _plane, deproject
from cup_grasp_demo.flow.green_image_rim import image_contour


QUALITY = dict(max_mean_edge_error_px=.8, max_view_edge_error_px=1.,
               edge_distance_px=2., min_edge_support=.9, ambiguity_score_margin_px=.15,
               max_competing_center_distance_mm=10., max_competing_diameter_difference_mm=6.,
               max_center_spread_mm=3., max_height_spread_mm=5., max_diameter_spread_mm=5.)


class RimEdgeQualityError(ValueError):
    """A fresh image can recover edge noise; never substitutes a stale target."""
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"Stereo rim edge support insufficient: {report['mean_edge_error_px']:.3f} px, "
            f"{report['edge_support']}, height={report['height_mm']:.1f}, "
            f"diameter={report['diameter_mm']:.1f}; failed={report['failed_checks']}")


def check_edge_quality(errors, quality, height_m, radius_m):
    score = float(errors.mean())
    views = errors.mean(axis=1)
    support = (errors < quality['edge_distance_px']).mean(axis=1)
    failed = []
    if score > quality['max_mean_edge_error_px']:
        failed.append('max_mean_edge_error_px')
    if views.max() > quality['max_view_edge_error_px']:
        failed.append('max_view_edge_error_px')
    if support.min() < quality['min_edge_support']:
        failed.append('min_edge_support')
    report = dict(mean_edge_error_px=score, view_mean_edge_error_px=views.tolist(),
                  edge_support=support.tolist(), height_mm=height_m*1000,
                  diameter_mm=radius_m*2000, failed_checks=failed,
                  thresholds={k: quality[k] for k in ('max_mean_edge_error_px',
                      'max_view_edge_error_px', 'edge_distance_px', 'min_edge_support')})
    if failed:
        raise RimEdgeQualityError(report)
    return report


def quality_options(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw)-set(QUALITY):
        raise ValueError('Unknown stereo_rim quality parameter')
    result = dict(QUALITY, **raw)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError('Invalid stereo_rim.'+key)
    if result['min_edge_support'] > 1:
        raise ValueError('min_edge_support must be <= 1')
    return result


def height_options(opts):
    mode = opts.get('height_mode', 'measured')
    if mode not in ('fixed', 'measured'):
        raise ValueError('green_cup.perception.height_mode must be fixed or measured')
    height = opts.get('fixed_height_mm', 65)
    if (isinstance(height, bool) or not isinstance(height, (float, int))
            or not np.isfinite(height) or height <= 0):
        raise ValueError('fixed_height_mm must be positive')
    if mode == 'fixed':
        if opts.get('geometry_method') != 'stereo_rim':
            raise ValueError('fixed height requires stereo_rim geometry')
        if not opts['height_range_mm'][0] <= height <= opts['height_range_mm'][1]:
            raise ValueError('fixed_height_mm outside height_range_mm')
    return mode, height / 1000


def table_from_base_scene(scene, T_base_camera):
    """Express the calibration-stage table plane in this camera's coordinates."""
    point = np.asarray(scene['cup_support_base_m'], dtype=float)
    normal = np.asarray(scene['cup_normal_base'], dtype=float)
    transform = np.asarray(T_base_camera, dtype=float)
    if (point.shape != (3,) or normal.shape != (3,) or transform.shape != (4, 4)
            or not np.isfinite(point).all() or not np.isfinite(normal).all()
            or not np.isfinite(transform).all()
            or abs(np.linalg.norm(normal)-1) > .01
            or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-3)):
        raise ValueError('Invalid calibrated table plane or camera transform')
    rotation = transform[:3, :3]
    camera_point = rotation.T @ (point-transform[:3, 3])
    camera_normal = rotation.T @ (normal/np.linalg.norm(normal))
    if camera_point[2] <= 0 or camera_normal @ camera_point >= 0:
        raise ValueError('Calibrated table plane does not face the camera')
    return dict(point=camera_point, normal=camera_normal,
                fit=scene.get('table_fit', {}), source='calibrated')


def resolve_table(depth, red, meta, plane_tolerance_mm, fixed_table=None):
    if fixed_table is None:
        table, normal, fraction, rms = _plane(
            deproject(depth, red, meta['intrinsics'], meta['depth_scale_m']),
            Config(plane_tolerance_m=plane_tolerance_mm/1000))
        return table, normal, fraction, rms, 'live_depth'
    table = np.asarray(fixed_table['point'], dtype=float)
    normal = np.asarray(fixed_table['normal'], dtype=float)
    fit = fixed_table.get('fit', {})
    rms_mm = fit.get('rms_mm')
    return (table, normal, fit.get('inlier_fraction'),
            None if rms_mm is None else rms_mm/1000, 'calibrated')


def make_view(image, record):
    k = record['intrinsics']
    if any(abs(float(x)) > 1e-8 for x in k['dist_coeffs']):
        raise ValueError('Stereo rim requires rectified zero-distortion IR images')
    if (not np.isfinite([k['fx'], k['fy'], k['cx'], k['cy']]).all()
            or min(k['fx'], k['fy']) <= 0):
        raise ValueError('Invalid stereo intrinsics')
    if image is None or image.shape != (k['height'], k['width']):
        raise ValueError('Stereo image shape mismatch')
    rotation = np.asarray(record['to_color_rotation'], float).reshape(3, 3, order='F')
    translation = np.asarray(record['to_color_translation'], float)
    if translation.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError('Invalid stereo extrinsics')
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)):
        raise ValueError('Non-orthogonal stereo extrinsics')
    edges = cv2.Canny(image, 25, 60)
    distance = cv2.distanceTransform(255-edges, cv2.DIST_L2, 5)
    return dict(k=k, R=rotation, t=translation, image=image, distance=distance,
)


def project(points, view):
    p = (points-view['t']) @ view['R']
    if np.any(p[:, 2] <= 0):
        raise ValueError('Stereo points behind camera')
    k = view['k']
    return np.c_[p[:, 0]/p[:, 2]*k['fx']+k['cx'],
                 p[:, 1]/p[:, 2]*k['fy']+k['cy']]


def fixed_rim_residual(views, origin, basis, unit):
    """Exact Jacobian of the bilinear distance-map residual at fixed height."""
    derivatives = np.stack((np.broadcast_to(basis[:, 0], unit.shape),
                            np.broadcast_to(basis[:, 1], unit.shape), unit), axis=-1)
    # The circle basis and stereo extrinsics are constant for every optimizer
    # iteration and seed; rotate their derivatives once per residual factory.
    count = len(unit)
    dc = np.concatenate([np.einsum('ij,nip->njp', v['R'], derivatives) for v in views])
    base = np.repeat(np.array([(origin-v['t']) @ v['R'] for v in views]), count, axis=0)
    fx, fy, cx, cy = [np.repeat([v['k'][key] for v in views], count)
                       for key in ('fx', 'fy', 'cx', 'cy')]
    heights = np.repeat([v['distance'].shape[0] for v in views], count)
    widths = np.repeat([v['distance'].shape[1] for v in views], count)
    sizes = [v['distance'].size for v in views]
    offsets = np.repeat(np.cumsum([0] + sizes[:-1]), count)
    pixels = np.concatenate([v['distance'].ravel() for v in views])
    cached = [None, None, None]

    def evaluate(q):
        if cached[0] is not None and np.array_equal(q, cached[0]):
            return cached[1], cached[2]
        camera = base + dc @ q
        z = camera[:, 2]
        if np.any(z <= 0):
            raise ValueError('Stereo points behind camera')
        xn, yn = camera[:, 0]/z, camera[:, 1]/z
        x, y = xn*fx+cx, yn*fy+cy
        inside = (x >= 0) & (x <= widths-1) & (y >= 0) & (y <= heights-1)
        ix = np.minimum(np.maximum(np.floor(x).astype(int), 0), widths-2)
        iy = np.minimum(np.maximum(np.floor(y).astype(int), 0), heights-2)
        u, v = x-ix, y-iy
        index = offsets + iy*widths + ix
        a, b = pixels[index], pixels[index+1]
        c, d = pixels[index+widths], pixels[index+widths+1]
        value = (1-v)*((1-u)*a+u*b)+v*((1-u)*c+u*d)
        gx, gy = (1-v)*(b-a)+v*(d-c), (1-u)*(c-a)+u*(d-b)
        dx = (fx/z)[:, None]*(dc[:, 0, :]-xn[:, None]*dc[:, 2, :])
        dy = (fy/z)[:, None]*(dc[:, 1, :]-yn[:, None]*dc[:, 2, :])
        jac = gx[:, None]*dx+gy[:, None]*dy
        cached[:] = [np.array(q, copy=True), np.where(inside, value, 50.),
                     np.where(inside[:, None], jac, 0.)]
        return cached[1], cached[2]

    return lambda q: evaluate(q)[0], lambda q: evaluate(q)[1]


def fit_circle(views, table, normal, seeds, radius_range, height_range, quality=None, *, fixed_height=None, analytic_jacobian=False):
    quality = quality_options(quality)
    if len(views) != 2 or np.linalg.norm(views[0]['t']-views[1]['t']) < 0.01:
        raise ValueError('Two independent calibrated IR cameras are required')
    normal = np.asarray(normal, float)
    normal = normal / np.linalg.norm(normal)
    u = np.cross(normal, np.eye(3)[np.argmin(abs(normal))]); u /= np.linalg.norm(u)
    basis = np.column_stack((u, np.cross(normal, u)))
    angles = np.linspace(0, 2*np.pi, 120, endpoint=False)
    unit = np.c_[np.cos(angles), np.sin(angles)] @ basis.T

    def residual(p):
        pts = p[:3]+p[3]*unit
        values = []
        for view in views:
            uv = project(pts, view)
            values.extend(map_coordinates(view['distance'], [uv[:, 1], uv[:, 0]],
                                          order=1, mode='constant', cval=50))
        return np.asarray(values, dtype=float)

    fits = []
    for seed in seeds:
        seed = np.asarray(seed, float)
        bounds = (np.r_[seed[:3]-.04, radius_range[0]], np.r_[seed[:3]+.04, radius_range[1]])
        if fixed_height is None:
            result = least_squares(residual, seed, bounds=bounds,
                loss='soft_l1', f_scale=1, max_nfev=120, diff_step=1e-4)
            p = result.x
        else:
            if not np.isfinite(fixed_height) or not height_range[0] <= fixed_height <= height_range[1]:
                raise ValueError('Fixed height outside configured cup range')
            origin = np.asarray(table) + fixed_height * normal
            start = np.r_[(seed[:3]-origin) @ basis, seed[3]]
            def expand(q):
                return np.r_[origin + basis @ q[:2], q[2]]
            fun = lambda q: residual(expand(q))
            jac = '2-point'
            if analytic_jacobian:
                fun, jac = fixed_rim_residual(views, origin, basis, unit)
            result = least_squares(fun, start, jac=jac,
                bounds=(np.r_[start[:2]-.04, radius_range[0]], np.r_[start[:2]+.04, radius_range[1]]),
                loss='soft_l1', f_scale=1, max_nfev=80, diff_step=1e-4)
            p = expand(result.x)
        h = float((p[:3]-table) @ normal)
        if not height_range[0] <= h <= height_range[1]:
            continue
        errors = residual(p).reshape(2, -1)
        fits.append((float(errors.mean()), p, errors, h))
    if not fits:
        raise ValueError('Stereo rim has no in-range solution')
    fits.sort(key=lambda row: row[0]); score, p, errors, h = fits[0]
    # Reject competing circles before consulting the manual dimensions.
    plausible = [row for row in fits if row[0] < score+quality['ambiguity_score_margin_px']]
    for other in plausible[1:]:
        if (np.linalg.norm(other[1][:3]-p[:3])*1000 > quality['max_competing_center_distance_mm']
                or abs(other[1][3]-p[3])*2000 > quality['max_competing_diameter_difference_mm']):
            raise ValueError(f'Stereo rim has ambiguous competing circles: best={score:.3f}px/{h*1000:.1f}mm, alternative={other[0]:.3f}px/{other[3]*1000:.1f}mm')
    edge_report = check_edge_quality(errors, quality, h, p[3])
    support = np.asarray(edge_report['edge_support'])
    pts = p[:3]+p[3]*unit
    for view in views:
        uv = project(pts, view)
        if ((uv[:, 0] < 2) | (uv[:, 1] < 2) |
            (uv[:, 0] >= view['k']['width']-2) |
            (uv[:, 1] >= view['k']['height']-2)).any():
            raise ValueError('Stereo rim clipped at image boundary')
    return dict(center=p[:3].tolist(), radius_m=float(p[3]), height_m=h,
                mean_edge_error_px=score, edge_support=support.tolist(),
                edge_quality=edge_report,
                basis=basis.tolist(), outline=pts.tolist(),
                competing_height_range_mm=[min(row[3] for row in plausible)*1000,
                                           max(row[3] for row in plausible)*1000])


def refine_sequence(views_by_frame, initial, table, normal, radius_range, height_range, quality):
    """Share search starts, never observations or fitted output across frames."""
    seeds = [np.r_[fit['center'], fit['radius_m']] for fit in initial]
    return [fit_circle(views, table, normal, seeds, radius_range, height_range, quality)
            for views in views_by_frame]


def fit_frame_sequence(views_by_frame, table, normal, seeds, radius_range, height_range,
                       quality, diagnostics):
    fitted = []
    diagnostics.update(frames=[], initial_frame_errors=[])
    for index, views in enumerate(views_by_frame):
        use_seeds = seeds if not fitted else [np.r_[fitted[0]['center'], fitted[0]['radius_m']]]
        try:
            fit = fit_circle(views, table, normal, use_seeds, radius_range, height_range, quality)
        except ValueError as exc:
            diagnostics['initial_frame_errors'].append(dict(frame=index, error=str(exc)))
            diagnostics['frames'].append(None)
            continue
        fitted.append(fit)
        diagnostics['frames'].append(fit)
    diagnostics['initial_frames'] = list(diagnostics['frames'])
    if not fitted:
        raise ValueError('No frame supplied a valid stereo rim search start; see initial_frame_errors')
    diagnostics['refinement_method'] = 'shared_initial_guesses_independent_frame_fit'
    refined = refine_sequence(views_by_frame, fitted, table, normal, radius_range, height_range, quality)
    diagnostics['frames'] = refined
    return refined


def validate_sequence(fitted, quality, diagnostics):
    centers = np.asarray([f['center'] for f in fitted]); center = np.median(centers, axis=0)
    heights = np.asarray([f['height_m'] for f in fitted]); radii = np.asarray([f['radius_m'] for f in fitted])
    spread = float(np.linalg.norm(centers-center, axis=1).max()*1000)
    diagnostics.update(center_spread_mm=spread, height_range_mm=(heights*1000).tolist(),
                       diameter_range_mm=(radii*2000).tolist())
    if (spread > quality['max_center_spread_mm'] or np.ptp(heights)*1000 > quality['max_height_spread_mm']
            or np.ptp(radii)*2000 > quality['max_diameter_spread_mm']):
        raise ValueError(
            f'Stereo rim is unstable across frames: center spread {spread:.2f} mm '
            f'(limit {quality["max_center_spread_mm"]:.2f}); '
            f'height span {np.ptp(heights)*1000:.2f} mm '
            f'(limit {quality["max_height_spread_mm"]:.2f}); '
            f'diameter span {np.ptp(radii)*2000:.2f} mm '
            f'(limit {quality["max_diameter_spread_mm"]:.2f})')
    return center, heights, radii, spread


def fit_fixed_sequence(views_by_frame, table, normal, seeds, radius_range,
                       height_range, quality, fixed_height, minimum, diagnostics):
    """Require a configured quorum of independently qualified observations."""
    if type(minimum) is not int or not 3 <= minimum <= 5:
        raise ValueError('fixed_min_valid_frames must be an integer from 3 to 5')
    fitted, indices = [], []
    diagnostics.update(frames=fitted, accepted_frame_indices=indices, rejected_frames=[],
                       minimum_valid_frames=minimum, attempted_frames=len(views_by_frame))
    for index, views in enumerate(views_by_frame):
        try:
            fit = fit_circle(views, table, normal, seeds, radius_range, height_range,
                             quality, fixed_height=fixed_height)
        except ValueError as exc:
            diagnostics['rejected_frames'].append(dict(frame_index=index, reason=str(exc)))
            continue
        fitted.append(fit)
        indices.append(index)
    if len(fitted) < minimum:
        raise ValueError(f'Stereo rim qualified frames {len(fitted)}/{len(views_by_frame)}, '
                         f'require {minimum}; see rejected_frames')
    validate_sequence(fitted, quality, diagnostics)
    return fitted, indices[0]


def detect_stereo(run, depth, image, meta, opts, plane_tolerance_mm, instances, diagnostics,
                  *, fixed_table=None):
    quality = quality_options(opts.get('stereo_rim'))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) &
           (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 50))
    table, normal, fraction, rms, table_source = resolve_table(
        depth, red, meta, plane_tolerance_mm, fixed_table)
    diagnostics['table_plane_source'] = table_source
    contours, _ = cv2.findContours(red.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    workspace = np.zeros(red.shape, np.uint8)
    cv2.drawContours(workspace, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    eligible = [item for item in instances if np.count_nonzero(item['mask']) >= opts['min_area_px']
                and np.count_nonzero(np.asarray(item['mask'], bool) & (workspace > 0)) /
                np.count_nonzero(item['mask']) >= .9]
    if len(eligible) != 1:
        raise ValueError('Stereo rim requires one YOLO cup in the red workspace')
    mask = np.asarray(eligible[0]['mask'], bool)
    k = meta['intrinsics']
    seed_ellipses = []
    for source, quantile in [('opening', .3), ('green_outline', .5)]:
        configured = dict(opts, image_rim=dict(opts.get('image_rim', {}),
                          contour_source=source, opening_value_quantile=quantile))
        seed_diagnostic = {}
        try:
            ellipse, _, contour = image_contour(image, mask, configured, seed_diagnostic)
            seed_ellipses.append(ellipse)
        except ValueError:
            # A partial opening can initialize a search, but cannot validate it.
            e = seed_diagnostic.get('image_ellipse')
            if e:
                seed_ellipses.append((e['center_uv'], e['diameters_px'], e['angle_deg']))
    if not seed_ellipses:
        raise ValueError('No complete image contour for stereo initialization')
    opening_seeds = seed_ellipses[:1]
    for ellipse in opening_seeds:
        for offset in (-4., 4.):
            seed_ellipses.append((np.asarray(ellipse[0])+offset, ellipse[1], ellipse[2]))
    mode, fixed_height = height_options(opts)
    height_range = np.asarray(opts['height_range_mm'])/1000
    radius_range = np.asarray(opts['diameter_range_mm'])/2000
    seeds = []
    for ellipse in seed_ellipses:
        ray = np.array([(ellipse[0][0]-k['cx'])/k['fx'], (ellipse[0][1]-k['cy'])/k['fy'], 1])
        for h in ([fixed_height] if mode == "fixed" else np.linspace(height_range[0], height_range[1], 5)[:-1]):
            center = ray*((table @ normal+h)/(ray @ normal))
            guess = np.mean(ellipse[1])/2*center[2]/np.mean([k['fx'], k['fy']])
            # RGB may outline the body rather than the mouth. Fixed height
            # removes height search, but must retain independent radius starts.
            for factor in (.8, 1, 1.2):
                radius = np.clip(guess*factor, radius_range[0]+.001, radius_range[1]-.001)
                seeds.append(np.r_[center, radius])
    frames = sorted((Path(run)/'rgbd').glob('frame_*.json'))
    frame_count = opts.get('frame_count', 5)
    if type(frame_count) is not int or frame_count not in (1, 5) or (frame_count == 1 and mode != 'fixed'):
        raise ValueError('Single-frame stereo requires fixed height; frame_count must be 1 or 5')
    if len(frames) < frame_count:
        raise ValueError(f'Stereo rim needs {frame_count} synchronized frame pairs')
    diagnostics.update(valid=False, geometry_method='stereo_rim', frames=[], initial_frame_errors=[])
    views_by_frame = []
    first_records = None
    pair_ids = set()
    for index, file in enumerate(frames[:frame_count]):
        current = json.loads(file.read_text())
        records = current.get('stereo')
        if not records or records['left_frame_number'] != records['right_frame_number']:
            raise ValueError('Missing or unsynchronized stereo pair')
        pair_id = records['left_frame_number']
        if pair_id in pair_ids:
            raise ValueError('Repeated stereo frame pair')
        pair_ids.add(pair_id)
        if abs(records['left_timestamp_ms']-records['right_timestamp_ms']) > 1:
            raise ValueError('IR pair timestamp mismatch')
        if current['serial'] != meta['serial'] or current['intrinsics'] != k:
            raise ValueError('Stereo batch camera changed')
        with np.load(file.with_suffix('.npz')) as data:
            views = [make_view(data['ir_'+side], records[side]) for side in ('left', 'right')]
        if index and any(records[side] != first_records[side] for side in ('left', 'right')):
            raise ValueError('Stereo calibration changed within batch')
        if index == 0:
            first_records = records
        views_by_frame.append(views)
    if frame_count == 1:
        fitted = [fit_circle(views_by_frame[0], table, normal, seeds, radius_range,
                             height_range, quality, fixed_height=fixed_height,
                             analytic_jacobian=opts.get('analytic_jacobian', False))]
        overlay_index = 0
        diagnostics.update(frames=fitted, accepted_frame_indices=[0], refinement_method='single_frame_known_height')
    elif mode == 'fixed':
        fitted, overlay_index = fit_fixed_sequence(
            views_by_frame, table, normal, seeds, radius_range, height_range, quality,
            fixed_height, opts.get('fixed_min_valid_frames', 5), diagnostics)
        diagnostics.update(frames=fitted, refinement_method='known_height_independent_frame_fit')
    else:
        overlay_index = 0
        fitted = fit_frame_sequence(views_by_frame, table, normal, seeds,
                                    radius_range, height_range, quality, diagnostics)
    diagnostics.update(height_mode=mode, height_source='configured' if mode == 'fixed' else 'stereo_measured')
    for side, view in (zip(('left', 'right'), views_by_frame[overlay_index]) if opts.get('save_debug_images', True) else ()):
        canvas = cv2.cvtColor(view['image'], cv2.COLOR_GRAY2BGR)
        uv = project(np.asarray(fitted[0]['outline']), view)
        cv2.polylines(canvas, [np.rint(uv).astype('int32')], True, (0, 255, 0), 1)
        cv2.imwrite(str(Path(run)/f'stereo_{side}.png'), canvas)
    center, heights, radii, spread = validate_sequence(fitted, quality, diagnostics)
    diagnostics['temporal_consistency_checked'] = frame_count > 1
    if frame_count == 1:
        spread = None
        diagnostics['center_spread_mm'] = None
    height = float(np.median(heights)); radius = float(np.median(radii))
    if 'reference_dimensions_mm' in opts:
        measured = {'height': height*1000, 'diameter': radius*2000}
        if any(abs(measured[key]-opts['reference_dimensions_mm'][key]) >
               opts['reference_tolerance_mm'][key] for key in measured):
            raise ValueError(f'Stereo dimensions disagree with independent reference: {measured}')
    theta = np.linspace(0, 2*np.pi, 180, endpoint=False)
    points = center+radius*(np.c_[np.cos(theta), np.sin(theta)] @ np.asarray(fitted[0]['basis']).T)
    rgb_view = dict(k=k, R=np.eye(3), t=np.zeros(3))
    uv = project(points, rgb_view)
    center_uv = np.rint(project(center[None, :], rgb_view)[0]).astype(int)
    if (not 0 <= center_uv[0] < mask.shape[1] or not 0 <= center_uv[1] < mask.shape[0]
            or not mask[center_uv[1], center_uv[0]]):
        raise ValueError('Stereo circle center falls outside the selected YOLO cup')
    diagnostics.update(valid=True, height_mm=height*1000, diameter_mm=radius*2000,
                       rim_center_camera_m=center.tolist())
    contour = np.rint(uv).astype('int32')[:, None, :]
    ellipse = cv2.fitEllipse(uv.astype('float32'))
    diagnostics['candidates'] = [dict(valid=True, image_ellipse=dict(
        center_uv=list(ellipse[0]), diameters_px=list(ellipse[1]), angle_deg=ellipse[2]))]
    geometry = dict(height_m=height, radius_m=radius, rim_center_camera_m=center.tolist(),
                    table_point_camera_m=table.tolist(), table_normal_camera=normal.tolist(),
                    table_fit={'inlier_fraction':fraction,
                               'rms_mm':None if rms is None else rms*1000,
                               'source':table_source},
                    rim_basis_camera=fitted[0]['basis'], geometry_method='stereo_rim', height_mode=mode,
                    height_source='configured' if mode == 'fixed' else 'stereo_measured',
                    section_height_fraction=1.0, upright_rim_parallel_table_assumption=True,
                    center_spread_mm=spread, hardware_grasp_verified=False)
    return geometry, mask.astype('uint8')*255, contour
