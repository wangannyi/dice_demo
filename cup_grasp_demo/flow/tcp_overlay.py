"""Project a feedback-derived, fixed open-hand TCP into a calibrated RGB image."""

import cv2
import numpy as np

from nero_calibration.core import inverse, matrix
from nero_revo2_control.kinematics import load_model


def project(point, session):
    """Use the registered, zero-distortion color intrinsics; never clamp a point."""
    intr = session['intrinsics']
    if np.any(np.asarray(intr.get('dist_coeffs', [])) != 0):
        raise ValueError('TCP overlay requires zero-distortion color intrinsics')
    p = (inverse(matrix(session['T_base_camera'])) @ np.r_[point, 1])[:3]
    if not np.isfinite(p).all():
        raise ValueError('Non-finite projected point')
    result = dict(base_m=np.asarray(point).tolist(), camera_m=p.tolist(), pixel=None)
    if p[2] <= 0:
        return dict(result, visibility='behind_camera')
    uv = np.array([intr['fx'] * p[0] / p[2] + intr['cx'],
                   intr['fy'] * p[1] / p[2] + intr['cy']])
    visible = 0 <= uv[0] < intr['width'] and 0 <= uv[1] < intr['height']
    return dict(result, pixel=uv.tolist(), visibility='in_frame' if visible else 'outside_image')


def draw_tcp(image, session, snapshot, target=None, hand_state='unknown'):
    """Return image and audit data. This does not detect a physical finger point."""
    intr = session['intrinsics']
    if image is None or image.shape[:2] != (intr['height'], intr['width']):
        raise ValueError('RGB image dimensions differ from calibration')
    q = np.asarray(snapshot['joints_rad'], dtype=float)
    if q.shape != (7,) or not np.isfinite(q).all():
        raise ValueError('TCP overlay requires seven finite feedback joints')
    transform = np.asarray(load_model().fk(q)) @ matrix(session['T_flange_tcp'])
    tcp = project(transform[:3, 3], session)
    report = dict(tcp=tcp, joints_rad=q.tolist(), T_base_tcp=transform.tolist(),
                  observed_epoch_s=snapshot.get('observed_epoch_s'),
                  source='joint_feedback_fk', hand_state=hand_state,
                  fixed_open_hand_tcp=True, physical_tcp_measured=False,
                  occlusion_checked=False)
    result = image.copy()
    if target is not None:
        report['target'] = project(target, session)
        if report['target']['visibility'] == 'in_frame':
            uv = tuple(np.rint(report['target']['pixel']).astype(int))
            cv2.drawMarker(result, uv, (0, 0, 0), cv2.MARKER_CROSS, 24, 4)
            cv2.drawMarker(result, uv, (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
    if tcp['visibility'] == 'in_frame':
        uv = tuple(np.rint(tcp['pixel']).astype(int))
        cv2.circle(result, uv, 9, (255, 255, 255), 4)
        cv2.circle(result, uv, 9, (255, 0, 255), 2)
        cv2.drawMarker(result, uv, (255, 0, 255), cv2.MARKER_CROSS, 10, 1)
    label = 'TCP reference (after close command)' if hand_state == 'after_close_command' else 'TCP now (open-hand model)'
    lines = [(f'MAGENTA: {label}; {tcp["visibility"]}', (255, 0, 255)),
             ('Joint feedback + calibration; physical point NOT measured', (255, 255, 255)),
             ('Valid for open fingers; occlusion NOT checked', (255, 255, 255))]
    if target is not None:
        lines.append((f'RED: frozen target; {report["target"]["visibility"]}', (0, 0, 255)))
    # At HOME the TCP can be near the top edge. Keep the legend away from it.
    legend_y = image.shape[0] - 30 - 18 * (len(lines) - 1) if (
        tcp['visibility'] == 'in_frame' and tcp['pixel'][1] < 110) else 18
    for row, (text, color) in enumerate(lines):
        origin = (8, legend_y + 18 * row)
        cv2.putText(result, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 3)
        cv2.putText(result, text, origin, cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1)
    return result, report
