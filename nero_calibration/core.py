"""Rigid TCP transforms and eye-to-hand solve. No hardware imports."""
import json
import math
from pathlib import Path
import numpy as np


def matrix(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(t[:3, :3]) - 1) > 1e-6):
        raise ValueError('Expected a finite rigid 4x4 transform')
    return t


def pose_matrix(pose):
    p = np.asarray(pose, dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError('pose must contain six finite values (m, rad)')
    r, t, y = p[3:]
    cr, sr, ct, st, cy, sy = math.cos(r), math.sin(r), math.cos(t), math.sin(t), math.cos(y), math.sin(y)
    out = np.eye(4)
    out[:3, :3] = [[cy*ct, cy*st*sr-sy*cr, cy*st*cr+sy*sr],
                   [sy*ct, sy*st*sr+cy*cr, sy*st*cr-cy*sr], [-st, ct*sr, ct*cr]]
    out[:3, 3] = p[:3]
    return out


def matrix_pose(t):
    t = matrix(t)
    r = t[:3, :3]
    pitch = math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0]))
    if abs(math.cos(pitch)) < 1e-8:
        roll, yaw = math.atan2(-r[1, 2], r[1, 1]), 0.
    else:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    return t[:3, 3].tolist() + [roll, pitch, yaw]


def inverse(t):
    t = matrix(t)
    out = np.eye(4)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def distance(a, b):
    a, b = matrix(a), matrix(b)
    angle = math.acos(float(np.clip((np.trace(a[:3, :3].T @ b[:3, :3])-1)/2, -1, 1)))
    return float(np.linalg.norm(a[:3, 3]-b[:3, 3])), math.degrees(angle)


HAND_BASE = pose_matrix([.031, 0, -.0235, -1.5708, 0, -1.5708]) @ pose_matrix([0, 0, .012, 0, 0, 3.1415926])
# Model-derived reference point; not a factory-defined or calibrated palm center.
HAND_PALM = pose_matrix([.007931305737821843, 0, .041068125, 0, 0, 0])
HAND_PALM[:3, :3] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]
PALM = HAND_BASE @ HAND_PALM


def tcp_transform(name='flange', custom=None):
    if custom is not None and name != 'custom':
        raise ValueError('--tcp-config requires --tcp custom')
    if name == 'flange':
        return np.eye(4)
    if name == 'palm':
        return PALM.copy()
    if name != 'custom' or custom is None:
        raise ValueError('custom TCP requires --tcp-config')
    cfg = json.loads(Path(custom).read_text())
    parents = {'flange': np.eye(4), 'hand_base': HAND_BASE, 'palm': PALM}
    if cfg.get('relative_to') not in parents:
        raise ValueError('relative_to must be flange, hand_base or palm')
    has_pose = 'pose' in cfg
    has_matrix = 'matrix' in cfg
    if has_pose == has_matrix:
        raise ValueError('TCP config must contain exactly one of pose or matrix')
    if has_pose:
        if cfg.get('units') != 'm_rad':
            raise ValueError('TCP pose config units must be m_rad')
        offset = pose_matrix(cfg['pose'])
    else:
        if cfg.get('units') != 'm':
            raise ValueError('TCP matrix translation units must be m')
        offset = matrix(cfg['matrix'])
    return matrix(parents[cfg['relative_to']] @ offset)


def average(ts):
    u, _, vt = np.linalg.svd(sum(t[:3, :3] for t in ts))
    out = np.eye(4)
    out[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    out[:3, 3] = np.mean([t[:3, 3] for t in ts], axis=0)
    return out


def diversity(ts):
    import cv2
    vectors = [cv2.Rodrigues(ts[0][:3, :3].T @ t[:3, :3])[0].ravel() for t in ts[1:]]
    s = np.linalg.svd(np.array(vectors), compute_uv=False)
    if len(s) < 2 or s[1] < .1:
        raise ValueError('Insufficient rotation diversity: use rotations about at least two axes')
    return s.tolist()


def fit(bt, cm):
    import cv2
    diversity(bt)
    tb = [inverse(t) for t in bt]
    # Same eye_to_hand input inversion as AgileX handeye_calibration_ros.
    r, t = cv2.calibrateHandEye([a[:3, :3] for a in tb], [a[:3, 3] for a in tb],
                               [a[:3, :3] for a in cm], [a[:3, 3] for a in cm],
                               method=cv2.CALIB_HAND_EYE_PARK)
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = r, np.asarray(t).ravel()
    return matrix(out)


def solve(samples, tcp, max_position_m=.005, max_angle_deg=2.):
    if len(samples) < 12:
        raise ValueError('Need at least 12 samples, including held-out validation poses')
    if not (0 < max_position_m < 1 and 0 < max_angle_deg < 180):
        raise ValueError('Invalid residual thresholds')
    tcp = matrix(tcp)
    bt = [matrix(s['T_base_flange']) @ tcp for s in samples]
    cm = [matrix(s['T_camera_board']) for s in samples]
    hold = [i for i in range(len(samples)) if i % 4 == 3]
    train = [i for i in range(len(samples)) if i not in hold]
    bc_train = fit([bt[i] for i in train], [cm[i] for i in train])
    tm_train = average([inverse(bt[i]) @ bc_train @ cm[i] for i in train])
    errors = [distance(bc_train @ cm[i], bt[i] @ tm_train) for i in hold]
    passed = all(p <= max_position_m and a <= max_angle_deg for p, a in errors)
    bc = fit(bt, cm)
    tm = average([inverse(a) @ bc @ b for a, b in zip(bt, cm)])
    residuals = [distance(bc @ b, a @ tm) for a, b in zip(bt, cm)]
    passed = passed and all(p <= max_position_m and a <= max_angle_deg for p, a in residuals)
    return {'schema': 1, 'mode': 'eye_to_hand', 'quality_passed': bool(passed),
            'direction': 'T_base_camera maps camera optical coordinates into arm base',
            'T_base_camera': bc.tolist(), 'T_tcp_board': tm.tolist(),
            'T_flange_tcp': tcp.tolist(), 'pose_base_camera_m_rad': matrix_pose(bc),
            'rotation_diversity_singular_values': diversity(bt),
            'holdout_indices': hold, 'holdout_errors_m_deg': errors,
            'all_sample_errors_m_deg': residuals,
            'thresholds': {'position_m': max_position_m, 'angle_deg': max_angle_deg}}
