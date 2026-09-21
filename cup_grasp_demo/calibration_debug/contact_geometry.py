"""Select a cup-side contact direction and turn the reference hand toward it."""

import math

import numpy as np


def contact_azimuth(constraint):
    """Validate optional XY azimuth; existing base-X mode stays at zero."""
    if constraint is None:
        return None
    if (not isinstance(constraint, dict)
            or constraint.get('mode') not in ('base_x_table_section', 'base_xy_angle_table_section')
            or constraint.get('side') not in ('positive', 'negative')):
        raise ValueError('Invalid contact_direction mode or positive/negative side')
    if constraint['mode'] == 'base_x_table_section':
        if 'azimuth_deg' in constraint:
            raise ValueError('contact_direction.azimuth_deg requires base_xy_angle_table_section')
        return 0.
    angle = constraint.get('azimuth_deg')
    if (isinstance(angle, bool) or not isinstance(angle, (float, int))
            or not math.isfinite(angle) or not -180 <= angle <= 180):
        raise ValueError('contact_direction.azimuth_deg must be finite and within [-180, 180]')
    return float(angle)


def contact_direction(rotation, palm_normal, table_normal, constraint=None):
    rotation = np.array(rotation, dtype=float, copy=True)
    normal = np.asarray(table_normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < .9:
        raise ValueError('Invalid table normal')
    normal = normal / np.linalg.norm(normal)
    outward = -(rotation @ np.asarray(palm_normal, dtype=float))
    outward -= (outward @ normal) * normal
    if not np.isfinite(outward).all() or np.linalg.norm(outward) < .1:
        raise ValueError('Invalid approach direction')
    outward /= np.linalg.norm(outward)
    if constraint is None:
        return outward, rotation, None
    azimuth = contact_azimuth(constraint)
    if normal[2] < .9:
        raise ValueError('Table too tilted for a base-XY side approach')
    # Choose the diameter angle in base XY, then solve Z to stay in the
    # measured table-parallel cross-section. Rotation about the normal alone
    # does not give an exact requested XY azimuth on a tilted table.
    dx, dy = math.cos(math.radians(azimuth)), math.sin(math.radians(azimuth))
    direction = np.array([dx, dy, -(normal[0] * dx + normal[1] * dy) / normal[2]])
    direction /= np.linalg.norm(direction)
    if constraint['side'] == 'negative':
        direction = -direction
    angle = math.atan2(float(normal @ np.cross(outward, direction)), float(outward @ direction))
    nx, ny, nz = normal
    skew = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
    turn = np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)
    metadata = dict(mode=constraint['mode'], frame='robot_base', side=constraint['side'],
                    parallel_in_base_xy=True, diameter_unit_base=direction.tolist(),
                    base_x_3d_deviation_deg=math.degrees(math.acos(float(abs(direction[0])))),
                    reference_yaw_adjustment_deg=math.degrees(angle))
    if constraint['mode'] == 'base_xy_angle_table_section':
        metadata.update(parallel_in_base_xy=abs(math.sin(math.radians(azimuth))) < 1e-12,
                        diameter_azimuth_deg=azimuth,
                        outward_azimuth_deg=math.degrees(math.atan2(direction[1], direction[0])))
    return direction, turn @ rotation, metadata
