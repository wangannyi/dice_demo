"""Select the robot's table surface; colour never assigns game participants."""
import copy

import cv2
import numpy as np


PRESETS = {
    'red': dict(hue_ranges=[[0, 14], [166, 179]], min_saturation=70,
                max_saturation=255, min_value=50, max_value=255,
                min_region_fraction=0., region_shape='contour'),
    'green': dict(hue_ranges=[[25, 55]], min_saturation=30,
                  max_saturation=255, min_value=40, max_value=255,
                  min_region_fraction=.05, region_shape='contour'),
    'gray': dict(hue_ranges=[[0, 179]], min_saturation=-1,
                 max_saturation=60, min_value=35, max_value=190,
                 min_region_fraction=.05, region_shape='contour'),
}


def workspace_options(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError('workspace must be an object')
    color = raw.get('color', 'red')
    if not isinstance(color, str) or color not in PRESETS:
        raise ValueError('workspace.color must be red, green or gray')
    if set(raw) - (set(PRESETS[color]) | {'color'}):
        raise ValueError('Unknown workspace parameter')
    result = dict(copy.deepcopy(PRESETS[color]), color=color, **{k: v for k, v in raw.items() if k != 'color'})
    for key in ('min_saturation', 'max_saturation', 'min_value', 'max_value'):
        value = result[key]
        low, high = (-1, 254) if key.startswith('min_') else (0, 255)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('workspace.' + key + ' outside integer range')
    if result['min_saturation'] >= result['max_saturation'] or result['min_value'] >= result['max_value']:
        raise ValueError('workspace minimum must be below maximum')
    ranges = result['hue_ranges']
    if (not isinstance(ranges, list) or not ranges
            or any(not isinstance(pair, (list, tuple)) or len(pair) != 2
                   or any(type(v) is not int for v in pair)
                   or not 0 <= pair[0] <= pair[1] <= 179 for pair in ranges)):
        raise ValueError('workspace.hue_ranges must contain integer [low, high] pairs in 0..179')
    fraction = result['min_region_fraction']
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError('workspace.min_region_fraction must be in 0..1')
    if result['region_shape'] not in ('contour', 'convex_hull'):
        raise ValueError('workspace.region_shape must be contour or convex_hull')
    return result


def perception_workspace(opts):
    """One current setting, with the old red-only key accepted for saved setups."""
    if 'workspace' in opts and 'red_workspace' in opts:
        raise ValueError('Use workspace only; remove legacy red_workspace')
    if 'workspace' in opts:
        return workspace_options(opts['workspace'])
    raw = opts.get('red_workspace')
    if raw is not None:
        if not isinstance(raw, dict) or set(raw) - {'min_saturation', 'min_value'}:
            raise ValueError('Unknown red_workspace parameter')
        for key, value in raw.items():
            if type(value) is not int or not 0 <= value < 255:
                raise ValueError('red_workspace.' + key + ' must be an integer in 0..254')
    return workspace_options(raw)


def workspace_masks(image, raw=None):
    options = workspace_options(raw)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = np.zeros(image.shape[:2], bool)
    for low, high in options['hue_ranges']:
        hue |= (hsv[:, :, 0] >= low) & (hsv[:, :, 0] <= high)
    surface = (hue & (hsv[:, :, 1] > options['min_saturation'])
               & (hsv[:, :, 1] <= options['max_saturation'])
               & (hsv[:, :, 2] > options['min_value'])
               & (hsv[:, :, 2] <= options['max_value']))
    contours, _ = cv2.findContours(surface.astype('uint8'), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    workspace = np.zeros(surface.shape, np.uint8)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(workspace, [contour], -1, 1, cv2.FILLED)
    if options['min_region_fraction'] > 0:
        # A green cup or a small gray machine part is not a tabletop. Preserve
        # holes in the depth mask, but fill them when selecting a cup location.
        surface &= workspace > 0
        if np.count_nonzero(surface) < surface.size * options['min_region_fraction']:
            surface[:] = False
            workspace[:] = 0
    if contours and workspace.any() and options['region_shape'] == 'convex_hull':
        # A reflection can connect the cup's hole to a table edge. Complete the
        # supported table envelope for cup selection, never for depth fitting.
        cv2.fillConvexPoly(workspace, cv2.convexHull(contour), 1)
    return surface, workspace
