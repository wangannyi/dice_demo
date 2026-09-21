"""Offline Revo2 static palm thickness. No hardware or motion interfaces."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np


def load_stl(path):
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + 50*count:
        raise ValueError('Expected binary STL')
    records = np.frombuffer(data, dtype=[('normal', '<f4', 3), ('v', '<f4', (3, 3)),
                                       ('attribute', '<u2')], count=count, offset=84)
    return records['v'].astype(float)


def intersect_x(triangles, y, z):
    """Intersect infinite X line with triangles, returning deduplicated X coordinates."""
    a = triangles[:, 0]
    u = triangles[:, 1]-a
    v = triangles[:, 2]-a
    det = u[:, 1]*v[:, 2]-u[:, 2]*v[:, 1]
    good = abs(det) > 1e-15
    s = np.zeros(len(a))
    t = s.copy()
    s[good] = ((y-a[good, 1])*v[good, 2]-(z-a[good, 2])*v[good, 1])/det[good]
    t[good] = (u[good, 1]*(z-a[good, 2])-u[good, 2]*(y-a[good, 1]))/det[good]
    hit = good & (s >= -1e-8) & (t >= -1e-8) & (s+t <= 1+1e-8)
    return np.unique(np.round(a[hit, 0]+s[hit]*u[hit, 0]+t[hit]*v[hit, 0], 10))


def section(triangles, y, z):
    hits = intersect_x(triangles, y, z)
    if len(hits) < 2:
        raise ValueError('Section lacks two outer surfaces')
    return {'y_m': float(y), 'z_m': float(z), 'back_x_m': float(hits.min()),
            'palm_x_m': float(hits.max()), 'thickness_m': float(hits.max()-hits.min())}


def derive(model_root):
    urdf = model_root/'urdf/revo2_right_hand.urdf'
    tree = ET.parse(urdf).getroot()
    link = tree.find("link[@name='right_base_link']")
    # Verify the meshes really share hand-base coordinates with no scaling.
    for visual in link.findall('visual'):
        origin = visual.find('origin')
        if any(float(x) != 0 for key in ('xyz', 'rpy') for x in origin.get(key).split()):
            raise ValueError('Nonidentity visual transform requires explicit handling')
        scale = visual.find('geometry/mesh').get('scale', '1 1 1')
        if any(float(x) != 1 for x in scale.split()):
            raise ValueError('Nonunit STL scale')
    heights = [float(tree.find(f"joint[@name='right_{n}_proximal_joint']/origin").get('xyz').split()[2])
               for n in ('index', 'middle', 'ring', 'pinky')]
    paths = [model_root/'meshes/revo2_right_hand'/name for name in
             ('right_base_link.STL', 'right_base_visual_link.STL')]
    mesh = np.concatenate([load_stl(path) for path in paths])
    z = float(np.mean(heights)/2)
    center = section(mesh, 0, z)
    patch = [section(mesh, y, zz) for y in np.linspace(-.005, .005, 5)
             for zz in np.linspace(z-.005, z+.005, 5)]
    return {'schema_version': 1, 'status': 'model_estimate_not_attachment_calibration',
            'marker_id': 40, 'marker_size_m': .030, 'marker_size_user_confirmed': True,
            'hand_frame': 'right_base_link',
            'definition': 'y=0, z=half mean four-finger root height; span along X between outer static visual surfaces',
            'palm_side': '+X', 'back_side': '-X', 'reference_section': center,
            'nominal_back_to_palm_m': center['thickness_m'],
            'central_10mm_patch_thickness_range_m': [min(x['thickness_m'] for x in patch),
                                                     max(x['thickness_m'] for x in patch)],
            'centerline_sections': [section(mesh, 0, zz/1000) for zz in (25, 35, 45, 55, 65, 75)],
            'T_marker_contact': None, 'motion_target_valid': False,
            'limitations': ['Thickness varies across shell; exact marker model location not measured',
                           'Tape/paper thickness, mounting tilt and model-to-hardware mismatch not measured',
                           'A scalar thickness does not determine lateral offset or full rigid transform',
                           'Visual template center is not verified physical contact center'],
            'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [urdf]+paths}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = derive(args.model_root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result))
