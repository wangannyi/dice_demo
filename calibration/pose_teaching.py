"""Export accepted joint waypoints; never infer missing historical joints."""
import json
from pathlib import Path
import numpy as np


def export_poses(dataset):
    dataset=Path(dataset)
    manifest=json.loads((dataset/'manifest.json').read_text())
    poses=[];missing=[]
    for path in sorted(dataset.glob('sample_[0-9][0-9][0-9][0-9].json')):
        sample=json.loads(path.read_text())
        if 'joints_rad' not in sample:
            missing.append(path.name)
            continue
        q=np.asarray(sample['joints_rad'],dtype=float)
        if q.shape!=(7,) or not np.isfinite(q).all():
            raise ValueError('Invalid taught joints: '+path.name)
        poses.append(dict(sample=path.name,joints_rad=q.tolist(),joints_deg=np.degrees(q).tolist(),
                          T_base_flange=sample['T_base_flange'],time_unix_s=sample['time_unix_s']))
    report=dict(schema=1,kind='calibration_joint_waypoints',poses=poses,
                missing_joint_samples=missing,camera=manifest['camera'],board=manifest['board'],
                execution_validated=False,records_manual_travel_path=False)
    temp=dataset/'teaching_poses.json.tmp'
    temp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temp.replace(dataset/'teaching_poses.json')
    return report
