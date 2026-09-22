"""Bind a freshly checked table capture to the selected pipeline calibration."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cup_grasp_demo.flow.planar_scene import verify
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,required=True)
p.add_argument('--table-scene',type=Path,required=True)
a=p.parse_args()
record,cfg=verify(a.table_scene,a.config)
out=ROOT/cfg['green_cup']['home_table_scene']
value=dict(scene=record['scene'],calibration_sha256=hashlib.sha256(Path(cfg['calibration']).read_bytes()).hexdigest(),source=str(a.table_scene.resolve()),requires_fixed_base_and_table=True)
out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():
    backup=out.with_suffix('.json.bak')
    backup.write_bytes(out.read_bytes())
tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(out)
print(out)
