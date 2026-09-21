"""Create a relocatable source bundle; no robot access, no virtualenv or logs."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TREES=('cup_grasp_demo','dice_cup_localization','nero_revo2_control','nero_calibration','rgb_hand_tracking','configs','scripts','docs')
SKIP={'build','CMakeFiles','.pytest_cache','.ruff_cache','datasets','output','runtime','diagnostics','__pycache__','.git','.deps','.venv'}

def build(destination):
    destination=destination.resolve()
    if destination.exists():
        raise ValueError('Output must not exist: '+str(destination))
    destination.mkdir(parents=True)
    stage=destination/'dice_demo';stage.mkdir()
    selected=set()
    for tree in TREES:
        for path in (ROOT/tree).rglob('*'):
            rel=path.relative_to(ROOT)
            if any(part in SKIP for part in rel.parts) or path.is_symlink() or not path.is_file():continue
            if path.suffix == '.md' and path.name not in ('README.md', 'README_DEBUG.md') and tree != 'docs':
                continue  # Development journals reference local-only captures.
            if path.suffix in ('.py','.sh','.json','.md','.onnx','.urdf','.xacro','.txt','.yaml') or path.name == 'LICENSE':
                selected.add(rel)
    for name in ('README.md','run.sh','run_feedback.sh','.gitignore','requirements-vision.txt','requirements-sdk.txt','THIRD_PARTY.md'):
        selected.add(Path(name))
    for name in ('README.md','build.sh','source_commit.json','source.sha256'):
        selected.add(Path('kernel_usbcan_20260921')/name)
    mesh=Path('agx_arm_ros/src/agx_arm_description/agx_arm_urdf')
    for path in (ROOT/mesh).rglob('*'):
        rel=path.relative_to(ROOT)
        if path.is_file() and not path.is_symlink() and '.git' not in rel.parts:
            if path.suffix.lower() in ('.stl','.dae','.obj','.urdf','.xacro','.png','.jpg','.yaml') or path.name=='LICENSE':selected.add(rel)
    # Only active installation artifacts, never whole sampling/log datasets.
    cfg=json.loads((ROOT/'configs/green_cup.json').read_text())
    for key in ('home','calibration','orientation_reference','tcp_candidate','grasp_config'):
        path=Path(cfg[key]);selected.add(path.relative_to(ROOT) if path.is_absolute() else path)
    for path in sorted(selected):
        source=(ROOT/path).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():raise ValueError('Missing source: '+str(path))
        target=stage/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    # Keep the active calibration outside ignored datasets for Git distribution.
    calibration=stage/'configs/installation/camera.json'
    calibration.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/cfg['calibration'],calibration)
    # Every delivered installation starts with an explicit calibration gate.
    for relative in ('configs/green_cup.json','cup_grasp_demo/calibration_debug/green_open_cup/stereo_config.json'):
        p=stage/relative;d=json.loads(p.read_text());d['calibration']='configs/installation/camera.json';d['green_cup']['installation_requires_calibration']=True;p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
    # Normalize whitespace in generated vendor XML only; geometry is unchanged.
    for path in stage.rglob('*'):
        if path.suffix in ('.urdf', '.xacro'):
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()).rstrip()+'\n')
    manifest={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
    (stage/'MANIFEST.sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=destination/'dice_demo-source.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:tar.add(stage,arcname='dice_demo')
    (destination/'SHA256SUMS').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
    print(json.dumps(dict(directory=str(stage),archive=str(archive),files=len(manifest),bytes=archive.stat().st_size)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.output)
