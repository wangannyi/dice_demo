"""Offline YOLO timing on one saved image. Never opens camera or CAN."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/green_cup.json')
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--provider', choices=('cpu','spacemit'), default='cpu')
    parser.add_argument('--ai-cpus', default='8,9')
    parser.add_argument('--cpus', help='Linux CPU IDs, e.g. 8,9. Omit to keep affinity.')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.cpus:
        cpus = {int(x) for x in args.cpus.split(',')}
        if not cpus or not cpus <= os.sched_getaffinity(0):
            parser.error('Requested CPUs are not available')
        os.sched_setaffinity(0, cpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    sys.path.insert(0, str(ROOT))
    import cv2
    import numpy as np
    from vision.inference.detector import configured_session, cap_outputs
    from cup_grasp_demo.flow.cup_perception import decode
    from vision.inference.yolo_seg import preprocess
    opts = json.loads(args.config.read_text())['green_cup']['perception']
    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError('Image could not be read')
    tensor, transform = preprocess(image)
    model_path = ROOT/opts['model']
    begin = time.perf_counter()
    opts.update(inference_threads=args.threads, inference_provider=args.provider,
                inference_cpu_ids=[int(x) for x in args.ai_cpus.split(',')] if args.provider == 'spacemit' else [])
    model = configured_session(opts)
    load_s = time.perf_counter()-begin
    feed = {model.get_inputs()[0].name: tensor}
    begin = time.perf_counter()
    model.run(None, feed)
    first_s = time.perf_counter()-begin
    durations = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        outputs = model.run(None, feed)
        durations.append(time.perf_counter()-begin)
    items = decode(cap_outputs(outputs), image.shape, transform, opts)
    masks = np.stack([x['mask'] for x in items]) if items else np.zeros((0,*image.shape[:2]), bool)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix('.npz'), masks=masks)
    report = dict(image=str(args.image), image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(),
                  threads=args.threads, cpus=sorted(os.sched_getaffinity(0)), providers=model.get_providers(),
                  load_s=load_s, first_run_s=first_s, inference_s=durations, median_s=statistics.median(durations),
                  detections=[{k:v for k,v in item.items() if k!='mask'} for item in items],
                  task_affinities={p.name: sorted(os.sched_getaffinity(int(p.name))) for p in Path('/proc/self/task').iterdir()})
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('threads','cpus','providers','load_s','first_run_s','median_s')}), flush=True)

if __name__ == '__main__':
    main()
