"""Execute original main.py with an isolated, opt-in Metrics adapter.

Run from the ORIGINAL checkout. Everything after '--' goes to its main.py.
An unmodified source hash manifest is recorded for reproducibility.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys
import time
import numpy as np
import torch

from aabb_soft import AABBMetrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--weight', type=float, default=None,
                        help='legacy alias: sets strength and band together')
    parser.add_argument('--strength', type=float, default=.1)
    parser.add_argument('--band', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('args', nargs=argparse.REMAINDER)
    opts = parser.parse_args()
    if opts.weight is not None:
        opts.strength = opts.band = opts.weight
    if not 0 <= opts.strength <= .25 or not 0 <= opts.band <= .5:
        parser.error('strength must be within [0,.25], band within [0,.5]')
    repo = Path(opts.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(repo/'third_party/FCGF'))
    random.seed(opts.seed); np.random.seed(opts.seed); torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.backends.cudnn.benchmark = False
    import Scene_rep as source
    original = source.Scene_rep
    scenes = []

    class ExperimentScene(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metrics = AABBMetrics(self.metrics, self, opts.strength, opts.band)
            self.aabb_update_times = []
            scenes.append(self)

        def update_masks(self, frame_id):
            self.metrics.frame = int(frame_id)
            torch.cuda.synchronize()
            start = time.perf_counter()
            super().update_masks(frame_id)
            torch.cuda.synchronize()
            self.aabb_update_times.append(dict(frame=int(frame_id), seconds=time.perf_counter()-start))

    source.Scene_rep = ExperimentScene
    raw_args = opts.args[1:] if opts.args[:1] == ['--'] else opts.args
    sys.argv = [str(repo/'main.py')] + raw_args
    text = (repo/'main.py').read_text()
    if opts.limit:
        assert text.count('max_frame_num = None') == 1
        text = text.replace('max_frame_num = None', 'max_frame_num = %d' % opts.limit)
    start = time.perf_counter()
    status = 'failed'
    try:
        exec(compile(text, str(repo/'main.py'), 'exec'),
             {'__name__': '__main__', '__file__': str(repo/'main.py')})
        status = 'complete'
    finally:
        for scene in scenes:
            files = ['main.py','Scene_rep.py','Metrics.py','MaskGraph.py','Instance.py',
                     'PC_extractor.py','voxelized_points.py','Dataset/dataset.py',
                     'tool/post_process.py']
            manifest = dict(status=status, seconds=time.perf_counter()-start,
                            weight=opts.weight, strength=opts.strength, band=opts.band,
                            seed=opts.seed, limit=opts.limit,
                            repo=str(repo), argv=raw_args,
                            original_hashes={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},
                            experiment_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                            config=scene.cfg, updates=scene.aabb_update_times,
                            evidence=scene.metrics.records,
                            note='Original TSDF, candidates, FCGF, MaskGraph and output filtering; soft AABB only for borderline overall similarity, no carving or independent tracks.')
            dest = Path(scene.seq_output_dir)/'aabb_manifest.json'
            dest.write_text(json.dumps(manifest, indent=2))
            print('AABB_RUN', status, 'seconds', manifest['seconds'], 'manifest', dest, flush=True)


if __name__ == '__main__':
    main()
