"""Four paired read-only detector runs. No deletion, no AP parameter changes."""
import json
import os
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
PY = '/root/miniconda3/envs/OASeg/bin/python'
refs = [r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text()) if r['seq'].startswith('pair_')]
assert len(refs) == 4
rows = []
lock = threading.Lock()
assert not (ROOT / 'results.json').exists()


def persist():
    with lock:
        p = ROOT / 'results.tmp'
        p.write_text(json.dumps(rows, indent=2))
        p.replace(ROOT / 'results.json')


def run(ref, gpu):
    row = dict(seq=ref['seq'], gpu=gpu, status='running', started=time.time())
    rows.append(row)
    dest = ROOT / 'paired'
    dest.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    cmd = [PY, 'run_ig_fixed_candidate.py', '--config', ref['config'], '--dataset', ref['frames'],
           '--instance_dir', ref['masks'], '--output_dir', str(dest), '--seq_name', ref['seq'],
           '--merge_selection', 'disjoint_pairs', '--deletion', 'off']
    row['command'] = cmd
    persist()
    try:
        with (ROOT / (ref['seq'] + '.log')).open('w') as log:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=2400)
        row['returncode'] = rc
        if rc:
            raise RuntimeError('Paired mapping failed; see unchanged log')
        meta = json.loads((dest / ref['seq'] / 'run_metadata.json').read_text())
        old = json.loads((Path('/root/autodl-tmp/output/ig_correction_20261008/off') / ref['seq'] / 'run_metadata.json').read_text())
        for k in ('selected_frame_ids', 'mapping_frame_interval', 'segmentation_frame_interval', 'tsdf_voxel_size_mm', 'seed'):
            assert meta[k] == old[k], k
        row.update(status='complete', main_loop_seconds=meta['main_loop_seconds'],
                   fixed=meta['fixed_candidate_summary'], original=meta['readonly_audit']['summary'],
                   mapping_interval=meta['mapping_frame_interval'], voxel_mm=meta['tsdf_voxel_size_mm'],
                   schedule_check='pass', selected_frames=len(meta['selected_frame_ids']))
    except Exception:
        row.update(status='failed', error=traceback.format_exc())
    row['wall_seconds'] = time.time() - row['started']
    persist()
    print(json.dumps({k: row.get(k) for k in ('seq', 'status', 'wall_seconds', 'error')}), flush=True)


def queue(gpu, assigned):
    for ref in assigned:
        run(ref, gpu)


with ThreadPoolExecutor(max_workers=2) as pool:
    tasks = [pool.submit(queue, gpu, refs[i::2]) for i, gpu in enumerate([3, 7])]
    for task in tasks:
        task.result()
(ROOT / 'done.json').write_text(json.dumps(dict(completed=sum(r['status'] == 'complete' for r in rows), total=len(rows))))
