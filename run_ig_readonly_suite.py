"""Four existing IG pairs, detection only; no deletion and no AP claims."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parent
ROOT = Path('/root/autodl-tmp/output/ig_readonly_20261008/full')
PY = '/root/miniconda3/envs/OASeg/bin/python'
PREVIOUS = Path('/root/autodl-tmp/output/ig_correction_20261008')
refs = json.loads((PREVIOUS/'sources.json').read_text())
refs = [x for x in refs if x['seq'].startswith('pair_')]
assert len(refs) == 4
ROOT.mkdir(parents=True, exist_ok=True)
if (ROOT/'results.json').exists(): raise FileExistsError('Do not overwrite an existing diagnostic run')
rows = []
lock = threading.Lock()


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def persist():
    with lock: write(ROOT/'results.json', rows)


def job(ref, gpu):
    seq = ref['seq']
    row = dict(seq=seq, gpu=gpu, status='running', started=time.time(), actual_deletion_enabled=False)
    with lock: rows.append(row); write(ROOT/'results.json', rows)
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(REPO),
            OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', PYTHONUNBUFFERED='1')
        cmd = [PY, 'run_ig_readonly.py', '--config',ref['config'],'--dataset',ref['frames'],
            '--instance_dir',ref['masks'],'--output_dir',str(ROOT),'--seq_name',seq,
            '--merge_selection','disjoint_pairs','--detection','on']
        row['command'] = cmd; persist()
        with (ROOT/(seq+'.log')).open('w') as f:
            rc = subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=rc, wall_seconds=time.time()-row['started'])
        log = (ROOT/(seq+'.log')).read_text(errors='replace')
        row['warnings'] = {s:log.count(s) for s in ['CUDA out of memory','Associated bucket and excess list full',
            'Size out of bounds','device-side assert','raycast failed','Could not generate full range map']}
        if rc: raise RuntimeError('Read-only run failed; retained log, no automatic parameter changes')
        for warning in ['CUDA out of memory','Associated bucket and excess list full','Size out of bounds','device-side assert']:
            if row['warnings'][warning]: raise RuntimeError('Reconstruction warning: '+warning)
        meta = json.loads((ROOT/seq/'run_metadata.json').read_text())
        old = json.loads((PREVIOUS/'off'/seq/'run_metadata.json').read_text())
        for k in ['selected_frame_ids','mapping_frame_interval','segmentation_frame_interval',
                  'tsdf_voxel_size_mm','integrated_frames','segmented_frames','final_merge','seed']:
            assert meta[k] == old[k], ('IG input/schedule mismatch', k)
        summary = json.loads((ROOT/seq/'remove_detection/summary.json').read_text())
        assert summary['completed'] and not summary['actual_instance_deletions'] and not summary['actual_tsdf_deletions']
        row.update(status='complete', metadata=meta, detection=summary, input_schedule_check='pass')
    except Exception:
        row.update(status='failed',error=traceback.format_exc())
    finally:
        persist()
        print(json.dumps({k:row.get(k) for k in ['seq','status','wall_seconds','error']}),flush=True)


def queue(gpu, assigned):
    for ref in assigned: job(ref, gpu)


with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(queue,gpu,refs[i::2]) for i,gpu in enumerate([4,7])]
    for future in futures: future.result()
write(ROOT/'done.json',dict(total=len(rows),complete=sum(x['status']=='complete' for x in rows),finished=time.time()))
