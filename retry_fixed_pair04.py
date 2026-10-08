"""Same protocol retry on GPU 3 after the GPU 7 run stopped advancing."""
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

ROOT=Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
REPO=Path(__file__).resolve().parent
old=next(r for r in json.loads((ROOT/'results.json').read_text()) if r['seq'].startswith('pair_04'))
row=dict(seq=old['seq'],gpu=3,status='running',started=time.time(),retry_reason='Original run stopped advancing at frame 541; no OOM reported. Same config, GPU 7 -> 3.')
cmd=list(old['command']);cmd[cmd.index('--output_dir')+1]=str(ROOT/'retry_pair04')
row['command']=cmd
def save():
    p=ROOT/'retry_result.tmp';p.write_text(json.dumps(row,indent=2));p.replace(ROOT/'retry_result.json')
save()
try:
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(REPO),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    with (ROOT/'retry_pair04.log').open('w') as log:
        rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    row['returncode']=rc
    if rc:raise RuntimeError('Retry failed, see log')
    meta=json.loads((ROOT/'retry_pair04'/row['seq']/'run_metadata.json').read_text())
    base=json.loads((Path('/root/autodl-tmp/output/ig_correction_20261008/off')/row['seq']/'run_metadata.json').read_text())
    for k in ('selected_frame_ids','mapping_frame_interval','segmentation_frame_interval','tsdf_voxel_size_mm','seed'):
        assert meta[k]==base[k],k
    row.update(status='complete',main_loop_seconds=meta['main_loop_seconds'],
        fixed=meta['fixed_candidate_summary'],original=meta['readonly_audit']['summary'],
        mapping_interval=meta['mapping_frame_interval'],voxel_mm=meta['tsdf_voxel_size_mm'],
        selected_frames=len(meta['selected_frame_ids']),schedule_check='pass')
except Exception:row.update(status='failed',error=traceback.format_exc())
finally:
    row['wall_seconds']=time.time()-row['started'];save()
if row['status']=='complete':
    original=json.loads((ROOT/'results.json').read_text())
    selected=[r for r in original if r['status']=='complete' and r['seq']!=row['seq']]+[row]
    assert len(selected)==4
    (ROOT/'selected_results.json').write_text(json.dumps(selected,indent=2))
    (ROOT/'selected_done.json').write_text(json.dumps(dict(completed=4,total=4,retained_stalled_attempts=1)))
print(json.dumps({k:row.get(k) for k in ('seq','status','error','wall_seconds')}))
