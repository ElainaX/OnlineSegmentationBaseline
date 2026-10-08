"""One fixed-config hybrid-retirement run, then original and full-prediction AP."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--gpu',required=True)
p.add_argument('--mode',choices=['hybrid','local_control'],default='hybrid');a=p.parse_args()
REPO=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/output/ig_whole_instance_20261008')
if a.mode=='local_control': ROOT=ROOT/'local_control_run'
OLD=Path('/root/autodl-tmp/output/ig_surface_deletion_20261008')
SEQ='pair_01_0cac767c_19f1a892'
ROOT.mkdir(exist_ok=True)
if (ROOT/'result.json').exists(): raise FileExistsError('Preserve previous result')
ref=next(r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text()) if r['seq']==SEQ)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
result=dict(status='running',seq=SEQ,gpu=a.gpu,started=time.time())
def save(): (ROOT/'result.json').write_text(json.dumps(result,indent=2))
save()
try:
    (ROOT/'gpu_before.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader'],text=True))
    cmd=[sys.executable,'run_ig_surface_deletion.py','--config',ref['config'],'--dataset',ref['frames'],
         '--instance_dir',ref['masks'],'--output_dir',str(ROOT/'hybrid'),'--seq_name',SEQ,
         '--merge_selection','disjoint_pairs','--detection','on','--deletion','on',
         '--whole-instance','on' if a.mode=='hybrid' else 'off','--verify-erasure','--detector-schedule-json',
         '/root/autodl-tmp/output/ig_rescan_dense_20261008/schedules.json']
    result['mapping_command']=cmd;save()
    with (ROOT/'mapping.log').open('w') as f:
        subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1200)
    out=ROOT/'hybrid'/SEQ;meta=json.loads((out/'run_metadata.json').read_text())
    old=json.loads((OLD/'on'/SEQ/'run_metadata.json').read_text())
    keys=['selected_frame_ids','mapping_frame_interval','segmentation_frame_interval','tsdf_voxel_size_mm',
          'integrated_frames','segmented_frames','final_merge','seed','detector_frame_ids','baseline_config_sha256']
    assert all(meta[k]==old[k] for k in keys),'Experiment schedule mismatch'
    bank=np.load(out/'surface_detection/bank.npz');oldbank=np.load(OLD/'on'/SEQ/'surface_detection/bank.npz')
    assert all(np.array_equal(bank[k],oldbank[k]) for k in bank.files),'Detector changed'
    result.update(status='evaluation',schedule_parity=True,detector_arrays_identical=True,
                  main_loop_seconds=meta['main_loop_seconds'],torch_peak_allocated_bytes=meta['torch_peak_allocated_bytes'],
                  local_deletion=meta['deletion_summary'],whole_retirement=meta['instance_retirement_summary'])
    save()
    cmd=[sys.executable,'-m','eval.evaluate_seqs','--result_dir',str(ROOT/'hybrid'),'--seq_name',SEQ,
         '--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply',
         '--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt']
    result['original_eval_command']=cmd;save()
    with (ROOT/'original_eval.log').open('w') as f:
        subprocess.run(cmd,cwd='/root/OnlineAnySegBaseline_original',env=dict(env,PYTHONPATH='/root/OnlineAnySegBaseline_original'),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
    match=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',(ROOT/'original_eval.log').read_text())
    if not match: raise RuntimeError('No AP in evaluator output')
    result['original_AP']=dict(zip(['AP','AP50','AP25'],map(float,match.groups())))
    baseline_full=json.loads((OLD/'full_prediction_ap/off.json').read_text())
    cmd=[sys.executable,'eval_full_prediction_ap.py','--prediction-dir',str(out),
         '--gt-cloud',str(Path(ref['gtroot'])/SEQ/(SEQ+'_vh_clean_2.ply')),
         '--gt-labels',str(Path(ref['gtseg'])/(SEQ+'.txt')),
         '--gt-to-prediction-transform',str(OLD/'full_prediction_ap/shared_GT_to_prediction.json'),
         '--valid-class-ids',','.join(map(str,baseline_full['config']['GT_valid_classes'])),
         '--min-gt-points','100','--voxel-size','.025','--projection-radius','.15','--output',str(ROOT/'full_prediction_AP.json')]
    result['strict_eval_command']=cmd;save()
    with (ROOT/'strict_eval.log').open('w') as f:
        subprocess.run(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=600)
    full=json.loads((ROOT/'full_prediction_AP.json').read_text())
    result.update(status='complete',FullPred_AP=full['FullPred_AP'],geometry=full['geometry'])
except Exception:
    result.update(status='failed',error=traceback.format_exc())
finally:
    result['wall_seconds']=time.time()-result['started'];save()
    print(json.dumps({k:result.get(k) for k in ['status','original_AP','main_loop_seconds','error']},indent=2),flush=True)
