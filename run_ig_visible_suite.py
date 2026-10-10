"""Matched input, separate detection-only/deletion runs; never overwrite outputs."""
import os,sys,json,time,subprocess,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
REPO=Path('/root/OnlineSegmentationBaseline_ig_deletion')
OUT=Path('/root/autodl-tmp/output/ig_visible_v3_20261010')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')
INPUT=Path('/root/autodl-tmp/output/six_scene_extension_20261008')
SEQS=['pair_09_0cac7676_7272e17c','pair_10_10b17940_5630cfe7']
rows=[];lock=threading.Lock()

def save(row):
 with lock:
  rows.append(row);temp=OUT/'run_results.tmp';temp.write_text(json.dumps(rows,indent=2));temp.replace(OUT/'run_results.json')

def worker(seq,gpu):
 for mode in ['detect','delete']:
  root=OUT/mode;root.mkdir(exist_ok=True)
  cmd=[sys.executable,'run_ig_visible.py','--config',str(INPUT/'configs'/(seq+'.yaml')),
    '--dataset',str(DATA/seq/'frames'),'--instance_dir',str(INPUT/'seg_result'/seq),
    '--output_dir',str(root),'--seq_name',seq,'--merge_selection','disjoint_pairs','--correction',mode]
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONPATH=str(REPO),PYTHONUNBUFFERED='1')
  start=time.time()
  with (root/(seq+'.run.log')).open('x') as f:rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
  row=dict(seq=seq,mode=mode,gpu=gpu,command=cmd,returncode=rc,seconds=time.time()-start)
  if not rc:
   s=json.loads((root/seq/'correction/summary.json').read_text())
   row.update(references=len(s['references']),candidate_references=sum(r['candidates']>0 for r in s['references'].values()),
    erased_voxels=s['unique_deleted_world_voxels'],failures=s['failure_reasons'])
  save(row);print(json.dumps(row),flush=True)
  if rc:return

if __name__=='__main__':
 OUT.mkdir(parents=True,exist_ok=True)
 if (OUT/'run_results.json').exists():raise RuntimeError('Benchmark already exists')
 with ThreadPoolExecutor(2) as pool:list(pool.map(lambda args:worker(*args),zip(SEQS,[4,7])))
 (OUT/'done.json').write_text(json.dumps(dict(complete=len(rows)==4 and all(r['returncode']==0 for r in rows),runs=len(rows))))
