import json, os, copy, re, subprocess, time, threading, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/root/autodl-tmp/output/it_submap_disjoint_20261007')
REPO=Path('/root/OnlineSegmentationBaseline_baseline_ablation')
ORIGINAL=Path('/root/OnlineAnySegBaseline_original')
PY='/root/miniconda3/envs/OASeg/bin/python'
rows=json.loads((ROOT/'results.json').read_text())
assert len(rows)==6 and all(x['status']=='complete' for x in rows)
refs={x['seq']:x for x in json.loads((ROOT/'baseline_reference.json').read_text())}
assert not (ROOT/'reexport_results.json').exists()
dest=ROOT/'IT_min1';dest.mkdir(exist_ok=True)
lock=threading.Lock();results=[]
def save():
    temp=ROOT/'reexport_results.tmp';temp.write_text(json.dumps(results,indent=2));temp.replace(ROOT/'reexport_results.json')
def run(old,gpu):
    seq=old['seq'];row=copy.deepcopy(old)
    row.update(status='reexporting',initial_IT_AP={k:old[k] for k in ['AP','AP50','AP25']},
        initial_it_audit=old['it_audit'],surface_strict_threshold=.5,export_only_repair=True,reexport_gpu=gpu)
    with lock:results.append(row);save()
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(REPO))
    try:
        cmd=[PY,'-u','reexport_it_surface.py','--source',str(ROOT/'IT'/seq),'--output',str(dest/seq)]
        with open(dest/(seq+'.reexport.log'),'w') as f:rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        row['reexport_returncode']=rc
        if rc:row['status']='reexport_failed';return
        m=json.loads((dest/seq/'it_manifest.json').read_text())
        row.update(it_audit=m['audit'],reexport_seconds=m['reexport_seconds'],
            recovered_fragments=m['new_surface_fragments_from_empty'],remaining_empty_fragments=m['remaining_empty_fragments'],
            original_export_parity=m['original_export_parity'],reexport_commit=m['code_commit'])
        r=refs[seq]
        cmd=[PY,'-m','eval.evaluate_seqs','--result_dir',str(dest),'--seq_name',seq,'--gt_dir',r['gtroot'],
             '--gt_pc_pattern','%s/%s_vh_clean_2.ply','--gt_seg_dir',r['gtseg'],'--gt_seg_pattern','%s.txt',
             '--output_file',str(dest/(seq+'_class_agnostic.txt'))]
        t=time.time()
        with open(dest/(seq+'.eval.log'),'w') as f:rc=subprocess.call(cmd,cwd=ORIGINAL,env=dict(env,PYTHONPATH=str(ORIGINAL)),stdout=f,stderr=subprocess.STDOUT)
        row['reexport_eval_seconds']=time.time()-t
        text=(dest/(seq+'.eval.log')).read_text(errors='replace')
        match=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',text)
        if rc or not match:raise RuntimeError('re-export AP failed')
        row.update(dict(zip(['AP','AP50','AP25'],map(float,match.groups()))));row['status']='complete'
    except Exception:row.update(status='failed',error=traceback.format_exc())
    finally:
        with lock:
            save();print(json.dumps({k:row.get(k) for k in ['seq','status','AP','AP50','AP25','reexport_seconds','recovered_fragments','error']}),flush=True)
def queue(gpu,assigned):
    for row in assigned:run(row,gpu)
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(queue,gpu,rows[i::2]) for i,gpu in enumerate([4,7])]
    for f in futures:f.result()
(ROOT/'reexport_done.json').write_text(json.dumps(dict(complete=sum(x['status']=='complete' for x in results),total=len(results))))
