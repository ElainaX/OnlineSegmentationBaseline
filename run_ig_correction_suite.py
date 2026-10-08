"""Matched four-pair IG controls and pre-fusion contradiction deletion."""
import os,json,subprocess,time,re,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path('/root/autodl-tmp/output/ig_correction_20261008')
REPO=Path('/root/OnlineSegmentationBaseline_ig_correction')
ORIGINAL=Path('/root/OnlineAnySegBaseline_original')
PY='/root/miniconda3/envs/OASeg/bin/python'
PREVIOUS=Path('/root/autodl-tmp/output/ig_submap_merge_ablation_20261007')
refs=[r for r in json.loads((PREVIOUS/'baseline_reference.json').read_text()) if r['seq'].startswith('pair')]
history={r['seq']:r for r in json.loads((PREVIOUS/'disjoint_pairs/results.json').read_text())}
lock=threading.Lock();rows=[]
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
ROOT.mkdir(exist_ok=True)
assert not (ROOT/'results.json').exists(),'Never overwrite a benchmark'
save(ROOT/'sources.json',refs)
def persist():
    with lock:save(ROOT/'results.json',rows)
def job(ref,mode,gpu):
    seq=ref['seq'];dest=ROOT/mode;dest.mkdir(exist_ok=True)
    row=dict(seq=seq,variant=mode,status='mapping',gpu=gpu,historical_IG={k:history[seq][k] for k in ['AP','AP50','AP25']})
    with lock:rows.append(row);save(ROOT/'results.json',rows)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
    try:
        cmd=[PY,'run_ig_correction.py','--config',ref['config'],'--dataset',ref['frames'],'--instance_dir',ref['masks'],
             '--output_dir',str(dest),'--seq_name',seq,'--merge_selection','disjoint_pairs','--correction',mode]
        row['command']=cmd;row['started']=time.time();persist()
        with (dest/(seq+'.run.log')).open('w') as f:rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(returncode=rc,wall_seconds=time.time()-row['started'])
        text=(dest/(seq+'.run.log')).read_text(errors='replace')
        row['warnings']={w:text.count(w) for w in ['CUDA out of memory','Associated bucket and excess list full','Size out of bounds','device-side assert','MemoryError']}
        if rc or any(row['warnings'].values()):raise RuntimeError('Mapping error/warning; no OOM repair or parameter change')
        meta=json.loads((dest/seq/'run_metadata.json').read_text());row['metadata']=meta
        old=history[seq]['run_metadata']
        for k in ['selected_frame_ids','mapping_frame_interval','segmentation_frame_interval','tsdf_voxel_size_mm','integrated_frames','segmented_frames','final_merge','seed']:
            assert meta[k]==old[k],('configuration mismatch',k)
        assert not meta['smoke_stop'];row['configuration_audit']='pass'
        if mode!='off':row['correction']=json.loads((dest/seq/'correction/summary.json').read_text())
        row['status']='evaluation';persist()
        cmd=[PY,'-m','eval.evaluate_seqs','--result_dir',str(dest),'--seq_name',seq,
             '--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply','--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt',
             '--output_file',str(dest/(seq+'_class_agnostic.txt'))]
        row['eval_command']=cmd
        with (dest/(seq+'.eval.log')).open('w') as f:rc=subprocess.call(cmd,cwd=ORIGINAL,env=dict(env,PYTHONPATH=str(ORIGINAL)),stdout=f,stderr=subprocess.STDOUT)
        log=(dest/(seq+'.eval.log')).read_text(errors='replace')
        m=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',log)
        if rc or not m:raise RuntimeError('Evaluation failed; inspect log')
        row.update(dict(zip(['AP','AP50','AP25'],map(float,m.groups()))));row['status']='complete'
    except Exception:row.update(status='failed',error=traceback.format_exc())
    finally:
        persist();print(json.dumps({k:row.get(k) for k in ['seq','variant','status','AP','AP50','AP25','wall_seconds','error']}),flush=True)
def queue(gpu,assigned):
    for ref in assigned:
        for mode in ['off','delete']:job(ref,mode,gpu)
with ThreadPoolExecutor(max_workers=2) as ex:
    fs=[ex.submit(queue,gpu,refs[i::2]) for i,gpu in enumerate([4,7])]
    for f in fs:f.result()
save(ROOT/'done.json',dict(complete=sum(r['status']=='complete' for r in rows),total=len(rows),finished=time.time()))
