"""Six ITSubmap benchmarks plus same-run IG association controls; no GT in mapping."""
import os, json, subprocess, time, re, traceback, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('/root/autodl-tmp/output/it_submap_disjoint_20261007')
PREVIOUS=Path('/root/autodl-tmp/output/ig_submap_merge_ablation_20261007')
REPO=Path('/root/OnlineSegmentationBaseline_baseline_ablation')
ORIGINAL=Path('/root/OnlineAnySegBaseline_original')
PY='/root/miniconda3/envs/OASeg/bin/python'
sources=json.loads((PREVIOUS/'baseline_reference.json').read_text())
sources.sort(key=lambda r:(0 if r['seq']=='scene0549_00' else 1 if r['seq']=='scene0050_00' else 2,r['seq']))
control={r['seq']:r for r in json.loads((PREVIOUS/'results.json').read_text()) if r['merge_selection']=='disjoint_pairs'}
lock=threading.Lock();rows=[]

def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

ROOT.mkdir(exist_ok=True)
assert not (ROOT/'results.json').exists(),'Do not overwrite an experiment'
save(ROOT/'ig_disjoint_reference.json',list(control.values()))
save(ROOT/'baseline_reference.json',sources)

def evaluate(seq,root,ref,env,log):
    command=[PY,'-m','eval.evaluate_seqs','--result_dir',str(root),'--seq_name',seq,
        '--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply',
        '--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt',
        '--output_file',str(root/(seq+'_class_agnostic.txt'))]
    start=time.time()
    with open(log,'w') as f:rc=subprocess.call(command,cwd=ORIGINAL,env=dict(env,PYTHONPATH=str(ORIGINAL)),stdout=f,stderr=subprocess.STDOUT)
    text=Path(log).read_text(errors='replace')
    m=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',text)
    if rc or not m:raise RuntimeError('Evaluation failed: '+str(log))
    return dict(zip(['AP','AP50','AP25'],map(float,m.groups()))),time.time()-start

def run_job(ref,gpu):
    seq=ref['seq'];dest=ROOT/'IT';dest.mkdir(exist_ok=True)
    row=dict(seq=seq,gpu=gpu,status='mapping',control={k:control[seq][k] for k in ['AP','AP50','AP25']})
    with lock:rows.append(row);save(ROOT/'results.json',rows)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
             MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
    try:
        config=dest/(seq+'.yaml');config.write_text(Path(ref['config']).read_text())
        cmd=[PY,'run_it_submap_ablation.py','--config',str(config),'--dataset',ref['frames'],
             '--instance_dir',ref['masks'],'--output_dir',str(dest),'--seq_name',seq,'--merge_selection','disjoint_pairs']
        save(ROOT/f'gpu{gpu}_current.json',dict(seq=seq,command=cmd,stage='mapping',started=time.time()))
        start=time.time()
        with open(dest/(seq+'.run.log'),'w') as f:
            rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(run_returncode=rc,run_wall_seconds=time.time()-start)
        text=(dest/(seq+'.run.log')).read_text(errors='replace')
        row['warnings']={w:text.count(w) for w in ['CUDA out of memory','Associated bucket and excess list full','Size out of bounds','device-side assert','MemoryError','misaligned address']}
        if rc:row['status']='run_failed';return
        meta=json.loads((dest/seq/'run_metadata.json').read_text());row['run_metadata']=meta
        manifest=json.loads((dest/seq/'it_manifest.json').read_text());row['it_audit']=manifest['audit']
        old=control[seq]['run_metadata']
        for k in ['selected_frame_ids','mapping_frame_interval','segmentation_frame_interval',
                  'tsdf_voxel_size_mm','baseline_config_sha256','integrated_frames','segmented_frames','final_merge','seed']:
            assert meta[k]==old[k],('control mismatch',k)
        assert not meta['smoke_frames'] and not meta['geometry_fallback'] and not meta['correction_enabled']
        assert not manifest['carving_aabb_enabled'] and not manifest['depth_shrink_enabled']
        row['configuration_audit']='pass'
        row['status']='evaluation'
        with lock:save(ROOT/'results.json',rows)
        save(ROOT/f'gpu{gpu}_current.json',dict(seq=seq,stage='evaluation',started=time.time()))
        scores,seconds=evaluate(seq,dest,ref,env,dest/(seq+'.eval.log'))
        row.update(scores);row['eval_wall_seconds']=seconds
        audit=ROOT/'IG_audit';audit.mkdir(exist_ok=True)
        (audit/seq).symlink_to(dest/seq/'ig_audit',target_is_directory=True)
        row['ig_same_run_AP'],row['ig_eval_seconds']=evaluate(seq,audit,ref,env,audit/(seq+'.eval.log'))
        row['ig_history_AP_equal']=row['ig_same_run_AP']==row['control']
        row['status']='complete'
    except Exception:row.update(status='failed',error=traceback.format_exc())
    finally:
        with lock:
            save(ROOT/'results.json',rows)
            print(json.dumps({k:row.get(k) for k in ['seq','status','AP','AP50','AP25','ig_history_AP_equal','run_wall_seconds']}),flush=True)

def queue(gpu,assigned):
    for ref in assigned:run_job(ref,gpu)
    save(ROOT/f'gpu{gpu}_current.json',dict(stage='done',finished=time.time()))

with ThreadPoolExecutor(max_workers=2) as executor:
    futures=[executor.submit(queue,gpu,sources[i::2]) for i,gpu in enumerate([4,7])]
    for f in futures:f.result()
save(ROOT/'done.json',dict(complete=sum(r['status']=='complete' for r in rows),total=len(rows),finished=time.time()))
