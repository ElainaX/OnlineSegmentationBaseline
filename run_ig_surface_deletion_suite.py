"""Strict surface-bank detection, matched local retirement off/on; original rescan GT evaluator."""
import json, os, re, subprocess, threading, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/output/ig_surface_deletion_20261008')
PY='/root/miniconda3/envs/OASeg/bin/python'
ORIGINAL=Path('/root/OnlineAnySegBaseline_original')
refs=json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text())
refs=[r for r in refs if r['seq'].startswith('pair_')]
assert len(refs)==4
ROOT.mkdir(parents=True,exist_ok=True)
assert not (ROOT/'results.json').exists()
rows=[]; lock=threading.Lock()

def save(path, obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def persist():
    with lock:save(ROOT/'results.json',rows)

def job(ref,mode,gpu):
    seq=ref['seq'];dest=ROOT/mode;dest.mkdir(exist_ok=True)
    row=dict(seq=seq,variant=mode,gpu=gpu,status='mapping',started=time.time())
    with lock:rows.append(row);save(ROOT/'results.json',rows)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONPATH=str(REPO),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    try:
        gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
        (dest/(seq+'.gpu_before.txt')).write_text(gpu_state)
        cmd=[PY,'run_ig_surface_deletion.py','--config',ref['config'],'--dataset',ref['frames'],
            '--instance_dir',ref['masks'],'--output_dir',str(dest),'--seq_name',seq,
            '--merge_selection','disjoint_pairs','--detection','on','--deletion',mode,
            '--detector-schedule-json','/root/autodl-tmp/output/ig_rescan_dense_20261008/schedules.json','--verify-erasure']
        row['command']=cmd;persist()
        with (dest/(seq+'.run.log')).open('w') as log:
            rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        row['mapping_returncode']=rc
        if rc:raise RuntimeError('Mapping failed, original log retained; no parameter changes')
        meta=json.loads((dest/seq/'run_metadata.json').read_text())
        old=json.loads((Path('/root/autodl-tmp/output/ig_correction_20261008/off')/seq/'run_metadata.json').read_text())
        for key in ['selected_frame_ids','mapping_frame_interval','segmentation_frame_interval','tsdf_voxel_size_mm',
                    'integrated_frames','segmented_frames','final_merge','seed']:
            assert meta[key]==old[key],('schedule mismatch',key)
        row.update(status='evaluation',main_loop_seconds=meta['main_loop_seconds'],
            mapping_interval=meta['mapping_frame_interval'],segmentation_interval=meta['segmentation_frame_interval'],
            voxel_mm=meta['tsdf_voxel_size_mm'],selected_frames=len(meta['selected_frame_ids']),
            deletion=meta['deletion_summary'],
            detection_seconds=meta['surface_detection_seconds'],
            torch_peak_allocated_bytes=meta['torch_peak_allocated_bytes'],schedule_check='pass')
        persist()
        cmd=[PY,'-m','eval.evaluate_seqs','--result_dir',str(dest),'--seq_name',seq,
            '--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply',
            '--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt',
            '--output_file',str(dest/(seq+'_class_agnostic.txt'))]
        row['eval_command']=cmd
        with (dest/(seq+'.eval.log')).open('w') as log:
            rc=subprocess.call(cmd,cwd=ORIGINAL,env=dict(env,PYTHONPATH=str(ORIGINAL)),stdout=log,stderr=subprocess.STDOUT,timeout=600)
        log=(dest/(seq+'.eval.log')).read_text(errors='replace')
        m=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',log)
        if rc or not m:raise RuntimeError('Evaluation failed; preserved log')
        row.update(dict(zip(['AP','AP50','AP25'],map(float,m.groups()))))
        row['status']='complete'
    except Exception:row.update(status='failed',error=traceback.format_exc())
    finally:
        row['wall_seconds']=time.time()-row['started'];persist()
        print(json.dumps({k:row.get(k) for k in ['seq','variant','status','AP','AP50','AP25','error']}),flush=True)

def queue(gpu,assigned):
    for ref in assigned:
        for mode in ['off','on']:job(ref,mode,gpu)

with ThreadPoolExecutor(max_workers=2) as pool:
    tasks=[pool.submit(queue,gpu,refs[i::2]) for i,gpu in enumerate([1,3])]
    for t in tasks:t.result()
save(ROOT/'done.json',dict(total=len(rows),complete=sum(r['status']=='complete' for r in rows),finished=time.time()))
