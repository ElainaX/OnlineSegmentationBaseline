"""Two-scene, four-arm IT study. No GT is supplied to reconstruction."""
import os,json,subprocess,time,re,traceback,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ROOT=Path('/root/autodl-tmp/output/it_factor_ablation_20261007')
REPO=Path('/root/OnlineSegmentationBaseline_baseline_ablation')
ORIGINAL=Path('/root/OnlineAnySegBaseline_original')
PY='/root/miniconda3/envs/OASeg/bin/python'
previous=Path('/root/autodl-tmp/output/it_submap_disjoint_20261007')
refs={r['seq']:r for r in json.loads((previous/'baseline_reference.json').read_text()) if r['seq'].startswith('scene')}
lock=threading.Lock(); rows=[]
def save(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def persist():
    with lock:save(ROOT/'results.json',rows)
ROOT.mkdir(exist_ok=True)
assert not (ROOT/'results.json').exists(),'Never overwrite existing runs'
save(ROOT/'sources.json',refs)
variants={'IT_control':(0,0),'IT_AABB':(1,0),'IT_depth_k70':(0,1),'IT_AABB_depth_k70':(1,1)}

def evaluate(seq,root,ref,env,log):
    cmd=[PY,'-m','eval.evaluate_seqs','--result_dir',str(root),'--seq_name',seq,
        '--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply',
        '--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt',
        '--output_file',str(root/(seq+'_class_agnostic.txt'))]
    start=time.time()
    with open(log,'w') as f:rc=subprocess.call(cmd,cwd=ORIGINAL,env=dict(env,PYTHONPATH=str(ORIGINAL)),stdout=f,stderr=subprocess.STDOUT)
    text=log.read_text(errors='replace')
    match=re.search(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',text)
    if rc or not match:raise RuntimeError('Evaluation failed: '+str(log))
    return dict(zip(['AP','AP50','AP25'],map(float,match.groups()))),time.time()-start

def job(seq,variant,gpu):
    ref=refs[seq];aabb,shrink=variants[variant]
    dest=ROOT/variant;dest.mkdir(exist_ok=True)
    row=dict(seq=seq,variant=variant,gpu=gpu,status='mapping',aabb=bool(aabb),depth_shrink=bool(shrink))
    with lock:rows.append(row);save(ROOT/'results.json',rows)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
             OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
    try:
        cfg=dest/(seq+'.yaml');cfg.write_text(Path(ref['config']).read_text())
        cmd=[PY,'run_it_submap_ablation.py','--config',str(cfg),'--dataset',ref['frames'],
             '--instance_dir',ref['masks'],'--output_dir',str(dest),'--seq_name',seq,
             '--merge_selection','disjoint_pairs','--aabb',str(aabb),'--depth-shrink',str(shrink)]
        row['command']=cmd;row['started']=time.time();persist()
        with open(dest/(seq+'.run.log'),'w') as f:rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        row.update(run_returncode=rc,run_wall_seconds=time.time()-row['started'])
        log=(dest/(seq+'.run.log')).read_text(errors='replace')
        row['warnings']={w:log.count(w) for w in ['CUDA out of memory','Associated bucket and excess list full','Size out of bounds','device-side assert','MemoryError','misaligned address']}
        if rc:raise RuntimeError('Mapping failed; see run log')
        meta=json.loads((dest/seq/'run_metadata.json').read_text());row['run_metadata']=meta
        manifest=json.loads((dest/seq/'it_manifest.json').read_text());row['it_audit']=manifest['audit']
        old=json.loads((previous/'IT'/seq/'run_metadata.json').read_text())
        for key in ['selected_frame_ids','mapping_frame_interval','segmentation_frame_interval',
                    'tsdf_voxel_size_mm','baseline_config_sha256','integrated_frames','segmented_frames','final_merge','seed']:
            assert old[key]==meta[key],('control mismatch',key)
        assert manifest['options']['extraction_weight']==.5
        assert manifest['carving_aabb_enabled']==bool(aabb) and manifest['depth_shrink_enabled']==bool(shrink)
        assert not meta['correction_enabled'] and not meta['geometry_fallback'] and not meta['smoke_frames']
        if aabb or shrink:
            factor=json.loads((dest/seq/'ig_audit'/'factor_manifest.json').read_text());row['factor']=factor
            assert factor['removed_pixels']==manifest['audit']['excluded_depth_pixels']
        row['configuration_audit']='pass';row['status']='evaluation';persist()
        scores,seconds=evaluate(seq,dest,ref,env,dest/(seq+'.eval.log'))
        row.update(scores);row['eval_seconds']=seconds
        audit=ROOT/(variant+'_IG_audit');audit.mkdir(exist_ok=True)
        (audit/seq).symlink_to(dest/seq/'ig_audit',target_is_directory=True)
        row['ig_same_run_AP'],row['ig_eval_seconds']=evaluate(seq,audit,ref,env,audit/(seq+'.eval.log'))
        row['status']='complete'
    except Exception:row.update(status='failed',error=traceback.format_exc())
    finally:
        persist()
        print(json.dumps({k:row.get(k) for k in ['seq','variant','status','AP','AP50','AP25','run_wall_seconds','error']}),flush=True)

jobs=[('scene0549_00','IT_control'),('scene0549_00','IT_AABB'),
      ('scene0050_00','IT_control'),('scene0050_00','IT_AABB'),
      ('scene0549_00','IT_depth_k70'),('scene0549_00','IT_AABB_depth_k70'),
      ('scene0050_00','IT_depth_k70'),('scene0050_00','IT_AABB_depth_k70')]
def queue(gpu,assigned):
    for seq,variant in assigned:job(seq,variant,gpu)
with ThreadPoolExecutor(max_workers=2) as executor:
    futures=[executor.submit(queue,gpu,jobs[i::2]) for i,gpu in enumerate([4,7])]
    for f in futures:f.result()
save(ROOT/'done.json',dict(complete=sum(r['status']=='complete' for r in rows),total=len(rows),finished=time.time()))
if all(r['status']=='complete' for r in rows):
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with open(ROOT/'coverage.log','w') as f:
        rc=subprocess.call([PY,str(ROOT/'measure_coverage.py')],env=env,stdout=f,stderr=subprocess.STDOUT)
    save(ROOT/'coverage_status.json',dict(returncode=rc))
