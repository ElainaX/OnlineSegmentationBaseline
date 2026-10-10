import json, os, subprocess, time, re, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO=Path('/root/OnlineSegmentationBaseline_ig_deletion')
PY='/root/miniconda3/envs/OASeg/bin/python'
REF_FILE=Path('/root/autodl-tmp/output/it_submap_disjoint_20261007/baseline_reference.json')
OUT=Path('/root/autodl-tmp/output/it_counterfactual_20261010')
OUT.mkdir(parents=True,exist_ok=True)
refs={x['seq']:x for x in json.loads(REF_FILE.read_text()) if x.get('seq','').startswith('pair_0')}
seqs=sorted(refs)
assert len(seqs)==4,seqs
lock=None
rows=[]

def save():
    (OUT/'results.json').write_text(json.dumps(rows,indent=2))

def parse(log):
    text=Path(log).read_text(errors='replace')
    m=re.findall(r'average\s*:\s*([\d.nan]+)\s+([\d.nan]+)\s+([\d.nan]+)',text)
    if not m: raise RuntimeError('No AP in '+str(log))
    return dict(zip(['AP','AP50','AP25'],map(float,m[-1])))

def eval_seq(root,ref,log):
    cmd=[PY,'-m','eval.evaluate_seqs','--result_dir',str(root),'--seq_name',ref['seq'],
         '--gt_dir',ref['gtroot'],'--gt_pc_pattern','%s/%s_vh_clean_2.ply',
         '--gt_seg_dir',ref['gtseg'],'--gt_seg_pattern','%s.txt',
         '--output_file',str(Path(root)/(ref['seq']+'_class_agnostic.txt'))]
    with open(log,'w') as f:
        rc=subprocess.call(cmd,cwd='/root/OnlineAnySegBaseline_original',env=dict(os.environ,PYTHONPATH='/root/OnlineAnySegBaseline_original'),stdout=f,stderr=subprocess.STDOUT)
    if rc: raise RuntimeError('evaluation failed')
    return parse(log)

def job(gpu,variant,assigned):
    for seq in assigned:
        ref=refs[seq]
        root=OUT/variant; root.mkdir(exist_ok=True)
        row={'group':variant,'seq':seq,'gpu':gpu,'status':'mapping'}
        rows.append(row);save()
        dest=root/seq
        try:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONUNBUFFERED='1',PYTHONPATH=str(REPO))
            cmd=[PY,'run_it_submap_ablation.py','--config',ref['config'],'--dataset',ref['frames'],'--instance_dir',ref['masks'],
                 '--output_dir',str(root),'--seq_name',seq,'--merge_selection','disjoint_pairs']
            if variant=='IT_counterfactual': cmd += ['--counterfactual','on']
            else: cmd += ['--counterfactual','off']
            row['command']=cmd; row['started']=time.time()
            with open(root/(seq+'.run.log'),'w') as f:
                rc=subprocess.call(cmd,cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
            row['returncode']=rc; row['run_seconds']=time.time()-row['started']
            if rc: raise RuntimeError('mapping failed')
            meta=json.loads((dest/'run_metadata.json').read_text())
            manifest=json.loads((dest/'it_manifest.json').read_text())
            row['run_metadata']={k:meta.get(k) for k in ['mapping_frame_interval','segmentation_frame_interval','tsdf_voxel_size_mm','reference_sampled_frames','rescan_sampled_frames','main_loop_seconds','total_seconds','export_points']}
            row['it_audit']=manifest.get('audit',{})
            if variant=='IT_counterfactual':
                c=json.loads((dest/'counterfactual'/'summary.json').read_text())
                row['counterfactual']={'decisions':len(c.get('decisions',[])),'retired':len(c.get('retired',[])),'kept':sum(1 for d in c.get('decisions',[]) if d.get('decision')!='retire'),'seconds':c.get('seconds',0.)}
            row['status']='evaluation'; save()
            row.update(eval_seq(root,ref,root/(seq+'.eval.log')))
            row['status']='complete'; row['eval_seconds']=0.; save()
        except Exception:
            row['status']='failed'; row['error']=traceback.format_exc(); save()
        print(json.dumps({k:row.get(k) for k in ['group','seq','status','AP','AP50','AP25','run_seconds','counterfactual','error']}),flush=True)

# Existing baseline runs are the same 25mm / sampling inputs and are used as the fixed baseline group.
for seq in seqs:
    x=refs[seq]
    rows.append({'group':'baseline_existing','seq':seq,'status':'complete','AP':x.get('AP'),'AP50':x.get('AP50'),'AP25':x.get('AP25'),
                 'config':x.get('config'),'frames':x.get('input_audit',{}).get('selected_frames')})
save()
assign=[seqs[::2],seqs[1::2]]
with ThreadPoolExecutor(max_workers=2) as ex:
    fs=[]
    for gpu,sc in zip([4,7],assign): fs.append(ex.submit(job,gpu,'IT_base',sc))
    for f in fs:f.result()
with ThreadPoolExecutor(max_workers=2) as ex:
    fs=[]
    for gpu,sc in zip([4,7],assign): fs.append(ex.submit(job,gpu,'IT_counterfactual',sc))
    for f in fs:f.result()
summary=[]
for r in rows:
    summary.append({k:r.get(k) for k in ['group','seq','status','AP','AP50','AP25','run_seconds','counterfactual','run_metadata']})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({'out':str(OUT),'rows':len(rows),'complete':sum(r.get('status')=='complete' for r in rows)},indent=2))
