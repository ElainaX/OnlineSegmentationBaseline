"""Frozen-association experiment; no GT is read until evaluation subprocesses.

Replay the exact saved track observations into local TSDFs. Export a fixed
identity 2.5/2.0cm ablation and a held-out observation selected mixture. This
does not measure end-to-end online runtime, nor change association/deletion.
"""
import argparse,copy,json,os,sys,time,subprocess,threading
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch
import open3d as o3d
sys.path.append('third_party/FCGF')
from Dataset.dataset import get_dataset
from submap_disjoint.it_adapter_multires import InstanceTSDFAdapter
from submap_disjoint.selective_local import split_views,score_surface,choose_resolution,assemble_outputs

p=argparse.ArgumentParser()
p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
p.add_argument('--source-batch',default='/root/autodl-tmp/multires_extra2_20261013')
p.add_argument('--evaluate',action='store_true')
a=p.parse_args();start=time.monotonic();out=Path(a.output)
if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing overwrite')
out.mkdir(parents=True,exist_ok=True)
source=Path(a.source_batch);seq=a.scene;source_scene=source/'results'/seq
ref=next(r for r in json.loads((source/'manifest.json').read_text()) if r['seq']==seq)
metadata=json.loads((source_scene/'run_metadata.json').read_text())
cfg=copy.deepcopy(metadata['config']);cfg['multi_resolution']['fine_voxel_size_m']=.020
coarse_dir=source_scene/'ig_audit';ids=json.loads((coarse_dir/'export_instance_ids.json').read_text())
tracks={r['id']:r for r in json.loads((coarse_dir/'submap_migration.json').read_text())['instances']}
cloud=o3d.io.read_point_cloud(str(coarse_dir/'final.ply'))
xyz=np.asarray(cloud.points);rgb=np.asarray(cloud.colors)
with np.load(coarse_dir/'ckpt_final.npz') as f:arrays={k:f[k] for k in f.files}
masks=arrays['pred_masks'];assert masks.shape==(len(xyz),len(ids))
data=get_dataset(ref['dataset'],ref['masks'],cfg,'cpu')
owner=SimpleNamespace(device='cuda:0',intrinsics=data.cam_intrinsic)
adapter=InstanceTSDFAdapter(cfg,owner,out/'scratch')
options={name:replace(adapter.options,voxel_m=v,trunc_voxels=.1/v,max_blocks=30000)
         for name,v in [('local25',.025),('local20',.020)]}
cache={};records=[];surfaces={'local25':{},'local20':{},'selected':{}}
peaks={'device_peak_mib':None,'compute_process_sum_peak_mib':None};stop=threading.Event()
def sample():
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    while not stop.is_set():
        try:
            val=int(subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
            peaks['device_peak_mib']=max(peaks['device_peak_mib'] or 0,val)
        except Exception:pass
        stop.wait(1.)
sampler=threading.Thread(target=sample,daemon=True);sampler.start()
def packet(frame):
    if frame not in cache:
        color,depth,pose,labels,features,flag=data[frame]
        if not flag or not torch.isfinite(pose).all():raise ValueError('Saved observation is unavailable')
        d=depth.numpy();scale=float(cfg['cam']['depth_scale'])
        d=(np.where(np.isfinite(d)&(d>0),d,0)*scale).astype(np.uint16).astype(np.float32)/scale
        cache[frame]=dict(frame=frame,rgb=(color.numpy()*255).astype(np.uint8),depth=d,
                          pose=pose.numpy().astype(np.float64),labels=labels.numpy())
    return cache[frame]
def integrate(views,kind):
    x,c,_=adapter._integrate_volume_with_options(views,options[kind])
    return (x,c/255.)
def write_variant(name,replacements):
    dst=out/name/seq;dst.mkdir(parents=True)
    x,c,m=assemble_outputs(xyz,rgb,masks,replacements)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(x);cloud.colors=o3d.utility.Vector3dVector(c)
    o3d.io.write_point_cloud(str(dst/'final.ply'),cloud)
    np.savez_compressed(dst/'ckpt_final.npz',**dict(arrays,pred_masks=m))
    (dst/'export_instance_ids.json').write_text(json.dumps(ids))
    (dst/'replay_manifest.json').write_text(json.dumps(dict(fixed_ids=ids,modified_columns=sorted(replacements),
        modified_track_ids=[ids[i] for i in sorted(replacements)],points=len(x),
        no_gt_selection=True,global_geometry_preserved=True,
        note='Target masks replaced by local surfaces. Ordinary masks and identity/score arrays unchanged.'),indent=2))
    return str(dst)

for col,mid in enumerate(ids):
    began=time.monotonic();by_frame={}
    for frame,raw in tracks[mid]['observations']:by_frame.setdefault(frame,[]).append(raw)
    views=[(packet(f),np.isin(packet(f)['labels'],raw)) for f,raw in sorted(by_frame.items())]
    train,val=split_views(views);row=dict(track_id=mid,column=col,views=len(views),choice='ig')
    if not val:
        row['reason']='insufficient_held_out_views';records.append(row);continue
    try:
        base_scores=score_surface(xyz[masks[:,col]],val,adapter.K)
        coarse_train=integrate(train,'local25')
        coarse_scores=score_surface(coarse_train[0],val,adapter.K)
        # Do not spend fine reconstruction on a poor local hypothesis or an
        # already saturated surface. This gate uses RGB-D only, not AP.
        fine_scores=None;fine_train=None
        if np.mean(coarse_scores)>=np.mean(base_scores)-.02 and np.mean(coarse_scores)<.98:
            fine_train=integrate(train,'local20')
            fine_scores=score_surface(fine_train[0],val,adapter.K)
        choice=choose_resolution(base_scores,coarse_scores,fine_scores)
        row.update(choice=choice,base_scores=base_scores,local25_scores=coarse_scores,
                   local20_scores=fine_scores,validation_frames=[p['frame'] for p,_ in val],
                   training_frames=[p['frame'] for p,_ in train],selection_seconds=time.monotonic()-began)
        # Full-data reconstructions are for the controlled ablation and final
        # refit only; held-out scores above never see these surfaces.
        coarse=integrate(views,'local25');fine=integrate(views,'local20')
        if not len(coarse[0]) or not len(fine[0]):raise ValueError('Empty paired ablation surface')
        surfaces['local25'][col]=coarse;surfaces['local20'][col]=fine
        if choice!='ig':surfaces['selected'][col]=coarse if choice=='local25' else fine
        row.update(local25_points=len(coarse[0]),local20_points=len(fine[0]))
    except MemoryError as exc:
        row.update(choice='ig',reason='block_budget_fallback',error=str(exc))
    row['total_seconds']=time.monotonic()-began;records.append(row)
    (out/'selection.json').write_text(json.dumps(records,indent=2))
    print('TRACK',mid,row['choice'],round(row['total_seconds'],2),flush=True)

construction_seconds=time.monotonic()-start
paths={k:write_variant(k,v) for k,v in surfaces.items()}
report=dict(scene=seq,kind='frozen_association_replay',status='export_complete',source=str(source_scene),
    source_commit=metadata['commit'],commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    coarse_m=.025,fine_m=.020,physical_truncation_m=.1,score_tolerance_m=.025,mean_improvement=.02,
    candidate_ids=ids,records=records,paths=paths,construction_seconds=construction_seconds,
    selection_seconds=sum(r.get('selection_seconds',0) for r in records),
    gpu_sampling=peaks,gpu_note='Card-level 1s samples may include external processes; not 4090 tested.',
    no_association_rerun=True,correction_enabled=False,deletion_enabled=False)
(out/'report.json').write_text(json.dumps(report,indent=2))
if a.evaluate:
    results={}
    for name in paths:
        cmd=[sys.executable,'-m','eval.evaluate_seqs','--result_dir',str(out/name),'--seq_name',seq,
             '--gt_dir',ref['gtroot'],'--gt_seg_dir',ref['gtseg'],'--pred_path',str(out/'predictions'/name),
             '--output_file',str(out/(name+'_ap.txt'))]
        with (out/(name+'_eval.log')).open('w') as log:
            subprocess.run(cmd,cwd='/root/OnlineAnySegBaseline_original',env=dict(os.environ,PYTHONPATH='/root/OnlineAnySegBaseline_original'),stdout=log,stderr=subprocess.STDOUT,check=True)
        # Official evaluator appends its evaluation mode to the requested stem.
        metric_file=out/(name+'_ap_class_agnostic.txt')
        last=metric_file.read_text().splitlines()[-1]
        results[name]=dict(zip(['AP','AP50','AP25'],map(float,last.split(','))))
    report['results']=results
stop.set();sampler.join(3)
report.update(status='complete',total_seconds=time.monotonic()-start,gpu_sampling=peaks)
(out/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['records','paths','candidate_ids']}),flush=True)
