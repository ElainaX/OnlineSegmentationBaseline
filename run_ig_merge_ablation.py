"""IG-Submap with baseline-matched schedule; only merge selection differs."""
import argparse, json, os, sys, time, random, subprocess, hashlib
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.append('third_party/FCGF')
from Dataset.dataset import get_dataset
from tool.config import load_config
from submap_disjoint.geometry import GeometryScene
from submap_disjoint.sparse_owner import SparseDisjointSubmapManager

p=argparse.ArgumentParser()
for flag in ['config','dataset','instance_dir','output_dir','seq_name']:
    p.add_argument('--'+flag,required=True)
p.add_argument('--merge_selection',choices=['disjoint_pairs','baseline_components'],required=True)
p.add_argument('--device',default='cuda:0')
a=p.parse_args()
out=Path(a.output_dir)/a.seq_name; out.mkdir(parents=True,exist_ok=True)
assert not (out/'ckpt_final.npz').exists()
start=time.perf_counter()
random.seed(27); np.random.seed(27); torch.manual_seed(27); torch.cuda.manual_seed_all(27)
cfg=load_config(a.config)
# Representation and experiment controls; baseline semantic/numeric parameters
# are read unchanged from the archived baseline config above.
cfg['ig_submap']={'merge_selection':a.merge_selection,'final_merge':False}
cfg['sparse_candidate']={'enabled':True,'verify':False}
cfg['visibility_storage']='bool'
cfg.setdefault('seg',{}).setdefault('depth_shrink',{})['enabled']=False
cfg['fcgf_diagnostics']={'enabled':False,'use_in_merge':True,'export_high_similarity_pairs':False}
data=get_dataset(a.dataset,a.instance_dir,cfg,a.device)
geometry=GeometryScene(cfg,a,data,a.device)
owner=SparseDisjointSubmapManager(cfg,a,data,geometry,a.device)
assert not hasattr(geometry,'maskGraph')
run_metadata=dict(method='IG-Submap-v1',merge_selection=a.merge_selection,config=cfg,
    baseline_config_path=a.config,baseline_config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
    mapping_frame_interval=cfg['mapping']['keyframe_freq'],segmentation_frame_interval=cfg['seg']['seg_add_interval'],
    tsdf_voxel_size_m=cfg['scene']['voxel_size'],tsdf_voxel_size_mm=cfg['scene']['voxel_size']*1000,
    final_merge=False,seed=27,visibility_storage='bool',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),raw_frame_count=len(data))
selected=list(range(0,len(data),cfg['mapping']['keyframe_freq']))
run_metadata['selected_frame_ids']=selected
boundary=Path(a.dataset)/'session_boundary.json'
if boundary.exists():
    b=json.loads(boundary.read_text()); cut=b['rescan_start_frame']
    run_metadata.update(reference_sampled_frames=sum(i<cut for i in selected),rescan_sampled_frames=sum(i>=cut for i in selected))
(out/'run_metadata.json').write_text(json.dumps(run_metadata,indent=2))
trace=[]; original_update=owner.update_masks
def update(fid):
    original_update(fid)
    trace.append(dict(frame=int(fid),count=owner.c_mask_num))
owner.update_masks=update
last_pose=torch.eye(4); integrated=0; observed=0
torch.cuda.synchronize(); loop_start=time.perf_counter()
for frame_id,(rgb,depth,pose,seg,features,flag) in tqdm(enumerate(DataLoader(data,batch_size=1,num_workers=0))):
    if not torch.isfinite(pose[0]).all(): pose[0]=last_pose
    else: last_pose=pose[0]
    if flag[0]: seg=seg[0]; features=features[0]
    if frame_id % cfg['mapping']['keyframe_freq']==0:
        with torch.no_grad():
            blocks,_=geometry.integrate_frame(frame_id,rgb[0],depth[0],torch.inverse(pose[0])); integrated+=1
            if flag[0] and frame_id % cfg['seg']['seg_add_interval']==0:
                owner.observe(frame_id,rgb[0],depth[0],pose[0],blocks,seg,features); observed+=1
    if cfg['save']['ckpt_interval']>0 and frame_id>0 and frame_id%cfg['save']['ckpt_interval']==0:
        ckpt=out/f'ckpt_{frame_id}'; ckpt.mkdir(exist_ok=True)
        owner.save_ckpt(frame_id,str(ckpt))
torch.cuda.synchronize(); loop_seconds=time.perf_counter()-loop_start
owner.save_outputs(len(data)-1,str(out))
torch.cuda.synchronize()
assert all(x['frame']%owner.merge_frame_interval==0 for x in trace),'unexpected off-schedule merge'
run_metadata.update(status='complete',integrated_frames=integrated,segmented_frames=observed,
                    main_loop_seconds=loop_seconds,total_seconds=time.perf_counter()-start,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    merge_audit=owner.migration_audit,export_points=len(owner.points))
(out/'run_metadata.json').write_text(json.dumps(run_metadata,indent=2))
(out/'merge_trace.json').write_text(json.dumps(trace,indent=2))
print('COMPLETE',json.dumps({k:run_metadata[k] for k in ['merge_selection','main_loop_seconds','total_seconds','merge_audit']}),flush=True)
