"""IGSubmap-InstDel with adaptive instance-local TSDFs.

This experimental runner keeps Feng's contradiction detector/deletion policy,
while preserving the final IG coarse export. Only final valid, stable small
instances replay raw RGB-D observations at fine resolution for supported
local substitution; ordinary instances keep their original geometry and masks.
"""
import argparse, json, os, sys, time, random, subprocess, hashlib
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
sys.path.append('third_party/FCGF')
from Dataset.dataset import get_dataset
from tool.config import load_config
from submap_disjoint.geometry import GeometryScene
from submap_disjoint.sparse_owner import SparseDisjointSubmapManager
from submap_disjoint.factor_owner import FactorSubmapManager
from feng_ig_adapter import FengIGCorrection as IGCorrection
from submap_disjoint.it_adapter_multires import InstanceTSDFAdapter
from submap_disjoint.online_fine_evidence import OnlineFineEvidence
from submap_disjoint.observation_refinement import ObservationRefinementAdapter

p=argparse.ArgumentParser()
for flag in ['config','dataset','instance_dir','output_dir','seq_name']:
    p.add_argument('--'+flag,required=True)
p.add_argument('--merge_selection',choices=['disjoint_pairs','baseline_components'],required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--aabb',type=int,choices=[0,1],default=0)
p.add_argument('--depth-shrink',type=int,choices=[0,1],default=0)
p.add_argument('--correction',choices=['off','detect','delete'],default='delete')
p.add_argument('--detector-config',default='feng_detector_config.json')
p.add_argument('--multires',action='store_true')
p.add_argument('--observation-refine',action='store_true',help='Refine existing instance masks using RGB-D agreement; no local TSDF')
p.add_argument('--online-fine-evidence',action='store_true',
               help='legacy compatibility flag; enabled by default for multires')
p.add_argument('--global-voxel-m',type=float,default=0.04)
p.add_argument('--fine-voxel-m',type=float,default=0.025)
p.add_argument('--small-mask-max-pixels',type=int,default=5000)
p.add_argument('--geometry-policy',choices=['hybrid'],default='hybrid',
               help='preserve IG coarse output; replace only final valid fine regions supported by coarse geometry and multiple views')
p.add_argument('--fine-max-extent-m',type=float,default=0.45,
               help='maximum robust multi-view 3-D diagonal for fine instances')
p.add_argument('--fine-min-views',type=int,default=3,
               help='minimum distinct observations before resolving fine/coarse')
p.add_argument('--fine-max-views',type=int,default=5,
               help='maximum observations used by the resolution decision')
p.add_argument('--disable-online-fine-evidence',action='store_true',
               help='keep fine geometry out of online score adjustment (ablation)')
p.add_argument('--frame-ids-file',default='',help='JSON list of original frame IDs to process')
p.add_argument('--smoke-stop',type=int,default=0)
a=p.parse_args()
if a.observation_refine and a.multires: raise ValueError("Choose observation refinement or multires, not both")
out=Path(a.output_dir)/a.seq_name; out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()): raise FileExistsError('Use a new output directory: '+str(out))
start=time.perf_counter(); random.seed(27); np.random.seed(27); torch.manual_seed(27); torch.cuda.manual_seed_all(27)
cfg=load_config(a.config)
cfg['scene']['voxel_size']=float(a.global_voxel_m)
cfg['multi_resolution']={
    'enabled':bool(a.multires), 'fine_voxel_size_m':float(a.fine_voxel_m),
    'small_mask_max_pixels':int(a.small_mask_max_pixels),
    'fine_block_count':min(int(cfg['scene']['block_count']),30000),
    'online_fine_block_count':4096, 'online_fine_max_surface_points':1500,
    'online_fine_min_surface_points':25, 'online_fine_max_observations':512,
    'online_fine_overlap_weight':0.20, 'online_fine_visibility_weight':0.10,
    'online_fine_contradiction_weight':0.15,
    'online_fine_contradiction_distance_m':0.12,
    'online_fine_contradiction_scale_m':0.20,
    'selection_rule':'multi_view_3d_extent',
    'legacy_pixel_threshold_ignored':True,
    'fine_min_views':int(a.fine_min_views),
    'fine_max_views':int(a.fine_max_views),
    'fine_max_extent_m':float(a.fine_max_extent_m),
    'fine_max_relative_spread':0.50,
    'physical_truncation_m':float(a.global_voxel_m) * float(cfg['scene'].get('trunc_voxel_multiplier',4.0)),
    'geometry_policy':a.geometry_policy,
    'min_fine_views':int(a.fine_min_views),
    'fine_replace_radius_m':0.05,
}
cfg['ig_submap']={'merge_selection':a.merge_selection,'final_merge':False,
                  'carving_aabb':{'enabled':bool(a.aabb),'strength':0.15,'band':0.10}}
cfg['sparse_candidate']={'enabled':True,'verify':False}; cfg['visibility_storage']='bool'
cfg.setdefault('seg',{})['depth_shrink']={'enabled':bool(a.depth_shrink),'front_quantile':0.20,'margin_m':0.08,'min_pixels':25,'min_keep_ratio':0.70,'remove_unknown':False}
cfg['fcgf_diagnostics']={'enabled':False,'use_in_merge':True,'export_high_similarity_pairs':False}
data=get_dataset(a.dataset,a.instance_dir,cfg,a.device)
owner_class=FactorSubmapManager if a.aabb or a.depth_shrink else SparseDisjointSubmapManager
geometry=GeometryScene(cfg,a,data,a.device); owner=owner_class(cfg,a,data,geometry,a.device)
correction=IGCorrection(owner,out/'correction',enabled=a.correction=='delete',options=json.loads(Path(a.detector_config).read_text())) if a.correction!='off' else None
it=(ObservationRefinementAdapter(cfg,owner,out/'observations') if a.observation_refine else
    InstanceTSDFAdapter(cfg,owner,out/'it_fragments') if a.multires else None)
online_enabled = bool(a.multires and not a.disable_online_fine_evidence)
online=OnlineFineEvidence(cfg,owner,out/'online_fine_evidence',enabled=online_enabled) if a.multires else None
if a.frame_ids_file:
    processed_frame_ids=[int(x) for x in json.loads(Path(a.frame_ids_file).read_text())]
else:
    processed_frame_ids=list(range(len(data)))
if not processed_frame_ids: raise ValueError('empty frame ID list')
run_metadata=dict(method='IGSubmap-ObsRefine' if a.observation_refine else 'IGSubmap-InstDel-MultiRes' if a.multires else 'IGSubmap-InstDel-global-control',
    config=cfg, baseline_config_path=a.config, baseline_config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
    mapping_frame_interval=cfg['mapping']['keyframe_freq'],segmentation_frame_interval=cfg['seg']['seg_add_interval'],
    tsdf_voxel_size_m=cfg['scene']['voxel_size'],tsdf_voxel_size_mm=cfg['scene']['voxel_size']*1000,
    multiresolution=bool(a.multires), observation_refinement=bool(a.observation_refine), online_fine_evidence=online_enabled,
    global_voxel_m=a.global_voxel_m,fine_voxel_m=a.fine_voxel_m,small_mask_max_pixels=a.small_mask_max_pixels,
    correction_mode=a.correction,correction_enabled=a.correction!='off',deletion_enabled=a.correction=='delete',
    merge_selection=a.merge_selection,final_merge=False,seed=27,visibility_storage='bool',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
    commit=(subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip() if (Path('.git').exists()) else 'working-tree'),raw_frame_count=len(data),
    processed_frame_ids=processed_frame_ids,processed_frame_count=len(processed_frame_ids),
    output_geometry='global_tsdf_observation_refined_masks' if a.observation_refine else 'global_tsdf_with_supported_fine_replacement' if a.multires else 'global_tsdf')
boundary=Path(a.dataset)/'session_boundary.json'
if not boundary.exists(): boundary=Path(a.dataset).parent/'session_boundary.json'
if boundary.exists():
    b=json.loads(boundary.read_text()); cut=b['rescan_start_frame']; run_metadata.update(reference_sampled_frames=sum(i<cut for i in processed_frame_ids),rescan_sampled_frames=sum(i>=cut for i in processed_frame_ids))
run_metadata['detector_options']=vars(correction.opt) if correction else None
(out/'run_metadata.json').write_text(json.dumps(run_metadata,indent=2))
stage_seconds = {}
def timed(stage, fn, *args, **kwargs):
    began = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        stage_seconds[stage] = stage_seconds.get(stage, 0.) + time.perf_counter() - began
# Host wall time includes synchronization performed inside each stage. GPU work
# that remains asynchronous may be charged to a later stage; no extra barriers.
profile_log = (out/'runtime_profile.jsonl').open('w', buffering=1)
trace=[]; skipped=[]; original_update=owner.update_masks
def update(fid):
    original_update(fid); trace.append({'frame':int(fid),'count':owner.c_mask_num})
owner.update_masks=update
if correction and not boundary.exists(): run_metadata.update(static_scene=True)
last_pose=torch.eye(4); integrated=0; observed=0; last_frame=-1
torch.cuda.synchronize(); loop_start=time.perf_counter()
for frame_id in tqdm(processed_frame_ids,total=len(processed_frame_ids)):
    if a.smoke_stop and frame_id>=a.smoke_stop: break
    frame_stage_start = dict(stage_seconds)
    rgb,depth,pose,seg,features,flag=timed('data_loading',data.__getitem__,frame_id); last_frame=frame_id
    valid_pose=bool(torch.isfinite(pose).all())
    if not valid_pose: pose=last_pose
    else: last_pose=pose
    if correction:
        if not correction.initialized: correction.freeze(frame_id)
        if valid_pose: correction.step(frame_id,rgb.cpu().numpy(),depth.cpu().numpy(),data.cam_intrinsic.cpu().numpy(),pose.cpu().numpy())
    if not valid_pose: continue
    if frame_id % cfg['mapping']['keyframe_freq'] != 0: continue
    with torch.no_grad():
        try: blocks,_=timed('global_tsdf',geometry.integrate_frame,frame_id,rgb,depth,torch.inverse(pose)); integrated+=1
        except RuntimeError as exc:
            if 'No block is touched in TSDF volume' not in str(exc): raise
            skipped.append(frame_id); continue
        if bool(flag) and frame_id % cfg['seg']['seg_add_interval']==0:
            timed('ig_association',owner.observe,frame_id,rgb,depth,pose,blocks,seg,features); observed+=1
            if it is not None:
                timed('observation_buffer',it.observe,frame_id,rgb,depth,pose,seg,owner.last_inserted_mask_pairs)
                if online is not None: timed('online_fine_refresh',online.refresh,it.packets)
        elif it is not None:
            timed('observation_buffer',it.observe,frame_id,rgb,depth,pose,None,[])
    if cfg['save']['ckpt_interval']>0 and frame_id>0 and frame_id%cfg['save']['ckpt_interval']==0:
        ckpt=out/f'ckpt_{frame_id}'; ckpt.mkdir(exist_ok=True); timed('periodic_checkpoint',owner.save_ckpt,frame_id,str(ckpt))
    profile_log.write(json.dumps(dict(frame=frame_id, elapsed_seconds=time.perf_counter()-loop_start,
        cumulative_stage_seconds=stage_seconds,
        frame_stage_seconds={k:v-frame_stage_start.get(k,0.) for k,v in stage_seconds.items()}))+'\n')
profile_log.close()

torch.cuda.synchronize(); loop_seconds=time.perf_counter()-loop_start
if it is None:
    owner.save_outputs(last_frame,str(out)); output_report=None
else:
    timed('ig_final_export',owner.save_outputs,last_frame,str(out/'ig_audit')); output_report=timed('observation_refinement_export' if a.observation_refine else 'fine_final_export',it.export,last_frame,out)
if correction: correction.finalize()
if online is not None: online_report=online.save()
else: online_report=None
torch.cuda.synchronize(); assert all(x['frame']%owner.merge_frame_interval==0 for x in trace)
run_metadata.update(runtime_stage_host_seconds=stage_seconds,runtime_stage_note='Host wall times; asynchronous GPU execution may be charged to later stages.',status='smoke_complete' if a.smoke_stop else 'complete',last_processed_frame=last_frame,integrated_frames=integrated,skipped_geometry_frames=skipped,segmented_frames=observed,main_loop_seconds=loop_seconds,total_seconds=time.perf_counter()-start,torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),merge_audit=owner.migration_audit,export_points=(output_report['audit']['export_points'] if output_report else len(owner.points)),online_fine_evidence_audit=online_report,correction_summary=(json.loads((out/'correction/summary.json').read_text()) if correction and (out/'correction/summary.json').exists() else None),output_geometry_report=(output_report if output_report else None))
(out/'run_metadata.json').write_text(json.dumps(run_metadata,indent=2)); (out/'merge_trace.json').write_text(json.dumps(trace,indent=2))
print('COMPLETE',json.dumps({'method':run_metadata['method'],'main_loop_seconds':loop_seconds,'total_seconds':run_metadata['total_seconds'],'merge_audit':owner.migration_audit,'correction_enabled':run_metadata['correction_enabled'],'deletion_enabled':run_metadata['deletion_enabled']}),flush=True)
