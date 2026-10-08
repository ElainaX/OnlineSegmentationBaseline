"""Matched IG mapping with strict surface memory and optional local retirement."""
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
from submap_disjoint.factor_owner import FactorSubmapManager
from submap_disjoint.surface_deletion import IGSurfaceDetector, IGSurfaceDeletion, DeletionOptions

p=argparse.ArgumentParser()
for flag in ['config','dataset','instance_dir','output_dir','seq_name']:
    p.add_argument('--'+flag,required=True)
p.add_argument('--merge_selection',choices=['disjoint_pairs','baseline_components'],required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--aabb',type=int,choices=[0,1],default=0)
p.add_argument('--depth-shrink',type=int,choices=[0,1],default=0)
p.add_argument('--deletion',choices=['off','on'],default='on')
p.add_argument('--detection',choices=['off','on'],default='on')
p.add_argument('--detection-config',default='config/ig_readonly_onlineSeg.yaml')
p.add_argument('--verify-readonly',action='store_true')
p.add_argument('--detector-schedule-json')
p.add_argument('--verify-erasure',action='store_true')
p.add_argument('--whole-instance',choices=['off','audit','on'],default='off')
p.add_argument('--instance-retirement-config',help='Optional JSON of RetirementOptions; saved in metadata')
p.add_argument('--smoke-stop',type=int,default=0)
a=p.parse_args()
out=Path(a.output_dir)/a.seq_name; out.mkdir(parents=True,exist_ok=True)
if any(out.iterdir()): raise FileExistsError(f'Use a new output directory: {out}')
start=time.perf_counter()
random.seed(27); np.random.seed(27); torch.manual_seed(27); torch.cuda.manual_seed_all(27)
cfg=load_config(a.config)
import yaml
detection_cfg=yaml.safe_load(Path(a.detection_config).read_text())['remove_detection']
cfg['remove_detection']=dict(detection_cfg,enabled=a.detection=='on')
# Representation and experiment controls; baseline semantic/numeric parameters
# are read unchanged from the archived baseline config above.
cfg['ig_submap']={'merge_selection':a.merge_selection,'final_merge':False}
cfg['ig_submap']['carving_aabb']={'enabled':bool(a.aabb),'strength':0.15,'band':0.10}
cfg['sparse_candidate']={'enabled':True,'verify':False}
cfg['visibility_storage']='bool'
cfg.setdefault('seg',{})['depth_shrink']={'enabled':bool(a.depth_shrink),
    'front_quantile':0.20,'margin_m':0.08,'min_pixels':25,
    'min_keep_ratio':0.70,'remove_unknown':False}
cfg['fcgf_diagnostics']={'enabled':False,'use_in_merge':True,'export_high_similarity_pairs':False}
data=get_dataset(a.dataset,a.instance_dir,cfg,a.device)
geometry=GeometryScene(cfg,a,data,a.device)
owner_class=FactorSubmapManager if a.aabb or a.depth_shrink else SparseDisjointSubmapManager
owner=owner_class(cfg,a,data,geometry,a.device)
assert not hasattr(geometry,'maskGraph')
detector=IGSurfaceDetector(owner,cfg,out/'surface_detection') if a.detection=='on' else None
if a.deletion=='on' and not detector: raise ValueError('Deletion requires detection')
controller=IGSurfaceDeletion(owner,detector,out/'surface_deletion',options=DeletionOptions(verify_state=a.verify_erasure),enabled=a.deletion=='on') if detector else None
whole=None
if a.whole_instance!='off':
    from submap_disjoint.instance_retirement import IGInstanceRetirement, RetirementOptions
    if detector is None: raise ValueError('Whole-instance retirement requires detection')
    if a.whole_instance=='on' and a.deletion!='on': raise ValueError('Hybrid whole-instance mode requires local deletion on')
    whole_options=json.loads(Path(a.instance_retirement_config).read_text()) if a.instance_retirement_config else {}
    whole_options['verify_state']=a.verify_erasure
    whole=IGInstanceRetirement(owner,detector,out/'instance_retirement',RetirementOptions(**whole_options),enabled=a.whole_instance=='on')
run_metadata=dict(method='IG-Submap-v1',merge_selection=a.merge_selection,config=cfg,
    baseline_config_path=a.config,baseline_config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
    mapping_frame_interval=cfg['mapping']['keyframe_freq'],segmentation_frame_interval=cfg['seg']['seg_add_interval'],
    tsdf_voxel_size_m=cfg['scene']['voxel_size'],tsdf_voxel_size_mm=cfg['scene']['voxel_size']*1000,
    final_merge=False,seed=27,visibility_storage='bool',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),raw_frame_count=len(data))
selected=list(range(0,len(data),cfg['mapping']['keyframe_freq']))
run_metadata['selected_frame_ids']=selected
detector_frames=json.loads(Path(a.detector_schedule_json).read_text())[a.seq_name] if a.detector_schedule_json else selected
if detector_frames!=sorted(set(detector_frames)) or not all(type(f) is int and 0<=f<len(data) for f in detector_frames):raise ValueError('Invalid detector schedule')
detector_set=set(detector_frames)
run_metadata['detector_frame_ids']=detector_frames
boundary=Path(a.dataset)/'session_boundary.json'
if not boundary.exists(): boundary=Path(a.dataset).parent/'session_boundary.json'
cut=None
if boundary.exists():
    b=json.loads(boundary.read_text()); cut=b['rescan_start_frame']
    run_metadata.update(reference_sampled_frames=sum(i<cut for i in selected),rescan_sampled_frames=sum(i>=cut for i in selected))
run_metadata.update(detection_mode=a.detection,deletion_enabled=a.deletion=='on',smoke_stop=a.smoke_stop,
    whole_instance_mode=a.whole_instance,
    detection_config_path=a.detection_config,
    detector_provenance={'surface_bank_sha256':hashlib.sha256(Path('submap_disjoint/surface_bank.py').read_bytes()).hexdigest(),'implementation':'strict surface bank; original confirmation unchanged'},
    detection_input_shape=[cfg['cam']['img_h'],cfg['cam']['img_w']],
    detection_schedule='pre-fusion; independent explicit detector schedule; mapping/segmentation intervals unchanged',
    protocol='Strict surface memory + lineage-bounded local surface retirement. GT/session labels are not passed to detector or deletion.')
(out/'run_metadata.json').write_text(json.dumps(run_metadata,indent=2))
trace=[]; original_update=owner.update_masks
def update(fid):
    original_update(fid)
    trace.append(dict(frame=int(fid),count=owner.c_mask_num))
owner.update_masks=update
last_pose=torch.eye(4); integrated=0; observed=0; last_frame=-1
torch.cuda.synchronize(); loop_start=time.perf_counter()
for frame_id,(rgb,depth,pose,seg,features,flag) in tqdm(enumerate(DataLoader(data,batch_size=1,num_workers=0))):
    if a.smoke_stop and frame_id>=a.smoke_stop: break
    last_frame=frame_id
    valid_pose=bool(torch.isfinite(pose[0]).all())
    if not valid_pose: pose[0]=last_pose
    else: last_pose=pose[0]
    if flag[0]: seg=seg[0]; features=features[0]
    if detector and frame_id in detector_set:
        with torch.no_grad():
            if detector.step(frame_id,rgb[0],depth[0],pose[0],valid_pose):
                if whole: whole.step(frame_id,depth[0],pose[0])
                controller.step(frame_id,depth[0],pose[0])
    if frame_id % cfg['mapping']['keyframe_freq']==0:
        with torch.no_grad():
            blocks,_=geometry.integrate_frame(frame_id,rgb[0],depth[0],torch.inverse(pose[0])); integrated+=1
            if flag[0] and frame_id % cfg['seg']['seg_add_interval']==0:
                owner.observe(frame_id,rgb[0],depth[0],pose[0],blocks,seg,features); observed+=1
    if cfg['save']['ckpt_interval']>0 and frame_id>0 and frame_id%cfg['save']['ckpt_interval']==0:
        ckpt=out/f'ckpt_{frame_id}'; ckpt.mkdir(exist_ok=True)
        owner.save_ckpt(frame_id,str(ckpt))
torch.cuda.synchronize(); loop_seconds=time.perf_counter()-loop_start
owner.save_outputs(last_frame,str(out))
detector_report=detector.finalize() if detector else None
deletion_report=controller.save() if controller else None
whole_report=whole.save() if whole else None
torch.cuda.synchronize()
assert all(x['frame']%owner.merge_frame_interval==0 for x in trace),'unexpected off-schedule merge'
run_metadata.update(status='smoke_complete' if a.smoke_stop else 'complete',last_processed_frame=last_frame,integrated_frames=integrated,segmented_frames=observed,
                    main_loop_seconds=loop_seconds,total_seconds=time.perf_counter()-start,
                    torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    merge_audit=owner.migration_audit,export_points=len(owner.points),
                    surface_detection_seconds=detector.seconds if detector else 0.,
                    instance_retirement_summary=whole_report,
                    deletion_summary=deletion_report)
(out/'run_metadata.json').write_text(json.dumps(run_metadata,indent=2))
(out/'merge_trace.json').write_text(json.dumps(trace,indent=2))
print('COMPLETE',json.dumps({k:run_metadata[k] for k in ['merge_selection','main_loop_seconds','total_seconds','merge_audit']}),flush=True)
