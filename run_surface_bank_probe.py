"""Raw-sensor candidate-entry ablation; no instance/map changes and no GT input."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from submap_disjoint.surface_bank import SurfaceBank
from submap_disjoint.readonly_detection_core import interior_mask
from submap_disjoint.detection_alignment import refine_detection_pose

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True)
parser.add_argument('--require-color',action='store_true')
parser.add_argument('--view-gate',choices=['camera','ray'],default='camera')
parser.add_argument('--refine-pose',action='store_true')
parser.add_argument('--surface-support',action='store_true')
parser.add_argument('--strict-history-support',action='store_true')
args=parser.parse_args()
OUT=Path(args.output_dir);OUT.mkdir(exist_ok=False)
source=Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
schedules={r['seq']:r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text())}
scenes=[]
for run in json.loads((source/'selected_results.json').read_text()):
    seq=run['seq'];cmd=run['command'];target=OUT/seq;target.mkdir()
    data=Path(cmd[cmd.index('--dataset')+1]);prior=Path(cmd[cmd.index('--output_dir')+1])/seq
    cfg_file=prior/'remove_detection/config.json'
    config=json.loads(cfg_file.read_text());cfg=config['effective_settings']
    cam=config['pipeline_config']['cam'];assert cam['h_crop']==cam['w_crop']==0
    k=np.loadtxt(data/'intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32)
    bank=SurfaceBank(config['pipeline_config']['scene']['voxel_size'],cfg,require_color=args.require_color,view_gate=args.view_gate,surface_support=args.surface_support,strict_history_support=args.strict_history_support)
    selected=schedules[seq]['sampling']['selected_frame_ids'];frames=[];start=time.perf_counter();poses=[];original_poses=[]
    for j,f in enumerate(selected):
        raw=cv2.imread(str(data/'depth'/f'{f}.png'),cv2.IMREAD_UNCHANGED)
        if raw is None:raise FileNotFoundError(data/'depth'/f'{f}.png')
        depth=raw.astype(np.float32)/cam['depth_scale']
        assert depth.shape==(cam['img_h'],cam['img_w'])
        valid=np.isfinite(depth)&(depth>cam['depth_near'])&(depth<cam['depth_far'])
        depth[~valid]=0
        inner=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
        pose=np.loadtxt(data/'pose'/f'{f}.txt').astype(np.float32)
        original_poses.append(pose.copy())
        align_start=time.perf_counter();alignment=dict(accepted=False,reason='disabled')
        if args.refine_pose:
            # Anchors predate this frame, have repeated support and no pending
            # disappearance. Spatial decimation limits tree construction cost.
            anchors=bank.points[(bank.support>=3)&(bank.streak==0)]
            if len(anchors):
                _,unique=np.unique(np.floor(anchors/.05).astype(np.int64),axis=0,return_index=True)
                anchors=anchors[unique]
                if len(anchors)>50000:anchors=anchors[np.linspace(0,len(anchors)-1,50000,dtype=int)]
            yy,xx=np.indices(depth.shape);y,x=np.nonzero(inner&(yy%4==0)&(xx%4==0))
            camera=(np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(k).T)*depth[y,x,None]
            cloud=camera@pose[:3,:3].T+pose[:3,3]
            pose,alignment=refine_detection_pose(cloud,anchors,pose)
        alignment['seconds']=time.perf_counter()-align_start
        poses.append(pose.copy())
        rgb=None
        if args.require_color:
            color=cv2.imread(str(data/'color'/f'{f}.jpg'))
            if color is None:raise FileNotFoundError(data/'color'/f'{f}.jpg')
            rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
            if rgb.shape[:2]!=depth.shape:rgb=cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)
        t=time.perf_counter();row=bank.step(f,pose,k,depth,inner,cam['depth_near'],cam['depth_far'],rgb)
        row['seconds']=time.perf_counter()-t;row['alignment']=alignment;frames.append(row)
        if j%100==0:print(seq,j,len(selected),'points',len(bank.points),flush=True)
    arrays=bank.arrays();np.savez_compressed(target/'bank.npz',**arrays)
    np.savez_compressed(target/'poses.npz',frames=selected,original_poses=original_poses,detector_poses=poses)
    report=dict(seq=seq,frames=frames,selected_frame_ids=selected,voxel_size=bank.voxel_size,
        cfg=cfg,seed_stride=bank.seed_stride,require_color=args.require_color,view_gate=args.view_gate,refine_pose=args.refine_pose,surface_support=args.surface_support,strict_history_support=args.strict_history_support,wall_seconds=time.perf_counter()-start,
        array_memory_bytes=sum(a.nbytes for a in arrays.values()),GT_used=False,actual_deletions=0,
        scope='experimental raw-depth local change proposals; not whole-instance deletion',
        source_config=str(cfg_file))
    (target/'summary.json').write_text(json.dumps(report,indent=2))
    scenes.append({k:report[k] for k in ['seq','wall_seconds','array_memory_bytes','voxel_size']})
    print(seq,'done',report['wall_seconds'],'active',int(bank.active.sum()),flush=True)
(OUT/'completed.json').write_text(json.dumps(dict(completed=True,scenes=scenes,GT_used=False),indent=2))
