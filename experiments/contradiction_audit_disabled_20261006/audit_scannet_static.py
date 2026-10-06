#!/usr/bin/env python3
"""Static ScanNet contradiction statistics; TSDF map is unchanged by auditing."""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, open3d as o3d, open3d.core as o3c
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO)); sys.path.insert(0,str(REPO/'experiments/contradiction_audit'))
from audit_3rscan import AuditDataset, raycast
from voxelized_points import VoxelBlockGrid
import tool.config as config

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--seq',required=True); ap.add_argument('--output-root',required=True); ap.add_argument('--config',default=str(REPO/'config/scannet_stride10_controlled.yaml')); ap.add_argument('--audit-interval',type=int,default=10); ap.add_argument('--tau-min',type=float,default=.035); ap.add_argument('--tau-rel',type=float,default=.04); ap.add_argument('--ratio-threshold',type=float,default=.05); ap.add_argument('--min-pixels',type=int,default=400); args=ap.parse_args()
 ds=AuditDataset(Path(args.data_root)/args.seq); cfg=config.load_config(args.config); wrap=VoxelBlockGrid(cfg,ds,'cuda:0'); grid=wrap.voxel_block_grids; K_o3c=o3c.Tensor.from_numpy(ds.K.astype(np.float64)); dev=o3c.Device('CUDA:0'); depth_scale=float(cfg['cam']['depth_scale']); depth_far=float(cfg['cam']['depth_far']); trunc=float(cfg['scene'].get('trunc_voxel_multiplier',4.))
 out=Path(args.output_root)/args.seq; out.mkdir(parents=True,exist_ok=True); frames=[]; last_pose=np.eye(4,dtype=np.float32)
 for n,fid in enumerate(ds.ids):
  depth=cv2.imread(str(ds.depths[fid]),cv2.IMREAD_UNCHANGED); color=cv2.imread(str(ds.colors[fid]),cv2.IMREAD_COLOR); pose=ds.poses[fid];
  if not np.isfinite(pose).all(): pose=last_pose
  else: last_pose=pose
  pw=np.linalg.inv(pose); stat=None
  if n % max(1,args.audit_interval)==0:
   pred,_=raycast(grid,K_o3c,pw,depth,ds.target_w,ds.target_h,depth_scale,depth_far,trunc); obs=depth.astype(np.float32)/depth_scale; both=(obs>0)&(pred>0); tau=np.maximum(args.tau_min,args.tau_rel*np.maximum(pred,.1)); free=both&(obs>pred+tau); fg=both&(obs<pred-tau); con=int(np.count_nonzero(free)+np.count_nonzero(fg)); stat={'frame':int(fid),'both_pixels':int(np.count_nonzero(both)),'observed_pixels':int(np.count_nonzero(obs>0)),'raycast_pixels':int(np.count_nonzero(pred>0)),'free_pixels':int(np.count_nonzero(free)),'foreground_pixels':int(np.count_nonzero(fg)),'occluded_pixels':int(np.count_nonzero(fg)),'free_ratio':float(np.count_nonzero(free)/max(1,np.count_nonzero(both))),'occluded_ratio':float(np.count_nonzero(fg)/max(1,np.count_nonzero(both))),'contradiction_pixels':int(np.count_nonzero(free)),'contradiction_ratio':float(np.count_nonzero(free)/max(1,np.count_nonzero(both))),'trigger':bool(np.count_nonzero(free)>=args.min_pixels and np.count_nonzero(free)/max(1,np.count_nonzero(both))>=args.ratio_threshold)}; frames.append(stat)
  # Integrate every frame, including frames not audited.
  rgb=cv2.cvtColor(color,cv2.COLOR_BGR2RGB); depth_img=o3d.t.geometry.Image(np.ascontiguousarray(depth)).to(o3c.uint16).to(dev); color_img=o3d.t.geometry.Image(np.ascontiguousarray(rgb)).to(o3c.uint8).to(dev); ext=o3c.Tensor.from_numpy(np.ascontiguousarray(pw.astype(np.float64))); fc=grid.compute_unique_block_coordinates(depth_img,K_o3c,ext,depth_scale,depth_far); grid.integrate(fc,depth_img,color_img,K_o3c,ext,depth_scale,depth_far,trunc_voxel_multiplier=trunc)
  if n%250==0: print(f'{args.seq}: {n+1}/{len(ds.ids)} frame={fid} blocks={grid.hashmap().size()}',flush=True)
 vals=np.asarray([x['contradiction_ratio'] for x in frames],dtype=float); runs=[]; cur=0
 for x in frames:
  cur=cur+1 if x['trigger'] else 0; runs.append(cur)
 summary={'sequence':args.seq,'frames_total':len(ds.ids),'audited_frames':len(frames),'audit_interval':args.audit_interval,'parameters':vars(args),'metrics':{'trigger_frames':int(sum(x['trigger'] for x in frames)),'trigger_rate':float(sum(x['trigger'] for x in frames)/max(1,len(frames))),'mean_free_ratio':float(vals.mean()),'median_free_ratio':float(np.median(vals)),'p90_free_ratio':float(np.percentile(vals,90)),'p95_free_ratio':float(np.percentile(vals,95)),'max_free_ratio':float(vals.max()),'mean_occluded_ratio':float(np.mean([s['occluded_ratio'] for s in frames])),'median_occluded_ratio':float(np.median([s['occluded_ratio'] for s in frames])),'max_consecutive_trigger_frames':int(max(runs) if runs else 0)},'frames':frames}; json.dump(summary,open(out/'static_contradiction_summary.json','w'),indent=2); print(json.dumps(summary['metrics'],indent=2),flush=True)
if __name__=='__main__': main()