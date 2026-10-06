"""Locate coverage loss with one fixed GT-to-full-reconstruction mapping."""
import sys, json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/output/independent_repair_20261007')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
SEQ='pair_04_752cc595_55551083'
OUT=ROOT/'audit'/SEQ
z=np.load(OUT/'ckpt_585/export_stage_audit.npz')
pc=o3d.io.read_point_cloud(str(OUT/'final.ply'))
gt=o3d.io.read_point_cloud(str(DATA/SEQ/(SEQ+'_vh_clean_2.ply')))
labels=np.loadtxt(DATA/'gt_seg'/(SEQ+'.txt'),dtype=np.int64)
points=np.asarray(pc.points)
original_gt=np.asarray(gt.points).copy()
fit=o3d.pipelines.registration.registration_icp(gt,pc,.05,np.eye(4),
      o3d.pipelines.registration.TransformationEstimationPointToPoint())
gt.transform(fit.transformation); gtpoints=np.asarray(gt.points)
dist,indices=cKDTree(points).query(gtpoints,workers=4)
masks={k:np.unpackbits(z[k],axis=1,count=int(z['point_count'])).astype(bool)
       for k in ['all_tracks','before_dbscan','after_dbscan']}
ids=[int(x) for x in np.unique(labels) if x>=1000 and (labels==x).sum()>=100]
target=np.isin(labels,[x for x in ids if x%1000 in (12,16)])
q=gtpoints[target]; original_q=original_gt[target]
raw_distance=np.full(len(q),np.inf); raw_distance_native=raw_distance.copy()
K=np.loadtxt(DATA/SEQ/'intrinsic/intrinsic_depth.txt')[:3,:3]
raw_count=0
for frame in range(586):
    depth=cv2.imread(str(DATA/SEQ/'depth'/('%d.png'%frame)),-1).astype(float)/1000
    pose=np.loadtxt(DATA/SEQ/'pose'/('%d.txt'%frame))
    v,u=np.indices(depth.shape); ok=(depth>.1)&(depth<5.)
    d=depth[ok]
    p=np.column_stack(((u[ok]-K[0,2])*d/K[0,0],(v[ok]-K[1,2])*d/K[1,1],d))
    p=p@pose[:3,:3].T+pose[:3,3]
    if len(p):
        tree=cKDTree(p)
        raw_distance=np.minimum(raw_distance,tree.query(q,workers=4)[0])
        raw_distance_native=np.minimum(raw_distance_native,tree.query(original_q,workers=4)[0])
        raw_count+=len(p)
    if frame%100==0: print('raw depth audit',frame,flush=True)
rows=[]
for label in ids:
    sel=labels==label; n=int(sel.sum()); per={}
    for threshold in (.025,.05,.10,.15):
        valid=dist<=threshold
        stage={}
        for name,mask in masks.items():
            mapped=mask[:,indices] & valid[None,:]
            intersections=mapped[:,sel].sum(1)
            union=int(mapped[:,sel].any(0).sum()) if len(mapped) else 0
            size=mapped.sum(1)
            iou=intersections/np.maximum(1,size+n-intersections)
            order=np.argsort(iou)[::-1][:5]
            stage[name]={'union_recall':union/n,
               'max_single_recall':float(intersections.max(initial=0)/n),
               'best_iou':float(iou.max(initial=0)),
               'top_predictions':[{'index':int(i),'intersection':int(intersections[i]),
                   'recall':float(intersections[i]/n),'purity':float(intersections[i]/max(1,size[i])),
                   'iou':float(iou[i]),'track_id':int(z['all_ids'][i]) if name=='all_tracks' else None}
                   for i in order if intersections[i]>0]}
        per[str(threshold)]={'full_geometry_recall':float(valid[sel].mean()),'stages':stage}
        if label%1000 in (12,16):
            subset=labels[target]==label
            per[str(threshold)]['raw_depth_recall']=float((raw_distance[subset]<=threshold).mean())
            per[str(threshold)]['raw_depth_recall_no_icp']=float((raw_distance_native[subset]<=threshold).mean())
    rows.append({'object_id':label%1000,'vertices':n,'by_distance_m':per})
report={'mapping':'GT -> full geometry nearest point; SAME mapping across export stages; diagnostic, not official AP mapping',
        'T_recon_gt':fit.transformation.tolist(),'raw_depth_frames':586,'raw_valid_depth_points':raw_count,
        'raw_depth_scope':'objects 12 and 16 only, all frames, .1m < depth < 5m',
        'stage_mask_counts':{k:len(v) for k,v in masks.items()},'instances':rows}
(ROOT/'pair04_coverage_diagnosis.json').write_text(json.dumps(report,indent=2))
sys.path.insert(0,'/root/OnlineSegmentationBaseline_baseline_ablation')
sys.argv=[sys.argv[0],'--result_dir',str(OUT),'--gt_dir',str(DATA),'--output_file',str(ROOT/'audit_AP.txt')]
from eval import evaluate_seqs as ev
np.random.seed(27)
g,p=ev.assign_instances_for_scan(SEQ,str(OUT/'ckpt_final.npz'),str(DATA/'gt_seg'/(SEQ+'.txt')),
       o3d.io.read_point_cloud(str(OUT/'final.ply')),
       o3d.io.read_point_cloud(str(DATA/SEQ/(SEQ+'_vh_clean_2.ply'))),save_seg_gt=False)
scores=ev.compute_averages(ev.evaluate_matches({'audit':dict(gt=g,pred=p)}))
report['official_AP']={k:float(scores[k]) for k in ('all_ap','all_ap_50%','all_ap_25%')}
(ROOT/'pair04_coverage_diagnosis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report['official_AP']),flush=True)
for r in rows:
    if r['object_id'] in (12,16): print(json.dumps(r),flush=True)
