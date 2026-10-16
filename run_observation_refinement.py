"""TSDF-free consistency refinement on the unchanged global IG point cloud.

Existing identities only. Three agreeing views, no opposing view; unknown or
occluded samples abstain. Refine only a 5cm expansion of each existing support
bbox, without taking a point already labelled by another instance. Evaluation
uses GT only after all decisions and output files have been finalized.
"""
import os,sys,json,time,pathlib,shutil,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,torch,open3d as o3d
torch.set_num_threads(4);torch.Tensor.cuda=lambda self,*args,**kwargs:self
sys.path.append('third_party/FCGF')
from Dataset.dataset import get_dataset
from submap_disjoint.selective_local import observation_votes,refine_mask
p=argparse.ArgumentParser();p.add_argument('--source-batch',required=True);p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
a=p.parse_args();source=pathlib.Path(a.source_batch);seq=a.scene;root=pathlib.Path(a.output)
if root.exists() and any(root.iterdir()):raise RuntimeError('Refusing overwrite')
root.mkdir(parents=True,exist_ok=True);started=time.monotonic()
ref=next(r for r in json.loads((source/'manifest.json').read_text()) if r['seq']==seq)
base=source/'results'/seq/'ig_audit';meta=json.loads((source/'results'/seq/'run_metadata.json').read_text());cfg=meta['config']
data=get_dataset(ref['dataset'],ref['masks'],cfg,'cpu');K=data.cam_intrinsic.numpy()
pc=o3d.io.read_point_cloud(str(base/'final.ply'));xyz=np.asarray(pc.points)
with np.load(base/'ckpt_final.npz') as f:arrays={k:f[k] for k in f.files}
original=arrays['pred_masks'];result=original.copy();ids=json.loads((base/'export_instance_ids.json').read_text())
tracks={r['id']:r for r in json.loads((base/'submap_migration.json').read_text())['instances']}
packets={};changes=[];label_count=original.sum(axis=1)
for col,mid in enumerate(ids):
    observations={}
    for frame,raw in tracks[mid]['observations']:observations.setdefault(frame,[]).append(raw)
    if len(observations)<3:continue
    views=[]
    for frame,raw in sorted(observations.items()):
        if frame not in packets:
            rgb,depth,pose,seg,_,flag=data[frame];assert flag
            d=depth.numpy();scale=cfg['cam']['depth_scale'];d=(d*scale).astype(np.uint16).astype(np.float32)/scale
            packets[frame]=dict(frame=frame,depth=d,pose=pose.numpy().astype(np.float64),labels=seg.numpy())
        pkt=packets[frame];views.append((pkt,np.isin(pkt['labels'],raw)))
    target=xyz[original[:,col]]
    roi=np.all((xyz>=target.min(axis=0)-.05)&(xyz<=target.max(axis=0)+.05),axis=1)
    idx=np.flatnonzero(roi);pos,neg=observation_votes(xyz[idx],views,K)
    other=(label_count[idx]-original[idx,col])>0
    refined,audit=refine_mask(original[idx,col],other,np.ones(len(idx),bool),pos,neg)
    if not refined.any():audit=dict(removed_points=0,added_points=0,empty_fallback=True);refined=original[idx,col]
    result[idx,col]=refined;changes.append(dict(track_id=mid,views=len(views),**audit))
# Resolve any simultaneous claim to an originally unlabelled point by abstaining.
ambiguous=(label_count==0)&(result.sum(axis=1)>1);result[ambiguous]=False
dst=root/seq;dst.mkdir();shutil.copy2(base/'final.ply',dst/'final.ply')
np.savez_compressed(dst/'ckpt_final.npz',**dict(arrays,pred_masks=result));(dst/'export_instance_ids.json').write_text(json.dumps(ids))
refine_seconds=time.monotonic()-started
report=dict(scene=seq,status='export_complete',source=str(source),refinement_seconds=refine_seconds,
    geometry='byte-identical IG',new_geometry=False,new_instances=False,local_tsdf=False,selection_uses_gt=False,
    ids=ids,changes=changes,removed_memberships=int(np.count_nonzero(original&~result)),added_memberships=int(np.count_nonzero(~original&result)),
    ambiguous_additions_abstained=int(ambiguous.sum()),tolerance_m=.025,bbox_margin_m=.05,min_agreeing_views=3)
(root/'report.json').write_text(json.dumps(report,indent=2))
sys.path.insert(0,'/root/OnlineAnySegBaseline_original');sys.argv=['eval','--result_dir',str(root),'--gt_dir',str(root),'--output_file',str(root/'eval.txt')]
from eval import evaluate_seqs as ev
gtpc=pathlib.Path(ref['gtroot'])/seq/(seq+'_vh_clean_2.ply');gtfile=pathlib.Path(ref['gtseg'])/(seq+'.txt')
gt,pred=ev.assign_instances_for_scan(seq,str(dst/'ckpt_final.npz'),str(gtfile),o3d.io.read_point_cloud(str(dst/'final.ply')),o3d.io.read_point_cloud(str(gtpc)))
avg=ev.compute_averages(ev.evaluate_matches({seq:{'gt':gt,'pred':pred}}))
report.update(status='complete',metrics=dict(AP=avg['all_ap'],AP50=avg['all_ap_50%'],AP25=avg['all_ap_25%']),total_seconds=time.monotonic()-started)
(root/'report.json').write_text(json.dumps(report,indent=2));print('COMPLETE',json.dumps({k:v for k,v in report.items() if k not in ['ids','changes']}),flush=True)
