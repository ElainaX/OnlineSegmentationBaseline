"""Re-use local replay surfaces; evaluate masks on unchanged IG geometry."""
import os,sys,json,time,pathlib,shutil
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='4';os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np,torch,open3d as o3d
from scipy.spatial import cKDTree
torch.set_num_threads(4);torch.Tensor.cuda=lambda self,*args,**kwargs:self
sys.path.append('third_party/FCGF')
from Dataset.dataset import get_dataset
from submap_disjoint.selective_local import observation_votes,refine_mask
root=pathlib.Path('/root/autodl-tmp/multires_selective2_20261013');source=pathlib.Path('/root/autodl-tmp/multires_extra2_20261013')
sys.path.insert(0,'/root/OnlineAnySegBaseline_original');sys.argv=['eval','--result_dir',str(root),'--gt_dir',str(root),'--output_file',str(root/'mask_eval.txt')]
from eval import evaluate_seqs as ev
refs=json.loads((source/'manifest.json').read_text());summaries=[]
for ref in refs:
    seq=ref['seq'];started=time.monotonic();base=source/'results'/seq/'ig_audit';output=root/seq
    meta=json.loads((source/'results'/seq/'run_metadata.json').read_text());cfg=meta['config'];data=get_dataset(ref['dataset'],ref['masks'],cfg,'cpu');K=data.cam_intrinsic.numpy()
    pc=o3d.io.read_point_cloud(str(base/'final.ply'));xyz=np.asarray(pc.points)
    with np.load(base/'ckpt_final.npz') as f:arrays={k:f[k] for k in f.files}
    original=arrays['pred_masks'];ids=json.loads((base/'export_instance_ids.json').read_text());tracks={r['id']:r for r in json.loads((base/'submap_migration.json').read_text())['instances']}
    selected=json.loads((output/'selection.json').read_text());changes=[];packets={}
    selected_ids={r['track_id'] for r in selected if r['choice']!='ig'}
    models={}
    for kind in ['local25','local20']:
        p=output/kind/seq;models[kind]=(np.asarray(o3d.io.read_point_cloud(str(p/'final.ply')).points),np.load(p/'ckpt_final.npz')['pred_masks'])
    masks={kind:original.copy() for kind in ['mask25','mask20']}
    for col,mid in enumerate(ids):
        if mid not in selected_ids:continue
        observations={}
        for frame,raw in tracks[mid]['observations']:observations.setdefault(frame,[]).append(raw)
        views=[]
        for frame,raw in sorted(observations.items()):
            if frame not in packets:
                rgb,depth,pose,seg,_,flag=data[frame];assert flag
                d=depth.numpy();scale=cfg['cam']['depth_scale'];d=(d*scale).astype(np.uint16).astype(np.float32)/scale
                packets[frame]=dict(frame=frame,depth=d,pose=pose.numpy().astype(np.float64),labels=seg.numpy())
            p=packets[frame];views.append((p,np.isin(p['labels'],raw)))
        points=models['local25'][0][models['local25'][1][:,col]]
        # Restrict work to the union of existing target support and local bbox.
        roi=original[:,col]|np.all((xyz>=points.min(axis=0)-.025)&(xyz<=points.max(axis=0)+.025),axis=1)
        idx=np.flatnonzero(roi);pos,neg=observation_votes(xyz[idx],views,K)
        occupied_other=np.any(original[idx][:,np.arange(len(ids))!=col],axis=1)
        for kind,model in [('mask25','local25'),('mask20','local20')]:
            pts,mm=models[model];near=cKDTree(pts[mm[:,col]]).query(xyz[idx],workers=1)[0]<=.025
            refined,audit=refine_mask(original[idx,col],occupied_other,near,pos,neg)
            # An empty prediction is never published as an instance.
            if not refined.any() and not original[~roi,col].any():
                audit=dict(removed_points=0,added_points=0,empty_fallback=True);refined=original[idx,col]
            masks[kind][idx,col]=refined;changes.append(dict(kind=kind,track_id=mid,**audit))
    construction=time.monotonic()-started;results={}
    for kind,mask in masks.items():
        dst=output/kind/seq;dst.mkdir(parents=True,exist_ok=False);shutil.copy2(base/'final.ply',dst/'final.ply')
        np.savez_compressed(dst/'ckpt_final.npz',**dict(arrays,pred_masks=mask));(dst/'export_instance_ids.json').write_text(json.dumps(ids))
        gtpc=pathlib.Path(ref['gtroot'])/seq/(seq+'_vh_clean_2.ply');gtfile=pathlib.Path(ref['gtseg'])/(seq+'.txt')
        gt,pred=ev.assign_instances_for_scan(seq,str(dst/'ckpt_final.npz'),str(gtfile),o3d.io.read_point_cloud(str(dst/'final.ply')),o3d.io.read_point_cloud(str(gtpc)))
        avg=ev.compute_averages(ev.evaluate_matches({seq:{'gt':gt,'pred':pred}}));results[kind]=dict(AP=avg['all_ap'],AP50=avg['all_ap_50%'],AP25=avg['all_ap_25%'])
    summary=dict(scene=seq,results=results,changes=changes,refinement_seconds=construction,total_seconds=time.monotonic()-started,geometry='byte-identical IG',new_geometry=False,new_instances=False,selection_uses_gt=False)
    summaries.append(summary);(root/'mask_refinement_results.json').write_text(json.dumps(summaries,indent=2));print('RESULT',json.dumps({k:v for k,v in summary.items() if k!='changes'}),flush=True)
