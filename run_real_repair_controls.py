"""GT-assisted representation test on reference/rescan, NOT an online detector.

GT-nearest assignments are depth-surface annotations, not exact pixel masks.
Unannotatable reference depth is excluded equally from every method.
"""
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from run_repair_controls import backproject, compact, run_case, distance
from submap_disjoint.independent_tsdf import TSDFOptions

DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')


def confirmed_free(points, after, margin=.10, required=2):
    votes=np.zeros(len(points),np.int32)
    for f in after:
        pc=(points-f['pose'][:3,3])@f['pose'][:3,:3]
        good=pc[:,2]>.1
        uv=pc[:,:2]/np.maximum(pc[:,2,None],1e-6)
        u=np.rint(uv[:,0]*f['K'][0,0]+f['K'][0,2]).astype(int)
        v=np.rint(uv[:,1]*f['K'][1,1]+f['K'][1,2]).astype(int)
        h,w=f['depth'].shape
        good &= (u>=2)&(u<w-2)&(v>=2)&(v<h-2)
        ids=np.flatnonzero(good)
        # Conservative 3x3 minimum: reject holes and nearby foreground edges.
        minimum=cv2.erode(f['depth'],np.ones((3,3),np.uint8))
        maximum=cv2.dilate(f['depth'],np.ones((3,3),np.uint8))
        d=minimum[v[ids],u[ids]]
        stable=(maximum[v[ids],u[ids]]-d)<.10
        votes[ids]+=(stable&(d>.1)&(pc[ids,2]+margin<d))
    return votes>=required


def prepare(seq, stride):
    root=DATA/seq
    boundary=json.loads((root/'session_boundary.json').read_text())
    # Dataset convention: frame_index explicitly names the reference/rescan origin.
    report=json.loads((root/'gt_report.json').read_text())
    K=np.loadtxt(root/'intrinsic/intrinsic_depth.txt')[:3,:3]
    posefiles=sorted((root/'pose').glob('*.txt'),key=lambda p:int(p.stem))
    splits={'pair_01_0cac767c_19f1a892':170,'pair_04_752cc595_55551083':408}
    split=splits[seq]
    gtp=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)%1000
    tree=cKDTree(gtp)
    candidates={int(x['reference']['object_id']) for x in report['changed_instances'].get('rigid',[])}
    candidates|={int(x['reference']['object_id']) for x in report['changed_instances'].get('removed',[])}
    changed=set(candidates)
    for entries in report['changed_instances'].values():
        for x in entries:
            if 'reference' in x: changed.add(int(x['reference']['object_id']))
    before=[]; after=[]; observed={k:[] for k in candidates}; static=[]
    total=accepted=0
    for path in posefiles:
        frame=int(path.stem)
        if frame%stride: continue
        depth=cv2.imread(str(root/'depth'/('%d.png'%frame)),-1).astype(np.float32)/1000
        depth[(depth<=.1)|(depth>=5.)]=0
        rgb=cv2.cvtColor(cv2.imread(str(root/'color'/('%d.jpg'%frame))),cv2.COLOR_BGR2RGB)
        if rgb.shape[:2]!=depth.shape:
            rgb=cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_LINEAR)
        pose=np.loadtxt(path)
        f={'frame':frame,'rgb':rgb,'depth':depth,'K':K,'pose':pose,
           'source':{'rgb':str(root/'color'/('%d.jpg'%frame)),
                     'depth':str(root/'depth'/('%d.png'%frame)), 'pose':str(path)}}
        if frame<split:
            p=backproject(depth,K,pose)
            d,idx=tree.query(p,workers=4)
            lab=labels[idx]; ok=d<=.075
            total+=len(p); accepted+=int(ok.sum())
            labelimg=np.full(depth.shape,-1,np.int32)
            valid=depth>0; labelimg[valid]=np.where(ok,lab,-1)
            f['depth']=np.where(labelimg>=0,depth,0).astype(np.float32)
            f['labels']=labelimg
            before.append(f)
            for k in candidates:
                q=p[ok&(lab==k)]
                if len(q): observed[k].append(q)
            q=p[ok&~np.isin(lab,list(changed))]
            static.append(compact(q,.025))
        else:
            f['mask']=np.zeros(depth.shape,bool)
            after.append(f)
    ranks=[]; probes={}
    for k,obs in observed.items():
        if not obs: continue
        p=compact(np.concatenate(obs),.025)
        free=p[confirmed_free(p,after)]
        probes[k]=free
        ranks.append({'object_id':k,'observed_surface_probes':len(p),'confirmed_free_probes':len(free)})
    ranks.sort(key=lambda x:x['confirmed_free_probes'],reverse=True)
    if not ranks or ranks[0]['confirmed_free_probes']<20:
        raise RuntimeError('insufficient measurable disappearing surface')
    target=ranks[0]['object_id']
    for f in before:
        f['mask']=f.pop('labels')==target
        f['source']['mask_recipe']={'kind':'reference_gt_nearest_surface','object_id':target,'max_distance_m':.075}
    print(seq,'oracle target',target,'candidate ranks',ranks,flush=True)
    meta={'data':'3RScan','sequence':seq,'stride':stride,'reference_end_exclusive':split,
          'oracle_target_object_id':target,'target_selection':'most GT-annotated changed surface confirmed in later free space; diagnostic selection, not a deployable detector',
          'candidate_counts':ranks,'reference_depth_valid_pixels':total,
          'reference_depth_accepted_pixels':accepted,'reference_depth_acceptance':accepted/max(total,1),
          'oracle_pixel_annotation':'nearest reference GT surface within 75mm; approximation, unknown pixels excluded in all methods; RGB resized to depth resolution',
          'free_space_metric':'old target probes, 2 or more later 3x3 depth-consistent views, >100mm in front of measured depth',
          'free_space_stale_probes':probes[target],
          'static_probes':compact(np.concatenate(static),.025)}
    return before,after,meta


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); p.add_argument('--stride',type=int,default=2)
    p.add_argument('--seq',required=True); a=p.parse_args()
    before,after,meta=prepare(a.seq,a.stride)
    run_case(before,after,Path(a.out)/a.seq,TSDFOptions(device='CUDA:0'),meta)
