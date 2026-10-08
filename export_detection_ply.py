"""Post-hoc PLY visualization. Does not mutate experiments or use GT online."""
from pathlib import Path
import json, colorsys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/autodl-tmp/output/ig_deletion_20261008')
OUT=ROOT/'visualization';OUT.mkdir(exist_ok=True)
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
STATUS={'cancelled':1,'suspected':2,'would_remove':3}
COLORS={1:[35,110,255],2:[255,170,0],3:[255,25,40]}


def write(path,points,colors,**fields):
    n=len(points)
    dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]
    dtype += [(k,'<i4') for k in fields]
    a=np.empty(n,dtype=dtype)
    for i,k in enumerate(['x','y','z']):a[k]=points[:,i]
    for i,k in enumerate(['red','green','blue']):a[k]=np.asarray(colors)[:,i]
    for k,v in fields.items():a[k]=v
    header=['ply','format binary_little_endian 1.0',f'element vertex {n}',
        'property float x','property float y','property float z',
        'property uchar red','property uchar green','property uchar blue']
    header += ['property int '+k for k in fields]
    header += ['end_header','']
    with path.open('wb') as f:f.write('\n'.join(header).encode());a.tofile(f)
    # Validate exported point count and standard RGB interoperability.
    check=o3d.io.read_point_cloud(str(path))
    assert len(check.points)==n and check.has_colors()


def xyz(path):return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def palette(n):
    return np.array([np.array(colorsys.hsv_to_rgb((i*.61803398875)%1,.40,.85))*255 for i in range(n)])


def preview(path,points,colors,title,focus=None):
    fig=plt.figure(figsize=(13,7),facecolor='white')
    for i,(el,az) in enumerate([(62,-65),(18,-65)]):
        ax=fig.add_subplot(1,2,i+1,projection='3d')
        ids=np.arange(len(points))
        if focus is not None:
            lo,hi=focus.min(0)-.4,focus.max(0)+.4
            ids=ids[((points>=lo)&(points<=hi)).all(1)]
        if len(ids)>90000:ids=ids[np.linspace(0,len(ids)-1,90000).astype(int)]
        p=points[ids];ax.scatter(p[:,0],p[:,1],p[:,2],c=np.asarray(colors)[ids]/255.,s=.5,depthshade=False,rasterized=True)
        ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),.1));ax.view_init(el,az)
        ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)');ax.set_zlabel('Z (m)')
    fig.suptitle(title,fontsize=12);fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)


manifest=[]
for row in json.loads((ROOT/'selected_results.json').read_text()):
    if row['variant']!='on':continue
    seq=row['seq'];short='pair'+seq.split('_')[1]
    cmd=row['command'];folder=Path(cmd[cmd.index('--output_dir')+1])/seq
    points=xyz(folder/'final.ply')
    ckpt=np.load(folder/'ckpt_final.npz');masks=ckpt['pred_masks']
    assert masks.shape[0]==len(points)
    # Resolve overlapping final predictions for display ONLY. AP masks unchanged.
    order=np.argsort(-ckpt['pred_score'],kind='stable')
    instance=np.full(len(points),-1,np.int32)
    for i in order:instance[(instance<0)&masks[:,i]]=int(i)
    color=np.full((len(points),3),[185,185,185],np.uint8)
    valid=instance>=0;color[valid]=palette(masks.shape[1])[instance[valid]].astype(np.uint8)
    base_color=color.copy()
    summary=json.loads((folder/'remove_detection/summary.json').read_text())
    records=[]
    for m in summary['instances']:
        d=folder/'remove_detection/candidates'/f"{m['candidate_id']:04d}"
        paths=sorted(d.glob('*.npz'))
        if not paths:continue
        f=paths[-1];p=np.load(f)
        records.append(dict(meta=m,points=p['reference_points'],evidence=p['evidence'],
            source=p['contradiction_source'],path=str(f)))
    status=np.zeros(len(points),np.int32);cid=np.full(len(points),-1,np.int32)
    # Proximity is only a rendering annotation, never a final identity lookup.
    for state in [1,2,3]:
        selected=[r for r in records if STATUS[r['meta']['state']]==state]
        if not selected:continue
        ref=np.concatenate([r['points'] for r in selected])
        labels=np.concatenate([np.full(len(r['points']),r['meta']['candidate_id']) for r in selected])
        dist,near=cKDTree(ref).query(points)
        match=dist<=.035
        status[match]=state;cid[match]=labels[near[match]];color[match]=COLORS[state]
    # Append frozen references explicitly: deleted geometry must remain visible.
    ref=np.concatenate([r['points'] for r in records])
    rstatus=np.concatenate([np.full(len(r['points']),STATUS[r['meta']['state']]) for r in records])
    rcid=np.concatenate([np.full(len(r['points']),r['meta']['candidate_id']) for r in records])
    full=np.concatenate([points,ref]);fullcolors=np.concatenate([color,np.array([COLORS[int(s)] for s in rstatus],np.uint8)])
    origin=np.r_[np.zeros(len(points),np.int32),np.ones(len(ref),np.int32)]
    write(OUT/(short+'_instances_detection.ply'),full,fullcolors,
        instance_id=np.r_[instance,np.full(len(ref),-1)],candidate_id=np.r_[cid,rcid],
        detection_status=np.r_[status,rstatus],point_origin=origin)
    write(OUT/(short+'_instances_original.ply'),points,base_color,instance_id=instance)
    confirmed=[r for r in records if r['meta']['state']=='would_remove']
    cp=np.concatenate([r['points'] for r in confirmed])
    cc=np.concatenate([np.tile(np.array(colorsys.hsv_to_rgb(i/max(1,len(confirmed)),.9,1))*255,(len(r['points']),1)) for i,r in enumerate(confirmed)])
    cg=np.concatenate([np.full(len(r['points']),r['meta']['candidate_id']) for r in confirmed])
    write(OUT/(short+'_confirmed_objects.ply'),np.r_[points,cp],np.r_[np.full((len(points),3),190),cc],
        candidate_id=np.r_[np.full(len(points),-1),cg],point_origin=np.r_[np.zeros(len(points)),np.ones(len(cp))])
    archives=sorted((folder/'deletion/archive').glob('*.npz'))
    deleted=np.unique(np.concatenate([np.load(p)['xyz'] for p in archives]),axis=0)
    write(OUT/(short+'_actual_erased_voxels.ply'),np.r_[points,deleted],
        np.r_[base_color,np.tile([255,0,0],(len(deleted),1))],
        point_origin=np.r_[np.zeros(len(points)),np.full(len(deleted),2)])
    item=dict(seq=seq,source=str(folder),final_points=len(points),frozen_reference_points=len(ref),
        candidates=[r['meta'] for r in records],confirmed_colors={str(r['meta']['candidate_id']):cc[sum(len(x['points']) for x in confirmed[:i])].astype(int).tolist() for i,r in enumerate(confirmed)},
        actual_erased_voxels=len(deleted))
    if short=='pair04':
        r=next(r for r in records if r['meta']['candidate_id']==12)
        p=r['points'];pal=np.array([[135,135,135],[25,210,70],[255,20,35],[30,110,255]],np.uint8)
        evcolor=pal[r['evidence']];evcolor[(r['source']&2)>0]=[255,170,0]
        lo,hi=p.min(0)-.6,p.max(0)+.6
        context=((points>=lo)&(points<=hi)).all(1)
        write(OUT/'pair04_C12_evidence.ply',np.r_[points[context],p],
            np.r_[np.full((context.sum(),3),[215,215,215]),evcolor],
            evidence_code=np.r_[np.full(context.sum(),-1),r['evidence']],
            point_origin=np.r_[np.zeros(context.sum()),np.ones(len(p))])
        rg=xyz(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))
        rl=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
        sg=xyz(DATA/seq/(seq+'_vh_clean_2.ply'))
        sl=np.loadtxt(DATA/'gt_seg_all_things'/(seq+'.txt'),dtype=np.int64)
        ref17=rg[(rl>0)&(rl%1000==17)];res17=sg[(sl>0)&(sl%1000==17)]
        dist_ref,_=cKDTree(ref17).query(p);dist_res,_=cKDTree(res17).query(p)
        confirmed_npz=np.load(r['path']);depth_pts=p[confirmed_npz['depth_contradicted']]
        dd,_=cKDTree(depth_pts).query(deleted);local_deleted=deleted[dd<=np.sqrt(3)*.025+1e-5]
        data=np.r_[ref17,res17,p,local_deleted]
        cols=np.r_[np.tile([255,200,0],(len(ref17),1)),np.tile([0,200,240],(len(res17),1)),
            np.tile([240,0,230],(len(p),1)),np.tile([255,0,0],(len(local_deleted),1))]
        write(OUT/'pair04_C12_GT17_old_new.ply',data,cols,
            point_origin=np.r_[np.full(len(ref17),3),np.full(len(res17),4),np.ones(len(p)),np.full(len(local_deleted),2)])
        item['C12_diagnostics']=dict(reference_points=len(p),reference_gt17_within_5cm=int((dist_ref<=.05).sum()),
            rescan_gt17_within_5cm=int((dist_res<=.05).sum()),
            reference_distance_median_m=float(np.median(dist_ref)),rescan_distance_median_m=float(np.median(dist_res)),
            evidence_counts={str(k):int((r['evidence']==k).sum()) for k in range(4)},
            cumulative_depth_contradiction_points=len(depth_pts),nearby_erased_voxels=len(local_deleted),
            nearby_erased_note='Spatial association to C12 depth references within one voxel diagonal; not a unique causal attribution.')
        preview(OUT/'pair04_detection_preview.png',full,fullcolors,
            'pair04: red = confirmed; orange = pending; blue = cancelled; pastel = final instances')
        preview(OUT/'pair04_C12_evidence_preview.png',np.r_[points[context],p],
            np.r_[np.full((context.sum(),3),[225,225,225]),evcolor],
            'C12 / GT17: red = depth contradiction; green = support; blue = occluded; gray = unknown',focus=p)
    manifest.append(item)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps([dict(seq=r['seq'],diagnosis=r.get('C12_diagnostics')) for r in manifest],indent=2))
