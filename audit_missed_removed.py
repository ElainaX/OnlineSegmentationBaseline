"""Post-hoc stage tracing and orange PLYs. GT never changes detector decisions."""
from pathlib import Path
import json, hashlib, types
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from submap_disjoint.readonly_detection_core import interior_mask

REPO=Path(__file__).resolve().parent
ROOT=Path('/root/autodl-tmp/output/ig_rescan_dense_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
OUT=ROOT/'missed_removed';OUT.mkdir(exist_ok=False)
load=lambda p:json.loads(p.read_text())
runs={r['seq']:r for r in load(Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008/selected_results.json'))}
comp=load(ROOT/'dense/comparison.json')['scenes']

# Instrument a private in-memory copy; every final array must equal the saved run.
source=(REPO/'submap_disjoint/surface_bank.py').read_text()
replacements=[
 ('n=len(self.points); before=self.active','n=len(self.points); before=self.active\n        diag_old_streak=self.streak.copy()'),
 ('            if self.require_color:\n', '            diag_raw=codes.copy()\n            if self.require_color:\n'),
 ('            if self.surface_support and np.any(codes==CONTRADICTION):','            diag_rgb=codes.copy()\n            if self.surface_support and np.any(codes==CONTRADICTION):'),
 ('            after=self.active; changed=np.flatnonzero(before!=after)',
  '            self.trace(frame,ids,diag_raw,diag_rgb,codes,accepted,free_ids,support_ids,verified_ids,diag_old_streak)\n            after=self.active; changed=np.flatnonzero(before!=after)')]
for old,new in replacements:
    assert source.count(old)==1,old
    source=source.replace(old,new)
module=types.ModuleType('submap_disjoint._missed_audit');module.__package__='submap_disjoint'
exec(compile(source,'<read-only-instrumented-surface-bank>','exec'),module.__dict__)

def write_ply(path,points,gids,missed,**fields):
    orange=np.isin(gids,list(missed))
    colors=np.full((len(points),3),160,np.uint8);colors[orange]=[255,145,0]
    fields=dict(gt_instance_id=gids,missed_removed=orange.astype(np.int32),**fields)
    dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')]+[(k,'<i4') for k in fields]
    arr=np.empty(len(points),dtype=dtype)
    for j,k in enumerate(['x','y','z']):arr[k]=points[:,j]
    for j,k in enumerate(['red','green','blue']):arr[k]=colors[:,j]
    for k,v in fields.items():arr[k]=v
    header=['ply','format binary_little_endian 1.0',f'element vertex {len(points)}','property float x','property float y','property float z','property uchar red','property uchar green','property uchar blue']
    header+=['property int '+k for k in fields]+['end_header','']
    with path.open('wb') as f:f.write('\n'.join(header).encode());arr.tofile(f)
    check=o3d.io.read_point_cloud(str(path))
    assert len(check.points)==len(points)
    if len(points):
        assert np.allclose(np.asarray(check.points),points,atol=1e-6)
        assert np.array_equal(np.rint(np.asarray(check.colors)*255).astype(np.uint8),colors)
    return int(orange.sum())

def preview(path,points,gids,missed,title):
    orange=np.isin(gids,list(missed));gray=np.flatnonzero(~orange);fg=np.flatnonzero(orange)
    if len(gray)>45000:gray=gray[np.linspace(0,len(gray)-1,45000,dtype=int)]
    fig=plt.figure(figsize=(14,7))
    for j,(el,az) in enumerate([(75,-60),(25,-60)]):
        ax=fig.add_subplot(1,2,j+1,projection='3d')
        p=points[gray];ax.scatter(*p.T,c='#aaaaaa',s=.15,alpha=.13,depthshade=False)
        if len(fg):ax.scatter(*points[fg].T,c='#ff9100',s=2,depthshade=False)
        for gid in sorted(missed):
            p=points[gids==gid]
            if len(p):ax.text(*np.median(p,axis=0),str(gid),fontsize=9,color='#9d4400')
        ax.set_box_aspect(np.maximum(np.ptp(points,axis=0),.1));ax.view_init(el,az)
        ax.set_xlabel('X / m');ax.set_ylabel('Y / m');ax.set_zlabel('Z / m')
    fig.suptitle(title+'\nOrange: reconstructed surfaces of missed removed GT; gray: context');fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)

all_scenes=[]
for scene in comp:
    seq=scene['seq'];short='pair'+seq.split('_')[1];folder=OUT/short;folder.mkdir()
    original=np.load(ROOT/'dense'/seq/'bank.npz')
    summary=load(ROOT/'dense'/seq/'summary.json');cfg=summary['cfg']
    config=load(Path(summary['source_config']));cam=config['pipeline_config']['cam']
    boundary=load(DATA/seq/'session_boundary.json');cut=boundary['rescan_start_frame']
    hit=set(scene['phases']['rescan']['metrics']['20']['removed_hits'])
    missed=set(scene['verifiable_removed'])-hit
    gt=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    gtids=np.where(labels>0,labels%1000,-1);tree=cKDTree(gt)
    def associate(p):
        d,i=tree.query(p);return np.where(d<=.05,gtids[i],-1).astype(np.int32)
    gids=associate(original['points']);old=original['created_frame']<cut;nold=int(old.sum())
    assert old[:nold].all() and not old[nold:].any()
    namespath=Path('/root/autodl-tmp/output/ig_readonly_20261008/gt_audit/raw_metadata')/boundary['reference_scan_id']/'semseg.v2.json'
    names={int(r['id']):r['label'] for r in load(namespath)['segGroups']} if namespath.exists() else {}
    flags={k:np.zeros(nold,bool) for k in ['in_projection','valid_depth','raw_free','occluded','depth_support','after_rgb','after_nearby','independent_free','history_verified_free','ever_active','cleared_after_vote','cleared_after_confirm']}
    free_counts=np.zeros(nold,np.int32);vote_counts=np.zeros(nold,np.int32)
    perframe=[]
    bank=module.SurfaceBank(.025,cfg,require_color=True,view_gate='ray',surface_support=True,strict_history_support=True)
    def trace(frame,ids,raw,rgb,codes,accepted,free,support,verified,old_streak):
        if frame<cut:return
        def mark(key,idx):
            idx=np.asarray(idx);flags[key][idx[idx<nold]]=True
        mark('in_projection',ids);mark('valid_depth',ids[raw!=0]);mark('raw_free',ids[raw==2]);mark('occluded',ids[raw==3]);mark('depth_support',ids[raw==1])
        mark('after_rgb',ids[rgb==2]);mark('after_nearby',ids[codes==2]);mark('independent_free',ids[accepted&(codes==2)]);mark('history_verified_free',free)
        mark('ever_active',np.flatnonzero(bank.active));mark('cleared_after_vote',support[old_streak[support]>0]);mark('cleared_after_confirm',support[old_streak[support]>=cfg['confirm_observations']])
        r=ids[raw==2];free_counts[r[r<nold]]+=1;v=free[free<nold];vote_counts[v]+=1
        for gid in sorted(missed):
            m=(gids[ids]==gid)&(ids<nold)
            perframe.append(dict(frame=frame,gt_id=gid,projected=int(m.sum()),raw_free=int((m&(raw==2)).sum()),after_rgb=int((m&(rgb==2)).sum()),after_nearby=int((m&(codes==2)).sum()),independent_free=int((m&accepted&(codes==2)).sum()),accepted_free=int((gids[v]==gid).sum()),confirmed=int(((gids[:nold]==gid)&bank.active[:nold]).sum())))
    bank.trace=trace
    k=np.loadtxt(DATA/seq/'intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32)
    reference_support=None
    for j,f in enumerate(summary['selected_frame_ids']):
        if f>=cut and reference_support is None:reference_support=bank.support.copy();assert len(reference_support)==nold
        depth=cv2.imread(str(DATA/seq/'depth'/f'{f}.png'),cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        valid=np.isfinite(depth)&(depth>cam['depth_near'])&(depth<cam['depth_far']);depth[~valid]=0
        inner=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
        pose=np.loadtxt(DATA/seq/'pose'/f'{f}.txt').astype(np.float32)
        rgb=cv2.cvtColor(cv2.imread(str(DATA/seq/'color'/f'{f}.jpg')),cv2.COLOR_BGR2RGB)
        if rgb.shape[:2]!=depth.shape:rgb=cv2.resize(rgb,(depth.shape[1],depth.shape[0]),interpolation=cv2.INTER_NEAREST)
        bank.step(f,pose,k,depth,inner,cam['depth_near'],cam['depth_far'],rgb)
    exact=all(np.array_equal(v,original[key]) for key,v in bank.arrays().items());assert exact,seq
    active=original['active']&(original['last_conflict_frame']>=cut)
    cmd=runs[seq]['command'];igpath=Path(cmd[cmd.index('--output_dir')+1])/seq/'final.ply'
    ig=np.asarray(o3d.io.read_point_cloud(str(igpath)).points);igids=associate(ig)
    # Final IG is a separate preserved reconstruction; do not claim it is a reference checkpoint.
    recon_counts={int(g):int((igids==g).sum()) for g in missed}
    igorange=write_ply(folder/(short+'_IG_reconstruction_missed_orange.ply'),ig,igids,missed)
    oldpoints=original['points'][:nold];oldgids=gids[:nold]
    write_ply(folder/(short+'_detector_reference_missed_orange.ply'),oldpoints,oldgids,missed,reference_support=reference_support,confirmed_at_end=active[:nold].astype(np.int32))
    orange=np.isin(oldgids,list(missed))
    if orange.any():write_ply(folder/(short+'_missed_objects_only.ply'),oldpoints[orange],oldgids[orange],missed)
    # GT reference is an explicitly separate companion; never mixed into reconstruction.
    write_ply(folder/(short+'_GT_reference_missed_orange.ply'),gt,gtids,missed)
    rows=[]
    for gid in sorted(missed):
        m=oldgids==gid;surface=oldpoints[m];gtp=gt[gtids==gid]
        cov=float((cKDTree(surface).query(gtp)[0]<=.05).mean()) if len(surface) else 0.
        counts={key:int(value[m].sum()) for key,value in flags.items()}
        region_details=[]
        for region in scene['phases']['rescan']['regions']:
            contribution=int(region['counts'].get(str(gid),0))
            if contribution:region_details.append(dict(region=region['region'],region_size=region['points'],own_points=contribution,dominant_gt=region['gt_id'],own_fraction=contribution/region['points']))
        row=dict(gt_id=gid,name=names.get(gid,'unknown'),gt_vertices=len(gtp),reference_surface_points=len(surface),reference_gt_coverage_5cm=cov,IG_final_reconstruction_points=recon_counts[gid],reference_verified_points=int(((reference_support>=cfg['confirm_observations'])&m).sum()),stage_unique_points=counts,
                 raw_free_point_observations=int(free_counts[m].sum()),accepted_free_point_observations=int(vote_counts[m].sum()),points_with_two_total_votes=int((vote_counts[m]>=2).sum()),
                 end_confirmed_reference_points=int((active[:nold]&m).sum()),end_confirmed_all_phase_points=int((active&(gids==gid)).sum()),regions_with_own_points=region_details)
        rows.append(row)
        if len(surface):write_ply(folder/(f'{short}_GT{gid}_missed_surface.ply'),surface,np.full(len(surface),gid),{gid})
    preview(folder/(short+'_preview.png'),ig,igids,missed,short+' | preserved IG final reconstruction')
    row=dict(seq=seq,missed_ids=sorted(missed),hit_ids=sorted(hit),excluded_or_missing_GT=sorted(set(scene['official_removed'])-set(scene['verifiable_removed'])),exact_detector_replay=exact,IG_geometry_source=str(igpath),IG_orange_points=igorange,instances=rows)
    (folder/'diagnosis.json').write_text(json.dumps(row,indent=2));(folder/'per_frame_counts.json').write_text(json.dumps(perframe))
    all_scenes.append(row);print(json.dumps(row),flush=True)
(OUT/'diagnosis.json').write_text(json.dumps(dict(scenes=all_scenes,GT_used_for_audit_only=True,actual_deletions=0,AP_computed=False,detector_run=str(ROOT/'dense'),matching='nearest reference GT <= 5cm; region >=20 and strict >50% majority for detection; orange is GT-associated geometry, not a predicted instance ID'),indent=2))
(OUT/'completed.json').write_text(json.dumps(dict(completed=True,files=[str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()]),indent=2))
