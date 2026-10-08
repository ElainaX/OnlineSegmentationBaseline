"""Opt-in whole IG instance retirement after frozen, instance-wide evidence.

Detector thresholds are unchanged. No GT or scan boundary enters this module.
This retires an identity and resets exclusive shared-TSDF cells; it is NOT exact
deintegration. A partly confirmed object may have unobserved cells, so the
whole-object conclusion is explicitly an inference with coverage/support gates.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
import json
import time

import numpy as np
from scipy.spatial import cKDTree
from .readonly_detection_core import as_numpy, interior_mask, classify_depth, SUPPORT, CONTRADICTION
from .ig_correction import corners, lattice_indices
from .surface_deletion import active_lineage


@dataclass(frozen=True)
class RetirementOptions:
    min_reference_points: int = 30
    max_reference_points: int = 4000
    min_confirmed_points: int = 20
    min_anchor_fraction: float = .70
    min_confirmed_fraction: float = .60
    min_contradiction_purity: float = .90
    max_recent_support_fraction: float = .05
    min_current_free_fraction: float = .10
    max_shared_fraction: float = .10
    evidence_max_age: int = 60
    max_candidate_age: int = 300
    retry_cooldown: int = 30
    max_candidates: int = 128
    max_new_candidates_per_step: int = 8
    anchor_radius_voxels: float = 1.75
    protect_radius_voxels: float = 2.
    max_abs_tsdf: float = .5
    verify_state: bool = False

    def __post_init__(self):
        for key in ['min_reference_points','max_reference_points','min_confirmed_points','evidence_max_age',
                    'max_candidate_age','retry_cooldown','max_candidates','max_new_candidates_per_step']:
            if getattr(self,key)<1: raise ValueError(key)
        if self.max_reference_points<self.min_reference_points: raise ValueError('Reference sample bounds')
        for key in ['min_anchor_fraction','min_confirmed_fraction','min_contradiction_purity',
                    'max_recent_support_fraction','min_current_free_fraction','max_shared_fraction']:
            if not 0<=getattr(self,key)<=1: raise ValueError(key)
        if self.anchor_radius_voxels<=0 or self.protect_radius_voxels<0 or not 0<self.max_abs_tsdf<=1:
            raise ValueError('Geometry options')


def resolve_frozen_track(source, mapping):
    """Row renumbering is allowed; newly merged observations require a new audit."""
    source=set(map(int,source));mapping=np.asarray(mapping)
    if not source or max(source)>=len(mapping): return None,'missing_lineage'
    rows=set(int(mapping[r]) for r in source)
    if rows=={-1}: return None,'already_removed'
    if len(rows)!=1 or min(rows)<0: return None,'split_or_partially_removed'
    row=next(iter(rows))
    if set(np.flatnonzero(mapping==row))!=source: return None,'lineage_expanded'
    return row,'stable'


def decision(confirmed, recent_support, current_free, anchored, options):
    """All fractions use the FROZEN full reference sample, never only visible points."""
    n=len(confirmed)
    c=int(np.count_nonzero(confirmed));s=int(np.count_nonzero(recent_support))
    stats=dict(reference_points=n,confirmed_points=c,recent_support_points=s,
               confirmed_fraction=c/max(n,1),support_fraction=s/max(n,1),
               contradiction_purity=c/max(c+s,1),anchor_fraction=float(np.mean(anchored)) if n else 0.,
               current_free_fraction=float(np.mean(current_free)) if n else 0.)
    checks=[('reference_too_small',n>=options.min_reference_points),
            ('insufficient_anchor_coverage',stats['anchor_fraction']>=options.min_anchor_fraction),
            ('too_few_confirmed_points',c>=options.min_confirmed_points),
            ('insufficient_whole_instance_coverage',stats['confirmed_fraction']>=options.min_confirmed_fraction),
            ('recent_surface_support',stats['support_fraction']<=options.max_recent_support_fraction),
            ('mixed_evidence',stats['contradiction_purity']>=options.min_contradiction_purity),
            ('no_current_free_evidence',stats['current_free_fraction']>=options.min_current_free_fraction)]
    reasons=[reason for reason,passed in checks if not passed]
    return not reasons,dict(**stats,reasons=reasons)


class DepthContext:
    """One measured-cloud KD tree per frame, shared by all candidates."""
    def __init__(self, depth, pose, K, cfg, near, far, protection_radius):
        self.depth=np.asarray(depth);self.pose=np.asarray(pose);self.K=K
        self.cfg=cfg;self.near=near;self.far=far;self.radius=protection_radius
        valid=np.isfinite(depth)&(depth>near)&(depth<far)
        self.inner=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
        y,x=np.nonzero(valid)
        cloud=(np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(K).T)*depth[y,x,None]
        self.tree=cKDTree(cloud) if len(cloud) else None

    def classify(self, points):
        camera=(np.asarray(points)-self.pose[:3,3])@self.pose[:3,:3]
        codes=np.zeros(len(points),np.uint8);protected=np.zeros(len(points),bool)
        z=camera[:,2];ids=np.flatnonzero(np.isfinite(camera).all(1)&(z>self.near)&(z<self.far))
        uv=np.rint((camera[ids]@self.K.T)[:,:2]/z[ids,None]).astype(np.int64)
        h,w=self.depth.shape;ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
        ids,uv=ids[ok],uv[ok];x,y=uv[:,0],uv[:,1]
        codes[ids]=classify_depth(self.depth[y,x],z[ids],self.inner[y,x],
                                 self.cfg['depth_epsilon_m'],self.cfg['depth_relative_epsilon'])
        if self.tree is not None and self.radius>0 and len(points):
            finite=np.flatnonzero(np.isfinite(camera).all(1))
            d,_=self.tree.query(camera[finite],distance_upper_bound=self.radius)
            protected[finite]=np.isfinite(d)
            codes[protected]=SUPPORT
        return codes


def refresh_after_retirement(owner):
    import torch
    from .sparse_edges import Edges
    graph=owner.maskGraph
    graph.mask_voxels_coords=owner.mask_voxels_coords
    owner.merged_mask_bbox.zero_();owner.edges=Edges()
    for mid,inst in graph.instance_dict.items():
        coords=inst.mask_voxel_coords
        if len(coords): owner.merged_mask_bbox[mid]=torch.cat((coords.min(0).values,coords.max(0).values))
        other,fwd,bwd=graph.query_mask_under_visible_part(graph.merged_mask_in_frame_mat[mid],coords,owner.global2merged_mask_id)
        if other.numel(): owner.edges.write_row_col(mid,other,fwd,bwd)
        owner.edges.set(mid,mid,1.)
    for obj in [owner,owner.geometry]:
        obj.pc_frame_id=obj.pc_feature_frame_id=-1
        obj.points=obj.colors=obj.pc_xyz=obj.pc_feature=None
    owner.pred_inst_mask_frame_id=-1;owner.pred_inst_masks=None


def retire_track(owner, executor, source, frame, context, archive_dir, max_shared_fraction=.10, verify=False):
    """Remove one frozen identity; erase exclusive, unsupported geometry only.

    Called only AFTER instance-wide confirmation. Shared/currently-supported
    cells remain geometric map content, even if the old identity is retired.
    """
    import torch
    import open3d.core as core
    source=set(map(int,source));row,reason=resolve_frozen_track(source,owner.global2merged_mask_id)
    if row is None: return dict(retired=False,reason=reason)
    voxel=owner.cfg['scene']['voxel_size'];res=owner.cfg['scene']['block_resolution'];graph=owner.maskGraph
    coords=as_numpy(graph.instance_dict[row].mask_voxel_coords).copy()
    # Include original support cache cells, which need not all survive boundary filtering.
    clouds=[coords]
    for r in source:
        old=graph.f_mask_voxels_clouds.get(owner.global_f_mask_ids[r])
        if old is not None and len(old): clouds.append(as_numpy(old))
    coords=np.unique(np.vstack(clouds),axis=0)
    ids,exists=lattice_indices(owner.voxel_block_grids,coords,voxel,res,owner.device_o3c)
    ids,coords=ids[exists],coords[exists]
    shared=np.array([bool(active_lineage(owner.voxel_hash_table.hash_table[int(i)],owner.global2merged_mask_id)-source) for i in ids])
    if len(shared) and float(shared.mean())>max_shared_fraction:
        return dict(retired=False,reason='shared_geometry_fraction',shared_fraction=float(shared.mean()))
    codes=context.classify(coords)
    removable=(~shared)&(codes!=SUPPORT)
    path=Path(archive_dir);path.mkdir(parents=True,exist_ok=True)
    name=f'{frame:06d}_{min(source):08d}'
    np.savez_compressed(path/(name+'.npz'),coords=coords,source_raw_ids=sorted(source),
                        shared=shared,current_support=codes==SUPPORT,erase=removable,
                        semantic_feature=as_numpy(owner.mask_features[row]).copy())
    (path/(name+'.json')).write_text(json.dumps(dict(frame=frame,old_row=row,raw_ids=sorted(source),
        memberships=[sorted(owner.voxel_hash_table.hash_table[int(i)] or ()) for i in ids]),indent=2))
    shared_before=None
    if verify and shared.any():
        index=core.Tensor(ids[shared],dtype=core.int64,device=owner.device_o3c)
        shared_before={a:owner.voxel_block_grids.attribute(a).reshape((-1,w))[index].cpu().numpy().copy()
                       for a,w in [('tsdf',1),('weight',1),('color',3)]}
    # Separate erase calls use separate frame archive namespaces to avoid overwrites.
    result=dict(voxels=0)
    if removable.any():
        started=time.perf_counter();result=executor.erase(frame,[(min(source),coords[removable])])
        executor.seconds+=time.perf_counter()-started;executor.frames+=1
        executor.events.append(dict(frame=frame,**result))
    rows=sorted({owner.global2merged_mask_id[r] for r in source if owner.global2merged_mask_id[r]>=0})
    if rows: owner.remove_masks(frame,torch.tensor(rows,device=owner.device,dtype=torch.long))
    no_owner=[]
    for i in ids:
        remaining=set(owner.voxel_hash_table.hash_table[int(i)] or ())-source
        remaining=set(active_lineage(remaining,owner.global2merged_mask_id))
        owner.voxel_hash_table.hash_table[int(i)]=remaining or None
        owner.voxel_hash_table.hash_item_flag[int(i)]=bool(remaining)
        if not remaining: no_owner.append(int(i))
    for r in source:
        key=owner.global_f_mask_ids[r]
        if key in graph.f_mask_voxels_clouds: graph.f_mask_voxels_clouds[key]=graph.f_mask_voxels_clouds[key][:0]
    if no_owner:
        idx=torch.tensor(no_owner,device=owner.device,dtype=torch.long)
        graph.voxel_in_frame_matrix[idx]=False
        if graph.boundary_voxel_indices is not None:
            graph.boundary_voxel_indices=graph.boundary_voxel_indices[~torch.isin(graph.boundary_voxel_indices,idx)]
    refresh_after_retirement(owner)
    if verify:
        assert all(owner.global2merged_mask_id[r]==-1 for r in source)
        assert all(not source.intersection(owner.voxel_hash_table.hash_table[int(i)] or ()) for i in ids)
        if shared_before is not None:
            for a,w in [('tsdf',1),('weight',1),('color',3)]:
                np.testing.assert_array_equal(shared_before[a],owner.voxel_block_grids.attribute(a).reshape((-1,w))[index].cpu().numpy())
        for mid,inst in graph.instance_dict.items():
            assert len(inst.mask_voxel_coords)==owner.merged_mask_size[mid]
    return dict(retired=True,raw_ids=sorted(source),erased_corners=result['voxels'],
                retained_shared_corners=int(shared.sum()),retained_supported_corners=int(((codes==SUPPORT)&~shared).sum()))


class IGInstanceRetirement:
    def __init__(self,owner,detector,output,options=None,enabled=False):
        self.owner=owner;self.detector=detector;self.bank=detector.bank
        self.options=options or RetirementOptions();self.enabled=enabled
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.pending={};self.cooldown={};self.events=[];self.reasons=Counter();self.seconds=0.;self.serial=0
        self.tree=None;self.tree_size=-1;self.last_frame=-1

    def log(self,event):
        with (self.out/'events.jsonl').open('a') as f: f.write(json.dumps(event)+'\n')

    def snapshot(self,row,source,frame):
        import open3d.core as core
        o=self.owner;b=self.bank;opt=self.options
        coords=as_numpy(o.maskGraph.instance_dict[row].mask_voxel_coords)
        ids,ok=lattice_indices(o.voxel_block_grids,coords,b.voxel_size,o.cfg['scene']['block_resolution'],o.device_o3c)
        coords,ids=coords[ok],ids[ok]
        if not len(ids): return None,'no_geometry'
        tid=core.Tensor(ids,dtype=core.int64,device=o.device_o3c)
        weight=o.voxel_block_grids.attribute('weight').reshape((-1,))[tid].cpu().numpy()
        tsdf=o.voxel_block_grids.attribute('tsdf').reshape((-1,))[tid].cpu().numpy()
        coords=coords[(weight>=self.detector.cfg['min_map_weight'])&np.isfinite(tsdf)&(np.abs(tsdf)<=opt.max_abs_tsdf)]
        if len(coords)<opt.min_reference_points: return None,'small_reference'
        if len(coords)>opt.max_reference_points: coords=coords[np.linspace(0,len(coords)-1,opt.max_reference_points).astype(int)]
        if self.tree_size!=len(b.points): self.tree=cKDTree(b.points);self.tree_size=len(b.points)
        distance,nearest=self.tree.query(coords,distance_upper_bound=opt.anchor_radius_voxels*b.voxel_size)
        anchored=np.isfinite(distance);anchors=np.full(len(coords),-1,np.int64)
        anchored[anchored]&=b.support[nearest[anchored]]>=self.detector.cfg['confirm_observations']
        anchors[anchored]=nearest[anchored]
        if anchored.mean()<opt.min_anchor_fraction: return None,'insufficient_anchor_coverage'
        self.serial+=1
        candidate=dict(id=self.serial,source=tuple(sorted(source)),created=frame,points=coords.copy(),anchors=anchors,
                       last_support=np.full(len(coords),-1,np.int64),last_stats={})
        np.savez_compressed(self.out/f'candidate_{self.serial:05d}.npz',points=coords,anchors=anchors,source_raw_ids=sorted(source),created=frame)
        return candidate,'created'

    def step(self,frame,depth,pose):
        if frame!=self.bank.last_frame or frame<=self.last_frame: raise ValueError('Must follow current detector step')
        self.last_frame=frame;start=time.perf_counter();b=self.bank;o=self.owner;opt=self.options
        mapping=np.asarray(o.global2merged_mask_id)
        # Freeze at first accepted conflict, before local deletion prunes the denominator.
        fresh=np.flatnonzero((b.streak>0)&(b.last_conflict_frame==frame))
        source_rows=Counter()
        if len(fresh):
            coords=corners(b.points[fresh],b.voxel_size).reshape(-1,3)
            ids,ok=lattice_indices(o.voxel_block_grids,coords,b.voxel_size,o.cfg['scene']['block_resolution'],o.device_o3c)
            for idx in ids[ok]:
                for raw in active_lineage(o.voxel_hash_table.hash_table[int(idx)],mapping): source_rows[int(mapping[raw])]+=1
        new=0
        for row,_ in source_rows.most_common():
            source=tuple(np.flatnonzero(mapping==row).tolist())
            if source in self.pending or frame<self.cooldown.get(source,-1): continue
            if len(self.pending)>=opt.max_candidates or new>=opt.max_new_candidates_per_step: break
            if float(o.merged_mask_weight[row])<self.detector.cfg['min_instance_weight']: continue
            candidate,reason=self.snapshot(row,source,frame);self.reasons[reason]+=1;new+=1
            if candidate is None: self.cooldown[source]=frame+opt.retry_cooldown
            else:
                self.pending[source]=candidate
                self.log(dict(frame=frame,event='candidate_created',candidate=candidate['id'],raw_ids=list(source),reference_points=len(candidate['points'])))
        context=None;retired=[];candidate_stats=[]
        for source,c in list(self.pending.items()):
            row,reason=resolve_frozen_track(source,o.global2merged_mask_id)
            if row is None or frame-c['created']>opt.max_candidate_age:
                reason=reason if row is None else 'expired'
                self.reasons[reason]+=1;self.pending.pop(source);self.cooldown[source]=frame+opt.retry_cooldown
                self.log(dict(frame=frame,event='candidate_closed',candidate=c['id'],reason=reason));continue
            if context is None:
                context=DepthContext(as_numpy(depth),as_numpy(pose),self.detector.K,self.detector.cfg,
                                     self.detector.cam['depth_near'],self.detector.cam['depth_far'],opt.protect_radius_voxels*b.voxel_size)
            codes=context.classify(c['points']);c['last_support'][codes==SUPPORT]=frame
            anchored=c['anchors']>=0;confirmed=np.zeros(len(anchored),bool)
            anchors=c['anchors'][anchored]
            confirmed[anchored]=b.active[anchors]&((frame-b.last_conflict_frame[anchors])<=opt.evidence_max_age)
            # A newer supporting observation must be followed by a fresh detector conflict.
            confirmed[anchored]&=b.last_conflict_frame[anchors]>c['last_support'][anchored]
            recent=(c['last_support']>=0)&(frame-c['last_support']<=opt.evidence_max_age)
            accept,stats=decision(confirmed,recent,codes==CONTRADICTION,anchored,opt)
            c['last_stats']=stats;candidate_stats.append(dict(candidate=c['id'],**stats))
            for reason in stats['reasons']: self.reasons[reason]+=1
            if not accept: continue
            event=dict(frame=frame,event='would_retire',candidate=c['id'],raw_ids=list(source),evidence=stats)
            if self.enabled:
                from .ig_deletion import IGDeletion
                executor=IGDeletion(o,None,self.out/'executors'/f'{frame:06d}_{c["id"]:05d}')
                result=retire_track(o,executor,source,frame,context,self.out/'archive',opt.max_shared_fraction,opt.verify_state)
                executor.save();event.update(result)
                if not result['retired']:
                    self.reasons[result['reason']]+=1;self.log(event);continue
                event['event']='retired';retired.append(event)
                np.savez_compressed(self.out/f'evidence_{frame:06d}_{c["id"]:05d}.npz',points=c['points'],anchors=c['anchors'],
                                    confirmed=confirmed,last_support=c['last_support'],current_codes=codes)
                self.pending.pop(source)
            self.events.append(event);self.log(event)
        self.seconds+=time.perf_counter()-start
        if candidate_stats:
            with (self.out/'candidate_stats.jsonl').open('a') as f:
                f.write(json.dumps(dict(frame=frame,candidates=candidate_stats))+'\n')
        return retired

    def save(self):
        report=dict(method='IG whole-instance retirement v1',enabled=self.enabled,GT_used=False,independent_TSDF_deactivation=False,
                    options=asdict(self.options),seconds=self.seconds,candidates_created=self.serial,
                    retired_instances=sum(e['event']=='retired' for e in self.events),pending_candidates=len(self.pending),
                    reason_counts=dict(self.reasons),events=self.events)
        (self.out/'summary.json').write_text(json.dumps(report,indent=2))
        return report
