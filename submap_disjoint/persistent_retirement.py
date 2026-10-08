"""Persistent whole-instance evidence; detector and numeric gates stay unchanged.

Evidence lives on fixed SurfaceBank samples, not nearby TSDF lattice corners.
Merged identities retain old samples and audit newly acquired surface separately.
GT and scan boundaries are never used. The v1 executor still checks exact lineage.
"""
import json
import time
from collections import Counter
from dataclasses import asdict
import numpy as np
from scipy.spatial import cKDTree
from .instance_retirement import (IGInstanceRetirement, DepthContext, decision,
                                  retire_track, RetirementOptions)
from .readonly_detection_core import as_numpy, SUPPORT, CONTRADICTION
from .ig_correction import lattice_indices, corners
from .surface_bank import voxel_keys
from .surface_deletion import active_lineage


def resolve_live_track(source, mapping):
    """Permit an expanded identity for AUDIT, never implicit erase permission."""
    source=set(map(int,source));mapping=np.asarray(mapping)
    if not source or max(source)>=len(mapping): return None,'missing_lineage'
    rows={int(mapping[r]) for r in source}
    if rows=={-1}: return None,'already_removed'
    if len(rows)!=1 or min(rows)<0: return None,'split_or_partially_removed'
    return next(iter(rows)),'stable'


def surface_reference(coords, bank, radius, minimum_support):
    """One vote per fixed surface cell; unanchored cells remain unknown area.

    Nearest anchor selection is deliberately unchanged from v1. No searching
    for a farther, favorable anchor. Actual bank samples replace lattice points.
    """
    coords=np.asarray(coords,float).reshape(-1,3)
    if not len(coords): return dict(points=coords,anchors=np.empty(0,int),keys=np.empty(0,np.int64))
    distance,idx=cKDTree(bank.points).query(coords,distance_upper_bound=radius)
    anchored=np.isfinite(distance)
    anchored[anchored]&=bank.support[idx[anchored]]>=minimum_support
    anchors=np.full(len(coords),-1,int);anchors[anchored]=idx[anchored]
    points=coords.copy();points[anchored]=bank.points[idx[anchored]]
    keys=voxel_keys(points,bank.voxel_size)
    # An anchored measurement takes precedence over an unknown lattice sample
    # in the same cell; duplicate corners cannot multiply one anchor's vote.
    order=np.lexsort((~anchored,keys))
    _,first=np.unique(keys[order],return_index=True);take=order[first]
    return dict(points=points[take],anchors=anchors[take],keys=keys[take])


def init_state(reference, bank, frame, inherit_existing):
    c={k:v.copy() for k,v in reference.items()}
    ok=c['anchors']>=0;streak=np.zeros(len(ok),np.int64)
    streak[ok]=bank.streak[c['anchors'][ok]]
    c.update(last_bank_streak=streak,local_streak=streak.copy() if inherit_existing else np.zeros(len(ok),np.int64),
             last_support=np.full(len(ok),-1,np.int64),added_frame=np.full(len(ok),frame,np.int64))
    return c


STATE_FIELDS=('points','anchors','keys','last_bank_streak','local_streak','last_support','added_frame')


def union_states(states):
    """Union frozen area without inheriting a neighbor's votes or duplicating it.

    At duplicates use the oldest surface record. A newer support observation
    overrides that record, so merging cannot resurrect revoked evidence.
    """
    data={k:np.concatenate([s[k] for s in states],axis=0) for k in STATE_FIELDS}
    order=np.lexsort((data['added_frame'],data['keys']))
    keys,first,inverse=np.unique(data['keys'][order],return_index=True,return_inverse=True)
    take=order[first];out={k:data[k][take].copy() for k in STATE_FIELDS}
    latest=np.full(len(keys),-1,np.int64);np.maximum.at(latest,inverse,data['last_support'][order])
    revoked=latest>out['last_support'];out['local_streak'][revoked]=0;out['last_support']=latest
    return out


def update_state(c, bank, codes, frame, confirmations):
    """Unknown/occluded frames do not erase votes. Real support does.

    Add only new votes ALREADY accepted by the unchanged RGB-D/ray detector.
    Newly added surface starts at zero, even if its bank sample was active.
    """
    anchored=c['anchors']>=0;now=np.zeros(len(anchored),np.int64)
    now[anchored]=bank.streak[c['anchors'][anchored]]
    reset=(now<c['last_bank_streak'])|(codes==SUPPORT)
    c['local_streak'][reset]=0
    delta=np.maximum(0,now-c['last_bank_streak'])
    c['local_streak'][~reset]+=delta[~reset]
    c['last_bank_streak']=now
    c['last_support'][codes==SUPPORT]=frame
    # The bank's own revocation is authoritative as well as same-point depth.
    confirmed=anchored&(c['local_streak']>=confirmations)&(now>=confirmations)
    return confirmed


class PersistentIGInstanceRetirement(IGInstanceRetirement):
    def reference(self,row,source):
        import open3d.core as core
        o=self.owner;b=self.bank;opt=self.options;g=o.maskGraph
        clouds=[as_numpy(g.instance_dict[row].mask_voxel_coords)]
        # Match the executor's potential erase domain, including raw support.
        for raw in source:
            old=g.f_mask_voxels_clouds.get(o.global_f_mask_ids[raw])
            if old is not None and len(old): clouds.append(as_numpy(old))
        coords=np.unique(np.vstack(clouds),axis=0)
        ids,ok=lattice_indices(o.voxel_block_grids,coords,b.voxel_size,o.cfg['scene']['block_resolution'],o.device_o3c)
        coords,ids=coords[ok],ids[ok]
        if len(ids):
            tid=core.Tensor(ids,dtype=core.int64,device=o.device_o3c)
            weight=o.voxel_block_grids.attribute('weight').reshape((-1,))[tid].cpu().numpy()
            tsdf=o.voxel_block_grids.attribute('tsdf').reshape((-1,))[tid].cpu().numpy()
            coords=coords[(weight>=self.detector.cfg['min_map_weight'])&np.isfinite(tsdf)&(np.abs(tsdf)<=opt.max_abs_tsdf)]
        return surface_reference(coords,b,opt.anchor_radius_voxels*b.voxel_size,self.detector.cfg['confirm_observations'])

    def archive_candidate(self,c,frame):
        np.savez_compressed(self.out/f'candidate_{c["id"]:05d}_{frame:06d}.npz',
                            **{k:c[k] for k in STATE_FIELDS},source_raw_ids=c['source'],created=c['created'])

    def synchronize(self,frame):
        """Coalesce candidates by live row. Expansion audits happen before erase."""
        groups={};o=self.owner
        for c in list(self.pending.values()):
            row,reason=resolve_live_track(c['source'],o.global2merged_mask_id)
            if row is None:
                self.reasons[reason]+=1
                self.log(dict(frame=frame,event='candidate_closed',candidate=c['id'],reason=reason))
                continue
            groups.setdefault(row,[]).append(c)
        current={}
        for row,records in groups.items():
            c=min(records,key=lambda x:x['created'])
            source=tuple(np.flatnonzero(np.asarray(o.global2merged_mask_id)==row).tolist())
            if len(records)>1 or c['source']!=source:
                # New surfaces get zero local evidence. Existing old surfaces
                # remain even after local erasure; denominator never shrinks.
                added=init_state(self.reference(row,source),self.bank,frame,False)
                old_size=len(c['points']);old_sources=[r['id'] for r in records]
                c.update(union_states(records+[added]));c['source']=source
                self.reasons['lineage_extended_preserved']+=1
                self.log(dict(frame=frame,event='lineage_extended_preserved',candidate=c['id'],
                              inherited_candidates=old_sources,old_points=old_size,reference_points=len(c['points']),raw_ids=list(source)))
                self.archive_candidate(c,frame)
            current[source]=c
        self.pending=current

    def step(self,frame,depth,pose):
        if frame!=self.bank.last_frame or frame<=self.last_frame: raise ValueError('Must follow current detector step')
        self.last_frame=frame;started=time.perf_counter();b=self.bank;o=self.owner;opt=self.options
        self.synchronize(frame)
        # EXACT v1 detector entry: current accepted contradiction plus raw owner.
        mapping=np.asarray(o.global2merged_mask_id);fresh=np.flatnonzero((b.streak>0)&(b.last_conflict_frame==frame))
        source_rows=Counter()
        if len(fresh):
            coords=corners(b.points[fresh],b.voxel_size).reshape(-1,3)
            ids,ok=lattice_indices(o.voxel_block_grids,coords,b.voxel_size,o.cfg['scene']['block_resolution'],o.device_o3c)
            for idx in ids[ok]:
                for raw in active_lineage(o.voxel_hash_table.hash_table[int(idx)],mapping):source_rows[int(mapping[raw])]+=1
        created=0
        for row,_ in source_rows.most_common():
            source=tuple(np.flatnonzero(mapping==row).tolist())
            if source in self.pending or frame<self.cooldown.get(source,-1):continue
            if len(self.pending)>=opt.max_candidates or created>=opt.max_new_candidates_per_step:break
            if float(o.merged_mask_weight[row])<self.detector.cfg['min_instance_weight']:continue
            ref=self.reference(row,source);created+=1
            reason='small_reference' if len(ref['points'])<opt.min_reference_points else None
            if reason is None and np.mean(ref['anchors']>=0)<opt.min_anchor_fraction:reason='insufficient_anchor_coverage'
            if reason:
                self.reasons[reason]+=1;self.cooldown[source]=frame+opt.retry_cooldown;continue
            self.serial+=1;c=init_state(ref,b,frame,True)
            c.update(id=self.serial,source=source,created=frame,last_stats={})
            self.pending[source]=c;self.archive_candidate(c,frame);self.reasons['created']+=1
            self.log(dict(frame=frame,event='candidate_created',candidate=c['id'],raw_ids=list(source),reference_points=len(c['points'])))
        context=None;retired=[];stats_rows=[]
        for source,c in list(self.pending.items()):
            if context is None:context=DepthContext(as_numpy(depth),as_numpy(pose),self.detector.K,self.detector.cfg,
                self.detector.cam['depth_near'],self.detector.cam['depth_far'],opt.protect_radius_voxels*b.voxel_size)
            codes=context.classify(c['points'])
            confirmed=update_state(c,b,codes,frame,self.detector.cfg['confirm_observations'])
            recent=(c['last_support']>=0)&(frame-c['last_support']<=opt.evidence_max_age)
            # Keep full ledger across extensions; evaluate <=4000 uniform cells.
            n=len(codes);sample=np.linspace(0,n-1,min(n,opt.max_reference_points)).astype(int)
            accept,stats=decision(confirmed[sample],recent[sample],codes[sample]==CONTRADICTION,c['anchors'][sample]>=0,opt)
            stats.update(full_reference_points=n,full_confirmed_points=int(confirmed.sum()),
                persistent_past_age=int(np.count_nonzero(confirmed&(frame-b.last_conflict_frame[np.maximum(c['anchors'],0)]>opt.evidence_max_age))),
                candidate_age=frame-c['created'])
            c['last_stats']=stats;stats_rows.append(dict(candidate=c['id'],**stats))
            for reason in stats['reasons']:self.reasons[reason]+=1
            if not accept:continue
            event=dict(frame=frame,event='would_retire',candidate=c['id'],raw_ids=list(source),evidence=stats)
            if self.enabled:
                from .ig_deletion import IGDeletion
                executor=IGDeletion(o,None,self.out/'executors'/f'{frame:06d}_{c["id"]:05d}')
                result=retire_track(o,executor,source,frame,context,self.out/'archive',opt.max_shared_fraction,opt.verify_state)
                executor.save();event.update(result)
                if not result['retired']:
                    self.reasons[result['reason']]+=1;self.log(event);continue
                event['event']='retired';retired.append(event)
                np.savez_compressed(self.out/f'evidence_{frame:06d}_{c["id"]:05d}.npz',**{k:c[k] for k in STATE_FIELDS},confirmed=confirmed,current_codes=codes)
                self.pending.pop(source)
            self.events.append(event);self.log(event)
        self.seconds+=time.perf_counter()-started
        if stats_rows:
            with (self.out/'candidate_stats.jsonl').open('a') as f:f.write(json.dumps(dict(frame=frame,candidates=stats_rows))+'\n')
        return retired

    def save(self):
        for c in self.pending.values():
            np.savez_compressed(self.out/f'pending_{c["id"]:05d}.npz',
                                **{k:c[k] for k in STATE_FIELDS},source_raw_ids=c['source'],created=c['created'])
        report=super().save()
        report.update(method='IG persistent surface retirement v2',
            evidence_expiration=False,lineage_expansion='union old frozen area with newly audited area; new area starts unconfirmed',
            point_reference='fixed measured surface-bank coordinates; one vote per cell',
            max_candidate_age_policy='no destructive timeout; max_candidates still caps resident candidates',
            pending_reference_points=sum(len(c['points']) for c in self.pending.values()))
        (self.out/'summary.json').write_text(json.dumps(report,indent=2));return report
