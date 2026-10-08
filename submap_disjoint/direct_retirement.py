"""Direct surface-ID receipts before local erase; experimental whole retirement.

No GT/session inputs. A candidate stores measured surface cells and actual TSDF
zero-crossing surface cells, not the volume of a near-zero TSDF lattice band.
Confirmed events keep the raw ownership recorded before local mutation.
"""
import json
import time
from collections import Counter
import numpy as np
from .persistent_retirement import (PersistentIGInstanceRetirement,resolve_live_track,
    init_state,update_state,STATE_FIELDS)
from .instance_retirement import DepthContext,decision,retire_track
from .readonly_detection_core import as_numpy,SUPPORT,CONTRADICTION
from .surface_bank import voxel_keys
from .surface_deletion import active_lineage
from .ig_correction import corners,lattice_indices


def union_direct(states):
    """Keep one physical cell. A measured anchor can upgrade an unknown cell."""
    data={k:np.concatenate([s[k] for s in states]) for k in STATE_FIELDS}
    order=np.lexsort((data['added_frame'],data['anchors']<0,data['keys']))
    keys,first,inv=np.unique(data['keys'][order],return_index=True,return_inverse=True)
    out={k:data[k][order[first]].copy() for k in STATE_FIELDS}
    latest=np.full(len(keys),-1,np.int64)
    np.maximum.at(latest,inv,data['last_support'][order])
    revoked=latest>out['last_support'];out['local_streak'][revoked]=0
    cursor=np.zeros(len(keys),np.int64)
    recent=data['last_support'][order]==latest[inv]
    np.maximum.at(cursor,inv[recent],data['last_bank_streak'][order][recent])
    out['last_bank_streak'][revoked]=cursor[revoked];out['last_support']=latest
    return out


def lease_authorized(lease,source):
    """A spatial neighbor or only a partial overlap of owners is insufficient."""
    return bool(lease) and set(lease)<=set(source)


class DirectIGInstanceRetirement(PersistentIGInstanceRetirement):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.receipts={};self.receipt_created={};self.surface_frame=-1
        self.reference_cache=None;self.receipt_rows=[]

    def memberships(self,points):
        """Actual current raw owners of the eight interpolation lattice corners."""
        o=self.owner;b=self.bank
        if not len(points):return []
        xyz=corners(points,b.voxel_size).reshape(-1,3)
        ids,ok=lattice_indices(o.voxel_block_grids,xyz,b.voxel_size,
                              o.cfg['scene']['block_resolution'],o.device_o3c)
        cache={int(i):active_lineage(o.voxel_hash_table.hash_table[int(i)],o.global2merged_mask_id)
               for i in np.unique(ids[ok])}
        return [frozenset().union(*(cache[int(i)] for i,v in zip(row,valid) if v))
                for row,valid in zip(ids.reshape(-1,8),ok.reshape(-1,8))]

    def receive(self,frame):
        """Record at first accepted contradiction, even if no candidate is ready."""
        b=self.bank;o=self.owner
        for p in [p for p in self.receipts if b.streak[p]==0]:
            del self.receipts[p];self.receipt_created.pop(p,None)
        fresh=np.flatnonzero((b.streak>0)&(b.last_conflict_frame==frame))
        missing=np.array([p for p in fresh if int(p) not in self.receipts],np.int64)
        owners=self.memberships(b.points[missing]);saved=[];flat=[];offset=[0]
        for p,lease in zip(missing,owners):
            if not lease:continue
            p=int(p);self.receipts[p]=lease;self.receipt_created[p]=frame
            saved.append(p);flat.extend(sorted(lease));offset.append(len(flat))
        if saved:
            np.savez_compressed(self.out/f'receipts_{frame:06d}.npz',surface_ids=saved,
                                raw_ids=flat,offsets=offset)
        rows=Counter()
        for p in fresh:
            for raw in self.receipts.get(int(p),()):
                if 0<=raw<len(o.global2merged_mask_id) and o.global2merged_mask_id[raw]>=0:
                    rows[int(o.global2merged_mask_id[raw])]+=1
        self.receipt_rows.append(dict(frame=frame,fresh=len(fresh),new_receipts=len(saved),
                                     stored_receipts=len(self.receipts),unowned_fresh=int(sum(int(p) not in self.receipts for p in fresh))))
        return rows

    def reference(self,row,source):
        """Freeze all owned measured surfaces, plus unmatched actual map surface."""
        o=self.owner;b=self.bank;g=o.maskGraph;source=set(source)
        clouds=[as_numpy(g.instance_dict[row].mask_voxel_coords)]
        for raw in source:
            old=g.f_mask_voxels_clouds.get(o.global_f_mask_ids[raw])
            if old is not None and len(old):clouds.append(as_numpy(old))
        coords=np.vstack(clouds)
        leased=np.array([p for p,lease in self.receipts.items() if lease_authorized(lease,source)],np.int64)
        if len(coords):
            lo=coords.min(0)-b.voxel_size;hi=coords.max(0)+b.voxel_size
            ids=np.flatnonzero(((b.points>=lo)&(b.points<=hi)).all(1))
            owners=self.memberships(b.points[ids])
            ids=ids[np.array([bool(lease&source) for lease in owners],bool)]
        else:
            ids=np.empty(0,np.int64);lo=np.zeros(3);hi=np.zeros(3)
        ids=np.unique(np.r_[ids,leased]);points=b.points[ids].copy();anchors=ids.copy()
        # Surface extraction is cached only within this pre-fusion frame. All
        # candidates audit the same pre-local-erasure geometry snapshot.
        if self.surface_frame!=self.last_frame:
            pc=o.voxel_block_grids.extract_point_cloud(weight_threshold=self.detector.cfg['min_map_weight'])
            self.reference_cache=pc.point.positions.cpu().numpy().copy()
            self.surface_frame=self.last_frame
        surface=self.reference_cache
        if len(coords) and len(surface):
            pts=surface[((surface>=lo)&(surface<=hi)).all(1)]
            memberships=self.memberships(pts)
            pts=pts[np.array([bool(lease&source) for lease in memberships],bool)]
            points=np.vstack((points,pts));anchors=np.r_[anchors,np.full(len(pts),-1,np.int64)]
        if not len(points):return dict(points=np.empty((0,3)),anchors=np.empty(0,np.int64),keys=np.empty(0,np.int64))
        keys=voxel_keys(points,b.voxel_size)
        # Use the detector's original grid key, which predates float32 storage.
        # Requantizing rounded coordinates could duplicate one surface ID.
        known=anchors>=0;keys[known]=b.keys[anchors[known]]
        order=np.lexsort((anchors<0,keys));_,first=np.unique(keys[order],return_index=True);take=order[first]
        return dict(points=points[take],anchors=anchors[take],keys=keys[take])

    def state(self,reference,source,frame):
        c=init_state(reference,self.bank,frame,True)
        # Existing confirmations are transferable only with original raw-owner
        # receipts. Unrelated or newly appearing surface gets no inherited vote.
        authorized=np.array([a>=0 and lease_authorized(self.receipts.get(int(a)),source)
                             for a in c['anchors']],bool)
        c['local_streak'][~authorized]=0
        return c

    def synchronize(self,frame):
        groups={};o=self.owner
        for c in self.pending.values():
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
                ids=[r['id'] for r in records];n=len(c['points'])
                c.update(union_direct(records+[self.state(self.reference(row,source),source,frame)]));c['source']=source
                self.reasons['lineage_extended_preserved']+=1
                self.log(dict(frame=frame,event='lineage_extended_preserved',candidate=c['id'],
                              inherited_candidates=ids,old_points=n,reference_points=len(c['points'])))
                self.archive_candidate(c,frame)
            current[source]=c
        self.pending=current

    def attach_receipts(self,c,frame):
        ids=np.array([p for p,lease in self.receipts.items() if lease_authorized(lease,c['source'])],np.int64)
        if not len(ids):return
        # Merge even if the grid cell exists as UNKNOWN. Point IDs, rather than
        # nearest neighbors, carry the confirmation through local TSDF erasure.
        ref=dict(points=self.bank.points[ids],anchors=ids,keys=self.bank.keys[ids])
        old=set(c['anchors'][c['anchors']>=0].tolist())
        new=set(ids.tolist())-old
        if new:
            c.update(union_direct([c,self.state(ref,c['source'],frame)]))
            self.reasons['direct_receipt_samples_attached']+=len(new)
            self.log(dict(frame=frame,event='direct_receipts_attached',candidate=c['id'],samples=len(new)))
            self.archive_candidate(c,frame)

    def decide_candidate(self,c,confirmed,recent,codes,mature,sample,frame,pose):
        return decision(confirmed[sample],recent[sample],codes[sample]==CONTRADICTION,mature[sample],self.options)

    def step(self,frame,depth,pose):
        if frame!=self.bank.last_frame or frame<=self.last_frame:raise ValueError('Must follow detector step')
        self.last_frame=frame;start=time.perf_counter();b=self.bank;o=self.owner;opt=self.options
        source_rows=self.receive(frame)
        self.synchronize(frame)
        created=0;mapping=np.asarray(o.global2merged_mask_id)
        for row,_ in source_rows.most_common():
            source=tuple(np.flatnonzero(mapping==row).tolist())
            if source in self.pending or frame<self.cooldown.get(source,-1):continue
            if len(self.pending)>=opt.max_candidates or created>=opt.max_new_candidates_per_step:break
            if float(o.merged_mask_weight[row])<self.detector.cfg['min_instance_weight']:continue
            ref=self.reference(row,source);created+=1
            if not len(ref['points']):
                self.reasons['empty_reference']+=1;self.cooldown[source]=frame+opt.retry_cooldown;continue
            # Capture before local erase even when coverage is insufficient.
            # ALL original size/coverage gates still apply at the decision.
            self.serial+=1;c=self.state(ref,source,frame)
            c.update(id=self.serial,source=source,created=frame,last_stats={})
            self.pending[source]=c;self.archive_candidate(c,frame);self.reasons['created']+=1
            self.log(dict(frame=frame,event='candidate_created',candidate=c['id'],raw_ids=list(source),reference_points=len(c['points'])))
        context=None;stats_rows=[];retired=[]
        for source,c in list(self.pending.items()):
            self.attach_receipts(c,frame)
            if context is None:context=DepthContext(as_numpy(depth),as_numpy(pose),self.detector.K,self.detector.cfg,
                self.detector.cam['depth_near'],self.detector.cam['depth_far'],opt.protect_radius_voxels*b.voxel_size)
            codes=context.classify(c['points'])
            confirmed=update_state(c,b,codes,frame,self.detector.cfg['confirm_observations'])
            authorized=np.array([a>=0 and lease_authorized(self.receipts.get(int(a)),source) for a in c['anchors']],bool)
            confirmed&=authorized
            recent=(c['last_support']>=0)&(frame-c['last_support']<=opt.evidence_max_age)
            anchored=c['anchors']>=0
            mature=anchored.copy();mature[anchored]&=b.support[c['anchors'][anchored]]>=self.detector.cfg['confirm_observations']
            n=len(codes);sample=np.linspace(0,n-1,min(n,opt.max_reference_points)).astype(int)
            accept,stats=self.decide_candidate(c,confirmed,recent,codes,mature,sample,frame,as_numpy(pose))
            stats.update(full_reference_points=n,full_confirmed_points=int(confirmed.sum()),
                         registered_receipt_points=int(authorized.sum()),candidate_age=frame-c['created'])
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
        self.seconds+=time.perf_counter()-start
        if stats_rows:
            with (self.out/'candidate_stats.jsonl').open('a') as f:f.write(json.dumps(dict(frame=frame,candidates=stats_rows))+'\n')
        return retired

    def save(self):
        report=super().save()
        report.update(method='IG direct surface-ID whole retirement v3',receipt_samples=len(self.receipts),
            evidence_registration='raw ownership frozen at first accepted conflict, before local erasure',
            reference_domain='owned measured surface cells plus actual TSDF zero-crossing cells; not TSDF volume band',
            lineage_expansion='preserve old evidence; inherit confirmation only with exact surface-ID raw-owner receipt')
        (self.out/'summary.json').write_text(json.dumps(report,indent=2))
        (self.out/'receipt_stats.json').write_text(json.dumps(self.receipt_rows,indent=2))
        return report
