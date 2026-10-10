"""Simple IG-only detector using a shared TSDF and instance surface references.

No local TSDF construction, RGB calibration, render ownership gate, global
conflict veto, maturity count or whole-object retirement threshold.
The inherited eraser retains its geometry/graph updates and archive format.
"""
from dataclasses import asdict, replace
import json,time
import numpy as np
from scipy.spatial import cKDTree
from .ig_correction import IGCorrection, frozen_grid_copy, lattice_indices, corners, unique_owner, free_now
from .contradiction import AuditOptions,sample_surface,FREE,SUPPORT
from .ig_visible_evidence import VisibleOptions,TwoViewEvidence,project_evidence


class VisibleIGCorrection(IGCorrection):
    def __init__(self,owner,directory,enabled=True,options=None):
        super().__init__(owner,directory,enabled=enabled,rgb=False)
        self.opt=options or VisibleOptions()
        self.front=replace(self.front,rgb_enabled=False,depth_epsilon_m=self.opt.depth_abs_m,
            depth_relative_epsilon=self.opt.depth_relative,edge_threshold_m=self.opt.edge_m,boundary_margin_px=0)
        self.audit=AuditOptions(near_m=owner.cfg['cam']['depth_near'],far_m=owner.depth_far,cell_m=self.voxel)
        self.counts={}

    def freeze(self,frame):
        started=time.perf_counter();o=self.owner
        grid=frozen_grid_copy(o.voxel_block_grids,self.voxel,self.res,o.device_o3c)
        self.frozen_grid=grid
        self.slots=np.full(int(grid.attribute('weight').num_elements()),-1,np.int32)
        memberships={}
        for mid,inst in o.maskGraph.instance_dict.items():
            ids,ok=lattice_indices(grid,inst.mask_voxel_coords.cpu().numpy(),self.voxel,self.res,o.device_o3c)
            ids=np.unique(ids[ok]);memberships[mid]=ids
            prior=self.slots[ids];self.slots[ids]=np.where(prior==-1,mid,np.where(prior==mid,mid,-2))
        pc=grid.extract_point_cloud(weight_threshold=self.front.min_map_weight)
        xyz=pc.point.positions.cpu().numpy()
        ids,ok=lattice_indices(grid,corners(xyz,self.voxel).reshape(-1,3),self.voxel,self.res,o.device_o3c)
        ids=ids.reshape(-1,8);ok=ok.reshape(-1,8)
        labels=unique_owner(self.slots[np.clip(ids,0,len(self.slots)-1)],ok)
        arrays=dict(all_points=xyz,all_owners=labels)
        for mid,inst in o.maskGraph.instance_dict.items():
            # Ambiguous membership is permitted for detection but not erasure.
            owned=np.any(np.isin(ids,memberships[mid])&ok,axis=1)
            full=xyz[owned];points=sample_surface(full,self.audit)
            if not len(points):
                self.skips[str(mid)]=dict(reason='no_reference_surface',cells=0);continue
            nearest=cKDTree(points).query(full)[1]
            self.refs[mid]=dict(points=points,full=full,nearest=nearest,window=TwoViewEvidence(len(points),self.opt),
                observations=len(set(f for f,_ in inst.ori_mask_list)),track_uid=int(inst.track_uid),
                raw_observations=inst.ori_mask_list,state='stable',candidates=0,max_free=0.,
                max_confirmed_free=0.,deleted_voxels=0,stats={})
            arrays['points_'+str(mid)]=points
        np.savez_compressed(self.out/'reference_surfaces.npz',**arrays)
        self.initialized=True;self.freeze_frame=int(frame);self.freeze_seconds=time.perf_counter()-started
        self.save()

    def step(self,frame,rgb,depth,K,pose):
        started=time.perf_counter();self.tick+=1;self.frame_count+=1
        rows=[];requests=[]
        for mid,r in self.refs.items():
            codes,d=project_evidence(r['points'],depth,K,pose,self.audit.near_m,self.audit.far_m,self.opt)
            confirmed=r['window'].update(self.tick,pose,codes)
            cf=int(confirmed.sum());den=d['visible_cells'];ratio=cf/max(den,1)
            d.update(reference_id=int(mid),confirmed_free_cells=cf,confirmed_visible_ratio=ratio,
                tested_reference=den,reference_conflict_ratio=d['visible_conflict_ratio'],
                visible_point_indices=np.flatnonzero((codes==FREE)|(codes==SUPPORT)).tolist(),
                free_point_indices=np.flatnonzero(codes==FREE).tolist(),
                confirmed_point_indices=np.flatnonzero(confirmed).tolist())
            r['max_free']=max(r['max_free'],d['visible_conflict_ratio'])
            r['max_confirmed_free']=max(r['max_confirmed_free'],ratio)
            r['candidates']+=int(d['candidate'])
            ready=d['candidate'] and ratio>=self.opt.conflict_ratio
            reasons=list(d['candidate_failures'])
            if d['candidate'] and not ready:reasons.append('awaiting_repeat_confirmation')
            r['state']='confirmed_local' if ready else ('suspect' if d['candidate'] else 'stable')
            d.update(state=r['state'],confirmed_region=bool(ready),eligible_erasure_corners=0)
            if ready:
                full=r['full'][confirmed[r['nearest']]]
                cs=np.unique(np.rint(corners(full,self.voxel).reshape(-1,3)/self.voxel).astype(np.int64),axis=0)*self.voxel
                ix,ok=lattice_indices(self.frozen_grid,cs,self.voxel,self.res,self.owner.device_o3c)
                owned=ok&(self.slots[np.clip(ix,0,len(self.slots)-1)]==mid)
                d['ambiguous_or_other_owner_corners']=int((~owned).sum())
                cs=cs[owned]
                cs=cs[free_now(cs,depth,K,pose,self.front,self.audit.near_m,self.audit.far_m)]
                d['eligible_erasure_corners']=len(cs)
                if not len(cs):reasons.append('no_safe_owned_free_corners')
                elif self.enabled:requests.append((mid,cs))
            d['failure_reasons']=reasons;r['stats']=d;rows.append(d)
            for reason in reasons:self.counts[reason]=self.counts.get(reason,0)+1
        deletion=self.erase(frame,requests) if requests else {}
        self.seconds+=time.perf_counter()-started
        with (self.out/'frames.jsonl').open('a') as f:
            f.write(json.dumps(dict(frame=int(frame),references=rows,deletion=deletion))+'\n')
        self.save()

    def save(self):
        refs={str(mid):{k:r[k] for k in ['observations','track_uid','raw_observations','state','candidates',
            'max_free','max_confirmed_free','deleted_voxels','stats']}|dict(cells=len(r['points']),surface_points=len(r['full'])) for mid,r in self.refs.items()}
        s=dict(method='IG visible-region depth contradiction',detector_version='ig-visible-v3',
            options=asdict(self.opt),deletion_enabled=self.enabled,ground_truth_used=False,
            fixed_reference_session=True,freeze_frame=getattr(self,'freeze_frame',None),
            freeze_seconds=getattr(self,'freeze_seconds',0.),audit_seconds=self.seconds,audited_frames=self.frame_count,
            unique_deleted_world_voxels=len(self.deleted),references=refs,skips=self.skips,events=self.events,
            failure_reasons=self.counts,scope='confirmed local cells only; no whole-instance retirement',
            denominator='current SUPPORT or FREE pixels; one closest fixed surface sample per pixel')
        (self.out/'summary.json').write_text(json.dumps(s,indent=2))
