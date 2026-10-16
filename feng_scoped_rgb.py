"""Bounded old-label retirement for RGB, bounded geometry erasure for free depth.

No GT/session boundary is used. Frozen candidate evidence never authorizes
erasing the entire current descendant. Original detector thresholds stay intact.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def canonical_cells(points,voxel):
    return np.rint(np.asarray(points,dtype=np.float64)/voxel).astype(np.int64)

def repeated_evidence(candidate,cfg):
    n=cfg['confirm_observations']
    unopposed=candidate.local_support==0
    depth=(candidate.local_depth>=n)&unopposed&(candidate.current_sources==1)&(candidate.current_codes==2)
    rgb=(candidate.local_rgb>=n)&unopposed&(candidate.current_sources==2)&(candidate.current_codes==2)
    return depth,rgb

def update_local(candidate,reason,codes,sources,accepted):
    if reason in ('support','cancelled'):
        candidate.local_depth[:]=0;candidate.local_rgb[:]=0;candidate.local_support[:]=0
    elif accepted:
        candidate.local_depth+=((codes==2)&((sources&1)!=0)).astype(np.int16)
        candidate.local_rgb+=((codes==2)&((sources&2)!=0)).astype(np.int16)
        candidate.local_support+=(codes==1).astype(np.int16)
    candidate.current_codes=np.asarray(codes).copy();candidate.current_sources=np.asarray(sources).copy()

def current_geometry_free(xyz,depth,K,pose,near,far,cfg):
    """The current depth must lie behind a lattice node; unknown/occlusion abstain."""
    from feng_remove_detection import interior_mask
    q=(xyz-pose[:3,3])@pose[:3,:3];z=q[:,2];ok=np.isfinite(q).all(1)&(z>near)&(z<far)
    result=np.zeros(len(xyz),bool);ids=np.flatnonzero(ok)
    if not len(ids):return result
    u=np.rint(K[0,0]*q[ids,0]/z[ids]+K[0,2]).astype(int);v=np.rint(K[1,1]*q[ids,1]/z[ids]+K[1,2]).astype(int)
    inside=(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0]);ids=ids[inside];u=u[inside];v=v[inside]
    valid=np.isfinite(depth)&(depth>near)&(depth<far)
    interior=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
    result[ids]=interior[v,u]&(depth[v,u]>z[ids]+cfg['depth_epsilon_m']+cfg['depth_relative_epsilon']*z[ids])
    return result

def install():
    import torch
    import open3d.core as core
    import feng_remove_detection as detector
    import feng_ig_adapter as adapter
    from submap_disjoint.ig_correction import lattice_indices
    from submap_disjoint.sparse_edges import Edges
    original_post=detector.Candidate.__post_init__;original_observe=detector.Candidate.observe;original_metadata=detector.Candidate.metadata
    def post(self):
        original_post(self)
        self.local_depth=np.zeros(len(self.points),np.int16);self.local_rgb=self.local_depth.copy();self.local_support=self.local_depth.copy()
        self.current_codes=np.zeros(len(self.points),np.uint8);self.current_sources=self.current_codes.copy()
    def observe(self,frame,pose,codes,sources,cfg):
        previous=self.last_observation_frame
        reason,stats=original_observe(self,frame,pose,codes,sources,cfg)
        update_local(self,reason,np.asarray(codes),np.asarray(sources),self.last_observation_frame!=previous)
        depth,rgb=repeated_evidence(self,cfg)
        stats.update(local_depth_confirmed=int(depth.sum()),local_rgb_confirmed=int(rgb.sum()))
        if reason=='would_remove' and not (depth.any() or rgb.any()):
            self.state='suspected';self.confirmed_frame=None;reason='pending_repeated_local_evidence'
        return reason,stats
    def metadata(self):
        result=original_metadata(self)
        result.update(local_depth_vote_max=int(self.local_depth.max(initial=0)),local_rgb_vote_max=int(self.local_rgb.max(initial=0)))
        return result
    detector.Candidate.__post_init__=post;detector.Candidate.observe=observe;detector.Candidate.metadata=metadata
    original_step=adapter.FengIGCorrection.step;original_save=adapter.FengIGCorrection.save
    def step(self,frame,rgb,depth,K,pose):
        self.current_depth=np.asarray(depth);self.current_K=np.asarray(K);self.current_pose=np.asarray(pose)
        return original_step(self,frame,rgb,depth,K,pose)
    def save(self):
        original_save(self)
        if not hasattr(self,'detector'):return
        p=self.out/'summary.json';d=json.loads(p.read_text());d.update(policy='Frozen local support; repeated independent per-point evidence. RGB retires old local memberships while preserving geometry; current free-depth erases exclusive local geometry. No whole-descendant retirement.',variant='scoped_rgb_v1',rgb_enabled=True,ground_truth_used=False)
        p.write_text(json.dumps(d,indent=2))

    def delete_candidate(self,frame,candidate):
        o=self.owner;graph=o.maskGraph;grid=o.voxel_block_grids;live=graph.instance_dict
        slots=adapter.resolve_slots(candidate.source_raw_ids,o.global2merged_mask_id,live)
        row=dict(frame=int(frame),candidate_id=candidate.candidate_id,source_slot=candidate.source_slot,source_raw_ids=sorted(candidate.source_raw_ids),resolved_slots=slots,scope='frozen_reference_local',actual_erased_voxels=0,archive=None,removed_instance_records=0)
        depth_ok,rgb_ok=repeated_evidence(candidate,self.settings)
        row.update(depth_confirmed_reference_points=int(depth_ok.sum()),rgb_confirmed_reference_points=int(rgb_ok.sum()))
        if not slots or not (depth_ok.any() or rgb_ok.any()):
            row['status']='no_confirmed_local_support';self.actions.append(row);return
        target=np.concatenate([live[s].mask_voxel_coords.detach().cpu().numpy() for s in slots])
        xyz=np.unique(canonical_cells(target,self.voxel),axis=0)*self.voxel
        distance,nearest=cKDTree(candidate.points).query(xyz)
        bounded=distance<=np.sqrt(3)*self.voxel+1e-6
        kinds=np.where(depth_ok[nearest],1,np.where(rgb_ok[nearest],2,0))
        eligible=bounded&(kinds>0);xyz=xyz[eligible];kinds=kinds[eligible]
        row.update(target_nodes=len(target),bounded_evidence_nodes=len(xyz))
        if not len(xyz):
            row['status']='no_local_nodes';self.actions.append(row);return
        ids,found=lattice_indices(grid,xyz,self.voxel,self.res,o.device_o3c);xyz=xyz[found];kinds=kinds[found];ids=ids[found]
        # Protect non-target support under BOTH canonical coordinates and the
        # legacy hash indexing used elsewhere. Do not rewrite global mapping.
        states={};blocked=[];mismatch_count=0
        for mid,inst in live.items():
            coords=inst.mask_voxel_coords
            canonical,valid=lattice_indices(grid,coords.detach().cpu().numpy(),self.voxel,self.res,o.device_o3c)
            legacy=o.voxel_hash_table.voxel_coords2voxel_indices(coords).detach().cpu().numpy()
            mismatch_count+=int(np.count_nonzero(valid&(canonical!=legacy)))
            if mid in slots:states[mid]=(coords,canonical,valid,legacy)
            else:blocked.extend([canonical[valid],legacy])
        blocked=np.unique(np.concatenate(blocked)) if blocked else np.empty(0,np.int64)
        protected=np.isin(ids,blocked)
        # A hash membership absent from current coordinate records is still a
        # reason to abstain, not permission to erase another track's support.
        mapping=o.global2merged_mask_id
        for j,index in enumerate(ids):
            raw=o.voxel_hash_table.hash_table[int(index)]
            if raw and any(0<=r<len(mapping) and int(mapping[r])>=0 and int(mapping[r]) not in slots for r in raw):protected[j]=True
        row.update(shared_or_index_ambiguous_nodes_protected=int(protected.sum()),canonical_legacy_index_mismatches=mismatch_count)
        xyz=xyz[~protected];kinds=kinds[~protected];ids=ids[~protected]
        free=current_geometry_free(xyz,self.current_depth,self.current_K,self.current_pose,self.detector.near,self.detector.far,self.settings)
        keep=(kinds==2)|((kinds==1)&free)
        row['current_depth_guard_rejected']=int(np.count_nonzero((kinds==1)&~free))
        xyz=xyz[keep];kinds=kinds[keep];ids=ids[keep]
        if not len(ids):
            row['status']='protected_or_not_currently_free';self.actions.append(row);return
        assert not np.isin(ids,blocked).any(),'Non-target membership would be modified'
        depth_idx=ids[kinds==1];depth_xyz=xyz[kinds==1]
        actiondir=self.out/'actions'/str(candidate.candidate_id);actiondir.mkdir(parents=True,exist_ok=True)
        # RGB is a label-change event, not proof of empty space. Preserve all
        # TSDF geometry in this branch, and let current RGB-D/masks refresh it.
        if len(depth_idx):
            oid=core.Tensor(depth_idx,dtype=core.int64,device=o.device_o3c)
            w=grid.attribute('weight').reshape((-1,))[oid].cpu().numpy()>0
            depth_idx=depth_idx[w];depth_xyz=depth_xyz[w]
            if len(depth_idx):
                oid=core.Tensor(depth_idx,dtype=core.int64,device=o.device_o3c)
                archived=dict(xyz=depth_xyz,indices=depth_idx)
                for attr,width in [('tsdf',1),('weight',1),('color',3)]:
                    values=grid.attribute(attr).reshape((-1,width));archived[attr]=values[oid].cpu().numpy();values[oid]=0.
                path=actiondir/f'{frame}.npz';np.savez_compressed(path,**archived);row['archive']=str(path)
                self.deleted.update(map(tuple,canonical_cells(depth_xyz,self.voxel)))
                tid=torch.as_tensor(depth_idx,device=o.device,dtype=torch.long);graph.voxel_in_frame_matrix[tid]=False
                if graph.boundary_voxel_indices is not None:graph.boundary_voxel_indices=graph.boundary_voxel_indices[~torch.isin(graph.boundary_voxel_indices,tid)]
        labelpath=actiondir/f'{frame}_labels.npz';np.savez_compressed(labelpath,xyz=xyz,indices=ids,evidence_kind=kinds)
        removed=0;affected=[];empty=[];legacy_drop=[];legacy_kept=[]
        for mid,(coords,canonical,valid,legacy) in states.items():
            drop=valid&np.isin(canonical,ids)
            legacy_drop.append(legacy[drop]);legacy_kept.append(legacy[~drop])
            if not drop.any():continue
            affected.append(int(mid));removed+=int(drop.sum())
            keep_t=torch.as_tensor(~drop,device=coords.device);inst=live[mid]
            inst.mask_voxel_coords=coords[keep_t]
            inst.mask_voxel_indices=torch.as_tensor(legacy[~drop],device=o.device,dtype=torch.long)
            o.mask_voxels_coords[mid]=inst.mask_voxel_coords;graph.mask_voxels_coords[mid]=inst.mask_voxel_coords
            o.merged_mask_size[mid]=int((~drop).sum());o.mask_geo_features[mid]=0;o.mask_geo_features_mask[mid]=0
            if not (~drop).any():empty.append(mid)
        # Remove retired raw-mask support at the legacy lookup addresses too,
        # while preserving addresses still used by a retained target node.
        hash_ids=np.unique(np.concatenate([ids]+legacy_drop))
        if legacy_kept:hash_ids=hash_ids[~np.isin(hash_ids,np.concatenate(legacy_kept))]
        for index in hash_ids:
            raw=o.voxel_hash_table.hash_table[int(index)]
            surviving={r for r in (raw or ()) if not (0<=r<len(mapping) and int(mapping[r]) in slots)}
            o.voxel_hash_table.hash_table[int(index)]=surviving or None
            o.voxel_hash_table.hash_item_flag[int(index)]=bool(surviving)
        if empty:o.remove_masks(frame,torch.tensor(empty,device=o.device,dtype=torch.long))
        o.edges=Edges()
        for mid,inst in graph.instance_dict.items():
            overlaps,forward,backward=graph.query_mask_under_visible_part(graph.merged_mask_in_frame_mat[mid],inst.mask_voxel_coords,o.global2merged_mask_id)
            if overlaps.numel():o.edges.write_row_col(mid,overlaps,forward,backward)
            o.edges.set(mid,mid,1.)
        if len(depth_idx):
            for obj in (o,o.geometry):
                obj.pc_frame_id=-1;obj.pc_feature_frame_id=-1;obj.points=None;obj.colors=None;obj.pc_xyz=None;obj.pc_feature=None
        o.pred_inst_mask_frame_id=-1;o.pred_inst_masks=None
        row.update(status='applied',actual_erased_voxels=int(len(depth_idx)),local_rgb_label_nodes=int(np.count_nonzero(kinds==2)),local_depth_label_nodes=int(np.count_nonzero(kinds==1)),removed_instance_records=len(empty),removed_instance_voxel_memberships=removed,affected_current_ids=affected,label_archive=str(labelpath),non_target_affected=0)
        self.actions.append(row)
    adapter.FengIGCorrection.step=step;adapter.FengIGCorrection.save=save;adapter.FengIGCorrection.delete_candidate=delete_candidate
