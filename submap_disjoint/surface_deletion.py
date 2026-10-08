"""Local stale-surface retirement, independent of GT and scan-session labels.

The detector proposes measured surface samples. A pending sample freezes raw
mask lineage on existing lattice corners. Confirmation never authorizes a whole
track/AABB: each corner needs fresh free-space, no nearby measured surface, and
no newer raw owner. Shared TSDF reset is not independent-submap deactivation.
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .readonly_detection_core import interior_mask, classify_depth, CONTRADICTION, as_numpy
from .surface_bank import SurfaceBank
from .ig_correction import corners, lattice_indices


@dataclass(frozen=True)
class DeletionOptions:
    min_component_points: int = 20
    component_radius_voxels: float = 3.
    protect_surface_radius_voxels: float = 2.
    max_abs_tsdf: float = .5
    verify_state: bool = False

    def __post_init__(self):
        if self.min_component_points<1 or self.component_radius_voxels<=0:
            raise ValueError('Invalid component policy')
        if self.protect_surface_radius_voxels<0 or not 0<self.max_abs_tsdf<=1:
            raise ValueError('Invalid surface protection policy')


def confirmed_components(points, minimum, radius):
    """Spatial grouping limits isolated speckles; grouping never fills geometry."""
    n=len(points)
    if n<minimum:return np.zeros(n,bool)
    edges=cKDTree(points).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(n,n))
    _,labels=connected_components(graph,directed=False)
    return np.bincount(labels)[labels]>=minimum


def active_lineage(members,mapping):
    return frozenset(int(i) for i in (members or ()) if 0<=int(i)<len(mapping) and mapping[int(i)]>=0)


def unchanged_lineage(members,frozen,mapping):
    """Merged row renumbering is fine. New raw observations are not old owners."""
    now=active_lineage(members,mapping)
    return bool(now) and bool(frozen) and now<=frozen


def free_without_nearby_surface(points,depth,K,pose,cfg,near,far,radius):
    """Camera Z in meters. Unknown/occluded/current occupied cells cannot erase."""
    points=np.asarray(points).reshape(-1,3);n=len(points)
    free=np.zeros(n,bool);protected=np.zeros(n,bool)
    camera=(points-pose[:3,3])@pose[:3,:3];z=camera[:,2]
    ids=np.flatnonzero(np.isfinite(camera).all(1)&(z>near)&(z<far))
    uv=np.rint((camera[ids]@K.T)[:,:2]/z[ids,None]).astype(np.int64)
    h,w=depth.shape;inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ids,uv=ids[inside],uv[inside]
    valid=np.isfinite(depth)&(depth>near)&(depth<far)
    interior=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
    free[ids]=classify_depth(depth[uv[:,1],uv[:,0]],z[ids],interior[uv[:,1],uv[:,0]],
                            cfg['depth_epsilon_m'],cfg['depth_relative_epsilon'])==CONTRADICTION
    candidates=np.flatnonzero(free)
    if len(candidates) and radius>0:
        y,x=np.nonzero(valid)
        cloud=(np.column_stack((x,y,np.ones(len(x))))@np.linalg.inv(K).T)*depth[y,x,None]
        distance,_=cKDTree(cloud).query(camera[candidates],distance_upper_bound=radius)
        protected[candidates[np.isfinite(distance)]]=True
    return free&~protected,dict(free_before_protection=int(free.sum()),nearby_veto=int(protected.sum()))


class IGSurfaceDetector:
    """Same strict raw RGB-D detector as the dense-rescan read-only experiment."""
    def __init__(self,owner,cfg,output):
        self.owner=owner;self.cfg=cfg['remove_detection'];self.cam=cfg['cam']
        self.K=as_numpy(owner.intrinsics).astype(np.float32)
        self.bank=SurfaceBank(cfg['scene']['voxel_size'],self.cfg,require_color=True,
            view_gate='ray',surface_support=True,strict_history_support=True)
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.rows=[];self.seconds=0.

    def step(self,frame,rgb,depth,pose,valid_pose=True):
        if not valid_pose:return False
        started=time.perf_counter();depth=as_numpy(depth).astype(np.float32).copy()
        pose=as_numpy(pose).astype(np.float32);rgb=as_numpy(rgb)
        if not np.isfinite(pose).all():return False
        valid=np.isfinite(depth)&(depth>self.cam['depth_near'])&(depth<self.cam['depth_far'])
        depth[~valid]=0
        inner=interior_mask(depth,valid,self.cfg['edge_threshold_m'],self.cfg['boundary_margin_px'])
        # Restore byte RGB before the same normalization/blur as the probe.
        if rgb.max(initial=0)<=1:rgb=np.rint(rgb*255).astype(np.uint8)
        self.rows.append(self.bank.step(frame,pose,self.K,depth,inner,self.cam['depth_near'],self.cam['depth_far'],rgb))
        self.seconds+=time.perf_counter()-started
        return True

    def finalize(self):
        np.savez_compressed(self.out/'bank.npz',**self.bank.arrays())
        report=dict(method='strict RGB-D surface memory',frames=self.rows,seconds=self.seconds,
                    selected_frame_ids=[r['frame'] for r in self.rows],GT_used=False,
                    cfg=self.cfg,voxel_size=self.bank.voxel_size)
        (self.out/'summary.json').write_text(json.dumps(report,indent=2))
        return report


class IGSurfaceDeletion:
    def __init__(self,owner,detector,output,options=None,enabled=False):
        self.owner=owner;self.detector=detector;self.bank=detector.bank
        self.options=options or DeletionOptions();self.enabled=enabled
        self.voxel=self.bank.voxel_size;self.res=owner.cfg['scene']['block_resolution']
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.pending={};self.events=[];self.seconds=0.;self.last_frame=-1
        self.executor=None
        if enabled:
            from .ig_deletion import IGDeletion
            self.executor=IGDeletion(owner,None,self.out/'executor')

    def _freeze(self,point_ids):
        """Freeze existing lattice ownership at the start of an evidence streak."""
        if not len(point_ids):return
        o=self.owner
        xyz=corners(self.bank.points[point_ids],self.voxel)
        ids,exists=lattice_indices(o.voxel_block_grids,xyz.reshape(-1,3),self.voxel,self.res,o.device_o3c)
        ids=ids.reshape(-1,8);exists=exists.reshape(-1,8)
        for p,cs,indices,valid in zip(point_ids,xyz,ids,exists):
            records=[]
            for c,idx,ok in zip(cs,indices,valid):
                if not ok:continue
                lineage=active_lineage(o.voxel_hash_table.hash_table[int(idx)],o.global2merged_mask_id)
                if lineage:records.append((tuple(np.rint(c/self.voxel).astype(int)),lineage))
            self.pending[int(p)]=records

    def step(self,frame,depth,pose):
        if frame!=self.bank.last_frame or frame<=self.last_frame:raise ValueError('Must follow exactly one current detector step')
        self.last_frame=frame;start=time.perf_counter();o=self.owner
        b=self.bank;cfg=self.detector.cfg;cam=self.detector.cam
        reset=[p for p in self.pending if b.streak[p]==0]
        for p in reset:del self.pending[p]
        # First accepted contradiction is observable from streak and timestamp;
        # no change to SurfaceBank or its evidence/confirmation thresholds.
        fresh=np.flatnonzero((b.streak>0)&(b.last_conflict_frame==frame))
        self._freeze(np.array([int(p) for p in fresh if int(p) not in self.pending],np.int64))
        active=np.flatnonzero(b.active)
        eligible=active[confirmed_components(b.points[active],self.options.min_component_points,
                                             self.options.component_radius_voxels*self.voxel)]
        # Old confirmation alone cannot delete something that returned later:
        # require the present depth to independently classify each corner as free.
        proposals={}
        for p in eligible:
            for key,lineage in self.pending.get(int(p),[]):
                # Multiple old samples may share a corner. Each lease is checked
                # separately; do not union unrelated lineage sets into permission.
                proposals.setdefault(key,[]).append((int(p),lineage))
        proposed=len(proposals);allowed_xyz=[];allowed_points=set()
        diagnostics=dict(free_before_protection=0,nearby_veto=0,existing=0,lineage_veto=0,tsdf_veto=0)
        if proposals:
            keys=list(proposals);xyz=np.asarray(keys,np.float64)*self.voxel
            free,check=free_without_nearby_surface(xyz,as_numpy(depth),self.detector.K,
                as_numpy(pose),cfg,cam['depth_near'],cam['depth_far'],self.options.protect_surface_radius_voxels*self.voxel)
            diagnostics.update(check)
            xyz=xyz[free];keys=[keys[i] for i in np.flatnonzero(free)]
            ids,exists=lattice_indices(o.voxel_block_grids,xyz,self.voxel,self.res,o.device_o3c) if len(xyz) else (np.empty(0,int),np.empty(0,bool))
            import open3d.core as core
            live_ids=ids[exists];diagnostics['existing']=len(live_ids)
            if len(live_ids):
                tid=core.Tensor(live_ids,dtype=core.int64,device=o.device_o3c)
                weight=o.voxel_block_grids.attribute('weight').reshape((-1,))[tid].cpu().numpy()
                tsdf=o.voxel_block_grids.attribute('tsdf').reshape((-1,))[tid].cpu().numpy()
                for key,pos,idx,w,t in zip([keys[i] for i in np.flatnonzero(exists)],xyz[exists],live_ids,weight,tsdf):
                    if not np.isfinite(w) or not np.isfinite(t) or w<cfg['min_map_weight'] or abs(t)>self.options.max_abs_tsdf:
                        diagnostics['tsdf_veto']+=1;continue
                    leases=[p for p,old in proposals[key] if unchanged_lineage(o.voxel_hash_table.hash_table[int(idx)],old,o.global2merged_mask_id)]
                    if not leases:diagnostics['lineage_veto']+=1;continue
                    allowed_xyz.append(pos);allowed_points.update(leases)
        result=dict(voxels=0)
        if allowed_xyz and self.enabled:
            xyz=np.asarray(allowed_xyz)
            # Archive source evidence before TSDF/semantic mutation, with world
            # coordinates (hash slots can be reused later).
            ids=np.array(sorted(allowed_points),np.int64)
            np.savez_compressed(self.out/f'evidence_{frame:06d}.npz',surface_ids=ids,
                points=b.points[ids],streak=b.streak[ids],support=b.support[ids],
                created_frame=b.created_frame[ids],last_conflict_frame=b.last_conflict_frame[ids],authorized_xyz=xyz)
            result=self.executor.erase(frame,[(int(p),xyz) for p in [-1]])
            if result['voxels']:
                # erase() prunes memberships and source-mask clouds, invalidates
                # features, removes empty rows and rebuilds containment edges.
                # Refresh per-row AABBs after any row compaction as well.
                import torch
                o.merged_mask_bbox.zero_()
                for mid,inst in o.maskGraph.instance_dict.items():
                    c=inst.mask_voxel_coords
                    if len(c):o.merged_mask_bbox[mid]=torch.cat((c.min(0).values,c.max(0).values))
                if self.options.verify_state:self.verify_erasure(xyz)
        elapsed=time.perf_counter()-start;self.seconds+=elapsed
        row=dict(frame=int(frame),reset_pending=len(reset),pending_surface_points=len(self.pending),
            confirmed_surface_points=len(active),component_eligible_points=len(eligible),
            proposed_corners=proposed,authorized_corners=len(allowed_xyz),enabled=self.enabled,
            seconds=elapsed,**diagnostics,**result)
        self.events.append(row)
        with (self.out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return row

    def verify_erasure(self,xyz):
        import torch
        import open3d.core as core
        o=self.owner;ids,ok=lattice_indices(o.voxel_block_grids,xyz,self.voxel,self.res,o.device_o3c)
        ids=ids[ok];oid=core.Tensor(ids,dtype=core.int64,device=o.device_o3c)
        assert (o.voxel_block_grids.attribute('weight').reshape((-1,))[oid].cpu().numpy()==0).all()
        assert all(not o.voxel_hash_table.hash_table[int(i)] for i in ids)
        tid=torch.as_tensor(ids,device=o.device,dtype=torch.long)
        assert not o.voxel_hash_table.hash_item_flag[tid].any()
        assert not o.maskGraph.voxel_in_frame_matrix[tid].any()
        for mid,inst in o.maskGraph.instance_dict.items():
            assert not torch.isin(inst.mask_voxel_indices,tid).any()
            assert len(inst.mask_voxel_coords)==o.merged_mask_size[mid]
        for coords in o.maskGraph.f_mask_voxels_clouds.values():
            if len(coords):assert not torch.isin(o.voxel_hash_table.voxel_coords2voxel_indices(coords),tid).any()

    def save(self):
        report=dict(method='IG local surface retirement v1',enabled=self.enabled,GT_used=False,
            whole_instance_deletion=False,independent_TSDF_deactivation=False,options=asdict(self.options),
            seconds=self.seconds,frames=len(self.events),deletion_frames=sum(r['voxels']>0 for r in self.events),
            erased_voxel_operations=sum(r['voxels'] for r in self.events),
            unique_erased_world_voxels=len(self.executor.deleted) if self.executor else 0,
            pending_surface_points=len(self.pending),events=self.events)
        (self.out/'summary.json').write_text(json.dumps(report,indent=2))
        return report
