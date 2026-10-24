"""Causal IG correction against a frozen reference-session shared TSDF.

No GT enters this module. Session boundary is supplied by the runner. RGB-only
conflicts propose suspicion; only multi-view FREE evidence authorizes erasure.
This is not reversible independent-TSDF deactivation: shared weights are reset.
"""
from dataclasses import asdict
from types import SimpleNamespace
from pathlib import Path
import itertools
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from .change_frontend import (FrontendOptions, prepare_context, fragment_evidence,
                             raycast_validity, classify, interior)
from .change_lifecycle import LifecycleOptions, EvidenceWindow
from .contradiction import AuditOptions, sample_surface, FREE, SUPPORT, validate_camera


def frozen_grid_copy(source, voxel, resolution, device):
    """Explicit independent attributes: this Open3D build aliases VBG.to(copy=True)."""
    import open3d as o3d
    import open3d.core as core
    active = source.hashmap().active_buf_indices().to(core.int64)
    keys = source.hashmap().key_tensor()[active]
    target = o3d.t.geometry.VoxelBlockGrid(('tsdf','weight','color'), (core.float32,)*3,
        (1,1,3), voxel, resolution, max(1,int(keys.shape[0])*2), device)
    target.hashmap().activate(keys)
    slots, ok = target.hashmap().find(keys)
    if not bool(ok.all().item()): raise RuntimeError('Snapshot allocation failed')
    slots = slots.to(core.int64)
    for attr, width in [('tsdf',1),('weight',1),('color',3)]:
        target.attribute(attr).reshape((-1,resolution**3,width))[slots] = source.attribute(attr).reshape((-1,resolution**3,width))[active].clone()
    return target


def lattice_indices(grid, xyz, voxel, resolution, device):
    """World lattice -> current hash slots. Never reuse snapshot buffer IDs."""
    import open3d.core as core
    cells = np.rint(np.asarray(xyz) / voxel).astype(np.int64)
    blocks = np.floor_divide(cells, resolution).astype(np.int32)
    offsets = cells - blocks.astype(np.int64) * resolution
    b, ok = grid.hashmap().find(core.Tensor(blocks, device=device))
    indices = b.cpu().numpy().astype(np.int64) * resolution**3
    indices += offsets[:, 0] + resolution * offsets[:, 1] + resolution**2 * offsets[:, 2]
    return indices, ok.cpu().numpy().astype(bool)


def corners(points, voxel):
    base = np.floor(np.asarray(points) / voxel).astype(np.int64)
    offsets = np.array(list(itertools.product((0, 1), repeat=3)))
    return (base[:, None, :] + offsets[None, :, :]) * voxel


def unique_owner(slots, useful):
    """Unknown corners are ignored; multiple owners or ambiguous corners reject."""
    valid = useful & (slots >= 0)
    lo = np.where(valid, slots, np.iinfo(np.int32).max).min(-1)
    hi = np.where(valid, slots, -1).max(-1)
    answer = np.where(hi < 0, -1, hi)
    answer[(hi >= 0) & (lo != hi) | ((slots == -2) & useful).any(-1)] = -2
    return answer


def confirmed_free(window, n, options):
    support, free = window.counts(n)
    return (free >= options.min_views) & (free / np.maximum(1, support + free) >= options.cell_consensus)


def free_now(points, depth, K, pose, options, near, far):
    """Safety check before erasure: occluded/new occupied geometry cannot pass."""
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    z = cam[:, 2]; result = np.zeros(len(points), bool)
    ids = np.flatnonzero(np.isfinite(cam).all(1) & (z > near) & (z < far))
    u = np.rint(K[0, 0] * cam[ids, 0] / z[ids] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * cam[ids, 1] / z[ids] + K[1, 2]).astype(int)
    valid = (u >= 0) & (v >= 0) & (u < depth.shape[1]) & (v < depth.shape[0])
    ids, u, v = ids[valid], u[valid], v[valid]
    safe = interior(depth, np.isfinite(depth) & (depth > near) & (depth < far),
                    options.edge_threshold_m, options.boundary_margin_px)
    result[ids] = classify(depth[v, u], z[ids], safe[v, u], options) == FREE
    return result


class IGCorrection:
    def __init__(self, owner, directory, enabled=True, rgb=True):
        self.owner = owner; self.out = Path(directory); self.out.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.front = FrontendOptions(enabled=True, interval=owner.cfg['mapping']['keyframe_freq'], rgb_enabled=rgb)
        self.life = LifecycleOptions()
        self.audit = AuditOptions(near_m=owner.cfg['cam']['depth_near'], far_m=owner.depth_far)
        self.voxel = owner.cfg['scene']['voxel_size']; self.res = owner.cfg['scene']['block_resolution']
        self.refs = {}; self.tick = 0; self.events = []; self.initialized = False
        self.deleted = set(); self.seconds = 0.; self.skips = {}; self.frame_count = 0
        self.archive = self.out / 'erasure_archive'; self.archive.mkdir(exist_ok=True)

    def freeze(self, frame):
        import open3d.core as core
        started = time.perf_counter(); o = self.owner
        grid = frozen_grid_copy(o.voxel_block_grids, self.voxel, self.res, o.device_o3c)
        self.volume = SimpleNamespace(grid=grid, device=o.device_o3c, anchor=np.eye(4),
            options=SimpleNamespace(near_m=self.audit.near_m, far_m=self.audit.far_m,
                                    trunc_voxels=o.trunc_voxel_multiplier))
        # Frozen ownership uses snapshot hash indices, NOT mutable merged IDs.
        self.slots = np.full(int(grid.attribute('weight').num_elements()), -1, np.int32)
        eligible = {}
        for mid, inst in o.maskGraph.instance_dict.items():
            observations = len(set(f for f, _ in inst.ori_mask_list))
            coords = inst.mask_voxel_coords.detach().cpu().numpy()
            ids, ok = lattice_indices(grid, coords, self.voxel, self.res, o.device_o3c)
            ids = ids[ok]
            prior = self.slots[ids]
            self.slots[ids] = np.where(prior == -1, mid, np.where(prior == mid, mid, -2))
            if observations >= self.front.min_instance_observations:
                eligible[mid] = dict(observations=observations, raw_observations=inst.ori_mask_list,
                                     track_uid=int(inst.track_uid))
        pc = grid.extract_point_cloud(weight_threshold=self.front.min_map_weight)
        xyz = pc.point.positions.cpu().numpy()
        labels = []
        for start in range(0, len(xyz), 50000):
            cs = corners(xyz[start:start+50000], self.voxel)
            ids, ok = lattice_indices(grid, cs.reshape(-1, 3), self.voxel, self.res, o.device_o3c)
            slots = self.slots[np.clip(ids, 0, len(self.slots)-1)].reshape(-1, 8)
            labels.append(unique_owner(slots, ok.reshape(-1, 8)))
        labels = np.concatenate(labels) if labels else np.zeros(0, np.int32)
        ref_arrays = dict(all_points=xyz, all_owners=labels)
        for mid, meta in eligible.items():
            full = xyz[labels == mid]; points = sample_surface(full, self.audit)
            if len(points) < self.front.min_reference_points:
                self.skips[str(mid)] = dict(reason='insufficient_reference_cells', cells=len(points)); continue
            distance, nearest = cKDTree(points).query(full)
            self.refs[mid] = dict(meta, points=points, full=full, nearest=nearest,
                window=EvidenceWindow(self.life), state='stable', candidates=0, max_free=0.,
                max_confirmed_free=0., deleted_voxels=0)
            ref_arrays['points_'+str(mid)] = points
        np.savez_compressed(self.out/'reference_surfaces.npz', **ref_arrays)
        self.initialized = True; self.freeze_frame = int(frame)
        self.freeze_seconds = time.perf_counter()-started
        self.save()

    def render(self, K, pose, shape):
        import open3d.core as core
        grid = self.volume.grid; h, w = shape
        active = grid.hashmap().active_buf_indices().to(core.int64)
        blocks = grid.hashmap().key_tensor()[active]
        intrinsic = core.Tensor(np.asarray(K, np.float64)); ext = core.Tensor(np.linalg.inv(pose).astype(np.float64))
        kwargs = dict(depth_scale=1., depth_min=self.audit.near_m, depth_max=self.audit.far_m,
            weight_threshold=self.front.min_map_weight,
            trunc_voxel_multiplier=self.owner.trunc_voxel_multiplier, range_map_down_factor=1)
        if self.front.raycast_warmup:
            b = blocks.cpu().numpy()
            grid.ray_cast(core.Tensor(np.concatenate((b, b)), device=self.volume.device),
                          intrinsic, ext, w, h, ['depth'], **kwargs)
        attrs = ['depth', 'index', 'mask', 'interp_ratio'] + (['color'] if self.front.rgb_enabled else [])
        result = grid.ray_cast(blocks, intrinsic, ext, w, h, attrs, **kwargs)
        depth = result['depth'].cpu().numpy().reshape(shape).copy()
        indices = result['index'].cpu().numpy(); ratio = result['interp_ratio'].cpu().numpy()
        valid = result['mask'].cpu().numpy() & (indices >= 0) & (indices < len(self.slots))
        safe = np.clip(indices, 0, len(self.slots)-1)
        sampled = grid.attribute('weight').reshape((-1,))[core.Tensor(safe.reshape(-1),device=self.volume.device)].cpu().numpy().reshape(indices.shape)
        quality, weight, _ = raycast_validity(depth, indices, valid, ratio, sampled, self.front, self.audit.near_m, self.audit.far_m)
        owners = unique_owner(self.slots[safe], valid & (ratio > 1e-6)).reshape(shape)
        # Shared baseline integrates uint8; Open3D stores extracted color in 0..255.
        color = np.clip(result['color'].cpu().numpy().reshape(shape+(3,))/255., 0, 1) if self.front.rgb_enabled else np.zeros(shape+(3,),np.float32)
        return dict(depth=depth, valid=quality, weight=weight, owners=owners, color=color)

    def step(self, frame, rgb, depth, K, pose):
        started = time.perf_counter(); self.tick += 1; self.frame_count += 1
        validate_camera(K, pose)
        history = self.render(K, pose, depth.shape)
        context = prepare_context(rgb, depth, history, self.front, self.audit.near_m, self.audit.far_m)
        rows = []; erasure = []
        for mid, r in self.refs.items():
            codes, appearance, diag = fragment_evidence(r['points'], rgb, depth, K, pose,
                history, history, context, mid, r['observations'], self.front,
                self.audit.near_m, self.audit.far_m, self.audit.map_gate_m)
            # Global anomaly must not update votes. Unknown still expires old votes.
            if context['skip_reasons']: codes[:] = 0
            r['window'].update(self.tick, pose, codes)
            free = confirmed_free(r['window'], len(codes), self.life)
            fraction = float(free.mean()); r['max_confirmed_free'] = max(r['max_confirmed_free'],fraction)
            r['max_free'] = max(r['max_free'], float((codes == FREE).mean()))
            before = r['state']
            if diag['candidate']:
                r['candidates'] += 1
                if before == 'stable': r['state'] = 'suspect'
            if (r['candidates'] and free.sum() >= self.life.min_cells and fraction >= self.life.retire_fraction):
                r['state'] = 'retired'
            if before != r['state']:
                self.events.append(dict(frame=int(frame), reference_id=mid, before=before, after=r['state'], confirmed_free_fraction=fraction))
            if r['state'] == 'retired' and self.enabled and not context['skip_reasons']:
                full = r['full'][free[r['nearest']]]
                # Clear only corners around confirmed old surface, not whole AABB/track.
                cs = np.unique(np.rint(corners(full,self.voxel).reshape(-1,3)/self.voxel).astype(np.int64),axis=0)*self.voxel
                cs = cs[free_now(cs, depth, K, pose, self.front, self.audit.near_m, self.audit.far_m)]
                if len(cs): erasure.append((mid,cs))
            rows.append(dict(reference_id=mid,state=r['state'],confirmed_free_cells=int(free.sum()),
                confirmed_free_fraction=fraction, **diag))
        deletion = self.erase(frame, erasure) if erasure else {}
        seconds = time.perf_counter()-started; self.seconds += seconds
        with (self.out/'frames.jsonl').open('a') as f:
            f.write(json.dumps(dict(frame=int(frame), seconds=seconds, global_conflict_ratio=context['global_ratio'],
                skip_reasons=context['skip_reasons'], references=rows, deletion=deletion))+'\n')
        self.save()

    def erase(self, frame, requests):
        import torch
        import open3d.core as core
        from .sparse_edges import Edges
        o = self.owner; grid = o.voxel_block_grids; graph = o.maskGraph
        xyz = np.unique(np.concatenate([cs for _, cs in requests]),axis=0)
        ids, ok = lattice_indices(grid, xyz, self.voxel, self.res, o.device_o3c)
        xyz, ids = xyz[ok], ids[ok]
        oid = core.Tensor(ids, dtype=core.int64, device=o.device_o3c)
        weights = grid.attribute('weight').reshape((-1,))[oid].cpu().numpy()
        use = weights > 0; xyz, ids = xyz[use], ids[use]
        if not len(ids): return dict(voxels=0)
        oid = core.Tensor(ids, dtype=core.int64, device=o.device_o3c)
        archived = dict(xyz=xyz, indices=ids)
        for attr, width in [('tsdf',1),('weight',1),('color',3)]:
            values=grid.attribute(attr).reshape((-1,width)); archived[attr]=values[oid].cpu().numpy()
            values[oid]=0.
        np.savez_compressed(self.archive/f'{frame}.npz', **archived)
        tid = torch.as_tensor(ids, device=o.device, dtype=torch.long)
        affected=[]; empty=[]; removed_support=0
        # Coordinates and indices are not guaranteed to have identical ordering
        # after original merge; recompute indices from coordinates before slicing.
        for mid, inst in list(graph.instance_dict.items()):
            coords=inst.mask_voxel_coords
            indices=o.voxel_hash_table.voxel_coords2voxel_indices(coords)
            keep=~torch.isin(indices,tid); count=int((~keep).sum())
            if not count: continue
            affected.append(mid); removed_support += count
            inst.mask_voxel_coords=coords[keep]; inst.mask_voxel_indices=indices[keep]
            o.mask_voxels_coords[mid]=inst.mask_voxel_coords
            graph.mask_voxels_coords[mid]=inst.mask_voxel_coords
            o.merged_mask_size[mid]=int(keep.sum())
            o.mask_geo_features[mid]=0; o.mask_geo_features_mask[mid]=0
            if not keep.any(): empty.append(mid)
        graph.voxel_in_frame_matrix[tid]=False
        if graph.boundary_voxel_indices is not None:
            graph.boundary_voxel_indices=graph.boundary_voxel_indices[~torch.isin(graph.boundary_voxel_indices,tid)]
        for i in ids.tolist(): o.voxel_hash_table.hash_table[i]=None
        o.voxel_hash_table.hash_item_flag[tid]=False
        if empty: o.remove_masks(frame,torch.tensor(empty,device=o.device,dtype=torch.long))
        # Rebuild sparse containment from surviving voxel evidence, no stale score.
        o.edges=Edges()
        for mid, inst in graph.instance_dict.items():
            overlaps, forward, backward=graph.query_mask_under_visible_part(
                graph.merged_mask_in_frame_mat[mid],inst.mask_voxel_coords,o.global2merged_mask_id)
            if overlaps.numel(): o.edges.write_row_col(mid,overlaps,forward,backward)
            o.edges.set(mid,mid,1.)
        for obj in (o,o.geometry):
            obj.pc_frame_id=-1; obj.pc_feature_frame_id=-1
            obj.points=None; obj.colors=None; obj.pc_xyz=None; obj.pc_feature=None
        o.pred_inst_mask_frame_id=-1; o.pred_inst_masks=None
        for mid,cs in requests:
            self.refs[mid]['deleted_voxels'] += int(len(cs))  # proposal count; total unique actual below
        self.deleted.update(map(tuple,np.rint(xyz/self.voxel).astype(np.int64)))
        result=dict(voxels=len(ids), unique_deleted_world_voxels=len(self.deleted), affected_current_ids=affected,
                    emptied_current_ids=empty, removed_instance_voxel_memberships=removed_support)
        self.events.append(dict(frame=int(frame),event='erasure',reference_ids=[x[0] for x in requests],**result))
        return result

    def save(self):
        refs={str(k):{f:r[f] for f in ['observations','track_uid','raw_observations','state','candidates','max_free','max_confirmed_free','deleted_voxels']} |
              dict(cells=len(r['points']),surface_points=len(r['full'])) for k,r in self.refs.items()}
        obj=dict(method='IG-Correction-v1', deletion_enabled=self.enabled, frontend=asdict(self.front),
            lifecycle=asdict(self.life), freeze_frame=getattr(self,'freeze_frame',None),
            freeze_seconds=getattr(self,'freeze_seconds',0.), audit_seconds=self.seconds,audited_frames=self.frame_count,
            unique_deleted_world_voxels=len(self.deleted), ground_truth_used=False,
            fixed_reference_session=True, reversible_tsdf_deactivation=False,references=refs,skips=self.skips,events=self.events)
        (self.out/'summary.json').write_text(json.dumps(obj,indent=2))
