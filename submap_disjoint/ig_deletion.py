"""Local erasure AFTER the unchanged onlineSeg detector confirms a candidate.

No GT, session labels, future observations or current merged row IDs authorize
erasure. Frozen raw lineage bounds the eligible old memberships. Geometric
freedom must be checked again against the current depth before every mutation.
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import open3d.core as core
from .ig_correction import corners, lattice_indices
from .readonly_detection_core import as_numpy, interior_mask, classify_depth, CONTRADICTION
from .sparse_edges import Edges


def currently_free(points, depth, intrinsic, pose, cfg, near, far):
    result = np.zeros(len(points), bool)
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    z = cam[:, 2]
    ids = np.flatnonzero(np.isfinite(cam).all(1) & (z > near) & (z < far))
    uv = np.rint((cam[ids] @ intrinsic.T)[:, :2] / z[ids, None]).astype(np.int64)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < depth.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < depth.shape[0])
    ids, uv = ids[ok], uv[ok]
    valid = np.isfinite(depth) & (depth > near) & (depth < far)
    inside = interior_mask(depth, valid, cfg['edge_threshold_m'], cfg['boundary_margin_px'])
    u, v = uv[:, 0], uv[:, 1]
    result[ids] = classify_depth(depth[v, u], z[ids], inside[v, u],
        cfg['depth_epsilon_m'], cfg['depth_relative_epsilon']) == CONTRADICTION
    return result


def eligible_lineage(members, source, mapping):
    active = {int(r) for r in (members or ()) if 0 <= r < len(mapping) and mapping[r] >= 0}
    return bool(active) and active <= source


class IGDeletion:
    def __init__(self, owner, detector, directory):
        self.owner, self.detector = owner, detector
        self.out = Path(directory); self.out.mkdir(parents=True, exist_ok=True)
        self.archive = self.out/'archive'; self.archive.mkdir(exist_ok=True)
        self.voxel = owner.cfg['scene']['voxel_size']
        self.res = owner.cfg['scene']['block_resolution']
        self.events, self.deleted = [], set()
        self.seconds = 0.; self.frames = 0
        self.save()

    def step(self, frame, depth, pose, valid_pose=True):
        start = time.perf_counter(); self.frames += 1
        d = self.detector.detector
        # Do not bypass invalid pose/depth, widespread conflict or render gates.
        if not valid_pose or d.frames_processed == getattr(self, 'last_processed', 0):
            self.last_processed = d.frames_processed
            return
        self.last_processed = d.frames_processed
        depth, pose = as_numpy(depth), as_numpy(pose).astype(np.float64)
        requests, diagnostics = [], []
        for c in d.candidates:
            if c.state != 'would_remove': continue
            pts = c.points[c.depth_contradicted]
            if not len(pts): continue
            xyz = np.unique(np.rint(corners(pts, self.voxel).reshape(-1, 3)/self.voxel).astype(np.int64), axis=0)*self.voxel
            count = len(xyz)
            xyz = xyz[currently_free(xyz, depth, d.intrinsic, pose, d.cfg, d.near, d.far)]
            ids, exists = lattice_indices(self.owner.voxel_block_grids, xyz, self.voxel, self.res, self.owner.device_o3c)
            ids, xyz = ids[exists], xyz[exists]
            allowed = np.array([eligible_lineage(self.owner.voxel_hash_table.hash_table[int(i)],
                c.source_raw_ids, self.owner.global2merged_mask_id) for i in ids], dtype=bool)
            diagnostics.append(dict(candidate=c.candidate_id, depth_reference_points=len(pts),
                proposed_corners=count, current_free_existing_corners=len(ids), lineage_allowed=int(allowed.sum())))
            if allowed.any(): requests.append((c.candidate_id, xyz[allowed]))
        result = self.erase(frame, requests) if requests else dict(voxels=0)
        elapsed = time.perf_counter()-start; self.seconds += elapsed
        self.events.append(dict(frame=int(frame), seconds=elapsed, candidates=diagnostics, **result))
        self.save()

    def erase(self, frame, requests):
        o, grid, graph = self.owner, self.owner.voxel_block_grids, self.owner.maskGraph
        xyz = np.unique(np.concatenate([x for _, x in requests]), axis=0)
        ids, ok = lattice_indices(grid, xyz, self.voxel, self.res, o.device_o3c)
        xyz, ids = xyz[ok], ids[ok]
        oid = core.Tensor(ids, dtype=core.int64, device=o.device_o3c)
        live = grid.attribute('weight').reshape((-1,))[oid].cpu().numpy() > 0
        xyz, ids = xyz[live], ids[live]
        if not len(ids): return dict(voxels=0)
        oid = core.Tensor(ids, dtype=core.int64, device=o.device_o3c)
        archived = dict(xyz=xyz, indices=ids)
        for attr, width in [('tsdf',1), ('weight',1), ('color',3)]:
            archived[attr] = grid.attribute(attr).reshape((-1,width))[oid].cpu().numpy()
        # Persist numeric state BEFORE changing shared TSDF. This archive is
        # evidence, not an automatic full graph rollback checkpoint.
        np.savez_compressed(self.archive/f'{frame:06d}.npz', **archived)
        raw = sorted({r for i in ids.tolist() for r in (o.voxel_hash_table.hash_table[i] or ())})
        (self.archive/f'{frame:06d}.json').write_text(json.dumps(dict(
            candidates=[c for c, _ in requests], raw_ids=raw,
            memberships=[sorted(o.voxel_hash_table.hash_table[i] or ()) for i in ids.tolist()])))
        for attr, width in [('tsdf',1), ('weight',1), ('color',3)]:
            grid.attribute(attr).reshape((-1,width))[oid] = 0.
        tid = torch.as_tensor(ids, device=o.device, dtype=torch.long)
        affected, empty, removed = [], [], 0
        for mid, inst in list(graph.instance_dict.items()):
            coords = inst.mask_voxel_coords
            indices = o.voxel_hash_table.voxel_coords2voxel_indices(coords)
            keep = ~torch.isin(indices, tid)
            count = int((~keep).sum())
            if not count: continue
            affected.append(mid); removed += count
            inst.mask_voxel_coords, inst.mask_voxel_indices = coords[keep], indices[keep]
            o.mask_voxels_coords[mid] = inst.mask_voxel_coords
            graph.mask_voxels_coords[mid] = inst.mask_voxel_coords
            o.merged_mask_size[mid] = int(keep.sum())
            o.mask_geo_features[mid] = 0; o.mask_geo_features_mask[mid] = 0
            if not keep.any(): empty.append(mid)
        # Original per-frame geometry is also pruned, so caches cannot restore
        # removed memberships. Historical 2D semantic embeddings stay unchanged.
        for r in raw:
            key = o.global_f_mask_ids[r]
            if key in graph.f_mask_voxels_clouds:
                coords = graph.f_mask_voxels_clouds[key]
                idx = o.voxel_hash_table.voxel_coords2voxel_indices(coords)
                graph.f_mask_voxels_clouds[key] = coords[~torch.isin(idx, tid)]
        graph.voxel_in_frame_matrix[tid] = False
        if graph.boundary_voxel_indices is not None:
            graph.boundary_voxel_indices = graph.boundary_voxel_indices[~torch.isin(graph.boundary_voxel_indices, tid)]
        for i in ids.tolist(): o.voxel_hash_table.hash_table[i] = None
        o.voxel_hash_table.hash_item_flag[tid] = False
        if empty: o.remove_masks(frame, torch.tensor(empty, device=o.device, dtype=torch.long))
        # Every pair touching altered support is recomputed; rebuilding all
        # sparse edges avoids stale supporter chains after row compaction.
        o.edges = Edges()
        for mid, inst in graph.instance_dict.items():
            other, forward, backward = graph.query_mask_under_visible_part(
                graph.merged_mask_in_frame_mat[mid], inst.mask_voxel_coords, o.global2merged_mask_id)
            if other.numel(): o.edges.write_row_col(mid, other, forward, backward)
            o.edges.set(mid, mid, 1.)
        for obj in (o, o.geometry):
            obj.pc_frame_id = -1; obj.pc_feature_frame_id = -1
            obj.points = obj.colors = obj.pc_xyz = obj.pc_feature = None
        o.pred_inst_mask_frame_id = -1; o.pred_inst_masks = None
        self.deleted.update(map(tuple, np.rint(xyz/self.voxel).astype(np.int64)))
        return dict(voxels=len(ids), affected_current_ids=affected, emptied_current_ids=empty,
            removed_instance_voxel_memberships=removed)

    def save(self):
        summary = dict(method='IG confirmed local depth erasure v1', enabled=True,
            ground_truth_used=False, deletion_frames=sum(e['voxels']>0 for e in self.events),
            voxel_erasure_operations=sum(e['voxels'] for e in self.events),
            unique_world_voxels=len(self.deleted), seconds=self.seconds,
            whole_instance_deletion=False, events=self.events)
        temp = self.out/'summary.tmp'; temp.write_text(json.dumps(summary, indent=2)); temp.replace(self.out/'summary.json')
