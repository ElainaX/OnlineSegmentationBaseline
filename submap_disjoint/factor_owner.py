"""Opt-in legacy depth refinement and Carving-AABB on the IG instance owner."""
import json
import time
from pathlib import Path
import torch
from depth_mask_refinement import shrink_segmentation_by_depth
from submap_disjoint.sparse_owner import SparseDisjointSubmapManager
from submap_disjoint.carved_pair_scores import adjust_carved_pair_scores


class FactorSubmapManager(SparseDisjointSubmapManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.carving_options = self.cfg.get('ig_submap', {}).get('carving_aabb', {})
        self.carving_enabled = bool(self.carving_options.get('enabled', False))
        self.depth_shrink_enabled = bool(self.cfg.get('seg', {}).get('depth_shrink', {}).get('enabled', False))
        self.latest_depth = self.latest_pose = self.aabb_boxes_override = None
        self.carve_stats = []
        self.depth_shrink_stats = []
        self.aabb_evidence = []
        self.factor_seconds = dict(depth_shrink=0., carved_boxes=0., pair_score=0.)
        self._carve_frame = -1

    def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w,
                         frustum_block_coords, seg_image, mask_features, **kwargs):
        if self.depth_shrink_enabled:
            start = time.perf_counter()
            seg_image, stats = shrink_segmentation_by_depth(seg_image, depth_img, self.cfg)
            self.factor_seconds['depth_shrink'] += time.perf_counter()-start
            self.depth_shrink_stats.append(dict(stats, frame=int(frame_id)))
        result = super().insert_seg_frame(frame_id, color_img, depth_img, pose_c2w,
                                         frustum_block_coords, seg_image, mask_features, **kwargs)
        if self.carving_enabled:
            self.latest_depth = depth_img.detach().clone()
            self.latest_pose = pose_c2w.detach().clone()
        return result

    @torch.no_grad()
    def _make_carved_boxes(self):
        # Same construction as experiments/run_disjoint_carving_aabb.py.
        # The point subset is temporary score evidence, never written to the map.
        n, device = self.c_mask_num, self.device
        boxes = torch.zeros((n, 2, 3), device=device, dtype=torch.float32)
        if n == 0 or self.latest_depth is None or self.latest_pose is None:
            return boxes
        depth = self.latest_depth.to(device).float()
        pose = self.latest_pose.to(device).float()
        while depth.ndim > 2:
            depth = depth[0]
        K = self.intrinsics.to(device).float()
        h, w = depth.shape[-2:]
        inv_pose = torch.linalg.inv(pose)
        tolerance = max(.06, 2.*float(self.cfg['scene']['voxel_size']))
        removed = total = fallback = 0
        for i in range(n):
            points = self.mask_voxels_coords[i].to(device).float()
            if points.numel() == 0:
                continue
            total += int(points.shape[0])
            ph = torch.cat((points, torch.ones((points.shape[0], 1), device=device)), dim=1)
            cam = (ph @ inv_pose.T)[:, :3]
            z = cam[:, 2]
            u = torch.round(cam[:, 0]/z.clamp_min(1e-6)*K[0, 0]+K[0, 2]).long()
            v = torch.round(cam[:, 1]/z.clamp_min(1e-6)*K[1, 1]+K[1, 2]).long()
            inside = (z > .1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            valid = torch.zeros_like(inside)
            valid[inside] = torch.isfinite(depth[v[inside], u[inside]]) & (depth[v[inside], u[inside]] > .1)
            d = torch.zeros_like(z)
            d[valid] = depth[v[valid], u[valid]]
            free = valid & (z < d-tolerance)
            kept = points[~free]
            if kept.shape[0] < max(16, int(.10*points.shape[0])):
                kept = points
                fallback += 1
            else:
                removed += int(free.sum())
            boxes[i] = torch.stack((torch.quantile(kept, .01, dim=0),
                                    torch.quantile(kept, .99, dim=0)))
        self.carve_stats.append(dict(frame=self._carve_frame, removed=removed,
            input_points=total, fallback_boxes=fallback))
        return boxes

    def update_masks(self, frame_id):
        self._carve_frame = int(frame_id)
        if self.carving_enabled:
            start = time.perf_counter()
            self.aabb_boxes_override = self._make_carved_boxes()
            self.factor_seconds['carved_boxes'] += time.perf_counter()-start
        try:
            return super().update_masks(frame_id)
        finally:
            # IDs change in the merge pass. Never reuse these boxes afterwards.
            self.aabb_boxes_override = None

    def adjust_candidate_scores(self, values, pairs, forward, backward):
        if not self.carving_enabled:
            return values
        start = time.perf_counter()
        boxes = self.aabb_boxes_override
        if boxes is None or len(boxes) != self.c_mask_num:
            raise RuntimeError('Carving-AABB boxes are missing or IDs are stale')
        result, stats = adjust_carved_pair_scores(values, pairs, forward, backward,
            boxes, torch.isfinite(boxes).all(dim=2).all(dim=1),
            self.sim_merge_thresh, self.merge_overlap_thresh,
            strength=self.carving_options['strength'], band=self.carving_options['band'])
        self.factor_seconds['pair_score'] += time.perf_counter()-start
        self.aabb_evidence.append(dict(stats, frame=self._carve_frame,
            scored_pairs=len(pairs), instances=self.c_mask_num))
        return result

    def save_outputs(self, frame_id, directory):
        super().save_outputs(frame_id, directory)
        report = dict(carving_aabb=self.carving_options,
            depth_shrink=self.cfg['seg']['depth_shrink'], carving=self.carve_stats,
            depth_shrink_frames=self.depth_shrink_stats, aabb_evidence=self.aabb_evidence,
            factor_host_seconds=self.factor_seconds,
            note='Host wall time includes synchronization at stats reads, not isolated CUDA kernel time.',
            removed_pixels=sum(x['removed_pixels'] for x in self.depth_shrink_stats),
            total_mask_pixels=sum(x.get('total_pixels', 0) for x in self.depth_shrink_stats),
            aabb_eligible_pairs=sum(x['eligible_pairs'] for x in self.aabb_evidence),
            aabb_promoted_pairs=sum(x['promoted_pairs'] for x in self.aabb_evidence),
            aabb_demoted_pairs=sum(x['demoted_pairs'] for x in self.aabb_evidence))
        (Path(directory)/'factor_manifest.json').write_text(json.dumps(report, indent=2))
