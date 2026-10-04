"""Submap pipeline with current-view Carving-AABB merge evidence.

The submap manager remains responsible for local object geometry and export.
Carving-AABB is an additional, bounded score term for the candidate pairs
already produced by MaskGraph.  It does not create candidates or delete the
baseline/submap voxels.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from aabb_soft import AABBMetrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--strength', type=float, default=.15)
    p.add_argument('--band', type=float, default=.10)
    p.add_argument('--positive-strength', type=float, default=None)
    p.add_argument('--negative-strength', type=float, default=None)
    p.add_argument('--confidence-threshold', type=float, default=.35)
    p.add_argument('--negative-confidence-floor', type=float, default=.25)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('args', nargs=argparse.REMAINDER)
    o = p.parse_args()
    if (not 0 <= o.strength <= .25 or not 0 <= o.band <= .5 or
            (o.positive_strength is not None and not 0 <= o.positive_strength <= .25) or
            (o.negative_strength is not None and not 0 <= o.negative_strength <= .25) or
            not 0 <= o.confidence_threshold <= 1):
        p.error('strengths must be within [0,.25], band within [0,.5], confidence within [0,1]')

    repo = Path(o.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(repo / 'third_party/FCGF'))
    random.seed(o.seed)
    np.random.seed(o.seed)
    torch.manual_seed(o.seed)
    torch.cuda.manual_seed_all(o.seed)

    import Scene_rep as source
    original = source.Scene_rep
    scenes = []

    class CarvedSubmapScene(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.metrics = AABBMetrics(self.metrics, self, o.strength, o.band,
                                       o.positive_strength, o.negative_strength,
                                       o.confidence_threshold)
            self.metrics.negative_confidence_floor = o.negative_confidence_floor
            self.aabb_boxes_override = None
            self.latest_depth = None
            self.latest_pose = None
            self.aabb_update_times = []
            self.carve_stats = []
            self.visibility_stats = []
            self.aabb_box_confidence = None
            scenes.append(self)

        @torch.no_grad()
        def _make_carved_boxes(self):
            n = self.c_mask_num
            device = self.device
            boxes = torch.zeros((n, 2, 3), device=device, dtype=torch.float32)
            confidence = torch.zeros((n,), device=device, dtype=torch.float32)
            if n == 0 or self.latest_depth is None or self.latest_pose is None:
                return boxes
            depth = self.latest_depth.to(device).float()
            pose = self.latest_pose.to(device).float()
            if depth.ndim == 3:
                depth = depth[0]
            K = self.intrinsics.to(device).float()
            h, w = depth.shape[-2:]
            inv_pose = torch.linalg.inv(pose)
            tolerance = max(.06, 2. * float(self.cfg['scene']['voxel_size']))
            removed = total = fallback = 0
            vis_points = occ_points = free_points = unknown_points = 0
            confidence_values = []
            for i in range(n):
                points = self.mask_voxels_coords[i].to(device).float()
                if points.numel() == 0:
                    continue
                total += int(points.shape[0])
                ph = torch.cat((points, torch.ones((points.shape[0], 1), device=device)), dim=1)
                cam = (ph @ inv_pose.T)[:, :3]
                z = cam[:, 2]
                u = torch.round(cam[:, 0] / z.clamp_min(1e-6) * K[0, 0] + K[0, 2]).long()
                v = torch.round(cam[:, 1] / z.clamp_min(1e-6) * K[1, 1] + K[1, 2]).long()
                inside = (z > .1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                valid = torch.zeros_like(inside)
                valid[inside] = torch.isfinite(depth[v[inside], u[inside]]) & (depth[v[inside], u[inside]] > .1)
                d = torch.zeros_like(z)
                d[valid] = depth[v[valid], u[valid]]
                visible = valid & ((z - d).abs() <= tolerance)
                occluded = valid & (z > d + tolerance)
                free = valid & (z < d - tolerance)
                point_count = max(int(points.shape[0]), 1)
                visible_ratio = float(visible.sum().item()) / point_count
                occluded_ratio = float(occluded.sum().item()) / point_count
                free_ratio = float(free.sum().item()) / point_count
                unknown_ratio = 1.0 - float(valid.sum().item()) / point_count
                # A view is trusted when it directly observes the instance;
                # occluded, free-space, and unknown projections reduce trust.
                view_confidence = visible_ratio * (1.0 - occluded_ratio) * (1.0 - unknown_ratio)
                confidence[i] = max(0.0, min(1.0, view_confidence))
                confidence_values.append(float(confidence[i].item()))
                vis_points += int(visible.sum().item())
                occ_points += int(occluded.sum().item())
                free_points += int(free.sum().item())
                unknown_points += int((~valid).sum().item())
                kept = points[~free]
                if kept.shape[0] < max(16, int(.10 * points.shape[0])):
                    kept = points
                    fallback += 1
                else:
                    removed += int(free.sum())
                boxes[i] = torch.stack((torch.quantile(kept, .01, dim=0),
                                        torch.quantile(kept, .99, dim=0)))
            self.aabb_box_confidence = confidence
            self.carve_stats.append(dict(frame=int(self._carve_frame), removed=removed,
                                         input_points=total, fallback_boxes=fallback))
            denom = max(total, 1)
            self.visibility_stats.append(dict(
                frame=int(self._carve_frame),
                visible_ratio=float(vis_points / denom),
                occluded_ratio=float(occ_points / denom),
                free_ratio=float(free_points / denom),
                unknown_ratio=float(unknown_points / denom),
                mean_confidence=float(np.mean(confidence_values)) if confidence_values else 0.0))
            return boxes

        def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w,
                             frustum_block_coords, seg_image, mask_features):
            result = super().insert_seg_frame(frame_id, color_img, depth_img, pose_c2w,
                                              frustum_block_coords, seg_image, mask_features)
            self._carve_frame = int(frame_id)
            self.latest_depth = depth_img.detach().clone()
            self.latest_pose = pose_c2w.detach().clone()
            return result

        def update_masks(self, frame_id):
            self._carve_frame = int(frame_id)
            self.aabb_boxes_override = self._make_carved_boxes()
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            self.metrics.frame = int(frame_id)
            result = super().update_masks(frame_id)
            if self.device.startswith('cuda'):
                torch.cuda.synchronize()
            self.aabb_update_times.append(dict(frame=int(frame_id),
                                               seconds=time.perf_counter() - start))
            return result

    source.Scene_rep = CarvedSubmapScene
    raw = o.args[1:] if o.args[:1] == ['--'] else o.args
    sys.argv = [str(repo / 'main.py')] + raw
    text = (repo / 'main.py').read_text()
    if o.limit:
        text = text.replace('max_frame_num = None', 'max_frame_num = %d' % o.limit, 1)
    start = time.perf_counter()
    status = 'failed'
    try:
        exec(compile(text, str(repo / 'main.py'), 'exec'),
             {'__name__': '__main__', '__file__': str(repo / 'main.py')})
        status = 'complete'
    finally:
        for scene in scenes:
            files = ['main.py', 'Scene_rep.py', 'submap_mapping.py', 'Metrics.py',
                     'MaskGraph.py', 'Instance.py', 'PC_extractor.py',
                     'voxelized_points.py', 'Dataset/dataset.py', 'tool/post_process.py']
            manifest = dict(
                status=status, seconds=time.perf_counter() - start,
                strength=o.strength, band=o.band,
                positive_strength=o.positive_strength,
                negative_strength=o.negative_strength,
                confidence_threshold=o.confidence_threshold,
                negative_confidence_floor=o.negative_confidence_floor,
                seed=o.seed, limit=o.limit,
                repo=str(repo), argv=raw,
                original_hashes={f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files},
                experiment_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in Path(__file__).parent.glob('*.py')},
                updates=scene.aabb_update_times, evidence=scene.metrics.records,
                carving=scene.carve_stats, visibility=scene.visibility_stats,
                note='Submap manager retained; current-view depth carving supplies only AABB score evidence; no baseline or submap voxels are deleted.')
            dest = Path(scene.seq_output_dir) / 'carved_aabb_manifest.json'
            dest.write_text(json.dumps(manifest, indent=2))
            print('SUBMAP_CARVED_AABB_RUN', status, 'seconds', manifest['seconds'],
                  'manifest', dest, flush=True)


if __name__ == '__main__':
    main()
