"""Runtime-only disjoint-pairs + Carving-AABB ablation runner.

This file does not modify the baseline or submap implementation on disk. It
wraps Scene_rep in memory, applies optional depth shrink and optional carved
AABB evidence, then applies the deterministic disjoint-pair selection.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--depth-shrink', type=int, choices=[0, 1], required=True)
    p.add_argument('--aabb', type=int, choices=[0, 1], required=True)
    p.add_argument('--strength', type=float, default=0.15)
    p.add_argument('--band', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--tag', default='disjoint_carving_ablation')
    p.add_argument('args', nargs=argparse.REMAINDER)
    o = p.parse_args()
    repo = Path(o.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(repo / 'third_party/FCGF'))
    aabb_dir = repo / 'experiments/baseline_aabb_soft_v2'
    sys.path.insert(0, str(aabb_dir))
    from aabb_soft import AABBMetrics
    from depth_mask_refinement import shrink_segmentation_by_depth
    random.seed(o.seed); np.random.seed(o.seed)
    torch.manual_seed(o.seed); torch.cuda.manual_seed_all(o.seed)

    import tool.config as config
    original_load = config.load_config
    effective_config = {}
    def load(*args, **kwargs):
        cfg = original_load(*args, **kwargs)
        cfg.setdefault('seg', {}).setdefault('depth_shrink', {})['enabled'] = bool(o.depth_shrink)
        cfg.setdefault('save', {})['ckpt_interval'] = 0
        cfg.setdefault('save', {})['save_interval'] = 0
        effective_config.update(cfg)
        return cfg
    config.load_config = load

    import Scene_rep as source
    original = source.Scene_rep
    scenes = []

    class DisjointCarvedScene(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if o.aabb:
                self.metrics = AABBMetrics(self.metrics, self, o.strength, o.band)
            self.aabb_boxes_override = None
            self.latest_depth = None
            self.latest_pose = None
            self.aabb_update_times = []
            self.carve_stats = []
            self.depth_shrink_stats = []
            self.ablation_stats = {
                'depth_shrink_enabled': bool(o.depth_shrink),
                'aabb_enabled': bool(o.aabb),
                'aabb_strength': float(o.strength),
                'aabb_band': float(o.band),
                'merge_calls': 0,
                'merge_edges_before_disjoint': 0,
                'selected_disjoint_pairs': 0,
                'disjoint_rejected_edges': 0,
            }
            scenes.append(self)

        @torch.no_grad()
        def _make_carved_boxes(self):
            n = self.c_mask_num
            device = self.device
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
            tolerance = max(0.06, 2.0 * float(self.cfg['scene']['voxel_size']))
            removed = 0; total = 0; fallback = 0
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
                inside = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                valid = torch.zeros_like(inside)
                valid[inside] = torch.isfinite(depth[v[inside], u[inside]]) & (depth[v[inside], u[inside]] > 0.1)
                d = torch.zeros_like(z)
                d[valid] = depth[v[valid], u[valid]]
                free = valid & (z < d - tolerance)
                kept = points[~free]
                if kept.shape[0] < max(16, int(0.10 * points.shape[0])):
                    kept = points
                    fallback += 1
                else:
                    removed += int(free.sum())
                boxes[i] = torch.stack((torch.quantile(kept, 0.01, dim=0),
                                        torch.quantile(kept, 0.99, dim=0)))
            self.carve_stats.append({'frame': int(getattr(self, '_carve_frame', -1)),
                                     'removed': removed, 'input_points': total,
                                     'fallback_boxes': fallback})
            return boxes

        def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w,
                             frustum_block_coords, seg_image, mask_features):
            if o.depth_shrink:
                seg_image, stats = shrink_segmentation_by_depth(seg_image, depth_img, self.cfg)
                stats['frame'] = int(frame_id)
                self.depth_shrink_stats.append(stats)
            result = super().insert_seg_frame(frame_id, color_img, depth_img, pose_c2w,
                                              frustum_block_coords, seg_image, mask_features)
            self._carve_frame = int(frame_id)
            self.latest_depth = depth_img.detach().clone()
            self.latest_pose = pose_c2w.detach().clone()
            return result

        def merge_masks(self, frame_id, merge_mat, c_mat, *args, **kwargs):
            self.ablation_stats['merge_calls'] += 1
            if not torch.is_tensor(merge_mat) or not int(torch.count_nonzero(merge_mat)):
                return super().merge_masks(frame_id, merge_mat, c_mat, *args, **kwargs)
            ij = torch.nonzero(merge_mat).tolist()
            self.ablation_stats['merge_edges_before_disjoint'] += len(ij)
            pairs = sorted(ij, key=lambda pair: (
                -float(c_mat[pair[0], pair[1]] + c_mat[pair[1], pair[0]]),
                pair[0], pair[1]))
            keep = torch.zeros_like(merge_mat)
            used = set()
            for i, j in pairs:
                if i != j and i not in used and j not in used:
                    keep[i, j] = True
                    used.update((i, j))
            self.ablation_stats['selected_disjoint_pairs'] += int(torch.count_nonzero(keep))
            self.ablation_stats['disjoint_rejected_edges'] += len(ij) - int(torch.count_nonzero(keep))
            return super().merge_masks(frame_id, keep, c_mat, *args, **kwargs)

        def update_masks(self, frame_id):
            self._carve_frame = int(frame_id)
            if o.aabb:
                self.aabb_boxes_override = self._make_carved_boxes()
                torch.cuda.synchronize()
            start = time.perf_counter()
            if o.aabb and hasattr(self.metrics, 'frame'):
                self.metrics.frame = int(frame_id)
            result = super().update_masks(frame_id)
            if o.aabb:
                torch.cuda.synchronize()
                self.aabb_update_times.append({'frame': int(frame_id),
                                               'seconds': time.perf_counter() - start})
            return result

    source.Scene_rep = DisjointCarvedScene
    raw = o.args[1:] if o.args[:1] == ['--'] else o.args
    sys.argv = [str(repo / 'main.py')] + raw
    main_src = (repo / 'main.py').read_text()
    if o.limit:
        main_src = main_src.replace('max_frame_num = None',
                                    'max_frame_num = %d' % o.limit, 1)
    start = time.perf_counter(); status = 'failed'
    try:
        exec(compile(main_src, str(repo / 'main.py'), 'exec'),
             {'__name__': '__main__', '__file__': str(repo / 'main.py')})
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        status = 'complete'
    finally:
        for scene in scenes:
            evidence = getattr(scene.metrics, 'records', []) if o.aabb else []
            manifest = {
                'status': status,
                'seconds': time.perf_counter() - start,
                'depth_shrink_enabled': bool(o.depth_shrink),
                'aabb_enabled': bool(o.aabb),
                'strength': float(o.strength),
                'band': float(o.band),
                'seed': int(o.seed),
                'argv': raw,
                'config': effective_config,
                'carving': scene.carve_stats,
                'depth_shrink': scene.depth_shrink_stats,
                'aabb_evidence': evidence,
                'aabb_candidate_pairs': sum(int(x.get('eligible_pairs', 0)) for x in evidence),
                'aabb_promoted_pairs': sum(int(x.get('promoted_pairs', 0)) for x in evidence),
                'aabb_demoted_pairs': sum(int(x.get('demoted_pairs', 0)) for x in evidence),
                'ablation_stats': scene.ablation_stats,
                'source_hash': hashlib.sha256((repo / 'Scene_rep.py').read_bytes()).hexdigest(),
                'runner_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            out = Path(scene.seq_output_dir) / 'disjoint_carving_manifest.json'
            out.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            print('DISJOINT_CARVING', status, 'seconds', manifest['seconds'],
                  'manifest', out, flush=True)


if __name__ == '__main__':
    main()
