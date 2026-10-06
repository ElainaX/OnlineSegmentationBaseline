#!/usr/bin/env python3
"""Diagnostic contradiction guard with post-fusion and temporal confirmation.

This module intentionally does not delete tracks or alter the production mapper.
A free-space residual is only a candidate. The current frame is integrated, the
same view is raycast again, and only residuals that survive integration and
repeat in consecutive audited views can be confirmed.
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, open3d as o3d, open3d.core as o3c
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / 'experiments/contradiction_audit'))
from audit_3rscan import AuditDataset, raycast
from voxelized_points import VoxelBlockGrid
import tool.config as config


def edge_mask(depth_m, threshold):
    """Reject depth discontinuities where ray ordering is least reliable."""
    valid = depth_m > 0
    edge = np.zeros_like(valid, dtype=bool)
    for axis in (0, 1):
        for shift in (-1, 1):
            shifted = np.roll(depth_m, shift, axis=axis)
            shifted_valid = np.roll(valid, shift, axis=axis)
            edge |= valid & shifted_valid & (np.abs(depth_m - shifted) > threshold)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    return edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--seq', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--config', default=str(REPO / 'config/scannet_stride10_controlled.yaml'))
    ap.add_argument('--audit-interval', type=int, default=10)
    ap.add_argument('--tau-min', type=float, default=.035)
    ap.add_argument('--tau-rel', type=float, default=.04)
    ap.add_argument('--ratio-threshold', type=float, default=.05)
    ap.add_argument('--min-pixels', type=int, default=400)
    ap.add_argument('--edge-threshold', type=float, default=.08)
    ap.add_argument('--min-post-persistence', type=float, default=.25)
    ap.add_argument('--min-consecutive', type=int, default=2)
    args = ap.parse_args()

    ds = AuditDataset(Path(args.data_root) / args.seq)
    cfg = config.load_config(args.config)
    wrap = VoxelBlockGrid(cfg, ds, 'cuda:0'); grid = wrap.voxel_block_grids
    K_o3c = o3c.Tensor.from_numpy(ds.K.astype(np.float64)); dev = o3c.Device('CUDA:0')
    depth_scale = float(cfg['cam']['depth_scale']); depth_far = float(cfg['cam']['depth_far'])
    trunc = float(cfg['scene'].get('trunc_voxel_multiplier', 4.))
    out = Path(args.output_root) / args.seq; out.mkdir(parents=True, exist_ok=True)
    frames = []; consecutive = 0; last_pose = np.eye(4, dtype=np.float32)

    for n, fid in enumerate(ds.ids):
        depth_raw = cv2.imread(str(ds.depths[fid]), cv2.IMREAD_UNCHANGED)
        color = cv2.imread(str(ds.colors[fid]), cv2.IMREAD_COLOR)
        pose = ds.poses[fid]
        if not np.isfinite(pose).all(): pose = last_pose
        else: last_pose = pose
        pose_w2c = np.linalg.inv(pose)
        pre = None
        if n % max(1, args.audit_interval) == 0:
            pre, _ = raycast(grid, K_o3c, pose_w2c, depth_raw, ds.target_w, ds.target_h,
                             depth_scale, depth_far, trunc)
            obs = depth_raw.astype(np.float32) / depth_scale
            pre_both = (obs > 0) & (pre > 0)
            pre_tau = np.maximum(args.tau_min, args.tau_rel * np.maximum(pre, .1))
            pre_free = pre_both & (obs > pre + pre_tau)
            edges = edge_mask(obs, args.edge_threshold)
            candidate = pre_free & ~edges
            support = pre_both & ~edges
        # Every frame is integrated, including non-audited frames.
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        depth_img = o3d.t.geometry.Image(np.ascontiguousarray(depth_raw)).to(o3c.uint16).to(dev)
        color_img = o3d.t.geometry.Image(np.ascontiguousarray(rgb)).to(o3c.uint8).to(dev)
        ext = o3c.Tensor.from_numpy(np.ascontiguousarray(pose_w2c.astype(np.float64)))
        fc = grid.compute_unique_block_coordinates(depth_img, K_o3c, ext, depth_scale, depth_far)
        grid.integrate(fc, depth_img, color_img, K_o3c, ext, depth_scale, depth_far,
                       trunc_voxel_multiplier=trunc)

        if pre is not None:
            # Same-view post-fusion check. This measures whether current integration
            # explains the residual immediately; it is not a second independent view.
            post, _ = raycast(grid, K_o3c, pose_w2c, depth_raw, ds.target_w, ds.target_h,
                              depth_scale, depth_far, trunc)
            post_both = (obs > 0) & (post > 0)
            post_tau = np.maximum(args.tau_min, args.tau_rel * np.maximum(post, .1))
            post_free = post_both & (obs > post + post_tau)
            persistent = candidate & post_free
            raw_ratio = float(pre_free.sum() / max(1, pre_both.sum()))
            supported_ratio = float(candidate.sum() / max(1, support.sum()))
            persistence = float(persistent.sum() / max(1, candidate.sum()))
            raw_trigger = bool(pre_free.sum() >= args.min_pixels and raw_ratio >= args.ratio_threshold)
            guarded_trigger = bool(candidate.sum() >= args.min_pixels and
                                   supported_ratio >= args.ratio_threshold and
                                   persistence >= args.min_post_persistence)
            consecutive = consecutive + 1 if guarded_trigger else 0
            confirmed = consecutive >= args.min_consecutive
            frames.append({
                'frame': int(fid), 'both_pixels': int(pre_both.sum()),
                'free_pixels': int(pre_free.sum()), 'support_pixels': int(support.sum()),
                'candidate_pixels': int(candidate.sum()), 'post_free_pixels': int(post_free.sum()),
                'persistent_pixels': int(persistent.sum()),
                'free_ratio': raw_ratio, 'supported_candidate_ratio': supported_ratio,
                'post_persistence_ratio': persistence,
                'post_decay_ratio': float(1.0 - persistence),
                'raw_trigger': raw_trigger, 'guarded_trigger': guarded_trigger,
                'consecutive_guarded': int(consecutive), 'confirmed': bool(confirmed),
                'map_blocks': int(grid.hashmap().size()),
            })
        if n % 250 == 0:
            print(f'{args.seq}: {n+1}/{len(ds.ids)} blocks={grid.hashmap().size()}', flush=True)

    ratios = np.asarray([x['free_ratio'] for x in frames], dtype=float)
    summary = {
        'sequence': args.seq, 'frames_total': len(ds.ids), 'audited_frames': len(frames),
        'parameters': vars(args),
        'meaning': {
            'raw_trigger': 'pre-integration free-space residual only',
            'guarded_trigger': 'candidate after edge/support filtering and same-view post-fusion persistence',
            'confirmed': 'guarded trigger in min-consecutive audited views; diagnostic only',
        },
        'metrics': {
            'raw_trigger_frames': int(sum(x['raw_trigger'] for x in frames)),
            'guarded_trigger_frames': int(sum(x['guarded_trigger'] for x in frames)),
            'confirmed_frames': int(sum(x['confirmed'] for x in frames)),
            'raw_trigger_rate': float(sum(x['raw_trigger'] for x in frames) / max(1, len(frames))),
            'guarded_trigger_rate': float(sum(x['guarded_trigger'] for x in frames) / max(1, len(frames))),
            'mean_free_ratio': float(ratios.mean()) if len(ratios) else 0.,
            'mean_post_persistence_ratio': float(np.mean([x['post_persistence_ratio'] for x in frames])) if frames else 0.,
            'p95_post_persistence_ratio': float(np.percentile([x['post_persistence_ratio'] for x in frames], 95)) if frames else 0.,
            'max_consecutive_guarded': int(max([x['consecutive_guarded'] for x in frames], default=0)),
        },
        'frames': frames,
    }
    json.dump(summary, open(out / 'guard_summary.json', 'w'), indent=2)
    print(json.dumps(summary['metrics'], indent=2), flush=True)

if __name__ == '__main__': main()