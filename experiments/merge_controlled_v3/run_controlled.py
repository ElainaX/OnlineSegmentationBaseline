"""Scale-matched merge ablations. Never modify baseline source files on disk."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types
import numpy as np
import torch
import torch.nn.functional as F

MODES = ['baseline', 'no_semantic', 'no_geometry', 'no_overlap_score',
         'no_supporter', 'no_containment', 'full_overlap', 'no_boundary',
         'disjoint_pairs', 'equal_original', 'quality_prototype',
         'disjoint_quality', 'disjoint_no_sem_geo']

def score(mode, overlap, sem, geo, use_overlap):
    terms = [sem, geo]
    if mode in ('no_semantic', 'disjoint_no_sem_geo'): terms = [geo]
    if mode in ('no_geometry', 'disjoint_no_sem_geo'): terms = [sem] if mode != 'disjoint_no_sem_geo' else []
    if use_overlap and mode != 'no_overlap_score': terms = [overlap] + terms
    if not terms:
        return torch.zeros_like(overlap)
    # Preserve the original maximum (3 or 2); no GT-tuned threshold.
    return sum(terms) * ((3.0 if use_overlap else 2.0) / len(terms))

def prototype(features, weights, gated):
    unit = F.normalize(features.float(), dim=-1)
    weights = weights.clamp_min(1e-6)
    anchor = int(weights.argmax())
    compatible = bool(((unit @ unit[anchor]) >= .8).all())
    if gated and not compatible:
        return unit[anchor], False
    return (unit * weights[:, None]).sum(0), True

def camera_quality(poses, i):
    """Dimensionless proxy from adjacent RAW frames, not seconds or visibility."""
    if not torch.isfinite(poses[max(0, i-2):i+1]).all(): return .01
    if i == 0: return 1.0
    def change(a, b):
        d = torch.linalg.inv(a) @ b
        t = float(d[:3, 3].norm())
        r = float(torch.acos(((torch.trace(d[:3, :3])-1)/2).clamp(-1, 1)))
        return t, r
    t, r = change(poses[i-1], poses[i])
    pt, pr = change(poses[i-2], poses[i-1]) if i > 1 else (t, r)
    # metres/raw-frame, radians/raw-frame; independent scales before addition.
    cost = t/.02 + r/.035 + abs(t-pt)/.01 + abs(r-pr)/.0175
    return max(.01, float(np.exp(-cost)))

def full_query(self, mask_frame, coords, mapping, valid_merge_mask_ids=None):
    if -1 in mapping: valid_merge_mask_ids = sorted(set(x for x in mapping if x != -1))
    ids, counts, _ = self.voxel_hash_table.query_mask_w_mapping(coords, mapping, valid_merge_mask_ids)
    if ids.numel() == 0: return ids, None, None
    sizes = counts.new_tensor([len(self.instance_dict[int(i)].mask_voxel_indices) for i in ids], dtype=torch.float32)
    forward = counts.float() / sizes.clamp_min(1)
    backward = counts.float() / max(1, len(coords))
    forward = torch.where(forward > self.contain_min_threshold, forward, 0).clamp(max=1)
    backward = torch.where(backward > self.contain_min_threshold, backward, 0).clamp(max=1)
    return ids, forward, backward

def full_matrix(self, ids, mapping):
    ids = [int(i) for i in ids]
    n = len(ids)
    out = torch.eye(n, device=self.device)
    lookup = {v:k for k,v in enumerate(ids)}
    for row, mid in enumerate(ids):
        overlap, forward, _ = full_query(self, None, self.instance_dict[mid].mask_voxel_coords, mapping, ids)
        if overlap.numel():
            # The voxel query may return tracks outside the reduced candidate
            # set used for this merge call.  They are valid for the raw query,
            # but cannot be written into this local matrix.
            keep = torch.tensor([int(i) in lookup for i in overlap],
                                device=overlap.device, dtype=torch.bool)
            if keep.any():
                overlap_kept = overlap[keep]
                forward_kept = forward[keep]
                col = torch.tensor([lookup[int(i)] for i in overlap_kept],
                                   device=self.device)
                out[row, col] = forward_kept
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--max_frames', type=int, default=None,
                   help='Process only the first N frames for fast ablations.')
    p.add_argument('args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    repo = Path(args.repo).resolve()
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo/'third_party/FCGF')]
    random.seed(27); np.random.seed(27); torch.manual_seed(27); torch.cuda.manual_seed_all(27)
    import tool.config as config
    orig_load = config.load_config
    effective_config = {}
    def load(*a, **kw):
        cfg = orig_load(*a, **kw)
        cfg['seg']['depth_shrink']['enabled'] = False
        # Suppress intermediate exports uniformly; final export is unchanged.
        cfg['save']['ckpt_interval'] = 0
        cfg['save']['save_interval'] = 0
        effective_config.update(cfg)
        return cfg
    config.load_config = load

    # Only a semantic-prototype line and passive counters change in the class.
    src = (repo/'Scene_rep.py').read_text()
    needle = 'cluster_feature_merged = cluster_features.sum(dim=0)'
    assert src.count(needle) == 1
    src = src.replace(needle, 'cluster_feature_merged = self.fuse_prototype(merged_mask_id_new, cluster_features)')
    needle = 'merge_mat = torch.logical_or(sim_merge_mat, supporter_num_merge_mat)'
    assert src.count(needle) == 1
    src = src.replace(needle, 'self.note_edges(sim_merge_mat, supporter_num_merge_mat)\n        '+needle)
    module = types.ModuleType('Scene_rep'); module.__file__ = str(repo/'Scene_rep.py')
    exec(compile(src, module.__file__, 'exec'), module.__dict__)
    sys.modules['Scene_rep'] = module
    from Metrics import Metrics
    from tool.helper_functions import retain_max_per_row
    class AblationMetrics(Metrics):
        def compute_final_sim_mat(self, c_mat, sem_feature_mat, geo_feature_mat, **kw):
            retain = kw.pop('retain_max', False)
            final, overlap, sem, geo = super().compute_final_sim_mat(c_mat, sem_feature_mat, geo_feature_mat, retain_max=False, **kw)
            if args.mode in ('no_semantic', 'no_geometry', 'no_overlap_score', 'disjoint_no_sem_geo'):
                final = score(args.mode, overlap, sem, geo, kw.get('use_IoU', True))
            if retain: final = retain_max_per_row(final)
            return final, overlap, sem, geo
    module.Metrics = AblationMetrics
    scenes = []
    class Scene(module.Scene_rep):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.audit = dict(sim_edges=0, supporter_edges=0, both_edges=0,
                              overall_edges=0, containment_edges=0, union_calls=0,
                              instances_before=0, instances_after=0,
                              prototype_fused=0, prototype_selected=0)
            self.observations = {}
            if args.mode == 'no_supporter': self.merge_supporter_num = float('inf')
            if args.mode == 'no_containment': self.merge_contain_iter_num = 0
            if args.mode == 'no_boundary': self.maskGraph.consider_bound_pts = False
            if args.mode == 'full_overlap':
                self.maskGraph.query_mask_under_visible_part = types.MethodType(full_query, self.maskGraph)
                self.maskGraph.compute_masks_contain_ratio = types.MethodType(full_matrix, self.maskGraph)
            self.raw_poses = self.dataset.poses.detach().float().cpu() if args.mode == 'quality_prototype' else None
            scenes.append(self)

        def note_edges(self, sim, support):
            self.audit['sim_edges'] += int(sim.sum())
            self.audit['supporter_edges'] += int(support.sum())
            self.audit['both_edges'] += int((sim & support).sum())

        def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w, blocks, seg_image, features):
            if args.mode in ('equal_original', 'quality_prototype', 'disjoint_quality'):
                quality = camera_quality(self.raw_poses, frame_id) if self.raw_poses is not None else 1.
                ids, counts = torch.unique(seg_image, return_counts=True)
                for mid, pixels in zip(ids.tolist(), counts.tolist()):
                    if mid > 0:
                        area = min(1., np.sqrt((pixels/seg_image.numel())/.05))
                        w = max(.0001, quality*area) if args.mode == 'quality_prototype' else 1.
                        self.observations[(int(frame_id), int(mid))] = (features[mid-1].detach().clone(), w)
            return super().insert_seg_frame(frame_id, color_img, depth_img, pose_c2w, blocks, seg_image, features)

        def fuse_prototype(self, new_id, cluster_features):
            if args.mode not in ('equal_original', 'quality_prototype', 'disjoint_quality'):
                return cluster_features.sum(0)
            inst = self.maskGraph.instance_dict[new_id]
            keys = sorted(set(tuple(x) for x in inst.ori_mask_list))
            fs, ws = zip(*(self.observations[k] for k in keys))
            features = torch.stack(fs)
            weights = features.new_tensor(ws)
            out, fused = prototype(features, weights,
                                    args.mode in ('quality_prototype', 'disjoint_quality'))
            self.audit['prototype_fused' if fused else 'prototype_selected'] += 1
            inst.mask_sem_feature = out
            return out

        def merge_masks(self, frame_id, merge_mat, c_mat, count_merge=False, full_geo_merge=True):
            edges = int(merge_mat.count_nonzero())
            self.audit['overall_edges' if count_merge else 'containment_edges'] += edges
            if args.mode in ('disjoint_pairs', 'disjoint_quality', 'disjoint_no_sem_geo') and edges:
                # Replace transitive components by deterministic disjoint pairs.
                ij = torch.nonzero(merge_mat).tolist()
                pairs = sorted(ij, key=lambda p: (-float(c_mat[p[0],p[1]]+c_mat[p[1],p[0]]), p[0], p[1]))
                keep = torch.zeros_like(merge_mat); used = set()
                for i,j in pairs:
                    if i != j and i not in used and j not in used:
                        keep[i,j] = True; used.update((i,j))
                merge_mat = keep
            before = self.c_mask_num
            result = super().merge_masks(frame_id, merge_mat, c_mat, count_merge=count_merge, full_geo_merge=full_geo_merge)
            if result[1]:
                self.audit['union_calls'] += 1
                self.audit['instances_before'] += before
                self.audit['instances_after'] += self.c_mask_num
            return result
    module.Scene_rep = Scene
    raw = args.args[1:] if args.args[:1] == ['--'] else args.args
    sys.argv = [str(repo/'main.py')] + raw
    # main.py currently exposes max_frame_num as a module constant.  Override
    # it textually so a short ablation sequence still executes the normal
    # save/evaluation path without changing the baseline source tree.
    main_src = (repo/'main.py').read_text()
    if args.max_frames is not None:
        main_src = main_src.replace('max_frame_num = None',
                                    f'max_frame_num = {int(args.max_frames)}', 1)
    start = time.perf_counter(); status = 'failed'
    try:
        exec(compile(main_src, str(repo/'main.py'), 'exec'), {'__name__':'__main__','__file__':str(repo/'main.py')})
        torch.cuda.synchronize(); status = 'complete'
    finally:
        for scene in scenes:
            data = dict(status=status, mode=args.mode, seed=27, seconds=time.perf_counter()-start,
                        audit=scene.audit, config=effective_config, argv=raw,
                        runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        source_hash=hashlib.sha256((repo/'Scene_rep.py').read_bytes()).hexdigest())
            dest = Path(scene.seq_output_dir)/'controlled_manifest.json'
            dest.write_text(json.dumps(data,indent=2))
            print('CONTROLLED', args.mode, status, data['seconds'], flush=True)

if __name__ == '__main__': main()
