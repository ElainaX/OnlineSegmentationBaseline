"""Controlled one-module-at-a-time ablation runner for original OnlineAnySeg."""
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
import torch.nn.functional as F

MODULES = {
    "baseline", "no_semantic", "no_geometry", "overlap_only",
    "no_supporter", "no_containment", "no_boundary", "no_visibility",
    "weighted_semantic",
}

def _clusters(mat):
    from tool.helper_functions import do_clustering
    if torch.is_tensor(mat):
        return do_clustering(mat)
    return []

def _weighted_feature(features, qualities, sizes, high=0.80):
    # features are pre-merge vectors; normalize only for comparison and output.
    f = features.float()
    if f.shape[0] == 1:
        return f[0]
    fn = F.normalize(f, dim=-1)
    sim = fn @ fn.T
    off = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    pair_mean = float(sim[off].mean().item()) if off.any() else 1.0
    q = torch.as_tensor(qualities, device=f.device, dtype=f.dtype).clamp_min(1e-4)
    s = torch.as_tensor(sizes, device=f.device, dtype=f.dtype).clamp_min(1.)
    # Saturate size so a wall cannot dominate a small object solely by area.
    w = q * torch.sqrt(s / s.mean().clamp_min(1.)).clamp(max=2.)
    if pair_mean >= high:
        out = (fn * w[:, None]).sum(dim=0)
    else:
        # Preserve the most reliable prototype when evidence is inconsistent.
        # Reliability is smooth-view quality times saturated visible support.
        out = fn[torch.argmax(w)]
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--module", choices=sorted(MODULES), default="baseline")
    p.add_argument("--seed", type=int, default=27)
    p.add_argument("--tag", default="")
    p.add_argument("args", nargs=argparse.REMAINDER)
    o = p.parse_args()
    repo = Path(o.repo).resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(repo / "third_party" / "FCGF"))
    random.seed(o.seed); np.random.seed(o.seed); torch.manual_seed(o.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(o.seed)

    import Scene_rep as source
    from Metrics import Metrics as BaseMetrics
    original_scene = source.Scene_rep
    scenes = []

    class AblationMetrics(BaseMetrics):
        @torch.no_grad()
        def compute_final_sim_mat(self, c_mat, sem_feature_mat, geo_feature_mat,
                                  masks_to_merged=None, masks_to_receive=None,
                                  retain_max=False, use_IoU=True):
            outputs = super().compute_final_sim_mat(
                c_mat, sem_feature_mat, geo_feature_mat,
                masks_to_merged=masks_to_merged,
                masks_to_receive=masks_to_receive,
                retain_max=retain_max, use_IoU=use_IoU)
            final, iou, sem, geo = outputs
            if o.module == "no_semantic":
                final = (iou + geo) if use_IoU else geo
            elif o.module == "no_geometry":
                final = (iou + sem) if use_IoU else sem
            elif o.module == "overlap_only":
                final = iou if use_IoU else torch.zeros_like(iou)
            return final, iou, sem, geo

    class AblationScene(original_scene):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.ablation_module = o.module
            self.ablation_counts = {"merge_calls": 0, "merge_pairs": 0}
            self.semantic_quality = {}
            self._prev_pose = None
            self._prev_delta = None
            if o.module == "no_supporter":
                self.merge_supporter_num = 10**9
            if o.module == "no_containment":
                self.merge_contain_iter_num = 0
            if o.module == "no_boundary":
                self.maskGraph.consider_bound_pts = False
            if o.module == "no_visibility":
                self.maskGraph.mask_visible_threshold = -1.0
                self.maskGraph.mask_visible_min_overlap = 0
                self.maskGraph.contain_min_threshold = 0.0
            scenes.append(self)

        @torch.no_grad()
        def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w,
                             frustum_block_coords, seg_image, mask_features):
            # Compute a reliability scalar for each newly added mask from
            # trajectory smoothness and saturated visible support.
            pose = pose_c2w.detach().float()
            smooth = 1.0
            if self._prev_pose is not None:
                rel = torch.linalg.inv(self._prev_pose) @ pose
                dt = float(torch.linalg.norm(rel[:3, 3]).item())
                tr_t = torch.trace(rel[:3, :3]).clamp(-1., 3.)
                tr = float(tr_t.item())
                dr = float(torch.acos(((tr_t - 1.) / 2.).clamp(-1., 1.)).item())
                delta = torch.tensor([dt, dr], device=pose.device)
                if self._prev_delta is not None:
                    accel = torch.linalg.norm(delta - self._prev_delta).item()
                    smooth = float(np.exp(-dt / .05 - dr / .12 - accel / .05))
                self._prev_delta = delta
            self._prev_pose = pose.clone()
            result = super().insert_seg_frame(
                frame_id, color_img, depth_img, pose_c2w,
                frustum_block_coords, seg_image, mask_features)
            if o.module == "weighted_semantic" and result[0]:
                sizes = [float(x.shape[0]) for x in result[1]]
                med = float(np.median(sizes)) if sizes else 1.0
                for mid, sz in zip(result[0], sizes):
                    self.semantic_quality[int(mid)] = smooth * min(2.0, np.sqrt(max(sz, 1.) / max(med, 1.)))
            return result

        @torch.no_grad()
        def merge_masks(self, frame_id, merge_mat, c_mat,
                        count_merge=False, full_geo_merge=True):
            self.ablation_counts["merge_calls"] += 1
            if torch.is_tensor(merge_mat):
                self.ablation_counts["merge_pairs"] += int(torch.count_nonzero(merge_mat).item()) // 2
            if o.module != "weighted_semantic":
                return super().merge_masks(frame_id, merge_mat, c_mat,
                                           count_merge=count_merge, full_geo_merge=full_geo_merge)
            clusters = _clusters(merge_mat)
            rows_to_keep = [int(c.item()) for c in clusters if c.shape[0] == 1]
            rows_to_merge = [c.tolist() for c in clusters if c.shape[0] > 1]
            old_features = self.mask_features[:self.c_mask_num].detach().clone()
            old_sizes = list(self.merged_mask_size)
            old_q = dict(self.semantic_quality)
            result = super().merge_masks(frame_id, merge_mat, c_mat,
                                         count_merge=count_merge, full_geo_merge=full_geo_merge)
            if result[1] and rows_to_merge:
                new_id = len(rows_to_keep)
                for cluster in rows_to_merge:
                    fs = old_features[cluster]
                    qs = [old_q.get(int(i), 1.0) for i in cluster]
                    ss = [old_sizes[int(i)] if int(i) < len(old_sizes) else 1.0 for i in cluster]
                    fused = _weighted_feature(fs, qs, ss)
                    self.mask_features[new_id] = fused
                    if new_id in self.maskGraph.instance_dict:
                        self.maskGraph.instance_dict[new_id].mask_sem_feature = fused
                    self.semantic_quality[new_id] = float(np.average(qs, weights=np.maximum(ss, 1.0)))
                    new_id += 1
                # Reindex qualities for kept rows and discard stale IDs.
                kept_q = {j: old_q.get(old, 1.0) for j, old in enumerate(rows_to_keep)}
                kept_q.update({k: v for k, v in self.semantic_quality.items() if k >= len(rows_to_keep) and k < self.c_mask_num})
                self.semantic_quality = kept_q
            return result

    source.Metrics = AblationMetrics
    source.Scene_rep = AblationScene
    raw = o.args[1:] if o.args[:1] == ["--"] else o.args
    sys.argv = [str(repo / "main.py")] + raw
    main_text = (repo / "main.py").read_text()
    start = time.perf_counter(); status = "failed"
    try:
        exec(compile(main_text, str(repo / "main.py"), "exec"),
             {"__name__": "__main__", "__file__": str(repo / "main.py")})
        status = "complete"
    finally:
        files = ["main.py", "Scene_rep.py", "Metrics.py", "MaskGraph.py",
                 "Instance.py", "PC_extractor.py", "voxelized_points.py",
                 "Dataset/dataset.py", "tool/post_process.py"]
        for scene in scenes:
            manifest = {
                "status": status, "seconds": time.perf_counter() - start,
                "module": o.module, "seed": o.seed, "repo": str(repo),
                "argv": raw,
                "original_hashes": {f: hashlib.sha256((repo / f).read_bytes()).hexdigest() for f in files},
                "ablation_counts": scene.ablation_counts,
                "note": "Original flow; selected module only. No Carving-AABB or depth shrink."
            }
            dest = Path(scene.seq_output_dir) / "merge_ablation_manifest.json"
            dest.write_text(json.dumps(manifest, indent=2))
            print("MERGE_ABLATION", status, "module", o.module,
                  "seconds", manifest["seconds"], "manifest", dest, flush=True)

if __name__ == "__main__":
    main()




