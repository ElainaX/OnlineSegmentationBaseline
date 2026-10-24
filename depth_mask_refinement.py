"""Conservative depth based refinement of 2-D segmentation masks.

This is an OVI-MAP style visibility prior: a 2-D mask is allowed to retain
only points close to its robust front depth.  Unknown depth is never invented;
when the rule would remove too much of a mask, the original mask is kept.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


@torch.no_grad()
def shrink_segmentation_by_depth(seg_image: torch.Tensor, depth_image: torch.Tensor, cfg: dict) -> Tuple[torch.Tensor, Dict[str, int]]:
    opts = cfg.get("seg", {}).get("depth_shrink", {})
    if not bool(opts.get("enabled", True)):
        return seg_image, {"masks": 0, "changed": 0, "removed_pixels": 0}

    seg = seg_image.clone()
    depth = depth_image
    while depth.ndim > 2:
        depth = depth[0]
    depth = depth.to(device=seg.device, dtype=torch.float32)
    valid_depth = torch.isfinite(depth) & (depth > 0)
    quantile = float(opts.get("front_quantile", 0.20))
    margin_m = float(opts.get("margin_m", 0.08))
    min_pixels = int(opts.get("min_pixels", 25))
    min_keep_ratio = float(opts.get("min_keep_ratio", 0.40))
    remove_unknown = bool(opts.get("remove_unknown", False))

    labels = torch.unique(seg)
    stats = {"masks": 0, "changed": 0, "removed_pixels": 0, "total_pixels": 0}
    for label in labels.tolist():
        if int(label) <= 0:
            continue
        mask = seg == int(label)
        pixel_count = int(mask.sum().item())
        if pixel_count < min_pixels:
            continue
        stats["masks"] += 1
        stats["total_pixels"] += pixel_count
        samples = depth[mask & valid_depth]
        if samples.numel() < min_pixels:
            continue
        front = torch.quantile(samples, quantile)
        keep = mask & valid_depth & (depth <= front + margin_m)
        if not remove_unknown:
            keep |= mask & ~valid_depth
        keep_count = int(keep.sum().item())
        # A front-depth prior is a guard against background leakage, not a
        # reason to erase a mostly unobserved object.
        if keep_count < max(min_pixels, int(min_keep_ratio * pixel_count)):
            continue
        removed = mask & ~keep
        removed_count = int(removed.sum().item())
        if removed_count:
            seg[removed] = 0
            stats["changed"] += 1
            stats["removed_pixels"] += removed_count
    return seg, stats

