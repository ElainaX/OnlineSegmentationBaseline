"""Sparse form of baseline_aabb_soft_v2.soft_similarity (confidence=1).

Only the existing mutual-overlap, near-threshold candidate scores change.
Supporter joins, containment passes, disjoint ranking and geometry stay intact.
"""
import torch


@torch.no_grad()
def adjust_carved_pair_scores(base, pairs, forward, backward, boxes, valid,
                              threshold, overlap_threshold, strength=.15, band=.10):
    if strength == 0 or band == 0 or base.numel() == 0:
        return base, dict(eligible_pairs=0, promoted_pairs=0, demoted_pairs=0)
    if not 0 <= strength <= .25 or not 0 < band <= .5:
        raise ValueError('strength must be in [0,.25], band in (0,.5]')
    left, right = pairs[:, 0], pairs[:, 1]
    eligible = ((forward > overlap_threshold) & (backward > overlap_threshold)
                & valid[left] & valid[right] & (left != right)
                & ((base - threshold).abs() <= band))
    q = eligible.nonzero(as_tuple=False).flatten()
    if not q.numel():
        return base, dict(eligible_pairs=0, promoted_pairs=0, demoted_pairs=0)
    a, b = boxes[left[q]], boxes[right[q]]
    intersection = (torch.minimum(a[:, 1], b[:, 1]) -
                    torch.maximum(a[:, 0], b[:, 0])).clamp_min(0).prod(-1)
    va, vb = (a[:, 1]-a[:, 0]).prod(-1), (b[:, 1]-b[:, 0]).prod(-1)
    iou = intersection / (va+vb-intersection).clamp_min(1e-12)
    coverage = intersection / (va*vb).sqrt().clamp_min(1e-12)
    evidence = (.5*iou + .5*coverage).clamp(0, 1)
    result = base.clone()
    result[q] += strength * (2*evidence - 1)
    return result, dict(eligible_pairs=int(q.numel()),
        promoted_pairs=int(((base[q] <= threshold) & (result[q] > threshold)).sum().item()),
        demoted_pairs=int(((base[q] > threshold) & (result[q] <= threshold)).sum().item()))
