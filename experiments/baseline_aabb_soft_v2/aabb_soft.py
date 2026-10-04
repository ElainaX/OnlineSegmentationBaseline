"""Bounded AABB evidence for the ORIGINAL MaskGraph similarity stage.

Inputs are world-space voxel origins in metres, not image boxes or indices.
No new candidate pairs, tracks, occupied volume or output masks are created.
"""
import torch


@torch.no_grad()
def soft_similarity(base, containment, boxes, valid, threshold, overlap_threshold,
                    strength=0.1, band=0.1, positive_strength=None,
                    negative_strength=None, confidence=None,
                    confidence_threshold=0.0, negative_confidence_floor=0.25):
    if positive_strength is None:
        positive_strength = strength
    if negative_strength is None:
        negative_strength = strength
    if (positive_strength == 0 and negative_strength == 0) or band == 0:
        return base, {}
    if not 0 <= positive_strength <= 0.25 or not 0 <= negative_strength <= 0.25 or not 0 < band <= 0.5:
        raise ValueError('strengths must be in [0,.25], band in (0,.5]')
    if confidence is None:
        confidence = torch.ones((boxes.shape[0],), device=boxes.device,
                                dtype=boxes.dtype)
    confidence = confidence.to(device=boxes.device, dtype=boxes.dtype)
    trusted = confidence >= float(confidence_threshold)
    # These are precisely the original overall-similarity eligibility rules.
    # Supporter and containment merge paths are left to the original code.
    eligible = ((containment > overlap_threshold) &
                (containment.T > overlap_threshold) &
                valid[:, None] & valid[None, :] &
                trusted[:, None] & trusted[None, :] &
                ((base - threshold).abs() <= band))
    eligible.fill_diagonal_(False)
    ij = eligible.nonzero(as_tuple=False)
    if ij.numel() == 0:
        return base, dict(eligible_pairs=0, promoted_pairs=0, demoted_pairs=0)
    a, b = boxes[ij[:, 0]], boxes[ij[:, 1]]
    intersection = (torch.minimum(a[:, 1], b[:, 1]) -
                    torch.maximum(a[:, 0], b[:, 0])).clamp_min(0).prod(-1)
    va, vb = (a[:, 1]-a[:, 0]).prod(-1), (b[:, 1]-b[:, 0]).prod(-1)
    iou = intersection / (va+vb-intersection).clamp_min(1e-12)
    coverage = intersection / (va*vb).sqrt().clamp_min(1e-12)
    evidence = (.5*iou + .5*coverage).clamp(0, 1)
    result = base.clone()
    pair_confidence = torch.minimum(confidence[ij[:, 0]], confidence[ij[:, 1]])
    signed = 2 * evidence - 1
    pair_strength = torch.where(
        signed >= 0,
        torch.as_tensor(positive_strength, device=base.device, dtype=base.dtype),
        torch.as_tensor(negative_strength, device=base.device, dtype=base.dtype))
    # Visibility confidence controls only negative evidence. A partial view
    # should weaken a claim that two masks are incompatible, but it should
    # not discard positive spatial agreement supported by the observed box.
    # Keep strong negative evidence for well-observed boxes, while retaining
    # a conservative floor for partial views.  Multiplying by the raw
    # confidence made the previous adaptive setting (.04) too weak even on
    # static scenes where the carving was reliable.
    negative_confidence_floor = float(max(0.0, min(1.0, negative_confidence_floor)))
    negative_multiplier = pair_confidence.clamp_min(negative_confidence_floor)
    confidence_multiplier = torch.where(signed >= 0,
                                        torch.ones_like(pair_confidence),
                                        negative_multiplier)
    result[ij[:, 0], ij[:, 1]] += pair_strength * signed * confidence_multiplier
    upper = torch.triu(eligible, diagonal=1)
    stats = dict(eligible_pairs=int(upper.sum().item()),
                 promoted_pairs=int((upper & (base <= threshold) & (result > threshold)).sum().item()),
                 demoted_pairs=int((upper & (base > threshold) & (result <= threshold)).sum().item()),
                 mean_confidence=float(pair_confidence.mean().item()))
    return result, stats


class AABBMetrics:
    def __init__(self, original, scene, strength, band, positive_strength=None,
                 negative_strength=None, confidence_threshold=0.0):
        self.original, self.scene = original, scene
        self.strength, self.band = strength, band
        self.positive_strength = strength if positive_strength is None else positive_strength
        self.negative_strength = strength if negative_strength is None else negative_strength
        self.confidence_threshold = confidence_threshold
        self.negative_confidence_floor = 0.25
        self.records = []
        self.frame = -1

    def __getattr__(self, name):
        return getattr(self.original, name)

    @torch.no_grad()
    def adjust_sparse_pair(self, base_score, left, right, containment):
        """Apply carved-AABB evidence to one sparse candidate."""
        scene = self.scene
        boxes = getattr(scene, 'aabb_boxes_override', None)
        if boxes is None or boxes.shape[0] != scene.c_mask_num:
            return base_score
        if left >= boxes.shape[0] or right >= boxes.shape[0]:
            return base_score
        a, b = boxes[left], boxes[right]
        if not torch.isfinite(torch.stack((a, b))).all():
            return base_score
        intersection = (torch.minimum(a[1], b[1]) -
                        torch.maximum(a[0], b[0])).clamp_min(0).prod()
        va = (a[1] - a[0]).clamp_min(1e-6).prod()
        vb = (b[1] - b[0]).clamp_min(1e-6).prod()
        iou = intersection / (va + vb - intersection).clamp_min(1e-12)
        coverage = intersection / (va * vb).sqrt().clamp_min(1e-12)
        evidence = (.5 * iou + .5 * coverage).clamp(0, 1)
        signed = 2 * evidence - 1
        strength = self.positive_strength if signed >= 0 else self.negative_strength
        return float(base_score) + float(strength * signed)

    @torch.no_grad()
    def compute_final_sim_mat(self, c_mat, sem_feature_mat, geo_feature_mat, **kwargs):
        outputs = self.original.compute_final_sim_mat(c_mat, sem_feature_mat,
                                                      geo_feature_mat, **kwargs)
        if ((self.positive_strength == 0 and self.negative_strength == 0) or
                self.band == 0 or not kwargs.get('use_IoU', True)):
            return outputs
        # This integration is intentionally limited to the full NxN call in
        # update_masks; never silently alter other callers or feature-only scores.
        if any(kwargs.get(k) is not None for k in ('masks_to_merged', 'masks_to_receive')) or kwargs.get('retain_max', False):
            raise ValueError('AABB adapter requires the full similarity matrix')
        scene = self.scene
        override = getattr(scene, 'aabb_boxes_override', None)
        if override is not None and override.shape[0] == scene.c_mask_num:
            boxes = override.to(device=c_mat.device, dtype=c_mat.dtype)
            valid = torch.isfinite(boxes).all(dim=2).all(dim=1)
        else:
            boxes = c_mat.new_zeros((scene.c_mask_num, 2, 3))
            valid = torch.zeros(scene.c_mask_num, dtype=torch.bool, device=c_mat.device)
            # Rebuild from the actual current instance membership, avoiding stale
            # IDs after baseline merges/removals. Boxes never introduce new voxels.
            half_voxel = float(scene.cfg['scene']['voxel_size']) / 2
            for i in range(scene.c_mask_num):
                points = scene.mask_voxels_coords[i]
                if len(points):
                    boxes[i, 0] = points.amin(dim=0)-half_voxel
                    boxes[i, 1] = points.amax(dim=0)+half_voxel
                    valid[i] = torch.isfinite(boxes[i]).all()
        confidence = getattr(scene, 'aabb_box_confidence', None)
        if confidence is None:
            confidence = torch.ones((scene.c_mask_num,), device=c_mat.device,
                                    dtype=c_mat.dtype)
        final, stats = soft_similarity(outputs[0], c_mat, boxes, valid,
                                       scene.sim_merge_thresh,
                                       scene.merge_overlap_thresh, self.strength,
                                       self.band,
                                       self.positive_strength,
                                       self.negative_strength,
                                       confidence,
                                       self.confidence_threshold,
                                       self.negative_confidence_floor)
        stats.update(frame=self.frame, instances=scene.c_mask_num)
        self.records.append(stats)
        return (final,) + outputs[1:]
