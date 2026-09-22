import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class KeyframeRecord:
    frame_id: int
    color: np.ndarray
    depth: np.ndarray
    segmentation: np.ndarray
    pose_c2w: np.ndarray
    intrinsics: np.ndarray


def _as_numpy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class MultiViewVerifier:
    """GaME-style historical-view verification for stale TSDF surfaces.

    The verifier keeps a bounded CPU keyframe bank. A current-frame depth/color
    conflict is accepted only when the same old surface agrees with RGB-D and
    the original 2D instance mask in several covisible historical keyframes.
    """

    def __init__(self, cfg):
        self.min_keyframe_gap = int(cfg.get("keyframe_min_frame_gap", 5))
        self.min_translation = float(cfg.get("keyframe_translation", 0.05))
        self.min_rotation_deg = float(cfg.get("keyframe_rotation_deg", 5.0))
        self.max_keyframes = int(cfg.get("max_keyframes", 200))
        self.max_validation_views = int(cfg.get("max_validation_views", 8))
        self.min_verified_views = int(cfg.get("min_verified_views", 2))
        self.min_covisible_pixels = int(cfg.get("min_covisible_pixels", 20))
        self.depth_tolerance = float(cfg.get("history_depth_tolerance", 0.075))
        self.color_tolerance = float(cfg.get("history_color_tolerance", 0.15))
        self.min_projected_agreement = float(
            cfg.get("history_projected_agreement", 0.20))
        self.min_mask_coverage = float(cfg.get("history_mask_coverage", 0.01))
        self.require_history_color = bool(cfg.get("require_history_color", True))
        self.keyframes = OrderedDict()
        self.occlusion_masks = {}

    @staticmethod
    def _rotation_difference_deg(pose_a, pose_b):
        relative = pose_a[:3, :3].T @ pose_b[:3, :3]
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        return math.degrees(math.acos(float(cosine)))

    def _is_keyframe(self, frame_id, pose_c2w):
        if not self.keyframes:
            return True
        last = next(reversed(self.keyframes.values()))
        if frame_id - last.frame_id < self.min_keyframe_gap:
            return False
        translation = np.linalg.norm(pose_c2w[:3, 3] - last.pose_c2w[:3, 3])
        rotation = self._rotation_difference_deg(last.pose_c2w, pose_c2w)
        return translation >= self.min_translation or rotation >= self.min_rotation_deg

    def add_keyframe(self, frame_id, color, depth, segmentation, pose_c2w,
                     intrinsics):
        if segmentation is None:
            return False
        pose_np = _as_numpy(pose_c2w).astype(np.float32, copy=True)
        if not self._is_keyframe(int(frame_id), pose_np):
            return False

        color_np = _as_numpy(color)
        if color_np.max(initial=0) <= 1.0:
            color_np = color_np * 255.0
        record = KeyframeRecord(
            frame_id=int(frame_id),
            color=np.clip(color_np, 0, 255).astype(np.uint8),
            depth=_as_numpy(depth).astype(np.float32, copy=True),
            segmentation=_as_numpy(segmentation).astype(np.int32, copy=True),
            pose_c2w=pose_np,
            intrinsics=_as_numpy(intrinsics).astype(np.float32, copy=True),
        )
        self.keyframes[record.frame_id] = record
        while len(self.keyframes) > self.max_keyframes:
            old_frame_id, _ = self.keyframes.popitem(last=False)
            self.occlusion_masks.pop(old_frame_id, None)
        return True

    @staticmethod
    def _rasterize_points(points_world, point_colors, record):
        """Project candidate TSDF samples and keep the nearest sample per pixel."""
        height, width = record.depth.shape
        if points_world.size == 0:
            return None

        pose_w2c = np.linalg.inv(record.pose_c2w)
        points_camera = points_world @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
        z = points_camera[:, 2]
        positive = z > 1e-6
        safe_z = np.maximum(z, 1e-6)
        xs = np.rint(
            record.intrinsics[0, 0] * points_camera[:, 0] / safe_z
            + record.intrinsics[0, 2]).astype(np.int64)
        ys = np.rint(
            record.intrinsics[1, 1] * points_camera[:, 1] / safe_z
            + record.intrinsics[1, 2]).astype(np.int64)
        valid = positive & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not valid.any():
            return None

        xs, ys, z = xs[valid], ys[valid], z[valid]
        colors = point_colors[valid]
        flat = ys * width + xs
        # Sorting by depth first makes the first occurrence of each pixel the
        # closest candidate surface sample.
        depth_order = np.argsort(z, kind="stable")
        _, first = np.unique(flat[depth_order], return_index=True)
        selected = depth_order[first]
        return ys[selected], xs[selected], z[selected], colors[selected]

    def _validate_record(self, record, points_world, point_colors,
                         source_mask_ids):
        projection = self._rasterize_points(points_world, point_colors, record)
        if projection is None:
            return None
        ys, xs, projected_depth, projected_color = projection
        projected_pixels = int(ys.size)
        if projected_pixels < self.min_covisible_pixels:
            return None

        observed_depth = record.depth[ys, xs]
        depth_valid = np.isfinite(observed_depth) & (observed_depth > 0)
        depth_agree = depth_valid & (
            np.abs(observed_depth - projected_depth) <= self.depth_tolerance)

        observed_color = record.color[ys, xs].astype(np.float32) / 255.0
        color_agree = (
            np.abs(observed_color - projected_color).mean(axis=1)
            <= self.color_tolerance)
        geometry_agree = depth_agree
        if self.require_history_color:
            geometry_agree &= color_agree

        source_mask_ids = sorted({int(mask_id) for mask_id in source_mask_ids})
        source_mask = np.isin(record.segmentation, source_mask_ids)
        sample_in_mask = source_mask[ys, xs]
        supported = geometry_agree & sample_in_mask
        support_pixels = int(supported.sum())
        mask_pixels = int(source_mask.sum())
        projected_agreement = support_pixels / max(projected_pixels, 1)
        mask_coverage = support_pixels / max(mask_pixels, 1)
        verified = (
            support_pixels >= self.min_covisible_pixels
            and projected_agreement >= self.min_projected_agreement
            and mask_coverage >= self.min_mask_coverage
        )
        return {
            "frame_id": int(record.frame_id),
            "source_mask_ids": source_mask_ids,
            "projected_pixels": projected_pixels,
            "support_pixels": support_pixels,
            "projected_agreement": float(projected_agreement),
            "mask_coverage": float(mask_coverage),
            "depth_agreement": float(depth_agree.sum() / max(projected_pixels, 1)),
            "color_agreement": float(color_agree.sum() / max(projected_pixels, 1)),
            "verified": bool(verified),
        }

    def verify(self, instance, conflict_points, conflict_colors):
        points_world = _as_numpy(conflict_points).astype(np.float32, copy=False)
        point_colors = _as_numpy(conflict_colors).astype(np.float32, copy=False)
        if points_world.size == 0 or point_colors.size == 0:
            return {
                "accepted": False,
                "verified_view_count": 0,
                "required_view_count": self.min_verified_views,
                "views": [],
            }

        source_masks_by_frame = {}
        for frame_id, mask_id in instance.ori_mask_list:
            source_masks_by_frame.setdefault(int(frame_id), set()).add(int(mask_id))

        candidate_records = []
        for frame_id, source_mask_ids in source_masks_by_frame.items():
            record = self.keyframes.get(frame_id)
            if record is None:
                continue
            result = self._validate_record(
                record, points_world, point_colors, source_mask_ids)
            if result is not None:
                candidate_records.append(result)

        candidate_records.sort(
            key=lambda item: item["projected_pixels"], reverse=True)
        candidate_records = candidate_records[:self.max_validation_views]
        verified = [item for item in candidate_records if item["verified"]]
        return {
            "accepted": len(verified) >= self.min_verified_views,
            "verified_view_count": len(verified),
            "required_view_count": self.min_verified_views,
            "views": candidate_records,
        }

    def mark_verified_regions_occluded(self, verification):
        """Keep GaME-style per-keyframe stale masks for later map repair."""
        for view in verification["views"]:
            if not view["verified"]:
                continue
            frame_id = int(view["frame_id"])
            record = self.keyframes.get(frame_id)
            if record is None:
                continue
            region = np.isin(record.segmentation, view["source_mask_ids"])
            previous = self.occlusion_masks.get(frame_id)
            self.occlusion_masks[frame_id] = (
                region if previous is None else (previous | region))
