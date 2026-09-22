import cv2
import numpy as np
import torch

from .debug_visualizer import ChangeDebugVisualizer
from .temporal_evidence import TemporalEvidenceBank
from .tsdf_depth_renderer import TSDFDepthRenderer


DEFAULT_CONFIG = {
    "enabled": False,
    "interval": 5,
    "warmup_frames": 30,
    "depth_threshold": 0.05,
    "min_region_pixels": 30,
    "morph_kernel": 3,
    "min_conflict_voxels": 50,
    "min_visible_voxels": 50,
    "min_conflict_ratio": 0.20,
    "visibility_depth_tolerance": 0.075,
    "confirm_hits": 3,
    "min_hit_frame_gap": 10,
    "min_hit_translation": 0.05,
    "min_hit_rotation_deg": 5.0,
    "min_map_weight": 3.0,
    "range_map_down_factor": 8,
    "debug": {
        "enabled": True,
        "preview_interval": 50,
        "max_history_frames": 6,
        "gray_value": 128,
    },
}


def _merged_config(user_cfg):
    cfg = dict(DEFAULT_CONFIG)
    user_cfg = user_cfg or {}
    for key, value in user_cfg.items():
        if key == "debug":
            debug_cfg = dict(DEFAULT_CONFIG["debug"])
            debug_cfg.update(value or {})
            cfg["debug"] = debug_cfg
        else:
            cfg[key] = value
    return cfg


class SimpleDepthChangeDetector:
    """GaME-inspired disappearance detection for OnlineAnySeg's TSDF map.

    This first-stage plugin is deliberately non-destructive: it records temporal
    evidence and diagnostics, but does not erase TSDF voxels or remove masks.
    """

    def __init__(self, cfg):
        self.cfg = _merged_config(cfg.get("change_detection", {}))
        self.enabled = bool(self.cfg["enabled"])
        self.renderer = TSDFDepthRenderer(self.cfg)
        self.evidence = TemporalEvidenceBank(self.cfg)
        self.visualizer = ChangeDebugVisualizer(self.cfg["debug"])

    def _detection_due(self, frame_id):
        return frame_id % int(self.cfg["interval"]) == 0

    def _should_run(self, frame_id, scene_rep):
        if not self.enabled:
            return False
        if frame_id < int(self.cfg["warmup_frames"]):
            return False
        if not (
                self._detection_due(frame_id)
                or self.visualizer.preview_due(frame_id)):
            return False
        return scene_rep.voxel_grids.get_block_num > 0

    def _clean_conflict_mask(self, mask):
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        kernel_size = int(self.cfg["morph_kernel"])
        if kernel_size > 1:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
            mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_np, connectivity=8)
        cleaned = np.zeros_like(mask_np)
        for component_id in range(1, component_count):
            if stats[component_id, cv2.CC_STAT_AREA] >= int(self.cfg["min_region_pixels"]):
                cleaned[labels == component_id] = 1
        return torch.from_numpy(cleaned.astype(bool)).to(mask.device)

    @staticmethod
    def _backproject_map_surface(conflict_mask, rendered_depth, intrinsics, pose_c2w):
        ys, xs = torch.where(conflict_mask)
        if ys.numel() == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=rendered_depth.device)
        z = rendered_depth[ys, xs]
        intrinsics = intrinsics.to(device=z.device, dtype=torch.float32)
        x = (xs.float() - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.float() - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_points = torch.stack((x, y, z), dim=1)
        pose_c2w = pose_c2w.to(device=z.device, dtype=torch.float32)
        return camera_points @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]

    def _visible_instance_voxels(self, instance, rendered_depth, pose_w2c, intrinsics):
        points = instance.mask_voxel_coords.to(rendered_depth.device)
        if points.numel() == 0:
            return 0, torch.zeros_like(rendered_depth, dtype=torch.bool)
        pose_w2c = pose_w2c.to(device=points.device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=points.device, dtype=torch.float32)
        camera_points = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
        z = camera_points[:, 2]
        positive = z > 0
        safe_z = z.clamp_min(1e-6)
        xs = torch.round(intrinsics[0, 0] * camera_points[:, 0] / safe_z + intrinsics[0, 2]).long()
        ys = torch.round(intrinsics[1, 1] * camera_points[:, 1] / safe_z + intrinsics[1, 2]).long()
        height, width = rendered_depth.shape
        in_frame = positive & (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not in_frame.any():
            return 0, torch.zeros_like(rendered_depth, dtype=torch.bool)

        valid_indices = torch.where(in_frame)[0]
        map_depth = rendered_depth[ys[valid_indices], xs[valid_indices]]
        tolerance = float(self.cfg["visibility_depth_tolerance"])
        visible = (map_depth > 0) & torch.isfinite(map_depth) & (
            (map_depth - z[valid_indices]).abs() <= tolerance)
        projection_mask = torch.zeros_like(rendered_depth, dtype=torch.bool)
        visible_indices = valid_indices[visible]
        projection_mask[ys[visible_indices], xs[visible_indices]] = True
        if projection_mask.any():
            kernel_size = max(int(self.cfg["morph_kernel"]), 1)
            projection_np = projection_mask.detach().cpu().numpy().astype(np.uint8)
            projection_np = cv2.dilate(
                projection_np, np.ones((kernel_size, kernel_size), np.uint8))
            projection_mask = torch.from_numpy(projection_np.astype(bool)).to(rendered_depth.device)
        return int(visible.sum().item()), projection_mask

    def _associate_instances(self, scene_rep, conflict_voxel_coords,
                             rendered_depth, pose_w2c):
        if conflict_voxel_coords.numel() == 0:
            return []
        valid_mask_ids = list(range(scene_rep.c_mask_num))
        mask_ids, counts, _ = scene_rep.voxel_hash_table.query_mask_w_mapping(
            conflict_voxel_coords,
            scene_rep.global2merged_mask_id,
            valid_mask_ids,
        )

        candidates = []
        for mask_id_tensor, count_tensor in zip(mask_ids, counts):
            mask_id = int(mask_id_tensor.item())
            conflict_count = int(count_tensor.item())
            instance = scene_rep.maskGraph.instance_dict.get(mask_id)
            if instance is None:
                continue
            visible_count, projection_mask = self._visible_instance_voxels(
                instance, rendered_depth, pose_w2c, scene_rep.intrinsics)
            if visible_count < int(self.cfg["min_visible_voxels"]):
                continue
            ratio = min(float(conflict_count) / max(visible_count, 1), 1.0)
            if conflict_count < int(self.cfg["min_conflict_voxels"]):
                continue
            if ratio < float(self.cfg["min_conflict_ratio"]):
                continue
            candidates.append({
                "merged_mask_id": mask_id,
                "track_uid": int(instance.track_uid),
                "conflict_voxels": conflict_count,
                "visible_voxels": visible_count,
                "conflict_ratio": ratio,
                "depth_threshold": float(self.cfg["depth_threshold"]),
                "instance_projection_mask": projection_mask,
            })
        return candidates

    @torch.no_grad()
    def before_integrate(self, frame_id, color_img, depth_img, pose_c2w, scene_rep):
        if not self._should_run(frame_id, scene_rep):
            return []

        pose_w2c = torch.inverse(pose_c2w)
        rendered = self.renderer.render(scene_rep, pose_w2c)
        if rendered is None:
            return []
        rendered_depth = rendered["depth"].to(depth_img.device)
        observed_depth = depth_img.to(rendered_depth.device)
        valid = (
            torch.isfinite(rendered_depth)
            & torch.isfinite(observed_depth)
            & (rendered_depth > 0)
            & (observed_depth > 0)
            & (observed_depth < scene_rep.depth_far)
        )

        # A farther observation means the current ray penetrates the old map
        # surface. The opposite sign indicates a new foreground/occluder.
        raw_conflict_mask = valid & (
            observed_depth - rendered_depth > float(self.cfg["depth_threshold"]))
        conflict_mask = self._clean_conflict_mask(raw_conflict_mask)
        self.visualizer.write_depth_preview(
            scene_rep, frame_id, color_img, observed_depth, rendered_depth,
            raw_conflict_mask, conflict_mask,
            float(self.cfg["depth_threshold"]))

        # A preview-only frame must not alter the temporal evidence cadence.
        if not self._detection_due(frame_id):
            return []
        if scene_rep.c_mask_num == 0 or not conflict_mask.any():
            return []

        conflict_points = self._backproject_map_surface(
            conflict_mask, rendered_depth, scene_rep.intrinsics, pose_c2w)
        conflict_voxel_coords = scene_rep.voxel_grids.world_coords2voxel_coords(conflict_points)
        conflict_voxel_coords = torch.unique(conflict_voxel_coords, dim=0)
        candidates = self._associate_instances(
            scene_rep, conflict_voxel_coords, rendered_depth, pose_w2c)

        events = []
        for candidate in candidates:
            instance = scene_rep.maskGraph.instance_dict[candidate["merged_mask_id"]]
            evidence_event = self.evidence.add(
                candidate["track_uid"], frame_id, pose_c2w,
                candidate["conflict_voxels"], candidate["visible_voxels"],
                candidate["conflict_ratio"])
            if evidence_event is None:
                continue
            events.append(evidence_event)
            print(
                "[ChangeDetection] frame_%d track_%d mask_%d: %d/%d voxels "
                "(%.3f), hit %d/%d -> %s" % (
                    frame_id, candidate["track_uid"], candidate["merged_mask_id"],
                    candidate["conflict_voxels"], candidate["visible_voxels"],
                    candidate["conflict_ratio"], evidence_event["independent_hit_count"],
                    int(self.cfg["confirm_hits"]), evidence_event["new_state"]))
            candidate_conflict_mask = (
                conflict_mask & candidate["instance_projection_mask"])
            candidate_conflict_points = self._backproject_map_surface(
                candidate_conflict_mask, rendered_depth,
                scene_rep.intrinsics, pose_c2w)
            self.visualizer.write_event(
                scene_rep, frame_id, color_img, observed_depth, rendered_depth,
                candidate_conflict_mask, candidate_conflict_points,
                instance, candidate, evidence_event)
        return events

    def after_mask_update(self, scene_rep):
        if not self.enabled:
            return
        self.evidence.merge_instance_aliases(scene_rep.maskGraph.instance_dict.values())
