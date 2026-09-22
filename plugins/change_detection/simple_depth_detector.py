import cv2
import numpy as np
import torch

from .debug_visualizer import ChangeDebugVisualizer
from .multiview_verifier import MultiViewVerifier
from .tsdf_depth_renderer import TSDFDepthRenderer


DEFAULT_CONFIG = {
    "enabled": False,
    "interval": 5,
    "warmup_frames": 30,
    "depth_threshold": 0.05,
    "color_error_threshold": 0.10,
    "require_current_color": True,
    "min_region_pixels": 30,
    "morph_kernel": 3,
    "min_conflict_voxels": 50,
    "min_visible_voxels": 50,
    "min_conflict_ratio": 0.20,
    "visibility_depth_tolerance": 0.075,
    "min_map_weight": 3.0,
    "range_map_down_factor": 8,
    "keyframe_min_frame_gap": 5,
    "keyframe_translation": 0.05,
    "keyframe_rotation_deg": 5.0,
    "max_keyframes": 200,
    "max_validation_views": 8,
    "min_verified_views": 2,
    "min_covisible_pixels": 20,
    "history_depth_tolerance": 0.075,
    "history_color_tolerance": 0.15,
    "history_projected_agreement": 0.20,
    "history_mask_coverage": 0.01,
    "require_history_color": True,
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


class MultiViewDepthChangeDetector:
    """GaME-style multi-view removal verification for OnlineAnySeg.

    A current TSDF depth/color conflict is accepted only when the old surface
    agrees with RGB-D and the original 2D instance mask in several covisible
    historical keyframes. This stage is intentionally non-destructive: it
    records stale tracks and keyframe occlusion masks but does not yet remove
    or re-fuse TSDF voxels.
    """

    def __init__(self, cfg):
        self.cfg = _merged_config(cfg.get("change_detection", {}))
        self.enabled = bool(self.cfg["enabled"])
        self.renderer = TSDFDepthRenderer(self.cfg)
        self.verifier = MultiViewVerifier(self.cfg)
        self.visualizer = ChangeDebugVisualizer(self.cfg["debug"])
        self.stale_tracks = set()
        self._warned_missing_color = False

    def _detection_due(self, frame_id):
        return frame_id % int(self.cfg["interval"]) == 0

    def _should_detect(self, frame_id, scene_rep):
        return (
            frame_id >= int(self.cfg["warmup_frames"])
            and (self._detection_due(frame_id)
                 or self.visualizer.preview_due(frame_id))
            and scene_rep.voxel_grids.get_block_num > 0
        )

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
            if stats[component_id, cv2.CC_STAT_AREA] >= int(
                    self.cfg["min_region_pixels"]):
                cleaned[labels == component_id] = 1
        return torch.from_numpy(cleaned.astype(bool)).to(mask.device)

    @staticmethod
    def _backproject_map_surface(conflict_mask, rendered_depth, rendered_color,
                                 intrinsics, pose_c2w):
        ys, xs = torch.where(conflict_mask)
        if ys.numel() == 0:
            empty = torch.empty(
                (0, 3), dtype=torch.float32, device=rendered_depth.device)
            return empty, empty.clone()
        z = rendered_depth[ys, xs]
        intrinsics = intrinsics.to(device=z.device, dtype=torch.float32)
        x = (xs.float() - intrinsics[0, 2]) * z / intrinsics[0, 0]
        y = (ys.float() - intrinsics[1, 2]) * z / intrinsics[1, 1]
        camera_points = torch.stack((x, y, z), dim=1)
        pose_c2w = pose_c2w.to(device=z.device, dtype=torch.float32)
        world_points = camera_points @ pose_c2w[:3, :3].T + pose_c2w[:3, 3]
        return world_points, rendered_color[ys, xs]

    def _visible_instance_voxels(self, instance, rendered_depth, pose_w2c,
                                 intrinsics):
        points = instance.mask_voxel_coords.to(rendered_depth.device)
        if points.numel() == 0:
            return 0, torch.zeros_like(rendered_depth, dtype=torch.bool)
        pose_w2c = pose_w2c.to(device=points.device, dtype=torch.float32)
        intrinsics = intrinsics.to(device=points.device, dtype=torch.float32)
        camera_points = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
        z = camera_points[:, 2]
        positive = z > 0
        safe_z = z.clamp_min(1e-6)
        xs = torch.round(
            intrinsics[0, 0] * camera_points[:, 0] / safe_z
            + intrinsics[0, 2]).long()
        ys = torch.round(
            intrinsics[1, 1] * camera_points[:, 1] / safe_z
            + intrinsics[1, 2]).long()
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
                projection_np,
                np.ones((kernel_size, kernel_size), np.uint8))
            projection_mask = torch.from_numpy(
                projection_np.astype(bool)).to(rendered_depth.device)
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
            aliases = {int(instance.track_uid)} | {
                int(uid) for uid in instance.parent_track_uids}
            if aliases & self.stale_tracks:
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

    def _current_conflict_mask(self, color_img, observed_depth, rendered):
        rendered_depth = rendered["depth"].to(observed_depth.device)
        rendered_color = rendered.get("color")
        if rendered_color is not None:
            rendered_color = rendered_color.to(color_img.device)
        valid = (
            torch.isfinite(rendered_depth)
            & torch.isfinite(observed_depth)
            & (rendered_depth > 0)
            & (observed_depth > 0)
        )
        depth_conflict = valid & (
            observed_depth - rendered_depth
            > float(self.cfg["depth_threshold"]))

        if rendered_color is None:
            if bool(self.cfg["require_current_color"]):
                if not self._warned_missing_color:
                    print(
                        "[ChangeDetection][WARN] TSDF color rendering is unavailable; "
                        "current color verification rejects all candidates.")
                    self._warned_missing_color = True
                return depth_conflict & False, rendered_depth, None
            return depth_conflict, rendered_depth, None

        color_error = torch.mean(
            torch.abs(color_img.to(rendered_color) - rendered_color), dim=-1)
        color_conflict = color_error > float(self.cfg["color_error_threshold"])
        if bool(self.cfg["require_current_color"]):
            depth_conflict &= color_conflict
        return depth_conflict, rendered_depth, rendered_color

    @torch.no_grad()
    def before_integrate(self, frame_id, color_img, depth_img, pose_c2w,
                         scene_rep, segmentation=None, seg_flag=False):
        if not self.enabled:
            return []

        events = []
        if self._should_detect(frame_id, scene_rep):
            pose_w2c = torch.inverse(pose_c2w)
            rendered = self.renderer.render(scene_rep, pose_w2c)
            if rendered is not None:
                raw_mask, rendered_depth, rendered_color = (
                    self._current_conflict_mask(color_img, depth_img, rendered))
                conflict_mask = self._clean_conflict_mask(raw_mask)
                self.visualizer.write_depth_preview(
                    scene_rep, frame_id, color_img, depth_img, rendered_depth,
                    raw_mask, conflict_mask,
                    float(self.cfg["depth_threshold"]))

                if (self._detection_due(frame_id)
                        and scene_rep.c_mask_num > 0
                        and conflict_mask.any()
                        and rendered_color is not None):
                    conflict_points, _ = self._backproject_map_surface(
                        conflict_mask, rendered_depth, rendered_color,
                        scene_rep.intrinsics, pose_c2w)
                    conflict_voxels = (
                        scene_rep.voxel_grids.world_coords2voxel_coords(
                            conflict_points))
                    conflict_voxels = torch.unique(conflict_voxels, dim=0)
                    candidates = self._associate_instances(
                        scene_rep, conflict_voxels, rendered_depth, pose_w2c)

                    for candidate in candidates:
                        instance = scene_rep.maskGraph.instance_dict[
                            candidate["merged_mask_id"]]
                        candidate_mask = (
                            conflict_mask & candidate["instance_projection_mask"])
                        candidate_points, candidate_colors = (
                            self._backproject_map_surface(
                                candidate_mask, rendered_depth, rendered_color,
                                scene_rep.intrinsics, pose_c2w))
                        verification = self.verifier.verify(
                            instance, candidate_points, candidate_colors)
                        if not verification["accepted"]:
                            print(
                                "[ChangeDetection] frame_%d track_%d mask_%d: "
                                "current conflict %.3f, history %d/%d -> REJECTED"
                                % (frame_id, candidate["track_uid"],
                                   candidate["merged_mask_id"],
                                   candidate["conflict_ratio"],
                                   verification["verified_view_count"],
                                   verification["required_view_count"]))
                            continue

                        track_uid = int(candidate["track_uid"])
                        self.stale_tracks.add(track_uid)
                        self.verifier.mark_verified_regions_occluded(verification)
                        event = {
                            "track_uid": track_uid,
                            "old_state": "ACTIVE",
                            "new_state": "STALE",
                            "verified_view_count": int(
                                verification["verified_view_count"]),
                            "required_view_count": int(
                                verification["required_view_count"]),
                            "views": verification["views"],
                            # Compatibility fields for existing debug output.
                            "independent_hit_count": int(
                                verification["verified_view_count"]),
                            "hits": verification["views"],
                        }
                        events.append(event)
                        print(
                            "[ChangeDetection] frame_%d track_%d mask_%d: "
                            "%d/%d voxels (%.3f), history %d/%d -> STALE"
                            % (frame_id, track_uid,
                               candidate["merged_mask_id"],
                               candidate["conflict_voxels"],
                               candidate["visible_voxels"],
                               candidate["conflict_ratio"],
                               verification["verified_view_count"],
                               verification["required_view_count"]))
                        self.visualizer.write_event(
                            scene_rep, frame_id, color_img, depth_img,
                            rendered_depth, candidate_mask, candidate_points,
                            instance, candidate, event)

        # Insert after detection so the current frame cannot validate itself.
        if bool(seg_flag) and segmentation is not None:
            self.verifier.add_keyframe(
                frame_id, color_img, depth_img, segmentation, pose_c2w,
                scene_rep.intrinsics)
        return events

    def after_mask_update(self, scene_rep):
        if not self.enabled or not self.stale_tracks:
            return
        stale_now = set(self.stale_tracks)
        for instance in scene_rep.maskGraph.instance_dict.values():
            aliases = {int(instance.track_uid)} | {
                int(uid) for uid in instance.parent_track_uids}
            if aliases & stale_now:
                stale_now.add(int(instance.track_uid))
        self.stale_tracks = stale_now


# Existing integrations importing the old name receive the new implementation.
SimpleDepthChangeDetector = MultiViewDepthChangeDetector
