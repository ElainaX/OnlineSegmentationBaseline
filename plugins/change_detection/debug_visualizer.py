import json
import os

import cv2
import numpy as np
import open3d as o3d
import torch


class ChangeDebugVisualizer:
    """Write periodic depth previews and event-based 2D/3D diagnostics."""

    def __init__(self, cfg):
        self.enabled = bool(cfg.get("enabled", True))
        self.preview_interval = int(cfg.get("preview_interval", 50))
        self.max_history_frames = int(cfg.get("max_history_frames", 6))
        self.gray_value = int(cfg.get("gray_value", 128))
        self.event_counter = 0

    def preview_due(self, frame_id):
        return (
            self.enabled
            and self.preview_interval > 0
            and frame_id % self.preview_interval == 0
        )

    @staticmethod
    def _rgb_uint8(image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        image = np.asarray(image)
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _mask_numpy(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        return np.asarray(mask).astype(bool)

    @staticmethod
    def _colorize_depth(depth, depth_max):
        depth = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        normalized = np.zeros_like(depth, dtype=np.uint8)
        normalized[valid] = np.clip(depth[valid] / depth_max * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    @staticmethod
    def _write_rgb(path, rgb):
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _colorize_signed_residual(residual, valid, scale):
        """Render closer observations blue and farther observations red."""
        residual = np.asarray(residual, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        scale = max(float(scale), 1e-6)
        normalized = np.clip(residual / scale, -1.0, 1.0)
        colored = np.zeros((*residual.shape, 3), dtype=np.uint8)
        colored[..., 0] = (np.clip(-normalized, 0.0, 1.0) * 255).astype(np.uint8)
        colored[..., 2] = (np.clip(normalized, 0.0, 1.0) * 255).astype(np.uint8)
        colored[~valid] = 0
        return colored

    def write_depth_preview(self, scene_rep, frame_id, color_img,
                            observed_depth, rendered_depth,
                            raw_conflict_mask, cleaned_conflict_mask,
                            depth_threshold):
        """Periodically save depth diagnostics even when no event is accepted."""
        if not self.preview_due(frame_id):
            return None

        preview_dir = os.path.join(
            scene_rep.seq_output_dir, "change_debug", "previews",
            "frame_%06d" % frame_id)
        os.makedirs(preview_dir, exist_ok=True)

        rgb = self._rgb_uint8(color_img)
        observed_np = observed_depth.detach().cpu().numpy()
        rendered_np = rendered_depth.detach().cpu().numpy()
        raw_conflict_np = self._mask_numpy(raw_conflict_mask)
        cleaned_conflict_np = self._mask_numpy(cleaned_conflict_mask)

        valid = (
            np.isfinite(observed_np)
            & np.isfinite(rendered_np)
            & (observed_np > 0)
            & (rendered_np > 0)
        )
        residual = observed_np - rendered_np
        residual_bgr = self._colorize_signed_residual(
            residual, valid, depth_threshold * 4.0)
        observed_bgr = self._colorize_depth(observed_np, scene_rep.depth_far)
        rendered_bgr = self._colorize_depth(rendered_np, scene_rep.depth_far)

        self._write_rgb(os.path.join(preview_dir, "current_rgb.png"), rgb)
        cv2.imwrite(os.path.join(preview_dir, "observed_depth.png"), observed_bgr)
        cv2.imwrite(os.path.join(preview_dir, "tsdf_rendered_depth.png"), rendered_bgr)
        cv2.imwrite(os.path.join(preview_dir, "signed_depth_residual.png"), residual_bgr)
        cv2.imwrite(
            os.path.join(preview_dir, "raw_conflict_mask.png"),
            raw_conflict_np.astype(np.uint8) * 255)
        cv2.imwrite(
            os.path.join(preview_dir, "cleaned_conflict_mask.png"),
            cleaned_conflict_np.astype(np.uint8) * 255)

        gray_rgb = rgb.copy()
        gray_rgb[cleaned_conflict_np] = self.gray_value
        contours, _ = cv2.findContours(
            cleaned_conflict_np.astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gray_rgb, contours, -1, (255, 0, 0), 2)
        cv2.putText(
            gray_rgb,
            "frame=%d raw=%d cleaned=%d" % (
                frame_id, raw_conflict_np.sum(), cleaned_conflict_np.sum()),
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 0), 1, cv2.LINE_AA)
        self._write_rgb(
            os.path.join(preview_dir, "current_conflict_gray.png"), gray_rgb)

        tiles = [
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            observed_bgr,
            rendered_bgr,
            residual_bgr,
            cv2.cvtColor(gray_rgb, cv2.COLOR_RGB2BGR),
        ]
        labels = ["RGB", "Observed", "TSDF", "Residual", "Conflict"]
        sheet_tiles = []
        for tile, label in zip(tiles, labels):
            tile = cv2.resize(tile, (320, 240))
            cv2.putText(
                tile, label, (8, 232), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
            sheet_tiles.append(tile)
        cv2.imwrite(
            os.path.join(preview_dir, "comparison_sheet.png"),
            np.concatenate(sheet_tiles, axis=1))

        valid_residual = residual[valid]
        summary = {
            "frame_id": int(frame_id),
            "depth_threshold": float(depth_threshold),
            "valid_depth_pixels": int(valid.sum()),
            "raw_conflict_pixels": int(raw_conflict_np.sum()),
            "cleaned_conflict_pixels": int(cleaned_conflict_np.sum()),
            "mean_signed_residual": (
                float(valid_residual.mean()) if valid_residual.size else None),
            "mean_absolute_residual": (
                float(np.abs(valid_residual).mean())
                if valid_residual.size else None),
        }
        with open(
                os.path.join(preview_dir, "summary.json"),
                "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

        print(
            "[ChangeDetection][Preview] frame_%d: raw=%d cleaned=%d -> %s"
            % (frame_id, raw_conflict_np.sum(),
               cleaned_conflict_np.sum(), preview_dir))
        return preview_dir

    @staticmethod
    def _select_history(ori_mask_list, limit):
        latest_per_frame = {}
        for frame_id, mask_id in sorted(ori_mask_list):
            latest_per_frame[int(frame_id)] = int(mask_id)
        items = list(latest_per_frame.items())
        if len(items) <= limit:
            return items
        indices = np.linspace(0, len(items) - 1, limit, dtype=int)
        return [items[index] for index in indices]

    @staticmethod
    def _project_world_points(points_world, pose_c2w, intrinsics, height, width):
        if points_world.size == 0:
            return np.zeros((height, width), dtype=bool)
        if isinstance(pose_c2w, torch.Tensor):
            pose_c2w = pose_c2w.detach().cpu().numpy()
        if isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.detach().cpu().numpy()
        pose_w2c = np.linalg.inv(np.asarray(pose_c2w))
        points_camera = points_world @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
        z = points_camera[:, 2]
        valid = z > 1e-6
        x = np.zeros_like(z)
        y = np.zeros_like(z)
        x[valid] = (
            intrinsics[0, 0] * points_camera[valid, 0] / z[valid]
            + intrinsics[0, 2])
        y[valid] = (
            intrinsics[1, 1] * points_camera[valid, 1] / z[valid]
            + intrinsics[1, 2])
        x = np.rint(x).astype(np.int64)
        y = np.rint(y).astype(np.int64)
        valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
        projection = np.zeros((height, width), dtype=np.uint8)
        projection[y[valid], x[valid]] = 1
        projection = cv2.dilate(projection, np.ones((5, 5), np.uint8))
        return projection.astype(bool)

    def _write_history(self, event_dir, dataset, instance, conflict_points,
                       evidence_event):
        history_dir = os.path.join(event_dir, "history")
        os.makedirs(history_dir, exist_ok=True)
        saved_images = []
        view_by_frame = {
            int(view["frame_id"]): view for view in evidence_event.get("views", [])
        }
        if view_by_frame:
            selected = []
            for frame_id, view in view_by_frame.items():
                for mask_id in view.get("source_mask_ids", []):
                    selected.append((frame_id, int(mask_id)))
            selected = self._select_history(selected, self.max_history_frames)
        else:
            selected = self._select_history(
                instance.ori_mask_list, self.max_history_frames)
        points_world = conflict_points.detach().cpu().numpy()

        latest_seg_img = getattr(dataset, "latest_seg_img", None)
        for frame_id, mask_id in selected:
            try:
                color, _, pose_c2w, segmentation, _, seg_flag = dataset[frame_id]
            except Exception as error:
                print("[ChangeDetection][WARN] Could not load history frame_%d: %s" % (frame_id, error))
                continue

            rgb = self._rgb_uint8(color)
            contradiction_mask = self._project_world_points(
                points_world, pose_c2w, dataset.cam_intrinsic,
                rgb.shape[0], rgb.shape[1])
            if bool(seg_flag):
                if isinstance(segmentation, torch.Tensor):
                    segmentation = segmentation.detach().cpu().numpy()
                mask = np.asarray(segmentation) == mask_id
                contradiction_mask &= mask
                gray = np.full_like(rgb, self.gray_value)
                rgb[contradiction_mask] = gray[contradiction_mask]
                instance_contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb, instance_contours, -1, (0, 128, 255), 2)
            else:
                gray = np.full_like(rgb, self.gray_value)
                rgb[contradiction_mask] = gray[contradiction_mask]
            contradiction_contours, _ = cv2.findContours(
                contradiction_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contradiction_contours, -1, (255, 0, 0), 2)

            cv2.putText(
                rgb, "frame=%d mask=%d" % (frame_id, mask_id), (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)
            validation = view_by_frame.get(int(frame_id))
            if validation is not None:
                cv2.putText(
                    rgb,
                    "%s proj=%.2f mask=%.3f d=%.2f c=%.2f" % (
                        "VERIFIED" if validation["verified"] else "REJECTED",
                        validation["projected_agreement"],
                        validation["mask_coverage"],
                        validation["depth_agreement"],
                        validation["color_agreement"]),
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 255, 0) if validation["verified"] else (255, 64, 64),
                    1, cv2.LINE_AA)
            save_path = os.path.join(history_dir, "frame_%06d_mask_%03d.png" % (frame_id, mask_id))
            self._write_rgb(save_path, rgb)
            saved_images.append(rgb)
        if latest_seg_img is not None:
            dataset.latest_seg_img = latest_seg_img
        return saved_images

    def write_event(self, scene_rep, frame_id, color_img, observed_depth,
                    rendered_depth, conflict_mask, conflict_points, instance,
                    candidate, evidence_event):
        if not self.enabled:
            return

        self.event_counter += 1
        event_name = "event_%06d_frame_%06d_track_%06d" % (
            self.event_counter, frame_id, instance.track_uid)
        event_dir = os.path.join(scene_rep.seq_output_dir, "change_debug", event_name)
        os.makedirs(event_dir, exist_ok=True)

        rgb = self._rgb_uint8(color_img)
        observed_np = observed_depth.detach().cpu().numpy()
        rendered_np = rendered_depth.detach().cpu().numpy()
        conflict_np = self._mask_numpy(conflict_mask)

        self._write_rgb(os.path.join(event_dir, "current_rgb.png"), rgb)
        cv2.imwrite(
            os.path.join(event_dir, "observed_depth.png"),
            self._colorize_depth(observed_np, scene_rep.depth_far))
        cv2.imwrite(
            os.path.join(event_dir, "tsdf_rendered_depth.png"),
            self._colorize_depth(rendered_np, scene_rep.depth_far))

        residual = observed_np - rendered_np
        residual_vis = np.zeros_like(residual, dtype=np.uint8)
        valid = np.isfinite(residual) & (rendered_np > 0) & (observed_np > 0)
        residual_vis[valid] = np.clip(
            residual[valid] / max(candidate["depth_threshold"] * 4.0, 1e-6) * 255.0,
            0, 255).astype(np.uint8)
        cv2.imwrite(
            os.path.join(event_dir, "signed_depth_residual.png"),
            cv2.applyColorMap(residual_vis, cv2.COLORMAP_HOT))

        gray_rgb = rgb.copy()
        gray_rgb[conflict_np] = self.gray_value
        contours, _ = cv2.findContours(
            conflict_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(gray_rgb, contours, -1, (255, 0, 0), 2)
        cv2.putText(
            gray_rgb,
            "track=%d state=%s views=%d/%d ratio=%.3f" % (
                instance.track_uid, evidence_event["new_state"],
                evidence_event.get("verified_view_count", 0),
                evidence_event.get("required_view_count", 0),
                candidate["conflict_ratio"]),
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        self._write_rgb(os.path.join(event_dir, "current_conflict_gray.png"), gray_rgb)

        instance_overlay = rgb.copy()
        projection_mask = self._mask_numpy(candidate["instance_projection_mask"])
        projection_contours, _ = cv2.findContours(
            projection_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(instance_overlay, projection_contours, -1, (255, 255, 0), 2)
        cv2.drawContours(instance_overlay, contours, -1, (255, 0, 0), 2)
        self._write_rgb(os.path.join(event_dir, "current_instance_overlay.png"), instance_overlay)

        if conflict_points.numel() > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(conflict_points.detach().cpu().numpy())
            pcd.paint_uniform_color([1.0, 0.0, 0.0])
            o3d.io.write_point_cloud(os.path.join(event_dir, "conflict_points.ply"), pcd)

        history_images = self._write_history(
            event_dir, scene_rep.dataset, instance, conflict_points,
            evidence_event)
        tiles = [gray_rgb] + history_images
        if tiles:
            target_h, target_w = 240, 320
            tiles = [cv2.resize(tile, (target_w, target_h)) for tile in tiles]
            contact_sheet = np.concatenate(tiles, axis=1)
            self._write_rgb(os.path.join(event_dir, "contact_sheet.png"), contact_sheet)

        summary = {
            "frame_id": int(frame_id),
            "track_uid": int(instance.track_uid),
            "merged_mask_id": int(candidate["merged_mask_id"]),
            "state": evidence_event["new_state"],
            "verified_view_count": evidence_event.get("verified_view_count", 0),
            "required_view_count": evidence_event.get("required_view_count", 0),
            "conflict_voxels": int(candidate["conflict_voxels"]),
            "visible_voxels": int(candidate["visible_voxels"]),
            "conflict_ratio": float(candidate["conflict_ratio"]),
            "depth_threshold": float(candidate["depth_threshold"]),
            "views": evidence_event.get("views", []),
            "historical_masks": [[int(f), int(m)] for f, m in instance.ori_mask_list],
        }
        with open(os.path.join(event_dir, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
