"""IG coarse export with supported fine substitution for final valid tracks.

Global geometry and ordinary instance masks are owned by IG. Only eligible
small tracks replay their original RGB-D observations into a fine TSDF.
"""
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c

from .independent_tsdf import LocalTSDF, TSDFOptions
from .multires_policy import MultiViewResolutionPolicy, project_mask_points, assess_track
from .coarse_preserving_export import multiview_surface_support, substitute_fine_regions


def reserve_window(volume, views, K):
    """Reserve all blocks touched by a replay batch before integration."""
    keys = []
    opt = volume.options
    intrinsic = o3c.Tensor(np.asarray(K, dtype=np.float64))
    for packet, mask in views:
        depth = packet['depth']
        valid = (
            np.isfinite(depth)
            & (depth > opt.near_m)
            & (depth < opt.far_m)
            & np.asarray(mask, dtype=bool)
        )
        if not valid.any():
            continue
        image = o3d.t.geometry.Image(
            np.ascontiguousarray(np.where(valid, depth, 0), dtype=np.float32)
        ).to(volume.device)
        extrinsic = o3c.Tensor(np.linalg.inv(packet['pose']) @ volume.anchor)
        try:
            coords = volume.grid.compute_unique_block_coordinates(
                image, intrinsic, extrinsic, 1.0, opt.far_m,
                trunc_voxel_multiplier=opt.trunc_voxels,
            )
        except RuntimeError as error:
            if 'No block is touched' in str(error):
                continue
            raise
        keys.append(coords.cpu().numpy())
    needed = len(np.unique(np.concatenate(keys), axis=0)) if keys else 0
    if needed > opt.max_blocks:
        raise MemoryError(
            'persistent IT volume exceeds block budget: %d > %d'
            % (needed, opt.max_blocks)
        )
    if needed * 4 > volume.grid.hashmap().capacity():
        volume.grid.hashmap().reserve(needed * 4 + 1)


class InstanceTSDFAdapter:
    """Buffer observations; preserve the final IG coarse export."""

    def __init__(self, cfg, owner, directory):
        self.owner = owner
        self.cfg = cfg
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.options = TSDFOptions(
            voxel_m=float(cfg['scene']['voxel_size']),
            block_resolution=int(cfg['scene']['block_resolution']),
            trunc_voxels=float(cfg['scene'].get('trunc_voxel_multiplier', 4.0)),
            near_m=float(cfg['cam']['depth_near']),
            far_m=float(cfg['cam']['depth_far']),
            max_blocks=int(cfg['scene']['block_count']),
            device=owner.device.upper(),
            extraction_weight=float(cfg['scene'].get('extraction_weight', 0.5)),
        )
        # The global TSDF and each instance volume deliberately have separate
        # resolutions.  Fine volumes are selected only after stable, metric
        # observations; a single projected mask area is not a 3-D size cue.
        mr = cfg.get('multi_resolution', {})
        self.multires_enabled = bool(mr.get('enabled', False))
        self.fine_voxel_m = float(mr.get('fine_voxel_size_m', self.options.voxel_m))
        self.fine_block_count = int(mr.get('fine_block_count', self.options.max_blocks))
        self.fine_min_views = int(mr.get('fine_min_views', mr.get('min_fine_views', 3)))
        self.fine_max_views = int(mr.get('fine_max_views', 5))
        self.fine_max_extent_m = float(mr.get('fine_max_extent_m', 0.45))
        self.fine_max_relative_spread = float(mr.get('fine_max_relative_spread', 0.50))
        self.small_mask_max_pixels = int(mr.get('small_mask_max_pixels', 1200))
        physical_truncation_m = float(mr.get(
            'physical_truncation_m', self.options.voxel_m * self.options.trunc_voxels))
        fine_trunc_voxels = physical_truncation_m / max(self.fine_voxel_m, 1e-8)
        self.fine_options = TSDFOptions(
            voxel_m=self.fine_voxel_m,
            block_resolution=self.options.block_resolution,
            # Keep the physical truncation band fixed when voxel size changes.
            trunc_voxels=fine_trunc_voxels,
            near_m=self.options.near_m, far_m=self.options.far_m,
            max_blocks=self.fine_block_count,
            device=self.options.device,
            extraction_weight=self.options.extraction_weight,
        )
        self.resolution_policy = MultiViewResolutionPolicy(
            min_views=self.fine_min_views, max_views=self.fine_max_views,
            max_extent_m=self.fine_max_extent_m,
            max_relative_spread=self.fine_max_relative_spread)
        self.global_resolution_by_id = {}
        self.global_first_observation = {}
        self.fine_track_ids = set()
        self.track_decisions = {}
        self.geometry_policy = str(mr.get('geometry_policy', 'hybrid'))
        if self.geometry_policy != 'hybrid':
            raise ValueError('This branch requires coarse-preserving hybrid export')
        self.min_fine_views = int(mr.get('min_fine_views', 3))
        self.fine_replace_radius_m = float(mr.get('fine_replace_radius_m', 0.05))
        self.K = owner.intrinsics.detach().cpu().numpy().astype(np.float64)
        self.packets = []
        self.windows = []
        self._last_flush_frame = None
        self.fragments = []
        self.audit = dict(
            buffer_seconds=0.0, local_tsdf_seconds=0.0,
            persistence_seconds=0.0, export_seconds=0.0,
            buffered_frames_peak=0, windows=0, fragments=0,
            accepted_masks=0, integrated_fragment_frames=0,
            raw_surface_points=0, peak_volume_attribute_bytes=0,
            volume_blocks_total=0, unresolved_resolution_observations=0,
        )

    def observe(self, frame_id, rgb, depth, pose, seg, pairs, excluded_pixels=None):
        started = time.perf_counter()
        rgb_np = (rgb.detach().cpu().numpy() * 255).astype(np.uint8)
        # Match the global GeometryScene's millimetre quantisation so a
        # resolution ablation does not also compare two depth numerics.
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        scale = float(self.cfg['cam']['depth_scale'])
        depth_np = np.where(np.isfinite(depth_np) & (depth_np > 0), depth_np, 0)
        depth_np = (depth_np * scale).astype(np.uint16).astype(np.float32) / scale
        pose_np = pose.detach().cpu().numpy().astype(np.float64)
        labels = (
            seg.detach().cpu().numpy().astype(np.int32)
            if seg is not None else np.zeros(depth_np.shape, dtype=np.int32)
        )
        accepted = [(int(x['mask_id']), int(x['global_mask_id'])) for x in pairs]
        if excluded_pixels is not None:
            rejected = np.asarray(excluded_pixels, dtype=bool)
            if rejected.shape != depth_np.shape:
                raise ValueError('excluded_pixels/depth shape mismatch')
            depth_np = np.where(rejected, 0.0, depth_np)
        # Observation IDs are append-only. Resolution is resolved on current
        # track groups (at online merge boundaries and again at export).
        for raw_id, gid in accepted:
            self.global_first_observation[gid] = dict(
                frame=int(frame_id), raw_mask_id=raw_id,
                observed_pixels=int(np.count_nonzero(labels == raw_id)))
        self.packets.append(dict(frame=int(frame_id), rgb=rgb_np, depth=depth_np,
                                 pose=pose_np, labels=labels, pairs=accepted))
        self.audit['accepted_masks'] += len(accepted)
        self.audit['buffered_frames_peak'] = max(
            self.audit['buffered_frames_peak'], len(self.packets)
        )
        self.audit['buffer_seconds'] += time.perf_counter() - started

    def flush(self, decision_frame):
        """Record a causal boundary; final-track geometry is built at export."""
        if self._last_flush_frame == int(decision_frame):
            return
        if self.packets:
            self.windows.append(dict(decision_frame=int(decision_frame),
                                     frames=[p['frame'] for p in self.packets]))
        self._last_flush_frame = int(decision_frame)
        self.audit['windows'] = len(self.windows)

    def _final_identity(self, global_id):
        try:
            return int(self.owner.global2merged_mask_id[int(global_id)])
        except (IndexError, KeyError, TypeError):
            return -1

    def _frame_masks(self, packet):
        labels = packet['labels']
        # Raw masks whose final identity was deleted must return to the
        # background residual.  Removing every raw ID before final deletion
        # decisions creates silent holes in the exported point cloud.
        live_raw_ids = [raw for raw, gid in packet['pairs']
                        if self._final_identity(gid) >= 0]
        residual = (~np.isin(labels, live_raw_ids) if live_raw_ids
                    else np.ones(labels.shape, dtype=bool))
        by_mid = defaultdict(list)
        for raw, gid in packet['pairs']:
            mid = self._final_identity(gid)
            if mid >= 0:
                by_mid[mid].append(raw)
        return residual, by_mid

    def _integrate_volume(self, views):
        return self._integrate_volume_with_options(views, self.options)

    def _integrate_volume_with_options(self, views, options):
        volume = LocalTSDF(options)
        started = time.perf_counter()
        reserve_window(volume, views, self.K)
        integrated = 0
        for packet, mask in views:
            if volume.integrate(packet['rgb'], packet['depth'], self.K,
                                packet['pose'], mask):
                integrated += 1
        xyz, colors = volume.surface()
        o3c.cuda.synchronize()
        self.audit['local_tsdf_seconds'] += time.perf_counter() - started
        self.audit['peak_volume_attribute_bytes'] = max(
            self.audit['peak_volume_attribute_bytes'],
            volume.bytes_allocated_attributes())
        self.audit['volume_blocks_total'] += int(volume.grid.hashmap().size())
        self.audit['integrated_fragment_frames'] += integrated
        self.audit['raw_surface_points'] += len(xyz)
        return xyz, colors, integrated

    def _track_is_fine(self, mid, views):
        if not self.multires_enabled:
            return False
        decision = assess_track(views, self.K, self.options.near_m,
                                self.options.far_m, self.fine_min_views,
                                self.fine_max_views, self.fine_max_extent_m,
                                self.fine_max_relative_spread)
        self.track_decisions[int(mid)] = decision
        return decision['resolution'] == 'fine'

    def export(self, frame_id, output):
        self.flush(frame_id)
        started = time.perf_counter()
        output = Path(output)
        coarse_dir = output / 'ig_audit'
        # The original export defines the valid instances and their ordering.
        # Do not rerun DBSCAN, mask ordering, feature export or global fusion.
        with np.load(coarse_dir / 'ckpt_final.npz') as saved:
            arrays = {key: saved[key].copy() for key in saved.files}
        cloud = o3d.io.read_point_cloud(str(coarse_dir / 'final.ply'))
        points = np.asarray(cloud.points).copy()
        colors = np.asarray(cloud.colors).copy()
        masks = arrays['pred_masks']
        ids = json.loads((coarse_dir / 'export_instance_ids.json').read_text())
        if masks.shape != (len(points), len(ids)) or len(set(ids)) != len(ids):
            raise ValueError('Final IG masks and exported track identities disagree')
        valid_ids = set(ids)
        track_views = defaultdict(dict)
        for packet in self.packets:
            _, by_mid = self._frame_masks(packet)
            for mid, raw_ids in by_mid.items():
                if mid in valid_ids:
                    track_views[mid][packet['frame']] = (
                        packet, np.isin(packet['labels'], raw_ids))

        candidates = {}
        self.fine_track_ids.clear()
        self.fragments = []
        for column, mid in enumerate(ids):
            views = [track_views[mid][f] for f in sorted(track_views[mid])]
            if not self._track_is_fine(mid, views):
                continue
            xyz, rgb, integrated = self._integrate_volume_with_options(views, self.fine_options)
            stable = multiview_surface_support(
                xyz, views, self.K, self.options.near_m, self.options.far_m,
                self.fine_replace_radius_m, self.fine_min_views)
            self.fragments.append(dict(global_id=mid, resolution='fine',
                surface_points=len(xyz), stable_surface_points=int(stable.sum()),
                observation_frames=[p['frame'] for p, _ in views], integrated_frames=integrated))
            if stable.any():
                candidates[column] = (xyz[stable], rgb[stable] / 255.0)

        xyz, rgb, final_masks, retained, replacements = substitute_fine_regions(
            points, colors, masks, candidates, self.fine_replace_radius_m)
        for record in replacements:
            record['global_id'] = ids[record['column']]
            self.fine_track_ids.add(record['global_id'])
        ordinary = [col for col, mid in enumerate(ids) if mid not in self.fine_track_ids]
        # Exact invariants on original coarse point support, including overlap.
        removed = np.ones(len(points), dtype=bool); removed[retained] = False
        if ordinary and masks[removed][:, ordinary].any():
            raise AssertionError('Fine replacement removed ordinary-instance support')
        if not np.array_equal(xyz[:len(retained)], points[retained]):
            raise AssertionError('Retained coarse coordinates changed')
        if not np.array_equal(final_masks[:len(retained)], masks[retained]):
            raise AssertionError('Retained coarse mask memberships changed')
        if ordinary and final_masks[len(retained):, ordinary].any():
            raise AssertionError('Fine points added membership to ordinary masks')
        if not replacements:
            # Byte-for-byte fallback includes exact ordering and all NPZ fields.
            for name in ['final.ply', 'ckpt_final.npz']:
                shutil.copyfile(coarse_dir / name, output / name)
        else:
            arrays['pred_masks'] = final_masks
            np.savez(output / 'ckpt_final.npz', **arrays)
            self.owner.save_pc(xyz, rgb, str(output / 'final.ply'))
        (output / 'export_instance_ids.json').write_text(json.dumps(ids))
        np.savez_compressed(output / 'coarse_export_provenance.npz',
                            retained_coarse_indices=retained,
                            original_point_count=len(points))
        self.audit.update(export_seconds=time.perf_counter() - started,
            fragments=len(self.fragments), export_points=len(xyz), export_instances=len(ids),
            removed_coarse_points=int(removed.sum()), added_fine_points=len(xyz)-len(retained),
            ordinary_instances_unchanged=True, fallback_to_ig=not bool(replacements))
        report = dict(method='IGSubmap-coarse-preserving-fine-export',
            options=dict(vars(self.options)), geometry_policy='hybrid',
            multiresolution=dict(enabled=self.multires_enabled,
                global_voxel_size_m=self.options.voxel_m, fine_voxel_size_m=self.fine_voxel_m,
                selection_rule='final_valid_track_multiview_3d_extent',
                fine_min_views=self.fine_min_views, fine_max_views=self.fine_max_views,
                fine_max_extent_m=self.fine_max_extent_m,
                fine_replace_radius_m=self.fine_replace_radius_m,
                fine_track_ids=sorted(self.fine_track_ids), track_decisions=self.track_decisions),
            export_geometry='original IG coarse surface with exclusive supported fine substitution',
            preserved_fields='instance ordering, ordinary masks, scores, classes, semantic features',
            coarse_source=str(coarse_dir), fragments=self.fragments,
            replacements=replacements, exported_instance_ids=ids, audit=self.audit)
        (output / 'it_manifest.json').write_text(json.dumps(report, indent=2))
        return report
