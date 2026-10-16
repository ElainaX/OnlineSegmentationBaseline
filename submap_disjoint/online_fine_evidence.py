"""Causal fine evidence built after coarse identity updates, used next frame.

Observation IDs never act as persistent track IDs. Every lookup resolves the
owner's current mapping, and changed track membership invalidates cached fine
surfaces until the next completed observation update.
"""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import open3d.core as o3c
from scipy.spatial import cKDTree
from .independent_tsdf import LocalTSDF, TSDFOptions
from .multires_policy import assess_track, project_mask_points
from .it_adapter_multires import reserve_window


class OnlineFineEvidence:
    def __init__(self, cfg, owner, directory, enabled=True):
        self.cfg, self.owner, self.enabled = cfg, owner, bool(enabled)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        mr = cfg['multi_resolution']
        self.voxel_m = float(mr['fine_voxel_size_m'])
        self.min_views = int(mr.get('fine_min_views', 3))
        self.max_views = int(mr.get('fine_max_views', 5))
        self.max_extent_m = float(mr.get('fine_max_extent_m', .45))
        self.max_relative_spread = float(mr.get('fine_max_relative_spread', .50))
        self.K = owner.intrinsics.detach().cpu().numpy().astype(np.float64)
        self.options = TSDFOptions(
            voxel_m=self.voxel_m,
            block_resolution=int(cfg['scene']['block_resolution']),
            trunc_voxels=float(mr['physical_truncation_m']) / self.voxel_m,
            near_m=float(cfg['cam']['depth_near']), far_m=float(cfg['cam']['depth_far']),
            max_blocks=int(mr.get('online_fine_block_count', 4096)),
            device=owner.device.upper(),
            extraction_weight=float(cfg['scene'].get('extraction_weight', .5)))
        self.cache = {}
        self._decision_cache = {}
        self.packet = None
        self._previous_adjust = getattr(owner, 'adjust_candidate_scores', None)
        self.audit = dict(enabled=self.enabled, updates=0, fine_builds=0,
                          score_queries=0, queries_with_evidence=0,
                          fine_build_seconds=0., skipped_block_budget=0,
                          refresh_seconds=0., score_adjust_seconds=0.,
                          resolution_cache_hits=0, resolution_evaluations=0,
                          materialized_masks=0, score_trees_built=0,
                          depth_evidence_evaluations=0,
                          last_decisions={})
        if self.enabled:
            owner.adjust_candidate_scores = self.adjust_candidate_scores

    def _mapping(self):
        return [int(x) for x in self.owner.global2merged_mask_id]

    def refresh(self, packets):
        if not self.enabled or not packets:
            return
        refresh_started = time.perf_counter()
        self.packet = packets[-1]
        mapping = self._mapping()
        groups = defaultdict(dict)
        members = defaultdict(list)
        for gid, mid in enumerate(mapping):
            if mid >= 0:
                members[mid].append(gid)
        # Only a bounded recent window of masks per track is replayed online.
        # Packet arrays are shared with the export adapter, not copied again.
        for packet in reversed(packets):
            raw_by_mid = defaultdict(list)
            for raw, gid in packet['pairs']:
                if gid < len(mapping) and mapping[gid] >= 0:
                    raw_by_mid[mapping[gid]].append(raw)
            for mid, raws in raw_by_mid.items():
                if len(groups[mid]) < self.max_views:
                    # Keep only references/IDs until a changed track actually
                    # needs geometry. Cached tracks never scan image pixels.
                    groups[mid][packet['frame']] = (packet, tuple(raws))
        new_cache = {}
        new_decision_cache = {}
        decisions = {}
        for mid, by_frame in groups.items():
            signature = tuple(members[mid])
            cached = self.cache.get(signature)
            if cached is not None:
                new_cache[signature] = cached
                continue
            recent = [by_frame[f] for f in sorted(by_frame)]
            # Packets are immutable and append-only. A changed membership or
            # observation window necessarily gets a different key, including
            # tracks that were previously coarse/unstable/too short.
            key = (signature, tuple((p['frame'], raws) for p, raws in recent))
            decision = self._decision_cache.get(key)
            views = None
            def materialize():
                self.audit['materialized_masks'] += len(recent)
                return [(p, p['labels'] == raws[0] if len(raws) == 1
                         else np.isin(p['labels'], raws)) for p, raws in recent]
            if decision is None:
                views = materialize()
                decision = assess_track(views, self.K, self.options.near_m,
                                        self.options.far_m, self.min_views,
                                        self.max_views, self.max_extent_m,
                                        self.max_relative_spread)
                self.audit['resolution_evaluations'] += 1
            else:
                self.audit['resolution_cache_hits'] += 1
            new_decision_cache[key] = decision
            decisions[mid] = decision
            if decision['resolution'] != 'fine':
                continue
            if views is None:
                views = materialize()
            started = time.perf_counter()
            torch.cuda.synchronize()
            o3c.cuda.synchronize()
            volume = LocalTSDF(self.options)
            try:
                # Reserve before any integration: growing stdgpu tables after
                # accumulating frames can invalidate in-flight kernel storage.
                reserve_window(volume, views, self.K)
                for packet, mask in views:
                    volume.integrate(packet['rgb'], packet['depth'], self.K,
                                     packet['pose'], mask)
                o3c.cuda.synchronize()
                points, _ = volume.surface()
            except MemoryError:
                self.audit['skipped_block_budget'] += 1
                continue
            finally:
                o3c.cuda.synchronize()
                del volume
                self.audit['fine_build_seconds'] += time.perf_counter() - started
            if len(points) < 25:
                continue
            if len(points) > 1500:
                points = points[np.linspace(0, len(points)-1, 1500).astype(int)]
            new_cache[signature] = points
            self.audit['fine_builds'] += 1
        self.cache = new_cache
        self._decision_cache = new_decision_cache
        self.audit['updates'] += 1
        self.audit['last_decisions'] = decisions
        self.audit['refresh_seconds'] += time.perf_counter() - refresh_started

    def _depth_evidence(self, points):
        """Visible support vs free space; occluded points provide no evidence."""
        p = self.packet
        pose = p['pose']
        cam = (points - pose[:3, 3]) @ pose[:3, :3]
        cam = cam[np.isfinite(cam).all(1) & (cam[:, 2] > self.options.near_m)]
        if not len(cam):
            return 0., 0.
        uv = cam[:, :2] / cam[:, 2:3] @ self.K[:2, :2].T + self.K[:2, 2]
        u, v = np.rint(uv).astype(int).T
        h, w = p['depth'].shape
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v, z = u[inside], v[inside], cam[inside, 2]
        d = p['depth'][v, u]
        valid = np.isfinite(d) & (d > self.options.near_m) & (d < self.options.far_m)
        tol = max(.03, 2 * self.voxel_m)
        visible = valid & (z <= d + tol)
        if not visible.any():
            return 0., 0.
        # Count projected pixels once, not once per fine point.
        pixels = v[visible] * w + u[visible]
        _, first = np.unique(pixels, return_index=True)
        dz = (d[visible] - z[visible])[first]
        return float(np.mean(np.abs(dz) <= tol)), float(np.mean(dz > tol))

    def adjust_candidate_scores(self, values, ij, ab, ba):
        if self._previous_adjust is not None:
            values = self._previous_adjust(values, ij, ab, ba)
        if not self.enabled or not self.cache or self.packet is None:
            return values
        adjust_started = time.perf_counter()
        mapping = self._mapping()
        current_members = defaultdict(list)
        for gid, mid in enumerate(mapping):
            if mid >= 0:
                current_members[mid].append(gid)
        fine = {mid: self.cache[tuple(gids)] for mid, gids in current_members.items()
                if tuple(gids) in self.cache}
        result = values.clone()
        geometry_cache, trees, evidence_cache = {}, {}, {}
        def geometry(mid):
            if mid not in geometry_cache:
                geometry_cache[mid] = (fine[mid] if mid in fine else
                    self.owner.mask_voxels_coords[mid].detach().cpu().numpy().astype(np.float64))
            return geometry_cache[mid]
        def tree(mid):
            if mid not in trees:
                trees[mid] = cKDTree(geometry(mid))
                self.audit['score_trees_built'] += 1
            return trees[mid]
        def evidence_for(mid):
            if mid not in evidence_cache:
                evidence_cache[mid] = self._depth_evidence(fine[mid])
                self.audit['depth_evidence_evaluations'] += 1
            return evidence_cache[mid]
        for k, (a, b) in enumerate(ij.detach().cpu().tolist()):
            self.audit['score_queries'] += 1
            if a not in fine and b not in fine:
                continue
            pa, pb = geometry(a), geometry(b)
            if not len(pa) or not len(pb):
                continue
            radius = max(.03, 2 * self.voxel_m)
            overlap = .5 * (np.mean(tree(a).query(pb)[0] <= radius) +
                            np.mean(tree(b).query(pa)[0] <= radius))
            evidence = [evidence_for(x) for x in (a, b) if x in fine]
            support = float(np.mean([x[0] for x in evidence]))
            contradiction = float(np.mean([x[1] for x in evidence]))
            result[k] += float(.20 * overlap + .10 * support - .15 * contradiction)
            self.audit['queries_with_evidence'] += 1
        self.audit['score_adjust_seconds'] += time.perf_counter() - adjust_started
        return result

    def save(self):
        result = dict(self.audit, selection_rule='current_track_multiview_3d',
                      cached_fine_tracks=len(self.cache),
                      min_views=self.min_views, max_extent_m=self.max_extent_m,
                      timing='past observations only; refreshed after coarse merge')
        (self.directory/'online_fine_evidence.json').write_text(json.dumps(result, indent=2))
        return result
