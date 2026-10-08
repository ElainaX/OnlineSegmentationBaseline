"""Experimental read-only spatial disappearance evidence, no GT in decisions.

Sparse world cells own frozen surface samples and independent-view signed votes.
Instance ownership is diagnostic metadata, never a prerequisite for evidence.
"""
import json
import time
from pathlib import Path
import numpy as np

from .readonly_detection_core import (
    as_numpy, classify_depth, interior_mask, SUPPORT, CONTRADICTION,
)


class EvidenceBank:
    """One vote per spatial cell per sufficiently separated camera observation.

Scores are bounded engineering evidence counts, NOT calibrated probabilities.
Unknown/occluded observations do nothing. Support only subtracts locally.
"""
    def __init__(self, voxel_size, cfg, capacity=200000):
        self.size = float(voxel_size)
        self.cfg = cfg
        self.capacity = capacity
        self.n = 0
        self.lookup = {}
        self.points = np.empty((capacity, 3), np.float64)
        self.keys = np.empty((capacity, 3), np.int64)
        self.score = np.zeros(capacity, np.int8)
        self.contradictions = np.zeros(capacity, np.int32)
        self.supports = np.zeros(capacity, np.int32)
        self.last_frame = np.full(capacity, -1000000, np.int64)
        self.last_pose = np.zeros((capacity, 4, 4), np.float64)
        self.first_frame = np.full(capacity, -1, np.int64)
        self.first_confirmed = np.full(capacity, -1, np.int64)
        self.last_confirmed = np.full(capacity, -1, np.int64)
        self.confirmed = np.zeros(capacity, bool)
        self.first_owner = np.full(capacity, -1, np.int64)
        self.first_owner_weight = np.zeros(capacity, np.float32)
        self.dropped = 0
        self.transitions = dict(confirmed=0, revoked=0)

    def seed(self, points, frame, owners=None, weights=None):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        keys = np.floor(points / self.size).astype(np.int64)
        _, first = np.unique(keys, axis=0, return_index=True)
        for j in first:
            key = tuple(keys[j])
            if key in self.lookup:
                continue
            if self.n == self.capacity:
                self.dropped += 1
                continue
            i = self.n
            self.n += 1
            self.lookup[key] = i
            self.points[i], self.keys[i], self.first_frame[i] = points[j], keys[j], frame
            if owners is not None:
                self.first_owner[i] = owners[j]
            if weights is not None:
                self.first_owner_weight[i] = weights[j]

    def observe(self, frame, pose, codes):
        codes = np.asarray(codes)
        assert len(codes) == self.n
        ids = np.flatnonzero((codes == SUPPORT) | (codes == CONTRADICTION))
        if not len(ids):
            return dict(accepted=0, confirmed=0, revoked=0)
        fresh = self.last_frame[ids] < 0
        gap = frame - self.last_frame[ids] >= self.cfg['min_observation_gap']
        trans = np.linalg.norm(self.last_pose[ids, :3, 3] - pose[:3, 3], axis=1)
        trace = np.einsum('nij,ij->n', self.last_pose[ids, :3, :3], pose[:3, :3])
        angle = np.rad2deg(np.arccos(np.clip((trace - 1) / 2, -1, 1)))
        movement = np.zeros(len(ids), bool)
        if self.cfg['min_translation_m'] > 0:
            movement |= trans >= self.cfg['min_translation_m']
        if self.cfg['min_rotation_deg'] > 0:
            movement |= angle >= self.cfg['min_rotation_deg']
        if self.cfg['min_translation_m'] <= 0 and self.cfg['min_rotation_deg'] <= 0:
            movement[:] = True
        ids = ids[fresh | (gap & movement)]
        bad = codes[ids] == CONTRADICTION
        self.last_frame[ids] = frame
        self.last_pose[ids] = pose
        self.contradictions[ids[bad]] += 1
        self.supports[ids[~bad]] += 1
        self.score[ids] = np.clip(self.score[ids].astype(int) + np.where(bad, 1, -1), -3, 3)
        was = self.confirmed[ids].copy()
        # Hysteresis: enter at +2; require counterevidence to reach 0 to revoke.
        self.confirmed[ids[self.score[ids] >= 2]] = True
        self.confirmed[ids[self.score[ids] <= 0]] = False
        new = ids[~was & self.confirmed[ids]]
        revoked = ids[was & ~self.confirmed[ids]]
        first = new[self.first_confirmed[new] < 0]
        self.first_confirmed[first] = frame
        self.last_confirmed[new] = frame
        self.transitions['confirmed'] += len(new)
        self.transitions['revoked'] += len(revoked)
        return dict(accepted=len(ids), confirmed=len(new), revoked=len(revoked))

    def arrays(self):
        names = ('points', 'keys', 'score', 'contradictions', 'supports',
                 'first_frame', 'first_confirmed', 'last_confirmed', 'confirmed', 'first_owner',
                 'first_owner_weight', 'last_frame')
        return {k: getattr(self, k)[:self.n].copy() for k in names}


def project_codes(points, intrinsic, pose, depth, valid_interior, cfg, near, far):
    """Current measured free space can contradict ALL old samples along a ray.

Do not let potentially stale TSDF geometry veto measured free-space evidence.
Invalid depth/occlusion remain unknown/non-votes; no color disappearance votes.
"""
    codes = np.zeros(len(points), np.uint8)
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    ids = np.flatnonzero(np.isfinite(camera).all(axis=1) & (z > near) & (z < far))
    if not len(ids):
        return codes
    uv = camera[ids] @ intrinsic.T
    uv = np.rint(uv[:, :2] / uv[:, 2:3]).astype(int)
    h, w = depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids, uv = ids[inside], uv[inside]
    x, y = uv.T
    codes[ids] = classify_depth(depth[y, x], z[ids], valid_interior[y, x],
        cfg['depth_epsilon_m'], cfg['depth_relative_epsilon'])
    return codes


def connected_regions(keys, selected):
    """26-connected world-cell regions. Region is a change area, NOT an instance."""
    lookup = {tuple(keys[i]): int(i) for i in selected}
    offsets = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1)
               for z in (-1, 0, 1) if x or y or z]
    regions = []
    while lookup:
        key, i = lookup.popitem()
        stack = [key]
        region = [i]
        while stack:
            k = stack.pop()
            for offset in offsets:
                neighbor = tuple(k[a] + offset[a] for a in range(3))
                j = lookup.pop(neighbor, None)
                if j is not None:
                    stack.append(neighbor)
                    region.append(j)
        regions.append(np.asarray(region, dtype=np.int64))
    return regions


class LocalEvidenceDetector:
    def __init__(self, scene, cfg, output_dir):
        self.scene = scene
        self.cfg = dict(cfg['remove_detection'])
        self.intrinsic = as_numpy(scene.intrinsics).astype(float)
        self.near, self.far = cfg['cam']['depth_near'], cfg['cam']['depth_far']
        self.bank = EvidenceBank(cfg['scene']['voxel_size'], self.cfg)
        self.out = Path(output_dir) / 'local_evidence'
        self.out.mkdir(exist_ok=False)
        self.frames = 0
        self.seconds = 0.
        self.skips = {}
        self.boundary_saved = False
        self.previous_session = None
        (self.out / 'config.json').write_text(json.dumps(dict(
            original_depth_settings=self.cfg, voxel_size_m=self.bank.size,
            score_range=[-3, 3], confirm_score=2, revoke_score=0,
            max_cells=self.bank.capacity, region_min_cells=[20, 100],
            mode='read_only', rgb_votes=False, instance_gates=False,
            history_occlusion_veto=False, GT_in_decisions=False), indent=2))

    def skip(self, frame, reason):
        self.skips[reason] = self.skips.get(reason, 0) + 1
        self.event(dict(frame=int(frame), skip=reason))

    def event(self, data):
        with (self.out / 'events.jsonl').open('a') as f:
            f.write(json.dumps(data) + '\n')

    def step(self, frame, color, depth, pose, pose_valid=True, session=None):
        start = time.perf_counter()
        try:
            self._step(frame, depth, pose, pose_valid, session)
        finally:
            self.seconds += time.perf_counter() - start

    def _step(self, frame, depth, pose, pose_valid, session):
        if self.previous_session == 'reference' and session == 'rescan':
            self.snapshot('reference_end')
            self.boundary_saved = True
        self.previous_session = session
        pose = as_numpy(pose).astype(float)
        if (not pose_valid or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=.02)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=.02)):
            return self.skip(frame, 'invalid_pose')
        depth = as_numpy(depth).astype(np.float32)
        valid = np.isfinite(depth) & (depth > self.near) & (depth < self.far)
        if valid.sum() < self.cfg['min_valid_pixels']:
            return self.skip(frame, 'insufficient_depth')
        interior = interior_mask(depth, valid, self.cfg['edge_threshold_m'], self.cfg['boundary_margin_px'])
        history = self.scene.voxel_grids.raycast_history(self.intrinsic,
            np.linalg.inv(pose), depth.shape[1], depth.shape[0], self.near, self.far,
            self.cfg['min_map_weight'], self.scene.trunc_voxel_multiplier)
        seeded = 0
        if history is not None and history['valid'].sum() >= self.cfg['min_valid_pixels']:
            old_interior = interior_mask(history['depth'], history['valid'],
                self.cfg['edge_threshold_m'], self.cfg['boundary_margin_px'])
            codes = classify_depth(depth, history['depth'], interior & old_interior,
                self.cfg['depth_epsilon_m'], self.cfg['depth_relative_epsilon'])
            tested = (codes == SUPPORT) | (codes == CONTRADICTION)
            if tested.sum() >= self.cfg['min_valid_pixels'] and (codes == CONTRADICTION).sum() / tested.sum() > self.cfg['max_global_conflict_fraction']:
                return self.skip(frame, 'widespread_depth_conflict')
            y, x = np.where(codes == CONTRADICTION)
            if len(x):
                cam = np.column_stack((x, y, np.ones(len(x)))) @ np.linalg.inv(self.intrinsic).T
                cam *= history['depth'][y, x, None]
                world = cam @ pose[:3, :3].T + pose[:3, 3]
                # Diagnostic owner attribution only; unknown/mixed cells are kept.
                indices, found = self.scene.voxel_grids.lookup_voxels_safe(world)
                members = self.scene.voxel_hash_table.query_active_instance_memberships(
                    indices, self.scene.global2merged_mask_id)
                owners = np.array([next(iter(m)) if len(m) == 1 else -2 if m else -1 for m in members])
                owners[~found] = -1
                w = as_numpy(self.scene.get_merged_mask_weight)
                weights = np.zeros(len(owners))
                known = (owners >= 0) & (owners < len(w))
                weights[known] = w[owners[known]]
                before = self.bank.n
                self.bank.seed(world, frame, owners, weights)
                seeded = self.bank.n - before
        # Frozen references are still tested when normal TSDF fusion erased them.
        codes = project_codes(self.bank.points[:self.bank.n], self.intrinsic,
            pose, depth, interior, self.cfg, self.near, self.far)
        votes = self.bank.observe(frame, pose, codes)
        self.frames += 1
        self.event(dict(frame=int(frame), session=session, seeded=seeded, cells=self.bank.n,
            active_confirmed=int(self.bank.confirmed[:self.bank.n].sum()),
            support=int((codes == SUPPORT).sum()), contradiction=int((codes == CONTRADICTION).sum()),
            **votes))

    def snapshot(self, name):
        arrays = self.bank.arrays()
        np.savez_compressed(self.out / (name + '.npz'), **arrays)
        regions = connected_regions(arrays['keys'], np.flatnonzero(arrays['confirmed']))
        rows = [dict(region_id=i, cells=len(ids), indices=ids.tolist())
                for i, ids in enumerate(regions) if len(ids) >= 20]
        (self.out / (name + '_regions.json')).write_text(json.dumps(rows))
        return dict(confirmed_cells=int(arrays['confirmed'].sum()),
                    regions_ge20=len(rows), regions_ge100=sum(r['cells'] >= 100 for r in rows))

    def finalize(self):
        regions = self.snapshot('final')
        summary = dict(completed=True, evaluated_frames=self.frames, seconds=self.seconds,
            cells=self.bank.n, capacity_rejected_new_cells=self.bank.dropped,
            transitions=self.bank.transitions, skipped_frames=self.skips,
            memory_array_bytes=sum(v.nbytes for v in vars(self.bank).values() if isinstance(v, np.ndarray)),
            reference_boundary_snapshot=self.boundary_saved, actual_deletions=0, **regions)
        (self.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary
