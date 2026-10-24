"""Read-only, pre-integration diagnostics for historical instance disappearance.

This module never integrates depth, deletes instances, or changes merge decisions.
Candidate IDs identify frozen diagnostic references, not mutable merged-mask rows.
Pure NumPy helpers are independently testable without loading reconstruction models.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

UNKNOWN, SUPPORT, CONTRADICTION, OCCLUDED = 0, 1, 2, 3
SOURCE_NONE, SOURCE_DEPTH, SOURCE_RGB = 0, 1, 2
PALETTE = np.array([[110, 110, 110], [40, 200, 70], [240, 45, 45], [45, 110, 245]], dtype=np.uint8)
RGB_CONTRADICTION_COLOR = np.array([245, 180, 30], dtype=np.uint8)
REQUIRED_SETTINGS = {
    'enabled', 'interval', 'depth_epsilon_m',
    'depth_relative_epsilon', 'min_map_weight', 'min_instance_weight',
    'edge_threshold_m', 'boundary_margin_px', 'min_valid_pixels',
    'contradiction_ratio', 'min_reference_points', 'max_reference_points',
    'min_reference_tested_points', 'min_cumulative_tested_fraction',
    'min_contradicted_fraction', 'confirm_observations',
    'min_observation_gap', 'min_translation_m', 'min_rotation_deg',
    'cancel_support_ratio', 'cancel_observations', 'candidate_cooldown_frames',
    'max_candidates', 'max_global_conflict_fraction', 'dump_interval',
    'max_scene_snapshot_points', 'final_match_radius_m', 'save_frame_arrays',
    'rgb_enabled', 'rgb_mad_multiplier', 'rgb_threshold_min',
    'rgb_threshold_max',
}


def as_numpy(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def checked_settings(settings):
    """Read all detection parameters from the dataset configuration."""
    unknown = set(settings) - REQUIRED_SETTINGS
    if unknown:
        raise ValueError(f'Unknown remove_detection options: {sorted(unknown)}')
    missing = REQUIRED_SETTINGS - set(settings)
    if missing:
        raise ValueError(f'Missing remove_detection options: {sorted(missing)}')
    cfg = dict(settings)
    for key in ('interval', 'min_valid_pixels', 'min_reference_points',
                'min_reference_tested_points', 'max_reference_points',
                'confirm_observations', 'min_observation_gap',
                'cancel_observations', 'max_candidates', 'dump_interval',
                'max_scene_snapshot_points'):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('boundary_margin_px', 'candidate_cooldown_frames'):
        if not isinstance(cfg[key], int) or cfg[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('contradiction_ratio', 'min_cumulative_tested_fraction',
                'min_contradicted_fraction', 'cancel_support_ratio',
                'max_global_conflict_fraction'):
        if not 0 < cfg[key] <= 1:
            raise ValueError(f'{key} must be in (0, 1]')
    for key in ('depth_epsilon_m', 'depth_relative_epsilon', 'min_map_weight',
                'min_instance_weight', 'edge_threshold_m', 'min_translation_m',
                'min_rotation_deg', 'final_match_radius_m', 'rgb_mad_multiplier',
                'rgb_threshold_min', 'rgb_threshold_max'):
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not isinstance(cfg['rgb_enabled'], bool):
        raise ValueError('rgb_enabled must be boolean')
    if cfg['min_map_weight'] <= 0 or cfg['final_match_radius_m'] <= 0:
        raise ValueError('Map confidence and final match radius must be positive')
    if cfg['rgb_threshold_max'] < cfg['rgb_threshold_min']:
        raise ValueError('rgb_threshold_max must be >= rgb_threshold_min')
    if cfg['contradiction_ratio'] + cfg['cancel_support_ratio'] <= 1:
        raise ValueError('Contradiction and cancellation thresholds must not overlap')
    if cfg['max_reference_points'] < cfg['min_reference_points']:
        raise ValueError('max_reference_points must be >= min_reference_points')
    return cfg


def classify_depth(current, historical, valid, epsilon_m, relative_epsilon):
    current, historical = np.asarray(current), np.asarray(historical)
    ok = (np.asarray(valid, dtype=bool) & np.isfinite(current) & np.isfinite(historical)
          & (current > 0) & (historical > 0))
    tolerance = epsilon_m + relative_epsilon * np.maximum(current, historical)
    delta = current - historical
    evidence = np.zeros(current.shape, dtype=np.uint8)
    evidence[ok & (np.abs(delta) <= tolerance)] = SUPPORT
    evidence[ok & (delta > tolerance)] = CONTRADICTION
    evidence[ok & (delta < -tolerance)] = OCCLUDED
    return evidence


def normalize_rgb(image):
    """Return finite float RGB in [0, 1] for either float or uint8 inputs."""
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.shape[-1] != 3:
        raise ValueError(f'Expected RGB with three channels, got {rgb.shape}')
    finite = np.isfinite(rgb)
    scale = 255.0 if finite.any() and np.nanpercentile(np.abs(rgb[finite]), 99) > 1.5 else 1.0
    return np.clip(np.nan_to_num(rgb / scale), 0.0, 1.0)


def rgb_to_lab(rgb):
    import cv2
    value = normalize_rgb(rgb)
    shape = value.shape
    return cv2.cvtColor(value.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(shape)


def build_rgb_context(current_rgb, historical_rgb, eligible, cfg):
    """Estimate a robust per-frame RGB threshold on depth-consistent pixels."""
    import cv2
    current = cv2.GaussianBlur(normalize_rgb(current_rgb), (3, 3), 0)
    historical = cv2.GaussianBlur(normalize_rgb(historical_rgb), (3, 3), 0)
    eligible = (np.asarray(eligible, dtype=bool)
                & np.isfinite(current).all(axis=-1)
                & np.isfinite(historical).all(axis=-1))
    offset = (np.median(historical[eligible] - current[eligible], axis=0)
              if eligible.any() else np.zeros(3, dtype=np.float32))
    current = np.clip(current + offset, 0.0, 1.0)
    current_lab = rgb_to_lab(current)
    historical_lab = rgb_to_lab(historical)
    residual = np.linalg.norm(current_lab - historical_lab, axis=-1)
    values = residual[eligible]
    if len(values):
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        threshold = median + cfg['rgb_mad_multiplier'] * 1.4826 * mad
        threshold = float(np.clip(threshold, cfg['rgb_threshold_min'],
                                  cfg['rgb_threshold_max']))
    else:
        median, mad, threshold = 0.0, 0.0, float(cfg['rgb_threshold_max'])
    contradiction = eligible & (residual > threshold)
    return dict(current_rgb=current, historical_rgb=historical,
                current_lab=current_lab, historical_lab=historical_lab,
                residual=residual, contradiction=contradiction,
                offset=offset, threshold=threshold, median=median, mad=mad,
                eligible_pixels=int(eligible.sum()),
                contradiction_pixels=int(contradiction.sum()))


def evidence_colors(codes, sources):
    colors = PALETTE[np.asarray(codes)].copy()
    colors[(np.asarray(sources) & SOURCE_RGB) != 0] = RGB_CONTRADICTION_COLOR
    return colors


def interior_mask(depth, valid, jump_m, margin):
    """Exclude holes and depth steps without smoothing/altering observations."""
    depth, valid = np.asarray(depth), np.asarray(valid, dtype=bool)
    bad = ~valid.copy()
    vertical = (np.abs(np.diff(depth, axis=0)) > jump_m) | (valid[1:] != valid[:-1])
    horizontal = (np.abs(np.diff(depth, axis=1)) > jump_m) | (valid[:, 1:] != valid[:, :-1])
    bad[1:] |= vertical
    bad[:-1] |= vertical
    bad[:, 1:] |= horizontal
    bad[:, :-1] |= horizontal
    for _ in range(margin):
        padded = np.pad(bad, 1, constant_values=True)
        bad = np.logical_or.reduce([padded[y:y+bad.shape[0], x:x+bad.shape[1]]
                                    for y in range(3) for x in range(3)])
    return valid & ~bad


def reliable_instance_mapping(id_mapping, weights, min_weight):
    """Exclude provisional per-frame slots from diagnostic ownership only."""
    return [int(slot) if 0 <= slot < len(weights) and weights[slot] >= min_weight else -1
            for slot in id_mapping]


def unambiguous_owners(indices, neighbor_valid, ratios, membership_query):
    """One owner per ray/point; ignore duplicate raw views, retain true ambiguity.

    -1 = unassigned; -2 = more than one active instance. No voting by raw-mask
    counts: many observations of one object must not swamp a second owner.
    """
    shape = indices.shape[:-1]
    ids = np.asarray(indices).reshape(-1, indices.shape[-1])
    useful = (np.asarray(neighbor_valid).reshape(ids.shape)
              & (np.asarray(ratios).reshape(ids.shape) > 1e-6) & (ids >= 0))
    keys = np.unique(ids[useful])
    lookup = dict(zip(keys.tolist(), membership_query(keys)))
    owners = np.full(len(ids), -1, dtype=np.int64)
    for row in np.flatnonzero(useful.any(axis=1)):
        members = set()
        for voxel_id in ids[row, useful[row]]:
            members.update(lookup[int(voxel_id)])
            if len(members) > 1:
                break
        if len(members) == 1:
            owners[row] = next(iter(members))
        elif members:
            owners[row] = -2
    return owners.reshape(shape)


def owner_interior(owners, margin):
    good = owners >= 0
    for _ in range(margin):
        padded = np.pad(owners, 1, constant_values=-3)
        previous = np.pad(good, 1, constant_values=False)
        good &= np.logical_and.reduce([
            (padded[y:y+owners.shape[0], x:x+owners.shape[1]] == owners)
            & previous[y:y+owners.shape[0], x:x+owners.shape[1]]
            for y, x in ((0, 1), (2, 1), (1, 0), (1, 2))])
    return good


def evidence_stats(codes, total=None, sources=None):
    counts = np.bincount(np.asarray(codes).reshape(-1), minlength=4)
    tested = int(counts[SUPPORT] + counts[CONTRADICTION])
    total = int(len(np.asarray(codes).reshape(-1)) if total is None else total)
    stats = dict(unknown=int(counts[UNKNOWN]), support=int(counts[SUPPORT]),
                 contradiction=int(counts[CONTRADICTION]), occluded=int(counts[OCCLUDED]),
                 tested=tested, reference_count=total,
                 contradiction_ratio=float(counts[CONTRADICTION] / max(tested, 1)),
                 support_ratio=float(counts[SUPPORT] / max(tested, 1)),
                 tested_fraction=float(tested / max(total, 1)))
    if sources is not None:
        source = np.asarray(sources).reshape(-1)
        stats['depth_contradiction'] = int(((source & SOURCE_DEPTH) != 0).sum())
        stats['rgb_contradiction'] = int(((source & SOURCE_RGB) != 0).sum())
    return stats


def independent_observation(pose, previous, frame, previous_frame, cfg):
    if previous is None:
        return True
    if frame - previous_frame < cfg['min_observation_gap']:
        return False
    translation = np.linalg.norm(pose[:3, 3] - previous[:3, 3])
    cosine = (np.trace(previous[:3, :3].T @ pose[:3, :3]) - 1) / 2
    rotation = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    # A zero threshold disables that motion criterion, not all motion gating.
    gates = []
    if cfg['min_translation_m'] > 0:
        gates.append(translation >= cfg['min_translation_m'])
    if cfg['min_rotation_deg'] > 0:
        gates.append(rotation >= cfg['min_rotation_deg'])
    return not gates or any(gates)


def project_reference(points, colors, intrinsic, pose_c2w, current_depth, history,
                      owners, current_interior, rgb_context, cfg, near, far):
    """Reobserve frozen world points; one front-most reference per image pixel."""
    height, width = current_depth.shape
    camera = (points - pose_c2w[:3, 3]) @ pose_c2w[:3, :3]
    z = camera[:, 2]
    ids = np.flatnonzero(np.isfinite(camera).all(axis=1) & (z > near) & (z < far))
    codes = np.zeros(len(points), dtype=np.uint8)
    sources = np.zeros(len(points), dtype=np.uint8)
    if not len(ids):
        return codes, sources
    projected = camera[ids] @ intrinsic.T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int64)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    ids, uv = ids[inside], uv[inside]
    # A valid camera depth does not imply visibility inside the image.
    if not len(ids):
        return codes, sources
    pixels = uv[:, 1] * width + uv[:, 0]
    order = np.lexsort((ids, z[ids], pixels))
    _, first = np.unique(pixels[order], return_index=True)
    selected = order[first]
    ids, uv = ids[selected], uv[selected]
    y, x = uv[:, 1], uv[:, 0]
    ok = current_interior[y, x] & (owners[y, x] != -2)
    local = classify_depth(current_depth[y, x], z[ids], ok,
                           cfg['depth_epsilon_m'], cfg['depth_relative_epsilon'])
    local_sources = np.zeros(len(ids), dtype=np.uint8)
    local_sources[local == CONTRADICTION] = SOURCE_DEPTH
    if cfg['rgb_enabled'] and np.isfinite(rgb_context['threshold']):
        reference_lab = rgb_to_lab(colors[ids])
        rgb_residual = np.linalg.norm(rgb_context['current_lab'][y, x] - reference_lab, axis=-1)
        rgb_bad = ok & (local == SUPPORT) & (rgb_residual > rgb_context['threshold'])
        local[rgb_bad] = CONTRADICTION
        local_sources[rgb_bad] = SOURCE_RGB
    # Map geometry in front of a reference can self-occlude it. Unknown history
    # does not invalidate the frozen reference: normal fusion may erase a ghost.
    margin = cfg['depth_epsilon_m'] + cfg['depth_relative_epsilon'] * z[ids]
    hidden = history['valid'][y, x] & (history['depth'][y, x] < z[ids] - margin)
    local[hidden & ok] = OCCLUDED
    local_sources[hidden & ok] = SOURCE_NONE
    codes[ids] = local
    sources[ids] = local_sources
    return codes, sources


@dataclass
class Candidate:
    candidate_id: int
    source_raw_ids: frozenset
    source_slot: int
    first_frame: int
    points: np.ndarray
    colors: np.ndarray
    state: str = 'suspected'
    hits: int = 0
    support_hits: int = 0
    last_observation_frame: int = -1
    last_observation_pose: object = None
    last_frame: int = -1
    confirmed_frame: object = None
    cancelled_frame: object = None
    contradicted: np.ndarray = field(init=False)
    tested: np.ndarray = field(init=False)
    depth_contradicted: np.ndarray = field(init=False)
    rgb_contradicted: np.ndarray = field(init=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float32).copy()
        self.colors = normalize_rgb(self.colors).copy()
        if len(self.colors) != len(self.points):
            raise ValueError('Candidate point/color count mismatch')
        self.tested = np.zeros(len(self.points), dtype=bool)
        self.contradicted = np.zeros(len(self.points), dtype=bool)
        self.depth_contradicted = np.zeros(len(self.points), dtype=bool)
        self.rgb_contradicted = np.zeros(len(self.points), dtype=bool)

    def observe(self, frame, pose, codes, sources, cfg):
        stats = evidence_stats(codes, sources=sources)
        stats['cumulative_tested_fraction'] = float(self.tested.mean()) if len(self.points) else 0.
        stats['cumulative_contradicted_fraction'] = float(self.contradicted.mean()) if len(self.points) else 0.
        stats['cumulative_depth_fraction'] = float(self.depth_contradicted.mean()) if len(self.points) else 0.
        stats['cumulative_rgb_fraction'] = float(self.rgb_contradicted.mean()) if len(self.points) else 0.
        self.last_frame = frame
        if self.state != 'suspected':
            return 'terminal', stats
        # Each independent observation needs a minimum number of reference
        # points that can actually be tested. Coverage across different views
        # is accumulated below instead of requiring 12% in one frame.
        if stats['tested'] < cfg['min_reference_tested_points']:
            return 'paused_insufficient_reference_points', stats
        if not independent_observation(pose, self.last_observation_pose, frame,
                                       self.last_observation_frame, cfg):
            return 'paused_correlated_observation', stats
        contradiction = stats['contradiction_ratio'] >= cfg['contradiction_ratio']
        support = stats['support_ratio'] >= cfg['cancel_support_ratio']
        if not contradiction and not support:
            return 'paused_mixed_evidence', stats
        self.last_observation_frame, self.last_observation_pose = frame, pose.copy()
        if support:
            self.support_hits += 1
            # Strong new support invalidates accumulated disappearance evidence.
            self.hits = 0
            self.tested[:] = False
            self.contradicted[:] = False
            self.depth_contradicted[:] = False
            self.rgb_contradicted[:] = False
            stats['cumulative_tested_fraction'] = 0.
            stats['cumulative_contradicted_fraction'] = 0.
            stats['cumulative_depth_fraction'] = 0.
            stats['cumulative_rgb_fraction'] = 0.
            if self.support_hits >= cfg['cancel_observations']:
                self.state, self.cancelled_frame = 'cancelled', frame
                return 'cancelled', stats
            return 'support', stats
        self.support_hits = 0
        self.hits += 1
        current_tested = (np.asarray(codes) == SUPPORT) | (np.asarray(codes) == CONTRADICTION)
        self.tested |= current_tested
        self.contradicted |= np.asarray(codes) == CONTRADICTION
        self.depth_contradicted |= (np.asarray(sources) & SOURCE_DEPTH) != 0
        self.rgb_contradicted |= (np.asarray(sources) & SOURCE_RGB) != 0
        stats['cumulative_tested_fraction'] = float(self.tested.mean())
        stats['cumulative_contradicted_fraction'] = float(self.contradicted.mean())
        stats['cumulative_depth_fraction'] = float(self.depth_contradicted.mean())
        stats['cumulative_rgb_fraction'] = float(self.rgb_contradicted.mean())
        if (self.hits >= cfg['confirm_observations']
                and stats['cumulative_tested_fraction'] >= cfg['min_cumulative_tested_fraction']
                and stats['cumulative_contradicted_fraction'] >= cfg['min_contradicted_fraction']):
            self.state, self.confirmed_frame = 'would_remove', frame
            return 'would_remove', stats
        return 'contradiction', stats

    def metadata(self):
        return dict(candidate_id=self.candidate_id, source_slot=self.source_slot,
                    source_raw_ids=sorted(self.source_raw_ids), first_frame=self.first_frame,
                    last_frame=self.last_frame, state=self.state, confirmations=self.hits,
                    confirmed_frame=self.confirmed_frame, cancelled_frame=self.cancelled_frame,
                    reference_points=len(self.points),
                    tested_fraction=float(self.tested.mean()) if len(self.points) else 0.,
                    contradicted_fraction=float(self.contradicted.mean()) if len(self.points) else 0.,
                    depth_contradicted_fraction=float(self.depth_contradicted.mean()) if len(self.points) else 0.,
                    rgb_contradicted_fraction=float(self.rgb_contradicted.mean()) if len(self.points) else 0.)


def write_ply(path, points, colors):
    """Binary PLY, including valid empty outputs; never changes input arrays."""
    points = np.asarray(points).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3)
    if len(colors) != len(points):
        raise ValueError('Point/color count mismatch')
    if np.issubdtype(colors.dtype, np.floating):
        colors = np.clip(colors * 255, 0, 255)
    vertices = np.empty(len(points), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                           ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    for i, name in enumerate(('x', 'y', 'z')):
        vertices[name] = points[:, i]
    for i, name in enumerate(('red', 'green', 'blue')):
        vertices[name] = colors[:, i]
    with Path(path).open('wb') as stream:
        stream.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\n'
                      'property float x\nproperty float y\nproperty float z\n'
                      'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                      'end_header\n' % len(points)).encode('ascii'))
        vertices.tofile(stream)


def subsample(points, limit):
    if len(points) <= limit:
        return np.arange(len(points))
    return np.linspace(0, len(points) - 1, limit).astype(np.int64)


def final_event_labels(points, candidates, radius):
    """Spatial event-area correspondence, NOT final instance identity."""
    from scipy.spatial import cKDTree
    labels = np.full(len(points), -1, dtype=np.int32)
    best = np.full(len(points), np.inf)
    matches = {}
    final_tree = cKDTree(points) if len(points) else None
    for candidate in candidates:
        if candidate.state != 'would_remove' or not len(candidate.points):
            continue
        if final_tree is None:
            matches[str(candidate.candidate_id)] = dict(matched_reference_points=0,
                                                       unmatched_reference_points=len(candidate.points))
            continue
        distances, _ = cKDTree(candidate.points).query(points, distance_upper_bound=radius)
        choose = (distances < best) & (distances <= radius)
        labels[choose] = candidate.candidate_id
        best[choose] = distances[choose]
        reference_dist, _ = final_tree.query(candidate.points, distance_upper_bound=radius)
        count = int(np.isfinite(reference_dist).sum())
        matches[str(candidate.candidate_id)] = dict(matched_reference_points=count,
                                                   unmatched_reference_points=len(candidate.points)-count)
    return labels, matches


class RemoveDetector:
    """Observation-only sidecar. No Scene_rep or map state is written."""

    def __init__(self, scene, cfg, output_dir):
        self.scene = scene
        self.cfg = checked_settings(cfg['remove_detection'])
        if not self.cfg['enabled']:
            raise ValueError('Construct RemoveDetector only when explicitly enabled')
        self.root = Path(output_dir)
        self.out = self.root / 'remove_detection'
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f'Diagnostics already exist at {self.out}; use a new -o directory')
        (self.out / 'frames').mkdir(parents=True, exist_ok=True)
        (self.out / 'candidates').mkdir(exist_ok=True)
        self.intrinsic = as_numpy(scene.intrinsics).astype(np.float64).copy()
        self.near, self.far = cfg['cam']['depth_near'], cfg['cam']['depth_far']
        self.voxel_size = cfg['scene']['voxel_size']
        self.candidates = []
        self.frames_processed = 0
        self.skips = {}
        self.last_frame = -1
        self.latest_stats = []
        self.total_seconds = 0.
        self._surface_cache = None
        self._diagnostic_mapping = []
        self._json(self.out / 'config.json', dict(
            mode='detection_only', effective_settings=self.cfg, pipeline_config=cfg,
            depth_units='meters, camera z', evidence_palette=PALETTE.tolist(),
            contradiction_source_palette=dict(
                depth=PALETTE[CONTRADICTION].tolist(),
                rgb=RGB_CONTRADICTION_COLOR.tolist()),
            final_match_meaning='spatial historical event area, not final instance identity'))
        self._event(-1, 'started', mode='detection_only')
        self._summary(False)

    @staticmethod
    def _json(path, value):
        temporary = Path(str(path) + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def _event(self, frame, event, **details):
        record = dict(frame=int(frame), event=event, **details)
        with (self.out / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')

    def _summary(self, completed):
        self._json(self.out / 'summary.json', dict(
            completed=completed, last_frame=self.last_frame,
            evaluated_frames=self.frames_processed, skipped_frames=self.skips,
            diagnostic_seconds=self.total_seconds, candidates=len(self.candidates),
            suspected=sum(c.state == 'suspected' for c in self.candidates),
            would_remove=sum(c.state == 'would_remove' for c in self.candidates),
            cancelled=sum(c.state == 'cancelled' for c in self.candidates),
            instances=[c.metadata() for c in self.candidates],
            actual_instance_deletions=0, actual_tsdf_deletions=0))

    def _skip(self, frame, reason, **details):
        self.skips[reason] = self.skips.get(reason, 0) + 1
        self._event(frame, 'frame_skipped', reason=reason, **details)
        self._summary(False)

    def _surface(self):
        if self._surface_cache is None:
            cloud = self.scene.voxel_block_grids.extract_point_cloud(
                weight_threshold=self.cfg['min_map_weight'])
            points = cloud.point.positions.cpu().numpy().copy()
            colors = normalize_rgb(cloud.point.colors.cpu().numpy()).copy()
            indices, found = self.scene.voxel_grids.lookup_voxels_safe(points)
            memberships = self.scene.voxel_hash_table.query_active_instance_memberships(
                indices, self._diagnostic_mapping)
            owners = np.array([next(iter(m)) if len(m) == 1 else -1 for m in memberships], dtype=np.int64)
            owners[~found] = -1
            self._surface_cache = points, owners, colors
        return self._surface_cache

    def _reference(self, slot):
        points, owners, colors = self._surface()
        selected = np.flatnonzero(owners == slot)
        if not len(selected):
            return points[selected], colors[selected]
        reference = points[selected]
        _, first = np.unique(np.floor(reference / self.voxel_size).astype(np.int64),
                             axis=0, return_index=True)
        selected = selected[np.sort(first)]
        selected = selected[subsample(selected, self.cfg['max_reference_points'])]
        return points[selected].copy(), colors[selected].copy()

    def _save_candidate(self, candidate, frame, reason, codes, sources, stats):
        directory = self.out / 'candidates' / f'{candidate.candidate_id:04d}'
        directory.mkdir(exist_ok=True)
        self._json(directory / 'metadata.json', candidate.metadata())
        stem = directory / f'{frame:06d}_{reason}'
        np.savez_compressed(str(stem) + '.npz', reference_points=candidate.points,
                            reference_colors=candidate.colors, evidence=codes,
                            contradiction_source=sources, tested=candidate.tested,
                            contradicted=candidate.contradicted,
                            depth_contradicted=candidate.depth_contradicted,
                            rgb_contradicted=candidate.rgb_contradicted,
                            source_raw_ids=np.array(sorted(candidate.source_raw_ids), dtype=np.int64))
        self._json(Path(str(stem) + '.json'), stats)
        write_ply(str(stem) + '_evidence.ply', candidate.points, evidence_colors(codes, sources))
        scene_points, _, _ = self._surface()
        scene_points = scene_points[subsample(scene_points, self.cfg['max_scene_snapshot_points'])]
        highlight = np.array([230, 30, 220] if candidate.state == 'would_remove' else [255, 155, 30], dtype=np.uint8)
        points = np.concatenate([scene_points, candidate.points])
        colors = np.concatenate([np.full((len(scene_points), 3), 150, dtype=np.uint8),
                                 np.tile(highlight, (len(candidate.points), 1))])
        write_ply(str(stem) + '_scene.ply', points, colors)

    def _blocking_candidates(self, raw_ids, frame):
        blocked = []
        for candidate in self.candidates:
            if not (candidate.source_raw_ids & raw_ids):
                continue
            if (candidate.state != 'cancelled'
                    or frame - candidate.cancelled_frame < self.cfg['candidate_cooldown_frames']):
                blocked.append(dict(candidate_id=candidate.candidate_id,
                                    state=candidate.state,
                                    cancelled_frame=candidate.cancelled_frame))
        return blocked

    def _dump_frame(self, frame, rgb, current, history, owners, evidence,
                    sources, rgb_context, stats):
        from PIL import Image, ImageDraw
        height, width = current.shape
        canvas = Image.new('RGB', (width * 6, height + 65), 'white')
        def depth_rgb(depth):
            normalized = np.clip(np.nan_to_num(depth) / self.far, 0, 1)
            result = np.repeat((normalized[..., None] * 255).astype(np.uint8), 3, axis=2)
            result[~np.isfinite(depth) | (depth <= 0)] = 0
            return result
        threshold = max(float(rgb_context['threshold']), 1e-6)
        residual_level = np.clip(rgb_context['residual'] / threshold, 0, 1)
        residual_rgb = np.repeat((residual_level[..., None] * 255).astype(np.uint8), 3, axis=2)
        residual_rgb[rgb_context['contradiction']] = RGB_CONTRADICTION_COLOR
        panels = [(normalize_rgb(rgb) * 255).astype(np.uint8),
                  (rgb_context['historical_rgb'] * 255).astype(np.uint8),
                  depth_rgb(current), depth_rgb(history['depth']), residual_rgb,
                  evidence_colors(evidence, sources)]
        titles = ['Current RGB', 'Historical RGB BEFORE fusion', 'Current depth (m)',
                  'Historical depth BEFORE fusion',
                  f'RGB residual / yellow > {rgb_context["threshold"]:.2f}',
                  'Green support / Red depth / Yellow RGB / Blue occluded']
        draw = ImageDraw.Draw(canvas)
        for i, (panel, title) in enumerate(zip(panels, titles)):
            canvas.paste(Image.fromarray(panel), (i * width, 25))
            draw.text((i * width + 4, 5), title, fill='black')
        for row in stats:
            slot = row.get('current_slot')
            if slot is None:
                continue
            y, x = np.where(owners == slot)
            if not len(x):
                continue
            color = (230, 30, 220) if row['state'] == 'would_remove' else (255, 155, 30)
            draw.rectangle((int(x.min()), int(y.min())+25, int(x.max()), int(y.max())+25), outline=color, width=2)
            draw.text((int(x.min()), int(y.min())+25),
                      f"C{row['candidate_id']} {row['state']} R={row['contradiction_ratio']:.2f} D={row.get('depth_contradiction', 0)} RGB={row.get('rgb_contradiction', 0)} hits={row['confirmations']}", fill=color)
        draw.text((4, height+32), f'Frame {frame}; grey = unknown. Boxes use current raw-mask lineage; frozen references are in candidates/. No deletion performed.', fill='black')
        path = self.out / 'frames' / f'{frame:06d}'
        canvas.save(str(path) + '.png')
        rgb_stats = dict(threshold=float(rgb_context['threshold']),
                         median=float(rgb_context['median']), mad=float(rgb_context['mad']),
                         eligible_pixels=rgb_context['eligible_pixels'],
                         contradiction_pixels=rgb_context['contradiction_pixels'],
                         color_offset=np.asarray(rgb_context['offset']).tolist())
        self._json(Path(str(path) + '.json'),
                   dict(frame=frame, rgb=rgb_stats, candidates=stats))
        if self.cfg['save_frame_arrays']:
            np.savez_compressed(str(path) + '.npz', current_depth=current,
                                historical_depth=history['depth'], historical_valid=history['valid'],
                                historical_weight=history['weight'],
                                historical_color=rgb_context['historical_rgb'],
                                rgb_residual=rgb_context['residual'],
                                rgb_contradiction=rgb_context['contradiction'],
                                instance_slot=owners, evidence=evidence,
                                contradiction_source=sources)

    def step(self, frame, color, depth, pose_c2w, pose_valid=True, session=None):
        if frame % self.cfg['interval']:
            return
        started = time.perf_counter()
        self.last_frame = frame
        self._surface_cache = None
        try:
            self._step(frame, color, depth, pose_c2w, pose_valid, session)
        finally:
            # Point cloud snapshots are per-observation temporaries, not history.
            self._surface_cache = None
            self.total_seconds += time.perf_counter() - started
            self._summary(False)

    def _step(self, frame, color, depth, pose_c2w, pose_valid, session):
        pose = as_numpy(pose_c2w).astype(np.float64)
        if (not pose_valid or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=.02)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=.02)):
            self._skip(frame, 'invalid_original_pose')
            return
        current = as_numpy(depth).astype(np.float32)
        current_valid = np.isfinite(current) & (current > self.near) & (current < self.far)
        if current_valid.sum() < self.cfg['min_valid_pixels']:
            self._skip(frame, 'insufficient_current_depth')
            return
        try:
            history = self.scene.voxel_grids.raycast_history(
                self.intrinsic, np.linalg.inv(pose), current.shape[1], current.shape[0],
                self.near, self.far, self.cfg['min_map_weight'], self.scene.trunc_voxel_multiplier)
        except RuntimeError as error:
            self._skip(frame, 'raycast_error', error=str(error))
            print(f'[REMOVE-DETECT] frame={frame} raycast failed; diagnostics skipped (see events.jsonl).')
            return
        if history is None:
            self._skip(frame, 'empty_history')
            return
        if history['valid'].sum() < self.cfg['min_valid_pixels']:
            self._skip(frame, 'insufficient_historical_render')
            return
        mapping = np.asarray(self.scene.global2merged_mask_id, dtype=np.int64)
        weights = as_numpy(self.scene.get_merged_mask_weight)
        self._diagnostic_mapping = reliable_instance_mapping(
            mapping, weights, self.cfg['min_instance_weight'])
        query = lambda ids: self.scene.voxel_hash_table.query_active_instance_memberships(
            ids, self._diagnostic_mapping)
        owners = unambiguous_owners(history['index'], history['neighbor_valid'],
                                    history['interp_ratio'], query)
        owners[~history['valid']] = -1
        current_interior = interior_mask(current, current_valid, self.cfg['edge_threshold_m'], self.cfg['boundary_margin_px'])
        historical_interior = interior_mask(history['depth'], history['valid'], self.cfg['edge_threshold_m'], self.cfg['boundary_margin_px'])
        reliable = current_interior & historical_interior
        depth_evidence = classify_depth(current, history['depth'], reliable,
                                        self.cfg['depth_epsilon_m'], self.cfg['depth_relative_epsilon'])
        rgb_eligible = ((depth_evidence == SUPPORT) if self.cfg['rgb_enabled']
                        else np.zeros_like(reliable))
        rgb_context = build_rgb_context(as_numpy(color), history['color'], rgb_eligible, self.cfg)
        evidence = depth_evidence.copy()
        sources = np.zeros(evidence.shape, dtype=np.uint8)
        sources[evidence == CONTRADICTION] = SOURCE_DEPTH
        rgb_bad = rgb_context['contradiction'] & (depth_evidence == SUPPORT)
        evidence[rgb_bad] = CONTRADICTION
        sources[rgb_bad] = SOURCE_RGB
        global_stats = evidence_stats(evidence, sources=sources)
        owner_good = owner_interior(owners, self.cfg['boundary_margin_px'])
        excluded = (evidence == CONTRADICTION) & ~owner_good
        excluded_stats = dict(
            unassigned=int((excluded & (owners == -1)).sum()),
            ambiguous=int((excluded & (owners == -2)).sum()),
            instance_boundary=int((excluded & (owners >= 0)).sum()))
        evidence[~owner_good] = UNKNOWN
        sources[~owner_good] = SOURCE_NONE
        if (global_stats['tested'] >= self.cfg['min_valid_pixels']
                and global_stats['contradiction_ratio'] > self.cfg['max_global_conflict_fraction']):
            self._dump_frame(frame, as_numpy(color), current, history, owners, evidence,
                             sources, rgb_context, [])
            self._skip(frame, 'widespread_conflict', stats=global_stats)
            return
        self.frames_processed += 1
        changed = False
        frame_stats = []
        # Existing candidates are evaluated against their frozen references even
        # if normal merging has changed the current instance's size/slot.
        for candidate in self.candidates:
            if candidate.state != 'suspected':
                continue
            codes, candidate_sources = project_reference(
                candidate.points, candidate.colors, self.intrinsic, pose, current,
                history, owners, current_interior, rgb_context, self.cfg, self.near, self.far)
            reason, stats = candidate.observe(frame, pose, codes, candidate_sources, self.cfg)
            descendants = sorted({int(mapping[r]) for r in candidate.source_raw_ids
                                  if r < len(mapping) and mapping[r] >= 0})
            row = dict(stats, candidate_id=candidate.candidate_id, state=candidate.state,
                       confirmations=candidate.hits, current_slot=descendants[0] if len(descendants) == 1 else None,
                       descendant_slots=descendants, reason=reason)
            frame_stats.append(row)
            self._event(frame, 'candidate_observation', session=session, **row)
            if reason in ('would_remove', 'cancelled'):
                changed = True
                self._save_candidate(candidate, frame, reason, codes, candidate_sources, stats)
                print(f'[REMOVE-DETECT] frame={frame} candidate={candidate.candidate_id} {reason} (diagnostic only)')
        for slot in np.unique(owners[evidence == CONTRADICTION]):
            slot = int(slot)
            pixel_stats = evidence_stats(evidence[owners == slot], sources=sources[owners == slot])
            if slot < 0 or slot >= len(weights) or weights[slot] < self.cfg['min_instance_weight']:
                self._event(frame, 'candidate_rejected', slot=slot, stage='instance',
                            reason='unreliable_instance', pixel_stats=pixel_stats)
                continue
            pixel_failures = []
            if pixel_stats['tested'] < self.cfg['min_valid_pixels']:
                pixel_failures.append('insufficient_pixel_count')
            if pixel_stats['contradiction_ratio'] < self.cfg['contradiction_ratio']:
                pixel_failures.append('low_pixel_contradiction_ratio')
            if pixel_failures:
                self._event(frame, 'candidate_rejected', slot=slot, stage='pixels',
                            reason=pixel_failures[0], failed_conditions=pixel_failures,
                            pixel_stats=pixel_stats)
                continue
            raw_ids = frozenset(np.flatnonzero(mapping == slot).tolist())
            if not raw_ids:
                self._event(frame, 'candidate_rejected', slot=slot, stage='association',
                            reason='no_source_raw_ids', pixel_stats=pixel_stats)
                continue
            blocking = self._blocking_candidates(raw_ids, frame)
            if blocking:
                self._event(frame, 'candidate_rejected', slot=slot, stage='association',
                            reason='existing_candidate_or_cooldown', pixel_stats=pixel_stats,
                            blocking_candidates=blocking)
                continue
            if len(self.candidates) >= self.cfg['max_candidates']:
                self._event(frame, 'candidate_limit_reached', slot=slot,
                            pixel_stats=pixel_stats)
                break
            reference, reference_colors = self._reference(slot)
            if len(reference) < self.cfg['min_reference_points']:
                self._event(frame, 'candidate_rejected', slot=slot, stage='reference',
                            reason='insufficient_reference_surface', points=len(reference),
                            pixel_stats=pixel_stats)
                continue
            codes, candidate_sources = project_reference(
                reference, reference_colors, self.intrinsic, pose, current, history,
                owners, current_interior, rgb_context, self.cfg, self.near, self.far)
            ref_stats = evidence_stats(codes, sources=candidate_sources)
            reference_failures = []
            if ref_stats['tested'] < self.cfg['min_reference_tested_points']:
                reference_failures.append('insufficient_reference_tested_points')
            if ref_stats['contradiction_ratio'] < self.cfg['contradiction_ratio']:
                reference_failures.append('low_reference_contradiction_ratio')
            if reference_failures:
                self._event(frame, 'candidate_rejected', slot=slot, stage='reference',
                            reason='insufficient_reference_evidence',
                            failed_conditions=reference_failures,
                            pixel_stats=pixel_stats, stats=ref_stats)
                continue
            candidate = Candidate(len(self.candidates), raw_ids, slot, frame,
                                  reference, reference_colors)
            self.candidates.append(candidate)
            reason, stats = candidate.observe(frame, pose, codes, candidate_sources, self.cfg)
            row = dict(stats, candidate_id=candidate.candidate_id, state=candidate.state,
                       confirmations=candidate.hits, current_slot=slot, reason='created')
            frame_stats.append(row)
            self._event(frame, 'candidate_created', session=session, **row,
                        pixel_stats=pixel_stats, source_raw_ids=sorted(raw_ids))
            self._save_candidate(candidate, frame, 'created', codes, candidate_sources, stats)
            if reason == 'would_remove':
                self._event(frame, 'candidate_observation', session=session, **dict(row, reason=reason))
                self._save_candidate(candidate, frame, reason, codes, candidate_sources, stats)
            changed = True
        self.latest_stats = frame_stats
        rgb_stats = dict(threshold=float(rgb_context['threshold']),
                         median=float(rgb_context['median']), mad=float(rgb_context['mad']),
                         eligible_pixels=rgb_context['eligible_pixels'],
                         contradiction_pixels=rgb_context['contradiction_pixels'])
        self._event(frame, 'frame_evaluated', session=session, pixel_stats=global_stats,
                    rgb_stats=rgb_stats, ambiguous_pixels=int((owners == -2).sum()),
                    excluded_contradiction_pixels=excluded_stats,
                    candidate_observations=len(frame_stats))
        if changed or frame % self.cfg['dump_interval'] == 0:
            self._dump_frame(frame, as_numpy(color), current, history, owners, evidence,
                             sources, rgb_context, frame_stats)

    def finalize(self, points, colors):
        """Write separate outputs. Original final.ply and scene arrays are untouched."""
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors).reshape(-1, 3).copy()
        labels, matches = final_event_labels(points, self.candidates, self.cfg['final_match_radius_m'])
        colors = np.clip(colors * 255, 0, 255).astype(np.uint8)
        colors[labels >= 0] = [230, 30, 220]
        write_ply(self.root / 'final_remove_detection.ply', points, colors)
        confirmed = [c for c in self.candidates if c.state == 'would_remove']
        reference = np.concatenate([c.points for c in confirmed]) if confirmed else np.empty((0, 3), np.float32)
        ref_ids = np.concatenate([np.full(len(c.points), c.candidate_id, np.int32) for c in confirmed]) if confirmed else np.empty(0, np.int32)
        # Event colors are reproducible and have a manifest, distinct from evidence colors.
        import colorsys
        event_colors = {c.candidate_id: np.array(colorsys.hsv_to_rgb((c.candidate_id * .61803398875) % 1, .7, 1)) for c in confirmed}
        ref_colors = np.concatenate([np.tile(event_colors[c.candidate_id], (len(c.points), 1)) for c in confirmed]) if confirmed else np.empty((0, 3))
        write_ply(self.root / 'remove_reference_instances.ply', reference, ref_colors)
        np.savez_compressed(self.out / 'final_event_labels.npz', candidate_id=labels,
                            reference_points=reference, reference_candidate_id=ref_ids)
        self._json(self.out / 'final_matches.json', dict(
            radius_m=self.cfg['final_match_radius_m'], matches=matches,
            candidate_colors={str(k): v.tolist() for k, v in event_colors.items()},
            meaning='Spatial correspondence to historical references, not final instance IDs. Nearby new surfaces can also match.',
            final_points=len(points), marked_points=int((labels >= 0).sum())))
        for candidate in self.candidates:
            self._json(self.out / 'candidates' / f'{candidate.candidate_id:04d}' / 'metadata.json', candidate.metadata())
        self._event(self.last_frame, 'finished', would_remove=len(confirmed), actual_deletions=0)
        self._summary(True)
        print(f'[REMOVE-DETECT] evaluated_frames={self.frames_processed} '
              f'would_remove={len(confirmed)} actual_deletions=0; diagnostics: {self.out}')
        if self.frames_processed == 0:
            print('[REMOVE-DETECT] No frame was successfully evaluated. Check skipped_frames in summary.json.')
