"""Causal, multi-view policy for selecting instance-local TSDF resolution.

The policy deliberately uses metric 3-D evidence instead of a single-view
image area.  A track remains coarse until enough independent observations have
arrived; this prevents one bad first mask from allocating or publishing a fine
surface.
"""
from dataclasses import dataclass, asdict
import numpy as np


def assess_track(views, K, near_m, far_m, min_views=3, max_views=5,
                 max_extent_m=0.45, max_relative_spread=0.50):
    """Recompute after identity merges; IDs label observations, not tracks.

    views contains (packet, mask), grouped by the owner's current identity.
    No frozen vote survives a later merge into a geometrically larger object.
    """
    clouds = []
    frames = []
    for packet, mask in sorted(views, key=lambda x: x[0]['frame']):
        if int(packet['frame']) in frames:
            continue
        points = project_mask_points(packet['depth'], packet['pose'], K, mask,
                                     near_m, far_m, max_points=1200)
        if len(points) < 25:
            continue
        clouds.append(points)
        frames.append(int(packet['frame']))
    result = dict(resolution='coarse', stable=False, frames=frames,
                  views=len(frames), reason='insufficient_views')
    if len(frames) < min_views:
        return result
    # Full current track extent rejects fine contagion from a newly merged
    # distant/large track. Quantiles suppress isolated depth outliers.
    combined = np.concatenate(clouds)
    extent = np.quantile(combined, .98, axis=0) - np.quantile(combined, .02, axis=0)
    diagonal = float(np.linalg.norm(extent))
    result['diagonal_m'] = diagonal
    if diagonal > max_extent_m:
        result.update(stable=True, reason='large_track')
        return result
    recent = clouds[-max_views:]
    diagonals = np.array([np.linalg.norm(np.quantile(p, .98, axis=0) -
                                         np.quantile(p, .02, axis=0)) for p in recent])
    # Require three consecutive consistent observed extents. This is a
    # conservative observation stability check, not a claim of full coverage.
    d = diagonals[-min_views:]
    spread = float((d.max() - d.min()) / max(float(np.median(d)), 1e-6))
    result['relative_spread'] = spread
    if spread > max_relative_spread:
        result['reason'] = 'unstable_extent'
        return result
    result.update(resolution='fine', stable=True, reason='stable_small_track')
    return result


def project_mask_points(depth, pose, K, mask, near_m, far_m, max_points=6000):
    """Project valid masked depth samples to world coordinates in metres."""
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    y, x = np.nonzero(mask & np.isfinite(depth) &
                      (depth > float(near_m)) & (depth < float(far_m)))
    if not len(y):
        return np.empty((0, 3), dtype=np.float32)
    z = depth[y, x]
    K = np.asarray(K, dtype=np.float64)
    cam = np.column_stack(((x - K[0, 2]) * z / K[0, 0],
                           (y - K[1, 2]) * z / K[1, 1], z))
    pose = np.asarray(pose, dtype=np.float64)
    points = cam @ pose[:3, :3].T + pose[:3, 3]
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > int(max_points):
        # Deterministic spatial sampling avoids making the decision depend on
        # the raster scan order of a particular mask.
        stride = max(1, len(points) // int(max_points))
        points = points[::stride][:int(max_points)]
    return points.astype(np.float32, copy=False)


@dataclass
class ObservationScale:
    frame: int
    pixels: int
    points: int
    diagonal_m: float
    median_depth_m: float


class MultiViewResolutionPolicy:
    """Resolve fine/coarse from stable metric observations.

    ``max_extent_m`` is the diagonal of the robust 3-D observed extent.  It is
    intentionally configurable because object size is a dataset convention,
    not a mathematical constant.
    """

    def __init__(self, min_views=3, max_views=5, max_extent_m=0.45,
                 max_relative_spread=0.50):
        self.min_views = max(1, int(min_views))
        self.max_views = max(self.min_views, int(max_views))
        self.max_extent_m = float(max_extent_m)
        self.max_relative_spread = float(max_relative_spread)
        self.records = {}

    def update(self, global_id, frame, pixels, points):
        gid = int(global_id)
        points = np.asarray(points, dtype=np.float32)
        if len(points):
            extent = points.max(axis=0) - points.min(axis=0)
            diagonal = float(np.linalg.norm(extent))
            median_depth = float(np.median(points[:, 2]))
        else:
            diagonal = 0.0
            median_depth = 0.0
        item = ObservationScale(int(frame), int(pixels), int(len(points)),
                                diagonal, median_depth)
        entry = self.records.setdefault(gid, {'observations': [],
                                               'resolution': None,
                                               'resolved_frame': None})
        if len(entry['observations']) < self.max_views:
            # One observation per frame.  Duplicate callbacks must not make a
            # track appear stable faster than it really is.
            if not any(x.frame == item.frame for x in entry['observations']):
                entry['observations'].append(item)
        obs = entry['observations']
        diagonals = np.asarray([x.diagonal_m for x in obs if x.points > 0],
                               dtype=np.float64)
        stable = len(obs) >= self.min_views and len(diagonals) >= self.min_views
        if stable:
            median = float(np.median(diagonals))
            spread = float((np.percentile(diagonals, 75) -
                            np.percentile(diagonals, 25)) /
                           max(median, 1e-6))
            stable = spread <= self.max_relative_spread
            if stable and entry['resolution'] is None:
                entry['resolution'] = ('fine' if median <= self.max_extent_m
                                       else 'coarse')
                entry['resolved_frame'] = int(frame)
        return self.status(gid)

    def status(self, global_id):
        entry = self.records.get(int(global_id), {})
        obs = entry.get('observations', [])
        return dict(global_id=int(global_id),
                    stable=bool(entry.get('resolution') is not None),
                    resolution=entry.get('resolution'),
                    observation_count=len(obs),
                    observations=[asdict(x) for x in obs],
                    resolved_frame=entry.get('resolved_frame'))

    def is_fine(self, global_id):
        return self.records.get(int(global_id), {}).get('resolution') == 'fine'

    def is_coarse(self, global_id):
        return self.records.get(int(global_id), {}).get('resolution') == 'coarse'

    def resolution(self, global_id):
        return self.records.get(int(global_id), {}).get('resolution')
