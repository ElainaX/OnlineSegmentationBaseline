"""Fine substitution over an immutable IG export; arrays use world metres."""
import numpy as np
from scipy.spatial import cKDTree


def multiview_surface_support(points, views, K, near_m, far_m, tolerance_m, min_views):
    """Count distinct frames agreeing in mask membership and camera-Z depth."""
    counts = np.zeros(len(points), dtype=np.int32)
    seen = set()
    for packet, mask in views:
        frame = int(packet['frame'])
        if frame in seen:
            continue
        seen.add(frame)
        pose = np.asarray(packet['pose'])
        camera = (points - pose[:3, 3]) @ pose[:3, :3]
        z = camera[:, 2]
        valid = np.isfinite(camera).all(axis=1) & (z > near_m) & (z < far_m)
        ids = np.flatnonzero(valid)
        if not len(ids):
            continue
        uv = camera[ids] @ np.asarray(K).T
        uv = np.rint(uv[:, :2] / uv[:, 2:3]).astype(np.int64)
        h, w = packet['depth'].shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ids, uv = ids[inside], uv[inside]
        x, y = uv[:, 0], uv[:, 1]
        observed = packet['depth'][y, x]
        consistent = (mask[y, x] & np.isfinite(observed) &
                      (observed > near_m) & (observed < far_m) &
                      (np.abs(observed - z[ids]) <= tolerance_m))
        counts[ids[consistent]] += 1
    return counts >= min_views


def substitute_fine_regions(points, colors, masks, candidates, radius_m):
    """Replace only exclusive points of the target IG instance.

    candidates maps an existing mask column to (fine xyz, normalized RGB).
    Background and shared coarse points are never removed. Every surviving
    coarse coordinate/color/membership is copied, without voxel averaging or
    any global re-filtering. Other masks get no membership on appended points.
    """
    if masks.shape[0] != len(points) or colors.shape != points.shape:
        raise ValueError('coarse export shape mismatch')
    if radius_m <= 0:
        raise ValueError('replacement radius must be positive')
    keep = np.ones(len(points), dtype=bool)
    records, appended_xyz, appended_rgb, appended_masks = [], [], [], []
    if len(points) and candidates:
        tree = cKDTree(points)
        exclusive = masks.sum(axis=1) == 1
        for column, (xyz, rgb) in sorted(candidates.items()):
            if not 0 <= column < masks.shape[1]:
                raise ValueError('fine candidate is not a final IG instance')
            if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3:
                raise ValueError('fine export shape mismatch')
            finite = np.isfinite(xyz).all(axis=1) & np.isfinite(rgb).all(axis=1)
            xyz, rgb = xyz[finite], rgb[finite]
            if not len(xyz):
                continue
            distance, nearest = tree.query(xyz, k=1, workers=1)
            supported = (distance <= radius_m) & exclusive[nearest] & masks[nearest, column]
            xyz, rgb = xyz[supported], rgb[supported]
            if not len(xyz):
                continue
            own = np.flatnonzero(exclusive & masks[:, column])
            covered = cKDTree(xyz).query(points[own], k=1, workers=1)[0] <= radius_m
            removed = own[covered]
            keep[removed] = False
            fm = np.zeros((len(xyz), masks.shape[1]), dtype=bool)
            fm[:, column] = True
            appended_xyz.append(xyz); appended_rgb.append(rgb); appended_masks.append(fm)
            records.append(dict(column=int(column), removed_coarse_points=int(len(removed)),
                                added_fine_points=int(len(xyz))))
    retained = np.flatnonzero(keep)
    if not records:
        return points, colors, masks, retained, records
    return (np.concatenate([points[keep]] + appended_xyz),
            np.concatenate([colors[keep]] + appended_rgb),
            np.concatenate([masks[keep]] + appended_masks), retained, records)
