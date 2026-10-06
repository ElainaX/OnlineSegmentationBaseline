"""Pre-fusion, GT-free auditing of FIXED world surface samples.

Adapted from the archived audit_3rscan/world_guard diagnostics. Depth is camera
Z in meters; poses are T_world_camera. Nearer measurements are occlusion, not
disappearance. Invalid measurements do not count as votes of either kind.
"""
from dataclasses import dataclass
import numpy as np

UNKNOWN, SUPPORT, FREE, OCCLUDED = range(4)


@dataclass(frozen=True)
class AuditOptions:
    tau_min_m: float = .035
    tau_relative: float = .04
    map_gate_m: float = .04
    edge_m: float = .08
    cell_m: float = .05
    near_m: float = .1
    far_m: float = 5.
    max_points: int = 20000

    def __post_init__(self):
        if min(self.tau_min_m, self.map_gate_m, self.edge_m, self.cell_m) <= 0:
            raise ValueError('positive metric thresholds required')
        if self.tau_relative < 0 or not 0 <= self.near_m < self.far_m or self.max_points < 1:
            raise ValueError('invalid audit limits')


def validate_camera(K, pose):
    K, pose = np.asarray(K), np.asarray(pose)
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('invalid intrinsic matrix')
    if not np.allclose(K[2], [0, 0, 1]) or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8:
        raise ValueError('requires rectified pinhole intrinsics, zero skew')
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('invalid pose; never substitute previous pose for change detection')
    R = pose[:3, :3]
    if not (np.allclose(pose[3], [0, 0, 0, 1]) and
            np.allclose(R.T @ R, np.eye(3), atol=1e-4) and np.linalg.det(R) > .999):
        raise ValueError('pose is not rigid')


def depth_edges(depth, threshold):
    """Exclude both sides of discontinuities AND holes, without wrap-around."""
    d = np.asarray(depth)
    valid = np.isfinite(d) & (d > 0)
    edge = ~valid
    for axis in (0, 1):
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(1, None); b[axis] = slice(None, -1)
        a, b = tuple(a), tuple(b)
        jump = (valid[a] != valid[b]) | (valid[a] & valid[b] & (np.abs(d[a] - d[b]) > threshold))
        edge[a] |= jump; edge[b] |= jump
    edge[0] = edge[-1] = True; edge[:, 0] = edge[:, -1] = True
    return edge


def sample_surface(points, options):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(1)]
    if not len(points): return np.empty((0, 3))
    # Canonical ordering makes GPU extraction order irrelevant.
    points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    _, idx = np.unique(np.floor(points / options.cell_m).astype(np.int64), axis=0, return_index=True)
    points = points[idx]
    if len(points) > options.max_points:
        points = points[np.linspace(0, len(points) - 1, options.max_points).astype(int)]
    return points.copy()


def raycast_volume(volume, K, pose, shape):
    """Query allocated OLD blocks, independent of current measured depth.

    Using only blocks touched by the new far depth can hide a vanished near
    surface. All allocated blocks are used here for correctness; spatial culling
    is a future optimization. GPU raycast, CPU evidence accumulation.
    """
    import open3d.core as core
    h, w = shape; grid = volume.grid
    if not grid.hashmap().size(): return np.zeros(shape, np.float32)
    active = grid.hashmap().active_buf_indices().to(core.int64)
    blocks = grid.hashmap().key_tensor()[active]
    out = grid.ray_cast(blocks, core.Tensor(np.asarray(K, dtype=np.float64)),
        core.Tensor(np.linalg.inv(pose) @ volume.anchor), w, h, ['depth'],
        depth_scale=1., depth_min=volume.options.near_m, depth_max=volume.options.far_m,
        weight_threshold=volume.options.extraction_weight,
        trunc_voxel_multiplier=volume.options.trunc_voxels, range_map_down_factor=1)
    depth = out['depth'].cpu().numpy().reshape(h, w).copy()
    depth[~np.isfinite(depth)] = 0
    return depth


def audit_surface(points, observed, reference, live_depth, K, pose, options):
    """One state per fixed old surface cell; no GT labels or instance vote."""
    validate_camera(K, pose)
    observed, reference, live_depth = map(np.asarray, (observed, reference, live_depth))
    if observed.ndim != 2 or reference.shape != observed.shape or live_depth.shape != observed.shape:
        raise ValueError('depth images must have matching H x W shapes')
    state = np.zeros(len(points), np.uint8)
    cam = (points - pose[:3, 3]) @ pose[:3, :3]
    z = cam[:, 2]; valid = np.isfinite(cam).all(1) & (z > options.near_m) & (z < options.far_m)
    idx = np.flatnonzero(valid)
    if not len(idx): return state
    u = np.rint(K[0, 0] * cam[idx, 0] / z[idx] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * cam[idx, 1] / z[idx] + K[1, 2]).astype(int)
    h, w = observed.shape
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    idx, u, v = idx[inside], u[inside], v[inside]
    d, r, front = observed[v, u], reference[v, u], live_depth[v, u]
    tau = np.maximum(options.tau_min_m, options.tau_relative * z[idx])
    valid = (np.isfinite(d) & np.isfinite(r) & (d > options.near_m) &
             (d < options.far_m) & (r > 0) & (np.abs(r - z[idx]) <= options.map_gate_m))
    valid &= ~depth_edges(observed, options.edge_m)[v, u]
    valid &= ~depth_edges(reference, options.edge_m)[v, u]
    hidden = (front > 0) & (front < z[idx] - options.map_gate_m)
    state[idx[valid & (hidden | (d < z[idx] - tau))]] = OCCLUDED
    visible = valid & ~hidden
    state[idx[visible & (np.abs(d - z[idx]) <= tau)]] = SUPPORT
    state[idx[visible & (d > z[idx] + tau)]] = FREE
    return state
