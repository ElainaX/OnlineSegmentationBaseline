"""RGB-D candidate frontend, adapted from onlineSeg e6907b8/remove_detection.py.

CPU-only evidence functions; Open3D is imported only by render_quality().
Appearance disagreement NEVER becomes FREE. Candidates are proposals for the
lifecycle controller, not an instruction to erase geometry. Poses: T_world_camera.
"""
from dataclasses import dataclass
import numpy as np
from .contradiction import UNKNOWN, SUPPORT, FREE, OCCLUDED, validate_camera


@dataclass(frozen=True)
class FrontendOptions:
    enabled: bool = False  # preserve the 42e6361 control unless explicitly enabled
    interval: int = 10  # RAW frame id; caller sampling must be aligned
    rgb_enabled: bool = True
    depth_epsilon_m: float = .08
    depth_relative_epsilon: float = .01
    min_map_weight: float = 3.
    min_instance_observations: int = 5
    interpolation_coverage: float = .99
    edge_threshold_m: float = .08
    boundary_margin_px: int = 1
    min_valid_pixels: int = 80
    min_reference_points: int = 100
    min_reference_tested_points: int = 100
    contradiction_ratio: float = .5
    max_global_conflict_fraction: float = .75
    rgb_mad_multiplier: float = 4.
    rgb_threshold_min: float = 8.
    rgb_threshold_max: float = 50.
    min_rgb_calibration_pixels: int = 80
    owner_tolerance_m: float = .025
    raycast_warmup: bool = True

    def __post_init__(self):
        for name in ('interval', 'min_instance_observations', 'min_valid_pixels',
                     'min_reference_points', 'min_reference_tested_points', 'min_rgb_calibration_pixels'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1: raise ValueError(name)
        if type(self.boundary_margin_px) is not int or self.boundary_margin_px < 0: raise ValueError('boundary margin')
        for name in ('interpolation_coverage', 'contradiction_ratio', 'max_global_conflict_fraction'):
            if not 0 < getattr(self, name) <= 1: raise ValueError(name)
        for name in ('depth_epsilon_m', 'depth_relative_epsilon', 'min_map_weight', 'edge_threshold_m',
                     'rgb_mad_multiplier', 'rgb_threshold_min', 'rgb_threshold_max', 'owner_tolerance_m'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0: raise ValueError(name)
        if self.min_map_weight <= 0 or self.rgb_threshold_min > self.rgb_threshold_max: raise ValueError('invalid thresholds')
        for name in ('enabled', 'rgb_enabled', 'raycast_warmup'):
            if type(getattr(self, name)) is not bool: raise ValueError(name)


def interior(depth, valid, jump, margin):
    bad = ~np.asarray(valid, bool).copy()
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(1, None), slice(None, -1)
        a, b = tuple(a), tuple(b)
        step = (np.abs(depth[a] - depth[b]) > jump) | (valid[a] != valid[b])
        bad[a] |= step; bad[b] |= step
    for _ in range(margin):
        p = np.pad(bad, 1, constant_values=True)
        bad = np.logical_or.reduce([p[y:y+bad.shape[0], x:x+bad.shape[1]] for y in range(3) for x in range(3)])
    return valid & ~bad


def owner_interior(owners, margin):
    good = owners >= 0
    for _ in range(margin):
        p = np.pad(owners, 1, constant_values=-3); g = np.pad(good, 1, constant_values=False)
        good &= np.logical_and.reduce([(p[y:y+owners.shape[0], x:x+owners.shape[1]] == owners)
            & g[y:y+owners.shape[0], x:x+owners.shape[1]] for y, x in ((0, 1), (2, 1), (1, 0), (1, 2))])
    return good


def owners_from_stencil(indices, valid, ratios, membership_query):
    """Baseline integration interface: unique CURRENT identity, not raw-mask votes.

    Callback returns a set of active instance IDs for each voxel index. Multiple
    observations of the same ID count once. -1 unknown, -2 ambiguous.
    """
    shape = indices.shape[:-1]; ids = np.asarray(indices).reshape(-1, indices.shape[-1])
    useful = np.asarray(valid).reshape(ids.shape) & (np.asarray(ratios).reshape(ids.shape) > 1e-6) & (ids >= 0)
    keys = np.unique(ids[useful]); lookup = dict(zip(keys.tolist(), membership_query(keys)))
    owners = np.full(len(ids), -1, np.int64)
    for row in np.flatnonzero(useful.any(1)):
        members = set()
        for key in ids[row, useful[row]]:
            members.update(lookup[int(key)])
            if len(members) > 1: break
        if len(members) == 1: owners[row] = next(iter(members))
        elif members: owners[row] = -2
    return owners.reshape(shape)


def classify(current, reference, valid, cfg):
    tau = cfg.depth_epsilon_m + cfg.depth_relative_epsilon * np.maximum(current, reference)
    good = valid & np.isfinite(current) & np.isfinite(reference) & (current > 0) & (reference > 0)
    code = np.zeros(current.shape, np.uint8); delta = current - reference
    code[good & (np.abs(delta) <= tau)] = SUPPORT
    code[good & (delta > tau)] = FREE
    code[good & (delta < -tau)] = OCCLUDED
    return code


def rgb01(rgb):
    rgb = np.asarray(rgb)
    if rgb.shape[-1] != 3: raise ValueError('RGB must end in 3 channels')
    # Explicit contract avoids confusing a dark uint8 image with normalized RGB.
    if rgb.dtype == np.uint8: return rgb.astype(np.float32) / 255.
    if not np.issubdtype(rgb.dtype, np.floating): raise ValueError('RGB must be uint8 or float [0,1]')
    finite = rgb[np.isfinite(rgb)]
    if len(finite) and (finite.min() < 0 or finite.max() > 1.001): raise ValueError('float RGB must be in [0,1]')
    return rgb.astype(np.float32)


def lab(rgb):
    import cv2
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2LAB)


def rgb_context(current, history, eligible, cfg):
    if not cfg.rgb_enabled:
        return dict(enabled=False, current_lab=None, threshold=cfg.rgb_threshold_max,
            bad=np.zeros(np.shape(eligible), bool), offset=[0., 0., 0.], median=0., mad=0., calibration_pixels=0)
    import cv2
    current, history = rgb01(current), rgb01(history)
    finite = np.isfinite(current).all(-1) & np.isfinite(history).all(-1)
    eligible = eligible & finite
    c = cv2.GaussianBlur(np.nan_to_num(current), (3, 3), 0)
    h = cv2.GaussianBlur(np.nan_to_num(history), (3, 3), 0)
    enabled = cfg.rgb_enabled and int(eligible.sum()) >= cfg.min_rgb_calibration_pixels
    offset = np.median(h[eligible] - c[eligible], axis=0) if enabled else np.zeros(3, np.float32)
    adjusted = np.clip(c + offset, 0, 1); cl, hl = lab(adjusted), lab(h)
    residual = np.linalg.norm(cl - hl, axis=-1)
    median = float(np.median(residual[eligible])) if enabled else 0.
    mad = float(np.median(np.abs(residual[eligible] - median))) if enabled else 0.
    threshold = float(np.clip(median + cfg.rgb_mad_multiplier * 1.4826 * mad,
        cfg.rgb_threshold_min, cfg.rgb_threshold_max)) if enabled else cfg.rgb_threshold_max
    return dict(enabled=enabled, current_lab=cl, threshold=threshold,
        bad=enabled & eligible & (residual > threshold), offset=offset.tolist(),
        median=median, mad=mad, calibration_pixels=int(eligible.sum()))


def raycast_validity(depth, indices, neighbor_valid, ratios, weight_values, cfg, near, far):
    """weight_values already gathered at safe indices; never copy the whole grid."""
    good = np.asarray(neighbor_valid, bool) & (indices >= 0) & np.isfinite(ratios) & (ratios >= 0)
    coeff = np.where(good, ratios, 0.); coverage = coeff.sum(-1)
    weight = (np.where(good, weight_values, 0) * coeff).sum(-1) / np.maximum(coverage, 1e-12)
    valid = np.isfinite(depth) & (depth > near) & (depth < far) & np.isfinite(weight)
    valid &= (coverage >= cfg.interpolation_coverage) & (weight >= cfg.min_map_weight)
    return valid, weight, coverage


def render_quality(volume, K, pose, shape, cfg):
    """All allocated old blocks; color normalized explicitly from this store's 0..255."""
    import open3d.core as core
    grid = volume.grid; h, w = shape
    empty = dict(depth=np.zeros(shape, np.float32), color=np.zeros(shape + (3,), np.float32),
        valid=np.zeros(shape, bool), weight=np.zeros(shape, np.float32))
    if not grid.hashmap().size(): return empty
    active = grid.hashmap().active_buf_indices().to(core.int64)
    blocks = grid.hashmap().key_tensor()[active]
    intrinsic = core.Tensor(np.asarray(K, dtype=np.float64)); ext = core.Tensor(np.linalg.inv(pose) @ volume.anchor)
    kwargs = dict(depth_scale=1., depth_min=volume.options.near_m, depth_max=volume.options.far_m,
        weight_threshold=cfg.min_map_weight, trunc_voxel_multiplier=volume.options.trunc_voxels, range_map_down_factor=1)
    if cfg.raycast_warmup:
        block_np = blocks.cpu().numpy()
        grid.ray_cast(core.Tensor(np.concatenate([block_np, block_np]), device=volume.device), intrinsic, ext,
            w, h, ['depth'], **kwargs)
    attributes = ['depth', 'index', 'mask', 'interp_ratio'] + (['color'] if cfg.rgb_enabled else [])
    result = grid.ray_cast(blocks, intrinsic, ext, w, h, attributes, **kwargs)
    depth = result['depth'].cpu().numpy().reshape(shape).copy()
    color = (np.clip(result['color'].cpu().numpy().reshape(shape + (3,)) / 255., 0., 1.)
             if cfg.rgb_enabled else empty['color'])
    idx = result['index']; indices = idx.cpu().numpy().copy()
    weights = grid.attribute('weight').reshape((-1,))
    safe = core.Tensor(np.clip(indices, 0, int(weights.shape[0]) - 1), dtype=core.int64, device=volume.device)
    sampled = weights[safe.reshape((-1,))].cpu().numpy().reshape(indices.shape)
    valid, weight, _ = raycast_validity(depth, indices,
        result['mask'].cpu().numpy() & (indices < int(weights.shape[0])),
        result['interp_ratio'].cpu().numpy(), sampled, cfg, volume.options.near_m, volume.options.far_m)
    return dict(depth=depth, color=color, valid=valid, weight=weight)


def compose_history(rendered, published_keys, slots, shape, cfg):
    """Z-composite independent volumes. Same-depth different fragments are ambiguous.

    Callers with a global voxel ownership map can instead supply owners obtained
    from owners_from_stencil. No arbitrary winning-fragment ownership vote.
    """
    depth = np.full(shape, np.inf, np.float32); color = np.zeros(shape + (3,), np.float32)
    owner = np.full(shape, -1, np.int64); valid = np.zeros(shape, bool)
    for key in published_keys:
        r = rendered[key]; positive = np.isfinite(r['depth']) & (r['depth'] > 0)
        nearer = positive & (r['depth'] < depth)
        depth[nearer] = r['depth'][nearer]; color[nearer] = r['color'][nearer]; valid[nearer] = r['valid'][nearer]
    for key in published_keys:
        r = rendered[key]
        close = r['valid'] & valid & (np.abs(r['depth'] - depth) <= cfg.owner_tolerance_m)
        conflict = close & (owner != -1) & (owner != slots[key])
        owner[close & (owner == -1)] = slots[key]; owner[conflict] = -2
    depth[~np.isfinite(depth)] = 0
    return dict(depth=depth, color=color, valid=valid, owners=owner)


def ownership_slots(store, keys):
    """Honor explicit identity links without fusing their independent geometry."""
    memberships = {k: [] for k in keys}
    for identity, members in getattr(store, 'identities', {}).items():
        for k in members:
            if k in memberships: memberships[k].append(str(identity))
    groups = {}; slots = {}
    for k in keys:
        if len(memberships[k]) > 1: raise ValueError('fragment has ambiguous explicit identities: ' + k)
        group = ('identity', memberships[k][0]) if memberships[k] else ('fragment', k)
        if group not in groups: groups[group] = len(groups)
        slots[k] = groups[group]
    return slots


def prepare_context(rgb, depth, history, cfg, near, far):
    current_valid = np.isfinite(depth) & (depth > near) & (depth < far)
    ci = interior(depth, current_valid, cfg.edge_threshold_m, cfg.boundary_margin_px)
    hi = interior(history['depth'], history['valid'], cfg.edge_threshold_m, cfg.boundary_margin_px)
    codes = classify(depth, history['depth'], ci & hi, cfg)
    colors = rgb_context(rgb, history['color'], codes == SUPPORT, cfg)
    tested = (codes == SUPPORT) | (codes == FREE)
    conflict = (codes == FREE) | colors['bad']
    ratio = float(conflict.sum() / max(1, tested.sum()))
    guard = tested.sum() >= cfg.min_valid_pixels and ratio > cfg.max_global_conflict_fraction
    reasons = []
    if current_valid.sum() < cfg.min_valid_pixels: reasons.append('insufficient_current_depth')
    if history['valid'].sum() < cfg.min_valid_pixels: reasons.append('insufficient_historical_render')
    if guard: reasons.append('widespread_conflict')
    return dict(current_interior=ci, depth_codes=codes, rgb=colors,
        owner_good=owner_interior(history['owners'], cfg.boundary_margin_px),
        skip_reasons=reasons, global_ratio=ratio)


def fragment_evidence(points, rgb, depth, K, pose, own, history, context, slot,
                      observations, cfg, near, far, map_gate_m, private=False):
    """Two levels: pixel/instance candidate proposal + fixed-point geometric audit."""
    validate_camera(K, pose)
    h, w = depth.shape; n = len(points)
    codes = np.zeros(n, np.uint8); appearance = np.zeros(n, bool)
    # Proposals/inactive references use their own rendering for ownership. They
    # are not published in the composite; current nearer depth still occludes.
    if private:
        owned = interior(own['depth'], own['valid'], cfg.edge_threshold_m, cfg.boundary_margin_px)
        owned &= history['owners'] != -2
        pixel_codes = classify(depth, own['depth'], context['current_interior'] & owned, cfg)
    else:
        owned = (history['owners'] == slot) & context['owner_good']
        pixel_codes = np.where(owned, context['depth_codes'], UNKNOWN)
    tested = (pixel_codes == SUPPORT) | (pixel_codes == FREE)
    if private and context['rgb']['enabled']:
        residual = np.linalg.norm(context['rgb']['current_lab'] - lab(np.nan_to_num(own['color'])), axis=-1)
        rgb_bad = owned & (pixel_codes == SUPPORT) & (residual > context['rgb']['threshold'])
    else:
        rgb_bad = owned & context['rgb']['bad'] & (pixel_codes == SUPPORT)
    total = int(tested.sum()); bad_count = int(((pixel_codes == FREE) | rgb_bad).sum())
    failures = []
    if observations < cfg.min_instance_observations: failures.append('immature_fragment')
    if n < cfg.min_reference_points: failures.append('insufficient_reference_points')
    if total < cfg.min_valid_pixels: failures.append('insufficient_pixels')
    if bad_count / max(1, total) < cfg.contradiction_ratio: failures.append('low_pixel_conflict_ratio')
    # Protect ALL later votes with per-reference quality, regardless of whether
    # there is enough contradiction to create a new candidate this frame.
    if observations >= cfg.min_instance_observations and n >= cfg.min_reference_points:
        cam = (points - pose[:3, 3]) @ pose[:3, :3]; z = cam[:, 2]
        ids = np.flatnonzero(np.isfinite(cam).all(1) & (z > near) & (z < far))
        u = np.rint(K[0,0]*cam[ids,0]/z[ids]+K[0,2]).astype(int)
        v = np.rint(K[1,1]*cam[ids,1]/z[ids]+K[1,2]).astype(int)
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ids, u, v = ids[inside], u[inside], v[inside]
        pixels = v*w+u; order = np.lexsort((ids, z[ids], pixels))
        _, first = np.unique(pixels[order], return_index=True); pick = order[first]
        ids, u, v = ids[pick], u[pick], v[pick]
        ok = owned[v,u] & context['current_interior'][v,u] & own['valid'][v,u]
        ok &= np.abs(own['depth'][v,u] - z[ids]) <= map_gate_m
        local = classify(depth[v,u], z[ids], ok, cfg)
        front = history['depth'][v,u]
        hidden = history['valid'][v,u] & (front < z[ids] - map_gate_m)
        local[ok & hidden] = OCCLUDED
        codes[ids] = local
        if cfg.rgb_enabled and context['rgb']['enabled']:
            own_lab = lab(np.nan_to_num(own['color']))
            residual = np.linalg.norm(context['rgb']['current_lab'][v,u] - own_lab[v,u], axis=-1)
            appearance[ids] = ((local == SUPPORT) & np.isfinite(own['color'][v,u]).all(-1)
                               & (residual > context['rgb']['threshold']))
    tested_points = int(((codes == SUPPORT) | (codes == FREE)).sum())
    if tested_points < cfg.min_reference_tested_points: failures.append('insufficient_tested_reference')
    union = int(((codes == FREE) | appearance).sum())
    if union / max(1, tested_points) < cfg.contradiction_ratio: failures.append('low_reference_conflict_ratio')
    candidate = not failures and not context['skip_reasons']
    return codes, appearance, dict(candidate=candidate, candidate_failures=failures,
        tested_pixels=total, depth_pixels=int((pixel_codes == FREE).sum()), rgb_pixels=int(rgb_bad.sum()),
        tested_reference=tested_points, depth_reference=int((codes == FREE).sum()),
        rgb_reference=int(appearance.sum()), reference_conflict_ratio=union / max(1, tested_points))
