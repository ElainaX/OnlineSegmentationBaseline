"""Unchanged diagnostic helper bodies copied from local onlineSeg."""
import numpy as np
import torch
import open3d.core as o3c

def lookup_voxels_safe(self, world_points):
        """Read-only lookup for diagnostics; unknown positions have index -1."""
        points = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        indices = np.full(len(points), -1, dtype=np.int64)
        found = np.zeros(len(points), dtype=bool)
        finite = np.isfinite(points).all(axis=1)
        if not finite.any() or self.voxel_block_grids.hashmap().size() == 0:
            return indices, found
        scaled = points[finite] / self.voxel_size
        # Float32 coordinates of exact negative grid nodes can lie just below
        # their integer index after conversion to float64 (e.g. -0.025f).
        nearest = np.rint(scaled)
        scaled = np.where(np.abs(scaled - nearest) < 1e-5, nearest, scaled)
        grid = np.floor(scaled).astype(np.int64)
        blocks = np.floor_divide(grid, self.block_resolution)
        local = grid - blocks * self.block_resolution
        query = o3c.Tensor(blocks.astype(np.int32), device=self.device_o3c)
        buffers, valid = self.voxel_block_grids.hashmap().find(query)
        buffers = buffers.cpu().numpy().astype(np.int64)
        valid = valid.cpu().numpy().astype(bool)
        offsets = local[:, 0] + self.block_resolution * local[:, 1] + self.block_resolution**2 * local[:, 2]
        rows = np.flatnonzero(finite)[valid]
        indices[rows] = buffers[valid] * self.block_resolution**3 + offsets[valid]
        found[rows] = True
        return indices, found

def raycast_history(self, intrinsic, pose_w2c, width, height,
                        depth_min, depth_max, min_weight, trunc_multiplier):
        """Render without activating/integrating blocks or editing TSDF attributes.

        Range estimation in Open3D 0.18 can truncate its first rendering while
        growing its scratch buffer. A duplicated range-only block list warms a
        larger buffer before rendering the real list. Duplicate coordinates do
        not allocate map blocks; ray_cast only reads the map.
        """
        vbg = self.voxel_block_grids
        if vbg.hashmap().size() == 0:
            return None
        active = vbg.hashmap().active_buf_indices().to(o3c.int64)
        blocks = vbg.hashmap().key_tensor()[active]
        k = o3c.Tensor(np.asarray(intrinsic, dtype=np.float64))
        ext = o3c.Tensor(np.asarray(pose_w2c, dtype=np.float64))
        kwargs = dict(depth_scale=1.0, depth_min=float(depth_min),
                      depth_max=float(depth_max), weight_threshold=float(min_weight),
                      trunc_voxel_multiplier=float(trunc_multiplier))
        # CPU concatenation avoids assumptions about version-specific Tensor APIs.
        block_array = blocks.cpu().numpy()
        warm_blocks = o3c.Tensor(np.concatenate([block_array, block_array]),
                                 device=self.device_o3c)
        vbg.ray_cast(warm_blocks, k, ext, width, height, ['depth'], **kwargs)
        result = vbg.ray_cast(blocks, k, ext, width, height,
                              ['depth', 'color', 'index', 'mask', 'interp_ratio'], **kwargs)
        depth = result['depth'].cpu().numpy()[..., 0].copy()
        color = result['color'].cpu().numpy().copy()
        index = result['index'].cpu().numpy().copy()
        mask = result['mask'].cpu().numpy().copy()
        ratio = result['interp_ratio'].cpu().numpy().copy()
        # Gather on the map's device instead of copying its full weight buffer.
        index_t = torch.utils.dlpack.from_dlpack(result['index'].to_dlpack()).long()
        weight_t = torch.utils.dlpack.from_dlpack(vbg.attribute('weight').to_dlpack()).reshape(-1)
        safe = index_t.clamp(min=0, max=len(weight_t) - 1)
        weight = weight_t[safe].cpu().numpy()
        mask &= (index >= 0) & (index < len(weight_t))
        coeff = np.where(mask & np.isfinite(ratio), ratio, 0.)
        coverage = coeff.sum(axis=-1)
        surface_weight = (weight * coeff).sum(axis=-1) / np.maximum(coverage, 1e-12)
        # A partial interpolation stencil is not adequate evidence for removal.
        valid = (np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
                 & (coverage >= .99) & (surface_weight >= min_weight))
        return dict(depth=depth, color=color, valid=valid, weight=surface_weight,
                    index=index, neighbor_valid=mask, interp_ratio=coeff)

def query_active_instance_memberships(self, flat_indices, id_mapping):
        """Per-voxel active membership, without changing the original hash table.

        Several raw masks mapping to one instance count as one membership.
        Missing voxels and tombstoned/out-of-range raw IDs return no membership.
        """
        memberships = []
        for index in np.asarray(flat_indices, dtype=np.int64).reshape(-1):
            if index < 0 or index >= len(self.hash_table):
                memberships.append(frozenset())
                continue
            raw_ids = self.hash_table[int(index)]
            active = set()
            if raw_ids:
                for raw in raw_ids:
                    if 0 <= raw < len(id_mapping) and id_mapping[raw] >= 0:
                        active.add(int(id_mapping[raw]))
            memberships.append(frozenset(active))
        return memberships
