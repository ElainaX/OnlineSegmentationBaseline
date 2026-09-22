import numpy as np
import open3d.core as o3c
import torch


def _o3d_tensor_to_torch(tensor):
    """Create a torch view of an Open3D tensor without a CPU round-trip."""
    return torch.utils.dlpack.from_dlpack(tensor.to_dlpack())


class TSDFDepthRenderer:
    """Render pre-integration TSDF depth and color from the current viewpoint."""

    def __init__(self, cfg):
        self.weight_threshold = float(cfg.get("min_map_weight", 3.0))
        self.range_map_down_factor = int(cfg.get("range_map_down_factor", 8))

    @torch.no_grad()
    def render(self, scene_rep, pose_w2c):
        vbg = scene_rep.voxel_block_grids
        block_count = int(vbg.hashmap().size())
        if block_count == 0:
            return None

        # Use all active blocks. Blocks derived only from the current observed
        # depth can omit a stale surface that is closer than the observation.
        block_coords = vbg.hashmap().key_tensor()[:block_count]
        pose_w2c_np = pose_w2c.detach().cpu().numpy().astype("float64")
        extrinsic = o3c.Tensor.from_numpy(np.ascontiguousarray(pose_w2c_np))

        result = vbg.ray_cast(
            block_coords=block_coords,
            intrinsic=scene_rep.intrinsic_o3c,
            extrinsic=extrinsic,
            width=scene_rep.img_w,
            height=scene_rep.img_h,
            render_attributes=["depth", "color"],
            depth_scale=scene_rep.depth_scale,
            depth_min=max(float(scene_rep.cfg["cam"].get("depth_near", 0.1)), 0.001),
            depth_max=scene_rep.depth_far,
            weight_threshold=self.weight_threshold,
            trunc_voxel_multiplier=scene_rep.trunc_voxel_multiplier,
            range_map_down_factor=self.range_map_down_factor,
        )

        rendered_depth = _o3d_tensor_to_torch(result["depth"]).clone().squeeze()
        rendered_depth = rendered_depth.to(dtype=torch.float32) / scene_rep.depth_scale

        rendered_color = _o3d_tensor_to_torch(result["color"]).clone().squeeze()
        rendered_color = rendered_color.to(dtype=torch.float32)
        # Open3D versions differ in whether VoxelBlockGrid ray-cast colors are
        # returned in [0, 1] or [0, 255]. OnlineAnySeg images use [0, 1].
        if rendered_color.numel() and rendered_color.max().item() > 1.5:
            rendered_color = rendered_color / 255.0
        rendered_color = rendered_color.clamp(0.0, 1.0)

        return {
            "depth": rendered_depth,
            "color": rendered_color,
        }
