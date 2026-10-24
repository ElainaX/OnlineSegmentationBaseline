"""Shared TSDF geometry service; instance state belongs to DisjointSubmapManager."""
import numpy as np
import torch
import open3d as o3d
import open3d.core as o3c
from voxelized_points import VoxelBlockGrid
from PC_extractor import PointFeatureExtractor

class GeometryScene:
    def __init__(self, cfg, args, dataset, device="cuda:0"):
        self.cfg = cfg
        self.args = args
        self.dataset = dataset
        self.device = device
        self.device_o3c = o3c.Device(self.device)
        self.img_h = dataset.target_h
        self.img_w = dataset.target_w
        self.depth_scale = self.cfg["cam"]["depth_scale"]
        self.depth_far = self.cfg["cam"]["depth_far"]
        self.intrinsics = dataset.cam_intrinsic  # Tensor(3, 3), dtype=float32
        self.intrinsic_o3c = o3c.Tensor.from_numpy( self.intrinsics.cpu().numpy().astype("float64") )
        self.pinhole_cam_intrinsic = dataset.pinhole_cam_intrinsic  # o3d.camera.PinholeCameraIntrinsic obj

        # VoxelBlockGrid representation
        self.voxel_grids = VoxelBlockGrid(self.cfg, dataset, self.device)  # ***
        self.voxel_block_grids = self.voxel_grids.voxel_block_grids
        self.voxel_hash_table = self.voxel_grids.voxel_hash_table
        self.trunc_voxel_multiplier = self.cfg["scene"]["trunc_voxel_multiplier"] if "trunc_voxel_multiplier" in self.cfg["scene"] else 4.

        # vars for corresponding (active) voxels in the latest frame
        self.active_frame_id = -1
        self.active_voxel_coords = None
        self.active_voxel_indices = None

        # Pointcloud geometric feature extractor
        self.pc_extractor = PointFeatureExtractor(self.cfg, self.voxel_grids, self.device)
        self.pc_xyz = None  # latest extracted pointcloud, Tensor(pts_num, 3)
        self.pc_feature = None  # latest extracted feature pointcloud, Tensor(pts_num, feature_dim)
        self.pc_feature_frame_id = -1  # latest frame_id that we extracted feature pointcloud, int

        self.pc_frame_id = -1
        self.points = self.colors = None
    def integrate_frame(self, frame_id, color_img, depth_img, pose_w2c):
        color_img = (color_img * 255.).cpu().numpy().astype("uint8")
        depth_img = (depth_img * self.depth_scale).cpu().numpy().astype("uint16")
        pose_w2c = pose_w2c.cpu().numpy().astype("float64")

        color_img = np.ascontiguousarray(color_img)
        depth_img = np.ascontiguousarray(depth_img)
        pose_w2c = np.ascontiguousarray(pose_w2c)
        color = o3d.t.geometry.Image(color_img).to(o3c.uint8).to(self.device_o3c)
        depth = o3d.t.geometry.Image(depth_img).to(o3c.uint16).to(self.device_o3c)
        extrinsic = o3c.Tensor.from_numpy(pose_w2c)

        # Get active frustum block coordinates from input frame, and them integrate input frame to TSDF Volume
        frustum_block_coords = self.voxel_block_grids.compute_unique_block_coordinates(depth, self.intrinsic_o3c, extrinsic, self.depth_scale, self.depth_far)  # active blocks of this frame
        self.voxel_block_grids.integrate(frustum_block_coords, depth, color, self.intrinsic_o3c, extrinsic, self.depth_scale, self.depth_far, trunc_voxel_multiplier=self.trunc_voxel_multiplier)

        return frustum_block_coords, extrinsic
    def get_pc(self):
        pcd = self.voxel_block_grids.extract_point_cloud()
        points = pcd.point.positions.cpu().numpy()
        colors = pcd.point.colors.cpu().numpy()
        return points, colors
