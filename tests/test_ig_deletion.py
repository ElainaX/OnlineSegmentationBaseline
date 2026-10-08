import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
import open3d as o3d
import open3d.core as core
from submap_disjoint.ig_deletion import IGDeletion, currently_free, eligible_lineage
from submap_disjoint.ig_correction import lattice_indices


class DeletionTests(unittest.TestCase):
    def test_depth_occlusion_and_holes(self):
        cfg=dict(depth_epsilon_m=.08, depth_relative_epsilon=.01,
                 edge_threshold_m=.08, boundary_margin_px=1)
        k=np.array([[20,0,10],[0,20,10],[0,0,1.]])
        points=np.array([[0,0,1],[0,0,2],[0,0,3.]])
        d=np.full((21,21),2.)
        np.testing.assert_equal(currently_free(points,d,k,np.eye(4),cfg,.1,5),[True,False,False])
        d[10,10]=0
        self.assertFalse(currently_free(points,d,k,np.eye(4),cfg,.1,5).any())

    def test_lineage_protects_new_observations_even_same_slot(self):
        self.assertTrue(eligible_lineage({0,1},{0,1},[4,4,4]))
        self.assertFalse(eligible_lineage({0,2},{0,1},[4,4,4]))
        self.assertFalse(eligible_lineage(None,{0,1},[4,4,4]))
        self.assertFalse(eligible_lineage({0},{0,1},[-1,4,4]))

    def test_no_free_corners_never_queries_empty_hashmap(self):
        # The owner deliberately has no VBG: an occluded confirmed candidate
        # must exit before touching any map or GPU hash lookup.
        cfg=dict(depth_epsilon_m=.08,depth_relative_epsilon=.01,
            edge_threshold_m=.08,boundary_margin_px=1)
        candidate=NS(state='would_remove',points=np.array([[0.,0,2.]]),
            depth_contradicted=np.array([True]),candidate_id=0)
        d=NS(frames_processed=1,candidates=[candidate],near=.1,far=5.,cfg=cfg,
            intrinsic=np.array([[20,0,10],[0,20,10],[0,0,1.]]))
        owner=NS(cfg=dict(scene=dict(voxel_size=.025,block_resolution=4)))
        with tempfile.TemporaryDirectory() as td:
            ctrl=IGDeletion(owner,NS(detector=d),td)
            ctrl.step(10,np.ones((21,21)),np.eye(4))
            self.assertEqual(ctrl.events[-1]['voxels'],0)

    def test_real_cuda_erasure_preserves_neighbors_and_cleans_indices(self):
        device=core.Device('CUDA:0')
        grid=o3d.t.geometry.VoxelBlockGrid(('tsdf','weight','color'),(core.float32,)*3,
            (1,1,3),.025,4,10,device)
        grid.hashmap().activate(core.Tensor([[0,0,0]],dtype=core.int32,device=device))
        grid.attribute('weight')[:]=5.;grid.attribute('tsdf')[:]=.2;grid.attribute('color')[:]=100.
        coords=torch.tensor([[0.,0,0],[.025,0,0],[.05,0,0]],device='cuda:0')
        def indices(x):
            ids, ok=lattice_indices(grid,x.cpu().numpy(),.025,4,device)
            assert ok.all()
            return torch.tensor(ids,device='cuda:0')
        ids=indices(coords)
        inst=NS(mask_voxel_coords=coords.clone(),mask_voxel_indices=ids.flip(0))
        graph=NS(instance_dict={0:inst},mask_voxels_coords={0:coords.clone()},
            f_mask_voxels_clouds={(0,1):coords.clone()},
            voxel_in_frame_matrix=torch.ones((640,3),dtype=torch.bool,device='cuda:0'),
            boundary_voxel_indices=ids.clone(),merged_mask_in_frame_mat=torch.ones((1,3),device='cuda:0'),
            query_mask_under_visible_part=lambda *a:(torch.zeros(0,dtype=torch.long,device='cuda:0'),None,None))
        table=NS(voxel_coords2voxel_indices=indices,hash_table=[{0} for _ in range(640)],
            hash_item_flag=torch.ones(640,dtype=torch.bool,device='cuda:0'))
        owner=NS(cfg=dict(scene=dict(voxel_size=.025,block_resolution=4)),
            voxel_block_grids=grid,device_o3c=device,device='cuda:0',maskGraph=graph,
            voxel_hash_table=table,mask_voxels_coords={0:coords.clone()},merged_mask_size=[3],
            mask_geo_features=torch.ones((1,16),device='cuda:0'),mask_geo_features_mask=torch.ones(1,device='cuda:0'),
            global2merged_mask_id=[0],global_f_mask_ids=[(0,1)],geometry=NS())
        with tempfile.TemporaryDirectory() as td:
            ctrl=IGDeletion(owner,None,td)
            result=ctrl.erase(1,[(0,coords[:1].cpu().numpy())])
            self.assertEqual(result['voxels'],1)
            self.assertEqual(float(grid.attribute('weight').reshape((-1,))[int(ids[0])].item()),0.)
            self.assertEqual(float(grid.attribute('weight').reshape((-1,))[int(ids[1])].item()),5.)
            torch.testing.assert_close(inst.mask_voxel_coords,coords[1:])
            torch.testing.assert_close(inst.mask_voxel_indices,ids[1:])
            torch.testing.assert_close(graph.f_mask_voxels_clouds[(0,1)],coords[1:])
            self.assertIsNone(table.hash_table[int(ids[0])])
            self.assertFalse(graph.voxel_in_frame_matrix[ids[0]].any())
            self.assertTrue(graph.voxel_in_frame_matrix[ids[1]].all())
            archive=np.load(Path(td)/'archive/000001.npz')
            self.assertEqual(archive['weight'].item(),5.)


if __name__=='__main__': unittest.main()
