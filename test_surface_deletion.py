import unittest
import numpy as np
from submap_disjoint.surface_deletion import (unchanged_lineage, confirmed_components,
                                             free_without_nearby_surface, DeletionOptions)

class SurfaceDeletionTests(unittest.TestCase):
    def test_raw_lineage_survives_merge_renumbering(self):
        self.assertTrue(unchanged_lineage({1,2},{1,2},[9,5,5]))
    def test_new_raw_owner_prevents_erasure(self):
        self.assertFalse(unchanged_lineage({1,2,3},{1,2},[0,0,0,0]))
    def test_no_owner_does_not_authorize_erasure(self):
        self.assertFalse(unchanged_lineage(set(),set(),[0]))
        self.assertFalse(unchanged_lineage({0},{0},[-1]))
    def test_unrelated_leases_must_not_be_unioned(self):
        self.assertFalse(unchanged_lineage({0,1},{0},[0,0]))
        self.assertFalse(unchanged_lineage({0,1},{1},[0,0]))
    def test_speckles_are_not_whole_object_permission(self):
        points=np.array([[0,0,0],[.01,0,0],[.02,0,0],[3,0,0]])
        np.testing.assert_equal(confirmed_components(points,3,.03),[True,True,True,False])
    def test_empty_points(self):
        self.assertEqual(len(confirmed_components(np.empty((0,3)),20,.1)),0)
    def setup_depth(self):
        cfg=dict(edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,depth_relative_epsilon=.01)
        K=np.array([[100,0,5],[0,100,5],[0,0,1]],float)
        return cfg,K,np.eye(4)
    def test_free_vs_occluded_vs_support(self):
        cfg,K,pose=self.setup_depth();depth=np.full((11,11),2.)
        pts=np.array([[0,0,1],[0,0,3],[0,0,2.]])
        free,_=free_without_nearby_surface(pts,depth,K,pose,cfg,.1,5,.05)
        np.testing.assert_equal(free,[True,False,False])
    def test_unknown_cannot_erase(self):
        cfg,K,pose=self.setup_depth()
        free,_=free_without_nearby_surface(np.array([[0,0,1]]),np.zeros((11,11)),K,pose,cfg,.1,5,.05)
        self.assertFalse(free.any())
    def test_nearby_current_surface_veto(self):
        cfg,K,pose=self.setup_depth();cfg['edge_threshold_m']=2.
        depth=np.full((11,11),2.);depth[5,6]=1.
        free,stats=free_without_nearby_surface(np.array([[0,0,1]]),depth,K,pose,cfg,.1,5,.05)
        self.assertFalse(free.any());self.assertEqual(stats['nearby_veto'],1)
    def test_world_pose_camera_z(self):
        cfg,K,pose=self.setup_depth();pose[:3,3]=[4,5,6]
        free,_=free_without_nearby_surface(np.array([[4,5,7.]]),np.full((11,11),2.),K,pose,cfg,.1,5,.05)
        self.assertTrue(free[0])
    def test_out_of_view_cannot_erase(self):
        cfg,K,pose=self.setup_depth()
        free,_=free_without_nearby_surface(np.array([[30,0,1],[0,0,-1]]),np.full((11,11),2.),K,pose,cfg,.1,5,.05)
        self.assertFalse(free.any())
    def test_invalid_options(self):
        with self.assertRaises(ValueError):DeletionOptions(min_component_points=0)

    def test_real_cpu_tsdf_and_semantic_caches_are_pruned_together(self):
        import tempfile
        import torch
        import open3d as o3d
        import open3d.core as core
        from types import SimpleNamespace as NS
        from submap_disjoint.ig_correction import corners,lattice_indices
        from submap_disjoint.surface_deletion import IGSurfaceDeletion
        device=core.Device('CPU:0');voxel=.025;res=4
        grid=o3d.t.geometry.VoxelBlockGrid(('tsdf','weight','color'),(core.float32,)*3,(1,1,3),voxel,res,2,device)
        grid.hashmap().activate(core.Tensor([[0,0,10]],dtype=core.int32))
        grid.attribute('tsdf')[:]=.1;grid.attribute('weight')[:]=10;grid.attribute('color')[:]=100
        erased=corners(np.array([[0,0,1.]]),voxel).reshape(-1,3)
        coords=torch.tensor(np.vstack((erased,[[.075,.075,1.075]])),dtype=torch.float32)
        def indices(c):
            ids,ok=lattice_indices(grid,c.numpy(),voxel,res,device);assert ok.all()
            return torch.as_tensor(ids,dtype=torch.long)
        ids=indices(coords);table=[None]*128
        for i in ids.tolist():table[i]={0}
        flags=torch.zeros(128,dtype=torch.bool);flags[ids]=True
        vt=NS(hash_table=table,hash_item_flag=flags,voxel_coords2voxel_indices=indices)
        inst=NS(mask_voxel_coords=coords.clone(),mask_voxel_indices=ids.clone())
        graph=NS(instance_dict={0:inst},mask_voxels_coords={0:coords.clone()},
                 f_mask_voxels_clouds={(0,0):coords.clone()},voxel_in_frame_matrix=torch.ones((128,1),dtype=torch.bool),
                 boundary_voxel_indices=ids.clone(),merged_mask_in_frame_mat=torch.ones((1,1)))
        graph.query_mask_under_visible_part=lambda *a:(torch.empty(0,dtype=torch.long),torch.empty(0),torch.empty(0))
        owner=NS(cfg={'scene':{'voxel_size':voxel,'block_resolution':res}},device='cpu',device_o3c=device,
                 voxel_block_grids=grid,voxel_hash_table=vt,maskGraph=graph,global2merged_mask_id=[0],global_f_mask_ids=[(0,0)],
                 mask_voxels_coords={0:coords.clone()},merged_mask_size=[len(coords)],mask_geo_features=torch.ones((1,3)),
                 mask_geo_features_mask=torch.ones(1),merged_mask_bbox=torch.ones((1,6)),geometry=NS())
        owner.remove_masks=lambda *a:self.fail('Surviving point should keep this instance')
        cfg,K,pose=self.setup_depth();cfg['min_map_weight']=3
        bank=NS(points=np.array([[0,0,1.]]),streak=np.array([2]),support=np.array([3]),
                last_conflict_frame=np.array([10]),active=np.array([True]),last_frame=10,created_frame=np.array([0]),voxel_size=voxel)
        detector=NS(bank=bank,K=K,cfg=cfg,cam={'depth_near':.1,'depth_far':5.})
        with tempfile.TemporaryDirectory() as tmp:
            controller=IGSurfaceDeletion(owner,detector,tmp,DeletionOptions(min_component_points=1,verify_state=True),enabled=True)
            r=controller.step(10,np.full((11,11),2.),pose)
            self.assertEqual(r['voxels'],8)
            self.assertEqual(len(inst.mask_voxel_coords),1)
            self.assertEqual(len(graph.f_mask_voxels_clouds[(0,0)]),1)
            self.assertEqual(owner.merged_mask_size,[1])
            self.assertEqual(int(grid.attribute('weight').reshape((-1,))[core.Tensor([int(ids[-1])])].item()),10)
            self.assertEqual(controller.save()['unique_erased_world_voxels'],8)
            import json
            from pathlib import Path
            self.assertEqual(json.loads((Path(tmp)/'executor/summary.json').read_text())['voxel_erasure_operations'],8)

if __name__=='__main__':unittest.main()
