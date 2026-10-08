import unittest
import numpy as np
from submap_disjoint.ig_correction import unique_owner, free_now, confirmed_free, corners
from submap_disjoint.change_frontend import FrontendOptions
from submap_disjoint.change_lifecycle import EvidenceWindow, LifecycleOptions
from submap_disjoint.contradiction import FREE, SUPPORT, OCCLUDED

class CorrectionTests(unittest.TestCase):
    def test_ownership_rejects_overlap_not_unknown(self):
        a=np.array([[1,1,-1],[1,2,-1],[1,-2,-1],[-1,-1,-1]])
        np.testing.assert_equal(unique_owner(a,np.ones_like(a,bool)),[1,-2,-2,-1])

    def test_views_and_occlusion(self):
        opt=LifecycleOptions(); w=EvidenceWindow(opt); pose=np.eye(4)
        for i in range(5):w.update(i,pose,np.array([FREE, SUPPORT],np.uint8))
        self.assertFalse(confirmed_free(w,2,opt).any())
        for j in range(2):
            pose=pose.copy();pose[0,3]+=.2;w.update(6+j,pose,np.array([FREE,OCCLUDED],np.uint8))
        np.testing.assert_equal(confirmed_free(w,2,opt),[True,False])
        w.update(100,pose,np.array([OCCLUDED,OCCLUDED],np.uint8))
        self.assertFalse(confirmed_free(w,2,opt).any())

    def test_only_free_can_be_deleted(self):
        K=np.array([[20,0,10],[0,20,10],[0,0,1.]])
        pts=np.array([[0,0,1.],[0,0,2.],[0,0,3.]])
        d=np.full((21,21),2.)
        np.testing.assert_equal(free_now(pts,d,K,np.eye(4),FrontendOptions(),.1,5),[True,False,False])
        d[10,10]=0
        self.assertFalse(free_now(pts,d,K,np.eye(4),FrontendOptions(),.1,5).any())

    def test_negative_world_lattice_corners(self):
        c=corners(np.array([[-.012,.012,.04]]),.025)
        self.assertEqual(c.shape,(1,8,3))
        np.testing.assert_allclose(c.min(1),[[-.025,0,.025]])
        np.testing.assert_allclose(c.max(1),[[0,.025,.05]])

    def test_actual_weight_erasure_and_index_consistency(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace as NS
        import torch
        import open3d as o3d
        import open3d.core as core
        from submap_disjoint.ig_correction import IGCorrection, lattice_indices, frozen_grid_copy
        device=core.Device('CPU:0')
        grid=o3d.t.geometry.VoxelBlockGrid(('tsdf','weight','color'),(core.float32,)*3,
            (1,1,3),.025,4,10,device)
        grid.hashmap().activate(core.Tensor([[0,0,0]],dtype=core.int32))
        grid.attribute('weight')[:]=5.;grid.attribute('tsdf')[:]=.2;grid.attribute('color')[:]=100.
        frozen=frozen_grid_copy(grid,.025,4,device)
        coords=torch.tensor([[0.,0,0],[.025,0,0],[.050,0,0]])
        def indices(x):
            a,b=lattice_indices(grid,x.numpy(),.025,4,device)
            assert b.all()
            return torch.tensor(a)
        ids=indices(coords)
        inst=NS(mask_voxel_coords=coords.clone(),mask_voxel_indices=ids.flip(0))
        graph=NS(instance_dict={0:inst},mask_voxels_coords={0:coords.clone()},
            voxel_in_frame_matrix=torch.ones((640,3),dtype=torch.bool),
            boundary_voxel_indices=ids.clone(),merged_mask_in_frame_mat=torch.ones((1,3)),
            query_mask_under_visible_part=lambda *a:(torch.zeros(0,dtype=torch.long),None,None))
        table=NS(voxel_coords2voxel_indices=indices,hash_table=[{0} for _ in range(640)],hash_item_flag=torch.ones(640,dtype=torch.bool))
        owner=NS(voxel_block_grids=grid,device_o3c=device,device='cpu',maskGraph=graph,
            voxel_hash_table=table,mask_voxels_coords={0:coords.clone()},merged_mask_size=[3],
            mask_geo_features=torch.ones((1,16)),mask_geo_features_mask=torch.ones(1),
            global2merged_mask_id=[0],geometry=NS())
        with tempfile.TemporaryDirectory() as td:
            ctrl=IGCorrection.__new__(IGCorrection)
            ctrl.owner=owner;ctrl.voxel=.025;ctrl.res=4;ctrl.archive=Path(td)
            ctrl.refs={0:{'deleted_voxels':0}};ctrl.deleted=set();ctrl.events=[]
            result=ctrl.erase(1,[(0,coords[:1].numpy())])
            self.assertEqual(result['voxels'],1)
            self.assertEqual(float(grid.attribute('weight').reshape((-1,))[int(ids[0])].item()),0.)
            frozen_ids,_=lattice_indices(frozen,coords.numpy(),.025,4,device)
            self.assertEqual(float(frozen.attribute('weight').reshape((-1,))[int(frozen_ids[0])].item()),5.)
            torch.testing.assert_close(inst.mask_voxel_coords,coords[1:])
            torch.testing.assert_close(inst.mask_voxel_indices,ids[1:])
            self.assertIsNone(table.hash_table[int(ids[0])])
            self.assertFalse(graph.voxel_in_frame_matrix[ids[0]].any())
            self.assertTrue(graph.voxel_in_frame_matrix[ids[1]].all())

if __name__=='__main__':unittest.main()
