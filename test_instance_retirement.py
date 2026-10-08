import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS, MethodType
import numpy as np
sys.path.append('third_party/FCGF')
from submap_disjoint.instance_retirement import (RetirementOptions, resolve_frozen_track,
    decision, DepthContext, retire_track)


class RetirementPolicyTests(unittest.TestCase):
    def test_row_compaction_is_not_identity_change(self):
        self.assertEqual(resolve_frozen_track({1,2},[8,0,0]),(0,'stable'))
        self.assertEqual(resolve_frozen_track({1,2},[0,0,0])[1],'lineage_expanded')
        self.assertEqual(resolve_frozen_track({1,2},[0,-1,-1])[1],'already_removed')
        self.assertEqual(resolve_frozen_track({1,2},[0,1,2])[1],'split_or_partially_removed')

    def policy(self,confirmed,supported=0,current=20,anchored=100):
        def flags(n): return np.arange(100)<n
        return decision(flags(confirmed),flags(supported),flags(current),flags(anchored),RetirementOptions())

    def test_local_patch_does_not_authorize_whole_object(self):
        ok,s=self.policy(20)
        self.assertFalse(ok);self.assertIn('insufficient_whole_instance_coverage',s['reasons'])

    def test_sufficient_coverage_passes(self):
        self.assertTrue(self.policy(70)[0])

    def test_surviving_surface_blocks_mixed_instance(self):
        ok,s=self.policy(70,supported=20)
        self.assertFalse(ok);self.assertIn('recent_surface_support',s['reasons'])

    def test_unknown_and_stale_evidence_cannot_confirm(self):
        self.assertFalse(self.policy(0)[0])
        self.assertFalse(self.policy(70,current=0)[0])
        self.assertFalse(self.policy(70,anchored=40)[0])

    def test_depth_support_free_occlusion_and_missing(self):
        cfg=dict(edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,depth_relative_epsilon=.01)
        K=np.array([[100,0,5],[0,100,5],[0,0,1.]])
        ctx=DepthContext(np.full((11,11),2.),np.eye(4),K,cfg,.1,5.,.05)
        np.testing.assert_array_equal(ctx.classify(np.array([[0,0,1.],[0,0,2.],[0,0,3.],[10,0,1.]])),[2,1,3,0])

    def test_invalid_options(self):
        with self.assertRaises(ValueError): RetirementOptions(min_confirmed_fraction=1.1)


def fixture():
    """Actual Open3D TSDF, actual owner.remove_masks and graph compaction methods."""
    import torch
    import open3d as o3d
    import open3d.core as core
    from submap_disjoint.sparse_owner import SparseDisjointSubmapManager
    from submap_disjoint.evidence_graph import SubmapEvidenceGraph
    from submap_disjoint.sparse_edges import Edges
    from submap_disjoint.ig_correction import lattice_indices
    voxel=.025;res=4;device=core.Device('CPU:0')
    grid=o3d.t.geometry.VoxelBlockGrid(('tsdf','weight','color'),(core.float32,)*3,(1,1,3),voxel,res,2,device)
    grid.hashmap().activate(core.Tensor([[0,0,10]],dtype=core.int32))
    grid.attribute('weight')[:]=10.;grid.attribute('tsdf')[:]=.1;grid.attribute('color')[:]=100.
    points=torch.tensor([[0,0,1],[.025,0,1],[.05,0,1],[.075,0,1]],dtype=torch.float32)
    def indices(x):
        ids,ok=lattice_indices(grid,x.numpy(),voxel,res,device);assert ok.all()
        return torch.tensor(ids,dtype=torch.long)
    ids=indices(points);table=[None]*128
    table[int(ids[0])]={0};table[int(ids[1])]={0};table[int(ids[2])]={0,1};table[int(ids[3])]={1}
    flags=torch.zeros(128,dtype=torch.bool);flags[ids]=True
    vt=NS(hash_table=table,hash_item_flag=flags,voxel_coords2voxel_indices=indices)
    o=object.__new__(SparseDisjointSubmapManager)
    o.cfg={'scene':{'voxel_size':voxel,'block_resolution':res},'mask':{'feature_dim':3},'pc_extractor':{'feature_dim':2}}
    o.device='cpu';o.device_o3c=device;o.voxel_block_grids=grid;o.voxel_hash_table=vt;o.geometry=NS()
    o.c_mask_num=2;o.t_mask_num=4;o.global2merged_mask_id=[0,1];o.global_f_mask_ids=[(0,0),(0,1)]
    o.mask_voxels_coords={0:points[:3].clone(),1:points[2:].clone()};o.merged_mask_size=[3,2]
    o.mask_features=torch.arange(12,dtype=torch.float32).reshape(4,3)
    o.mask_geo_features=torch.ones((4,2));o.mask_geo_features_mask=torch.ones(4)
    o.merged_mask_weight=torch.tensor([6.,7.,0.,0.]);o.merged_mask_last_frame=torch.tensor([5,6,-1,-1])
    o.merged_mask_bbox=torch.ones((4,6));o.edges=Edges({(0,0):1.,(1,1):1.,(0,1):.5})
    g=object.__new__(SubmapEvidenceGraph);g.device='cpu';g.seg_frame_num=2
    g.instance_dict={i:NS(mask_id=i,mask_voxel_coords=p.clone(),mask_voxel_indices=indices(p)) for i,p in o.mask_voxels_coords.items()}
    g.mask_voxels_coords=dict(o.mask_voxels_coords)
    g.f_mask_voxels_clouds={(0,i):p.clone() for i,p in o.mask_voxels_coords.items()}
    g.merged_mask_in_frame_mat=torch.tensor([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
    g.voxel_in_frame_matrix=torch.ones((128,2),dtype=torch.bool);g.boundary_voxel_indices=ids[2:3].clone()
    g.query_mask_under_visible_part=lambda *a:(torch.empty(0,dtype=torch.long),torch.empty(0),torch.empty(0))
    o.maskGraph=g
    return o,ids,points


class RetirementMutationTests(unittest.TestCase):
    def test_controller_waits_for_multiview_confirmation_then_retires(self):
        from submap_disjoint.instance_retirement import IGInstanceRetirement
        o,ids,p=fixture()
        cfg=dict(edge_threshold_m=.08,boundary_margin_px=0,depth_epsilon_m=.08,depth_relative_epsilon=.01,
                 confirm_observations=2,min_map_weight=3,min_instance_weight=5)
        bank=NS(points=p.numpy(),support=np.full(4,3),streak=np.array([1,1,0,0]),
                active=np.zeros(4,bool),last_conflict_frame=np.array([5,5,-1,-1]),last_frame=5,voxel_size=.025)
        detector=NS(bank=bank,K=np.array([[100,0,5],[0,100,5],[0,0,1.]]),cfg=cfg,cam=dict(depth_near=.1,depth_far=5.))
        options=RetirementOptions(min_reference_points=3,max_reference_points=3,min_confirmed_points=2,max_shared_fraction=.5,verify_state=True)
        with tempfile.TemporaryDirectory() as tmp:
            controller=IGInstanceRetirement(o,detector,tmp,options,enabled=True)
            self.assertEqual(controller.step(5,np.full((11,11),2.),np.eye(4)),[])
            self.assertEqual(o.global2merged_mask_id,[0,1])
            bank.streak[:2]=2;bank.active[:2]=True;bank.last_conflict_frame[:2]=10;bank.last_frame=10
            events=controller.step(10,np.full((11,11),2.),np.eye(4))
            self.assertEqual(len(events),1);self.assertEqual(o.global2merged_mask_id,[-1,0])
            self.assertEqual(controller.save()['retired_instances'],1)

    def test_exclusive_geometry_erased_shared_survives_and_caches_removed(self):
        import open3d.core as core
        from submap_disjoint.ig_deletion import IGDeletion
        o,ids,p=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            ctx=NS(classify=lambda points:np.full(len(points),2,np.uint8))
            result=retire_track(o,IGDeletion(o,None,Path(tmp)/'executor'),{0},10,ctx,Path(tmp)/'archive',.5,True)
            self.assertTrue(result['retired']);self.assertEqual(result['erased_corners'],2)
            self.assertEqual(o.global2merged_mask_id,[-1,0]);self.assertEqual(o.c_mask_num,1)
            self.assertEqual(o.voxel_hash_table.hash_table[int(ids[2])],{1})
            self.assertEqual(len(o.maskGraph.f_mask_voxels_clouds[(0,0)]),0)
            self.assertEqual(o.merged_mask_size,[2])
            np.testing.assert_array_equal(o.mask_features[0].numpy(),[3,4,5])
            weights=o.voxel_block_grids.attribute('weight').reshape((-1,))[core.Tensor(ids.numpy())].numpy()
            np.testing.assert_array_equal(weights,[0,0,10,10])
            self.assertTrue(o.maskGraph.voxel_in_frame_matrix[ids[2]].all())
            self.assertEqual(set(o.edges.values),{(0,0)})

    def test_current_surface_preserved_and_shared_fraction_can_veto(self):
        import open3d.core as core
        from submap_disjoint.ig_deletion import IGDeletion
        o,ids,p=fixture()
        with tempfile.TemporaryDirectory() as tmp:
            executor=IGDeletion(o,None,Path(tmp)/'executor')
            ctx=NS(classify=lambda points:np.ones(len(points),np.uint8))
            denied=retire_track(o,executor,{0},10,ctx,Path(tmp)/'archive',.1,True)
            self.assertFalse(denied['retired']);self.assertEqual(o.global2merged_mask_id,[0,1])
            result=retire_track(o,executor,{0},11,ctx,Path(tmp)/'archive',.5,True)
            self.assertTrue(result['retired']);self.assertEqual(result['erased_corners'],0)
            weights=o.voxel_block_grids.attribute('weight').reshape((-1,))[core.Tensor(ids.numpy())].numpy()
            np.testing.assert_array_equal(weights,[10,10,10,10])

    def test_retired_raw_ids_do_not_authorize_new_observation(self):
        o,ids,p=fixture();o.global2merged_mask_id.append(0)
        with tempfile.TemporaryDirectory() as tmp:
            result=retire_track(o,None,{0},10,None,tmp,.5,True)
            self.assertFalse(result['retired']);self.assertEqual(result['reason'],'lineage_expanded')


if __name__=='__main__': unittest.main()
