import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import open3d as o3d
from submap_disjoint.it_adapter import (group_observations,resolve_fragment,
                                       coalesce_surfaces,InstanceTSDFAdapter)
from submap_disjoint.independent_tsdf import IndependentTSDFStore


class AdapterTests(unittest.TestCase):
    def test_removed_masks_do_not_become_one_identity(self):
        packets=[{'pairs':[(1,0),(2,1),(3,2)]},{'pairs':[(4,3)]}]
        groups=group_observations(packets,[-1,-1,7,7])
        self.assertEqual(len(groups[('instance',7)]),2)
        self.assertIn(('orphan',0),groups); self.assertIn(('orphan',1),groups)

    def test_identity_remapping_and_split_guard(self):
        self.assertEqual(resolve_fragment([0,2],[5,3,5]),5)
        self.assertEqual(resolve_fragment([0,2],[-1,3,-1]),-1)
        self.assertEqual(resolve_fragment([],[]),-1)
        with self.assertRaises(RuntimeError): resolve_fragment([0,2],[5,3,6])

    def test_export_inverse_preserves_overlapping_support(self):
        p=np.array([[.001,0,0],[.002,0,0],[-.001,0,0]])
        q,c,inv=coalesce_surfaces(p,np.ones((3,3))*128,.025)
        self.assertEqual(len(q),2)
        self.assertEqual(inv[0],inv[1]); self.assertNotEqual(inv[0],inv[2])
        np.testing.assert_allclose(c,128)
        np.testing.assert_array_equal(p,np.array([[.001,0,0],[.002,0,0],[-.001,0,0]]))

    def test_causal_flush_union_and_restore(self):
        cfg={'scene':{'voxel_size':.025,'block_resolution':4,'block_count':30000},
             'cam':{'depth_near':.1,'depth_far':5},'seg':{'mask_weight_threshold':2},
             'mask':{'feature_dim':4}}
        # Pixel footprint at 2 m is 20 mm, below the 25 mm fusion grid.
        K=torch.tensor([[100.,0,31.5],[0,100.,31.5],[0,0,1.]])
        feature=SimpleNamespace(get_semantic_feature=torch.ones(4))
        owner=SimpleNamespace(device='cuda:0',intrinsics=K,global2merged_mask_id=[0,0,0],
            merge_frame_interval=2,merge_time=0,c_mask_num=1,merged_mask_weight=torch.tensor([3.]),
            maskGraph=SimpleNamespace(instance_dict={0:feature}))
        def save_pc(p,c,path):
            pc=o3d.geometry.PointCloud(); pc.points=o3d.utility.Vector3dVector(p)
            pc.colors=o3d.utility.Vector3dVector(c);o3d.io.write_point_cloud(path,pc)
        owner.save_pc=save_pc
        rgb=torch.ones((64,64,3))*.4; depth=torch.ones((64,64))*2.013
        seg=torch.zeros((64,64),dtype=torch.int32);seg[:,8:32]=1;seg[:,32:56]=2
        pairs=[{'mask_id':1,'global_mask_id':0},{'mask_id':2,'global_mask_id':1}]
        with tempfile.TemporaryDirectory() as d:
            adapter=InstanceTSDFAdapter(cfg,owner,Path(d)/'fragments')
            adapter.observe(0,rgb,depth,torch.eye(4),seg,pairs)
            self.assertEqual(len(adapter.fragments),0)
            adapter.observe(2,rgb,depth,torch.eye(4),seg,[{'mask_id':1,'global_mask_id':2}])
            self.assertEqual(len(adapter.fragments),2)
            target=next(f for f in adapter.fragments if f['kind']=='instance')
            restored=IndependentTSDFStore.load(Path(d)/'fragments'/target['key'])
            fragment=restored.fragments[target['key']]
            self.assertEqual(len(fragment.observations),2)
            self.assertEqual(fragment.observations[0]['pixels'],64*48)
            self.assertEqual(fragment.status,'frozen')
            report=adapter.export(2,d)
            self.assertFalse(report['geometry_fallback'])
            self.assertGreater(report['audit']['export_points'],200)
            arr=np.load(Path(d)/'ckpt_final.npz')
            self.assertEqual(arr['pred_masks'].shape[0],report['audit']['export_points'])
            self.assertEqual(arr['pred_masks'].shape[1],1)


if __name__=='__main__': unittest.main()
