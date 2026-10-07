import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from submap_disjoint.it_adapter import InstanceTSDFAdapter
from submap_disjoint.independent_tsdf import IndependentTSDFStore

class FactorWiringTests(unittest.TestCase):
    def test_rejected_pixels_do_not_reenter_background(self):
        cfg={'scene':{'voxel_size':.025,'block_resolution':4,'block_count':30000},
             'cam':{'depth_near':.1,'depth_far':5}}
        owner=SimpleNamespace(device='cuda:0',intrinsics=torch.tensor(
            [[100.,0,31.5],[0,100.,31.5],[0,0,1.]]),
            global2merged_mask_id=[0],merge_frame_interval=50)
        rgb=torch.ones((64,64,3))*.4; depth=torch.ones((64,64))*2.013
        refined=torch.zeros((64,64),dtype=torch.int32);refined[:,:32]=1
        excluded=torch.zeros((64,64),dtype=torch.bool);excluded[:,32:40]=True
        pairs=[{'mask_id':1,'global_mask_id':0}]
        with tempfile.TemporaryDirectory() as d:
            adapter=InstanceTSDFAdapter(cfg,owner,d)
            adapter.observe(0,rgb,depth,torch.eye(4),refined,pairs,excluded)
            self.assertTrue(torch.all(depth==2.013))  # shared caller data unchanged
            self.assertEqual(adapter.audit['excluded_depth_pixels'],512)
            adapter.flush(0)
            pixels={}
            for record in adapter.fragments:
                restored=IndependentTSDFStore.load(Path(d)/record['key'])
                fragment=restored.fragments[record['key']]
                pixels[record['kind']]=int(fragment.observations[0]['pixels'])
            self.assertEqual(pixels,{'instance':2048,'background':1536})
            self.assertEqual(sum(pixels.values()),4096-512)

if __name__=='__main__':unittest.main()
