"""Ablation invariants: immediate trigger, shared protection and no whole L retirement."""
import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'adapters')]
from feng_remove_detection import Candidate,CONTRADICTION,OCCLUDED,SOURCE_DEPTH
from immediate_confirmation import observe as immediate

class AblationContracts(unittest.TestCase):
    def test_same_detector_parameters_and_distinct_switches(self):
        cfg=json.loads((ROOT/'configs/detector_H.json').read_text())
        for name in ['F0','O0','L']:
            self.assertEqual(cfg,json.loads((ROOT/f'configs/detector_{name}.json').read_text()))
        m=json.loads((ROOT/'configs/ablations.json').read_text())
        self.assertEqual(m['f0']['correction'],'off');self.assertTrue(m['f0']['refine'])
        self.assertEqual(m['o0']['correction'],'delete');self.assertFalse(m['o0']['refine'])
        self.assertEqual(m['l']['env_method'],'L');self.assertTrue(m['t1-any']['cuda_launch_blocking'])

    def test_immediate_one_owned_pixel_not_aggregate_threshold(self):
        cfg=json.loads((ROOT/'configs/detector_H.json').read_text())
        c=Candidate(1,frozenset({0}),0,0,np.ones((200,3)),np.ones((200,3)))
        codes=np.full(200,OCCLUDED,np.uint8);sources=np.zeros(200,np.uint8)
        # Candidate reference sampling need not contain the one valid owned trigger.
        reason,stats=immediate(c,0,np.eye(4),codes,sources,cfg,trigger_stats={'contradiction':1})
        self.assertEqual(reason,'would_remove');self.assertEqual(c.hits,1)
        self.assertEqual(c.confirmed_frame,0)

    def test_immediate_without_contradiction_is_not_confirmed(self):
        cfg=json.loads((ROOT/'configs/detector_H.json').read_text())
        c=Candidate(1,frozenset({0}),0,0,np.ones((200,3)),np.ones((200,3)))
        state,_=immediate(c,0,np.eye(4),np.full(200,OCCLUDED,np.uint8),np.zeros(200,np.uint8),cfg)
        self.assertEqual(state,'no_valid_contradiction')

    def test_local_negative_cells_and_shared_protection(self):
        from local_erasure import local_cells
        got=local_cells([[-.025,0,0],[0,0,0],[.025,0,0],[.1,0,0]],[[0,0,0]],[[-.012,.001,.001],[.101,.001,.001]],[True,False],.025)
        np.testing.assert_allclose(got,[[-.025,0,0]])

    def test_local_eraser_keeps_remaining_instance(self):
        import torch
        from local_erasure import delete_local
        with tempfile.TemporaryDirectory() as d:
            live={2:NS(mask_voxel_coords=torch.tensor([[0.,0,0],[.1,0,0]]))}
            def forbidden(*args):raise AssertionError('L must not force whole-instance retirement')
            owner=NS(maskGraph=NS(instance_dict=live),global2merged_mask_id=[2],remove_masks=forbidden)
            calls=[]
            def erase(frame,requests):
                calls.append(requests);np.testing.assert_allclose(requests[0][1],[[0,0,0]])
                live[2].mask_voxel_coords=live[2].mask_voxel_coords[1:]
                return {'voxels':1,'emptied_current_ids':[]}
            obj=NS(owner=owner,out=Path(d),voxel=.025,refs={},actions=[],erase=erase)
            c=NS(candidate_id=1,source_slot=99,source_raw_ids={0},points=np.array([[.001,.001,.001]]),contradicted=np.array([True]))
            delete_local(obj,10,c)
            self.assertEqual(len(calls),1);self.assertIn(2,live)
            self.assertFalse(obj.actions[0]['forced_whole_instance_retirement'])

if __name__=='__main__':unittest.main()
