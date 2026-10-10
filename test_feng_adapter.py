import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from feng_ig_adapter import exclusive_cells,resolve_slots,FengIGCorrection

class ActionTests(unittest.TestCase):
    def test_lineage_ignores_stale_negative_and_unknown_slots(self):
        self.assertEqual(resolve_slots({0,1,2,3,99},[-1,5,5,8],{5:object()}),[5])
    def test_shared_and_negative_lattice_nodes_are_preserved(self):
        out=exclusive_cells([[-.025,0,0],[0,0,0],[.025,0,0],[.025,0,0]],[[0,0,0]],.025)
        np.testing.assert_allclose(out,[[-.025,0,0],[.025,0,0]])
    def test_no_live_descendant_does_not_delete_an_unrelated_slot(self):
        x=object.__new__(FengIGCorrection);x.actions=[]
        x.owner=NS(maskGraph=NS(instance_dict={}),global2merged_mask_id=[-1])
        x.delete_candidate(9,NS(candidate_id=1,source_slot=7,source_raw_ids={0}))
        self.assertEqual(x.actions[0]['status'],'no_live_descendant')
    def test_erase_slot_compaction_is_followed_by_track_uid(self):
        with tempfile.TemporaryDirectory() as d:
            x=object.__new__(FengIGCorrection);x.actions=[];x.refs={};x.out=Path(d);x.voxel=.025
            def inst(uid,coords):return NS(track_uid=uid,mask_voxel_coords=torch.tensor(coords,dtype=torch.float32))
            live={0:inst(10,[[0,0,0],[.025,0,0]]),1:inst(20,[[0,0,0],[.05,0,0]])}
            removed=[];o=NS(maskGraph=NS(instance_dict=live),global2merged_mask_id=[0,1],device='cpu')
            o.remove_masks=lambda frame,ids:removed.extend(ids.tolist());x.owner=o
            def erase(frame,requests):
                np.testing.assert_allclose(requests[0][1],[[.025,0,0]])
                # Simulate slot renumbering while an unrelated instance survives.
                o.maskGraph.instance_dict={0:inst(20,[[0,0,0]]),1:inst(10,[[0,0,0]])}
                return dict(voxels=1)
            x.erase=erase
            x.delete_candidate(11,NS(candidate_id=2,source_slot=99,source_raw_ids={0}))
            self.assertEqual(removed,[1]);self.assertEqual(x.actions[0]['shared_nodes_preserved'],1)

if __name__=='__main__':unittest.main()
