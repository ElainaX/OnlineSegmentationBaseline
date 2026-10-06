"""Check actual visibility/overlap routines on identical history, CPU and CUDA."""
import sys
from types import SimpleNamespace
sys.path.insert(0, '/root/OnlineSegmentationBaseline_baseline_ablation')
import torch
from submap_disjoint.evidence_graph import SubmapEvidenceGraph
from submap_disjoint.sparse_edges import compute_sparse_containment

class Lookup:
    def __init__(self, members): self.members=members
    def query_mask_w_mapping(self, coords, mapping, valid=None):
        ids=coords[:,0].long(); allowed=list(self.members) if valid is None else valid
        overlaps=[];counts=[]
        for j in allowed:
            count=torch.isin(ids,self.members[j]).sum().item()
            if count: overlaps.append(j);counts.append(count)
        return torch.tensor(overlaps,dtype=torch.long,device=ids.device),torch.tensor(counts,device=ids.device),ids

for device in ['cpu']+(['cuda:0'] if torch.cuda.is_available() else []):
    for seed in range(5):
        torch.manual_seed(seed)
        history=torch.randint(0,300,(100,13),device=device).float()
        history[torch.rand_like(history)<.7]=0
        history[0]=0  # never observed
        history[:,0]=0  # empty observation
        history[1,1]=100000 # large positive IDs must still be valid visibility
        members={i:torch.randperm(100,device=device)[:30] for i in range(6)}
        frames=torch.randint(0,2,(6,13),device=device).float()
        def graph(storage):
            g=SubmapEvidenceGraph.__new__(SubmapEvidenceGraph)
            g.device=device;g.voxel_in_frame_matrix=storage
            g.merged_mask_in_frame_mat=frames
            g.mask_visible_threshold=.2;g.mask_visible_min_overlap=5;g.contain_min_threshold=.025
            g.voxel_hash_table=Lookup(members)
            g.instance_dict={i:SimpleNamespace(mask_voxel_indices=v,mask_voxel_coords=v[:,None]) for i,v in members.items()}
            return g
        a=graph(history);b=graph(history.bool())
        for i in members:
            args=(frames[i],members[i][:,None],list(members))
            for left,right in zip(a.query_mask_under_visible_part(*args),b.query_mask_under_visible_part(*args)):
                torch.testing.assert_close(left,right,rtol=0,atol=0,equal_nan=True)
        args=(list(members),list(members))
        torch.testing.assert_close(a.compute_masks_contain_ratio(*args),b.compute_masks_contain_ratio(*args),rtol=0,atol=0,equal_nan=True)
        sa=compute_sparse_containment(a,*args);sb=compute_sparse_containment(b,*args)
        assert sa.values==sb.values
    print(device,'PASS: query, dense containment, sparse containment; 5 seeded fixtures, exact values')
print('Storage: float32 4 bytes/cell -> bool 1 byte/cell; no frame truncation')
