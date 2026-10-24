"""Exact sparse containment algebra. Zero entries are implicit; no top-K."""
from collections import defaultdict
from itertools import combinations
import sys
import numpy as np
import torch
from tool.geometric_helpers import nonlinear_mapping

class Edges:
    def __init__(self, values=None):
        self.values = {} if values is None else dict(values)

    def __getitem__(self, key):
        i,j=key
        return self.values.get((int(i),int(j)),0.0)

    def set(self, i, j, value):
        key=(int(i),int(j)); value=float(value)
        if value != 0: self.values[key]=value
        else: self.values.pop(key,None)

    def write_row_col(self, q, ids, forward, backward):
        # Callers use fresh IDs, hence absence means zero (no stale row).
        ids=ids.detach().cpu().tolist()
        fs=forward.detach().cpu().tolist()
        bs=backward.detach().cpu().tolist()
        for i,f,b in zip(ids,fs,bs):
            self.set(q,i,f); self.set(i,q,b)

    def subset(self, ids):
        if hasattr(ids,'tolist'): ids=ids.tolist()
        mapping={int(old):new for new,old in enumerate(ids)}
        return Edges({(mapping[i],mapping[j]):v for (i,j),v in self.values.items()
                      if i in mapping and j in mapping})

    def insert_block(self, block, ids):
        for (i,j),v in block.values.items(): self.set(ids[i],ids[j],v)

    def nbytes_estimate(self):
        # CPython dictionary, key tuple, integer referents, Python float.
        return sys.getsizeof(self.values)+sum(
            sys.getsizeof(k)+sum(sys.getsizeof(x) for x in k)+sys.getsizeof(v)
            for k,v in self.values.items())

    def dense_for_audit(self,n,device):
        out=torch.zeros((n,n),dtype=torch.float32,device=device)
        for (i,j),v in self.values.items():
            if i<n and j<n: out[i,j]=v
        return out

def sparse_clusters(pairs,n,edges,audit,mode='disjoint_pairs'):
    if mode == 'baseline_components':
        # Same undirected connected components as baseline do_clustering,
        # without materializing its dense adjacency matrix. Include isolates.
        parent=list(range(n))
        def find(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]
                i=parent[i]
            return i
        candidates=sorted(set(pairs))
        for i,j in candidates:
            ri,rj=find(i),find(j)
            if ri!=rj: parent[max(ri,rj)]=min(ri,rj)
        groups=defaultdict(list)
        for i in range(n): groups[find(i)].append(i)
        result=sorted((np.asarray(g,dtype=np.int64) for g in groups.values()),key=lambda g:int(g[0]))
        merged=[g for g in result if len(g)>1]
        audit['merge_calls']+=1
        audit['candidate_edges']+=len(candidates)
        audit['multi_merge_groups']=audit.get('multi_merge_groups',0)+len(merged)
        audit['merged_instances']=audit.get('merged_instances',0)+sum(len(g) for g in merged)
        return result
    if mode != 'disjoint_pairs':
        raise ValueError('Unknown merge selection mode: '+str(mode))
    # Match row-major torch.nonzero tie order and float32 addition.
    ordered=sorted(set(pairs),key=lambda p:(
        -float(np.float32(np.float32(edges[p])+np.float32(edges[p[1],p[0]]))),p[0],p[1]))
    used=set(); groups=[]
    for i,j in ordered:
        if i!=j and i not in used and j not in used:
            groups.append(np.array(sorted((i,j)),dtype=np.int64)); used.update((i,j))
    audit['merge_calls']+=1
    audit['candidate_edges']+=len(ordered)
    audit['selected_pairs']+=len(groups)
    groups.extend(np.array([i],dtype=np.int64) for i in range(n) if i not in used)
    return sorted(groups,key=lambda x:int(x[0]))

def sparse_scores(owner,pairs,use_overlap=True):
    if not pairs: return np.empty(0,np.float32)
    # Chunk candidate descriptors: workspace O(chunk_size * feature_dim).
    sem=owner.get_mask_features
    geo=owner.get_mask_geo_features if owner.fcgf_diag_use_in_merge else torch.zeros_like(owner.get_mask_geo_features)
    sem=sem/(sem.norm(dim=-1,keepdim=True)+1e-7)
    geo=geo/(geo.norm(dim=-1,keepdim=True)+1e-7)
    results=[]
    for start in range(0,len(pairs),4096):
        part=pairs[start:start+4096]
        ij=torch.tensor(part,device=sem.device,dtype=torch.long)
        ss=(sem[ij[:,0]]*sem[ij[:,1]]).sum(-1)
        gs=(geo[ij[:,0]]*geo[ij[:,1]]).sum(-1)
        values=nonlinear_mapping(ss,min_value=owner.metrics.seman_min_value,max_value=1.)+gs
        if use_overlap:
            ab=values.new_tensor([owner.edges[i,j] for i,j in part])
            ba=values.new_tensor([owner.edges[j,i] for i,j in part])
            values=values+(ab+ba)/2
            adjust = getattr(owner, 'adjust_candidate_scores', None)
            if adjust is not None:
                values = adjust(values, ij, ab, ba)
        results.append(values.detach().cpu().numpy())
    return np.concatenate(results)

def select_initial(owner):
    e=owner.edges
    valid=set(owner.get_mask_w_geo_feature.cpu().tolist())
    mutual=sorted((i,j) for (i,j),v in e.values.items()
                  if i>j and i in valid and j in valid and
                  v>owner.merge_overlap_thresh and e[j,i]>owner.merge_overlap_thresh)
    scores=sparse_scores(owner,mutual)
    selected={p for p,s in zip(mutual,scores) if s>owner.sim_merge_thresh}
    # A[i,k] means k supports i. A @ diag(weight) @ A.T is a
    # weighted common-neighbor join, including pairs with C[i,j]==0.
    supporters=defaultdict(list)
    for (i,k),v in e.values.items():
        if i!=k and v>owner.contained_ratio and e[k,i]>owner.containing_ratio:
            supporters[k].append(i)
    weights=owner.get_merged_mask_weight.detach().cpu().numpy()
    totals=defaultdict(float)
    for k,neighbors in sorted(supporters.items()):
        for a,b in combinations(sorted(set(neighbors)),2):
            if a in valid and b in valid:
                totals[(b,a)]+=float(weights[k])
    selected.update(p for p,v in totals.items() if v>=owner.merge_supporter_num)
    owner.sparse_audit['direct_candidates']+=len(mutual)
    owner.sparse_audit['supporter_candidates']+=len(totals)
    owner.sparse_audit['supporter_only_candidates']+=sum(e[p]==0 and e[p[1],p[0]]==0 for p in totals)
    owner.sparse_audit['possible_dense_pairs']+=owner.c_mask_num*(owner.c_mask_num-1)//2
    return sorted(selected)

def select_containment(owner,valid_ids):
    valid=set(valid_ids.cpu().tolist())
    best={}
    for (i,j),v in owner.edges.values.items():
        if i!=j and i in valid and j in valid and v>owner.merge_contain_ratio:
            if j not in best or v>best[j][1] or (v==best[j][1] and i<best[j][0]):
                best[j]=(i,v)
    geo=set(owner.get_mask_w_geo_feature.cpu().tolist())
    # The reference first thresholds C at merge_contain_ratio, then retains
    # max per column and ORs feature gates on that SAME retained matrix.
    # The feature gates add no new entries under its current positive thresholds.
    return sorted((i,j) for j,(i,v) in best.items() if i in geo and j in geo)

def compute_sparse_containment(graph,merged_mask_ids,id_mapping):
    """Preserve compute_masks_contain_ratio, including transpose and no clamp."""
    ids=[int(i) for i in merged_mask_ids]
    lookup={mid:i for i,mid in enumerate(ids)}
    result=Edges()
    for source in ids:
        inst=graph.instance_dict[source]
        overlap,counts,_=graph.voxel_hash_table.query_mask_w_mapping(
            inst.mask_voxel_coords,id_mapping,ids)
        if not overlap.numel(): continue
        keep=torch.tensor([int(i) in lookup for i in overlap],device=overlap.device)
        overlap=overlap[keep]; counts=counts[keep]
        if not overlap.numel(): continue
        frames=graph.merged_mask_in_frame_mat[overlap]
        vf=graph.voxel_in_frame_matrix[inst.mask_voxel_indices].float().clamp(max=1.)
        vis=(vf@frames.T).clamp(max=1.).sum(0)
        ratio=counts.float()/vis
        condition=((vis/len(inst.mask_voxel_indices)>graph.mask_visible_threshold) |
                   (vis>graph.mask_visible_min_overlap)) & (ratio>graph.contain_min_threshold)
        values=torch.where(condition,ratio,torch.zeros_like(ratio))
        # Original method transposes source-denominator values.
        for target,value in zip(overlap.cpu().tolist(),values.cpu().tolist()):
            result.set(lookup[int(target)],lookup[source],value)
    return result

