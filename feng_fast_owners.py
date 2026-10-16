"""Equivalent owner reduction; no detector or deletion policy changes."""
import json,time
from pathlib import Path
import numpy as np

def reduce_owners(indices,neighbor_valid,ratios,memberships):
    shape=indices.shape[:-1];ids=np.asarray(indices).reshape(-1,indices.shape[-1])
    useful=np.asarray(neighbor_valid).reshape(ids.shape)&(np.asarray(ratios).reshape(ids.shape)>1e-6)&(ids>=0)
    keys,inverse=np.unique(ids[useful],return_inverse=True)
    groups=memberships(keys)
    sentinel=np.iinfo(np.int64).max
    lower=np.fromiter((min(m) if m else sentinel for m in groups),dtype=np.int64,count=len(keys))
    upper=np.fromiter((max(m) if m else -1 for m in groups),dtype=np.int64,count=len(keys))
    lows=np.full(ids.shape,sentinel,dtype=np.int64);highs=np.full(ids.shape,-1,dtype=np.int64)
    lows[useful]=lower[inverse];highs[useful]=upper[inverse]
    lo=lows.min(axis=1);hi=highs.max(axis=1)
    return np.where(hi<0,-1,np.where(lo==hi,lo,-2)).reshape(shape)

def install(directory):
    import feng_remove_detection as module
    original=module.unambiguous_owners
    audit={'calls':0,'checked_calls':[],'verification_seconds':0.,'mismatches':0}
    path=Path(directory).parent/(Path(directory).name+'.owner_equivalence.json')
    def optimized(indices,neighbor_valid,ratios,membership_query):
        audit['calls']+=1
        if audit['calls'] in (20,80,150,220,300):
            cache={}
            def query(keys):
                groups=membership_query(keys);cache.update(zip(map(int,keys),groups));return groups
            result=reduce_owners(indices,neighbor_valid,ratios,query)
            start=time.perf_counter()
            expected=original(indices,neighbor_valid,ratios,lambda keys:[cache[int(k)] for k in keys])
            audit['verification_seconds']+=time.perf_counter()-start
            audit['mismatches']+=int(np.count_nonzero(result!=expected));audit['checked_calls'].append(audit['calls'])
            assert np.array_equal(result,expected),'Owner reduction mismatch on actual scene input'
        else:result=reduce_owners(indices,neighbor_valid,ratios,membership_query)
        path.write_text(json.dumps(audit,indent=2))
        return result
    module.unambiguous_owners=optimized
