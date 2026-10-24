"""Copied CPU surface extraction only; no mapping or GT loads."""
import numpy as np
from scipy.spatial import cKDTree
V=.025

def atoms(xyz,labels=None):
    keys,first,inv=np.unique(np.floor(xyz/V).astype(np.int64),axis=0,return_index=True,return_inverse=True)
    n=np.bincount(inv)
    pts=np.column_stack([np.bincount(inv,weights=xyz[:,i])/n for i in range(3)])
    # One spatial atom, one label: closest original point to atom centroid.
    if labels is not None: lab=labels[cKDTree(xyz).query(pts)[1]]
    else:lab=None
    return pts,keys,lab

def archived_surface(z):
    """Actual pre-erasure zero-crossings; occupied volume nodes are NOT surfaces."""
    xyz=z['xyz'];s=np.asarray(z['tsdf']).reshape(-1);w=np.asarray(z['weight']).reshape(-1)
    good=np.isfinite(xyz).all(axis=1)&np.isfinite(s)&(w>0)
    erased=np.asarray(z['erased'],bool)[good] if 'erased' in z else np.ones(int(good.sum()),bool)
    xyz=xyz[good];s=s[good];k=np.rint(xyz/V).astype(np.int64)
    assert np.max(np.abs(xyz-k*V),initial=0)<1e-5
    lookup={tuple(v):i for i,v in enumerate(k)};out=[]
    used=set()
    for i,ki in enumerate(k):
        if erased[i] and abs(s[i])<1e-8:out.append(xyz[i]);used.add(i)
        for ax in range(3):
            other=ki.copy();other[ax]+=1;j=lookup.get(tuple(other))
            if j is None or s[i]*s[j]>=0 or not (erased[i] or erased[j]):continue
            t=s[i]/(s[i]-s[j]);out.append(xyz[i]+t*(xyz[j]-xyz[i]));used.update([i,j])
    pts=np.asarray(out,float).reshape(-1,3)
    if len(pts):pts=atoms(pts)[0]
    return pts,dict(archived_valid_nodes=len(xyz),zero_crossing_surface_atoms=len(pts),nodes_supporting_zero_crossings=len(used))
