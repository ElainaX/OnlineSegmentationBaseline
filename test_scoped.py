"""Targeted semantic checks for owner union and local evidence lifetime."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from feng_fast_owners import reduce_owners
from feng_scoped_rgb import canonical_cells,repeated_evidence,update_local

def main():
    # Owner union must preserve true ambiguity, not majority votes.
    ids=np.array([[[0,1,2],[3,4,-1],[1,1,1],[5,5,5]]])
    groups=[frozenset([2]),frozenset([2]),frozenset([8]),frozenset(),frozenset(),frozenset([2,8])]
    got=reduce_owners(ids,ids>=0,np.ones(ids.shape),lambda keys:[groups[k] for k in keys])
    assert got.tolist()==[[ -2,-1,2,-2]]
    c=SimpleNamespace(local_depth=np.zeros(4,np.int16),local_rgb=np.zeros(4,np.int16),local_support=np.zeros(4,np.int16))
    code=np.array([2,2,1,0]);source=np.array([1,2,0,0])
    update_local(c,'contradiction',code,source,True)
    assert not any(x.any() for x in repeated_evidence(c,{'confirm_observations':2}))
    update_local(c,'paused_correlated_observation',code,source,False)
    assert c.local_rgb[1]==1 # repeated correlated frames must not count
    update_local(c,'would_remove',code,source,True)
    d,r=repeated_evidence(c,{'confirm_observations':2});assert d.tolist()==[True,False,False,False];assert r.tolist()==[False,True,False,False]
    update_local(c,'paused_insufficient_reference_points',np.zeros(4),np.zeros(4),False)
    assert not any(x.any() for x in repeated_evidence(c,{'confirm_observations':2})) # currently invisible abstains
    update_local(c,'support',code,source,True)
    assert not c.local_rgb.any() and not c.local_depth.any()
    cells=np.arange(-400,401);xyz=np.c_[cells,np.zeros_like(cells),np.zeros_like(cells)].astype(np.float32)*np.float32(.025)
    canonical=canonical_cells(xyz,.025);assert np.array_equal(canonical[:,0],cells)
    offsets=canonical-np.floor_divide(canonical,4)*4;assert ((offsets>=0)&(offsets<4)).all()
    print('PASS: ambiguity, RGB retention, independent views, current visibility, support reset, negative lattice coordinates')

if __name__=='__main__':main()
