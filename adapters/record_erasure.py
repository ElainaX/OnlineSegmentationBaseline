"""Read-only pre-erasure one-ring TSDF audit. No GT or algorithm state writes."""
import json,time,functools
from pathlib import Path
import numpy as np

def install():
    import open3d.core as core
    from submap_disjoint.ig_correction import IGCorrection,lattice_indices
    original=IGCorrection.erase
    @functools.wraps(original)
    def recorded(self,frame,requests):
        started=time.perf_counter();o=self.owner;grid=o.voxel_block_grids
        xyz=np.unique(np.concatenate([cs for _,cs in requests]),axis=0)
        ids,ok=lattice_indices(grid,xyz,self.voxel,self.res,o.device_o3c)
        xyz,ids=xyz[ok],ids[ok]
        weights=grid.attribute('weight').reshape((-1,))[core.Tensor(ids,dtype=core.int64,device=o.device_o3c)].cpu().numpy()
        xyz=xyz[weights>0];ids=ids[weights>0]
        ev=None
        if len(ids):
            cells=np.rint(xyz/self.voxel).astype(np.int64)
            delta=np.concatenate([np.zeros((1,3),int),np.eye(3,dtype=int),-np.eye(3,dtype=int)])
            ring=np.unique((cells[:,None,:]+delta[None,:,:]).reshape(-1,3),axis=0)
            pos=ring*self.voxel
            ix,valid=lattice_indices(grid,pos,self.voxel,self.res,o.device_o3c)
            pos,ix,ring=pos[valid],ix[valid],ring[valid]
            oi=core.Tensor(ix,dtype=core.int64,device=o.device_o3c)
            ev=dict(xyz=pos,indices=ix,erased=np.isin(ix,ids),
                    tsdf=grid.attribute('tsdf').reshape((-1,))[oi].cpu().numpy().copy(),
                    weight=grid.attribute('weight').reshape((-1,))[oi].cpu().numpy().copy())
        before=time.perf_counter()-started
        result=original(self,frame,requests)
        started=time.perf_counter()
        if result['voxels']:
            assert ev is not None
            p=self.archive/f'{frame}.npz'
            with np.load(p) as z:
                ix=np.flatnonzero(ev['erased']);order=np.argsort(ev['indices'][ix]);ix=ix[order]
                zo=np.argsort(z['indices'])
                assert np.array_equal(ev['indices'][ix],z['indices'][zo])
                assert np.array_equal(ev['weight'][ix],z['weight'].reshape(-1)[zo])
                assert np.array_equal(ev['tsdf'][ix],z['tsdf'].reshape(-1)[zo])
            np.savez_compressed(self.archive/f'{frame}.evidence.npz',**ev)
        row=dict(frame=int(frame),archive=str(self.archive/f'{frame}.npz'),actual_erased_voxels=result['voxels'],
                 pre_read_seconds=before,post_verify_save_seconds=time.perf_counter()-started,
                 protocol='All 6-neighbour TSDF edges incident on actual erased positive-weight nodes; read-only; exact archive equivalence asserted')
        with (self.out/'evidence_timing.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        return result
    IGCorrection.erase=recorded
