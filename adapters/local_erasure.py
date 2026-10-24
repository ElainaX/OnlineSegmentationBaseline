"""L ablation: restrict the unchanged Feng action to contradicted surface cells.

Detector confirmation, lineage resolution, sharing protection and the recorded
eraser are unchanged. Only the spatial extent and forced retirement are changed.
"""
import json
import numpy as np


def local_cells(target, other, points, contradicted, voxel):
    from feng_ig_adapter import exclusive_cells
    from submap_disjoint.ig_correction import corners
    points=np.asarray(points).reshape(-1,3)
    contradicted=np.asarray(contradicted,dtype=bool)
    assert contradicted.shape==(len(points),)
    eligible=exclusive_cells(target,other,voxel)
    if not contradicted.any() or not len(eligible):return np.empty((0,3))
    # Surface points are interpolated inside a TSDF cube: bound the action to
    # its eight incident lattice nodes, including negative world coordinates.
    support=corners(points[contradicted],voxel).reshape(-1,3)
    allowed={tuple(x) for x in np.rint(support/voxel).astype(np.int64)}
    cells=np.rint(eligible/voxel).astype(np.int64)
    keep=np.fromiter((tuple(x) in allowed for x in cells),bool,count=len(cells))
    return eligible[keep]


def delete_local(self,frame,candidate):
    from feng_ig_adapter import resolve_slots
    live=self.owner.maskGraph.instance_dict
    slots=resolve_slots(candidate.source_raw_ids,self.owner.global2merged_mask_id,live)
    row=dict(frame=int(frame),candidate_id=candidate.candidate_id,
        source_slot=candidate.source_slot,source_raw_ids=sorted(candidate.source_raw_ids),
        resolved_slots=slots,scope='contradicted_reference_surface_cube_corners',
        forced_whole_instance_retirement=False,actual_erased_voxels=0,
        confirmed_reference_points=int(np.count_nonzero(candidate.contradicted)))
    if not slots:
        row['status']='no_live_descendant';self.actions.append(row);return
    target=np.concatenate([live[s].mask_voxel_coords.cpu().numpy() for s in slots])
    rest=[inst.mask_voxel_coords.cpu().numpy() for s,inst in live.items() if s not in slots]
    other=np.concatenate(rest) if rest else np.empty((0,3))
    xyz=local_cells(target,other,candidate.points,candidate.contradicted,self.voxel)
    row.update(target_nodes=len(np.unique(np.rint(target/self.voxel).astype(np.int64),axis=0)),local_exclusive_nodes=len(xyz))
    self.archive=self.out/'actions'/str(candidate.candidate_id)
    self.archive.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(self.archive/f'{frame}.local_scope.npz',
        reference_points=candidate.points,contradicted=candidate.contradicted,proposal=xyz)
    self.refs[slots[0]]=dict(deleted_voxels=0)
    result=self.erase(frame,[(slots[0],xyz)]) if len(xyz) else dict(voxels=0)
    # erase maintains indices, sparse edges, hashes and export caches, and can
    # remove genuinely empty instances. Never retire remaining target support.
    row.update(status='applied',actual_erased_voxels=int(result['voxels']),
        archive=str(self.archive/f'{frame}.npz') if result['voxels'] else None,
        removed_instance_records=len(result.get('emptied_current_ids',[])),erasure=result)
    self.actions.append(row)


def install():
    import feng_ig_adapter as adapter
    original_save=adapter.FengIGCorrection.save
    def save(self):
        original_save(self)
        if not hasattr(self,'detector'):return
        path=self.out/'summary.json';data=json.loads(path.read_text())
        data.update(ablation='L',policy='Unchanged Feng confirmation -> lineage -> exclusive contradicted surface cube corners -> same recorded eraser; no forced whole-instance retirement.',
            scope_note='Same confirmation rule, not replayed Full events: downstream observations can differ after map updates.')
        path.write_text(json.dumps(data,indent=2))
    adapter.FengIGCorrection.delete_candidate=delete_local
    adapter.FengIGCorrection.save=save
