"""Original Feng detector on IG geometry; whole-instance action adapter.

The detector and its thresholds are unchanged. All mutation is here. Resolve
frozen raw-mask lineage to current slots at action time; never reuse old slots.
Only exclusive current instance lattice nodes are erased from shared TSDF.
Whole target instance records are removed even if shared nodes are preserved.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from feng_remove_detection import RemoveDetector
from feng_geometry_helpers import lookup_voxels_safe,raycast_history,query_active_instance_memberships
from submap_disjoint.ig_correction import IGCorrection


class GeometryView:
    lookup_voxels_safe=lookup_voxels_safe
    raycast_history=raycast_history
    def __init__(self,owner):
        self.voxel_block_grids=owner.voxel_block_grids
        self.voxel_size=owner.cfg['scene']['voxel_size']
        self.block_resolution=owner.cfg['scene']['block_resolution']
        self.device_o3c=owner.device_o3c


class HashView:
    query_active_instance_memberships=query_active_instance_memberships
    def __init__(self,owner):self.owner=owner
    @property
    def hash_table(self):return self.owner.voxel_hash_table.hash_table


class SceneView:
    def __init__(self,owner):
        self.owner=owner
        self.voxel_grids=GeometryView(owner)
        self.voxel_hash_table=HashView(owner)
    def __getattr__(self,name):return getattr(self.owner,name)


def resolve_slots(raw_ids,mapping,live):
    return sorted({int(mapping[r]) for r in raw_ids
                   if 0<=r<len(mapping) and int(mapping[r]) in live})


def exclusive_cells(target,other,voxel):
    """Integer lattice set subtraction; no dependence on hash-buffer layout."""
    cells=np.unique(np.rint(np.asarray(target).reshape(-1,3)/voxel).astype(np.int64),axis=0)
    blocked={tuple(x) for x in np.rint(np.asarray(other).reshape(-1,3)/voxel).astype(np.int64)}
    return np.asarray([x for x in cells if tuple(x) not in blocked],dtype=np.int64).reshape(-1,3)*voxel


class FengIGCorrection(IGCorrection):
    def __init__(self,owner,directory,enabled=True,options=None):
        super().__init__(owner,directory,enabled=enabled,rgb=True)
        self.settings=options or json.loads(Path('feng_detector_config.json').read_text())
        self.opt=SimpleNamespace(**self.settings)
        cfg=dict(owner.cfg,remove_detection=self.settings)
        self.detector=RemoveDetector(SceneView(owner),cfg,Path(directory).parent)
        self.handled=set();self.actions=[]

    def freeze(self,frame):
        # Runner invokes this once before the first observation, including frame 0.
        self.initialized=True;self.freeze_frame=int(frame)

    def step(self,frame,rgb,depth,K,pose):
        self.detector.step(frame,rgb,depth,pose,pose_valid=True)
        new=[c for c in self.detector.candidates if c.state=='would_remove' and c.candidate_id not in self.handled]
        for c in new:
            self.handled.add(c.candidate_id)
            if self.enabled:self.delete_candidate(frame,c)
        self.save()

    def delete_candidate(self,frame,candidate):
        o=self.owner;live=o.maskGraph.instance_dict
        slots=resolve_slots(candidate.source_raw_ids,o.global2merged_mask_id,live)
        tracks={int(live[s].track_uid) for s in slots}
        row=dict(frame=int(frame),candidate_id=candidate.candidate_id,source_slot=candidate.source_slot,
                 source_raw_ids=sorted(candidate.source_raw_ids),resolved_slots=slots,track_uids=sorted(tracks))
        if not slots:
            row.update(status='no_live_descendant',actual_erased_voxels=0);self.actions.append(row);return
        target=np.concatenate([live[s].mask_voxel_coords.cpu().numpy() for s in slots])
        rest=[inst.mask_voxel_coords.cpu().numpy() for s,inst in live.items() if s not in slots]
        other=np.concatenate(rest) if rest else np.empty((0,3))
        xyz=exclusive_cells(target,other,self.voxel)
        row.update(target_nodes=len(np.unique(np.rint(target/self.voxel).astype(np.int64),axis=0)),
                   exclusive_nodes=len(xyz),shared_nodes_preserved=len(np.unique(np.rint(target/self.voxel).astype(np.int64),axis=0))-len(xyz))
        # Per-candidate archive directories prevent two same-frame actions from
        # overwriting each other. The eraser archives only nodes with weight>0.
        self.archive=self.out/'actions'/str(candidate.candidate_id);self.archive.mkdir(parents=True,exist_ok=True)
        self.refs[slots[0]]=dict(deleted_voxels=0)
        result=self.erase(frame,[(slots[0],xyz)]) if len(xyz) else dict(voxels=0)
        # erase() may remove empties and renumber slots. Track UID is persistent.
        remaining=[s for s,inst in o.maskGraph.instance_dict.items() if int(inst.track_uid) in tracks]
        if remaining:o.remove_masks(frame,torch.tensor(remaining,device=o.device,dtype=torch.long))
        row.update(status='applied',actual_erased_voxels=int(result['voxels']),
                   archive=str(self.archive/f'{frame}.npz') if result['voxels'] else None,
                   removed_instance_records=len(slots),erasure=result)
        self.actions.append(row)

    def save(self):
        if not hasattr(self,'detector'):return
        data=dict(detector_version='feng-original-on-IG',deletion_enabled=self.enabled,
                  detector_settings=self.settings,unique_deleted_world_voxels=len(self.deleted),
                  candidates=[c.metadata() for c in self.detector.candidates],actions=self.actions,
                  skips=self.detector.skips,evaluated_frames=self.detector.frames_processed,
                  policy='would_remove -> resolve current descendants -> exclusive TSDF nodes + whole instance records',
                  ground_truth_used=False)
        (self.out/'summary.json').write_text(json.dumps(data,indent=2))

    def finalize(self):
        self.detector.finalize(self.owner.points,self.owner.colors)
        self.save()
