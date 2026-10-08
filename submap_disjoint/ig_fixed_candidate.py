"""Read-only adapter from live IG state to the unchanged onlineSeg detector.

No integration, erasure, graph update, or observation voting is implemented here.
The owner retains all geometry and instance lifetimes. This adapter owns only
diagnostic references, files, and optional non-mutation checks.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .fixed_candidate_core import FixedCandidateRemoveDetector
from .readonly_queries import ReadOnlyVoxelQueries, ReadOnlyMembershipQueries


class ReadOnlySceneAdapter:
    def __init__(self, owner):
        self._owner = owner
        self.intrinsics = owner.intrinsics.detach().cpu().numpy().copy()
        self.voxel_block_grids = owner.voxel_block_grids
        self.trunc_voxel_multiplier = owner.trunc_voxel_multiplier
        self.voxel_grids = ReadOnlyVoxelQueries(owner)
        self.voxel_hash_table = ReadOnlyMembershipQueries(owner)

    @property
    def global2merged_mask_id(self):
        return list(self._owner.global2merged_mask_id)

    @property
    def get_merged_mask_weight(self):
        return self._owner.get_merged_mask_weight.detach().cpu().numpy().copy()


def state_digest(owner):
    """Expensive validation only: live TSDF, memberships and graph state.

Raycasting may resize Open3D scratch memory. Scratch allocation is deliberately
excluded; map keys, voxel attributes and semantic state must remain unchanged.
"""
    h = hashlib.sha256()

    def add(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            h.update(str((value.dtype.str, value.shape)).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                add(key); add(value[key])
        elif isinstance(value, (set, frozenset)):
            add(sorted(value))
        elif isinstance(value, (list, tuple)):
            h.update(str(len(value)).encode())
            for item in value: add(item)
        elif value is None or isinstance(value, (int, float, str, bool, np.number)):
            h.update(repr(value).encode())
        else:
            raise TypeError(f'Unsupported validation state: {type(value)}')

    grid = owner.voxel_block_grids
    # CUDA hash enumeration order is not stable; compare the same buffer rows.
    core = __import__('open3d').core
    rows = np.sort(grid.hashmap().active_buf_indices().cpu().numpy().astype(np.int64))
    active = core.Tensor(rows, device=owner.device_o3c)
    add(active.cpu().numpy())
    add(grid.hashmap().key_tensor()[active].cpu().numpy())
    for name in ('tsdf', 'weight', 'color'):
        add(name); add(grid.attribute(name)[active].cpu().numpy())
    table = owner.voxel_hash_table
    add(table.hash_item_flag)
    # Include nonempty rows even if a stale occupancy flag is false.
    add({i: sorted(x) for i, x in enumerate(table.hash_table) if x})
    for name in ('global2merged_mask_id', 'global_f_mask_ids', 'merged_mask_size',
                 'merged_mask_weight', 'mask_voxels_coords', 'mask_features',
                 'mask_geo_features', 'mask_geo_features_mask', 'merged_mask_bbox',
                 'merged_mask_last_frame', 'c_mask_num', 'g_mask_num',
                 'edge_rows', 'edge_cols', 'edge_values', 'contain_matrix',
                 'points', 'colors', 'migration_audit'):
        if hasattr(owner, name): add(name); add(getattr(owner, name))
    for mid, instance in sorted(owner.maskGraph.instance_dict.items()):
        add(mid)
        for name, value in sorted(vars(instance).items()):
            if torch.is_tensor(value) or isinstance(value, (np.ndarray, list, tuple, dict,
                    set, frozenset, int, float, str, bool, np.number)) or value is None:
                add(name); add(value)
    return h.hexdigest()


class IGFixedCandidateDetector:
    def __init__(self, owner, cfg, output_dir, verify=False):
        self.owner = owner
        self.verify = verify
        self.checks = []
        self.validation_seconds = 0.
        self.root = Path(output_dir)
        self.provenance = json.loads((Path(__file__).parents[1] / 'config/ig_readonly_provenance.json').read_text())
        core_path = Path(__file__).with_name('readonly_detection_core.py')
        if hashlib.sha256(core_path.read_bytes()).hexdigest() != self.provenance['detector_sha256']:
            raise RuntimeError('Vendored onlineSeg detector differs from the recorded source')
        self.detector = FixedCandidateRemoveDetector(ReadOnlySceneAdapter(owner), cfg, output_dir)
        self.cfg = self.detector.cfg

    def _checked_call(self, name, callback):
        before = None
        if self.verify:
            start = time.perf_counter(); before = state_digest(self.owner)
            self.validation_seconds += time.perf_counter() - start
        result = callback()
        if self.verify:
            start = time.perf_counter(); after = state_digest(self.owner)
            self.validation_seconds += time.perf_counter() - start
            self.checks.append(dict(call=name, unchanged=before == after, before=before, after=after))
            self._save_audit()
            if before != after: raise AssertionError(f'Read-only detector modified IG state: {name}')
        return result

    def step(self, frame, color, depth, pose, pose_valid=True, session=None):
        return self._checked_call(f'frame_{frame}', lambda: self.detector.step(
            frame, color, depth, pose, pose_valid, session))

    def finalize(self, points, colors):
        result = self._checked_call('finalize', lambda: self.detector.finalize(points, colors))
        self._save_audit()
        return result

    def report(self):
        return dict(mode='detection_only', detector_sha256=self.provenance['detector_sha256'],
            source_commit=self.provenance['source_commit'], live_geometry=True,
            all_sessions=True, deletion_enabled=False, mutation_checks=len(self.checks),
            mutation_check_passed=all(x['unchanged'] for x in self.checks) if self.checks else None,
            verification_enabled=self.verify, validation_seconds=self.validation_seconds,
            detection_seconds=self.detector.total_seconds,
            summary=json.loads((self.detector.out/'summary.json').read_text()), checks=self.checks)

    def _save_audit(self):
        path = self.root / 'readonly_alignment.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.report(), indent=2))
        temporary.replace(path)
