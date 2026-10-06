"""Sparse mask evidence graph for submap-owned instance reasoning.

The original MaskGraph stores voxel-by-segmentation-frame evidence in dense
matrices.  This module stores the same information as observation nodes,
voxel postings, and sparse pair edges.  The directed containment equations
are kept unchanged; only the storage and candidate enumeration are sparse.
"""
from __future__ import annotations

from collections import defaultdict
import json
import numpy as np


class SparseSubmapMaskGraph:
    def __init__(self, cfg):
        mask_cfg = cfg.get('mask', {})
        self.visible_threshold = float(mask_cfg.get('mask_visible_threshold', 0.2))
        self.visible_min_overlap = int(mask_cfg.get('mask_visible_min_overlap', 500))
        self.contain_min_threshold = float(mask_cfg.get('contain_min_threshold', 0.025))
        self.enabled = bool(cfg.get('submaps', {}).get('sparse_mask_graph', {}).get('enabled', True))
        self.nodes = {}
        self.parent = {}
        self.voxel_postings = defaultdict(set)
        self.voxel_frames = defaultdict(set)
        self.edges = {}
        self.worst_examples = []
        self.audit = {
            'observations': 0, 'candidate_pairs': 0, 'scored_pairs': 0,
            'nonzero_pairs': 0, 'dense_pairs': 0, 'compared_pairs': 0,
            'abs_error_sum': 0.0, 'max_abs_error': 0.0,
            'dense_nonzero': 0, 'sparse_nonzero': 0,
            'dense_nonzero_error_sum': 0.0,
            'dense_nonzero_compared': 0,
            'dense_nonzero_recovered': 0,
        }

    @staticmethod
    def _voxels(values):
        if values is None:
            return np.empty((0,), dtype=np.int64)
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        arr = np.asarray(values)
        if arr.ndim == 2:
            # A voxel coordinate is made hashable without depending on the
            # runtime-specific flattened index mapping.
            arr = arr.astype(np.int64, copy=False)
            arr = arr[:, 0] * 1000000000000 + arr[:, 1] * 1000000 + arr[:, 2]
        return np.unique(arr.astype(np.int64, copy=False))

    def add_observation(self, observation_id, frame_id, voxel_values):
        if not self.enabled:
            return
        uid = int(observation_id)
        voxels = self._voxels(voxel_values)
        if uid in self.nodes:
            raise ValueError('duplicate sparse mask observation id: %d' % uid)
        self.nodes[uid] = {
            'id': uid, 'frame': int(frame_id), 'frames': {int(frame_id)},
            'voxels': voxels, 'members': [uid],
        }
        self.parent[uid] = uid
        for voxel in voxels.tolist():
            voxel = int(voxel)
            self.voxel_postings[voxel].add(uid)
            self.voxel_frames[voxel].add(int(frame_id))
        self.audit['observations'] += 1

    def add_observations(self, frame_id, pairs, voxel_observations,
                         visibility_voxels=None):
        if not self.enabled:
            return
        # MaskGraph records visibility for every raw 2-D mask before size
        # filtering. Keep that history without creating track nodes for the
        # rejected masks.
        if hasattr(visibility_voxels, 'values'):
            for values in visibility_voxels.values():
                for voxel in self._voxels(values).tolist():
                    self.voxel_frames[int(voxel)].add(int(frame_id))
        pairs = list(pairs or [])
        for index, pair in enumerate(pairs):
            mid = int(pair['mask_id'])
            uid = int(pair['global_mask_id'])
            if hasattr(voxel_observations, 'get'):
                values = voxel_observations.get(mid)
            else:
                values = voxel_observations[index] if index < len(voxel_observations or []) else None
            self.add_observation(uid, frame_id, values)

    def root(self, uid):
        uid = int(uid)
        while self.parent.get(uid, uid) != uid:
            self.parent[uid] = self.parent[self.parent[uid]]
            uid = self.parent[uid]
        return uid

    def _aggregate(self, uid):
        root = self.root(uid)
        member_ids = self.nodes[root].get('members', [root])
        voxels = np.unique(np.concatenate([self.nodes[x]['voxels'] for x in member_ids])) if member_ids else np.empty((0,), dtype=np.int64)
        frames = set()
        for x in member_ids:
            frames.update(self.nodes[x].get('frames', ()))
        return root, voxels, frames

    def union(self, ids):
        ids = sorted({self.root(int(x)) for x in ids if int(x) in self.parent})
        if len(ids) < 2:
            return ids[0] if ids else None
        target = ids[0]
        for source in ids[1:]:
            source = self.root(source)
            target = self.root(target)
            if source == target:
                continue
            self.parent[source] = target
            self.nodes[target]['members'].extend(self.nodes[source].get('members', [source]))
            self.nodes[target]['frames'].update(self.nodes[source].get('frames', ()))
        return target

    def _visible_count(self, voxels, target_frames):
        if not target_frames:
            return 0
        return sum(1 for voxel in voxels
                   if self.voxel_frames.get(int(voxel), set()) & target_frames)

    def score_pair(self, first_id, second_id):
        """Return the MaskGraph directed containment scores for two roots."""
        a, vox_a, frames_a = self._aggregate(first_id)
        b, vox_b, frames_b = self._aggregate(second_id)
        if a == b:
            return {'a': a, 'b': b, 'overlap_count': int(len(vox_a)),
                    'a_in_b_visible': int(len(vox_a)),
                    'b_in_a_visible': int(len(vox_b)),
                    'a_contains_b': 1.0, 'b_contains_a': 1.0,
                    'valid_a_contains_b': True, 'valid_b_contains_a': True}
        if len(vox_a) == 0 or len(vox_b) == 0:
            return {'a': a, 'b': b, 'overlap_count': 0,
                    'a_in_b_visible': 0, 'b_in_a_visible': 0,
                    'a_contains_b': 0.0, 'b_contains_a': 0.0,
                    'valid_a_contains_b': False, 'valid_b_contains_a': False}
        # Both arrays are sorted and unique, so intersect1d is exact and
        # considerably smaller than a dense voxel x frame multiplication.
        overlap = np.intersect1d(vox_a, vox_b, assume_unique=True)
        overlap_count = int(len(overlap))
        a_visible = self._visible_count(vox_a, frames_b)
        b_visible = self._visible_count(vox_b, frames_a)
        a_ratio = float(overlap_count / a_visible) if a_visible else 0.0
        b_ratio = float(overlap_count / b_visible) if b_visible else 0.0
        a_visible_ratio = float(a_visible / len(vox_a)) if len(vox_a) else 0.0
        b_visible_ratio = float(b_visible / len(vox_b)) if len(vox_b) else 0.0
        valid_a = (a_visible_ratio > self.visible_threshold or a_visible > self.visible_min_overlap) and a_ratio > self.contain_min_threshold
        valid_b = (b_visible_ratio > self.visible_threshold or b_visible > self.visible_min_overlap) and b_ratio > self.contain_min_threshold
        return {
            'a': a, 'b': b, 'overlap_count': overlap_count,
            'a_in_b_visible': a_visible, 'b_in_a_visible': b_visible,
            'a_visible_ratio': a_visible_ratio, 'b_visible_ratio': b_visible_ratio,
            # Matrix[i,j] in MaskGraph means i contains j, so its denominator
            # is j visible on i's frame set.
            'a_contains_b': float(min(1.0, b_ratio)),
            'b_contains_a': float(min(1.0, a_ratio)),
            'valid_a_contains_b': bool(valid_b), 'valid_b_contains_a': bool(valid_a),
        }

    def score_member_sets(self, first_ids, second_ids):
        """Score explicit MaskGraph members without expanding their track roots."""
        first_ids = [int(x) for x in first_ids if int(x) in self.nodes]
        second_ids = [int(x) for x in second_ids if int(x) in self.nodes]
        vox_a = np.unique(np.concatenate([self.nodes[x]['voxels'] for x in first_ids])) if first_ids else np.empty((0,), dtype=np.int64)
        vox_b = np.unique(np.concatenate([self.nodes[x]['voxels'] for x in second_ids])) if second_ids else np.empty((0,), dtype=np.int64)
        frames_a = {f for x in first_ids for f in self.nodes[x].get('frames', ())}
        frames_b = {f for x in second_ids for f in self.nodes[x].get('frames', ())}
        if len(vox_a) == 0 or len(vox_b) == 0:
            return {'a_contains_b': 0.0, 'b_contains_a': 0.0,
                    'overlap_count': 0, 'a_in_b_visible': 0,
                    'b_in_a_visible': 0}
        overlap_count = int(len(np.intersect1d(vox_a, vox_b, assume_unique=True)))
        a_visible = self._visible_count(vox_a, frames_b)
        b_visible = self._visible_count(vox_b, frames_a)
        a_ratio = float(overlap_count / a_visible) if a_visible else 0.0
        b_ratio = float(overlap_count / b_visible) if b_visible else 0.0
        a_valid = ((a_visible / len(vox_a) > self.visible_threshold or
                    a_visible > self.visible_min_overlap) and
                   a_ratio > self.contain_min_threshold)
        b_valid = ((b_visible / len(vox_b) > self.visible_threshold or
                    b_visible > self.visible_min_overlap) and
                   b_ratio > self.contain_min_threshold)
        return {'a_contains_b': float(min(1.0, b_ratio)),
                'b_contains_a': float(min(1.0, a_ratio)),
                'overlap_count': overlap_count,
                'a_in_b_visible': a_visible,
                'b_in_a_visible': b_visible,
                'valid_a_contains_b': bool(b_valid),
                'valid_b_contains_a': bool(a_valid)}

    def candidate_pairs(self, observation_id):
        uid = int(observation_id)
        if uid not in self.nodes:
            return []
        candidates = set()
        for voxel in self.nodes[uid]['voxels'].tolist():
            candidates.update(self.voxel_postings.get(int(voxel), ()))
        candidates.discard(uid)
        roots = sorted({self.root(x) for x in candidates})
        self.audit['candidate_pairs'] += len(roots)
        return roots

    def score_observation_against_tracks(self, observation_id):
        pairs = []
        for root in self.candidate_pairs(observation_id):
            score = self.score_pair(observation_id, root)
            self.edges[(min(self.root(observation_id), root), max(self.root(observation_id), root))] = score
            self.audit['scored_pairs'] += 1
            self.audit['nonzero_pairs'] += int(score['overlap_count'] > 0)
            pairs.append(score)
        return pairs

    def compare_dense_matrix(self, instance_to_ids, dense_matrix):
        """Compare sparse scores to a MaskGraph contain matrix.

        ``instance_to_ids`` maps each dense MaskGraph instance id to the
        append-only observation ids that compose it.  The matrix convention is
        preserved: [i,j] means instance i contains instance j.
        """
        if not self.enabled or dense_matrix is None:
            return
        if hasattr(dense_matrix, 'detach'):
            dense_matrix = dense_matrix.detach().cpu().numpy()
        dense = np.asarray(dense_matrix)
        ids = sorted(int(x) for x in instance_to_ids if int(x) < dense.shape[0])
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                first = instance_to_ids[i]
                second = instance_to_ids[j]
                if not first or not second:
                    continue
                score = self.score_member_sets(first, second)
                sparse_value = (float(score['a_contains_b'])
                                if score.get('valid_a_contains_b', True) else 0.0)
                dense_value = float(dense[i, j])
                err = abs(sparse_value - dense_value)
                self.audit['compared_pairs'] += 1
                self.audit['abs_error_sum'] += err
                self.audit['max_abs_error'] = max(self.audit['max_abs_error'], err)
                if dense_value > 0 and len(self.worst_examples) < 20:
                    self.worst_examples.append({
                        'dense_i': int(i), 'dense_j': int(j),
                        'dense': dense_value, 'sparse': sparse_value,
                        'error': err, 'overlap_count': score.get('overlap_count', 0),
                        'a_visible': score.get('b_in_a_visible', 0),
                        'b_visible': score.get('a_in_b_visible', 0),
                    })
                    self.worst_examples.sort(key=lambda x: x['error'], reverse=True)
                    del self.worst_examples[20:]
                self.audit['dense_pairs'] += int(dense_value > 0)
                self.audit['sparse_nonzero'] += int(sparse_value > 0)
                self.audit['dense_nonzero'] += int(dense_value > 0)
                if dense_value > 0:
                    self.audit['dense_nonzero_error_sum'] += err
                    self.audit['dense_nonzero_compared'] += 1
                    self.audit['dense_nonzero_recovered'] += int(sparse_value > 0)

    def summary(self):
        out = dict(self.audit)
        out['mean_abs_error'] = (out['abs_error_sum'] / out['compared_pairs']
                                 if out['compared_pairs'] else 0.0)
        out['dense_nonzero_mean_abs_error'] = (
            out['dense_nonzero_error_sum'] / out['dense_nonzero_compared']
            if out['dense_nonzero_compared'] else 0.0)
        out['dense_nonzero_recall'] = (
            out['dense_nonzero_recovered'] / out['dense_nonzero_compared']
            if out['dense_nonzero_compared'] else 0.0)
        out['stored_edges'] = len(self.edges)
        out['active_roots'] = len({self.root(x) for x in self.parent})
        out['worst_examples'] = self.worst_examples
        return out

    def save(self, path):
        Path = __import__('pathlib').Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2, sort_keys=True))
