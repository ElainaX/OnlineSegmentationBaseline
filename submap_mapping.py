"""Mask-driven object submaps, persistent merge lineage, and first-view atlas.

Tracks are online; object TSDFs are materialized sequentially from recorded
masked RGB-D observations at export. No camera-motion partition, GT input,
class label, or new model inference is used. 'First observed' and historical
free-space appearance evidence are explicitly different labels.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict, deque
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def cpu(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def sample_points(points, limit):
    if len(points) <= limit:
        return points.copy()
    return points[np.linspace(0, len(points) - 1, limit, dtype=int)].copy()


class SubmapManager:
    def __init__(self, cfg, dataset, device='cuda:0', pc_extractor=None):
        self.enabled = bool(cfg.get('submaps', {}).get('enabled', False))
        self.dataset, self.device = dataset, device
        self.cfg = cfg
        self.opt = cfg.get('submaps', {})
        # FCGF is optional at the submap layer.  Reuse Scene_rep's already
        # loaded extractor so enabling this path does not load a second model.
        self.pc_extractor = pc_extractor
        self.fcgf_opt = self.opt.get('fcgf_matching', {})
        self.geometry_backend = self.fcgf_opt.get('backend', 'fcgf')
        if self.geometry_backend not in ('fcgf', 'd3feat'):
            raise ValueError('Unknown submap geometry backend: ' + self.geometry_backend)
        if self.enabled and self.fcgf_opt.get('enabled', False) and self.geometry_backend == 'd3feat':
            from tool.d3feat_backend import D3FeatExtractor
            self.pc_extractor = D3FeatExtractor(self.fcgf_opt['d3feat'])
        self.fcgf_enabled = bool(self.fcgf_opt.get('enabled', False) and
                                  self.pc_extractor is not None)
        self.fcgf_require = bool(self.fcgf_opt.get('require_valid', False))
        self.fcgf_scene_cache = bool(self.fcgf_opt.get('scene_feature_cache', True))
        self.fcgf_weight = float(self.fcgf_opt.get('score_weight', 0.25))
        self.fcgf_min_similarity = float(self.fcgf_opt.get('min_similarity', 0.25))
        self.d3feat_opt = self.fcgf_opt.get('d3feat_matching', {})
        self.d3feat_max_keypoints = max(4, int(self.d3feat_opt.get('max_keypoints', 48)))
        self.d3feat_min_keypoints = max(2, int(self.d3feat_opt.get('min_keypoints', 4)))
        self.d3feat_min_pair_similarity = float(
            self.d3feat_opt.get('min_pair_similarity', 0.55))
        self.d3feat_min_mutual_matches = max(
            2, int(self.d3feat_opt.get('min_mutual_matches', 3)))
        self.d3feat_min_match_coverage = float(
            self.d3feat_opt.get('min_match_coverage', 0.10))
        self.d3feat_score_weight = min(max(float(
            self.d3feat_opt.get('score_weight', 0.35)), 0.0), 1.0)
        self.d3feat_geom_weight = min(max(float(
            self.d3feat_opt.get('geometry_weight', 0.15)), 0.0), 1.0)
        # Local descriptors are useful evidence, but a missing mutual match
        # should not discard an otherwise valid RGB-D/embedding match.
        self.d3feat_require_match = bool(
            self.d3feat_opt.get('require_match', True))
        self.fcgf_limits = [max(8, int(x)) for x in self.fcgf_opt.get(
            'input_point_limits', [32, 96, 256])]
        self.fcgf_thresholds = [max(0, int(x)) for x in self.fcgf_opt.get(
            'point_count_thresholds', [96, 384])]
        self.fcgf_min_points = [max(1, int(x)) for x in self.fcgf_opt.get(
            'min_points_by_level', [12, 20, 32])]
        if len(self.fcgf_thresholds) != max(0, len(self.fcgf_limits) - 1):
            raise ValueError('fcgf_matching.point_count_thresholds must have one fewer value than input_point_limits')
        if len(self.fcgf_min_points) != len(self.fcgf_limits):
            raise ValueError('fcgf_matching.min_points_by_level must match input_point_limits')
        self.K = cpu(dataset.cam_intrinsic).astype(np.float64)
        self.voxel_size = float(cfg['scene']['voxel_size'])
        self.near = max(0.0, float(cfg['cam'].get('depth_near', 0.1)))
        self.far = float(cfg['cam']['depth_far'])
        # Track confirmation can use the historical distinct-frame rule or a
        # confidence accumulator. Keep the old rule as the default so older
        # configs remain reproducible; the new mode confirms strong matches
        # quickly while requiring several weaker matches to accumulate.
        self.confirmation_mode = str(
            self.opt.get('confirmation_mode', 'legacy_frames')).lower()
        self.min_frames = int(self.opt.get('confirmation_frames', 3))
        self.confirmation_threshold = float(
            self.opt.get('confirmation_confidence_threshold', 1.0))
        self.confirmation_score_floor = float(
            self.opt.get('confirmation_score_floor', 0.45))
        self.confirmation_decay = float(np.clip(
            self.opt.get('confirmation_confidence_decay', 0.995), 0.0, 1.0))
        self.confirmation_high_score = float(
            self.opt.get('confirmation_high_score', 0.88))
        self.confirmation_high_min_frames = max(1, int(
            self.opt.get('confirmation_high_min_frames', 1)))
        self.confirmation_min_evidence_frames = max(1, int(
            self.opt.get('confirmation_min_evidence_frames', 1)))
        self.reject_covisible_graph_merge = bool(self.opt.get('reject_covisible_graph_merge', False))
        self.graph_merge_stats = {'accepted_edges': 0, 'rejected_covisible_edges': 0}
        self.sample_limit = int(self.opt.get('max_track_points', 4096))
        self.match_distance = float(self.opt.get('match_distance_m', 0.06))
        self.match_coverage = float(self.opt.get('match_coverage', 0.30))
        self.match_feature = float(self.opt.get('match_feature_similarity', 0.65))
        self.margin = float(self.opt.get('match_ambiguity_margin', 0.06))
        self.min_pixels = int(self.opt.get('min_mask_pixels', 50))
        self.free_margin = float(self.opt.get('free_space_margin_m', 0.10))
        self.free_ratio = float(self.opt.get('free_space_ratio', 0.60))
        self.history = deque(maxlen=int(self.opt.get('history_frames', 32)))
        if self.confirmation_mode not in ('legacy_frames', 'cumulative_confidence'):
            raise ValueError('Unknown submap confirmation_mode: ' + self.confirmation_mode)
        if (self.confirmation_mode == 'legacy_frames' and self.min_frames < 2) or \
                self.confirmation_threshold <= 0 or self.sample_limit < 32 or self.match_distance <= 0:
            raise ValueError('Invalid object submap confirmation/sampling configuration')
        self.observations = {}
        self.key_to_id = {}
        self.parent = {}
        self.tracks = {}
        self.events = []
        self.graph_history = []
        self.latest_graph = {}
        self.frame_cache = OrderedDict()
        self.pending_history = None
        self.max_blocks_seen = 0
        self.boundary_replays = 0
        self.mv3dis_stats = {'enabled': False, 'candidate_edges': 0,
                             'accepted_edges': 0, 'merged_tracks': 0,
                             'guide_observations': 0, 'guide_cells': 0}
        self.fcgf_stats = {
            'enabled': self.fcgf_enabled, 'backend': self.geometry_backend,
            'candidates': 0, 'valid_descriptors': 0, 'below_min_points': 0,
            'extraction_errors': 0, 'matches_considered': 0,
            'scene_feature_cache_hits': 0, 'scene_feature_cache_misses': 0,
            'matches_with_fcgf': 0, 'fcgf_rejected': 0,
            'fallback_matches': 0, 'accepted_matches': 0,
            'd3feat_mutual_matches': 0, 'd3feat_rejected': 0,
            'd3feat_match_scores': [],
            'levels': [{'candidates': 0, 'valid': 0, 'below_min_points': 0,
                        'input_point_limit': self.fcgf_limits[i],
                        'min_points': self.fcgf_min_points[i]}
                       for i in range(len(self.fcgf_limits))]}

    def root(self, uid):
        if self.parent[uid] != uid:
            self.parent[uid] = self.root(self.parent[uid])
        return self.parent[uid]

    def _confirm(self, uid, now):
        track = self.tracks[uid]
        if self.confirmation_mode == 'cumulative_confidence':
            frames = len(track.get('confirmation_frames', []))
            confident = track.get('confirmation_confidence', 0.0) >= self.confirmation_threshold
            high_quality = (track.get('max_match_score', 0.0) >= self.confirmation_high_score and
                            frames >= self.confirmation_high_min_frames)
            ready = ((confident and frames >= self.confirmation_min_evidence_frames) or
                     high_quality)
        else:
            frames = {self.observations[i]['frame'] for i in track['members']}
            ready = len(frames) >= self.min_frames
        if ready and track['confirmed_frame'] is None:
            track['confirmed_frame'] = int(now)
            self.events.append({'type': 'confirmed', 'frame': int(now), 'object_id': uid,
                                'confirmation_mode': self.confirmation_mode,
                                'confidence': round(float(track.get('confirmation_confidence', 0.0)), 5),
                                'evidence_frames': int(frames) if isinstance(frames, int) else len(frames)})

    def _add_confirmation_evidence(self, uid, now, score):
        """Accumulate one independent temporal match for a track."""
        track = self.tracks[uid]
        frame = int(now)
        frames = track.setdefault('confirmation_frames', [])
        if frame in frames:
            return
        previous = track.get('last_confirmation_frame')
        if previous is not None:
            gap = max(1, frame - int(previous))
            track['confirmation_confidence'] *= self.confirmation_decay ** gap
        floor = min(max(self.confirmation_score_floor, 0.0), 0.99)
        gain = np.clip((float(score) - floor) / max(1.0 - floor, 1e-6), 0.0, 1.0)
        track['confirmation_confidence'] += float(gain)
        track['confirmation_evidence_count'] = int(track.get('confirmation_evidence_count', 0)) + 1
        frames.append(frame)
        track['last_confirmation_frame'] = frame
        track['max_match_score'] = max(float(track.get('max_match_score', 0.0)), float(score))
        self._confirm(uid, frame)

    def _union(self, ids, now, reason, evidence_score=None):
        roots = sorted({self.root(i) for i in ids})
        if not roots:
            return
        target = roots[0]
        if len(roots) == 1:
            self._confirm(target, now)
            return
        keep = self.tracks[target]
        accepted_sources = []
        for source in roots[1:]:
            other = self.tracks[source]
            if reason == 'OnlineAnySeg_mask_graph':
                if self.reject_covisible_graph_merge:
                    keep_frames = {self.observations[i]['frame'] for i in keep['members']}
                    other_frames = {self.observations[i]['frame'] for i in other['members']}
                    if keep_frames & other_frames:
                        # Conservative cannot-link constraint. Different masks
                        # seen together may be different objects. This opt-in
                        # ablation also blocks true within-frame split masks.
                        self.graph_merge_stats['rejected_covisible_edges'] += 1
                        continue
                self.graph_merge_stats['accepted_edges'] += 1
            other = self.tracks.pop(source)
            accepted_sources.append(source)
            self.parent[source] = target
            keep['members'].extend(other['members'])
            keep['points'] = sample_points(np.concatenate((keep['points'], other['points'])), self.sample_limit)
            keep['feature_sum'] += other['feature_sum']
            if self.geometry_backend == 'd3feat':
                keep['d3feat_payload'] = self._merge_d3feat_payload(
                    keep.get('d3feat_payload'), other.get('d3feat_payload'))
            elif other.get('fcgf_feature_sum') is not None:
                if keep.get('fcgf_feature_sum') is None:
                    keep['fcgf_feature_sum'] = other['fcgf_feature_sum'].copy()
                else:
                    keep['fcgf_feature_sum'] += other['fcgf_feature_sum']
                keep['fcgf_feature_count'] = (keep.get('fcgf_feature_count', 0) +
                                              other.get('fcgf_feature_count', 0))
            keep['last_frame'] = max(keep['last_frame'], other['last_frame'])
            if other['confirmed_frame'] is not None:
                keep['confirmed_frame'] = min(x for x in (keep['confirmed_frame'], other['confirmed_frame']) if x is not None)
            if self.confirmation_mode == 'cumulative_confidence':
                keep['confirmation_confidence'] = (
                    float(keep.get('confirmation_confidence', 0.0)) +
                    float(other.get('confirmation_confidence', 0.0)))
                keep['confirmation_evidence_count'] = (
                    int(keep.get('confirmation_evidence_count', 0)) +
                    int(other.get('confirmation_evidence_count', 0)))
                keep['confirmation_frames'] = sorted(set(
                    keep.get('confirmation_frames', []) + other.get('confirmation_frames', [])))
                keep['last_confirmation_frame'] = max(
                    [x for x in (keep.get('last_confirmation_frame'),
                                  other.get('last_confirmation_frame')) if x is not None],
                    default=None)
                keep['max_match_score'] = max(float(keep.get('max_match_score', 0.0)),
                                              float(other.get('max_match_score', 0.0)))
        keep['members'].sort(key=lambda i: (self.observations[i]['frame'], i))
        if accepted_sources:
            self.events.append({'type': 'merge', 'frame': int(now), 'target': target,
                                'sources': accepted_sources, 'reason': reason})
        if evidence_score is not None:
            self._add_confirmation_evidence(target, now, evidence_score)
        self._confirm(target, now)

    def graph_update(self, frame_id, instances):
        """Called after EVERY MaskGraph merge/prune, before IDs can be reused.

        Deleted instances remain in our lineage; deletion from the segmentation
        working set must not erase an object's first observation.
        """
        if not self.enabled:
            return
        snapshot = []
        for merged_id, inst in instances.items():
            ids = sorted(self.key_to_id[k] for k in inst.ori_mask_list if k in self.key_to_id)
            if not ids:
                continue
            self._union(ids, frame_id, 'OnlineAnySeg_mask_graph')
            snapshot.append({'merged_mask_id': int(merged_id), 'observation_ids': ids})
        self.graph_history.append({'frame': int(frame_id), 'instances': snapshot})
        self.latest_graph = {x['merged_mask_id']: x['observation_ids'] for x in snapshot}

    def _project_mask(self, depth, pose, mask):
        y, x = np.nonzero(mask & np.isfinite(depth) & (depth > self.near) & (depth < self.far))
        z = depth[y, x]
        cam = np.column_stack(((x-self.K[0,2])*z/self.K[0,0],
                               (y-self.K[1,2])*z/self.K[1,1], z))
        points = cam @ pose[:3,:3].T + pose[:3,3]
        if not len(points):
            return points.astype(np.float32)
        _, idx = np.unique(np.floor(points/self.voxel_size).astype(np.int32), axis=0, return_index=True)
        return sample_points(points[np.sort(idx)].astype(np.float32), self.sample_limit)

    def _cap_d3feat_payload(self, payload):
        if payload is None:
            return None
        n = len(payload['keypoints'])
        if n <= self.d3feat_max_keypoints:
            return payload
        order = np.argsort(payload['scores'])[::-1][:self.d3feat_max_keypoints]
        return {k: np.asarray(v)[order].copy() for k, v in payload.items()}

    def _merge_d3feat_payload(self, first, second):
        if first is None:
            return self._cap_d3feat_payload(second)
        if second is None:
            return self._cap_d3feat_payload(first)
        merged = {k: np.concatenate((first[k], second[k]), axis=0)
                  for k in ('keypoints', 'features', 'scores')}
        return self._cap_d3feat_payload(merged)

    def _d3feat_payload(self, points, sample):
        result = self.pc_extractor.extract_local(sample)
        keypoints = np.asarray(result['points'], dtype=np.float32)
        descriptors = np.asarray(result['features'], dtype=np.float32)
        scores = np.asarray(result['scores'], dtype=np.float32).reshape(-1)
        if descriptors.ndim != 2 or len(keypoints) != len(descriptors) or len(scores) != len(keypoints):
            raise ValueError('D3Feat returned inconsistent keypoint arrays')
        if not np.isfinite(keypoints).all() or not np.isfinite(descriptors).all() or not np.isfinite(scores).all():
            raise ValueError('D3Feat returned non-finite keypoints, descriptors or scores')
        # Scores are detector confidences. Keep the strongest keypoints rather
        # than collapsing local descriptors into one global average.
        order = np.argsort(scores)[::-1]
        keep = order[:self.d3feat_max_keypoints]
        keypoints, descriptors, scores = keypoints[keep], descriptors[keep], scores[keep]
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / np.maximum(norms, 1e-8)
        return {'keypoints': keypoints, 'features': descriptors, 'scores': scores}

    @staticmethod
    def _confidence_weights(scores):
        scores = np.asarray(scores, dtype=np.float32)
        if len(scores) == 1 or float(scores.max() - scores.min()) < 1e-6:
            return np.ones(len(scores), dtype=np.float32)
        values = (scores - scores.min()) / (scores.max() - scores.min())
        return 0.25 + 0.75 * values

    def _d3feat_similarity(self, candidate, track):
        if candidate is None or track is None:
            return None
        a, b = candidate['features'], track['features']
        if len(a) < self.d3feat_min_keypoints or len(b) < self.d3feat_min_keypoints:
            self.fcgf_stats['d3feat_rejected'] += 1
            return None
        similarity = np.clip(a @ b.T, -1.0, 1.0)
        ab = np.argmax(similarity, axis=1)
        ba = np.argmax(similarity, axis=0)
        rows = np.arange(len(a))
        mutual = ba[ab] == rows
        mutual &= similarity[rows, ab] >= self.d3feat_min_pair_similarity
        count = int(mutual.sum())
        coverage = count / float(max(1, min(len(a), len(b))))
        if count < self.d3feat_min_mutual_matches or coverage < self.d3feat_min_match_coverage:
            self.fcgf_stats['d3feat_rejected'] += 1
            return None
        conf_a = self._confidence_weights(candidate['scores'])
        conf_b = self._confidence_weights(track['scores'])
        pair_weights = conf_a[mutual] * conf_b[ab[mutual]]
        pair_sim = similarity[rows[mutual], ab[mutual]]
        descriptor_score = float(np.sum(pair_weights * pair_sim) /
                                max(float(pair_weights.sum()), 1e-8))
        # Compare centered keypoint locations so a translating object can still
        # match; this is only a soft geometric tie-breaker.
        pa = candidate['keypoints'][mutual]
        pb = track['keypoints'][ab[mutual]]
        residual = np.linalg.norm((pa - pa.mean(0)) - (pb - pb.mean(0)), axis=1)
        geometry_score = float(np.exp(-np.median(residual) /
                                      max(2.0 * self.match_distance, 1e-4)))
        score = ((1.0 - self.d3feat_geom_weight) * descriptor_score +
                 self.d3feat_geom_weight * geometry_score)
        score = float(score * min(1.0, np.sqrt(coverage / 0.25)))
        self.fcgf_stats['d3feat_mutual_matches'] += count
        self.fcgf_stats['d3feat_match_scores'].append(round(score, 6))
        return score, count, coverage, descriptor_score, geometry_score

    def _fcgf_descriptor(self, points, scene_feature_cache=None):
        """Extract FCGF pooled or D3Feat local geometry evidence.

        When the caller already has the scene-level FCGF point features from
        Scene_rep, reuse them instead of launching one sparse-conv network
        call per mask. This preserves the same pooled descriptor definition
        while reducing thousands of tiny GPU launches to one scene extraction.
        """
        if not self.fcgf_enabled:
            return None, -1, 0
        count = int(len(points))
        self.fcgf_stats['candidates'] += 1
        level = 0
        while level < len(self.fcgf_thresholds) and count > self.fcgf_thresholds[level]:
            level += 1
        level_stats = self.fcgf_stats['levels'][level]
        level_stats['candidates'] += 1
        if count < self.fcgf_min_points[level]:
            self.fcgf_stats['below_min_points'] += 1
            level_stats['below_min_points'] += 1
            return None, level, count
        sample = sample_points(np.asarray(points, dtype=np.float32), self.fcgf_limits[level])
        if self.geometry_backend == 'fcgf' and scene_feature_cache is not None:
            cache_tree, cache_points, cache_features, radius = scene_feature_cache
            distances, indices = cache_tree.query(sample, k=1, distance_upper_bound=radius,
                                                  workers=1)
            valid = np.isfinite(distances)
            if int(valid.sum()) >= self.fcgf_min_points[level]:
                values = np.asarray(cache_features[np.asarray(indices[valid], dtype=np.int64)],
                                    dtype=np.float32)
                descriptor = values.mean(axis=0)
                norm = float(np.linalg.norm(descriptor))
                if norm > 1e-8 and np.isfinite(norm):
                    self.fcgf_stats['scene_feature_cache_hits'] += 1
                    self.fcgf_stats['valid_descriptors'] += 1
                    level_stats['valid'] += 1
                    return descriptor / norm, level, count
            self.fcgf_stats['scene_feature_cache_misses'] += 1
        try:
            if self.geometry_backend == 'd3feat':
                geometry = self._d3feat_payload(points, sample)
                if len(geometry['features']) < self.d3feat_min_keypoints:
                    raise ValueError('D3Feat returned too few keypoints')
            else:
                values = cpu(self.pc_extractor.infer_pts_feature(sample)).astype(np.float32)
                if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
                    raise ValueError('Geometry backend returned an empty or non-finite feature array')
                descriptor = values.mean(axis=0)
                norm = float(np.linalg.norm(descriptor))
                if norm <= 1e-8 or not np.isfinite(norm):
                    raise ValueError('Geometry pooled descriptor has zero norm')
                geometry = descriptor / norm
        except Exception as error:
            if self.geometry_backend == 'd3feat':
                raise RuntimeError('D3Feat submap extraction failed') from error
            self.fcgf_stats['extraction_errors'] += 1
            self.events.append({'type': 'fcgf_extraction_error', 'error': str(error),
                                'point_count': count, 'level': level})
            return None, level, count
        self.fcgf_stats['valid_descriptors'] += 1
        level_stats['valid'] += 1
        return geometry, level, count

    def _history_evidence(self, points):
        """Past RGB-D ray evidence, measured BEFORE insertion of this frame.

        Missing depth/outside-frustum/occluded samples are unknown, never free.
        Depth behind a current surface is positive free-space evidence only
        within the same historical camera ray. Require two separated views.
        """
        pts = sample_points(points, 512)
        free_views, known_views = [], []
        selected = []
        for old in reversed(self.history):
            w2c = old['w2c']
            p = pts @ w2c[:3,:3].T + w2c[:3,3]
            good = p[:,2] > self.near
            p = p[good]
            if not len(p):
                continue
            x = np.rint(p[:,0]*self.K[0,0]/p[:,2]+self.K[0,2]).astype(int)
            y = np.rint(p[:,1]*self.K[1,1]/p[:,2]+self.K[1,2]).astype(int)
            h,w = old['depth'].shape
            inside = (x>=0)&(x<w)&(y>=0)&(y<h)
            x,y,p = x[inside],y[inside],p[inside]
            d = old['depth'][y,x]
            valid = np.isfinite(d)&(d>self.near)&(d<self.far)
            if valid.sum() < max(30, int(0.2*len(pts))):
                continue
            difference = d[valid]-p[valid,2]
            free = float(np.mean(difference > self.free_margin))
            same = float(np.mean(np.abs(difference) < self.match_distance))
            if same >= 0.5:
                known_views.append(old['frame'])
            if free < self.free_ratio:
                continue
            # Repeated nearly identical historical views count once.
            pose = old['pose']
            distinct = True
            for prior in selected:
                angle = np.degrees(np.arccos(np.clip((np.trace(prior[:3,:3].T@pose[:3,:3])-1)/2,-1,1)))
                if np.linalg.norm(prior[:3,3]-pose[:3,3]) < 0.1 and angle < 5:
                    distinct = False
                    break
            if distinct:
                selected.append(pose)
                free_views.append({'frame': old['frame'], 'free_ratio': round(free,4)})
        return {'historical_free_views': free_views, 'historical_surface_views': known_views,
                'appearance_evidence': len(free_views)>=2,
                'interpretation': 'free_space_candidate' if len(free_views)>=2 else
                                  ('previous_surface' if known_views else 'unknown_or_occluded')}

    def before_frame(self, frame_id, depth, pose_c2w):
        """Prepare bounded depth history on mask frames, without self-validation."""
        if not self.enabled:
            return
        pose = cpu(pose_c2w).astype(np.float64)
        self.pending_history = {'frame': int(frame_id), 'depth': cpu(depth).copy(),
                                'pose': pose, 'w2c': np.linalg.inv(pose)}

    def observe(self, frame_id, color, depth, pose_c2w, seg_image,
                mask_features=None, merged_pairs=None,
                scene_feature_points=None, scene_feature_values=None):
        if not self.enabled:
            return
        segmentation, dep, pose = cpu(seg_image), cpu(depth), cpu(pose_c2w)
        feats = cpu(mask_features)
        scene_feature_cache = None
        if (self.geometry_backend == 'fcgf' and self.fcgf_enabled and
                self.fcgf_scene_cache and scene_feature_points is not None and
                scene_feature_values is not None):
            cache_points = cpu(scene_feature_points).astype(np.float32)
            cache_features = cpu(scene_feature_values).astype(np.float32)
            if (cache_points.ndim == 2 and cache_points.shape[1] == 3 and
                    cache_features.ndim == 2 and len(cache_points) == len(cache_features) and
                    len(cache_points)):
                scene_feature_cache = (
                    cKDTree(cache_points), cache_points, cache_features,
                    float(self.fcgf_opt.get('scene_feature_cache_radius_m', 0.08)))
        candidates = []
        # Empty accepted-pairs means no observation; do not silently use rejected masks.
        for pair in merged_pairs or []:
            mid, uid = int(pair['mask_id']), int(pair['global_mask_id'])
            mask = segmentation==mid
            if mask.sum() < self.min_pixels:
                continue
            points = self._project_mask(dep, pose, mask)
            if len(points)<12:
                continue
            feature = feats[mid-1].astype(np.float32)
            feature /= max(np.linalg.norm(feature), 1e-8)
            geo_feature, geo_level, geo_count = self._fcgf_descriptor(
                points, scene_feature_cache=scene_feature_cache)
            candidates.append((uid, mid, mask, points, feature, pair,
                               geo_feature, geo_level, geo_count))
        active = sorted(self.tracks)
        scores = np.full((len(candidates), len(active)), -1.0)
        for j, track_id in enumerate(active):
            t = self.tracks[track_id]
            tree = cKDTree(t['points'])
            embedding = t['feature_sum']/max(np.linalg.norm(t['feature_sum']),1e-8)
            if self.geometry_backend == 'd3feat':
                track_geo = t.get('d3feat_payload')
            else:
                track_geo = t.get('fcgf_feature_sum')
                if track_geo is not None:
                    track_geo = track_geo/max(np.linalg.norm(track_geo), 1e-8)
            for i, (_, _, _, pts, feature, _, geo_feature, _, _) in enumerate(candidates):
                self.fcgf_stats['matches_considered'] += 1
                if np.any(pts.min(0)>t['points'].max(0)+self.match_distance) or np.any(pts.max(0)<t['points'].min(0)-self.match_distance):
                    continue
                similarity = float(feature@embedding)
                if similarity < self.match_feature:
                    continue
                coverage = float(np.mean(tree.query(pts, workers=1)[0]<self.match_distance))
                if coverage < self.match_coverage:
                    continue
                base_score = 0.75*coverage+0.25*similarity
                if geo_feature is not None and track_geo is not None:
                    if self.geometry_backend == 'd3feat':
                        evidence = self._d3feat_similarity(geo_feature, track_geo)
                        if evidence is None:
                            # D3Feat is a soft geometric cue.  Hard rejection
                            # can fragment objects when only a few local
                            # keypoints survive occlusion or viewpoint change.
                            if self.d3feat_require_match:
                                continue
                            self.fcgf_stats['fallback_matches'] += 1
                            scores[i,j] = base_score
                            continue
                        geo_similarity, _, _, _, _ = evidence
                        weight = self.d3feat_score_weight
                    else:
                        geo_similarity = float(geo_feature@track_geo)
                        if geo_similarity < self.fcgf_min_similarity:
                            self.fcgf_stats['fcgf_rejected'] += 1
                            continue
                        self.fcgf_stats['matches_with_fcgf'] += 1
                        weight = min(max(self.fcgf_weight, 0.0), 1.0)
                    scores[i,j] = (1.0-weight)*base_score + weight*geo_similarity
                elif self.fcgf_require:
                    self.fcgf_stats['fcgf_rejected'] += 1
                    continue
                else:
                    self.fcgf_stats['fallback_matches'] += 1
                    scores[i,j] = base_score
        matches = {}
        if len(candidates) and len(active):
            # Reject ambiguous neighbors before one-to-one assignment.
            for i in range(len(candidates)):
                ranked = sorted(scores[i][scores[i]>=0], reverse=True)
                if len(ranked)>1 and ranked[0]-ranked[1]<self.margin:
                    scores[i,:] = -1
            rows, cols = linear_sum_assignment(-np.concatenate((scores, np.zeros((len(candidates),len(candidates)))),axis=1))
            for i,j in zip(rows,cols):
                if j<len(active) and scores[i,j]>0:
                    matches[i] = active[j]
        for i,(uid,mid,mask,pts,feature,pair,geo_feature,geo_level,geo_count) in enumerate(candidates):
            if uid in self.observations:
                raise ValueError('Duplicate append-only observation ID')
            self.parent[uid] = uid
            evidence = self._history_evidence(pts)
            self.observations[uid] = {'id':uid,'frame':int(frame_id),'mask_id':mid,
                'merged_mask_id_at_observation':int(pair['merged_mask_id']),
                'mask_shape':list(mask.shape),'mask_bits':np.packbits(mask),
                'feature':feature.astype(np.float32).tolist(),
                'fcgf_feature': (geo_feature.astype(np.float32).tolist()
                                 if self.geometry_backend != 'd3feat' and geo_feature is not None else None),
                'fcgf_valid': bool(geo_feature is not None),
                'geometry_backend': self.geometry_backend,
                'geometry_keypoint_count': (len(geo_feature['features'])
                                             if self.geometry_backend == 'd3feat' and geo_feature is not None else 0),
                'fcgf_level': int(geo_level),
                'fcgf_point_count': int(geo_count),
                'pose_c2w':pose.astype(float).tolist(), 'novelty':evidence}
            self.key_to_id[(int(frame_id),mid)] = uid
            self.tracks[uid] = {
                'members':[uid], 'points':pts, 'feature_sum':feature.copy(),
                'fcgf_feature_sum': (geo_feature.copy() if self.geometry_backend != 'd3feat' and geo_feature is not None else None),
                'fcgf_feature_count': int(geo_feature is not None),
                'd3feat_payload': (geo_feature if self.geometry_backend == 'd3feat' else None),
                'last_frame':int(frame_id), 'confirmed_frame':None,
                'confirmation_confidence':0.0, 'confirmation_evidence_count':0,
                'confirmation_frames':[], 'last_confirmation_frame':None,
                'max_match_score':0.0}
            self.events.append({'type':'candidate','frame':int(frame_id),'object_id':uid,'mask_id':mid})
            if i in matches:
                matched_track = matches[i]
                matched_column = active.index(matched_track)
                self._union([uid, matched_track], frame_id, 'geometry_and_embedding',
                            evidence_score=float(scores[i, matched_column]))
                self.fcgf_stats['accepted_matches'] += 1
        if self.pending_history is not None:
            self.history.append(self.pending_history)
            self.pending_history = None

    def _frame(self, frame_id):
        if frame_id not in self.frame_cache:
            color, depth, pose, _, _, _ = self.dataset[frame_id]
            self.frame_cache[frame_id] = ((cpu(color)*255).clip(0,255).astype(np.uint8),
                                          cpu(depth).copy(),cpu(pose).astype(np.float64))
            if len(self.frame_cache)>4:
                self.frame_cache.popitem(last=False)
                self.frame_cache.popitem(last=False)
        else:
            self.frame_cache.move_to_end(frame_id)
        return self.frame_cache[frame_id]

    @staticmethod
    def _weighted_cosine(first, second):
        """Cosine similarity between sparse coarse-region vectors."""
        if not first or not second:
            return 0.0
        if len(first) > len(second):
            first, second = second, first
        dot = sum(value * second.get(key, 0.0) for key, value in first.items())
        n_first = math.sqrt(sum(value * value for value in first.values()))
        n_second = math.sqrt(sum(value * value for value in second.values()))
        return float(dot / max(n_first * n_second, 1e-8))

    def _mv3dis_observation_guide(self, observation, global_points, global_tree):
        """Build a view-independent coarse 3D coverage vector for one mask.

        MV3DIS uses coarse 3D regions as a common reference for masks from
        different views.  Here the existing global TSDF surface is the guide;
        no ground-truth points or extra inference are needed.  A small coarse
        cell dilation makes opposite visible sides of one object share guide
        regions while the continuous nearest-surface weight suppresses depth
        outliers and occlusions.
        """
        _, depth, pose = self._frame(observation['frame'])
        mask = self._mask(observation)
        points = self._project_mask(depth, pose, mask)
        if not len(points) or not len(global_points):
            return {}
        guide_radius = float(self.opt.get('mv3dis_guide_radius_m', 0.14))
        guide_limit = int(self.opt.get('mv3dis_guide_sample_points', 768))
        points = sample_points(points, guide_limit)
        distance, indices = global_tree.query(points, k=1, workers=1)
        valid = np.isfinite(distance) & (distance <= guide_radius)
        if not valid.any():
            return {}
        nearest = global_points[np.asarray(indices[valid], dtype=np.int64)]
        distance = np.asarray(distance[valid], dtype=np.float32)
        cell_size = float(self.opt.get('mv3dis_coarse_voxel_m', 0.32))
        sigma = max(float(self.opt.get('mv3dis_depth_sigma_m', 0.08)), 1e-4)
        cells = np.floor(nearest / cell_size).astype(np.int64)
        # Continuous confidence, analogous to MV3DIS depth consistency.
        confidence = np.exp(-0.5 * (distance / sigma) ** 2).astype(np.float32)
        base = {}
        for cell, value in zip(cells, confidence):
            key = tuple(int(x) for x in cell)
            base[key] = max(base.get(key, 0.0), float(value))
        dilation = max(0, int(self.opt.get('mv3dis_cell_dilation', 1)))
        if dilation == 0:
            return base
        guide = dict(base)
        # Manhattan dilation is bounded and less aggressive than filling a
        # complete 3-D cube around every surface cell.
        offsets = [(0, 0, 0)]
        for radius in range(1, dilation + 1):
            offsets.extend([(radius, 0, 0), (-radius, 0, 0),
                            (0, radius, 0), (0, -radius, 0),
                            (0, 0, radius), (0, 0, -radius)])
        for (x, y, z), value in base.items():
            for dx, dy, dz in offsets[1:]:
                key = (x + dx, y + dy, z + dz)
                guide[key] = max(guide.get(key, 0.0), value * 0.55 ** (abs(dx) + abs(dy) + abs(dz)))
        return guide

    def _mv3dis_guided_reconcile(self, global_points):
        """Merge fragmented tracks using coarse 3D mask coverage.

        This is an export-time, geometry-preserving refinement.  It leaves the
        online TSDF and MaskGraph untouched and only changes which historical
        observations are re-fused into one object TSDF.  Thus the existing
        baseline fallback can still protect AP when the guide is uninformative.
        """
        enabled = bool(self.opt.get('mv3dis_guided_merge', False))
        self.mv3dis_stats = {'enabled': enabled, 'candidate_edges': 0,
                             'accepted_edges': 0, 'merged_tracks': 0,
                             'guide_observations': 0, 'guide_cells': 0}
        if not enabled or not self.tracks or global_points is None:
            return self.mv3dis_stats
        points = cpu(global_points).astype(np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
            return self.mv3dis_stats
        tree = cKDTree(points)
        guides = {}
        for observation_id, observation in self.observations.items():
            guide = self._mv3dis_observation_guide(observation, points, tree)
            if guide:
                guides[observation_id] = guide
                self.mv3dis_stats['guide_observations'] += 1
                self.mv3dis_stats['guide_cells'] += len(guide)

        roots = sorted(self.tracks)
        if len(roots) < 2:
            return self.mv3dis_stats
        track_guides = {}
        track_features = {}
        for root in roots:
            members = [i for i in self.tracks[root]['members'] if i in guides]
            track_guides[root] = members
            feature = self.tracks[root]['feature_sum']
            track_features[root] = feature / max(np.linalg.norm(feature), 1e-8)

        min_cosine = float(self.opt.get('mv3dis_min_guide_cosine', 0.30))
        min_feature = float(self.opt.get('mv3dis_min_feature_similarity', 0.72))
        min_score = float(self.opt.get('mv3dis_min_merge_score', 0.50))
        min_support = float(self.opt.get('mv3dis_min_observation_support', 0.50))
        only_unconfirmed = bool(self.opt.get('mv3dis_only_unconfirmed_merge', False))
        fragment_max_observations = int(self.opt.get('mv3dis_fragment_max_observations', 0))
        edges = []
        for left_index, left in enumerate(roots):
            if not track_guides[left]:
                continue
            for right in roots[left_index + 1:]:
                if not track_guides[right]:
                    continue
                if (only_unconfirmed and
                        self.tracks[left]['confirmed_frame'] is not None and
                        self.tracks[right]['confirmed_frame'] is not None):
                    continue
                if fragment_max_observations > 0:
                    left_confirmed = self.tracks[left]['confirmed_frame'] is not None
                    right_confirmed = self.tracks[right]['confirmed_frame'] is not None
                    if ((left_confirmed and len(self.tracks[right]['members']) > fragment_max_observations) or
                            (right_confirmed and len(self.tracks[left]['members']) > fragment_max_observations) or
                            (not left_confirmed and not right_confirmed and
                             (len(self.tracks[left]['members']) > fragment_max_observations or
                              len(self.tracks[right]['members']) > fragment_max_observations))):
                        continue
                best_cosine = 0.0
                best_pair = None
                left_support, right_support = [], []
                for left_observation in track_guides[left]:
                    best_left = 0.0
                    for right_observation in track_guides[right]:
                        cosine = self._weighted_cosine(guides[left_observation], guides[right_observation])
                        best_left = max(best_left, cosine)
                        if cosine > best_cosine:
                            best_cosine = cosine
                            best_pair = (left_observation, right_observation)
                    left_support.append(best_left)
                for right_observation in track_guides[right]:
                    right_support.append(max(
                        self._weighted_cosine(guides[left_observation], guides[right_observation])
                        for left_observation in track_guides[left]))
                self.mv3dis_stats['candidate_edges'] += 1
                if best_pair is None or best_cosine < min_cosine:
                    continue
                if len(left_support) > 1 and np.mean(np.asarray(left_support) >= min_cosine) < min_support:
                    continue
                if len(right_support) > 1 and np.mean(np.asarray(right_support) >= min_cosine) < min_support:
                    continue
                left_feature = np.asarray(self.observations[best_pair[0]]['feature'], dtype=np.float32)
                right_feature = np.asarray(self.observations[best_pair[1]]['feature'], dtype=np.float32)
                feature_similarity = float(left_feature @ right_feature)
                if feature_similarity < min_feature:
                    continue
                score = 0.60 * best_cosine + 0.40 * feature_similarity
                if score < min_score:
                    continue
                edges.append((score, left, right, best_cosine, feature_similarity, best_pair))
        edges.sort(reverse=True)
        for score, left, right, cosine, feature_similarity, pair in edges:
            left_root, right_root = self.root(left), self.root(right)
            if left_root == right_root:
                continue
            old_left, old_right = left_root, right_root
            if (only_unconfirmed and
                    self.tracks[old_left]['confirmed_frame'] is not None and
                    self.tracks[old_right]['confirmed_frame'] is not None):
                continue
            if fragment_max_observations > 0:
                left_confirmed = self.tracks[old_left]['confirmed_frame'] is not None
                right_confirmed = self.tracks[old_right]['confirmed_frame'] is not None
                if ((left_confirmed and len(self.tracks[old_right]['members']) > fragment_max_observations) or
                        (right_confirmed and len(self.tracks[old_left]['members']) > fragment_max_observations) or
                        (not left_confirmed and not right_confirmed and
                         (len(self.tracks[old_left]['members']) > fragment_max_observations or
                          len(self.tracks[old_right]['members']) > fragment_max_observations))):
                    continue
            self._union([left_root, right_root],
                        max(self.observations[pair[0]]['frame'], self.observations[pair[1]]['frame']),
                        'mv3dis_3d_guide', evidence_score=float(score))
            self.mv3dis_stats['accepted_edges'] += 1
            self.mv3dis_stats['merged_tracks'] += 1
            self.events.append({'type': 'mv3dis_guide_match',
                                'target': int(old_left),
                                'source': int(old_right),
                                'score': round(float(score), 5),
                                'guide_cosine': round(float(cosine), 5),
                                'feature_similarity': round(float(feature_similarity), 5),
                                'observation_ids': [int(pair[0]), int(pair[1])]})
        return self.mv3dis_stats

    @staticmethod
    def _mask(obs):
        return np.unpackbits(obs['mask_bits'], count=math.prod(obs['mask_shape'])).reshape(obs['mask_shape']).astype(bool)

    def _build_tsdf(self, observations, directory):
        # A single compact grid is alive at a time; all coordinates are local to
        # the earliest camera. Re-fusion after lineage merges prevents duplicate
        # integration of an observation and avoids resampling unrelated TSDFs.
        anchor = np.asarray(observations[0]['pose_c2w'])
        device = o3c.Device(self.device)
        grid = o3d.t.geometry.VoxelBlockGrid(('tsdf','weight','color'),
            (o3c.float32,)*3, ((1,),(1,),(3,)), self.voxel_size, 8, 2048, device=device)
        intrinsic = o3c.Tensor.from_numpy(self.K)
        grouped = {}
        for obs in observations:
            grouped.setdefault(obs['frame'],[]).append(obs)
        cap = int(self.opt.get('max_object_blocks',30000))
        integrated = []
        for frame, views in sorted(grouped.items()):
            rgb,dep,pose = self._frame(frame)
            mask = np.logical_or.reduce([self._mask(o) for o in views])
            # Conservative one-pixel erosion avoids mixing adjacent object edges.
            inner = cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
            if inner.sum()<12:
                inner = mask
            valid = inner & np.isfinite(dep)&(dep>self.near)&(dep<self.far)
            if not valid.any():
                continue
            depth_u16 = np.ascontiguousarray(
                np.where(valid, dep * self.dataset.depth_scale, 0)
                .clip(0, 65535).astype(np.uint16))
            depth = o3d.t.geometry.Image(depth_u16).to(o3c.uint16).to(device)
            color = o3d.t.geometry.Image(np.ascontiguousarray(rgb)).to(device)
            extrinsic = o3c.Tensor(np.linalg.inv(pose)@anchor)
            try:
                blocks = grid.compute_unique_block_coordinates(
                    depth, intrinsic, extrinsic, self.dataset.depth_scale, self.far)
            except RuntimeError as error:
                # Sparse masks can leave a valid observation with no depth ray
                # touching this object's local grid.  It contributes no TSDF
                # geometry and should not abort the whole sequence export.
                if 'No block is touched' in str(error):
                    continue
                raise
            if blocks.shape[0] == 0:
                continue
            _, exists = grid.hashmap().find(blocks)
            extra = int((~exists.cpu().numpy()).sum())
            if grid.hashmap().size()+extra>cap:
                raise RuntimeError('Object TSDF exceeds configured block budget; refusing to drop geometry')
            grid.integrate(blocks, depth, color, intrinsic, extrinsic,
                          self.dataset.depth_scale, self.far)
            integrated.append(frame)
        block_count = int(grid.hashmap().size())
        self.max_blocks_seen = max(self.max_blocks_seen,block_count)
        if not integrated:
            del grid
            return (np.empty((0, 3), dtype=np.float32),
                    np.empty((0, 3), dtype=np.float32),
                    {'block_count': 0, 'surface_points': 0,
                     'integrated_frames': [], 'T_world_submap': anchor.tolist(),
                     'voxel_size_m': self.voxel_size, 'extraction_weight': 1.0})
        cloud = grid.extract_point_cloud(weight_threshold=1.0)
        local = cloud.point.positions.cpu().numpy().copy()
        rgb = cloud.point.colors.cpu().numpy().copy()
        if bool(self.opt.get('save_tsdf', True)):
            grid.save(str(directory/'tsdf_local.npz'))
        world = local@anchor[:3,:3].T+anchor[:3,3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(world)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb,0,1))
        if len(world):
            if not o3d.io.write_point_cloud(str(directory/'surface_world.ply'),pcd):
                raise RuntimeError('Failed writing object surface')
        del cloud, grid
        return world,rgb,{'block_count':block_count,'surface_points':len(world),
            'integrated_frames':integrated,'T_world_submap':anchor.tolist(),
            'voxel_size_m':self.voxel_size,'extraction_weight':1.0}

    def _render(self, points, colors, obs, depth):
        h,w = depth.shape
        result = np.full((h,w,3),24,dtype=np.uint8)
        coverage = np.zeros((h,w),bool)
        if not len(points):
            return result,coverage
        w2c = np.linalg.inv(np.asarray(obs['pose_c2w']))
        p = points@w2c[:3,:3].T+w2c[:3,3]
        good = p[:,2]>self.near
        p,c = p[good], colors[good]
        x = np.rint(p[:,0]*self.K[0,0]/p[:,2]+self.K[0,2]).astype(int)
        y = np.rint(p[:,1]*self.K[1,1]/p[:,2]+self.K[1,2]).astype(int)
        valid = (x>=0)&(x<w)&(y>=0)&(y<h)
        x,y,p,c = x[valid],y[valid],p[valid],c[valid]
        observed = depth[y,x]
        visible = (observed<=0)|(p[:,2]<=observed+0.06)
        x,y,p,c = x[visible],y[visible],p[visible],c[visible]
        # Surface splats with a z-buffer, matching first-view occlusion.
        zbuf = np.full((h,w),np.inf)
        for dx,dy in [(0,0),(1,0),(-1,0),(0,1),(0,-1)]:
            u,v=x+dx,y+dy
            valid=(u>=0)&(u<w)&(v>=0)&(v<h)
            u,v,z,col=u[valid],v[valid],p[valid,2],c[valid]
            order=np.argsort(z)
            pixel=v[order]*w+u[order]
            _,first=np.unique(pixel,return_index=True)
            ix=order[first]
            win=z[ix]<zbuf[v[ix],u[ix]]
            ix=ix[win]
            result[v[ix],u[ix]]=(np.clip(col[ix],0,1)*255).astype(np.uint8)
            zbuf[v[ix],u[ix]]=z[ix]
            coverage[v[ix],u[ix]]=True
        return result,coverage

    @staticmethod
    def _font(size):
        path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
        return ImageFont.truetype(path,size) if Path(path).exists() else ImageFont.load_default()

    def _export_global_instances(self, output_dir, global_points, global_colors,
                                 object_surfaces):
        """Project confirmed object surfaces back onto the global TSDF points.

        The global reconstruction remains the geometry source.  Submap TSDF
        surfaces only decide which global points belong to each persistent
        object, so the result can be consumed by the existing AP evaluator.
        Each point is assigned to its closest object surface within a strict
        radius; overlapping object hypotheses therefore cannot double-label a
        point.
        """
        points = cpu(global_points).astype(np.float32)
        colors = cpu(global_colors).astype(np.float32) if global_colors is not None else None
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError('Global reconstruction points are empty or malformed')
        assign_distance = float(self.opt.get('global_assign_distance_m', 0.075))
        min_points = int(self.opt.get('global_min_points', 100))
        best_distance = np.full(len(points), np.inf, dtype=np.float32)
        best_object = np.full(len(points), -1, dtype=np.int32)
        for uid, surface, feature, entry in object_surfaces:
            if len(surface) < 3:
                continue
            tree = cKDTree(np.asarray(surface, dtype=np.float32))
            distance, _ = tree.query(points, k=1, distance_upper_bound=assign_distance,
                                     workers=1)
            valid = np.isfinite(distance) & (distance < best_distance)
            best_distance[valid] = distance[valid]
            best_object[valid] = int(uid)

        selected = []
        for uid, surface, feature, entry in object_surfaces:
            mask = best_object == int(uid)
            if int(mask.sum()) < min_points:
                continue
            selected.append((uid, mask, feature, entry))
        selected.sort(key=lambda item: item[0])
        if selected:
            pred_masks = np.stack([item[1] for item in selected], axis=1)
            features = []
            for _, _, feature, _ in selected:
                value = np.asarray(feature, dtype=np.float32)
                value = value / max(float(np.linalg.norm(value)), 1e-8)
                features.append(value)
            pred_sem_features = np.stack(features, axis=1)
        else:
            pred_masks = np.zeros((len(points), 0), dtype=bool)
            pred_sem_features = np.zeros((0, 0), dtype=np.float32)

        # Conservative hybrid export: use the original global Scene_rep
        # instances only to fill points that a confident submap already
        # identifies as the same object.  This keeps the submap instance count
        # and avoids adding duplicate baseline instances, while recovering
        # holes caused by sparse views and the surface-distance assignment.
        hybrid = bool(self.opt.get('hybrid_baseline_fill', False))
        hybrid_filled_points = 0
        hybrid_matched = 0
        baseline_path = Path(output_dir) / 'ckpt_scene_rep_final.npz'
        if hybrid and pred_masks.shape[1] and baseline_path.exists():
            baseline = np.load(baseline_path)
            baseline_masks = np.asarray(baseline.get('pred_masks'), dtype=bool)
            if baseline_masks.ndim == 2 and baseline_masks.shape[0] == len(points):
                min_overlap = float(self.opt.get('hybrid_min_baseline_overlap', 0.50))
                min_iou = float(self.opt.get('hybrid_min_baseline_iou', 0.20))
                covered = pred_masks.any(axis=1)
                for col in range(baseline_masks.shape[1]):
                    base_mask = baseline_masks[:, col]
                    base_count = int(base_mask.sum())
                    if base_count == 0:
                        continue
                    inter = (pred_masks & base_mask[:, None]).sum(axis=0)
                    best = int(inter.argmax())
                    overlap = float(inter[best]) / max(base_count, 1)
                    union = int(pred_masks[:, best].sum()) + base_count - int(inter[best])
                    iou = float(inter[best]) / max(union, 1)
                    if overlap < min_overlap or iou < min_iou:
                        continue
                    fill = base_mask & ~covered
                    added = int(fill.sum())
                    if added:
                        pred_masks[fill, best] = True
                        covered[fill] = True
                        hybrid_filled_points += added
                    hybrid_matched += 1

        export_mode = 'submap'
        export_score = np.ones(pred_masks.shape[1], dtype=np.float64)
        export_classes = np.zeros(pred_masks.shape[1], dtype=np.int32)
        baseline_path = Path(output_dir) / 'ckpt_scene_rep_final.npz'
        if bool(self.opt.get('baseline_fallback_when_submaps_not_richer', False)) and baseline_path.exists():
            baseline = np.load(baseline_path)
            baseline_masks = np.asarray(baseline.get('pred_masks'), dtype=bool)
            ratio_limit = float(self.opt.get('baseline_fallback_submap_count_ratio', 1.25))
            min_match_fraction = float(
                self.opt.get('baseline_fallback_min_match_fraction', 0.0))
            baseline_count = int(baseline_masks.shape[1]) if baseline_masks.ndim == 2 else 0
            matched_fraction = (float(hybrid_matched) / baseline_count
                                if baseline_count > 0 else 0.0)
            if (baseline_masks.ndim == 2 and baseline_masks.shape[0] == len(points) and
                    baseline_masks.shape[1] > 0 and
                    (pred_masks.shape[1] <= ratio_limit * baseline_masks.shape[1] or
                     matched_fraction < min_match_fraction)):
                # A submap result that is no richer than the global graph is
                # usually fragmented or missing low-support objects. Also keep
                # the global graph labels when the submap does not account for
                # enough baseline instances; instance count alone is not an
                # AP-quality guarantee.
                pred_masks = baseline_masks
                base_scores = baseline.get('pred_score')
                base_classes = baseline.get('pred_classes')
                export_score = (np.asarray(base_scores, dtype=np.float64)
                                if base_scores is not None and len(base_scores) == pred_masks.shape[1]
                                else np.ones(pred_masks.shape[1], dtype=np.float64))
                export_classes = (np.asarray(base_classes, dtype=np.int32)
                                  if base_classes is not None and len(base_classes) == pred_masks.shape[1]
                                  else np.zeros(pred_masks.shape[1], dtype=np.int32))
                base_features = baseline.get('pred_sem_features')
                if base_features is not None and np.asarray(base_features).ndim == 2:
                    pred_sem_features = np.asarray(base_features, dtype=np.float32)
                else:
                    pred_sem_features = np.zeros((0, pred_masks.shape[1]), dtype=np.float32)
                export_mode = 'baseline_fallback_submaps_not_richer'

        checkpoint_path = Path(output_dir) / 'ckpt_final.npz'
        np.savez_compressed(
            checkpoint_path,
            pred_masks=pred_masks,
            pred_score=export_score,
            pred_classes=export_classes,
            pred_sem_features=pred_sem_features)
        # Keep the geometry identical to the global reconstruction.  Only the
        # instance labels in ckpt_final.npz are replaced by persistent objects.
        if colors is not None and len(colors) == len(points):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
            o3d.io.write_point_cloud(str(Path(output_dir) / 'final.ply'), pcd)
        mapping = {
            'assign_distance_m': assign_distance,
            'global_point_count': int(len(points)),
            'exported_instance_count': int(pred_masks.shape[1]),
            'export_mode': export_mode,
            'instance_point_counts': {
                str(int(uid)): int(mask.sum()) for uid, mask, _, _ in selected},
            'source': 'confirmed object submap TSDF surfaces',
            'hybrid_baseline_fill': hybrid,
            'hybrid_matched_baseline_instances': int(hybrid_matched),
            'hybrid_filled_points': int(hybrid_filled_points),
        }
        (Path(output_dir) / 'global_submap_instances.json').write_text(
            json.dumps(mapping, indent=2), encoding='utf-8')
        return mapping

    def save_outputs(self, output_dir, global_points=None, global_colors=None):
        if not self.enabled:
            return
        out=Path(output_dir)
        objects_dir=out/'objects'
        objects_dir.mkdir(parents=True,exist_ok=True)
        if global_points is not None:
            self._mv3dis_guided_reconcile(global_points)
        groups=sorted(self.tracks.items(),key=lambda kv:(self.observations[kv[1]['members'][0]]['frame'],kv[0]))
        entries,tiles=[],[]
        object_surfaces=[]
        latest_for_obs={i:mid for mid,ids in self.latest_graph.items() for i in ids}
        for uid,track in groups:
            members=sorted(set(track['members']),key=lambda i:(self.observations[i]['frame'],i))
            views=[self.observations[i] for i in members]
            first_frame=views[0]['frame']
            entry={'object_id':uid,'first_frame':first_frame,
                   'first_mask_ids':[o['mask_id'] for o in views if o['frame']==first_frame],
                   'confirmed_frame':track['confirmed_frame'],
                   'status':'confirmed' if track['confirmed_frame'] is not None else 'tentative',
                   'observation_ids':members,'distinct_frames':len({o['frame'] for o in views}),
                   'final_merged_mask_ids':sorted({latest_for_obs[i] for i in members if i in latest_for_obs})}
            free_observations=[o for o in views if o['novelty']['appearance_evidence']]
            entry['appearance_evidence_frames']=sorted({o['frame'] for o in free_observations})
            entry['kind']='first_observed'
            # Appearance must be present at the first recorded observation and
            # supported again in another current frame, not just years later.
            first_evidence=any(o['novelty']['appearance_evidence'] for o in views if o['frame']==first_frame)
            if first_evidence and len(entry['appearance_evidence_frames'])>=2:
                entry['kind']='free_space_appearance_candidate'
            entries.append(entry)
            if entry['status']!='confirmed':
                continue
            directory=objects_dir/('object_%04d'%uid)
            directory.mkdir(exist_ok=True)
            rgb,depth,_=self._frame(first_frame)
            mask=np.logical_or.reduce([self._mask(o) for o in views if o['frame']==first_frame])
            Image.fromarray(rgb).save(directory/'first_rgb.png')
            Image.fromarray(mask.astype(np.uint8)*255).save(directory/'first_mask.png')
            overlay=(rgb.astype(np.float32)*0.30).astype(np.uint8)
            overlay[mask]=rgb[mask]
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(255,210,30),1)
            Image.fromarray(overlay).save(directory/'first_overlay.png')
            points,colors,tsdf=self._build_tsdf(views,directory)
            rendered,coverage=self._render(points,colors,views[0],depth)
            Image.fromarray(rendered).save(directory/'first_view_tsdf.png')
            entry['tsdf']=tsdf
            entry['render_mask_coverage']=float((coverage&mask).sum()/max(mask.sum(),1))
            entry['directory']=str(directory.relative_to(out))
            object_surfaces.append((uid, points, track['feature_sum'], entry))
            with (directory/'metadata.json').open('w') as f:
                json.dump(entry,f,indent=2)
            # Full first camera view, no arbitrary crop or change of viewpoint.
            h,w=mask.shape
            tile=Image.new('RGB',(600,320),(24,24,24))
            draw=ImageDraw.Draw(tile)
            draw.text((12,8),f"Object {uid:03d}  |  first frame {first_frame}",font=self._font(19),fill='white')
            for x,img,label in [(10,overlay,'First RGB + mask'),(310,rendered,'Object TSDF: same camera')]:
                image=Image.fromarray(img)
                image.thumbnail((280,218))
                image=image.resize((280,round(h*280/w)),Image.Resampling.NEAREST)
                tile.paste(image,(x,65))
                draw.text((x,39),label,font=self._font(15),fill=(215,215,215))
            draw.text((12,286),f"{entry['distinct_frames']} views | {entry['kind']}",font=self._font(14),fill=(235,235,235))
            tiles.append(tile)
            print('[ObjectSubmaps] object=%d first=%d views=%d blocks=%d points=%d'%(
                uid,first_frame,entry['distinct_frames'],tsdf['block_count'],tsdf['surface_points']),flush=True)
        columns=3
        rows=max(1,math.ceil(len(tiles)/columns))
        atlas=Image.new('RGB',(600*columns,rows*320+70),(24,24,24))
        title=ImageDraw.Draw(atlas)
        title.text((15,10),f"Object submaps: {len(tiles)} confirmed | earliest observed mask frame",font=self._font(26),fill='white')
        title.text((15,44),'Yellow: source mask. Right: TSDF surface rendered in that first camera. IDs may skip after merges.',font=self._font(17),fill=(220,220,220))
        for i,tile in enumerate(tiles):
            atlas.paste(tile,((i%columns)*600,70+(i//columns)*320))
        if not tiles:
            title.text((15,100),'No tracks reached multi-frame confirmation.',font=self._font(20),fill='white')
        atlas.save(out/'new_objects_first_view_atlas.png')
        global_mapping = None
        if global_points is not None:
            global_mapping = self._export_global_instances(
                out, global_points, global_colors, object_surfaces)
        np.savez_compressed(out/'observation_masks.npz',**{'obs_%05d'%i:o['mask_bits'] for i,o in self.observations.items()})
        observations=[]
        for i,o in self.observations.items():
            item={k:v for k,v in o.items() if k!='mask_bits'}
            item['final_object_id']=self.root(i)
            item['final_merged_mask_id']=latest_for_obs.get(i)
            observations.append(item)
        metadata={'schema_version':2,'mode':'online_object_lineage_sequential_masked_tsdf_refusion',
                  'groundtruth_used':False,'confirmation_mode':self.confirmation_mode,
                  'confirmation_distinct_frames':self.min_frames,
                  'confirmation_confidence_threshold':self.confirmation_threshold,
                  'fcgf_online_matching':self.fcgf_stats,
                  'graph_merge_stats':self.graph_merge_stats,
                  'reject_covisible_graph_merge':self.reject_covisible_graph_merge,
                  'mv3dis_guided_merge':self.mv3dis_stats,
                  'observations':observations,'events':self.events,'mask_graph_history':self.graph_history,
                  'objects':entries,'confirmed_count':len(tiles),'tentative_count':len(entries)-len(tiles),
                  'max_object_blocks_seen':self.max_blocks_seen,'max_concurrent_object_tsdfs':1,
                  'global_instance_export':global_mapping,
                  'source_dataset':str(self.dataset.data_location),
                  'source_masks':str(self.dataset.instance_dir),
                  'note':'First observed is not proof of physical arrival. No GT or session annotations used. Mask graph false merges remain possible.'}
        (out/'new_object_tracks.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        summary = dict(self.fcgf_stats)
        if self.geometry_backend == 'd3feat' and self.pc_extractor is not None and hasattr(self.pc_extractor, 'runtime_summary'):
            summary['d3feat_runtime'] = self.pc_extractor.runtime_summary()
        if summary.get('d3feat_match_scores'):
            values = summary['d3feat_match_scores']
            summary['d3feat_match_score_mean'] = float(np.mean(values))
            summary['d3feat_match_score_p50'] = float(np.percentile(values, 50))
            summary['d3feat_match_score_p90'] = float(np.percentile(values, 90))
        (out/'submap_fcgf_summary.json').write_text(
            json.dumps(summary, indent=2), encoding='utf-8')
        print('[ObjectSubmaps] confirmed=%d tentative=%d atlas=%s'%(len(tiles),len(entries)-len(tiles),out/'new_objects_first_view_atlas.png'),flush=True)

