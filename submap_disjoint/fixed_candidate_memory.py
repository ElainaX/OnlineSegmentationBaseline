"""Same candidates and per-frame evidence; replace only evidence bookkeeping."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .readonly_detection_core import (
    Candidate, evidence_stats, independent_observation,
    SUPPORT, CONTRADICTION, SOURCE_DEPTH, SOURCE_RGB,
)


from .batched_evidence import BatchedCandidate
from .sensor_reference import SurfaceMemory


ACCEPTED = {'support', 'cancelled', 'contradiction', 'would_remove'}


class PointMemory:
    def __init__(self, n):
        self.score = np.zeros(n, np.int8)
        self.depth_score = np.zeros(n, np.int8)
        self.rgb_score = np.zeros(n, np.int8)
        self.tested = np.zeros(n, bool)
        self.last_pose = None
        self.last_frame = -1
        self.accepted_frames = []
        self.confirmed = False
        self.confirmed_frame = None
        self.first_confirmed_frame = None
        self.confirmations = 0
        self.revocations = 0

    def eligible(self, frame, pose, codes, sources, cfg):
        stats = evidence_stats(codes, sources=sources)
        return (stats['tested'] >= cfg['min_reference_tested_points']
                and independent_observation(pose, self.last_pose, frame, self.last_frame, cfg)
                and (stats['contradiction_ratio'] >= cfg['contradiction_ratio']
                     or stats['support_ratio'] >= cfg['cancel_support_ratio']))

    def update(self, frame, pose, codes, sources, cfg):
        codes, sources = np.asarray(codes), np.asarray(sources)
        support = codes == SUPPORT
        conflict = codes == CONTRADICTION
        self.tested |= support | conflict
        self.score = np.clip(self.score.astype(int) + conflict - support.astype(int), -3, 3).astype(np.int8)
        self.depth_score = np.clip(self.depth_score.astype(int)
            + ((sources & SOURCE_DEPTH) != 0) - support.astype(int), -3, 3).astype(np.int8)
        self.rgb_score = np.clip(self.rgb_score.astype(int)
            + ((sources & SOURCE_RGB) != 0) - support.astype(int), -3, 3).astype(np.int8)
        self.last_frame, self.last_pose = frame, pose.copy()
        self.accepted_frames.append(frame)
        # Keep original candidate-level coverage and observation thresholds.
        active = self.score >= cfg['confirm_observations']
        now = (len(self.accepted_frames) >= cfg['confirm_observations']
               and self.tested.mean() >= cfg['min_cumulative_tested_fraction']
               and active.mean() >= cfg['min_contradicted_fraction'])
        if now and not self.confirmed:
            self.confirmations += 1
            self.confirmed_frame = frame
            if self.first_confirmed_frame is None:
                self.first_confirmed_frame = frame
        if self.confirmed and not now:
            self.revocations += 1
        self.confirmed = bool(now)

    def metadata(self, cfg):
        stable = self.score >= cfg['confirm_observations']
        return dict(state='would_remove' if self.confirmed else 'suspected',
                    confirmed_frame=self.confirmed_frame,
                    first_confirmed_frame=self.first_confirmed_frame,
                    accepted_frames=self.accepted_frames,
                    confirmations=self.confirmations, revocations=self.revocations,
                    tested_fraction=float(self.tested.mean()),
                    stable_fraction=float(stable.mean()),
                    stable_depth_fraction=float((stable & (self.depth_score > 0)).mean()))


class FixedCandidateObserver:
    """Original detector is the sole gatekeeper of candidate identities/lifetimes.

common: exactly original accepted observation frames; extended: same frame gates
but also inspect frozen candidates after original cancellation/confirmation.
The latter is reported separately, never credited to the strict common test.
"""
    def __init__(self, output_dir, cfg):
        self.out = Path(output_dir) / 'fixed_candidate_memory'
        self.out.mkdir(exist_ok=False)
        self.cfg = cfg
        self.records = {}
        self.seconds = 0.
        self.replay_checks = 0
        self.common_frame_checks = 0

    def observe(self, candidate, frame, pose, codes, sources):
        start = time.perf_counter()
        cid = candidate.candidate_id
        if cid not in self.records:
            assert frame == candidate.first_frame
            original = Candidate(cid, candidate.source_raw_ids, candidate.source_slot,
                                 candidate.first_frame, candidate.points, candidate.colors)
            self.records[cid] = dict(original=original,
                batched=BatchedCandidate(cid, candidate.source_raw_ids, candidate.source_slot,
                                         candidate.first_frame, candidate.points, candidate.colors),
                common=PointMemory(len(candidate.points)), extended=PointMemory(len(candidate.points)),
                frames=[], poses=[], codes=[], sources=[], common_frames=[], original_accepted=[])
        r = self.records[cid]
        r['batched'].observe(frame, pose, codes, sources, self.cfg)
        if 'surface_depth' not in r:
            r['surface_depth'] = SurfaceMemory(len(candidate.points))
        depth_codes = np.asarray(codes, np.uint8).copy()
        depth_codes[(np.asarray(sources) & SOURCE_RGB) != 0] = SUPPORT
        r['surface_depth'].observe(frame, pose, depth_codes, self.cfg)
        common = candidate.state == 'suspected'
        r['frames'].append(frame)
        r['poses'].append(pose.copy())
        r['codes'].append(np.asarray(codes, np.uint8).copy())
        r['sources'].append(np.asarray(sources, np.uint8).copy())
        r['common_frames'].append(common)
        reason = 'terminal'
        if common:
            reason, _ = r['original'].observe(frame, pose, codes, sources, self.cfg)
        accepted = common and reason in ACCEPTED
        r['original_accepted'].append(accepted)
        if accepted:
            r['common'].update(frame, pose, codes, sources, self.cfg)
        eligible = r['extended'].eligible(frame, pose, codes, sources, self.cfg)
        # Until original terminates, independent view selection must coincide.
        if common and not any(not x for x in r['common_frames']):
            assert eligible == accepted, (cid, frame, reason)
            self.common_frame_checks += 1
        if eligible:
            r['extended'].update(frame, pose, codes, sources, self.cfg)
        self.seconds += time.perf_counter() - start

    def verify_original(self, candidate):
        """Every actual baseline update must equal unmodified rule replay."""
        old = self.records[candidate.candidate_id]['original']
        for name in ('state', 'hits', 'support_hits', 'confirmed_frame', 'cancelled_frame', 'last_observation_frame'):
            assert getattr(candidate, name) == getattr(old, name), (candidate.candidate_id, name)
        for name in ('tested', 'contradicted', 'depth_contradicted', 'rgb_contradicted'):
            assert np.array_equal(getattr(candidate, name), getattr(old, name)), name
        self.replay_checks += 1

    def finalize(self, originals):
        assert len(originals) == len(self.records)
        records = []
        for candidate in originals:
            cid = candidate.candidate_id
            r = self.records[cid]
            self.verify_original(candidate)
            arrays = dict(reference_points=candidate.points, reference_colors=candidate.colors,
                frames=np.array(r['frames']), poses=np.array(r['poses']),
                evidence=np.array(r['codes']), sources=np.array(r['sources']),
                original_live=np.array(r['common_frames']), original_accepted=np.array(r['original_accepted']),
                original_depth_contradicted=candidate.depth_contradicted,
                original_rgb_contradicted=candidate.rgb_contradicted)
            variants = {}
            for name in ('common', 'extended'):
                m = r[name]
                variants[name] = m.metadata(self.cfg)
                arrays[name+'_score'] = m.score
                arrays[name+'_depth_score'] = m.depth_score
                arrays[name+'_rgb_score'] = m.rgb_score
                arrays[name+'_tested'] = m.tested
            assert np.array_equal(np.array(r['common'].accepted_frames), np.array(r['frames'])[r['original_accepted']])
            arrays['batched_depth_contradicted'] = r['batched'].depth_contradicted
            arrays['batched_rgb_contradicted'] = r['batched'].rgb_contradicted
            arrays['batched_pending_codes'] = r['batched'].pending_codes
            arrays['surface_depth_stable'] = r['surface_depth'].stable
            arrays['surface_depth_streak'] = r['surface_depth'].streak
            np.savez_compressed(self.out / f'{cid:04d}.npz', **arrays)
            records.append(dict(candidate_id=cid, first_frame=candidate.first_frame,
                source_raw_ids=sorted(candidate.source_raw_ids),
                point_sha256=hashlib.sha256(candidate.points.tobytes()).hexdigest(),
                original=candidate.metadata(), batched=r['batched'].metadata(),
                surface_depth=r['surface_depth'].metadata(), **variants))
        summary = dict(completed=True, candidates=len(records), original_replay_checks=self.replay_checks,
            common_frame_checks=self.common_frame_checks, observation_sets_identical=True,
            accumulator_seconds=self.seconds, actual_deletions=0, records=records,
            protocol='original proposals only; common exact accepted observations; extended future observations separately; same RGB/depth/occlusion/gates; batched variant collects small views into 100-point packets')
        (self.out / 'summary.json').write_text(json.dumps(summary, indent=2))
        return summary
