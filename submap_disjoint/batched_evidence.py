"""Collect small reliable observations before applying existing candidate rules.

No new candidates, GT, geometry mutation, or lower packet-size threshold.
Distinct frozen reference indices count once per packet. Their latest reliable
measurement replaces older evidence; unknown/occluded observations do not vote.
"""
import numpy as np
from .local_clear_only import LocalClearCandidate
from .readonly_detection_core import (
    SUPPORT, CONTRADICTION, UNKNOWN, independent_observation, evidence_stats,
)


class BatchedCandidate:
    def __init__(self, candidate_id, source_raw_ids, source_slot, first_frame, points, colors):
        self.inner = LocalClearCandidate(candidate_id, source_raw_ids, source_slot,
                                        first_frame, points, colors)
        n = len(self.inner.points)
        self.pending_codes = np.full(n, UNKNOWN, np.uint8)
        self.pending_sources = np.zeros(n, np.uint8)
        self.pending_frames = np.full(n, -1, np.int64)
        self.ever_tested = np.zeros(n, bool)
        self.last_input_pose = None
        self.last_input_frame = -1
        self.accepted_input_frames = []
        self.rejected_correlated_frames = 0
        self.packet_events = []
        self.last_seen_frame = -1

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def observe(self, frame, pose, codes, sources, cfg):
        self.last_seen_frame = int(frame)
        if self.inner.state != 'suspected':
            return 'terminal', {}
        codes = np.asarray(codes, np.uint8)
        sources = np.asarray(sources, np.uint8)
        if len(codes) != len(self.pending_codes) or len(sources) != len(codes):
            raise ValueError('Frozen reference point count changed')
        reliable = (codes == SUPPORT) | (codes == CONTRADICTION)
        if not reliable.any():
            return 'paused_no_reliable_points', {}
        # A small view must never consume the eligibility of a later full view.
        # Full observations use the original candidate's committed-view clock.
        # On acceptance they supersede the uncommitted small-view packet.
        if int(reliable.sum()) >= cfg['min_reference_tested_points']:
            reason, stats = self.inner.observe(frame, pose, codes, sources, cfg)
            if reason in ('support', 'cancelled', 'contradiction', 'would_remove'):
                self.last_input_pose = np.asarray(pose).copy()
                self.last_input_frame = int(frame)
                self.accepted_input_frames.append(int(frame))
                self.ever_tested |= reliable
                self.packet_events.append(dict(frame=int(frame), first_evidence_frame=int(frame),
                    last_evidence_frame=int(frame), distinct_source_frames=1,
                    distinct_points=int(reliable.sum()), reason=reason,
                    contradiction=int(stats['contradiction']), support=int(stats['support']),
                    contradiction_ratio=float(stats['contradiction_ratio']),
                    candidate_hits=int(self.inner.hits), kind='full_frame'))
                self.pending_codes.fill(UNKNOWN)
                self.pending_sources.fill(0)
                self.pending_frames.fill(-1)
            return reason, stats
        if not independent_observation(pose, self.last_input_pose, frame, self.last_input_frame, cfg):
            self.rejected_correlated_frames += 1
            return 'paused_correlated_input', {}
        self.last_input_pose = np.asarray(pose).copy()
        self.last_input_frame = int(frame)
        self.accepted_input_frames.append(int(frame))
        # Overwrite per point, never add duplicate point counts across frames.
        self.pending_codes[reliable] = codes[reliable]
        self.pending_sources[reliable] = sources[reliable]
        self.pending_frames[reliable] = frame
        self.ever_tested |= reliable
        pending = self.pending_codes != UNKNOWN
        count = int(pending.sum())
        if count < cfg['min_reference_tested_points']:
            return 'collecting_small_observations', dict(pending_distinct_points=count)
        stats = evidence_stats(self.pending_codes, sources=self.pending_sources)
        # A packet needs the same contradiction/support ratios as an old frame.
        # Mixed packets remain open; later reliable evidence can resolve them.
        reason, _ = self.inner.observe(frame, pose, self.pending_codes, self.pending_sources, cfg)
        if reason not in ('support', 'cancelled', 'contradiction', 'would_remove'):
            return reason, dict(pending_distinct_points=count)
        used_frames = self.pending_frames[pending]
        self.packet_events.append(dict(
            frame=int(frame), first_evidence_frame=int(used_frames.min()),
            last_evidence_frame=int(used_frames.max()),
            distinct_source_frames=int(len(np.unique(used_frames))),
            distinct_points=count, reason=reason,
            contradiction=int(stats['contradiction']), support=int(stats['support']),
            contradiction_ratio=float(stats['contradiction_ratio']),
            candidate_hits=int(self.inner.hits), kind='small_views_packet'))
        self.pending_codes.fill(UNKNOWN)
        self.pending_sources.fill(0)
        self.pending_frames.fill(-1)
        return reason, stats

    def metadata(self):
        return dict(self.inner.metadata(),
            accepted_input_frames=self.accepted_input_frames,
            last_seen_frame=self.last_seen_frame,
            rejected_correlated_frames=self.rejected_correlated_frames,
            pending_distinct_points=int((self.pending_codes != UNKNOWN).sum()),
            ever_tested_distinct_points=int(self.ever_tested.sum()),
            packet_events=self.packet_events)
