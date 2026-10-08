"""Strict one-change ablation: support clears only the observed support points.

All hit counters, cancellation, view gates, union evidence and confirm thresholds
are identical to original Candidate.observe. No point-score threshold is added.
"""
from .readonly_detection_core import *


class LocalClearCandidate(Candidate):
    def observe(self, frame, pose, codes, sources, cfg):
        stats = evidence_stats(codes, sources=sources)
        stats['cumulative_tested_fraction'] = float(self.tested.mean()) if len(self.points) else 0.
        stats['cumulative_contradicted_fraction'] = float(self.contradicted.mean()) if len(self.points) else 0.
        stats['cumulative_depth_fraction'] = float(self.depth_contradicted.mean()) if len(self.points) else 0.
        stats['cumulative_rgb_fraction'] = float(self.rgb_contradicted.mean()) if len(self.points) else 0.
        self.last_frame = frame
        if self.state != 'suspected':
            return 'terminal', stats
        # Each independent observation needs a minimum number of reference
        # points that can actually be tested. Coverage across different views
        # is accumulated below instead of requiring 12% in one frame.
        if stats['tested'] < cfg['min_reference_tested_points']:
            return 'paused_insufficient_reference_points', stats
        if not independent_observation(pose, self.last_observation_pose, frame,
                                       self.last_observation_frame, cfg):
            return 'paused_correlated_observation', stats
        contradiction = stats['contradiction_ratio'] >= cfg['contradiction_ratio']
        support = stats['support_ratio'] >= cfg['cancel_support_ratio']
        if not contradiction and not support:
            return 'paused_mixed_evidence', stats
        self.last_observation_frame, self.last_observation_pose = frame, pose.copy()
        if support:
            self.support_hits += 1
            # Only this surface part loses its stored evidence. All counters and decision gates remain unchanged.
            self.hits = 0
            self.tested[np.asarray(codes) == SUPPORT] = False
            self.contradicted[np.asarray(codes) == SUPPORT] = False
            self.depth_contradicted[np.asarray(codes) == SUPPORT] = False
            self.rgb_contradicted[np.asarray(codes) == SUPPORT] = False
            stats['cumulative_tested_fraction'] = float(self.tested.mean())
            stats['cumulative_contradicted_fraction'] = float(self.contradicted.mean())
            stats['cumulative_depth_fraction'] = float(self.depth_contradicted.mean())
            stats['cumulative_rgb_fraction'] = float(self.rgb_contradicted.mean())
            if self.support_hits >= cfg['cancel_observations']:
                self.state, self.cancelled_frame = 'cancelled', frame
                return 'cancelled', stats
            return 'support', stats
        self.support_hits = 0
        self.hits += 1
        current_tested = (np.asarray(codes) == SUPPORT) | (np.asarray(codes) == CONTRADICTION)
        self.tested |= current_tested
        self.contradicted |= np.asarray(codes) == CONTRADICTION
        self.depth_contradicted |= (np.asarray(sources) & SOURCE_DEPTH) != 0
        self.rgb_contradicted |= (np.asarray(sources) & SOURCE_RGB) != 0
        stats['cumulative_tested_fraction'] = float(self.tested.mean())
        stats['cumulative_contradicted_fraction'] = float(self.contradicted.mean())
        stats['cumulative_depth_fraction'] = float(self.depth_contradicted.mean())
        stats['cumulative_rgb_fraction'] = float(self.rgb_contradicted.mean())
        if (self.hits >= cfg['confirm_observations']
                and stats['cumulative_tested_fraction'] >= cfg['min_cumulative_tested_fraction']
                and stats['cumulative_contradicted_fraction'] >= cfg['min_contradicted_fraction']):
            self.state, self.confirmed_frame = 'would_remove', frame
            return 'would_remove', stats
        return 'contradiction', stats
