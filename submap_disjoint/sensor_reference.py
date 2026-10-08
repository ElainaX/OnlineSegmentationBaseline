"""Causal raw-depth checks of frozen world-space reference points.

The current sensor, not the old map or instance ownership, decides occlusion.
Coordinates: c2w, camera Z depth in metres, integer nearest-pixel sampling.
This is an experimental read-only alternative, not a deletion authority.
"""
import numpy as np
from .readonly_detection_core import (
    SUPPORT, CONTRADICTION, SOURCE_DEPTH, classify_depth,
    independent_observation,
)


def sensor_projection(points, intrinsic, pose, depth, interior, cfg, near, far):
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    ids = np.flatnonzero(np.isfinite(camera).all(1) & (z > near) & (z < far))
    codes = np.zeros(len(points), np.uint8)
    if not len(ids):
        return codes, codes.copy()
    projected = camera[ids] @ intrinsic.T
    uv = np.rint(projected[:, :2] / projected[:, 2:3]).astype(np.int64)
    h, w = depth.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    ids, uv = ids[inside], uv[inside]
    pixels = uv[:, 1] * w + uv[:, 0]
    order = np.lexsort((ids, z[ids], pixels))
    _, first = np.unique(pixels[order], return_index=True)
    chosen = order[first]
    ids, uv = ids[chosen], uv[chosen]
    y, x = uv[:, 1], uv[:, 0]
    codes[ids] = classify_depth(depth[y, x], z[ids], interior[y, x],
                                cfg['depth_epsilon_m'], cfg['depth_relative_epsilon'])
    sources = np.where(codes == CONTRADICTION, SOURCE_DEPTH, 0).astype(np.uint8)
    return codes, sources


class PointViewVotes:
    """Count independent views per point, including small observations.

    A view of another part of the candidate cannot consume this point's clock.
    No vote for unknown/occluded points. uint32 counters avoid byte overflow.
    """
    def __init__(self, n):
        self.support = np.zeros(n, np.uint32)
        self.conflict = np.zeros(n, np.uint32)
        self.last = np.full(n, -1, np.int32)
        self.views = []
        self.last_frame = -1

    def observe(self, frame, pose, codes, cfg):
        if frame <= self.last_frame:
            raise ValueError('Observations must have strictly increasing frame IDs')
        self.last_frame = frame
        reliable = (codes == SUPPORT) | (codes == CONTRADICTION)
        accepted = np.zeros(len(codes), bool)
        for previous in np.unique(self.last[reliable]):
            previous_frame, previous_pose = (-1, None) if previous < 0 else self.views[previous]
            if independent_observation(pose, previous_pose, frame, previous_frame, cfg):
                accepted |= reliable & (self.last == previous)
        if accepted.any():
            self.last[accepted] = len(self.views)
            self.views.append((frame, np.asarray(pose).copy()))
            self.support += accepted & (codes == SUPPORT)
            self.conflict += accepted & (codes == CONTRADICTION)
        return accepted


class SurfaceMemory:
    """Small depth votes stay at their own 3D point; support resets that point.

    A stable point needs the original number of independent confirmations.
    Candidate reporting retains the original 20% spatial coverage requirement.
    This only certifies the selected local surface, never an entire object.
    State can be revoked if subsequent reliable support reappears.
    """
    def __init__(self, n):
        self.votes = PointViewVotes(n)
        self.streak = np.zeros(n, np.uint32)
        self.stable = np.zeros(n, bool)
        self.ever_tested = np.zeros(n, bool)
        self.state = 'suspected'
        self.confirmed_frame = None
        self.first_confirmed_frame = None
        self.transitions = []

    def observe(self, frame, pose, codes, cfg):
        accepted = self.votes.observe(frame, pose, codes, cfg)
        support = accepted & (codes == SUPPORT)
        conflict = accepted & (codes == CONTRADICTION)
        self.ever_tested |= accepted
        self.streak[support] = 0
        self.streak[conflict] += 1
        self.stable = self.streak >= cfg['confirm_observations']
        confirmed = (self.stable.mean() >= cfg['min_contradicted_fraction']
                     and self.ever_tested.mean() >= cfg['min_cumulative_tested_fraction'])
        state = 'would_remove' if confirmed else 'suspected'
        if state != self.state:
            self.transitions.append(dict(frame=int(frame),before=self.state,after=state))
            if confirmed:
                self.confirmed_frame = int(frame)
                if self.first_confirmed_frame is None: self.first_confirmed_frame = int(frame)
            else:
                self.confirmed_frame = None
        self.state = state

    def metadata(self):
        return dict(state=self.state,confirmed_frame=self.confirmed_frame,
            first_confirmed_frame=self.first_confirmed_frame,transitions=self.transitions,
            tested_fraction=float(self.ever_tested.mean()),
            stable_fraction=float(self.stable.mean()), stable_points=int(self.stable.sum()),
            scope='confirmed depth-conflicting surface only; no whole-object deletion')
