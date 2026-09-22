import math
from dataclasses import dataclass, field

import torch


@dataclass
class EvidenceHit:
    frame_id: int
    conflict_voxels: int
    visible_voxels: int
    conflict_ratio: float
    pose_c2w: torch.Tensor = field(repr=False)

    def to_json(self):
        return {
            "frame_id": self.frame_id,
            "conflict_voxels": self.conflict_voxels,
            "visible_voxels": self.visible_voxels,
            "conflict_ratio": self.conflict_ratio,
        }


@dataclass
class InstanceEvidence:
    track_uid: int
    state: str = "ACTIVE"
    hits: list = field(default_factory=list)


class TemporalEvidenceBank:
    """Accumulate independent disappearance observations per stable instance."""

    def __init__(self, cfg):
        self.confirm_hits = int(cfg.get("confirm_hits", 3))
        self.min_frame_gap = int(cfg.get("min_hit_frame_gap", 10))
        self.min_translation = float(cfg.get("min_hit_translation", 0.05))
        self.min_rotation_deg = float(cfg.get("min_hit_rotation_deg", 5.0))
        self.states = {}

    @staticmethod
    def _rotation_difference_deg(pose_a, pose_b):
        relative = pose_a[:3, :3].T @ pose_b[:3, :3]
        cosine = ((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0)
        return math.degrees(float(torch.acos(cosine).item()))

    def _is_independent(self, last_hit, frame_id, pose_c2w):
        if frame_id - last_hit.frame_id < self.min_frame_gap:
            return False
        translation = torch.linalg.norm(
            pose_c2w[:3, 3] - last_hit.pose_c2w[:3, 3]
        ).item()
        rotation_deg = self._rotation_difference_deg(last_hit.pose_c2w, pose_c2w)
        return translation >= self.min_translation or rotation_deg >= self.min_rotation_deg

    def add(self, track_uid, frame_id, pose_c2w, conflict_voxels,
            visible_voxels, conflict_ratio):
        state = self.states.setdefault(track_uid, InstanceEvidence(track_uid))
        pose_cpu = pose_c2w.detach().cpu().clone()
        if state.hits and not self._is_independent(state.hits[-1], frame_id, pose_cpu):
            return None

        old_state = state.state
        state.hits.append(EvidenceHit(
            frame_id=int(frame_id),
            conflict_voxels=int(conflict_voxels),
            visible_voxels=int(visible_voxels),
            conflict_ratio=float(conflict_ratio),
            pose_c2w=pose_cpu,
        ))
        state.state = "STALE" if len(state.hits) >= self.confirm_hits else "SUSPECT"
        return {
            "track_uid": int(track_uid),
            "old_state": old_state,
            "new_state": state.state,
            "independent_hit_count": len(state.hits),
            "hits": [hit.to_json() for hit in state.hits],
        }

    def merge_instance_aliases(self, instances):
        """Merge evidence when OnlineAnySeg merges several instance tracks."""
        for instance in instances:
            root_uid = int(instance.track_uid)
            parent_uids = {int(uid) for uid in instance.parent_track_uids}
            parent_uids.add(root_uid)
            parent_states = [self.states[uid] for uid in parent_uids if uid in self.states]
            if not parent_states:
                continue

            all_hits = []
            for parent_state in parent_states:
                all_hits.extend(parent_state.hits)
            hits_by_frame = {}
            for hit in all_hits:
                previous = hits_by_frame.get(hit.frame_id)
                if previous is None or hit.conflict_ratio > previous.conflict_ratio:
                    hits_by_frame[hit.frame_id] = hit
            merged_hits = sorted(hits_by_frame.values(), key=lambda hit: hit.frame_id)

            state = self.states.setdefault(root_uid, InstanceEvidence(root_uid))
            state.hits = merged_hits
            state.state = "STALE" if len(merged_hits) >= self.confirm_hits else "SUSPECT"
            for uid in parent_uids:
                if uid != root_uid:
                    self.states.pop(uid, None)
