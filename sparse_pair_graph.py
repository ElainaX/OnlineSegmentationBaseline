"""Sparse temporal evidence graph for online mask association.

The old implementation kept one dense containment value for every pair of
active masks.  Most pairs are spatially impossible, so this module stores only
observed candidates and their temporal evidence.  A pair owns a 65-slot
window: the first 32 slots are history, slot 32 is the current anchor, and the
last 32 slots are reserved for future observations.  Empty future slots do not
count as negative evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class PairEvidence:
    left: int
    right: int
    start_frame: int
    evidence: np.ndarray = field(default_factory=lambda: np.zeros(65, dtype=np.float32))
    valid: np.ndarray = field(default_factory=lambda: np.zeros(65, dtype=np.bool_))
    positive: np.ndarray = field(default_factory=lambda: np.zeros(65, dtype=np.float32))
    negative: np.ndarray = field(default_factory=lambda: np.zeros(65, dtype=np.float32))
    status: str = "free"  # free, merging, merged
    last_frame: int = -1
    last_left_in_right: float = 0.0
    last_right_in_left: float = 0.0

    def score(self, tau: float = 12.0) -> float:
        idx = np.arange(65, dtype=np.float32)
        weights = np.exp(-np.abs(idx - 32.0) / max(float(tau), 1e-3))
        weights *= self.valid.astype(np.float32)
        denom = float(weights.sum())
        return 0.0 if denom <= 1e-6 else float((weights * self.evidence).sum() / denom)

    @property
    def valid_count(self) -> int:
        return int(self.valid.sum())

    def temporal_smoothness(self) -> float:
        """Continuity of the observed contact/overlap trajectory.

        This is deliberately a weak prior.  It rewards a pair whose visible
        overlap changes smoothly, but it cannot by itself prove that adjacent
        coplanar objects are one instance.
        """
        values = self.evidence[self.valid]
        if values.size < 2:
            return 1.0
        return float(np.exp(-np.abs(np.diff(values)).mean()))


class SparsePairGraph:
    """Candidate pair graph with bounded temporal evidence and union status."""

    def __init__(self, cfg: Optional[dict] = None):
        opts = (cfg or {}).get("seg", {}).get("pair_graph", {})
        self.window = 65
        self.half_window = 32
        self.tau = float(opts.get("weight_tau", 12.0))
        self.max_candidates_per_mask = int(opts.get("max_candidates_per_mask", 16))
        self.positive_threshold = float(opts.get("positive_threshold", 0.20))
        self.negative_threshold = float(opts.get("negative_threshold", 0.20))
        self.commit_threshold = float(opts.get("commit_threshold", 0.55))
        self.min_valid_for_commit = int(opts.get("min_valid_for_commit", 3))
        self.commit_delay = int(opts.get("commit_delay", 8))
        self.reset_after = int(opts.get("reset_after", 64))
        self.pairs: Dict[Tuple[int, int], PairEvidence] = {}
        self.total_observations = 0
        self.new_pair_count = 0
        self.status_transitions = {"free_to_merging": 0, "merging_to_merged": 0}

    @staticmethod
    def key(left: int, right: int) -> Tuple[int, int]:
        left, right = int(left), int(right)
        return (left, right) if left < right else (right, left)

    def _reset_window(self, pair: PairEvidence, frame_id: int) -> None:
        # Keep the latest 32 observations and make them the historical half of
        # a fresh window.  Future slots are empty and therefore neutral.
        keep = pair.evidence[33:65].copy()
        keep_valid = pair.valid[33:65].copy()
        keep_pos = pair.positive[33:65].copy()
        keep_neg = pair.negative[33:65].copy()
        pair.evidence.fill(0.0)
        pair.valid.fill(False)
        pair.positive.fill(0.0)
        pair.negative.fill(0.0)
        n = min(32, keep.shape[0])
        pair.evidence[32 - n:32] = keep[-n:]
        pair.valid[32 - n:32] = keep_valid[-n:]
        pair.positive[32 - n:32] = keep_pos[-n:]
        pair.negative[32 - n:32] = keep_neg[-n:]
        pair.start_frame = int(frame_id) - 32

    def _slot(self, pair: PairEvidence, frame_id: int) -> int:
        if frame_id > pair.start_frame + self.reset_after:
            self._reset_window(pair, frame_id)
        return int(np.clip(int(frame_id) - pair.start_frame, 0, 64))

    def observe(
        self,
        frame_id: int,
        left: int,
        right: int,
        left_in_right: float,
        right_in_left: float,
        *,
        visible: bool = True,
        positive: Optional[float] = None,
        negative: Optional[float] = None,
    ) -> None:
        if int(left) == int(right):
            return
        key = self.key(left, right)
        if key not in self.pairs:
            self.pairs[key] = PairEvidence(key[0], key[1], int(frame_id) - 32)
            self.new_pair_count += 1
        pair = self.pairs[key]
        self.total_observations += 1
        slot = self._slot(pair, int(frame_id))
        overlap = float(np.clip((float(left_in_right) + float(right_in_left)) * 0.5, 0.0, 1.0))
        pos = overlap if positive is None else float(np.clip(positive, 0.0, 1.0))
        neg = 0.0 if negative is None else float(np.clip(negative, 0.0, 1.0))
        if not visible:
            return
        pair.evidence[slot] = float(np.clip(pos - neg, -1.0, 1.0))
        pair.positive[slot] = pos
        pair.negative[slot] = neg
        pair.valid[slot] = True
        pair.last_frame = int(frame_id)
        if int(left) == pair.left:
            pair.last_left_in_right = float(left_in_right)
            pair.last_right_in_left = float(right_in_left)
        else:
            pair.last_left_in_right = float(right_in_left)
            pair.last_right_in_left = float(left_in_right)
        if pair.status == "free" and pos >= self.positive_threshold:
            pair.status = "merging"
            self.status_transitions["free_to_merging"] += 1

    def observe_candidates(
        self,
        frame_id: int,
        query_id: int,
        overlap_ids: Iterable[int],
        query_to_other: Iterable[float],
        other_to_query: Iterable[float],
    ) -> None:
        rows = []
        for other, q2o, o2q in zip(overlap_ids, query_to_other, other_to_query):
            score = (float(q2o) + float(o2q)) * 0.5
            rows.append((score, int(other), float(q2o), float(o2q)))
        rows.sort(reverse=True)
        for _, other, q2o, o2q in rows[: self.max_candidates_per_mask]:
            self.observe(frame_id, query_id, other, q2o, o2q)

    def observe_separation(self, frame_id: int, left: int, right: int, confidence: float) -> None:
        """Add negative evidence after an independent depth visibility audit.

        Callers must not use this for an unobserved or occluded pair: missing
        evidence is neutral, while this method explicitly asserts that the two
        surfaces were simultaneously visible and separated.
        """
        self.observe(frame_id, left, right, 0.0, 0.0,
                     visible=float(confidence) > 0.0,
                     positive=0.0, negative=float(confidence))

    def advance(self, frame_id: int) -> None:
        for pair in self.pairs.values():
            self._slot(pair, int(frame_id))

    def iter_candidates(self, current_frame: Optional[int] = None) -> List[PairEvidence]:
        if current_frame is not None:
            self.advance(int(current_frame))
        return [p for p in self.pairs.values() if p.status in ("free", "merging", "merged") and p.valid_count > 0]

    def score(self, pair: PairEvidence) -> float:
        return pair.score(self.tau)

    def to_dense(self, count: int, device=None):
        """Build a compatibility matrix only when a legacy merge needs it."""
        import torch
        out = torch.eye(int(count), dtype=torch.float32, device=device)
        for pair in self.pairs.values():
            if pair.left >= count or pair.right >= count:
                continue
            out[pair.left, pair.right] = float(pair.last_left_in_right)
            out[pair.right, pair.left] = float(pair.last_right_in_left)
        return out

    def mark_merged(self, pairs: Iterable[PairEvidence]) -> None:
        for pair in pairs:
            if pair.status != "merged":
                self.status_transitions["merging_to_merged"] += 1
            pair.status = "merged"

    def remap(self, old_to_new: Dict[int, int], new_count: int) -> None:
        new_pairs: Dict[Tuple[int, int], PairEvidence] = {}
        for pair in self.pairs.values():
            if pair.left not in old_to_new or pair.right not in old_to_new:
                continue
            key = self.key(old_to_new[pair.left], old_to_new[pair.right])
            if key[0] == key[1]:
                continue
            if key not in new_pairs:
                copied = PairEvidence(key[0], key[1], pair.start_frame)
                copied.evidence[:] = pair.evidence
                copied.valid[:] = pair.valid
                copied.positive[:] = pair.positive
                copied.negative[:] = pair.negative
                copied.status = pair.status
                copied.last_frame = pair.last_frame
                copied.last_left_in_right = pair.last_left_in_right
                copied.last_right_in_left = pair.last_right_in_left
                new_pairs[key] = copied
            else:
                dst = new_pairs[key]
                dst.evidence = np.maximum(dst.evidence, pair.evidence)
                dst.positive = np.maximum(dst.positive, pair.positive)
                dst.negative = np.maximum(dst.negative, pair.negative)
                dst.valid |= pair.valid
                dst.status = "merged" if "merged" in (dst.status, pair.status) else "merging"
                dst.last_frame = max(dst.last_frame, pair.last_frame)
        self.pairs = new_pairs

    def prune(self, valid_ids: Iterable[int]) -> None:
        valid = set(int(i) for i in valid_ids)
        self.pairs = {k: p for k, p in self.pairs.items() if p.left in valid and p.right in valid}

