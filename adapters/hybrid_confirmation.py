"""External H confirmation policy; original point classification/eraser untouched."""
import functools
import hashlib
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent

def observe(original, candidate, frame, pose, codes, sources, cfg, sink=None):
    from feng_remove_detection import CONTRADICTION, SOURCE_DEPTH, SOURCE_RGB, independent_observation
    assert cfg['confirm_observations'] == 2
    codes, sources = np.asarray(codes), np.asarray(sources)
    n = len(candidate.points)
    assert len(codes) == len(sources) == n and n > 0
    bad = codes == CONTRADICTION
    depth = int(np.count_nonzero(bad & ((sources & SOURCE_DEPTH) != 0)))
    rgb = int(np.count_nonzero(bad & ((sources & SOURCE_RGB) != 0)))
    # The original point classifier assigns exactly one source per contradicted point.
    assert not np.any(bad & ((sources & SOURCE_DEPTH) != 0) & ((sources & SOURCE_RGB) != 0))
    before = int(candidate.hits)
    first = before == 0
    dual = depth / n >= cfg['min_contradicted_fraction'] and rgb / n >= cfg['min_contradicted_fraction']
    independent = independent_observation(pose, candidate.last_observation_pose, frame,
                                          candidate.last_observation_frame, cfg)
    effective = 1 if first and dual else 2
    settings = dict(cfg, confirm_observations=effective)
    reason, stats = original(candidate, frame, pose, codes, sources, settings)
    stats.update(hybrid_first=first, hybrid_dual_current=dual,
                 hybrid_effective_confirmations=effective,
                 hybrid_depth_reference_fraction=depth/n,
                 hybrid_rgb_reference_fraction=rgb/n)
    if sink is not None:
        sink(dict(frame=int(frame), candidate_id=int(candidate.candidate_id),
                  reason=reason, hits_before=before, hits_after=int(candidate.hits),
                  independent=bool(independent), reference_points=n,
                  depth_points=depth, rgb_points=rgb, stats=stats))
    return reason, stats

def persist_policy(out):
    out = Path(out)
    if out.exists():
        data = (R/'hybrid_policy.json').read_bytes()
        (out/'hybrid_policy.json').write_bytes(data)
        (out/'hybrid_adapter_sha256.txt').write_text(
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()+'\n')

def install(out):
    from feng_remove_detection import Candidate
    original = Candidate.observe
    assert not getattr(original, '_hybrid_installed', False), 'Duplicate H policy install'
    out = Path(out)
    def sink(row):
        with (out/'hybrid_observations.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
    @functools.wraps(original)
    def wrapped(self, frame, pose, codes, sources, cfg):
        return observe(original, self, frame, pose, codes, sources, cfg, sink)
    wrapped._hybrid_installed = True
    Candidate.observe = wrapped
