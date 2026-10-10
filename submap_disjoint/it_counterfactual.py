"""Counterfactual submap retirement for ITSubmap.

The controller keeps the independent-TSDF representation intact. It selects a
visible fragment with strong disappearance evidence, then evaluates the same
future RGB-D frames with that fragment included and excluded. A fragment is
retired only after the whole observation window remained visible; otherwise it
is kept. This is deliberately an experiment module, not a GT-assisted oracle.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json, time
import numpy as np

@dataclass(frozen=True)
class CounterfactualOptions:
    min_surface_points: int = 30
    min_tested_points: int = 30
    trigger_ratio: float = 0.50
    horizon_frames: int = 4
    min_visible_fraction: float = 0.25
    decision_margin: float = 0.02
    min_age_frames: int = 2
    max_surface_points: int = 12000
    depth_abs_m: float = 0.08
    depth_relative: float = 0.01
    max_active_candidates: int = 1

class ITCounterfactualController:
    def __init__(self, adapter, K, near_m, far_m, output, options=None):
        self.adapter = adapter
        self.K = np.asarray(K, dtype=np.float64)
        self.Kinv = np.linalg.inv(self.K)
        self.near = float(near_m); self.far = float(far_m)
        self.options = options or CounterfactualOptions()
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        self.pending = None
        self.events = []
        self.decisions = []
        self.frames = 0
        self.seconds = 0.0

    def _record(self, key):
        return next((r for r in self.adapter.fragments if r['key'] == key), None)

    def _points(self, record):
        key = record['key']
        if key not in self.cache:
            path = self.adapter.directory / key / 'surface.npz'
            with np.load(path) as f: p = np.asarray(f['points'], dtype=np.float32)
            if len(p) > self.options.max_surface_points:
                ids = np.linspace(0, len(p)-1, self.options.max_surface_points).astype(np.int64)
                p = p[ids]
            self.cache[key] = p
        return self.cache[key]

    def _project(self, points, pose, shape, depth=None):
        h,w = shape
        camera = (points - pose[:3,3]) @ pose[:3,:3]
        z = camera[:,2]
        valid = np.isfinite(camera).all(1) & (z > self.near) & (z < self.far)
        uv = np.zeros((len(points),2), dtype=np.int64)
        if valid.any():
            q = camera[valid] @ self.K.T
            uv[valid] = np.rint(q[:,:2] / q[:,2,None]).astype(np.int64)
            valid[valid] &= (uv[valid,0] >= 0) & (uv[valid,0] < w) & (uv[valid,1] >= 0) & (uv[valid,1] < h)
        if depth is not None:
            ids = np.flatnonzero(valid)
            if len(ids):
                d = depth[uv[ids,1], uv[ids,0]]
                valid[ids] &= np.isfinite(d) & (d > self.near) & (d < self.far)
        return camera, uv, valid

    def _surface_stats(self, points, depth, pose):
        cam, uv, valid = self._project(points, pose, depth.shape, depth)
        ids = np.flatnonzero(valid)
        if not len(ids): return dict(tested=0, conflicts=0, ratio=0.0, visible_fraction=0.0)
        z = cam[ids,2]; d = depth[uv[ids,1], uv[ids,0]]
        tol = self.options.depth_abs_m + self.options.depth_relative*np.maximum(z,d)
        conflict = d > z + tol
        projected = self._project(points, pose, depth.shape)[2]
        return dict(tested=int(len(ids)), conflicts=int(conflict.sum()), ratio=float(conflict.mean()),
                    visible_fraction=float(len(ids)/max(int(projected.sum()),1)))

    def _render(self, records, depth, pose, skip_key=None):
        """Sparse z-buffer from independent submap surfaces.

        This is a deterministic surface raycast used for the counterfactual
        comparison. The actual IT volumes remain unchanged until a decision.
        """
        h,w = depth.shape
        best = np.full((h,w), np.inf, np.float32)
        owner = np.full((h,w), '', dtype=object)
        for record in records:
            key = record['key']
            if key == skip_key or record.get('status','active') == 'inactive': continue
            p = self._points(record)
            cam, uv, valid = self._project(p, pose, depth.shape)
            ids = np.flatnonzero(valid)
            if not len(ids): continue
            pix = uv[ids,1] * w + uv[ids,0]; z = cam[ids,2]
            order = np.lexsort((z,pix))
            pix, z, ids = pix[order], z[order], ids[order]
            first = np.r_[True, pix[1:] != pix[:-1]]
            pix, z = pix[first], z[first]
            yy,xx = pix//w,pix%w
            best[yy,xx] = z
            owner[yy,xx] = key
        valid = np.isfinite(best) & np.isfinite(depth) & (depth > self.near) & (depth < self.far)
        tol = self.options.depth_abs_m + self.options.depth_relative*np.maximum(best, np.where(valid,depth,0))
        contradiction = valid & (depth > best + tol)
        coverage = float(valid.sum()/max(int(np.isfinite(depth).sum()),1))
        ratio = float(contradiction.sum()/max(int(valid.sum()),1))
        return dict(depth=best, owner=owner, valid=valid, contradiction=contradiction,
                    contradiction_rate=ratio, coverage=coverage)

    def _candidate(self, depth, pose, frame):
        best = None
        for record in self.adapter.fragments:
            if record.get('status','active') == 'inactive': continue
            obs = record.get('observation_frames', [])
            if not obs or frame - max(obs) < self.options.min_age_frames: continue
            p = self._points(record)
            if len(p) < self.options.min_surface_points: continue
            stat = self._surface_stats(p, depth, pose)
            if stat['tested'] < self.options.min_tested_points: continue
            if stat['visible_fraction'] < self.options.min_visible_fraction: continue
            if stat['ratio'] < self.options.trigger_ratio: continue
            score = stat['ratio'] * np.log1p(stat['tested'])
            if best is None or score > best['score']:
                best = dict(key=record['key'], score=float(score), **stat)
        return best

    def _observe_pending(self, depth, pose, frame):
        p = self.pending
        key = p['key']; record = self._record(key)
        if record is None or record.get('status','active') == 'inactive':
            self.pending = None; return
        stat = self._surface_stats(self._points(record), depth, pose)
        if stat['tested'] < self.options.min_tested_points or stat['visible_fraction'] < self.options.min_visible_fraction:
            p['missing_visibility'] = True
            p['samples'].append(dict(frame=int(frame), visible=False, tested=stat['tested']))
            self.events.append(dict(frame=int(frame), event='counterfactual_visibility_lost', key=key, **stat))
            return
        records = [r for r in self.adapter.fragments if r.get('status','active') != 'inactive']
        keep = self._render(records, depth, pose)
        drop = self._render(records, depth, pose, skip_key=key)
        sample = dict(frame=int(frame), visible=True, tested=stat['tested'], candidate_ratio=stat['ratio'],
                      keep_contradiction=keep['contradiction_rate'], drop_contradiction=drop['contradiction_rate'],
                      keep_coverage=keep['coverage'], drop_coverage=drop['coverage'])
        p['samples'].append(sample)
        self.events.append(dict(event='counterfactual_sample', key=key, **sample))
        if len(p['samples']) < self.options.horizon_frames: return
        valid = [x for x in p['samples'] if x.get('visible')]
        if len(valid) < self.options.horizon_frames or p.get('missing_visibility'):
            decision = 'keep_visibility_lost'
        else:
            ck = float(np.mean([x['keep_contradiction'] for x in valid])); cd = float(np.mean([x['drop_contradiction'] for x in valid]))
            covk = float(np.mean([x['keep_coverage'] for x in valid])); covd = float(np.mean([x['drop_coverage'] for x in valid]))
            delta = ck-cd
            # Removing is accepted when it improves contradiction or is neutral
            # without collapsing observable map coverage. A clear degradation restores.
            if delta >= -self.options.decision_margin and covd >= 0.5*max(covk,1e-6): decision='retire'
            else: decision='keep_degraded'
            p.update(mean_keep_contradiction=ck, mean_drop_contradiction=cd,
                     mean_keep_coverage=covk, mean_drop_coverage=covd, delta=delta)
        if decision == 'retire':
            record['status'] = 'inactive'
        p['decision'] = decision
        p['end_frame'] = int(frame)
        decision_row = {k:v for k,v in p.items() if k not in ('samples',)}
        decision_row['key'] = key
        decision_row['decision'] = decision
        self.decisions.append(decision_row)
        self.events.append(dict(frame=int(frame), event='counterfactual_decision', key=key, decision=decision,
                                samples=p['samples']))
        self.pending = None

    def observe(self, frame, depth, pose):
        started = time.perf_counter(); self.frames += 1
        depth = np.asarray(depth, dtype=np.float32); pose = np.asarray(pose, dtype=np.float64)
        if self.pending is not None:
            self._observe_pending(depth, pose, int(frame))
        if self.pending is None:
            cand = self._candidate(depth, pose, int(frame))
            if cand is not None:
                self.pending = dict(key=cand['key'], start_frame=int(frame), trigger=cand,
                                    samples=[], missing_visibility=False)
                self.events.append(dict(frame=int(frame), event='counterfactual_candidate', **cand))
        self.seconds += time.perf_counter()-started

    def finalize(self):
        report=dict(method='ITSubmap counterfactual visible-window retirement', options=asdict(self.options),
                    frames=self.frames, seconds=self.seconds, decisions=self.decisions,
                    pending=self.pending, events=self.events,
                    retired=[r['key'] for r in self.adapter.fragments if r.get('status','active') == 'inactive'])
        (self.output/'summary.json').write_text(json.dumps(report, indent=2))
        return report
