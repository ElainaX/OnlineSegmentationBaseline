"""Reversible decisions over independent temporal TSDF fragments.

Public entry point: ChangeLifecycle.step(), always audit BEFORE integration.
Assignments are provisional fragment IDs -> 2D masks from an upstream tracker;
they are NOT automatically equated with historical instance IDs. No GT input.
An accepted fragment is sealed; later observations need a fresh fragment key.
"""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time
import numpy as np
from .contradiction import (AuditOptions, UNKNOWN, SUPPORT, FREE, OCCLUDED,
    validate_camera, sample_surface, raycast_volume, audit_surface)
from .change_frontend import (FrontendOptions, render_quality, compose_history,
    prepare_context, fragment_evidence, ownership_slots)


@dataclass(frozen=True)
class LifecycleOptions:
    min_reference_observations: int = 3
    min_views: int = 3
    view_distance_m: float = .1
    view_angle_deg: float = 10.
    window_views: int = 16
    history_age: int = 64  # audited steps, not raw frame indices
    min_cells: int = 20
    suspect_fraction: float = .05
    retire_fraction: float = .60  # denominator: ALL fixed reference cells
    accept_fraction: float = .50
    restore_fraction: float = .60
    cell_consensus: float = .80
    max_pending_steps: int = 64
    max_fragments: int = 128  # retained inactive volumes also consume memory
    max_attribute_bytes: int = 2 * 1024**3

    def __post_init__(self):
        if min(self.min_reference_observations, self.min_views, self.window_views,
               self.history_age, self.min_cells, self.max_pending_steps, self.max_fragments) < 1:
            raise ValueError('positive lifecycle limits required')
        if self.min_views > self.window_views or self.max_attribute_bytes < 1:
            raise ValueError('invalid evidence/memory budget')
        for x in (self.suspect_fraction, self.retire_fraction, self.accept_fraction,
                  self.restore_fraction, self.cell_consensus):
            if not 0 < x <= 1: raise ValueError('fractions must be in (0,1]')
        if self.cell_consensus <= .5 or self.view_distance_m <= 0 or not 0 < self.view_angle_deg < 180:
            raise ValueError('invalid consensus/view thresholds')


@dataclass
class EvidenceWindow:
    options: LifecycleOptions
    records: list = field(default_factory=list)

    def update(self, tick, pose, state):
        self.records = [r for r in self.records if tick - r[0] < self.options.history_age]
        for _, _, old_state, cell_tick in self.records:
            old_state[tick - cell_tick >= self.options.history_age] = UNKNOWN
        # Unknown / occluded views cannot remove useful earlier evidence just
        # by returning to a camera position. No visibility is not a negative.
        if not np.any((state == SUPPORT) | (state == FREE)): return False
        matches = []
        for i, (_, old_pose, _, _) in enumerate(self.records):
            dist = np.linalg.norm(pose[:3, 3] - old_pose[:3, 3])
            angle = np.degrees(np.arccos(np.clip(pose[:3, 2] @ old_pose[:3, 2], -1, 1)))
            if dist < self.options.view_distance_m and angle < self.options.view_angle_deg:
                matches.append((dist / self.options.view_distance_m + angle / self.options.view_angle_deg, i))
        independent = not matches
        if matches:
            i = min(matches)[1]
            _, anchor, previous, cell_tick = self.records.pop(i)
            # Keep the original view anchor: slowly moving cameras must not
            # chase a rolling representative forever and never gain a new view.
            informative = (state == SUPPORT) | (state == FREE)
            cell_tick = np.where(informative, tick, cell_tick)
            state = np.where(informative, state, previous)
            self.records.append((tick, anchor, state.copy(), cell_tick))
        else:
            self.records.append((tick, pose.copy(), state.copy(), np.full(len(state), tick, np.int64)))
        self.records = self.records[-self.options.window_views:]
        return independent

    def counts(self, n):
        if not self.records: return np.zeros(n, int), np.zeros(n, int)
        s = np.stack([r[2] for r in self.records])
        return (s == SUPPORT).sum(0), (s == FREE).sum(0)


@dataclass
class TrackedFragment:
    key: str
    role: str  # reference or proposal
    state: str  # stable, suspect, retired, pending, accepted, rejected, unresolved
    created: int
    points: np.ndarray
    history: EvidenceWindow
    last_stats: dict = field(default_factory=dict)


class ReferenceNotReady(ValueError):
    """A fragment can stay in the map without being safe to use as change evidence."""


class ChangeLifecycle:
    def __init__(self, store, options=None, audit_options=None, frontend_options=None):
        self.store = store
        self.options = options or LifecycleOptions()
        self.audit_options = audit_options or AuditOptions(near_m=store.options.near_m, far_m=store.options.far_m)
        self.frontend_options = frontend_options or FrontendOptions()
        self.tracks = {}; self.events = []; self.tick = 0; self.last_frame = None
        self.reference_skips = {}

    @property
    def required_observations(self):
        return max(self.options.min_reference_observations,
                   self.frontend_options.min_instance_observations if self.frontend_options.enabled else 0)

    def _surface_samples(self, fragment):
        points = (fragment.volume.surface(weight_threshold=self.frontend_options.min_map_weight)[0]
                  if self.frontend_options.enabled else fragment.volume.surface()[0])
        return sample_surface(points, self.audit_options)

    def register_reference(self, key):
        key = str(key)
        if key in self.tracks: raise ValueError('already registered')
        f = self.store.fragments[key]
        if f.status not in ('active', 'frozen') or len(f.observations) < self.required_observations:
            raise ReferenceNotReady('reference must be live and have sufficient prior observations')
        points = self._surface_samples(f)
        minimum = max(self.options.min_cells, self.frontend_options.min_reference_points if self.frontend_options.enabled else 0)
        if len(points) < minimum: raise ReferenceNotReady('insufficient reliable reference surface')
        # Close the historical temporal segment before any new evidence arrives.
        self.store.set_status(key, 'frozen', 'seal historical reference before change audit')
        self.tracks[key] = TrackedFragment(key, 'reference', 'stable', self.tick, points, EvidenceWindow(self.options))

    def register_available_references(self):
        """Low-quality/small references are retained, logged, and excluded from auditing."""
        for key, f in self.store.fragments.items():
            if key in self.tracks: continue
            try:
                self.register_reference(key)
            except ReferenceNotReady as error:
                self.reference_skips[key] = str(error)
                if f.status == 'active': self.store.set_status(key, 'frozen', 'historical segment retained but not reliable enough to audit')
        return dict(self.reference_skips)

    def _transition(self, tr, state, reason, frame):
        if tr.state == state: return
        self.events.append(dict(frame=int(frame), tick=self.tick, fragment=tr.key,
            before=tr.state, after=state, reason=reason, evidence=dict(tr.last_stats)))
        tr.state = state
        status = ('inactive' if state in ('retired', 'rejected', 'unresolved') else
                  'pending' if state == 'pending' else 'frozen')
        self.store.set_status(tr.key, status, reason)

    def _decide(self, tr, state, pose, frame, appearance_suspect=False):
        independent = tr.history.update(self.tick, pose, state)
        support, free = tr.history.counts(len(tr.points)); count = support + free
        confirmed_free = (free >= self.options.min_views) & (free / np.maximum(count, 1) >= self.options.cell_consensus)
        confirmed_support = (support >= self.options.min_views) & (support / np.maximum(count, 1) >= self.options.cell_consensus)
        nf, ns = int(confirmed_free.sum()), int(confirmed_support.sum())
        denom = max(1, len(tr.points)); ff, sf = nf / denom, ns / denom
        tr.last_stats = dict(cells=len(tr.points), observed_cells=int((count > 0).sum()),
            confirmed_free_cells=nf, confirmed_support_cells=ns, free_fraction=ff,
            support_fraction=sf, independent_view=independent, retained_views=len(tr.history.records),
            unknown=int((state == UNKNOWN).sum()), support=int((state == SUPPORT).sum()),
            free=int((state == FREE).sum()), occluded=int((state == OCCLUDED).sum()))
        enough_free = nf >= self.options.min_cells and ff >= self.options.retire_fraction
        enough_support = ns >= self.options.min_cells
        if tr.state == 'pending':
            if enough_free:
                self._transition(tr, 'rejected', 'proposal geometry contradicted by later independent views', frame)
            elif enough_support and sf >= self.options.accept_fraction and not appearance_suspect:
                self._transition(tr, 'accepted', 'proposal geometry supported before current-frame fusion', frame)
        elif tr.state == 'retired':
            if enough_support and sf >= self.options.restore_fraction and not appearance_suspect:
                self._transition(tr, 'stable', 'later observations support retained historical surface', frame)
        elif tr.state not in ('rejected', 'unresolved'):
            if enough_free:
                self._transition(tr, 'retired', 'multi-view free-space evidence covers reference surface', frame)
            elif enough_support and sf >= self.options.restore_fraction and not appearance_suspect:
                if tr.state == 'suspect': self._transition(tr, 'stable', 'contradiction refuted by later support', frame)
            elif int((state == FREE).sum()) >= self.options.min_cells and (state == FREE).mean() >= self.options.suspect_fraction:
                self._transition(tr, 'suspect', 'partial contradiction; keep geometry while awaiting evidence', frame)
        if appearance_suspect and tr.state in ('stable', 'accepted'):
            self._transition(tr, 'suspect', 'appearance differs at consistent depth; no geometry deletion authorized', frame)

    def step(self, frame, rgb, depth_m, K, pose, assignments=None):
        """Audit all sealed references and provisional volumes, then integrate.

        All assignments are new temporal fragments. Historical keys are rejected
        so a caller cannot accidentally overwrite the very reference being tested.
        Returns diagnostic counts/transitions/timings, not calibrated probabilities.
        """
        started = time.perf_counter()
        validate_camera(K, pose)
        depth_m = np.asarray(depth_m); assignments = assignments or {}
        if depth_m.ndim != 2 or rgb.shape != depth_m.shape + (3,) or rgb.dtype != np.uint8:
            raise ValueError('requires aligned RGB uint8 and HxW metric depth')
        if self.last_frame is not None and int(frame) <= self.last_frame:
            raise ValueError('frames must be strictly increasing')
        masks = {str(k): np.asarray(v, bool) for k, v in assignments.items()}
        union = np.zeros(depth_m.shape, bool)
        new_keys = []
        for key, mask in masks.items():
            if mask.shape != depth_m.shape or np.any(union & mask): raise ValueError('assignment masks must be aligned and disjoint')
            union |= mask
            if key in self.store.fragments:
                if key not in self.tracks or self.tracks[key].state != 'pending':
                    raise ValueError('assignment requires a fresh temporal key or pending proposal: ' + key)
            else: new_keys.append(key)
        if len(self.store.fragments) + len(new_keys) > self.options.max_fragments:
            raise MemoryError('retained fragment budget exceeded; no silent history deletion')
        total_bytes = sum(f.volume.bytes_allocated_attributes() for f in self.store.fragments.values())
        if total_bytes > self.options.max_attribute_bytes:
            raise MemoryError('TSDF attribute budget exceeded; checkpoint or explicitly release history')
        front = self.frontend_options
        audit_due = not front.enabled or int(frame) % front.interval == 0
        self.tick += int(audit_due); self.last_frame = int(frame); event_start = len(self.events)
        rendered = {}
        if audit_due:
            for key, f in self.store.fragments.items():
                if key in self.tracks and self.tracks[key].state in ('rejected', 'unresolved'): continue
                rendered[key] = (render_quality(f.volume, K, pose, depth_m.shape, front) if front.enabled
                                 else raycast_volume(f.volume, K, pose, depth_m.shape))
        # Compose only published geometry. Retired and pending volumes must not
        # mask each other's audit visibility. Measured nearer depth still occludes.
        diagnostics = dict(audit_due=audit_due, enabled=front.enabled, skip_reasons=[])
        context = None
        if front.enabled and audit_due:
            slots = ownership_slots(self.store, rendered)
            published = [k for k in rendered if self.store.fragments[k].status in ('active', 'frozen')]
            history = compose_history(rendered, published, slots, depth_m.shape, front)
            # If all history is retired, keep auditing it for recovery. A private
            # render supplies calibration/quality, never republishes geometry.
            if not published:
                history = compose_history(rendered, list(rendered), slots, depth_m.shape, front)
            context = prepare_context(rgb, depth_m, history, front, self.audit_options.near_m, self.audit_options.far_m)
            diagnostics.update(skip_reasons=context['skip_reasons'], global_conflict_ratio=context['global_ratio'],
                rgb={k: v for k, v in context['rgb'].items() if k not in ('current_lab', 'bad')},
                ambiguous_pixels=int((history['owners'] == -2).sum()))
        else:
            live = np.full(depth_m.shape, np.inf, np.float32)
            for key, pred in rendered.items():
                if self.store.fragments[key].status in ('active', 'frozen'):
                    live = np.minimum(live, np.where(pred > 0, pred, np.inf))
            live[~np.isfinite(live)] = 0
        for key, tr in self.tracks.items():
            if key not in rendered: continue
            f = self.store.fragments[key]
            if tr.state == 'pending' and not len(tr.points) and len(f.observations) >= self.required_observations:
                tr.points = self._surface_samples(f)
            if len(tr.points):
                if front.enabled:
                    if context['skip_reasons']:
                        tr.last_stats['frontend'] = dict(skipped=True, reasons=context['skip_reasons'])
                    else:
                        state, appearance, detail = fragment_evidence(tr.points, rgb, depth_m, K, pose,
                            rendered[key], history, context, slots[key], len(f.observations), front,
                            self.audit_options.near_m, self.audit_options.far_m, self.audit_options.map_gate_m,
                            private=f.status in ('pending', 'inactive'))
                        candidate_seen = tr.last_stats.get('candidate_started', False) or detail['candidate']
                        # Candidate screening must precede destructive evidence.
                        # Support can still disprove a candidate or restore history.
                        if not candidate_seen: state = np.where(state == FREE, UNKNOWN, state).astype(np.uint8)
                        appearance_suspect = (int(appearance.sum()) >= self.options.min_cells and
                            int(appearance.sum()) / max(1, detail['tested_reference']) >= front.contradiction_ratio)
                        if detail['tested_reference'] >= front.min_reference_tested_points:
                            self._decide(tr, state, pose, frame, appearance_suspect=appearance_suspect and candidate_seen)
                        else:
                            detail['paused'] = 'insufficient_tested_reference'
                        tr.last_stats.update(frontend=detail, candidate_started=bool(candidate_seen),
                            appearance_warning=bool(appearance_suspect), rgb_cannot_authorize_retirement=True)
                        for event in self.events[event_start:]:
                            if event['fragment'] == key: event['evidence'] = dict(tr.last_stats)
                else:
                    state = audit_surface(tr.points, depth_m, rendered[key], live, K, pose, self.audit_options)
                    self._decide(tr, state, pose, frame)
            if tr.state == 'pending' and self.tick - tr.created >= self.options.max_pending_steps:
                self._transition(tr, 'unresolved', 'proposal timed out without enough evidence; not evidence of disappearance', frame)
        audit_seconds = time.perf_counter() - started
        integrated = []
        for key, mask in masks.items():
            if key not in self.tracks:
                # Empty/invalid masks do not allocate a volume.
                if not np.any(mask & np.isfinite(depth_m) & (depth_m > self.store.options.near_m) & (depth_m < self.store.options.far_m)): continue
                self.store.create(key)
                self.store.set_status(key, 'pending', 'quarantine new temporal observations')
                self.tracks[key] = TrackedFragment(key, 'proposal', 'pending', self.tick,
                    np.empty((0, 3)), EvidenceWindow(self.options))
            tr = self.tracks[key]
            # If just accepted/rejected this step, do NOT write the deciding
            # observation into it. Next segment must use a new provisional key.
            # Once samples are fixed, seal proposal geometry too. Otherwise a
            # small verified core could authorize untested geometry fused later.
            if tr.state == 'pending' and not len(tr.points):
                if self.store.integrate(key, frame, rgb, depth_m, K, pose, mask,
                        {'kind': 'online_quarantine', 'tick': self.tick}): integrated.append(key)
        total_bytes = sum(f.volume.bytes_allocated_attributes() for f in self.store.fragments.values())
        if total_bytes > self.options.max_attribute_bytes:
            # Open3D allocation isn't transactional. Explicitly fail rather than
            # silently evicting evidence; caller must stop, not continue this step.
            raise MemoryError('post-integration attribute budget exceeded')
        return dict(frame=int(frame), tick=self.tick, fragments={k: dict(state=t.state, **t.last_stats) for k, t in self.tracks.items()},
            events=self.events[event_start:], integrated=integrated, attribute_bytes=total_bytes, frontend=diagnostics,
            sealed_proposals=[k for k, t in self.tracks.items() if t.state == 'pending' and len(t.points)],
            audit_seconds=audit_seconds, total_seconds=time.perf_counter() - started)

    def save(self, directory):
        """Persist geometry, masks, exact fixed samples and bounded vote history."""
        directory = Path(directory); self.store.save(directory, save_volumes=True)
        arrays = {}; tracks = []
        for i, tr in enumerate(self.tracks.values()):
            arrays[f'points_{i}'] = tr.points
            arrays[f'states_{i}'] = np.stack([r[2] for r in tr.history.records]) if tr.history.records else np.empty((0, len(tr.points)), np.uint8)
            arrays[f'cell_ticks_{i}'] = np.stack([r[3] for r in tr.history.records]) if tr.history.records else np.empty((0, len(tr.points)), np.int64)
            tracks.append(dict(key=tr.key, role=tr.role, state=tr.state, created=tr.created,
                history=[dict(tick=r[0], pose=r[1].tolist()) for r in tr.history.records], last_stats=tr.last_stats))
        np.savez_compressed(directory / 'change_evidence.npz', **arrays)
        (directory / 'change_lifecycle.json').write_text(json.dumps(dict(format=2,
            options=asdict(self.options), audit_options=asdict(self.audit_options),
            frontend_options=asdict(self.frontend_options), tick=self.tick,
            last_frame=self.last_frame, tracks=tracks, events=self.events, reference_skips=self.reference_skips), indent=2))

    @classmethod
    def load(cls, directory):
        from .independent_tsdf import IndependentTSDFStore
        directory = Path(directory); data = json.loads((directory / 'change_lifecycle.json').read_text())
        if data['format'] not in (1, 2): raise ValueError('unsupported lifecycle checkpoint')
        obj = cls(IndependentTSDFStore.load(directory), LifecycleOptions(**data['options']), AuditOptions(**data['audit_options']),
                  FrontendOptions(**data.get('frontend_options', {})))
        obj.tick, obj.last_frame, obj.events = data['tick'], data['last_frame'], data['events']
        obj.reference_skips = data.get('reference_skips', {})
        with np.load(directory / 'change_evidence.npz') as arrays:
            for i, item in enumerate(data['tracks']):
                window = EvidenceWindow(obj.options)
                window.records = [(r['tick'], np.array(r['pose']), arrays[f'states_{i}'][j].copy(), arrays[f'cell_ticks_{i}'][j].copy()) for j, r in enumerate(item['history'])]
                obj.tracks[item['key']] = TrackedFragment(item['key'], item['role'], item['state'], item['created'], arrays[f'points_{i}'].copy(), window, item['last_stats'])
        return obj
