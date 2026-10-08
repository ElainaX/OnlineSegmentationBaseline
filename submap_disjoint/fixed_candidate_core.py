"""Original _step plus observation taps; no proposal or baseline rule edits.

Derived from frozen readonly_detection_core.py, with three asserted substitutions.
"""
from .readonly_detection_core import *
from .fixed_candidate_memory import FixedCandidateObserver


class FixedCandidateRemoveDetector(RemoveDetector):
    def __init__(self, scene, cfg, output_dir):
        super().__init__(scene, cfg, output_dir)
        self.fixed = FixedCandidateObserver(output_dir, self.cfg)

    def _step(self, frame, color, depth, pose_c2w, pose_valid, session):
        pose = as_numpy(pose_c2w).astype(np.float64)
        if (not pose_valid or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=.02)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=.02)):
            self._skip(frame, 'invalid_original_pose')
            return
        current = as_numpy(depth).astype(np.float32)
        current_valid = np.isfinite(current) & (current > self.near) & (current < self.far)
        if current_valid.sum() < self.cfg['min_valid_pixels']:
            self._skip(frame, 'insufficient_current_depth')
            return
        try:
            history = self.scene.voxel_grids.raycast_history(
                self.intrinsic, np.linalg.inv(pose), current.shape[1], current.shape[0],
                self.near, self.far, self.cfg['min_map_weight'], self.scene.trunc_voxel_multiplier)
        except RuntimeError as error:
            self._skip(frame, 'raycast_error', error=str(error))
            print(f'[REMOVE-DETECT] frame={frame} raycast failed; diagnostics skipped (see events.jsonl).')
            return
        if history is None:
            self._skip(frame, 'empty_history')
            return
        if history['valid'].sum() < self.cfg['min_valid_pixels']:
            self._skip(frame, 'insufficient_historical_render')
            return
        mapping = np.asarray(self.scene.global2merged_mask_id, dtype=np.int64)
        weights = as_numpy(self.scene.get_merged_mask_weight)
        self._diagnostic_mapping = reliable_instance_mapping(
            mapping, weights, self.cfg['min_instance_weight'])
        query = lambda ids: self.scene.voxel_hash_table.query_active_instance_memberships(
            ids, self._diagnostic_mapping)
        owners = unambiguous_owners(history['index'], history['neighbor_valid'],
                                    history['interp_ratio'], query)
        owners[~history['valid']] = -1
        current_interior = interior_mask(current, current_valid, self.cfg['edge_threshold_m'], self.cfg['boundary_margin_px'])
        historical_interior = interior_mask(history['depth'], history['valid'], self.cfg['edge_threshold_m'], self.cfg['boundary_margin_px'])
        reliable = current_interior & historical_interior
        depth_evidence = classify_depth(current, history['depth'], reliable,
                                        self.cfg['depth_epsilon_m'], self.cfg['depth_relative_epsilon'])
        rgb_eligible = ((depth_evidence == SUPPORT) if self.cfg['rgb_enabled']
                        else np.zeros_like(reliable))
        rgb_context = build_rgb_context(as_numpy(color), history['color'], rgb_eligible, self.cfg)
        evidence = depth_evidence.copy()
        sources = np.zeros(evidence.shape, dtype=np.uint8)
        sources[evidence == CONTRADICTION] = SOURCE_DEPTH
        rgb_bad = rgb_context['contradiction'] & (depth_evidence == SUPPORT)
        evidence[rgb_bad] = CONTRADICTION
        sources[rgb_bad] = SOURCE_RGB
        global_stats = evidence_stats(evidence, sources=sources)
        owner_good = owner_interior(owners, self.cfg['boundary_margin_px'])
        excluded = (evidence == CONTRADICTION) & ~owner_good
        excluded_stats = dict(
            unassigned=int((excluded & (owners == -1)).sum()),
            ambiguous=int((excluded & (owners == -2)).sum()),
            instance_boundary=int((excluded & (owners >= 0)).sum()))
        evidence[~owner_good] = UNKNOWN
        sources[~owner_good] = SOURCE_NONE
        if (global_stats['tested'] >= self.cfg['min_valid_pixels']
                and global_stats['contradiction_ratio'] > self.cfg['max_global_conflict_fraction']):
            self._dump_frame(frame, as_numpy(color), current, history, owners, evidence,
                             sources, rgb_context, [])
            self._skip(frame, 'widespread_conflict', stats=global_stats)
            return
        self.frames_processed += 1
        changed = False
        frame_stats = []
        # Existing candidates are evaluated against their frozen references even
        # if normal merging has changed the current instance's size/slot.
        for candidate in self.candidates:
            codes, candidate_sources = project_reference(
                candidate.points, candidate.colors, self.intrinsic, pose, current,
                history, owners, current_interior, rgb_context, self.cfg, self.near, self.far)
            self.fixed.observe(candidate, frame, pose, codes, candidate_sources)
            if candidate.state != 'suspected':
                continue
            reason, stats = candidate.observe(frame, pose, codes, candidate_sources, self.cfg)
            self.fixed.verify_original(candidate)
            descendants = sorted({int(mapping[r]) for r in candidate.source_raw_ids
                                  if r < len(mapping) and mapping[r] >= 0})
            row = dict(stats, candidate_id=candidate.candidate_id, state=candidate.state,
                       confirmations=candidate.hits, current_slot=descendants[0] if len(descendants) == 1 else None,
                       descendant_slots=descendants, reason=reason)
            frame_stats.append(row)
            self._event(frame, 'candidate_observation', session=session, **row)
            if reason in ('would_remove', 'cancelled'):
                changed = True
                self._save_candidate(candidate, frame, reason, codes, candidate_sources, stats)
                print(f'[REMOVE-DETECT] frame={frame} candidate={candidate.candidate_id} {reason} (diagnostic only)')
        for slot in np.unique(owners[evidence == CONTRADICTION]):
            slot = int(slot)
            pixel_stats = evidence_stats(evidence[owners == slot], sources=sources[owners == slot])
            if slot < 0 or slot >= len(weights) or weights[slot] < self.cfg['min_instance_weight']:
                self._event(frame, 'candidate_rejected', slot=slot, stage='instance',
                            reason='unreliable_instance', pixel_stats=pixel_stats)
                continue
            pixel_failures = []
            if pixel_stats['tested'] < self.cfg['min_valid_pixels']:
                pixel_failures.append('insufficient_pixel_count')
            if pixel_stats['contradiction_ratio'] < self.cfg['contradiction_ratio']:
                pixel_failures.append('low_pixel_contradiction_ratio')
            if pixel_failures:
                self._event(frame, 'candidate_rejected', slot=slot, stage='pixels',
                            reason=pixel_failures[0], failed_conditions=pixel_failures,
                            pixel_stats=pixel_stats)
                continue
            raw_ids = frozenset(np.flatnonzero(mapping == slot).tolist())
            if not raw_ids:
                self._event(frame, 'candidate_rejected', slot=slot, stage='association',
                            reason='no_source_raw_ids', pixel_stats=pixel_stats)
                continue
            blocking = self._blocking_candidates(raw_ids, frame)
            if blocking:
                self._event(frame, 'candidate_rejected', slot=slot, stage='association',
                            reason='existing_candidate_or_cooldown', pixel_stats=pixel_stats,
                            blocking_candidates=blocking)
                continue
            if len(self.candidates) >= self.cfg['max_candidates']:
                self._event(frame, 'candidate_limit_reached', slot=slot,
                            pixel_stats=pixel_stats)
                break
            reference, reference_colors = self._reference(slot)
            if len(reference) < self.cfg['min_reference_points']:
                self._event(frame, 'candidate_rejected', slot=slot, stage='reference',
                            reason='insufficient_reference_surface', points=len(reference),
                            pixel_stats=pixel_stats)
                continue
            codes, candidate_sources = project_reference(
                reference, reference_colors, self.intrinsic, pose, current, history,
                owners, current_interior, rgb_context, self.cfg, self.near, self.far)
            ref_stats = evidence_stats(codes, sources=candidate_sources)
            reference_failures = []
            if ref_stats['tested'] < self.cfg['min_reference_tested_points']:
                reference_failures.append('insufficient_reference_tested_points')
            if ref_stats['contradiction_ratio'] < self.cfg['contradiction_ratio']:
                reference_failures.append('low_reference_contradiction_ratio')
            if reference_failures:
                self._event(frame, 'candidate_rejected', slot=slot, stage='reference',
                            reason='insufficient_reference_evidence',
                            failed_conditions=reference_failures,
                            pixel_stats=pixel_stats, stats=ref_stats)
                continue
            candidate = Candidate(len(self.candidates), raw_ids, slot, frame,
                                  reference, reference_colors)
            self.candidates.append(candidate)
            self.fixed.observe(candidate, frame, pose, codes, candidate_sources)
            reason, stats = candidate.observe(frame, pose, codes, candidate_sources, self.cfg)
            self.fixed.verify_original(candidate)
            row = dict(stats, candidate_id=candidate.candidate_id, state=candidate.state,
                       confirmations=candidate.hits, current_slot=slot, reason='created')
            frame_stats.append(row)
            self._event(frame, 'candidate_created', session=session, **row,
                        pixel_stats=pixel_stats, source_raw_ids=sorted(raw_ids))
            self._save_candidate(candidate, frame, 'created', codes, candidate_sources, stats)
            if reason == 'would_remove':
                self._event(frame, 'candidate_observation', session=session, **dict(row, reason=reason))
                self._save_candidate(candidate, frame, reason, codes, candidate_sources, stats)
            changed = True
        self.latest_stats = frame_stats
        rgb_stats = dict(threshold=float(rgb_context['threshold']),
                         median=float(rgb_context['median']), mad=float(rgb_context['mad']),
                         eligible_pixels=rgb_context['eligible_pixels'],
                         contradiction_pixels=rgb_context['contradiction_pixels'])
        self._event(frame, 'frame_evaluated', session=session, pixel_stats=global_stats,
                    rgb_stats=rgb_stats, ambiguous_pixels=int((owners == -2).sum()),
                    excluded_contradiction_pixels=excluded_stats,
                    candidate_observations=len(frame_stats))
        if changed or frame % self.cfg['dump_interval'] == 0:
            self._dump_frame(frame, as_numpy(color), current, history, owners, evidence,
                             sources, rgb_context, frame_stats)
