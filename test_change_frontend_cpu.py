"""CPU evidence/controller regressions. Fake rendering is NOT a GPU raycast test."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from submap_disjoint.contradiction import AuditOptions, SUPPORT, FREE, OCCLUDED
from submap_disjoint.change_frontend import (FrontendOptions, classify, interior,
    owners_from_stencil, owner_interior, rgb_context, rgb01, raycast_validity,
    compose_history, prepare_context, fragment_evidence, ownership_slots)
from submap_disjoint.change_lifecycle import ChangeLifecycle, LifecycleOptions


def pose(x=0):
    T = np.eye(4); T[0, 3] = x; return T


class FakeVolume:
    def __init__(self, rendered, points):
        self.rendered = rendered; self.points = points; self.integrations = 5
    def surface(self, weight_threshold=None): return self.points.copy(), np.zeros_like(self.points)
    def bytes_allocated_attributes(self): return 1000


class FakeStore:
    saved = {}
    def __init__(self):
        self.options = SimpleNamespace(near_m=.1, far_m=5.)
        self.fragments = {}; self.events = []
    def set_status(self, key, status, reason):
        self.fragments[key].status = status
        self.events.append((key, status, reason))
    def create(self, key):
        src = copy.deepcopy(self.fragments['old'])
        src.status = 'active'; src.observations = []; src.volume.integrations = 0
        self.fragments[key] = src
    def integrate(self, key, frame, *args):
        f = self.fragments[key]; f.observations.append({'frame': frame}); f.volume.integrations += 1
        return True
    def save(self, directory, save_volumes=False):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.saved[str(directory)] = copy.deepcopy(self)
    @classmethod
    def load(cls, directory): return copy.deepcopy(cls.saved[str(directory)])


def fixture():
    h, w = 40, 64; K = np.array([[30., 0, 31.5], [0, 30., 19.5], [0, 0, 1]])
    roi = np.zeros((h, w), bool); roi[8:32, 8:32] = True
    render = dict(depth=np.where(roi, 2., 0.).astype(np.float32), color=np.full((h,w,3), .5, np.float32),
        valid=roi.copy(), weight=np.where(roi, 5., 0.))
    background = dict(depth=np.full((h,w), 4., np.float32), color=np.full((h,w,3), .5, np.float32),
        valid=np.ones((h,w), bool), weight=np.full((h,w), 5.))
    v,u = np.nonzero(roi); points = np.column_stack(((u-K[0,2])*2/K[0,0], (v-K[1,2])*2/K[1,1], np.full(len(u), 2.)))
    store = FakeStore()
    for key, r in [('old', render), ('background', background)]:
        store.fragments[key] = SimpleNamespace(volume=FakeVolume(r, points), status='active', observations=[{'frame': i} for i in range(5)])
    opt = FrontendOptions(enabled=True, interval=1, min_valid_pixels=10, min_reference_points=10,
        min_reference_tested_points=10, min_rgb_calibration_pixels=10)
    ctl = ChangeLifecycle(store, LifecycleOptions(min_views=2, min_cells=5, view_distance_m=.05), AuditOptions(), opt)
    ctl.register_reference('old')
    rgb = np.full((h,w,3), 128, np.uint8); depth = np.where(roi, 2., 4.).astype(np.float32)
    return ctl, store, roi, rgb, depth, K


class PureFrontendTests(unittest.TestCase):
    def test_depth_sign_and_additive_tolerance(self):
        code = classify(np.array([2.1, 2.3, 1., 0., np.nan]), np.full(5, 2.), np.ones(5, bool), FrontendOptions())
        np.testing.assert_array_equal(code, [SUPPORT, FREE, OCCLUDED, 0, 0])

    def test_dark_uint8_not_mistaken_for_normalized_color(self):
        np.testing.assert_allclose(rgb01(np.ones((2,2,3), np.uint8)), 1/255)
        with self.assertRaises(ValueError): rgb01(np.full((2,2,3), 128., np.float32))

    def test_holes_and_owner_boundaries_are_excluded(self):
        depth = np.ones((7,7)); depth[3,3] = 0
        result = interior(depth, depth > 0, .08, 1)
        self.assertFalse(result[3,4]); self.assertFalse(result[0,0]); self.assertTrue(result[1,1])
        owners = np.zeros((7,7), int); owners[:,4:] = 1
        keep = owner_interior(owners, 1)
        self.assertFalse(keep[3,3]); self.assertTrue(keep[3,1])

    def test_stencil_multiple_raw_ids_do_not_outvote_another_instance(self):
        idx = np.array([[1,2,3], [1,2,4], [-1,-1,-1]])
        mapping = {1:{7}, 2:{7}, 3:{7}, 4:{8}}
        out = owners_from_stencil(idx, idx >= 0, np.full(idx.shape, 1/3), lambda keys: [mapping[int(k)] for k in keys])
        np.testing.assert_array_equal(out, [7, -2, -1])

    def test_partial_interpolation_and_low_weight_are_rejected(self):
        ids = np.tile(np.arange(8), (3,1)); mask = np.ones_like(ids, bool); mask[0,0] = False
        weights = np.full(ids.shape, 4.); weights[1] = 2.
        valid, _, _ = raycast_validity(np.full(3,2.), ids, mask, np.full(ids.shape, .125), weights, FrontendOptions(), .1, 5.)
        np.testing.assert_array_equal(valid, [False, False, True])

    def test_same_depth_fragments_are_ambiguous_order_independently(self):
        _, store, _, _, depth, _ = fixture(); a = store.fragments['old'].volume.rendered
        for order in (['a','b'], ['b','a']):
            h = compose_history({'a':a,'b':a}, order, {'a':0,'b':1}, depth.shape, FrontendOptions())
            self.assertTrue(np.all(h['owners'][a['valid']] == -2))

    def test_explicit_identity_links_do_not_create_false_owner_conflicts(self):
        _, store, _, _, depth, _ = fixture(); a = store.fragments['old'].volume.rendered
        store.identities = {'object': {'a', 'b'}}
        slots = ownership_slots(store, ['a','b'])
        h = compose_history({'a':a,'b':a}, ['a','b'], slots, depth.shape, FrontendOptions())
        self.assertTrue(np.all(h['owners'][a['valid']] >= 0))
        store.identities['other'] = {'b'}
        with self.assertRaises(ValueError): ownership_slots(store, ['a','b'])

    def test_rgb_compensates_global_offset_but_detects_local_change(self):
        history = np.full((30,40,3), .4, np.float32)
        current = history + .1; eligible = np.ones((30,40), bool)
        cfg = FrontendOptions(min_rgb_calibration_pixels=10)
        self.assertEqual(int(rgb_context(current, history, eligible, cfg)['bad'].sum()), 0)
        current[10:20,10:20] = [1.,0.,0.]
        result = rgb_context(current, history, eligible, cfg)
        self.assertGreater(int(result['bad'][12:18,12:18].sum()), 0)
        self.assertEqual(int(result['bad'][:5,:5].sum()), 0)

    def test_nan_color_not_eligible_and_small_calibration_set_disables_rgb(self):
        c = np.full((5,5,3), .5, np.float32); c[2,2] = np.nan
        r = rgb_context(c, np.zeros_like(c), np.ones((5,5), bool), FrontendOptions())
        self.assertFalse(r['enabled']); self.assertEqual(r['calibration_pixels'], 24)
        self.assertFalse(r['bad'].any())

    def test_rgb_disabled_does_not_import_or_execute_color_processing(self):
        with patch.dict('sys.modules', {'cv2': None}):
            result = rgb_context(None, None, np.ones((5,5), bool), FrontendOptions(rgb_enabled=False))
        self.assertFalse(result['enabled']); self.assertIsNone(result['current_lab'])

    def test_options_fail_early(self):
        with self.assertRaises(ValueError): FrontendOptions(interval=0)
        with self.assertRaises(ValueError): FrontendOptions(min_map_weight=0)
        with self.assertRaises(ValueError): FrontendOptions(rgb_enabled=1)


class ControllerCpuTests(unittest.TestCase):
    def setUp(self):
        self.ctl, self.store, self.roi, self.rgb, self.depth, self.K = fixture()
        self.mock = patch('submap_disjoint.change_lifecycle.render_quality', side_effect=lambda v,*args: v.rendered)
        self.renderer = self.mock.start(); self.addCleanup(self.mock.stop)

    def step(self, frame, depth=None, rgb=None, x=0, assignments=None):
        return self.ctl.step(frame, self.rgb if rgb is None else rgb,
            self.depth if depth is None else depth, self.K, pose(x), assignments)

    def test_rgb_only_candidate_never_authorizes_geometry_retirement(self):
        rgb = self.rgb.copy(); rgb[self.roi] = [255,0,0]
        for i,x in enumerate([0.,.1,-.1,.2,-.2]): row = self.step(i+10, rgb=rgb, x=x)
        tr = self.ctl.tracks['old']
        self.assertEqual(tr.state, 'suspect')
        self.assertTrue(tr.last_stats['candidate_started'])
        self.assertGreater(tr.last_stats['frontend']['rgb_reference'], 0)
        self.assertEqual(tr.last_stats['confirmed_free_cells'], 0)
        self.assertNotEqual(self.store.fragments['old'].status, 'inactive')

    def test_depth_candidate_drives_existing_reversible_backend(self):
        dep = self.depth.copy(); dep[self.roi] = 4.
        for i,x in enumerate([0.,.1,-.1]): row = self.step(i+10, depth=dep, x=x)
        self.assertEqual(self.ctl.tracks['old'].state, 'retired')
        for i,x in enumerate([0.,.1,-.1]): row = self.step(i+20, x=x)
        self.assertEqual(self.ctl.tracks['old'].state, 'stable')

    def test_occlusion_does_not_produce_disappearance_candidate(self):
        dep = self.depth.copy(); dep[self.roi] = 1.
        for i,x in enumerate([0.,.1,-.1]): row = self.step(i+10, depth=dep, x=x)
        self.assertEqual(self.ctl.tracks['old'].state, 'stable')
        self.assertFalse(self.ctl.tracks['old'].last_stats['candidate_started'])

    def test_widespread_conflict_pauses_state_updates(self):
        before = self.ctl.tracks['old'].state
        row = self.step(10, depth=np.full_like(self.depth, 4.9))
        self.assertIn('widespread_conflict', row['frontend']['skip_reasons'])
        self.assertEqual(self.ctl.tracks['old'].state, before)
        self.assertEqual(len(self.ctl.tracks['old'].history.records), 0)

    def test_quality_failure_cannot_add_votes(self):
        self.store.fragments['old'].volume.rendered['valid'][:] = False
        row = self.step(10)
        self.assertEqual(len(self.ctl.tracks['old'].history.records), 0)

    def test_candidate_requires_pixel_and_reference_evidence(self):
        self.ctl.frontend_options = replace(self.ctl.frontend_options, min_valid_pixels=5000)
        dep = self.depth.copy(); dep[self.roi] = 4.
        row = self.step(10, depth=dep)
        self.assertEqual(self.ctl.tracks['old'].state, 'stable')
        self.assertFalse(row['events'])

    def test_existing_candidate_still_requires_enough_tested_reference_points(self):
        tr = self.ctl.tracks['old']; tr.state = 'suspect'; tr.last_stats['candidate_started'] = True
        code = np.zeros(len(tr.points), np.uint8); code[0] = FREE
        detail = dict(candidate=False, tested_reference=1, candidate_failures=['insufficient_tested_reference'])
        with patch('submap_disjoint.change_lifecycle.fragment_evidence', return_value=(code, np.zeros_like(code, bool), detail)):
            self.step(10)
        self.assertEqual(len(tr.history.records), 0)
        self.assertEqual(tr.last_stats['frontend']['paused'], 'insufficient_tested_reference')

    def test_private_reference_cannot_vote_through_ambiguous_ownership(self):
        own = self.store.fragments['old'].volume.rendered
        slots = {'old':0, 'background':1}
        rendered = {k:f.volume.rendered for k,f in self.store.fragments.items()}
        history = compose_history(rendered, list(rendered), slots, self.depth.shape, self.ctl.frontend_options)
        history['owners'][self.roi] = -2
        ctx = prepare_context(self.rgb, self.depth, history, self.ctl.frontend_options, .1, 5.)
        codes, appearance, detail = fragment_evidence(self.ctl.tracks['old'].points, self.rgb, self.depth,
            self.K, pose(), own, history, ctx, 0, 5, self.ctl.frontend_options, .1, 5., .04, private=True)
        self.assertFalse(codes.any()); self.assertFalse(appearance.any())

    def test_sampling_skips_rendering_and_evidence_tick(self):
        self.ctl.frontend_options = replace(self.ctl.frontend_options, interval=10)
        row = self.step(1)
        self.assertFalse(row['frontend']['audit_due']); self.assertEqual(self.ctl.tick, 0)
        self.renderer.assert_not_called()
        row = self.step(10); self.assertEqual(self.ctl.tick, 1)

    def test_immature_reference_fails_explicitly(self):
        self.store.fragments['background'].observations = [{}, {}, {}]
        with self.assertRaises(ValueError): self.ctl.register_reference('background')

    def test_batch_registration_retains_small_or_immature_reference(self):
        self.store.fragments['background'].observations = [{}, {}, {}]
        excluded = self.ctl.register_available_references()
        self.assertIn('background', excluded)
        self.assertEqual(self.store.fragments['background'].status, 'frozen')
        self.assertIn('old', self.ctl.tracks)
        self.assertNotIn('background', self.ctl.tracks)

    def test_disabled_frontend_preserves_previous_depth_path(self):
        self.ctl.frontend_options = FrontendOptions(enabled=False)
        with patch('submap_disjoint.change_lifecycle.raycast_volume', side_effect=lambda volume,*args: volume.rendered['depth']):
            row = self.step(10)
        self.renderer.assert_not_called()
        self.assertFalse(row['frontend']['enabled'])
        self.assertEqual(self.ctl.tracks['old'].state, 'stable')

    def test_pending_matures_before_sealing(self):
        for i in range(4): self.step(i+10, assignments={'new': self.roi})
        self.assertEqual(len(self.ctl.tracks['new'].points), 0)
        self.step(14, assignments={'new': self.roi})
        self.assertEqual(len(self.store.fragments['new'].observations), 5)
        self.step(15, assignments={'new': self.roi})
        self.assertGreater(len(self.ctl.tracks['new'].points), 0)
        self.assertEqual(len(self.store.fragments['new'].observations), 5)

    def test_save_restore_preserves_frontend_and_candidate_evidence(self):
        rgb = self.rgb.copy(); rgb[self.roi] = [255,0,0]
        self.step(10, rgb=rgb)
        with tempfile.TemporaryDirectory() as tmp:
            self.ctl.save(tmp)
            module = SimpleNamespace(IndependentTSDFStore=FakeStore)
            with patch.dict('sys.modules', {'submap_disjoint.independent_tsdf': module}):
                restored = ChangeLifecycle.load(tmp)
            self.assertEqual(restored.frontend_options, self.ctl.frontend_options)
            a = self.step(11, rgb=rgb, x=.1)
            b = restored.step(11, rgb, self.depth, self.K, pose(.1))
            self.assertEqual(a['fragments'], b['fragments'])
            data = json.loads((Path(tmp)/'change_lifecycle.json').read_text())
            data['format'] = 1; data.pop('frontend_options')
            (Path(tmp)/'change_lifecycle.json').write_text(json.dumps(data))
            with patch.dict('sys.modules', {'submap_disjoint.independent_tsdf': module}):
                legacy = ChangeLifecycle.load(tmp)
            self.assertFalse(legacy.frontend_options.enabled)


if __name__ == '__main__': unittest.main()
