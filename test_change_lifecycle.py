"""Behavioral regressions, including actual CUDA TSDF/raycast round trips."""
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from submap_disjoint.contradiction import (AuditOptions, SUPPORT, FREE, OCCLUDED,
    audit_surface, depth_edges, raycast_volume)
from submap_disjoint.change_lifecycle import ChangeLifecycle, LifecycleOptions, EvidenceWindow
from submap_disjoint.independent_tsdf import TSDFOptions, IndependentTSDFStore


def view(x=0):
    T = np.eye(4); T[0, 3] = x
    return T


class EvidenceTests(unittest.TestCase):
    def test_depth_sign_unknown_edges_and_visibility(self):
        K = np.array([[10., 0, 5], [0, 10., 5], [0, 0, 1]])
        ref = np.full((11, 11), 2.)
        pts = np.array([[0., 0, 2]])
        for depth, expected in [(2., SUPPORT), (3., FREE), (1., OCCLUDED), (0., 0), (np.nan, 0)]:
            state = audit_surface(pts, np.full_like(ref, depth), ref, ref, K, view(), AuditOptions())
            self.assertEqual(int(state[0]), expected)
        obs = ref.copy(); obs[5, 6] = 0
        self.assertEqual(audit_surface(pts, obs, ref, ref, K, view(), AuditOptions())[0], 0)
        self.assertEqual(audit_surface(np.array([[0., 0, 3]]), ref * 2, ref, ref, K, view(), AuditOptions())[0], 0)

    def test_repeated_views_do_not_inflate_and_later_support_replaces_free(self):
        w = EvidenceWindow(LifecycleOptions())
        for t in range(20): w.update(t, view(), np.full(10, FREE, np.uint8))
        self.assertEqual(w.counts(10)[1].max(), 1)
        w.update(20, view(), np.full(10, SUPPORT, np.uint8))
        self.assertEqual(w.counts(10)[1].max(), 0)
        for i in range(1, 4): w.update(20 + i, view(i * .2), np.full(10, SUPPORT, np.uint8))
        self.assertEqual(w.counts(10)[0].max(), 4)

    def test_unknown_is_neutral_and_expiry_is_bounded(self):
        w = EvidenceWindow(LifecycleOptions(history_age=4))
        w.update(0, view(), np.full(3, FREE, np.uint8))
        w.update(1, view(), np.full(3, OCCLUDED, np.uint8))
        self.assertEqual(w.counts(3)[1].max(), 1)
        w.update(5, view(), np.zeros(3, np.uint8))
        self.assertEqual(len(w.records), 0)

    def test_observing_one_cell_does_not_refresh_stale_other_cells(self):
        w = EvidenceWindow(LifecycleOptions(history_age=4))
        w.update(0, view(), np.array([FREE, FREE], np.uint8))
        for t in range(1, 6): w.update(t, view(), np.array([SUPPORT, 0], np.uint8))
        support, free = w.counts(2)
        np.testing.assert_array_equal(support, [1, 0])
        np.testing.assert_array_equal(free, [0, 0])

    def test_slow_camera_motion_eventually_adds_a_distinct_view(self):
        w = EvidenceWindow(LifecycleOptions())
        for t in range(8): w.update(t, view(.02 * t), np.full(2, FREE, np.uint8))
        self.assertGreaterEqual(len(w.records), 2)


class GeometryLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.store = IndependentTSDFStore(TSDFOptions(device='CUDA:0', initial_blocks=128))
        self.K = np.array([[65., 0, 31.5], [0, 65., 23.5], [0, 0, 1]])
        self.rgb = np.full((48, 64, 3), 150, np.uint8)
        # Avoid an exact zero crossing on the TSDF lattice: Open3D point-cloud
        # extraction can miss a perfectly grid-aligned synthetic plane.
        self.depth = np.full((48, 64), 2.013, np.float32)
        self.mask = np.zeros((48, 64), bool); self.mask[10:38, 15:49] = True
        self.store.create('old')
        for i in range(3): self.store.integrate('old', i, self.rgb, self.depth, self.K, view(), self.mask)
        self.controller = ChangeLifecycle(self.store)
        self.controller.register_reference('old')

    def step(self, f, z, x=0, assignments=None):
        return self.controller.step(f, self.rgb, np.full_like(self.depth, z + .013 if z > 0 else z), self.K, view(x), assignments)

    def test_old_raycast_does_not_depend_on_new_depth_blocks(self):
        pred = raycast_volume(self.store.fragments['old'].volume, self.K, view(), self.depth.shape)
        self.assertAlmostEqual(float(pred[24, 32]), 2., delta=.03)

    def test_static_occluded_missing_do_not_retire(self):
        for f, z in enumerate([2., 1., 0., 2., 1., 0.]): self.step(f + 10, z, (f % 3) * .15)
        self.assertNotEqual(self.controller.tracks['old'].state, 'retired')
        self.assertEqual(self.store.fragments['old'].volume.integrations, 3)

    def test_multiview_disappearance_and_recovery(self):
        for i in range(5): self.step(10 + i, 3., 0)
        self.assertNotEqual(self.controller.tracks['old'].state, 'retired')
        self.step(20, 3., .15); self.step(21, 3., -.15)
        self.assertEqual(self.controller.tracks['old'].state, 'retired')
        self.assertEqual(len(self.store.surface()[0]), 0)
        for i, x in enumerate([0., .15, -.15]): self.step(30 + i, 2., x)
        self.assertEqual(self.controller.tracks['old'].state, 'stable')
        self.assertGreater(len(self.store.surface()[0]), 0)

    def test_small_local_contradiction_does_not_delete_whole_fragment(self):
        dep = self.depth.copy(); dep[22:27, 30:35] = 3.
        for i, x in enumerate([0., .15, -.15]):
            self.controller.step(10 + i, self.rgb, dep, self.K, view(x))
        self.assertNotEqual(self.controller.tracks['old'].state, 'retired')

    def test_proposal_acceptance_does_not_retire_occluded_old(self):
        old_count = len(self.store.surface()[0])
        for i in range(3): self.step(10 + i, 1., 0, {'new': self.mask})
        self.assertEqual(self.store.fragments['new'].status, 'pending')
        self.assertEqual(len(self.store.surface()[0]), old_count)
        for i, x in enumerate([0., .12, -.12, .25, -.25]):
            row = self.step(20 + i, 1., x)
        self.assertEqual(self.controller.tracks['new'].state, 'accepted')
        self.assertNotEqual(self.controller.tracks['old'].state, 'retired')
        self.assertEqual(self.store.fragments['new'].volume.integrations, 3)

    def test_pending_rejection_and_timeout(self):
        for i in range(3): self.step(10 + i, 1., 0, {'bad': self.mask})
        for i, x in enumerate([0., .12, -.12, .25, -.25]): self.step(20 + i, 3., x)
        self.assertEqual(self.controller.tracks['bad'].state, 'rejected')
        self.assertEqual(self.store.fragments['bad'].status, 'inactive')
        short = ChangeLifecycle(self.store, LifecycleOptions(max_pending_steps=2))
        for i in range(3): short.step(100 + i, self.rgb, self.depth, self.K, view(), {'unseen': self.mask} if i == 0 else {})
        self.assertEqual(short.tracks['unseen'].state, 'unresolved')

    def test_proposal_geometry_stops_growing_after_sampling(self):
        for i in range(3): self.step(10 + i, 1., 0, {'new': self.mask})
        # Same view cannot accept it; a greatly expanded assignment must not
        # add unverified geometry to the now-fixed reference samples.
        row = self.step(13, 1., 0, {'new': np.ones_like(self.mask)})
        self.assertIn('new', row['sealed_proposals'])
        self.assertEqual(self.store.fragments['new'].volume.integrations, 3)

    def test_save_reload_continues_same_decision(self):
        self.step(10, 3.)
        with tempfile.TemporaryDirectory() as tmp:
            self.controller.save(tmp)
            restored = ChangeLifecycle.load(tmp)
            np.testing.assert_array_equal(restored.tracks['old'].points, self.controller.tracks['old'].points)
            for i, x in enumerate([.15, -.15]):
                a = self.step(20 + i, 3., x)
                b = restored.step(20 + i, self.rgb, np.full_like(self.depth, 3.013), self.K, view(x))
                self.assertEqual(a['fragments'], b['fragments'])
            self.assertEqual(restored.tracks['old'].state, 'retired')

    def test_validation_before_mutation(self):
        pose = view(); pose[0, 3] = np.nan
        with self.assertRaises(ValueError): self.controller.step(10, self.rgb, self.depth, self.K, pose)
        with self.assertRaises(ValueError): self.step(10, 2., assignments={'old': self.mask})
        with self.assertRaises(ValueError): self.step(10, 2., assignments={'a': self.mask, 'b': self.mask})
        self.assertEqual(self.controller.tick, 0)
        self.assertEqual(len(self.store.fragments), 1)
        self.controller.options = LifecycleOptions(max_fragments=1)
        with self.assertRaises(MemoryError): self.step(10, 2., assignments={'new': self.mask})

    def test_manifest_runner_and_resume(self):
        from run_change_lifecycle import run_manifest
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows = []
            for frame in range(6):
                packet = root / ('%d.npz' % frame)
                data = dict(rgb=self.rgb, depth_m=self.depth if frame < 3 else np.full_like(self.depth, 3.013),
                    K=self.K, pose=view(0 if frame < 3 else [0., .15, -.15][frame - 3]))
                if frame < 3: data['mask__old'] = self.mask
                np.savez_compressed(packet, **data)
                rows.append(dict(frame=frame, packet=packet.name, phase='reference' if frame < 3 else 'live'))
            manifest = root / 'input.jsonl'; manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            out = root / 'first'; out.mkdir()
            run_manifest(SimpleNamespace(config=None, resume=None, device='CUDA:0', manifest=str(manifest)), out)
            self.assertEqual(json.loads((out / 'summary.json').read_text())['states']['old'], 'retired')
            rows = []
            for i, x in enumerate([0., .15, -.15]):
                packet = root / ('return%d.npz' % i)
                np.savez_compressed(packet, rgb=self.rgb, depth_m=self.depth, K=self.K, pose=view(x))
                rows.append(dict(frame=10 + i, packet=packet.name, phase='live'))
            manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            resumed = root / 'resumed'; resumed.mkdir()
            run_manifest(SimpleNamespace(config=None, resume=str(out / 'checkpoint'), device='CUDA:0', manifest=str(manifest)), resumed)
            self.assertEqual(json.loads((resumed / 'summary.json').read_text())['states']['old'], 'stable')


if __name__ == '__main__': unittest.main()
