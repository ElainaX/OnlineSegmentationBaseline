"""Geometry/lifecycle regressions; no claims about segmentation accuracy."""
import tempfile, unittest
import numpy as np
from scipy.spatial import cKDTree
from submap_disjoint.independent_tsdf import TSDFOptions, IndependentTSDFStore, LocalTSDF
from run_repair_controls import render_synthetic


class IndependentGeometryTests(unittest.TestCase):
    def setUp(self):
        self.options=TSDFOptions(device='CUDA:0')
        self.store=IndependentTSDFStore(self.options)
        for k in ('background','old'): self.store.create(k)
        self.frames=[render_synthetic(2.203,2.013,i) for i in range(6)]
        for f in self.frames:
            for key,mask in [('background',~f['mask']),('old',f['mask'])]:
                self.store.integrate(key,f['frame'],f['rgb'],f['depth'],f['K'],f['pose'],mask)

    def test_link_does_not_fuse_and_deactivate_does_not_change_neighbor(self):
        bg=self.store.fragments['background'].volume.surface()[0].copy()
        obj=self.store.fragments['old'].volume.surface()[0].copy()
        self.assertGreater(len(bg),100); self.assertGreater(len(obj),50)
        self.store.link('possibly_same_identity',['background','old'])
        self.assertEqual(len(self.store.fragments),2)
        self.store.set_status('old','inactive','oracle disappearance')
        live=self.store.surface()[0]
        self.assertEqual(len(live),len(bg))
        self.assertLess(cKDTree(live).query(bg)[0].max(),1e-7)
        self.store.set_status('old','frozen','manual reactivation for regression')
        self.assertEqual(len(self.store.surface()[0]),len(bg)+len(obj))
        # GPU extraction order is not stable; compare the point sets.
        self.assertLess(cKDTree(bg).query(self.store.fragments['background'].volume.surface()[0])[0].max(),1e-7)

    def test_duplicate_observation_and_frozen_write_guard(self):
        f=self.frames[0]
        self.assertFalse(self.store.integrate('old',0,f['rgb'],f['depth'],f['K'],f['pose'],f['mask']))
        with self.assertRaises(ValueError):
            self.store.integrate('old',0,f['rgb'],f['depth'],f['K'],f['pose'],~f['mask'])
        self.store.set_status('old','frozen','end of state')
        with self.assertRaises(RuntimeError):
            self.store.integrate('old',7,f['rgb'],f['depth'],f['K'],f['pose'],f['mask'])

    def test_save_restore_preserves_inactive_state(self):
        self.store.set_status('old','inactive','test')
        with tempfile.TemporaryDirectory() as tmp:
            self.store.save(tmp,save_volumes=True)
            recovered=IndependentTSDFStore.load(tmp)
            self.assertEqual(recovered.fragments['old'].status,'inactive')
            np.testing.assert_array_equal(recovered.fragments['old'].masks[0],self.store.fragments['old'].masks[0])
            a=self.store.surface()[0]; b=recovered.surface()[0]
            self.assertEqual(len(a),len(b))
            self.assertLess(cKDTree(a).query(b)[0].max(),1e-7)

    def test_local_coordinate_anchor_invariance(self):
        # Integer-voxel translation leaves the sampling lattice identical.
        T=np.eye(4); T[:3,3]=[.2,-.1,.05]
        shifted=LocalTSDF(self.options,T)
        for f in self.frames:
            shifted.integrate(f['rgb'],f['depth'],f['K'],f['pose'],~f['mask'])
        a=self.store.fragments['background'].volume.surface()[0]; b=shifted.surface()[0]
        self.assertLess(np.quantile(cKDTree(a).query(b)[0],.99),1e-4)

    def test_budget_failure_is_explicit(self):
        tiny=LocalTSDF(TSDFOptions(initial_blocks=1,max_blocks=1,device='CUDA:0'))
        f=self.frames[0]
        with self.assertRaises(MemoryError): tiny.integrate(f['rgb'],f['depth'],f['K'],f['pose'])

    def test_spatial_clear_and_contribution_support(self):
        global_volume=LocalTSDF(self.options)
        for f in self.frames: global_volume.integrate(f['rgb'],f['depth'],f['K'],f['pose'])
        before=len(global_volume.surface()[0])
        self.assertEqual(global_volume.clear_aabb([10,10,10],[11,11,11]),0)
        self.assertEqual(len(global_volume.surface()[0]),before)
        changed=global_volume.clear_observed_support(self.store.fragments['old'].volume)
        self.assertGreater(changed,0)
        self.assertGreater(len(global_volume.surface()[0]),100)
        self.assertLess(len(global_volume.surface()[0]),before)


if __name__=='__main__': unittest.main()
