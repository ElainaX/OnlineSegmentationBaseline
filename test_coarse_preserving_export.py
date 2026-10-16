import ast
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

from submap_disjoint.coarse_preserving_export import (
    substitute_fine_regions, multiview_surface_support)
from submap_disjoint.it_adapter_multires import InstanceTSDFAdapter
from tool.post_process import filter_instances, export_instance_mask


class PreserveCoarseTests(unittest.TestCase):
    def setUp(self):
        self.xyz = np.array([[0., 0., 1.], [.025, 0., 1.], [.05, 0., 1.],
                             [.075, 0., 1.], [1., 0., 1.]])
        self.rgb = np.ones_like(self.xyz) * .5
        # Ordinary point, fine point, shared boundary, fine point, background.
        self.masks = np.array([[1, 0], [0, 1], [1, 1], [0, 1], [0, 0]], bool)

    def test_no_candidate_is_exact_identity(self):
        x, c, m, keep, records = substitute_fine_regions(
            self.xyz, self.rgb, self.masks, {}, .05)
        self.assertIs(x, self.xyz); self.assertIs(m, self.masks)
        self.assertFalse(records)
        np.testing.assert_array_equal(keep, np.arange(5))

    def test_ordinary_shared_background_and_uncovered_points_preserved(self):
        fine = np.array([[.024, 0., 1.], [.026, 0., 1.]])
        x, c, m, keep, records = substitute_fine_regions(
            self.xyz, self.rgb, self.masks, {1: (fine, np.ones_like(fine))}, .02)
        np.testing.assert_array_equal(keep, [0, 2, 3, 4])
        np.testing.assert_array_equal(x[m[:, 0]], self.xyz[self.masks[:, 0]])
        np.testing.assert_array_equal(c[:len(keep)], self.rgb[keep])
        np.testing.assert_array_equal(m[:len(keep)], self.masks[keep])
        self.assertEqual(records[0]['removed_coarse_points'], 1)

    def test_no_supported_fine_falls_back(self):
        fine = np.array([[0., 0., 1.], [.05, 0., 1.], [1., 0., 1.], [9., 0., 1.]])
        result = substitute_fine_regions(self.xyz, self.rgb, self.masks,
                                          {1: (fine, fine)}, .05)
        self.assertIs(result[0], self.xyz); self.assertFalse(result[-1])

    def test_distinct_frames_depth_and_mask_are_required(self):
        pts = np.array([[0., 0., 1.], [1., 0., 1.], [0., 0., 1.2], [0., 0., -1.]])
        K = np.eye(3); mask = np.array([[1, 0], [1, 1]], bool)
        def view(f):
            return ({'frame': f, 'depth': np.ones((2, 2)), 'pose': np.eye(4)}, mask)
        r = multiview_surface_support(pts, [view(1), view(1), view(2)], K, .1, 5., .05, 3)
        self.assertFalse(r.any())
        r = multiview_surface_support(pts, [view(1), view(2), view(3)], K, .1, 5., .05, 3)
        np.testing.assert_array_equal(r, [1, 0, 0, 0])

    def test_camera_pose_translation(self):
        pose = np.eye(4); pose[:3, 3] = [10, 20, 30]
        view = ({'frame': 0, 'depth': np.ones((1, 1)), 'pose': pose}, np.ones((1, 1), bool))
        result = multiview_surface_support(np.array([[10., 20., 31.]]), [view], np.eye(3), .1, 5., .05, 1)
        self.assertTrue(result[0])

    def adapter(self, root, fine=False):
        audit = root / 'ig_audit'; audit.mkdir()
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(self.xyz))
        pc.colors = o3d.utility.Vector3dVector(self.rgb)
        o3d.io.write_point_cloud(str(audit / 'final.ply'), pc)
        np.savez(audit / 'ckpt_final.npz', pred_masks=self.masks,
                 pred_score=np.array([.3, .9]), pred_classes=np.array([3, 9]),
                 pred_sem_features=np.arange(8).reshape(4, 2))
        (audit / 'export_instance_ids.json').write_text('[7, 42]')
        obj = InstanceTSDFAdapter.__new__(InstanceTSDFAdapter)
        obj.flush = lambda _: None
        obj.packets = [dict(frame=f, labels=np.array([[1, 2, 3]])) for f in range(3)]
        obj._frame_masks = lambda p: (None, {7: [1], 42: [2], 99: [3]})
        checked = []
        obj._track_is_fine = lambda mid, views: (checked.append(mid) or (fine and mid == 42))
        def replay(views, options):
            if not fine:
                raise AssertionError('No fine candidate must never replay TSDF')
            return np.array([[.024, 0., 1.]]), np.array([[128., 128., 128.]]), 3
        obj._integrate_volume_with_options = replay
        obj.fine_options = None; obj.fine_track_ids = set(); obj.track_decisions = {}
        obj.options = types.SimpleNamespace(near_m=.1, far_m=5., voxel_m=.025)
        obj.K = np.eye(3); obj.fine_replace_radius_m = .02; obj.fine_min_views = 3
        obj.fine_max_views = 5; obj.fine_max_extent_m = .45; obj.fine_voxel_m = .015
        obj.multires_enabled = True; obj.audit = {}
        def save_pc(x, c, name):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(x))
            pc.colors = o3d.utility.Vector3dVector(c); o3d.io.write_point_cloud(name, pc)
        obj.owner = types.SimpleNamespace(save_pc=save_pc)
        # Put the target raw mask at the projected fine pixel.
        obj.packets = [dict(frame=f, labels=np.array([[2, 1, 3]]), depth=np.ones((1, 3)), pose=np.eye(4)) for f in range(3)]
        return obj, checked

    def test_adapter_exact_fallback_and_final_valid_ids_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); obj, checked = self.adapter(root)
            report = obj.export(2, root)
            self.assertEqual(checked, [7, 42])
            self.assertTrue(report['audit']['fallback_to_ig'])
            for name in ['final.ply', 'ckpt_final.npz']:
                self.assertEqual((root / name).read_bytes(), (root / 'ig_audit' / name).read_bytes())

    def test_adapter_fine_export_keeps_all_metadata_and_ordinary_support(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); obj, checked = self.adapter(root, fine=True)
            report = obj.export(2, root)
            self.assertEqual(report['multiresolution']['fine_track_ids'], [42])
            old = np.load(root / 'ig_audit/ckpt_final.npz'); new = np.load(root / 'ckpt_final.npz')
            for key in ['pred_score', 'pred_classes', 'pred_sem_features']:
                np.testing.assert_array_equal(old[key], new[key])
            pc = o3d.io.read_point_cloud(str(root / 'final.ply'))
            np.testing.assert_array_equal(np.asarray(pc.points)[new['pred_masks'][:, 0]], self.xyz[self.masks[:, 0]])

    def test_owner_export_identity_metadata_does_not_change_npz(self):
        old_path = Path('/root/OnlineSegmentationBaseline_multires_corrected7_20261013/submap_disjoint/owner.py')
        new_path = Path(__file__).parent / 'submap_disjoint/owner.py'
        if not old_path.exists():
            self.skipTest('Remote original IG owner fixture required')
        methods = []
        for path in [old_path, new_path]:
            tree = ast.parse(path.read_text())
            nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in ['save_ckpt', 'process_mask_boundary_pts']]
            ns = dict(torch=torch, np=np, os=__import__('os'), filter_instances=filter_instances,
                      export_instance_mask=export_instance_mask, keep_min_rows=lambda x: x)
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
            methods.append(ns)
        rng = np.random.default_rng(0)
        points = np.concatenate([rng.normal(0, .001, (301, 3)), rng.normal(1, .001, (50, 3)), rng.normal(2, .001, (210, 3))])
        masks = [torch.tensor(np.arange(561) < 301), torch.tensor((np.arange(561) >= 301) & (np.arange(561) < 351)), torch.tensor(np.arange(561) >= 351)]
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for i, ns in enumerate(methods):
                obj = types.SimpleNamespace(pc_frame_id=1, points=points, colors=np.zeros_like(points), device='cpu', cfg={})
                obj.get_valid_instances = lambda **kw: (masks, torch.tensor([9, 3, 20]), [], [torch.tensor([float(i)]) for i in range(3)], [])
                obj.save_pc = lambda *args: None
                obj.process_mask_boundary_pts = types.MethodType(ns['process_mask_boundary_pts'], obj)
                name = str(Path(directory) / ('out%d.npz' % i))
                ns['save_ckpt'](obj, 1, directory, ckpt_path=name)
                files.append(np.load(name))
                if i == 1:
                    self.assertEqual(obj.last_exported_instance_ids, [20, 9])
            for key in files[0].files:
                np.testing.assert_array_equal(files[0][key], files[1][key])


if __name__ == '__main__':
    unittest.main(verbosity=2)
