"""Bounded three-frame synthetic plane; no dataset/scene mapping run."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import open3d as o3d
from submap_disjoint.it_adapter_multires import InstanceTSDFAdapter


def save_pc(xyz, rgb, filename):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    cloud.colors = o3d.utility.Vector3dVector(rgb)
    assert o3d.io.write_point_cloud(filename, cloud)


K = np.array([[100., 0., 32.], [0., 100., 32.], [0., 0., 1.]])
cfg = dict(scene=dict(voxel_size=.025, block_resolution=4, block_count=10000),
           cam=dict(depth_near=.1, depth_far=5., depth_scale=1000.),
           multi_resolution=dict(enabled=True, fine_voxel_size_m=.015,
             fine_block_count=10000, geometry_policy='hybrid', fine_min_views=3,
             fine_max_views=5, fine_max_extent_m=.45, fine_replace_radius_m=.05))
owner = SimpleNamespace(device='cuda:0', intrinsics=torch.tensor(K),
                        global2merged_mask_id=[42, 42, 42], save_pc=save_pc)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory); (root/'ig_audit').mkdir()
    grid = np.arange(-.10, .101, .025)
    xx, yy = np.meshgrid(grid, grid)
    target = np.column_stack([xx.ravel(), yy.ravel(), np.ones(xx.size)])
    ordinary = target + [1., 0., 0.]
    xyz = np.concatenate([ordinary, target, [[2., 2., 1.]]])
    rgb = np.ones_like(xyz)*.5
    masks = np.zeros((len(xyz), 2), bool)
    masks[:len(ordinary), 0] = True
    masks[len(ordinary):-1, 1] = True
    save_pc(xyz, rgb, str(root/'ig_audit/final.ply'))
    np.savez(root/'ig_audit/ckpt_final.npz', pred_masks=masks,
             pred_score=np.array([.4, .8]), pred_classes=np.array([0, 0]),
             pred_sem_features=np.zeros((4, 2)))
    (root/'ig_audit/export_instance_ids.json').write_text('[7, 42]')
    adapter = InstanceTSDFAdapter(cfg, owner, root/'it_fragments')
    for frame in range(3):
        depth = torch.ones((64, 64)); color = torch.ones((64, 64, 3))*.5
        labels = torch.zeros((64, 64), dtype=torch.int32); labels[21:44, 21:44] = 1
        pose = torch.eye(4); pose[0, 3] = (frame-1)*.005
        adapter.observe(frame, color, depth, pose, labels,
                        [dict(mask_id=1, global_mask_id=frame)])
    report = adapter.export(2, root)
    assert report['multiresolution']['fine_track_ids'] == [42], report
    assert not report['audit']['fallback_to_ig']
    npz = np.load(root/'ckpt_final.npz')
    cloud = o3d.io.read_point_cloud(str(root/'final.ply'))
    np.testing.assert_array_equal(np.asarray(cloud.points)[npz['pred_masks'][:, 0]], ordinary)
    np.testing.assert_array_equal(npz['pred_score'], [.4, .8])
    print(json.dumps(dict(status='passed', synthetic_frames=3, replacements=report['replacements'],
                          audit=report['audit']), indent=2))
