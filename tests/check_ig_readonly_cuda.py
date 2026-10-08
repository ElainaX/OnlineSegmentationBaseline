"""Real CUDA TSDF + synthetic disappearance; check unchanged owner/map state."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import open3d as o3d
import open3d.core as c
import torch
import yaml

from submap_disjoint.ig_readonly import IGReadOnlyDetector

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
device = c.Device('CUDA:0')
voxel, resolution = .025, 4
grid = o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf','weight','color'),
    attr_dtypes=(c.float32,c.float32,c.float32), attr_channels=((1,),(1,),(3,)),
    voxel_size=voxel, block_resolution=resolution, block_count=10000, device=device)
k = np.array([[150.,0,111.5],[0,150.,85.5],[0,0,1.]])
kt, ext = c.Tensor(k), c.Tensor(np.eye(4))
rgb = np.full((172,224,3), .4, np.float32)
image = o3d.t.geometry.Image(np.full((172,224,3),102,np.uint8)).to(device)
# Avoid an exact grid-node zero crossing, which Open3D point extraction omits.
depth = o3d.t.geometry.Image(np.full((172,224),1013,np.uint16)).to(device)
blocks = grid.compute_unique_block_coordinates(depth,kt,ext,1000.,5.)
for _ in range(5): grid.integrate(blocks,depth,image,kt,ext,1000.,5.)
coords, flat = grid.voxel_coordinates_and_flattened_indices()
xyz = coords.cpu().numpy(); ids = flat.cpu().numpy().reshape(-1)
selected = ids[(np.abs(xyz[:,0]) < .23) & (np.abs(xyz[:,1]) < .23)]
capacity = int(grid.attribute('weight').num_elements())
memberships = [None] * capacity
for idx in selected: memberships[int(idx)] = {0}
flags = torch.zeros(capacity,dtype=torch.bool,device='cuda:0')
flags[torch.as_tensor(selected.astype(np.int64),device='cuda:0')] = True
cfg = yaml.safe_load(Path('config/ig_readonly_onlineSeg.yaml').read_text())
cfg.update(cam=dict(depth_near=.1,depth_far=5.), scene=dict(voxel_size=voxel,block_resolution=resolution))
owner = SimpleNamespace(cfg=cfg, voxel_block_grids=grid, device_o3c=device,
    intrinsics=torch.tensor(k), trunc_voxel_multiplier=4.,
    voxel_hash_table=SimpleNamespace(hash_table=memberships,hash_item_flag=flags),
    global2merged_mask_id=[0], get_merged_mask_weight=torch.tensor([5.]),
    maskGraph=SimpleNamespace(instance_dict={}))
detector = IGReadOnlyDetector(owner,cfg,a.output,verify=True)
current = np.full((172,224),1.013,np.float32)
current[44:128,70:154] = 2.
detector.step(0,rgb,current,np.eye(4))
assert detector.detector.candidates, 'Synthetic scene did not create a disappearance candidate'
# A graph row renumbering is a legitimate external mutation between checks.
# Frozen references and source-mask lineage must survive it.
owner.global2merged_mask_id = [1]
owner.get_merged_mask_weight = torch.tensor([0.,5.])
pose = np.eye(4);pose[0,3] = .03
detector.step(5,rgb,current,pose)
assert any(x.state=='would_remove' for x in detector.detector.candidates)
pc = grid.extract_point_cloud(weight_threshold=3.)
detector.finalize(pc.point.positions.cpu().numpy(),pc.point.colors.cpu().numpy())
report=detector.report()
assert report['mutation_check_passed'] and report['mutation_checks']==3
assert report['summary']['actual_tsdf_deletions']==0
Path(a.output,'cuda_test.json').write_text(json.dumps(report,indent=2))
print('CUDA_READONLY_PASS',json.dumps({k:report[k] for k in ['mutation_checks','mutation_check_passed']}))
