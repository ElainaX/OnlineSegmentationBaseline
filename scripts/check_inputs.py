"""Validate all prepared RGB-D/pose files and all 191 predicted mask frames."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from common import check_layout
from prepare_masks import validate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--frames',required=True,type=Path);p.add_argument('--masks',required=True,type=Path);a=p.parse_args()
    report=check_layout(a.frames,a.masks)
    K=np.loadtxt(a.frames/'intrinsic/intrinsic_depth.txt')
    if K.shape!=(4,4) or not np.array_equal(K[:3,:3],[[320,0,320],[0,320,240],[0,0,1]]):raise ValueError('Unexpected Flat intrinsics')
    for i in range(762):
        color=cv2.imread(str(a.frames/'color'/f'{i}.jpg'));depth=cv2.imread(str(a.frames/'depth'/f'{i}.png'),-1);pose=np.loadtxt(a.frames/'pose'/f'{i}.txt')
        if color is None or color.shape!=(480,640,3):raise ValueError(f'Invalid RGB {i}')
        if depth is None or depth.shape!=(480,640) or depth.dtype!=np.uint16:raise ValueError(f'Depth {i} must be 640x480 uint16 millimetres')
        if pose.shape!=(4,4) or not np.isfinite(pose).all() or not np.allclose(pose[3],[0,0,0,1]):raise ValueError(f'Invalid pose {i}')
        if not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-4):raise ValueError(f'Non-rigid pose {i}')
    records=validate(a.masks);report.update(mask_frames=len(records),all_files_decoded=True,ground_truth_used=False)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
