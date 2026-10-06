"""Create a lightweight stride-N ScanNet input by symlinking selected frames."""
from pathlib import Path
import argparse, os, shutil
import numpy as np
import cv2

def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)

def prepare(scene: str, demo_root: Path, seg_root: Path, out_demo: Path,
            out_seg: Path, stride: int):
    src_scene = demo_root / scene
    src_frames = src_scene / 'frames'
    src_color = src_scene / 'color_640'
    if not src_color.exists():
        src_color = src_scene / 'color'
    src_depth = src_scene / 'depth'
    src_pose = src_scene / 'pose'
    src_seg = seg_root / scene
    color_paths = list(src_color.glob('*.jpg')) or list(src_color.glob('*.png'))
    color_ext = color_paths[0].suffix
    color_ids = sorted(int(p.stem) for p in color_paths)
    selected = color_ids[::stride]
    if selected[-1] != color_ids[-1] and color_ids[-1] % stride == 0:
        selected.append(color_ids[-1])
    # Drop frames that cannot be integrated safely.  Some ScanNet sequences
    # contain non-finite poses or all-zero depth spans; retaining them would
    # make the TSDF integrator fail rather than produce a meaningful ablation.
    valid = []
    for original in selected:
        pose_path = src_pose / f'{original}.txt'
        depth_path = src_depth / f'{original}.png'
        if not pose_path.exists() or not depth_path.exists():
            continue
        try:
            pose = np.loadtxt(pose_path)
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if not np.isfinite(pose).all() or depth is None or not np.any(depth > 0):
                continue
        except Exception:
            continue
        valid.append(original)
    selected = valid
    dst_scene = out_demo / scene
    dst_frames = dst_scene / 'frames'
    for sub in ('color', 'depth', 'pose'):
        (dst_frames / sub).mkdir(parents=True, exist_ok=True)
    intrinsic = src_scene / 'intrinsic_640.txt'
    if not intrinsic.exists():
        intrinsic = src_scene / 'intrinsic' / 'intrinsic_depth.txt'
    link(intrinsic, dst_frames / 'intrinsic' / 'intrinsic_depth.txt')
    # Keep the sequence name and GT mesh lookup unchanged.
    if (src_scene / f'{scene}_vh_clean_2.ply').exists():
        link(src_scene / f'{scene}_vh_clean_2.ply', dst_scene / f'{scene}_vh_clean_2.ply')
    for local, original in enumerate(selected):
        link(src_color / f'{original}{color_ext}', dst_frames / 'color' / f'{local}{color_ext}')
        depth_ext = '.png' if (src_depth / f'{original}.png').exists() else '.jpg'
        link(src_depth / f'{original}{depth_ext}', dst_frames / 'depth' / f'{local}{depth_ext}')
        link(src_pose / f'{original}.txt', dst_frames / 'pose' / f'{local}.txt')
    dst_seg = out_seg / scene
    for sub in ('mask', 'mask_embeddings'):
        (dst_seg / sub).mkdir(parents=True, exist_ok=True)
    for local, original in enumerate(selected):
        # Segmentation is available at the original frame id. If a frame is
        # missing, leave it as an unsegmented frame; the dataset supports this.
        for sub, ext in (('mask', '.png'), ('mask_embeddings', '.pt')):
            src = src_seg / sub / f'{original}{ext}'
            if src.exists():
                link(src, dst_seg / sub / f'{local}{ext}')
    print(scene, 'selected', len(selected), 'original range', selected[0], selected[-1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--demo-root', type=Path, required=True)
    ap.add_argument('--seg-root', type=Path, required=True)
    ap.add_argument('--out-demo', type=Path, required=True)
    ap.add_argument('--out-seg', type=Path, required=True)
    ap.add_argument('--stride', type=int, default=10)
    ap.add_argument('scenes', nargs='+')
    a = ap.parse_args()
    for scene in a.scenes:
        prepare(scene, a.demo_root, a.seg_root, a.out_demo, a.out_seg, a.stride)

if __name__ == '__main__': main()
