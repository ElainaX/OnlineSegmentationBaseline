"""Portable, audited entry point for the current single-global-TSDF system."""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('config', 'dataset', 'instance_dir', 'output_dir'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--seq_name', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--detector-config', type=Path, default=Path(__file__).resolve().parents[2] / 'feng_detector_config.json')
    p.add_argument('--correction', choices=['off', 'detect', 'delete'], default='delete')
    p.add_argument('--observation-refine', choices=['on', 'off'], default='on')
    p.add_argument('--merge-selection', choices=['disjoint_pairs', 'baseline_components'], default='disjoint_pairs')
    a = p.parse_args()
    a.detector_config = a.detector_config.resolve()
    if not a.detector_config.is_file():
        p.error(f'Missing detector config: {a.detector_config}')
    root = Path(__file__).resolve().parents[2]
    here = Path(__file__).resolve().parent
    paths = {k: str(getattr(a, k).resolve()) for k in ('config', 'dataset', 'instance_dir', 'output_dir')}
    for k in ('config', 'dataset', 'instance_dir'):
        if not Path(paths[k]).exists():
            p.error(f'Missing {k}: {paths[k]}')
    if Path(a.seq_name).name != a.seq_name or a.seq_name in ('.', '..'):
        p.error('seq_name must be a single directory name')
    out = Path(paths['output_dir']) / a.seq_name
    if out.exists() and any(out.iterdir()):
        p.error(f'Refusing nonempty output: {out}')
    sys.path[:0] = [str(root), str(here), str(root / 'third_party/FCGF')]
    os.chdir(root)
    from raycast_bounds_guard import install as raycast_guard
    from fcgf_singleton_guard import install as singleton_guard
    from record_erasure import install as record_erasure
    raycast_guard(out)
    singleton_guard()
    record_erasure()
    os.environ['FENG_VARIANT'] = 'fast'
    sys.argv = ['run_instdel_multires.py']
    for k, v in paths.items():
        sys.argv += ['--' + k, v]
    sys.argv += ['--seq_name', a.seq_name, '--device', a.device,
                 '--correction', a.correction, '--merge_selection', a.merge_selection,
                 '--global-voxel-m', '0.025', '--aabb', '0', '--depth-shrink', '0',
                 '--detector-config', str(a.detector_config)]
    if a.observation_refine == 'on':
        sys.argv += ['--observation-refine']
    runpy.run_path(str(here / 'run_feng_optimized.py'), run_name='__main__')


if __name__ == '__main__':
    main()
