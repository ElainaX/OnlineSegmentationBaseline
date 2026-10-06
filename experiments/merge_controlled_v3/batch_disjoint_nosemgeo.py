import json, os, subprocess, time
from pathlib import Path

ROOT = Path('/root/autodl-tmp/output/merge_disjoint_nosemgeo_stride10')
REPO = Path('/root/OnlineSegmentationBaseline_baseline_ablation')
PY = '/root/miniconda3/envs/OASeg/bin/python'
HERE = Path(__file__).parent
SCENES = ['scene0050_00', 'scene0089_00', 'scene0549_00']
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)

def paths(scene):
    if scene == 'scene0089_00':
        return (f'/root/autodl-tmp/additional_stride10_v2/{scene}/frames',
                f'/root/autodl-tmp/additional_seg_stride10_v2/{scene}')
    return (f'/root/autodl-tmp/demo_stride10/{scene}/frames',
            f'/root/autodl-tmp/seg_result_stride10/{scene}')

for scene in SCENES:
    data, seg = paths(scene)
    manifest = ROOT/scene/'controlled_manifest.json'
    if not (manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete'):
        print('BEGIN', scene, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
        log = ROOT/f'{scene}.run.log'
        with open(log, 'w') as f:
            subprocess.run([PY, '-u', str(HERE/'run_controlled.py'), '--repo', str(REPO),
                            '--mode', 'disjoint_no_sem_geo', '--',
                            '--config', 'config/scannet_stride10_controlled.yaml', '--seq_name', scene,
                            '-d', data, '-i', seg, '-o', str(ROOT)],
                           stdout=f, stderr=subprocess.STDOUT, check=True, timeout=3600)
    report = ROOT/f'{scene}_eval_class_agnostic.txt'
    if not report.exists():
        log = ROOT/f'{scene}.eval.log'
        with open(log, 'w') as f:
            subprocess.run([PY, str(HERE/'evaluate_controlled.py'), '--result_dir', str(ROOT),
                            '--seq_name', scene, '--gt_dir', '/root/autodl-tmp/ScanNet_subset_onlineanyseg/groundtruth',
                            '--gt_pc_pattern', '%s/%s_vh_clean_2.ply', '--gt_seg_dir', '/root/autodl-tmp/ScanNet_subset_onlineanyseg/gt_seg',
                            '--gt_seg_pattern', '%s.txt', '--output_file', str(report)],
                           stdout=f, stderr=subprocess.STDOUT, check=True, timeout=3600)
    print('DONE', scene, report.read_text().splitlines()[-1], flush=True)
print('ALL DISJOINT NO SEM GEO COMPLETE', flush=True)
