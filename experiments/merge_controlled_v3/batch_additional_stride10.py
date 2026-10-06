import json, os, subprocess, time
from pathlib import Path

ROOT = Path('/root/autodl-tmp/output/merge_additional_stride10_v2')
REPO = Path('/root/OnlineSegmentationBaseline_baseline_ablation')
PY = '/root/miniconda3/envs/OASeg/bin/python'
HERE = Path(__file__).parent
MODES = ['baseline', 'disjoint_pairs', 'quality_prototype', 'disjoint_quality']
SCENES = ['scene0089_00', 'scene0243_00']
ROOT.mkdir(parents=True, exist_ok=True)
os.chdir(REPO)

def run(cmd, log):
    with open(log, 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=3600)

for mode in MODES:
    for scene in SCENES:
        manifest = ROOT/mode/scene/'controlled_manifest.json'
        finished = manifest.exists() and json.loads(manifest.read_text())['status'] == 'complete'
        if not finished:
            print('BEGIN', mode, scene, time.strftime('%Y-%m-%d %H:%M:%S'), flush=True)
            run([PY, '-u', str(HERE/'run_controlled.py'), '--repo', str(REPO), '--mode', mode, '--',
                 '--config', 'config/scannet_stride10_controlled.yaml', '--seq_name', scene,
                 '-d', f'/root/autodl-tmp/additional_stride10_v2/{scene}/frames',
                 '-i', f'/root/autodl-tmp/additional_seg_stride10_v2/{scene}', '-o', str(ROOT/mode)],
                ROOT/f'{mode}_{scene}.run.log')
        report = ROOT/f'{mode}_{scene}_eval_class_agnostic.txt'
        if not report.exists():
            run([PY, str(HERE/'evaluate_controlled.py'), '--result_dir', str(ROOT/mode),
                 '--seq_name', scene, '--gt_dir', '/root/autodl-tmp/ScanNet_subset_onlineanyseg/groundtruth',
                 '--gt_pc_pattern', '%s/%s_vh_clean_2.ply', '--gt_seg_dir', '/root/autodl-tmp/ScanNet_subset_onlineanyseg/gt_seg',
                 '--gt_seg_pattern', '%s.txt', '--output_file', str(report)], ROOT/f'{mode}_{scene}.eval.log')
        print('DONE', mode, scene, report.read_text().splitlines()[-1], flush=True)
print('ALL ADDITIONAL STRIDE10 COMPLETE', flush=True)
