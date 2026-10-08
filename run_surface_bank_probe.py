"""Raw-sensor candidate-entry ablation; no instance/map changes and no GT input."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from submap_disjoint.surface_bank import SurfaceBank
from submap_disjoint.readonly_detection_core import interior_mask

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',required=True)
OUT=Path(parser.parse_args().output_dir);OUT.mkdir(exist_ok=False)
source=Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
schedules={r['seq']:r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text())}
scenes=[]
for run in json.loads((source/'selected_results.json').read_text()):
    seq=run['seq'];cmd=run['command'];target=OUT/seq;target.mkdir()
    data=Path(cmd[cmd.index('--dataset')+1]);prior=Path(cmd[cmd.index('--output_dir')+1])/seq
    cfg_file=prior/'remove_detection/config.json'
    config=json.loads(cfg_file.read_text());cfg=config['effective_settings']
    cam=config['pipeline_config']['cam'];assert cam['h_crop']==cam['w_crop']==0
    k=np.loadtxt(data/'intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32)
    bank=SurfaceBank(config['pipeline_config']['scene']['voxel_size'],cfg)
    selected=schedules[seq]['sampling']['selected_frame_ids'];frames=[];start=time.perf_counter()
    for j,f in enumerate(selected):
        raw=cv2.imread(str(data/'depth'/f'{f}.png'),cv2.IMREAD_UNCHANGED)
        if raw is None:raise FileNotFoundError(data/'depth'/f'{f}.png')
        depth=raw.astype(np.float32)/cam['depth_scale']
        assert depth.shape==(cam['img_h'],cam['img_w'])
        valid=np.isfinite(depth)&(depth>cam['depth_near'])&(depth<cam['depth_far'])
        depth[~valid]=0
        inner=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
        pose=np.loadtxt(data/'pose'/f'{f}.txt').astype(np.float32)
        t=time.perf_counter();row=bank.step(f,pose,k,depth,inner,cam['depth_near'],cam['depth_far'])
        row['seconds']=time.perf_counter()-t;frames.append(row)
        if j%100==0:print(seq,j,len(selected),'points',len(bank.points),flush=True)
    arrays=bank.arrays();np.savez_compressed(target/'bank.npz',**arrays)
    report=dict(seq=seq,frames=frames,selected_frame_ids=selected,voxel_size=bank.voxel_size,
        cfg=cfg,seed_stride=bank.seed_stride,wall_seconds=time.perf_counter()-start,
        array_memory_bytes=sum(a.nbytes for a in arrays.values()),GT_used=False,actual_deletions=0,
        scope='experimental raw-depth local change proposals; not whole-instance deletion',
        source_config=str(cfg_file))
    (target/'summary.json').write_text(json.dumps(report,indent=2))
    scenes.append({k:report[k] for k in ['seq','wall_seconds','array_memory_bytes','voxel_size']})
    print(seq,'done',report['wall_seconds'],'active',int(bank.active.sum()),flush=True)
(OUT/'completed.json').write_text(json.dumps(dict(completed=True,scenes=scenes,GT_used=False),indent=2))
