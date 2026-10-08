"""CPU-only paired post-hoc diagnostic; fixed shared alignment and GT scope."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import open3d as o3d

ROOT=Path('/root/autodl-tmp/output/ig_surface_deletion_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
SEQ='pair_01_0cac767c_19f1a892'
OUT=ROOT/'full_prediction_ap';OUT.mkdir(exist_ok=True)
gt=DATA/SEQ/(SEQ+'_vh_clean_2.ply')
transform_file=OUT/'shared_GT_to_prediction.json'
if not transform_file.exists():
    target=ROOT/'off'/SEQ/'final.ply'
    registration=o3d.pipelines.registration.registration_icp(
        o3d.io.read_point_cloud(str(gt)),o3d.io.read_point_cloud(str(target)),.05,np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transform_file.write_text(json.dumps(dict(T_prediction_GT=registration.transformation.tolist(),
         source='one ICP from rescan GT to deletion-OFF cloud; frozen for both methods',
         reference_prediction=str(target),threshold_m=.05,fitness=registration.fitness,
         inlier_rmse=registration.inlier_rmse),indent=2))
tree=ast.parse(Path('/root/OnlineAnySegBaseline_original/eval/constants.py').read_text())
classes=None
for node in tree.body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SCANNET_IDS' for t in node.targets):
        classes=ast.literal_eval(node.value)
if classes is None: raise RuntimeError('Cannot read existing evaluator class list')
report={}
for mode in ['off','on']:
    output=OUT/(mode+'.json')
    cmd=[sys.executable,str(Path(__file__).with_name('eval_full_prediction_ap.py')),
         '--prediction-dir',str(ROOT/mode/SEQ),'--gt-cloud',str(gt),
         '--gt-labels',str(DATA/'gt_seg'/(SEQ+'.txt')),
         '--gt-to-prediction-transform',str(transform_file),
         '--valid-class-ids',','.join(map(str,classes)),
         '--min-gt-points','100','--voxel-size','.025','--projection-radius','.15',
         '--with-experimental-voxel-ap','--output',str(output)]
    started=time.perf_counter()
    with (OUT/(mode+'.log')).open('w') as f:
        subprocess.run(cmd,check=True,stdout=f,stderr=subprocess.STDOUT,
                       env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    result=json.loads(output.read_text())
    report[mode]=dict(seconds=time.perf_counter()-started,command=cmd,
         **{k:result[k] for k in ['FullPred_AP','VoxelUnion_AP','geometry']})
    print(mode,{k:{metric:result[k][metric] for metric in ['AP','AP50','AP25','predictions','GT_instances']} for k in ['FullPred_AP','VoxelUnion_AP']},flush=True)
(OUT/'comparison.json').write_text(json.dumps(report,indent=2))
