"""CPU-only checkpoint loader and evaluator; no ICP or GT access during mapping."""
import json,sys,time
from pathlib import Path
import numpy as np
from flat_scoring import score_reconstruction
from flat_change_scoring import evaluate_actions,digest

def load_prediction(root):
    import open3d as o3d
    root=Path(root)
    cloud=np.asarray(o3d.io.read_point_cloud(str(root/'final.ply')).points)
    assert cloud.ndim==2 and cloud.shape[1]==3 and np.isfinite(cloud).all()
    with np.load(root/'ckpt_final.npz',allow_pickle=False) as z:
        masks=z['pred_masks'];scores=z['pred_score']
        assert masks.shape==(len(cloud),len(scores)) and scores.ndim==1
        assert np.isfinite(scores).all()
        assert masks.dtype==bool or np.isin(masks,[0,1]).all(),'Non-binary checkpoint mask'
        masks=np.asarray(masks,bool)
    return cloud,masks,scores

def evaluate(root,method):
    start=time.time();root=Path(root).resolve();e=Path(__file__).resolve().parent
    assert method in ['baseline','H']
    assert not (root/'flat_metrics.json').exists(),'Preserve prior evaluation'
    cloud,masks,scores=load_prediction(root)
    with np.load(e/'derived_gt_v1/run2_domain.npz',allow_pickle=False) as z:
        ap=score_reconstruction(z,cloud,masks,scores)
    with np.load(e/'derived_change_gt_v1/reference_domain.npz',allow_pickle=False) as z:
        deletion=evaluate_actions(root,z,cloud,method)
    D=deletion['D_observed']
    for group in ['primary','secondary_all_sizes']:
        A=ap[group]['AP'];ap[group]['GAP_observed']=A+(1-A)*D if A is not None and D is not None else None
    result=dict(method=method,AP=ap,deletion=deletion,not_official_3D_benchmark=True,
        ICP_used=False,vertices=len(cloud),instances=len(scores),seconds=time.time()-start,
        source_sha256={str(p):digest(p) for p in [root/'final.ply',root/'ckpt_final.npz',
           e/'derived_gt_v1/run2_domain.npz',e/'derived_change_gt_v1/reference_domain.npz']},
        metric_scope='AP on fixed reliably labelled run2 surface; D/GAP_observed on conservative independently frozen old-surface target. Unknown and off-domain deletions unscored and disclosed. Not directly comparable to 3RScan GAP-v3.')
    (root/'flat_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(method=method,AP=ap['primary'],deletion_status=deletion['status'],D_observed=D)),flush=True)
    return result
if __name__=='__main__':evaluate(sys.argv[1],sys.argv[2])
