"""Offline-only per-GT IoU and erased-lattice proximity diagnostics."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/output/ig_deletion_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
rows=json.loads((ROOT/'selected_results.json').read_text())
refs={r['seq']:r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text())}
instances=[]; deletions=[]

def folder(r):
    cmd=r['command'];return Path(cmd[cmd.index('--output_dir')+1])/r['seq']

for seq in sorted({r['seq'] for r in rows}):
    pair={r['variant']:r for r in rows if r['seq']==seq and r['status']=='complete'}
    if not all(k in pair for k in ['off','on']):continue
    labels=np.loadtxt(Path(refs[seq]['gtseg'])/(seq+'.txt'),dtype=np.int64)
    ids,counts=np.unique(labels,return_counts=True)
    ids=ids[(ids>=1000)&(counts>=100)]
    masks={k:np.load(folder(r)/'pred_instance_info.npz')['pred_instance_masks'] for k,r in pair.items()}
    assert all(p.shape[1]==len(labels) for p in masks.values()),'Expected evaluator masks on GT vertices'
    for gid in ids:
        g=labels==gid;row=dict(seq=seq,gt_id=int(gid%1000),gt_vertices=int(g.sum()))
        for mode,p in masks.items():
            intersection=p[:,g].sum(1);size=p.sum(1)
            iou=intersection/np.maximum(1,size+g.sum()-intersection)
            best=int(iou.argmax()) if len(p) else -1
            row[mode]=dict(prediction_index=best,iou=float(iou[best]) if best>=0 else 0.,
                recall=float(intersection[best]/g.sum()) if best>=0 else 0.,
                purity=float(intersection[best]/max(1,size[best])) if best>=0 else 0.)
        row['delta_best_iou']=row['on']['iou']-row['off']['iou'];instances.append(row)
    clouds=[np.load(p)['xyz'] for p in sorted((folder(pair['on'])/'deletion/archive').glob('*.npz'))]
    if not clouds:
        deletions.append(dict(seq=seq,unique_erased_corners=0));continue
    xyz=np.unique(np.concatenate(clouds),axis=0)
    gt=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    gtlabels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(gt)==len(gtlabels)
    changes=json.loads((DATA/seq/'session_boundary.json').read_text())['changed_instances']
    changed=set(changes['removed'])|set(changes['nonrigid'])|{r['instance_reference'] for r in changes['rigid']}
    distances,nearest=cKDTree(gt).query(xyz)
    near=distances<=.05;known=near&(gtlabels[nearest]>0)
    obj=gtlabels[nearest]%1000;change=known&np.isin(obj,list(changed))
    deletions.append(dict(seq=seq,unique_erased_corners=len(xyz),within_5cm=int(near.sum()),
        near_change_annotated=int(change.sum()),near_not_change_annotated=int((known&~change).sum()),
        near_unlabelled=int((near&~known).sum()),far=int((~near).sum()),
        object_counts=dict(Counter(obj[known].tolist())),
        note='TSDF lattice corners, not surface samples or detection precision; no-change annotation is not proof of physical stasis.'))
(ROOT/'per_instance_iou.json').write_text(json.dumps(instances,indent=2))
(ROOT/'erased_voxel_gt_audit.json').write_text(json.dumps(deletions,indent=2))
print(json.dumps(dict(deletions=deletions,biggest_iou_changes=sorted(instances,key=lambda r:abs(r['delta_best_iou']),reverse=True)[:12]),indent=2))
