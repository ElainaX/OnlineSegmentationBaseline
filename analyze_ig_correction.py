"""Post-hoc GT evaluation only. Never imported by the online runner."""
from pathlib import Path
from collections import Counter
import json,csv
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
ROOT=Path('/root/autodl-tmp/output/ig_correction_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
rows=json.loads((ROOT/'results.json').read_text())
instances=[]; diagnostics=[]
def xyz(path):return np.asarray(o3d.io.read_point_cloud(str(path)).points)
def fractions(points,tree,labels,changed):
    if not len(points):return dict(points=0)
    dist,idx=tree.query(points);near=dist<=.10
    obj=labels[idx]%1000;known=near&(labels[idx]>0)
    changed_mask=known&np.isin(obj,list(changed))
    counts=Counter(obj[known].tolist())
    return dict(points=len(points),near_reference_gt=int(near.sum()),changed_annotated=int(changed_mask.sum()),
        known_unchanged=int((known&~changed_mask).sum()),unlabelled_near=int((near&~known).sum()),far_from_reference_gt=int((~near).sum()),
        dominant_object_id=int(counts.most_common(1)[0][0]) if counts else None,object_point_counts=dict(counts))
for seq in sorted({r['seq'] for r in rows}):
    completed={r['variant']:r for r in rows if r['seq']==seq and r['status']=='complete'}
    gt=np.loadtxt(DATA/'gt_seg'/(seq+'.txt'),dtype=np.int64)
    ids,counts=np.unique(gt,return_counts=True)
    ids=ids[(ids>=1000)&(counts>=100)]  # dataset already class-agnostic remapped valid labels
    masks={m:np.load(ROOT/m/seq/'pred_instance_info.npz')['pred_instance_masks'] for m in completed}
    for gid in ids:
        g=gt==gid;item=dict(seq=seq,gt_instance=int(gid),gt_vertices=int(g.sum()))
        for mode,p in masks.items():
            intersection=p[:,g].sum(1);size=p.sum(1);union=size+g.sum()-intersection
            iou=intersection/np.maximum(1,union);best=int(iou.argmax()) if len(p) else -1
            item[mode]=dict(prediction_index=best,iou=float(iou[best]) if best>=0 else 0.,
                recall=float(intersection[best]/g.sum()) if best>=0 else 0.,
                purity=float(intersection[best]/max(1,size[best])) if best>=0 else 0.)
        if 'off' in item and 'delete' in item:item['delta_best_iou']=item['delete']['iou']-item['off']['iou']
        instances.append(item)
    folder=ROOT/'delete'/seq/'correction'
    if not (folder/'summary.json').exists():continue
    summary=json.loads((folder/'summary.json').read_text())
    b=json.loads((DATA/seq/'session_boundary.json').read_text())
    changed=set(b['changed_instances']['removed'])|set(b['changed_instances']['nonrigid'])|{r['instance_reference'] for r in b['changed_instances']['rigid']}
    refxyz=xyz(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(labels)==len(refxyz);tree=cKDTree(refxyz)
    surf=np.load(folder/'reference_surfaces.npz');reasons=Counter();nframes=0;guard=0
    tested=Counter();candidate=Counter();free=Counter()
    for line in (folder/'frames.jsonl').read_text().splitlines():
        frame=json.loads(line);nframes+=1;guard+=bool(frame['skip_reasons'])
        for r in frame['references']:
            reasons.update(r['candidate_failures']);k=str(r['reference_id'])
            tested[k]+=r['tested_reference'];free[k]+=r['depth_reference'];candidate[k]+=r['candidate']
    reference_rows=[]
    for key,r in summary['references'].items():
        reference_rows.append(dict(reference_id=int(key),**r,posthoc_gt=fractions(surf['points_'+key],tree,labels,changed),
            total_tested_cell_observations=tested[key],total_free_cell_observations=free[key]))
    clouds=[np.load(f)['xyz'] for f in sorted((folder/'erasure_archive').glob('*.npz'))]
    deleted=np.unique(np.concatenate(clouds),axis=0) if clouds else np.zeros((0,3))
    diagnostics.append(dict(seq=seq,audited_frames=nframes,guarded_frames=guard,reference_count=len(reference_rows),
        registered_skips=summary['skips'],candidate_rejection_counts=dict(reasons),
        retirement_count=sum(r['state']=='retired' for r in reference_rows),candidate_count=sum(r['candidates']>0 for r in reference_rows),
        audit_seconds=summary['audit_seconds'],freeze_seconds=summary['freeze_seconds'],references=reference_rows,
        erased_voxel_gt=fractions(deleted,tree,labels,changed),
        caution='Nearest reference GT within 10 cm; unchanged means not change-annotated, not guaranteed physically static. Unlabelled surfaces are not counted as true negatives.'))
(ROOT/'per_instance_comparison.json').write_text(json.dumps(instances,indent=2))
(ROOT/'deletion_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
with (ROOT/'per_instance_comparison.csv').open('w',newline='') as f:
    fields=['scene','gt_instance','gt_vertices','off_iou','delete_iou','delta_iou','off_purity','delete_purity','off_recall','delete_recall']
    w=csv.writer(f);w.writerow(fields)
    for r in instances:
        a=r.get('off',{});b=r.get('delete',{})
        w.writerow([r['seq'],r['gt_instance'],r['gt_vertices'],a.get('iou'),b.get('iou'),r.get('delta_best_iou'),a.get('purity'),b.get('purity'),a.get('recall'),b.get('recall')])
print(json.dumps([{k:r[k] for k in ['seq','audited_frames','reference_count','candidate_count','retirement_count','erased_voxel_gt']} for r in diagnostics],indent=2))
