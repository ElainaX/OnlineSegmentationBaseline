"""Offline only: map reference evidence to all raw GT object IDs.

This audit never runs inside the detector and never modifies decisions.
Report instance-stage hits and sample coverage separately, including unchanged
objects. A stage hit needs three distinct 25mm cells, an evaluation convention.
"""
import json
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.spatial import cKDTree
from eval_gap_v2 import cloud,raw_reference
from audit_ig_recall import audit

DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_Additional_20261008')
RAW=Path('/root/autodl-tmp/3RScan/additional_raw_20261008')
OUT=Path('/root/autodl-tmp/output/ig_visible_v3_20261010')
SEQS=['pair_09_0cac7676_7272e17c','pair_10_10b17940_5630cfe7']
STAGES=('reference','visible','free','candidate','confirmed','ratio_pass','executed')

def one(seq,mode):
 root=OUT/mode/seq/'correction';audit(root)
 summary=json.loads((root/'summary.json').read_text())
 boundary=json.loads((DATA/seq/'session_boundary.json').read_text())
 ref=cloud(DATA/'gt_reference'/(seq+'_vh_clean_2.ply'))
 labels,nyu=raw_reference(RAW/boundary['reference_scan_id']/'labels.instances.annotated.v2.ply',ref)
 tree=cKDTree(ref);sample_labels={};sample_keys={}
 groups=defaultdict(lambda:{stage:set() for stage in STAGES})
 def add(mid,indices,stage):
  for i in indices:
   g=int(sample_labels[mid][i])
   if g>0:groups[g][stage].add(tuple(sample_keys[mid][i]))
 with np.load(root/'reference_surfaces.npz') as saved:
  for mid in summary['references']:
   p=saved['points_'+mid];d,j=tree.query(p)
   sample_labels[mid]=np.where(d<=.05,labels[j],0)
   sample_keys[mid]=np.floor(p/.025).astype(np.int64)
   add(mid,range(len(p)),'reference')
 stats=Counter();ref_stages={name:set() for name in ['projected','visible','candidate','confirmed','action_ready']}
 for line in (root/'frames.jsonl').open():
  f=json.loads(line)
  if f['frame'] < boundary['rescan_start_frame']:continue
  for r in f['references']:
   mid=str(r['reference_id'])
   for name,field in [('visible','visible_point_indices'),('free','free_point_indices'),('confirmed','confirmed_point_indices')]:
    add(mid,r.get(field,[]),name)
   if r['candidate']:add(mid,r['free_point_indices'],'candidate')
   if r['confirmed_region']:add(mid,r['confirmed_point_indices'],'ratio_pass')
   for name,passed in [('projected',r['projected_cells']>0),('visible',r['visible_cells']>=summary['options']['min_visible']),
       ('candidate',r['candidate']),('confirmed',r['confirmed_region']),('action_ready',r['eligible_erasure_corners']>0)]:
    if passed:ref_stages[name].add(mid)
   stats.update(r.get('failure_reasons',[]))
 # Actual action archives are independent of candidate/confirmation counters.
 for archive in sorted((root/'erasure_archive').glob('*.npz')):
  if int(archive.stem)<boundary['rescan_start_frame']:continue
  xyz=np.load(archive)['xyz'];dist,idx=tree.query(xyz)
  for p,g in zip(xyz[dist<=.05],labels[idx[dist<=.05]]):
   if g>0:groups[int(g)]['executed'].add(tuple(np.floor(p/.025).astype(np.int64)))
 ch=boundary['changed_instances'];removed=set(map(int,ch['removed']));moving=set(map(int,ch['nonrigid']))|{int(x['instance_reference']) for x in ch['rigid']}
 rows=[]
 for g in sorted(set(map(int,np.unique(labels)))-{0}):
  cat='removed' if g in removed else ('moving' if g in moving else 'unchanged_or_unannotated_change')
  counts={stage:len(groups[g][stage]) for stage in STAGES}
  rows.append(dict(GT_id=g,category=cat,NYU40=np.unique(nyu[labels==g]).tolist(),cells=counts,
      hits={stage:counts[stage]>=3 for stage in STAGES},
      miss_reason=('no_reference_surface' if counts['reference']<3 else 'not_observed' if counts['visible']<3 else
       'no_depth_free' if counts['free']<3 else 'candidate_gate' if counts['candidate']<3 else
       'no_repeat_confirmation' if counts['confirmed']<3 else 'instance_ratio_gate' if counts['ratio_pass']<3 else
       'no_archived_action_near_GT' if counts['executed']<3 and mode=='delete' else None)))
 grouped={cat:{stage:sum(r['hits'][stage] for r in rows if r['category']==cat) for stage in STAGES}
    for cat in ['removed','moving','unchanged_or_unannotated_change']}
 result=dict(sequence=seq,mode=mode,evaluation=dict(voxel_m=.025,GT_tolerance_m=.05,min_GT_stage_cells=3,
    note='Post-rescan spatial stage-hit diagnostics, not whole-object deletion accuracy. Executed counts archive corners within 5cm of raw GT; these are not surface coverage or final removal. Confirmed cells can exist below the instance candidate ratio.'),
   references_registered=len(summary['references']),references_skipped=len(summary['skips']),
   unique_reference_funnel={k:len(v) for k,v in ref_stages.items()},
   failure_reasons=dict(stats),GT_funnel=grouped,per_GT=rows,
   erased_voxels=summary['unique_deleted_world_voxels'])
 (root/'gt_recall_audit.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 all_results={s:{m:one(s,m) for m in ['detect','delete']} for s in SEQS}
 (OUT/'gt_recall_audit.json').write_text(json.dumps(all_results,indent=2))
