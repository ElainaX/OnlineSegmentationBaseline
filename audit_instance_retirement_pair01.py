"""Offline-only GT audit for retired predicted tracks; never feeds execution."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/output/ig_whole_instance_20261008')
OLD=Path('/root/autodl-tmp/output/ig_surface_deletion_20261008')
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
SEQ='pair_01_0cac767c_19f1a892'
out=ROOT/'hybrid'/SEQ
result=json.loads((ROOT/'result.json').read_text())
if result['status']!='complete': raise RuntimeError('Experiment not complete')
summary=json.loads((out/'instance_retirement/summary.json').read_text())
boundary=json.loads((DATA/SEQ/'session_boundary.json').read_text())
changes=boundary['changed_instances'];removed=set(map(int,changes['removed']))
changed=removed|set(map(int,changes['nonrigid']))|{int(x['instance_reference']) for x in changes['rigid']}
ref=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/SEQ/(SEQ+'_vh_clean_2.ply'))).points)
labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(SEQ+'.txt'),dtype=np.int64)
tree=cKDTree(ref)
events=[]
for event in summary['events']:
    if event['event']!='retired': continue
    candidate=np.load(out/'instance_retirement'/f'candidate_{event["candidate"]:05d}.npz')
    pts=candidate['points'];dist,near=tree.query(pts)
    ids=np.where((dist<=.05)&(labels[near]>0),labels[near]%1000,-1)
    counts=Counter(map(int,ids));majority=counts.most_common(1)[0] if counts else (-1,0)
    events.append(dict(**event,GT_reference_counts=dict(counts),
                       majority_GT=majority[0],majority_fraction=majority[1]/max(len(pts),1),
                       near_removed_fraction=float(np.isin(ids,list(removed)).mean()),
                       near_changed_fraction=float(np.isin(ids,list(changed)).mean()),
                       near_not_change_annotated_fraction=float(((ids>=0)&~np.isin(ids,list(changed))).mean())))
best={}
stats_path=out/'instance_retirement/candidate_stats.jsonl'
if stats_path.exists():
    for line in stats_path.read_text().splitlines():
        record=json.loads(line)
        for c in record['candidates']:
            cid=c['candidate']
            if cid not in best or c['confirmed_fraction']>best[cid]['confirmed_fraction']:
                best[cid]=dict(frame=record['frame'],**c)
old=json.loads((OLD/'on'/SEQ/'run_metadata.json').read_text())
new=json.loads((out/'run_metadata.json').read_text())
candidate_GT=[]
for path in sorted((out/'instance_retirement').glob('candidate_*.npz')):
    cid=int(path.stem.split('_')[-1]);candidate=np.load(path);pts=candidate['points']
    dist,near=tree.query(pts);ids=np.where((dist<=.05)&(labels[near]>0),labels[near]%1000,-1)
    counts=Counter(map(int,ids));majority=counts.most_common(1)[0] if counts else (-1,0)
    candidate_GT.append(dict(candidate=cid,created=int(candidate['created']),
        source_raw_ids=candidate['source_raw_ids'].tolist(),GT_counts=dict(counts),majority_GT=majority[0],
        majority_fraction=majority[1]/max(len(pts),1),majority_is_removed=majority[0] in removed,
        best_evidence=best.get(cid)))
report=dict(GT_used_only_offline=True,retired=events,
            candidate_GT_audit=candidate_GT,
            candidates_created=summary['candidates_created'],reason_counts=summary['reason_counts'],
            best_candidate_evidence=sorted(best.values(),key=lambda c:c['confirmed_fraction'],reverse=True),
            before_rescan_retirements=sum(e['frame']<boundary['rescan_start_frame'] for e in events),
            local_only_main_loop_seconds=old['main_loop_seconds'],hybrid_main_loop_seconds=new['main_loop_seconds'],
            hybrid_whole_retirement_seconds=summary['seconds'],hybrid_detector_seconds=new['surface_detection_seconds'],
            hybrid_local_deletion_seconds=new['deletion_summary']['seconds'],
            time_comparison_caveat='Shared GPU contention and CPU thread count differ from historical local run; not a controlled speedup/slowdown estimate',
            original_AP=result['original_AP'],FullPred_AP={k:result['FullPred_AP'][k] for k in ['AP','AP50','AP25','predictions','GT_instances']},
            geometry=result['geometry'],
            interpretation='GT point association of reference samples does not prove every erased voxel was correct. Unannotated change is not certified unchanged.')
(ROOT/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['candidates_created','before_rescan_retirements','original_AP','FullPred_AP',
    'hybrid_main_loop_seconds','hybrid_whole_retirement_seconds']},indent=2))
print('retired',len(events),'best candidates',report['best_candidate_evidence'][:5])
