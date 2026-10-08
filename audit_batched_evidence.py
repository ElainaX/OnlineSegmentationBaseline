"""Offline GT audit only, following causal replay without GT."""
import json
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',default='/root/autodl-tmp/output/ig_batched_evidence_20261008')
ROOT=Path(parser.parse_args().output_dir)
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
VARIANTS=['original','local_clear','lower_single_frame_gate','batched']


def associate(points,tree,labels):
    if not len(points):return dict(gt_id=None,purity=0.,evidence_points=0)
    distance,idx=tree.query(points)
    ok=(distance<=.05)&(labels[idx]>0)
    counts=Counter((labels[idx[ok]]%1000).tolist())
    gid,n=counts.most_common(1)[0] if counts else (None,0)
    return dict(gt_id=gid if n>.5*len(points) else None,purity=n/len(points),evidence_points=len(points))


scenes=[]
for source in json.loads((ROOT/'replay.json').read_text())['scenes']:
    seq=source['seq'];summary=json.loads((ROOT/seq/'summary.json').read_text())
    b=json.loads((DATA/seq/'session_boundary.json').read_text());cut=b['rescan_start_frame']
    removed=set(map(int,b['changed_instances']['removed']))
    changed=removed|set(map(int,b['changed_instances']['nonrigid']))|{int(x['instance_reference']) for x in b['changed_instances']['rigid']}
    gt=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(gt)==len(labels)
    present=set((labels[labels>0]%1000).tolist());tree=cKDTree(gt)
    predictions={k:[] for k in VARIANTS};pending=[]
    for record in summary['records']:
        cid=record['candidate_id'];a=np.load(ROOT/seq/f'{cid:04d}.npz');points=a['reference_points']
        identity=associate(points,tree,labels)
        for variant in VARIANTS:
            status=record[variant];evidence=points[a[variant+'_depth']]
            match=associate(evidence,tree,labels)
            gid=match['gt_id'];coverage=None
            if gid is not None:
                object_gt=gt[(labels>0)&(labels%1000==gid)]
                d,_=cKDTree(evidence).query(object_gt);coverage=float((d<=.05).mean())
            predictions[variant].append(dict(candidate_id=cid,state=status['state'],
                confirmed_frame=status['confirmed_frame'],
                phase='rescan' if status['confirmed_frame'] is not None and status['confirmed_frame']>=cut else 'reference',
                reference_gt=identity['gt_id'],coverage_5cm=coverage,**match))
        if identity['gt_id'] in removed and record['original']['state']=='suspected':
            pending.append(dict(candidate_id=cid,gt_id=identity['gt_id'],
                original_state=record['original']['state'],lower_gate_state=record['lower_single_frame_gate']['state'],
                batch_state=record['batched']['state'],batch_hits=record['batched']['confirmations'],
                packets=record['batched']['packet_events'],
                accepted_input_frames=len(record['batched']['accepted_input_frames']),
                pending_distinct_points=record['batched']['pending_distinct_points']))
    metrics={}
    for variant,rows in predictions.items():
        selected=[r for r in rows if r['state']=='would_remove' and r['phase']=='rescan']
        ids={r['gt_id'] for r in selected if r['gt_id'] is not None}
        metrics[variant]=dict(confirmed_candidates=len(selected),removed_hits=sorted(ids&removed),
            any_changed_hits=sorted(ids&changed),no_change_annotation_ids=sorted(ids-changed),
            mixed_or_no_depth=sum(r['gt_id'] is None for r in selected),
            reference_confirmed=sum(r['state']=='would_remove' and r['phase']=='reference' for r in rows))
    scenes.append(dict(seq=seq,official_removed=sorted(removed),verifiable_removed=sorted(removed&present),
        metrics=metrics,predictions=predictions,previously_pending=pending,
        candidates=source['candidates'],replay_seconds=source['replay_seconds']))
out=dict(protocol='Same frozen candidates and raw observation stream. Replay is causal, no GT decisions, no deletion. 5 cm depth-evidence majority and per-scene GT dedup.',scenes=scenes)
(ROOT/'comparison.json').write_text(json.dumps(out,indent=2))
for scene in scenes:
    print(scene['seq'],json.dumps(scene['metrics']))
    print('previous pending',[(p['gt_id'],p['batch_state'],p['pending_distinct_points']) for p in scene['previously_pending']])
