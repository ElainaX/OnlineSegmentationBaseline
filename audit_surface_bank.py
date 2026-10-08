"""GT only after detection: spatial components, strict removed and static alarms."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True)
ROOT=Path(parser.parse_args().output_dir)
DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')


def regions(points,tree,labels,radius):
    if not len(points):return [],{}
    d,idx=tree.query(points);gid=np.where((d<=.05)&(labels[idx]>0),labels[idx]%1000,-1)
    edges=cKDTree(points).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges),np.uint8),(edges[:,0],edges[:,1])),shape=(len(points),len(points)))
    n,comp=connected_components(graph,directed=False)
    rows=[]
    for i in range(n):
        members=np.flatnonzero(comp==i)
        if len(members)<20:continue
        counts=Counter(gid[members].tolist());identity,count=counts.most_common(1)[0]
        matched=identity>=0 and count>.5*len(members)
        rows.append(dict(region=i,points=len(members),gt_id=int(identity) if matched else None,
            purity=count/len(members),counts=dict(counts),centroid=points[members].mean(0).tolist()))
    return rows,dict(Counter(gid.tolist()))


scenes=[]
for run in json.loads((ROOT/'completed.json').read_text())['scenes']:
    seq=run['seq'];a=np.load(ROOT/seq/'bank.npz')
    summary=json.loads((ROOT/seq/'summary.json').read_text())
    boundary=json.loads((DATA/seq/'session_boundary.json').read_text());cut=boundary['rescan_start_frame']
    removed=set(map(int,boundary['changed_instances']['removed']))
    changed=removed|set(map(int,boundary['changed_instances']['nonrigid']))|{int(r['instance_reference']) for r in boundary['changed_instances']['rigid']}
    gt=np.asarray(o3d.io.read_point_cloud(str(DATA/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(DATA/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    assert len(gt)==len(labels)
    tree=cKDTree(gt);present=set((labels[labels>0]%1000).tolist())
    reference=np.zeros(len(a['points']),bool)
    for f in np.unique(a['event_frames'][a['event_frames']<cut]):
        mask=a['event_frames']==f
        reference[a['event_indices'][mask]]=a['event_states'][mask]
    # Require a current-phase free-space observation; do not count a stale alarm
    # that was last observed only during reference acquisition as rescan recall.
    masks=dict(reference=reference,rescan=a['active']&(a['last_conflict_frame']>=cut))
    phases={}
    for phase,mask in masks.items():
        rows,counts=regions(a['points'][mask],tree,labels,3*summary['voxel_size'])
        metrics={}
        for size in [20,100]:
            selected=[r for r in rows if r['points']>=size]
            ids={r['gt_id'] for r in selected if r['gt_id'] is not None}
            metrics[str(size)]=dict(regions=len(selected),gt_ids=sorted(ids),
                removed_hits=sorted(ids&removed),any_changed_hits=sorted(ids&changed),
                no_change_annotation_ids=sorted(ids-changed),
                mixed_or_unmatched_regions=sum(r['gt_id'] is None for r in selected))
        phases[phase]=dict(active_points=int(mask.sum()),regions=rows,point_gt_counts=counts,metrics=metrics)
    scene=dict(seq=seq,official_removed=sorted(removed),verifiable_removed=sorted(removed&present),
        phases=phases,wall_seconds=summary['wall_seconds'],array_memory_bytes=summary['array_memory_bytes'])
    scenes.append(scene)
    print(seq,'reference',json.dumps(phases['reference']['metrics']),'rescan',json.dumps(phases['rescan']['metrics']),flush=True)
(ROOT/'comparison.json').write_text(json.dumps(dict(scenes=scenes,
    protocol='New candidate entry. 25mm raw-depth memory; connected components radius 75mm; majority reference GT within 5cm; GT only in audit; local change not entire removal.'),indent=2))
