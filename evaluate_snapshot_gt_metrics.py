"""Offline audit of read-only snapshot disappearance proposals against 3RScan GT.

The detector never receives these labels.  This script only joins frozen
candidate points to the aligned reference GT after a run has completed.
"""
from __future__ import annotations
import argparse, json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d


def read_points(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points, dtype=np.float32)


def load_events(path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def gt_removed(report, strict=False):
    result=[]
    for x in report['changed_instances'].get('removed', []):
        ref=x['reference']; cur=x.get('rescan_same_id', {})
        if ref.get('vertex_count', 0) <= 0:
            continue
        if strict and cur.get('vertex_count', 0) > 10:
            continue
        result.append(dict(id=int(ref['object_id']), label=ref.get('label'),
                           reference_points=int(ref.get('vertex_count', 0)),
                           rescan_points=int(cur.get('vertex_count', 0))))
    return result


def one_scene(seq, run_root, data_root, radius=.05, match_fraction=.30):
    run=Path(run_root)/seq/'snapshot_absence'
    data=Path(data_root)/seq
    summary=json.loads((run/'summary.json').read_text())
    report=json.loads((data/'gt_report.json').read_text())
    ref=read_points(Path(data_root)/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))
    labels=np.loadtxt(Path(data_root)/'gt_seg_reference_all_things'/(seq+'.txt'), dtype=np.int64)
    assert len(ref)==len(labels)
    tree=cKDTree(ref)
    events=load_events(run/'events.jsonl')
    by_file={}
    for e in events:
        f=e.get('candidate_file')
        if not f: continue
        # use the latest evidence for a candidate, while preserving any
        # positive proposal seen earlier.
        old=by_file.get(f)
        if old is None or (e['frame'], e['status']=='would_remove') >= (old['frame'], old['status']=='would_remove'):
            by_file[f]=e
        elif e['status']=='would_remove':
            old['status']='would_remove'
    removed=gt_removed(report, strict=False)
    strict_removed=gt_removed(report, strict=True)
    removed_ids={x['id'] for x in removed}; strict_ids={x['id'] for x in strict_removed}
    candidates=[]
    for f,e in by_file.items():
        fp=Path(f)
        if not fp.exists():
            continue
        with np.load(fp) as z: pts=z['points'].astype(np.float32)
        dist,idx=tree.query(pts, distance_upper_bound=radius)
        valid=np.isfinite(dist)&(dist<=radius)&(idx<len(labels))
        safe_idx=np.minimum(idx, len(labels)-1)
        valid &= labels[safe_idx]>0
        ids=(labels[safe_idx[valid]]%1000).astype(np.int64)
        counts=Counter(ids.tolist())
        total=max(1,len(pts)); top=counts.most_common(8)
        top_id=int(top[0][0]) if top else None
        top_n=int(top[0][1]) if top else 0
        row=dict(candidate_file=f, frame=int(e['frame']), row=int(e['row']), raw_ids=e['raw_ids'],
                 status=e['status'], reasons=e.get('reasons', []), points=int(len(pts)),
                 matched_reference_points=int(valid.sum()), match_fraction=float(valid.mean()),
                 dominant_gt_id=top_id, dominant_gt_fraction=float(top_n/total),
                 gt_objects=[dict(id=int(i),points=int(n),fraction=float(n/total),
                                  removed=int(i in removed_ids), strict_removed=int(i in strict_ids)) for i,n in top],
                 dino_mean_similarity=e.get('dino_mean_similarity'),
                 dino_min_similarity=e.get('dino_min_similarity'),
                 dino_max_similarity=e.get('dino_max_similarity'),
                 current_free_fraction=e.get('current_free_fraction'),
                 current_support_fraction=e.get('current_support_fraction'),
                 tested_fraction=e.get('tested_fraction'),
                 confirmed_fraction=e.get('confirmed_fraction'),
                 owned_free_fraction=e.get('owned_free_fraction'),
                 owned_support_fraction=e.get('owned_support_fraction'),
                 current_joint_absence_points=e.get('current_joint_absence_points'),
                 unknown_points=e.get('unknown_points'), occluded_points=e.get('occluded_points'))
        candidates.append(row)
    # A candidate can be a mixed track.  Associate it to a GT object only if
    # that object owns at least match_fraction of its frozen points.
    assoc=defaultdict(list)
    for c in candidates:
        for g in c['gt_objects']:
            if g['fraction']>=match_fraction: assoc[g['id']].append(c)
    def object_status(obj, ids):
        cs=assoc.get(obj['id'],[])
        pos=[c for c in cs if c['status']=='would_remove']
        return dict(id=obj['id'],label=obj['label'],reference_points=obj['reference_points'],
                    rescan_points=obj['rescan_points'],candidate_count=len(cs),
                    would_remove_count=len(pos),status=('detected' if pos else ('unknown' if cs else 'unobserved')),
                    candidate_files=[c['candidate_file'] for c in cs])
    objects=[object_status(o,removed_ids) for o in removed]
    strict_objects=[object_status(o,strict_ids) for o in strict_removed]
    def weighted(rows, key):
        den=sum(x['reference_points'] for x in rows)
        return sum(x['reference_points'] for x in rows if key(x))/max(1,den)
    positives=[c for c in candidates if c['status']=='would_remove']
    false=[c for c in positives if not any(g['id'] in removed_ids and g['fraction']>=match_fraction for g in c['gt_objects'])]
    false_strict=[c for c in positives if not any(g['id'] in strict_ids and g['fraction']>=match_fraction for g in c['gt_objects'])]
    n_events=len(events); pending_events=sum(e['status']!='would_remove' for e in events)
    result=dict(sequence=seq, protocol=dict(radius_m=radius, candidate_match_fraction=match_fraction,
        gt_removed='gt_report.changed_instances.removed; object100/zero-reference excluded',
        strict_removed='same list with rescan_same_id.vertex_count <= 10',
        unknown_definition='matched candidate exists but no would_remove; unobserved is reported separately'),
        gt_removed_count=len(removed),strict_gt_removed_count=len(strict_removed),
        gt_removed_reference_points=sum(x['reference_points'] for x in removed),
        strict_gt_removed_reference_points=sum(x['reference_points'] for x in strict_removed),
        objects=objects,strict_objects=strict_objects,candidates=candidates,
        metrics=dict(
            size_weighted_recall=float(weighted(objects,lambda x:x['status']=='detected')),
            strict_size_weighted_recall=float(weighted(strict_objects,lambda x:x['status']=='detected')),
            gt_unknown_weight=float(weighted(objects,lambda x:x['status']=='unknown')),
            gt_unobserved_weight=float(weighted(objects,lambda x:x['status']=='unobserved')),
            candidate_unknown_fraction=float(pending_events/max(1,n_events)),
            candidate_positive_count=len(positives),
            false_deletion_rate=(float(len(false)/len(positives)) if positives else None),
            strict_false_deletion_rate=(float(len(false_strict)/len(positives)) if positives else None),
            false_deletion_denominator=len(positives),
            average_bbox_candidates=float(summary['counters'].get('bbox_candidates',0)/max(1,summary['counters'].get('frames',0))),
            average_tested_candidates=float(summary['counters'].get('tested_candidates',0)/max(1,summary['counters'].get('frames',0))),
            unique_candidates=len(candidates),event_count=n_events),
        summary_counters=summary['counters'],summary_runtime={k:summary.get(k) for k in ['total_seconds','feature_seconds','feature_extractions','cache_peak_bytes']})
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);p.add_argument('--scenes',nargs='+',required=True);a=p.parse_args()
    results=[one_scene(s,a.run_root,a.data_root) for s in a.scenes]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(results=results),indent=2,ensure_ascii=False))
    for x in results:
        print(json.dumps(dict(seq=x['sequence'],metrics=x['metrics'],gt=x['gt_removed_count'],strict_gt=x['strict_gt_removed_count']),ensure_ascii=False))

if __name__=='__main__':main()
