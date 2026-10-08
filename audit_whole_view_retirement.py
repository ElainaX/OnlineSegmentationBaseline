"""Offline GT audit of whole-view retirement; GT is never used for inference."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from eval_global_ap import cloud

ROOT = Path('/root/autodl-tmp/output/ig_direct_retirement_20261009')
DATA = Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
SEQ = 'pair_01_0cac767c_19f1a892'


def main():
    ref = cloud(DATA/'gt_reference'/SEQ/(SEQ+'_vh_clean_2.ply'))
    labels = np.loadtxt(DATA/'gt_seg_reference_all_things'/(SEQ+'.txt'), dtype=int)
    current = cloud(DATA/SEQ/(SEQ+'_vh_clean_2.ply'))
    rt, ct = cKDTree(ref), cKDTree(current)
    folder = ROOT/'instance_views/hybrid'/SEQ/'instance_retirement'
    events = [json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
    rows = []
    for event in events:
        if event['event'] != 'retired':
            continue
        frame, cid = event['frame'], event['candidate']
        evidence = np.load(folder/f'evidence_{frame:06d}_{cid:05d}.npz')
        archive = np.load(folder/'archive'/f'{frame:06d}_{min(event["raw_ids"]):08d}.npz')
        row = dict(event=event)
        for name, points in [('reference_surface', evidence['points']),
                             ('erased_TSDF_corners', archive['coords'][archive['erase']])]:
            d, near = rt.query(points)
            owner = np.where((d <= .05) & (labels[near] > 0), labels[near] % 1000, -1)
            ids, counts = np.unique(owner, return_counts=True)
            row[name] = dict(count=len(points), reference_GT_counts={str(g):int(n) for g,n in zip(ids,counts)},
                near_current_GT_5cm=int((ct.query(points)[0] <= .05).sum()))
        rows.append(row)
    gap = json.loads((ROOT/'gap/results.json').read_text())
    summary = dict(tolerance_m=.05, GT_used_online=False, retirements=rows, variants={})
    for mode in ['direct','instance_views']:
        r = json.loads((ROOT/mode/'result.json').read_text())
        s = json.loads((ROOT/mode/'hybrid'/SEQ/'instance_retirement/summary.json').read_text())
        g = gap['variants'][mode]
        summary['variants'][mode] = dict(standard_AP=r['original_AP'],G_AP=g['G_AP'],
            deletion_score=g['deletion']['score'], retention_score=g['retention']['score'],
            main_seconds=r['main_loop_seconds'], whole_module_seconds=s['seconds'],
            retired=s['retired_instances'], schedule_parity=r['schedule_parity'],
            detector_arrays_identical=r['detector_arrays_identical'],
            removed_GT_domain=g['deletion']['per_removed_GT_domain'],
            removed_GT_matches=g['deletion']['per_GT'])
    final = ROOT/'instance_views/hybrid'/SEQ
    points = cloud(final/'final.ply')
    masks = np.load(final/'ckpt_final.npz')['pred_masks']
    distances, near = rt.query(points)
    stool = (distances <= .05) & (labels[near] > 0) & (labels[near] % 1000 == 2)
    counts = masks[stool].sum(axis=0)
    summary['remaining_stool_geometry'] = dict(final_points_near_stool=int(stool.sum()),
        final_instance_indices_with_stool_points={str(i):int(n) for i,n in enumerate(counts) if n},
        unassigned_stool_points=int((~masks[stool].any(axis=1)).sum()),
        note='Prediction indices are final array columns, not persistent raw IDs; masks can overlap.')
    out = ROOT/'whole_view_audit.json'
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
