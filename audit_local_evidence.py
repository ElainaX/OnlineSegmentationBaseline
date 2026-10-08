"""GT used offline only; compare paired old candidates and new change regions."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from submap_disjoint.local_evidence import connected_regions

ROOT = Path('/root/autodl-tmp/output/ig_local_evidence_20261008')
DATA = Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
results = []


def association(points, tree, labels, radius=.05):
    if not len(points):
        return dict(majority_id=None, majority_fraction=0., points=0, counts={})
    d, idx = tree.query(points)
    ok = (d <= radius) & (labels[idx] > 0)
    counts = Counter((labels[idx[ok]] % 1000).tolist())
    gid, n = counts.most_common(1)[0] if counts else (None, 0)
    return dict(majority_id=gid if n / len(points) > .5 else None,
                majority_fraction=n / len(points), points=len(points), counts=dict(counts))


for run in json.loads((ROOT / 'results.json').read_text()):
    if run['status'] != 'complete':
        continue
    seq = run['seq']
    folder = ROOT / 'paired' / seq
    b = json.loads((DATA / seq / 'session_boundary.json').read_text())
    cut = b['rescan_start_frame']
    changes = b['changed_instances']
    removed = set(map(int, changes['removed']))
    changed = removed | set(map(int, changes['nonrigid'])) | {int(r['instance_reference']) for r in changes['rigid']}
    points = np.asarray(o3d.io.read_point_cloud(str(DATA / 'gt_reference' / seq / (seq + '_vh_clean_2.ply'))).points)
    labels = np.loadtxt(DATA / 'gt_seg_reference_all_things' / (seq + '.txt'), dtype=np.int64)
    assert len(points) == len(labels)
    tree = cKDTree(points)
    present = set((labels[labels > 0] % 1000).tolist())
    variants = {}
    old = json.loads((folder / 'remove_detection/summary.json').read_text())
    pred = []
    for c in old['instances']:
        if c['state'] != 'would_remove':
            continue
        saved = np.load(sorted((folder / 'remove_detection/candidates' / f"{c['candidate_id']:04d}").glob('*.npz'))[-1])
        p = saved['reference_points'][saved['depth_contradicted']]
        pred.append(dict(candidate_id=c['candidate_id'], phase='rescan' if c['confirmed_frame'] >= cut else 'reference',
                         **association(p, tree, labels)))
    variants['original'] = pred
    arrays = np.load(folder / 'local_evidence/final.npz')
    # Score states stay reversible; include currently confirmed cells only.
    # Compare confirmations in rescan, not static reference alerts counted as changes.
    for limit in (20, 100):
        active = np.flatnonzero(arrays['confirmed'] & (arrays['last_confirmed'] >= cut))
        regions = connected_regions(arrays['keys'], active)
        variants[f'local_ge{limit}'] = [dict(region_id=i, phase='rescan',
            **association(arrays['points'][ids], tree, labels)) for i, ids in enumerate(regions) if len(ids) >= limit]
    static = np.load(folder / 'local_evidence/reference_end.npz')
    static_regions = connected_regions(static['keys'], np.flatnonzero(static['confirmed']))
    static_stats = dict(confirmed_cells=int(static['confirmed'].sum()),
        regions_ge20=sum(len(r) >= 20 for r in static_regions), regions_ge100=sum(len(r) >= 100 for r in static_regions))
    summaries = {}
    for name, rows in variants.items():
        rr = [r for r in rows if r['phase'] == 'rescan']
        ids = {r['majority_id'] for r in rr if r['majority_id'] is not None}
        summaries[name] = dict(regions_or_candidates=len(rr),
            removed_hits=sorted(ids & removed), removed_recall=len(ids & removed) / len(removed & present),
            any_changed_hits=sorted(ids & changed),
            no_change_annotation_ids=sorted(ids - changed),
            unknown_or_mixed=sum(r['majority_id'] is None for r in rr),
            majority_changed_regions=sum(r['majority_id'] in changed for r in rr),
            reference_confirmed_candidates=sum(r['phase'] == 'reference' for r in rows))
    results.append(dict(seq=seq, official_removed=sorted(removed), verifiable_removed=sorted(removed & present),
        summaries=summaries, original_static_candidates=old['would_remove']-sum(r['phase']=='rescan' for r in pred),
        local_static_alerts=static_stats, predictions=variants,
        runtime=dict(original_seconds=run['original']['diagnostic_seconds'], local_seconds=run['local']['seconds'],
                     main_loop_seconds=run['main_loop_seconds'], sampled_frames=run['selected_frames']),
        bank=run['local']))

out = dict(protocol='Paired same-map read-only; reference GT majority within 5 cm; rescan confirmations only; deduplicated GT. No-change annotation is not proven false positive. Regions are not object identities.',
           scenes=results)
(ROOT / 'comparison.json').write_text(json.dumps(out, indent=2))
for r in results:
    print(r['seq'], json.dumps(r['summaries']), 'static', r['local_static_alerts'])
