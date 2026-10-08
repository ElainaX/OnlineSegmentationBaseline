"""GT-only offline audit of the fixed candidate memory ablation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from submap_disjoint.local_clear_only import LocalClearCandidate
from submap_disjoint.readonly_detection_core import Candidate

ROOT = Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
DATA = Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')


def match(points, tree, labels):
    if not len(points):
        return dict(gt_id=None, purity=0., points=0)
    dist, idx = tree.query(points)
    valid = (dist <= .05) & (labels[idx] > 0)
    counts = Counter((labels[idx[valid]] % 1000).tolist())
    gid, n = counts.most_common(1)[0] if counts else (None, 0)
    return dict(gt_id=gid if n > .5 * len(points) else None, purity=n / len(points), points=len(points))


scenes = []
result_file = ROOT / 'selected_results.json'
if not result_file.exists(): result_file = ROOT / 'results.json'
for run in json.loads(result_file.read_text()):
    if run['status'] != 'complete':
        continue
    seq = run['seq']
    folder = Path(run['command'][run['command'].index('--output_dir')+1]) / seq
    b = json.loads((DATA / seq / 'session_boundary.json').read_text())
    removed = set(map(int, b['changed_instances']['removed']))
    changed = removed | set(map(int, b['changed_instances']['nonrigid'])) | {int(r['instance_reference']) for r in b['changed_instances']['rigid']}
    cut = b['rescan_start_frame']
    gt = np.asarray(o3d.io.read_point_cloud(str(DATA / 'gt_reference' / seq / (seq + '_vh_clean_2.ply'))).points)
    labels = np.loadtxt(DATA / 'gt_seg_reference_all_things' / (seq + '.txt'), dtype=np.int64)
    assert len(gt) == len(labels)
    present = set((labels[labels > 0] % 1000).tolist())
    tree = cKDTree(gt)
    summary = json.loads((folder / 'fixed_candidate_memory/summary.json').read_text())
    cfg = json.loads((folder / 'remove_detection/config.json').read_text())['effective_settings']
    variants = {v: [] for v in ('original', 'local_clear_only', 'common', 'extended')}
    pool = []
    for record in summary['records']:
        cid = record['candidate_id']
        a = np.load(folder / 'fixed_candidate_memory' / f'{cid:04d}.npz')
        reference_match = match(a['reference_points'], tree, labels)
        pool.append(dict(candidate_id=cid, **reference_match))
        clear = LocalClearCandidate(cid, frozenset(record['source_raw_ids']), record['original']['source_slot'],
                                    record['first_frame'], a['reference_points'], a['reference_colors'])
        replay = Candidate(cid, clear.source_raw_ids, clear.source_slot, clear.first_frame, clear.points, clear.colors)
        clear_support_events = []
        for i, frame in enumerate(a['frames']):
            if not a['original_live'][i]:
                continue
            args = (int(frame), a['poses'][i], a['evidence'][i], a['sources'][i], cfg)
            reason, _ = clear.observe(*args)
            replay.observe(*args)
            if reason in ('support', 'cancelled'):
                clear_support_events.append(dict(frame=int(frame), state=clear.state,
                    retained_contradiction_points=int(clear.contradicted.sum()),
                    original_contradiction_points=int(replay.contradicted.sum()),
                    hits=clear.hits, support_hits=clear.support_hits))
        assert replay.metadata() == record['original'], ('offline original replay', cid)
        record['local_clear_only'] = clear.metadata()
        pool[-1]['local_clear_support_events'] = clear_support_events
        for variant in variants:
            status = record[variant]
            if variant == 'original':
                mask = a['original_depth_contradicted']
            elif variant == 'local_clear_only':
                mask = clear.depth_contradicted
            else:
                # Keep original source semantics: currently stable contradictory
                # points with positive depth evidence; RGB-only cannot be a hit.
                mask = (a[variant+'_score'] >= 2) & (a[variant+'_depth_score'] > 0)
            pred = a['reference_points'][mask]
            m = match(pred, tree, labels)
            phase = 'rescan' if status['confirmed_frame'] is not None and status['confirmed_frame'] >= cut else 'reference'
            coverage = None
            if m['gt_id'] is not None:
                local_gt = gt[(labels > 0) & (labels % 1000 == m['gt_id'])]
                distance, _ = cKDTree(pred).query(local_gt)
                coverage = float((distance <= .05).mean())
            variants[variant].append(dict(candidate_id=cid, state=status['state'],
                confirmed_frame=status['confirmed_frame'], phase=phase,
                fixed_reference_gt=reference_match['gt_id'], gt_vertex_coverage_5cm=coverage, **m))
    metrics = {}
    for variant, rows in variants.items():
        confirmed = [r for r in rows if r['state'] == 'would_remove' and r['phase'] == 'rescan']
        ids = {r['gt_id'] for r in confirmed if r['gt_id'] is not None}
        fixed_ids = {r['fixed_reference_gt'] for r in confirmed if r['points'] > 0 and r['fixed_reference_gt'] is not None}
        metrics[variant] = dict(confirmed_rescan_candidates=len(confirmed),
            removed_hits=sorted(ids & removed), any_changed_hits=sorted(ids & changed),
            no_change_annotation_ids=sorted(ids - changed),
            mixed_or_no_depth_candidates=sum(r['gt_id'] is None for r in confirmed),
            fixed_reference_removed_hits=sorted(fixed_ids & removed),
            reference_confirmed_candidates=sum(r['state']=='would_remove' and r['phase']=='reference' for r in rows))
    pool_ids = {r['gt_id'] for r in pool if r['gt_id'] is not None}
    events = [json.loads(line) for line in (folder / 'remove_detection/events.jsonl').read_text().splitlines()]
    by_id = {p['candidate_id']:p['gt_id'] for p in pool}
    pending = []
    for record in summary['records']:
        cid = record['candidate_id']
        if record['original']['state'] != 'suspected' or by_id[cid] not in removed:
            continue
        rows = [e for e in events if e['event'] == 'candidate_observation' and e['candidate_id'] == cid]
        pending.append(dict(candidate_id=cid, gt_id=by_id[cid],
            accepted_contradiction_observations=record['original']['confirmations'],
            reference_points=record['original']['reference_points'],
            reason_counts=dict(Counter(e['reason'] for e in rows)),
            max_cumulative_contradicted_fraction=max((e.get('cumulative_contradicted_fraction',0) for e in rows), default=0.)))
    scenes.append(dict(seq=seq, official_removed=sorted(removed), verifiable_removed=sorted(removed & present),
        candidates=summary['candidates'], candidate_pool_removed_gt=sorted(pool_ids & removed),
        checks=dict(original_replay_checks=summary['original_replay_checks'],
                    common_frame_checks=summary['common_frame_checks'],
                    observation_sets_identical=summary['observation_sets_identical']),
        metrics=metrics, predictions=variants, candidate_pool=pool, pending_diagnostics=pending,
        main_loop_seconds=run['main_loop_seconds'],
        accumulator_seconds=summary['accumulator_seconds']))
out = dict(protocol='Original candidate entry frozen. common uses exact original accepted observations. extended adds later observations separately. Majority of depth-evidence points within 5 cm; GT IDs deduplicated.',scenes=scenes)
(ROOT / 'comparison.json').write_text(json.dumps(out, indent=2))
for r in scenes:
    print(r['seq'], 'pool', r['candidate_pool_removed_gt'], 'metrics', json.dumps(r['metrics']))
