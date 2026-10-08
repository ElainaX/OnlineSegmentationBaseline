"""Audit a rescan-only sampling change, including unchanged reference evidence."""
import json
from pathlib import Path
import numpy as np

root = Path('/root/autodl-tmp/output/ig_rescan_dense_20261008')
previous = Path('/root/autodl-tmp/output/ig_surface_bank_strict_20261008')
load = lambda p: json.loads(p.read_text())
sampling = {r['seq']: r for r in load(root/'sampling.json')}
comparisons = {name: {r['seq']: r for r in load(path/'comparison.json')['scenes']}
               for name, path in [('previous', previous), ('control', root/'control'), ('dense', root/'dense')]}
scenes = []
for seq, schedule in sampling.items():
    cut = schedule['boundary']
    a = np.load(root/'control'/seq/'bank.npz')
    b = np.load(root/'dense'/seq/'bank.npz')
    old = np.load(previous/seq/'bank.npz')
    exact_control = all(np.array_equal(a[k], old[k]) for k in a.files)
    assert exact_control, (seq, 'Control did not reproduce previous detector')
    ar, br = a['created_frame'] < cut, b['created_frame'] < cut
    reference_equal = all(np.array_equal(a[k][ar], b[k][br])
                          for k in ['points', 'colors', 'keys', 'created_frame'])
    ae, be = a['event_frames'] < cut, b['event_frames'] < cut
    reference_equal &= all(np.array_equal(a[k][ae], b[k][be])
                           for k in ['event_frames', 'event_indices', 'event_states'])
    assert reference_equal, (seq, 'Reference geometry or decisions changed')
    summaries = {n: load(root/n/seq/'summary.json') for n in ['control', 'dense']}
    assert summaries['control']['cfg'] == summaries['dense']['cfg']
    assert summaries['control']['voxel_size'] == summaries['dense']['voxel_size'] == .025
    row = dict(seq=seq, reference_frames=schedule['reference_frames'],
               original_rescan_frames=schedule['original_rescan_frames'],
               dense_rescan_frames=schedule['dense_rescan_frames'],
               exact_control_reproduction=exact_control, unchanged_reference=reference_equal,
               evaluable_removed=len(comparisons['control'][seq]['verifiable_removed']))
    for name in ['control', 'dense']:
        scene = comparisons[name][seq]
        row[name] = dict(metrics=scene['phases']['rescan']['metrics'],
                         active_points=scene['phases']['rescan']['active_points'],
                         seconds=scene['wall_seconds'], array_bytes=scene['array_memory_bytes'])
    for size in ['20', '100']:
        ac, bc = row['control']['metrics'][size], row['dense']['metrics'][size]
        row['changes_'+size] = {
            'gained_removed': sorted(set(bc['removed_hits'])-set(ac['removed_hits'])),
            'lost_removed': sorted(set(ac['removed_hits'])-set(bc['removed_hits'])),
            'gained_unannotated': sorted(set(bc['no_change_annotation_ids'])-set(ac['no_change_annotation_ids'])),
            'lost_unannotated': sorted(set(ac['no_change_annotation_ids'])-set(bc['no_change_annotation_ids']))}
    scenes.append(row)
totals = {}
for name in ['control', 'dense']:
    totals[name] = {}
    for size in ['20', '100']:
        totals[name][size] = dict(
            removed_hits=sum(len(s[name]['metrics'][size]['removed_hits']) for s in scenes),
            evaluable_removed=sum(s['evaluable_removed'] for s in scenes),
            unannotated_objects=sum(len(s[name]['metrics'][size]['no_change_annotation_ids']) for s in scenes),
            mixed_regions=sum(s[name]['metrics'][size]['mixed_or_unmatched_regions'] for s in scenes),
            regions=sum(s[name]['metrics'][size]['regions'] for s in scenes))
report = dict(scenes=scenes, totals=totals, actual_deletions=0, new_AP=False,
              note='Same raw scan sessions, denser rescan only; independent-view thresholds unchanged. Expanded-entry read-only prototype, not an end-to-end IG AP run.')
(root/'density_comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
