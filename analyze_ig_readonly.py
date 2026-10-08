"""Summarize diagnostic events; do not treat would_remove as GT correctness."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import open3d as o3d

p = argparse.ArgumentParser()
p.add_argument('--root',default='/root/autodl-tmp/output/ig_readonly_20261008/full')
a = p.parse_args()
root = Path(a.root)
rows = json.loads((root/'results.json').read_text())
attempts = [dict(r, artifact_directory=str(root/r['seq'])) for r in rows]
retry_root = root.parent/'trace_retry'
if (retry_root/'results.json').exists():
    attempts += [dict(r, artifact_directory=str(retry_root/r['seq']))
                 for r in json.loads((retry_root/'results.json').read_text())]
selected = {}
for r in attempts:
    if r['seq'] not in selected or r['status']=='complete': selected[r['seq']] = r
rows = [selected[k] for k in sorted(selected)]
report = []
for row in rows:
    if row['status'] != 'complete':
        report.append(dict(seq=row['seq'],status=row['status'])); continue
    directory = Path(row['artifact_directory'])
    meta = row['metadata']; summary = row['detection']
    events = [json.loads(x) for x in (directory/'remove_detection/events.jsonl').read_text().splitlines()]
    cut_path = Path(row['command'][row['command'].index('--dataset')+1])/'session_boundary.json'
    if not cut_path.exists(): cut_path = cut_path.parent.parent/'session_boundary.json'
    cut = json.loads(cut_path.read_text())['rescan_start_frame']
    phase = {'reference':Counter(), 'rescan':Counter(), 'unknown':Counter()}
    rejects = Counter()
    for event in events:
        session = 'reference' if 0 <= event['frame'] < cut else 'rescan' if event['frame'] >= cut else 'unknown'
        if event['event']=='frame_evaluated': phase[session]['evaluated_frames']+=1
        if event['event']=='candidate_created': phase[session]['created']+=1
        if event['event']=='candidate_observation' and event.get('reason') in ('would_remove','cancelled'):
            phase[session][event['reason']]+=1
        if event['event']=='candidate_rejected': rejects[event.get('reason','unknown')]+=1
    confirmed = [x for x in summary['instances'] if x['state']=='would_remove']
    sources = Counter()
    for x in confirmed:
        depth=x['depth_contradicted_fraction']>0
        rgb=x['rgb_contradicted_fraction']>0
        sources['depth_and_rgb' if depth and rgb else 'depth_only' if depth else 'rgb_only' if rgb else 'none']+=1
    current = np.asarray(o3d.io.read_point_cloud(str(directory/'final.ply')).points)
    previous = np.asarray(o3d.io.read_point_cloud(str(Path('/root/autodl-tmp/output/ig_correction_20261008/off')/row['seq']/'final.ply')).points)
    def ordered(x): return x[np.lexsort((x[:,2],x[:,1],x[:,0]))]
    same_count = len(current)==len(previous)
    maxdiff = float(np.abs(ordered(current)-ordered(previous)).max()) if same_count and len(current) else None
    report.append(dict(seq=row['seq'],status='complete',
        artifact_directory=str(directory),attempt_history=[dict(status=r['status'],directory=r['artifact_directory'],
            error=r.get('error'),gpu=r.get('gpu')) for r in attempts if r['seq']==row['seq']],
        interval=meta['mapping_frame_interval'],voxel_mm=meta['tsdf_voxel_size_mm'],
        selected_frames=len(meta['selected_frame_ids']),evaluated_frames=summary['evaluated_frames'],
        skipped_frames=summary['skipped_frames'],candidates=summary['candidates'],
        would_remove=summary['would_remove'],cancelled=summary['cancelled'],suspected=summary['suspected'],
        diagnostic_seconds=summary['diagnostic_seconds'],main_loop_seconds=meta['main_loop_seconds'],
        source_counts=dict(sources),by_phase={k:dict(v) for k,v in phase.items()},
        candidate_rejection_counts=dict(rejects),actual_tsdf_deletions=summary['actual_tsdf_deletions'],
        final_geometry_comparison=dict(reference='prior IG off run',points=len(current),reference_points=len(previous),
            sorted_coordinates_identical=same_count and maxdiff==0,maximum_coordinate_difference_m=maxdiff)))
(root/'diagnostic_analysis.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
