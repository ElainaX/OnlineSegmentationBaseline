"""Summarize matched AP and actual mutations without reading GT in mapping."""
import json
from pathlib import Path

root=Path('/root/autodl-tmp/output/ig_deletion_20261008')
rows=json.loads((root/'results.json').read_text())
attempts=list(rows)
retry=root/'retry_pair01/results.json'
if retry.exists():
    for r in json.loads(retry.read_text()):
        attempts.append(r)
        if r['status']=='complete':
            rows=[x for x in rows if (x['seq'],x['variant'])!=(r['seq'],r['variant'])]
            rows.append(r)
old=json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/results.json').read_text())
historical={r['seq']:r for r in old if r['variant']=='off'}
comparison=[]
for seq in sorted({r['seq'] for r in rows}):
    pair={r['variant']:r for r in rows if r['seq']==seq}
    result=dict(seq=seq,variants={k:{f:r.get(f) for f in ['status','AP','AP50','AP25','main_loop_seconds',
        'detection_seconds','voxel_mm','mapping_interval','segmentation_interval','selected_frames','error']}
        for k,r in pair.items()})
    if all(k in pair and pair[k]['status']=='complete' for k in ['off','on']):
        off,on=pair['off'],pair['on']
        result['delta_percentage_points']={k:round(100*(on[k]-off[k]),4) for k in ['AP','AP50','AP25']}
        result['historical_off']={k:historical[seq][k] for k in ['AP','AP50','AP25']}
        result['off_repeat_delta_percentage_points']={k:round(100*(off[k]-historical[seq][k]),4) for k in ['AP','AP50','AP25']}
        d=on['deletion']
        result['deletion']={k:d[k] for k in ['deletion_frames','voxel_erasure_operations','unique_world_voxels','seconds']}
        result['deletion']['instance_memberships_removed']=sum(e.get('removed_instance_voxel_memberships',0) for e in d['events'])
        result['deletion']['tracks_emptied']=sum(len(e.get('emptied_current_ids',[])) for e in d['events'])
        result['deletion']['candidate_ids_with_authorized_corners']=sorted({c['candidate'] for e in d['events'] for c in e['candidates'] if c['lineage_allowed']>0})
        def folder(r):
            cmd=r['command'];return Path(cmd[cmd.index('--output_dir')+1])/seq
        first=min((e['frame'] for e in d['events'] if e['voxels']>0), default=None)
        traces={mode:{t['frame']:t['count'] for t in json.loads((folder(pair[mode])/'merge_trace.json').read_text())}
            for mode in ['off','on']}
        differences=[f for f in sorted(set(traces['off'])&set(traces['on'])) if traces['off'][f]!=traces['on'][f]]
        result['repeatability']=dict(first_erasure_frame=first,
            first_merge_count_divergence=differences[0] if differences else None,
            pre_erasure_merge_count_differences=[f for f in differences if first is None or f<first])
        result['detections']={mode:pair[mode]['detection'] for mode in ['off','on']}
    comparison.append(result)
(root/'comparison.json').write_text(json.dumps(comparison,indent=2))
(root/'selected_results.json').write_text(json.dumps(rows,indent=2))
(root/'all_attempts.json').write_text(json.dumps(attempts,indent=2))
print(json.dumps(comparison,indent=2))
