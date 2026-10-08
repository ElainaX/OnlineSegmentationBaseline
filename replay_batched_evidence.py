"""Causal replay: frozen candidates and RGB-D evidence; no mapping or GT inputs."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from submap_disjoint.batched_evidence import BatchedCandidate
from submap_disjoint.local_clear_only import LocalClearCandidate
from submap_disjoint.readonly_detection_core import Candidate

SOURCE = Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
OUT = Path('/root/autodl-tmp/output/ig_batched_evidence_20261008')
OUT.mkdir(exist_ok=True)
if (OUT/'replay.json').exists():raise FileExistsError('Do not overwrite an existing replay')
scenes=[]
for run in json.loads((SOURCE/'selected_results.json').read_text()):
    seq=run['seq']; source=Path(run['command'][run['command'].index('--output_dir')+1])/seq
    target=OUT/seq;target.mkdir(exist_ok=False)
    cfg=json.loads((source/'remove_detection/config.json').read_text())['effective_settings']
    low_cfg=dict(cfg,min_reference_tested_points=1)
    meta=json.loads((source/'fixed_candidate_memory/summary.json').read_text())
    records=[];start=time.perf_counter()
    for item in meta['records']:
        cid=item['candidate_id'];path=source/'fixed_candidate_memory'/f'{cid:04d}.npz'
        a=np.load(path)
        assert hashlib.sha256(a['reference_points'].tobytes()).hexdigest()==item['point_sha256']
        args=(cid,frozenset(item['source_raw_ids']),item['original']['source_slot'],
              item['first_frame'],a['reference_points'],a['reference_colors'])
        original=Candidate(*args)
        clear=LocalClearCandidate(*args)
        low=LocalClearCandidate(*args)
        batch=BatchedCandidate(*args)
        trace=[]
        for i,f in enumerate(a['frames']):
            frame=int(f);pose=a['poses'][i];codes=a['evidence'][i];sources=a['sources'][i]
            if a['original_live'][i]:original.observe(frame,pose,codes,sources,cfg)
            clear.observe(frame,pose,codes,sources,cfg)
            low.observe(frame,pose,codes,sources,low_cfg)
            before=batch.state
            reason,stats=batch.observe(frame,pose,codes,sources,cfg)
            trace.append(dict(frame=frame,reason=reason,state=batch.state,
                pending_points=int((batch.pending_codes!=0).sum()),
                hits=batch.hits,state_changed=before!=batch.state))
        assert original.metadata()==item['original'],('original replay differs',seq,cid)
        row=dict(candidate_id=cid,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 reference_points=len(original.points),original=original.metadata(),
                 local_clear=clear.metadata(),lower_single_frame_gate=low.metadata(),
                 batched=batch.metadata(),batch_trace=trace)
        records.append(row)
        np.savez_compressed(target/f'{cid:04d}.npz',reference_points=original.points,
            original_depth=original.depth_contradicted,local_clear_depth=clear.depth_contradicted,
            lower_single_frame_gate_depth=low.depth_contradicted,batched_depth=batch.depth_contradicted,
            pending_codes=batch.pending_codes,ever_tested=batch.ever_tested)
    scene=dict(seq=seq,candidates=len(records),records=records,replay_seconds=time.perf_counter()-start,
               cfg=cfg,source=str(source),original_replay_exact=True,source_commit=run.get('commit'))
    (target/'summary.json').write_text(json.dumps(scene,indent=2))
    scenes.append(dict(seq=seq,candidates=len(records),replay_seconds=scene['replay_seconds'],
                       source=str(source),original_replay_exact=True))
    print(seq,'candidates',len(records),'seconds',scene['replay_seconds'],flush=True)
(OUT/'replay.json').write_text(json.dumps(dict(completed=True,GT_used=False,actual_deletions=0,scenes=scenes),indent=2))
