"""Fixed-candidate diagnostic: depth-only vs raw sensor vs past surface witnesses.

No GT or session boundary is read. Historical witnesses use only selected input
frames strictly before candidate creation. Mapping/deletion are never changed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from submap_disjoint.batched_evidence import BatchedCandidate
from submap_disjoint.sensor_reference import sensor_projection, PointViewVotes
from submap_disjoint.readonly_detection_core import (
    SUPPORT, CONTRADICTION, SOURCE_DEPTH, SOURCE_RGB, interior_mask,
)

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True)
args = parser.parse_args()
OUT = Path(args.output_dir)
OUT.mkdir(exist_ok=False)
SOURCE = Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
SCHEDULES = {r['seq']: r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text())}
variants = ['batched', 'depth_original', 'depth_sensor', 'depth_sensor_witness']
scenes = []
for run in json.loads((SOURCE/'selected_results.json').read_text()):
    started = time.perf_counter()
    seq = run['seq']; cmd = run['command']
    data = Path(cmd[cmd.index('--dataset')+1])
    source = Path(cmd[cmd.index('--output_dir')+1])/seq
    target = OUT/seq; target.mkdir()
    config = json.loads((source/'remove_detection/config.json').read_text())
    cfg = config['effective_settings']; cam = config['pipeline_config']['cam']
    assert cam['h_crop'] == cam['w_crop'] == 0, 'Cropping needs explicit calibration adjustment'
    k = np.loadtxt(data/'intrinsic/intrinsic_depth.txt')[:3, :3].astype(np.float32)
    selected = SCHEDULES[seq]['sampling']['selected_frame_ids']
    observations = {}
    # Store exactly the inputs already used by mapping, no densification of time.
    for f in selected:
        raw = cv2.imread(str(data/'depth'/f'{f}.png'), cv2.IMREAD_UNCHANGED)
        if raw is None: raise FileNotFoundError(data/'depth'/f'{f}.png')
        depth = raw.astype(np.float32)/cam['depth_scale']
        assert depth.shape == (cam['img_h'], cam['img_w'])
        valid = np.isfinite(depth) & (depth > cam['depth_near']) & (depth < cam['depth_far'])
        depth[~valid] = 0
        inner = interior_mask(depth, valid, cfg['edge_threshold_m'], cfg['boundary_margin_px'])
        observations[f] = (depth, inner, np.loadtxt(data/'pose'/f'{f}.txt').astype(np.float32))
    records = []
    meta = json.loads((source/'fixed_candidate_memory/summary.json').read_text())
    for item in meta['records']:
        cid = item['candidate_id']; path = source/'fixed_candidate_memory'/f'{cid:04d}.npz'
        with np.load(path) as a:
            points = a['reference_points']; colors = a['reference_colors']
            assert hashlib.sha256(points.tobytes()).hexdigest() == item['point_sha256']
            make = (cid, frozenset(item['source_raw_ids']), item['original']['source_slot'], item['first_frame'], points, colors)
            objects = {v: BatchedCandidate(*make) for v in variants}
            past = PointViewVotes(len(points))
            for f in selected:
                if f >= item['first_frame']: break
                depth, inner, pose = observations[f]
                codes, _ = sensor_projection(points,k,pose,depth,inner,cfg,cam['depth_near'],cam['depth_far'])
                past.observe(f,pose,codes,cfg)
            witnessed = past.support >= cfg['confirm_observations']
            transitions = np.zeros((4,4), np.int64)
            traces = []
            for i, f0 in enumerate(a['frames']):
                f = int(f0); depth, inner, pose = observations[f]
                np.testing.assert_allclose(pose, a['poses'][i], atol=1e-6)
                old = a['evidence'][i].copy(); src = a['sources'][i].copy()
                # Pure appearance change is depth support, not disappearance.
                old_depth = old.copy(); old_depth[(src & SOURCE_RGB)!=0] = SUPPORT
                dep_src = np.where(old_depth==CONTRADICTION,SOURCE_DEPTH,0).astype(np.uint8)
                raw_codes, raw_src = sensor_projection(points,k,pose,depth,inner,cfg,cam['depth_near'],cam['depth_far'])
                transitions += np.bincount(old_depth.astype(int)*4+raw_codes, minlength=16).reshape(4,4)
                checked = raw_codes.copy(); checked[~witnessed] = 0
                checked_src = raw_src.copy(); checked_src[~witnessed] = 0
                inputs = [(old,src),(old_depth,dep_src),(raw_codes,raw_src),(checked,checked_src)]
                trace = dict(frame=f, old_tested=int(np.isin(old_depth,[1,2]).sum()),
                    sensor_tested=int(np.isin(raw_codes,[1,2]).sum()), sensor_conflict=int((raw_codes==2).sum()),
                    witnessed_conflict=int((checked==2).sum()))
                for v, (codes,sources) in zip(variants,inputs):
                    reason,_ = objects[v].observe(f,pose,codes,sources,cfg)
                    trace[v] = reason
                traces.append(trace)
            row = dict(candidate_id=cid, first_frame=item['first_frame'], source=str(path),
                reference_points=len(points), witnessed_points=int(witnessed.sum()),
                code_transition_matrix=transitions.tolist(), traces=traces,
                **{v:o.metadata() for v,o in objects.items()})
            records.append(row)
            np.savez_compressed(target/f'{cid:04d}.npz', reference_points=points,
                past_support=past.support,past_conflict=past.conflict,
                **{v+'_depth':o.depth_contradicted for v,o in objects.items()})
    summary = dict(seq=seq,candidates=len(records),records=records,cfg=cfg,
        replay_seconds=time.perf_counter()-started,source=str(source),
        variants=variants, historical_witness_rule='>=2 independent depth supports strictly before candidate creation',
        selected_frame_ids=selected, voxel_size=config['pipeline_config']['scene']['voxel_size'])
    (target/'summary.json').write_text(json.dumps(summary,indent=2))
    scenes.append({k:summary[k] for k in ['seq','candidates','replay_seconds','source']})
    print(seq, 'done', summary['replay_seconds'],flush=True)
(OUT/'replay.json').write_text(json.dumps(dict(completed=True,GT_used=False,actual_deletions=0,variants=variants,scenes=scenes),indent=2))
