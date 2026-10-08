"""GT probe audit ONLY: can selected RGB-D frames test annotated removed surfaces?

This is not a detector and not an AP upper bound. GT and the session boundary
are deliberately used here, never fed back into the detection implementation.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from submap_disjoint.sensor_reference import sensor_projection, PointViewVotes
from submap_disjoint.readonly_detection_core import interior_mask, SUPPORT, CONTRADICTION

parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
output=Path(parser.parse_args().output)
if output.exists():raise FileExistsError(output)
D=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
R=Path('/root/autodl-tmp/output/ig_fixed_candidates_20261008')
schedules={r['seq']:r for r in json.loads(Path('/root/autodl-tmp/output/ig_correction_20261008/sources.json').read_text())}
records=[]
for run in json.loads((R/'selected_results.json').read_text()):
    seq=run['seq'];cmd=run['command'];source=Path(cmd[cmd.index('--output_dir')+1])/seq
    config=json.loads((source/'remove_detection/config.json').read_text());cfg=config['effective_settings'];cam=config['pipeline_config']['cam']
    boundary=json.loads((D/seq/'session_boundary.json').read_text());cut=boundary['rescan_start_frame']
    gt=np.asarray(o3d.io.read_point_cloud(str(D/'gt_reference'/seq/(seq+'_vh_clean_2.ply'))).points)
    labels=np.loadtxt(D/'gt_seg_reference_all_things'/(seq+'.txt'),dtype=np.int64)
    k=np.loadtxt(D/seq/'intrinsic/intrinsic_depth.txt')[:3,:3].astype(np.float32)
    frames=[]
    for f in schedules[seq]['sampling']['selected_frame_ids']:
        depth=cv2.imread(str(D/seq/'depth'/f'{f}.png'),cv2.IMREAD_UNCHANGED).astype(np.float32)/cam['depth_scale']
        valid=np.isfinite(depth)&(depth>cam['depth_near'])&(depth<cam['depth_far']);depth[~valid]=0
        inner=interior_mask(depth,valid,cfg['edge_threshold_m'],cfg['boundary_margin_px'])
        frames.append((f,depth,inner,np.loadtxt(D/seq/'pose'/f'{f}.txt').astype(np.float32)))
    for gid in map(int,boundary['changed_instances']['removed']):
        pts=gt[(labels>0)&(labels%1000==gid)];total=len(pts)
        if not total:
            records.append(dict(seq=seq,gt_id=gid,gt_vertices=0,category='not_in_evaluation_GT'));continue
        if total>6000:pts=pts[np.linspace(0,total-1,6000,dtype=int)]
        reference=PointViewVotes(len(pts));rescan=PointViewVotes(len(pts))
        ever_codes=np.zeros((4,len(pts)),bool)
        for f,depth,inner,pose in frames:
            codes,_=sensor_projection(pts,k,pose,depth,inner,cfg,cam['depth_near'],cam['depth_far'])
            if f<cut:reference.observe(f,pose,codes,cfg)
            else:
                rescan.observe(f,pose,codes,cfg)
                for code in range(4):ever_codes[code]|=codes==code
        witnessed=reference.support>=2
        tested=(rescan.support+rescan.conflict)>0
        free=rescan.conflict>=2
        joint=witnessed&free
        category=('multi_view_free_space' if joint.sum()>=20 else
                  'insufficient_reference_support' if witnessed.sum()<20 else
                  'no_multiview_free_space_on_verified_surface')
        row=dict(seq=seq,gt_id=gid,gt_vertices=total,probes=len(pts),category=category,
            reference_supported_once=int((reference.support>=1).sum()),reference_supported_twice=int(witnessed.sum()),
            rescan_tested=int(tested.sum()),rescan_support_twice=int((rescan.support>=2).sum()),
            rescan_free_once=int((rescan.conflict>=1).sum()),rescan_free_twice=int(free.sum()),
            verified_then_free_twice=int(joint.sum()),verified_then_free_fraction=float(joint.mean()),
            ever_rescan_occluded=int(ever_codes[3].sum()))
        records.append(row);print(json.dumps(row),flush=True)
output.write_text(json.dumps(dict(GT_used=True,diagnostic_only=True,
    limitations='Projection depends on scan registration and GT surface accuracy. Counts are potential geometric evidence, not guaranteed true changes, full-object visibility, or AP bounds.',records=records),indent=2))
