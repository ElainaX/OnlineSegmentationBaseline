"""Collect checked results without changing the experiment or its predictions."""
import hashlib, json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ROOT=Path('/root/autodl-tmp/output/independent_repair_20261007')
diag_path=ROOT/'pair04_coverage_diagnosis.json'
diag=json.loads(diag_path.read_text())
seq='pair_04_752cc595_55551083'
audit=ROOT/'audit'/seq
data=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
gt=np.asarray(o3d.io.read_point_cloud(str(data/seq/(seq+'_vh_clean_2.ply'))).points)
T=np.array(diag['T_recon_gt']); gt=gt@T[:3,:3].T+T[:3,3]
labels=np.loadtxt(data/'gt_seg'/(seq+'.txt'),dtype=np.int64)%1000
points=np.asarray(o3d.io.read_point_cloud(str(audit/'final.ply')).points)
z=np.load(audit/'ckpt_585/export_stage_audit.npz')
for name in ['all_tracks','before_dbscan','after_dbscan']:
    masks=np.unpackbits(z[name],axis=1,count=int(z['point_count'])).astype(bool)
    union=masks.any(0)
    d=cKDTree(points[union]).query(gt,workers=4)[0] if union.any() else np.full(len(gt),np.inf)
    for row in diag['instances']:
        for t,stage in row['by_distance_m'].items():
            stage['stages'][name]['nearest_labeled_surface_recall']=float((d[labels==row['object_id']]<=float(t)).mean())
diag['proximity_note']='nearest_labeled_surface_recall searches the union of labeled surfaces independently of the fixed full-geometry nearest point; not an instance purity/AP measurement'
diag_path.write_text(json.dumps(diag,indent=2))
combined=[]
for p in sorted(ROOT.glob('synthetic_final/*/results.json'))+sorted(ROOT.glob('real_final/*/results.json')):
    r=json.loads(p.read_text()); case=p.parent
    a=np.asarray(o3d.io.read_point_cloud(str(case/'global_replay_immediate.ply')).points)
    b=np.asarray(o3d.io.read_point_cloud(str(case/'independent_deactivate_immediate.ply')).points)
    equivalence={'replay_points':len(a),'independent_points':len(b),
                 'max_bidirectional_distance_m':float(max(cKDTree(a).query(b,workers=4)[0].max(),cKDTree(b).query(a,workers=4)[0].max()))}
    r['replay_independent_equivalence']=equivalence
    r['source_results']=str(p);r['case']=case.name
    for method in r['results']:
        b0=method['stages'][0]['stale_residual_fraction']
        for s in method['stages']:
            s['stale_residual_relative_to_pre_repair']=(s['stale_residual_fraction']/b0 if b0 else None)
        method['recovery_sampled_new_frames']=next((s['new_frames'] for s in method['stages'][1:]
                 if s['stale_residual_fraction'] is not None and s['stale_residual_fraction']<.05
                 and s['lost_previously_supported_static_fraction']<.01),None)
    combined.append(r)
report={'cases':combined,'diagnosis':diag,'rejected_runs':{
         'synthetic':'grid-aligned ideal wall had no extracted geometry; invalid negative control',
         'synthetic_validated':'geometry valid but earlier source version, superseded by synthetic_final',
         'synthetic_checked and real_checked':'superseded: added spatial clear queries and exact target-contribution support control',
         'real':'GPU hash capacity warnings; rejected as performance/quality evidence, pair01 interrupted'},
         'checks':{'required_cases':4,'completed_cases':len(combined),
                   'unit_test_log':(ROOT/'tests_final.log').read_text()},
         'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
             Path('submap_disjoint/independent_tsdf.py'),Path('run_repair_controls.py'),
             Path('run_real_repair_controls.py'),Path('test_independent_tsdf.py')]}}
(ROOT/'summary.json').write_text(json.dumps(report,indent=2))
for r in combined:
    print(r['case'],r['replay_independent_equivalence'])
    for m in r['results']:
        s=m['stages'][1]
        print(m['method'],round(m['repair_seconds'],4),round(s['stale_residual_fraction'],4),
              round(s['lost_previously_supported_static_fraction'],4),m['recovery_sampled_new_frames'])
