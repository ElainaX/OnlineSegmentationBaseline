"""Post-hoc aligned-world geometry audit; no evaluation ICP here."""
from pathlib import Path
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

root=Path('/root/autodl-tmp/output/ig_deletion_20261008')
seq='pair_04_752cc595_55551083'
data=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg')
def xyz(p):return np.asarray(o3d.io.read_point_cloud(str(p)).points)
points=xyz(data/seq/(seq+'_vh_clean_2.ply'))
labels=np.loadtxt(data/'gt_seg_all_things'/(seq+'.txt'),dtype=np.int64)
gt=points[(labels>0)&(labels%1000==17)]
out=dict(gt17_rescan_vertices=len(gt),coordinate_protocol='Existing aligned world coordinates; no extra ICP; all geometry including unlabeled points')
nearest_points={};distances={}
for mode in ['off','on']:
    p=xyz(root/mode/seq/'final.ply');d,idx=cKDTree(p).query(gt)
    nearest_points[mode]=p[idx];distances[mode]=d
    out[mode]=dict(coverage={str(r):float((d<=r).mean()) for r in [.025,.05,.10,.15]},median_distance_m=float(np.median(d)))
out['same_nearest_geometry']=dict(max_distance_difference_m=float(np.max(np.abs(distances['off']-distances['on']))),
    same_nearest_position_fraction=float((np.linalg.norm(nearest_points['off']-nearest_points['on'],axis=1)<1e-6).mean()))
c=np.load(root/'on'/seq/'remove_detection/candidates/0012/000532_would_remove.npz')
p=c['reference_points'];tree=cKDTree(gt)
for name,use in [('confirmation_depth',(c['contradiction_source']&1)>0),('cumulative_depth',c['depth_contradicted'])]:
    d,_=tree.query(p[use]);out[name]=dict(points=len(d),within_5cm=int((d<=.05).sum()),within_10cm=int((d<=.10).sum()),median_distance_m=float(np.median(d)))
archives=[np.load(p)['xyz'] for p in (root/'on'/seq/'deletion/archive').glob('*.npz')]
erased=np.unique(np.concatenate(archives),axis=0)
dist,_=cKDTree(p[c['depth_contradicted']]).query(erased)
local=erased[dist<=np.sqrt(3)*.025+1e-5]
d,_=tree.query(local)
out['erased_near_C12']=dict(points=len(d),within_5cm_of_rescan_gt17=int((d<=.05).sum()),within_10cm=int((d<=.10).sum()),median_distance_m=float(np.median(d)),note='Spatial attribution, TSDF corners; proximity alone does not prove harmful deletion.')
(root/'visualization/pair04_C12_geometry_audit.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
