"""Export-time RGB-D consistency refinement, with no local TSDF allocation."""
import copy,json,shutil,time
from pathlib import Path
import numpy as np
import open3d as o3d
from .it_adapter_multires import InstanceTSDFAdapter
from .selective_local import refine_instances

class ObservationRefinementAdapter(InstanceTSDFAdapter):
    def __init__(self,cfg,owner,directory):
        cfg=copy.deepcopy(cfg)
        cfg.setdefault('multi_resolution',{})['geometry_policy']='hybrid'
        super().__init__(cfg,owner,directory)

    def observe(self,*args,**kwargs):
        super().observe(*args,**kwargs)
        # Only depth/pose/raw mask labels are needed; do not retain RGB images.
        self.packets[-1].pop('rgb',None)

    def export(self,frame_id,output):
        started=time.perf_counter();output=Path(output);base=output/'ig_audit'
        cloud=o3d.io.read_point_cloud(str(base/'final.ply'));xyz=np.asarray(cloud.points)
        with np.load(base/'ckpt_final.npz') as f:arrays={k:f[k] for k in f.files}
        ids=json.loads((base/'export_instance_ids.json').read_text())
        tracks={r['id']:r for r in json.loads((base/'submap_migration.json').read_text())['instances']}
        packets={p['frame']:p for p in self.packets}
        masks,audit=refine_instances(xyz,arrays['pred_masks'],ids,tracks,packets.__getitem__,self.K)
        shutil.copy2(base/'final.ply',output/'final.ply')
        if np.array_equal(masks,arrays['pred_masks']):
            shutil.copy2(base/'ckpt_final.npz',output/'ckpt_final.npz')
        else:np.savez_compressed(output/'ckpt_final.npz',**dict(arrays,pred_masks=masks))
        (output/'export_instance_ids.json').write_text(json.dumps(ids))
        audit.update(export_points=len(xyz),export_instances=len(ids),export_seconds=time.perf_counter()-started,
                     local_tsdf_seconds=0.,geometry_unchanged=True,identity_unchanged=True,
                     fallback_to_ig=bool(np.array_equal(masks,arrays['pred_masks'])))
        report=dict(method='IGSubmap-ObsRefine',audit=audit,exported_instance_ids=ids,
                    tolerance_m=.025,bbox_margin_m=.05,min_agreeing_views=3,
                    local_tsdf=False,new_instances=False,gt_used=False)
        (output/'observation_refinement.json').write_text(json.dumps(report,indent=2))
        return report
