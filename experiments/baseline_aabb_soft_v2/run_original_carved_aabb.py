"""Original OnlineAnySeg + conservative current-view carved AABB evidence.

The baseline still owns candidates, tracks, merges and export. Carving only
provides a temporary AABB for the soft score during the next merge update; it
does not delete baseline voxels or create an independent submap.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

from aabb_soft import AABBMetrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--strength', type=float, default=.15)
    p.add_argument('--band', type=float, default=.10)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--depth-shrink', type=int, choices=[0,1], default=1)
    p.add_argument('--tag', default='ablation')
    p.add_argument('args', nargs=argparse.REMAINDER)
    o=p.parse_args()
    if not 0 <= o.strength <= .25 or not 0 <= o.band <= .5:
        p.error('strength must be within [0,.25], band within [0,.5]')
    repo=Path(o.repo).resolve(); os.chdir(repo)
    sys.path.insert(0,str(repo)); sys.path.insert(1,str(repo/'third_party/FCGF'))
    random.seed(o.seed); np.random.seed(o.seed); torch.manual_seed(o.seed); torch.cuda.manual_seed_all(o.seed)
    from depth_mask_refinement import shrink_segmentation_by_depth
    import Scene_rep as source
    original=source.Scene_rep; scenes=[]

    class CarvedScene(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.metrics=AABBMetrics(self.metrics,self,o.strength,o.band)
            self.aabb_boxes_override=None; self.latest_depth=None; self.latest_pose=None
            self.latest_seg=None; self.aabb_update_times=[]; self.carve_stats=[]
            self.ablation_stats={'temporary_matrix_triggers':0,'candidate_pairs':0,
                                 'selected_pairs':0,'merge_calls':0,'merge_pairs':0,
                                 'baseline_status_transitions':{'merge_calls':0,'pairs_selected':0}}
            self.depth_shrink_stats=[]
            self._ablation_depth_shrink=bool(o.depth_shrink)
            if hasattr(self, 'depth_shrink_enabled'):
                self.depth_shrink_enabled=self._ablation_depth_shrink
            scenes.append(self)

        @torch.no_grad()
        def _make_carved_boxes(self):
            n=self.c_mask_num; device=self.device
            boxes=torch.zeros((n,2,3),device=device,dtype=torch.float32)
            if n == 0 or self.latest_depth is None or self.latest_pose is None:
                return boxes
            depth=self.latest_depth.to(device).float(); pose=self.latest_pose.to(device).float()
            if depth.ndim == 3: depth=depth[0]
            K=self.intrinsics.to(device).float(); h,w=depth.shape[-2:]
            inv_pose=torch.linalg.inv(pose)
            tolerance=max(.06,2.*float(self.cfg['scene']['voxel_size']))
            removed=0; total=0; fallback=0
            for i in range(n):
                points=self.mask_voxels_coords[i].to(device).float()
                if points.numel()==0: continue
                total += int(points.shape[0])
                ph=torch.cat((points,torch.ones((points.shape[0],1),device=device)),dim=1)
                cam=(ph@inv_pose.T)[:,:3]; z=cam[:,2]
                u=torch.round(cam[:,0]/z.clamp_min(1e-6)*K[0,0]+K[0,2]).long()
                v=torch.round(cam[:,1]/z.clamp_min(1e-6)*K[1,1]+K[1,2]).long()
                inside=(z>.1)&(u>=0)&(u<w)&(v>=0)&(v<h)
                valid=torch.zeros_like(inside)
                valid[inside]=torch.isfinite(depth[v[inside],u[inside]]) & (depth[v[inside],u[inside]]>.1)
                d=torch.zeros_like(z); d[valid]=depth[v[valid],u[valid]]
                free=valid&(z<d-tolerance)
                kept=points[~free]
                # Unknown, out-of-view and behind-surface points survive. If
                # one frame would remove most of an instance, keep its raw box.
                if kept.shape[0] < max(16,int(.10*points.shape[0])):
                    kept=points; fallback += 1
                else:
                    removed += int(free.sum())
                lo=torch.quantile(kept,.01,dim=0); hi=torch.quantile(kept,.99,dim=0)
                boxes[i]=torch.stack((lo,hi))
            self.carve_stats.append(dict(frame=int(self._carve_frame), removed=removed,
                                         input_points=total, fallback_boxes=fallback))
            return boxes

        @property
        def get_containing_mat(self):
            self.ablation_stats['temporary_matrix_triggers'] += 1
            return super().get_containing_mat

        def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w,
                             frustum_block_coords, seg_image, mask_features):
            if not hasattr(self, 'depth_shrink_enabled') and self._ablation_depth_shrink:
                seg_image, stats = shrink_segmentation_by_depth(seg_image, depth_img, self.cfg)
                stats['frame']=int(frame_id)
                self.depth_shrink_stats.append(stats)
            result=super().insert_seg_frame(frame_id,color_img,depth_img,pose_c2w,
                                            frustum_block_coords,seg_image,mask_features)
            self._carve_frame=int(frame_id)
            self.latest_depth=depth_img.detach().clone()
            self.latest_pose=pose_c2w.detach().clone()
            self.latest_seg=seg_image.detach().clone() if torch.is_tensor(seg_image) else None
            return result

        def merge_masks(self, frame_id, merge_mat, c_mat, *args, **kwargs):
            self.ablation_stats['merge_calls'] += 1
            if torch.is_tensor(merge_mat):
                self.ablation_stats['merge_pairs'] += int(torch.count_nonzero(merge_mat).item()) // 2
            return super().merge_masks(frame_id, merge_mat, c_mat, *args, **kwargs)

        def _select_sparse_merge_pairs(self, frame_id):
            result=super()._select_sparse_merge_pairs(frame_id)
            self.ablation_stats['candidate_pairs'] += len(getattr(self, 'pair_graph', {}).pairs) if hasattr(getattr(self, 'pair_graph', None), 'pairs') else 0
            self.ablation_stats['selected_pairs'] += len(result)
            return result

        def update_masks(self, frame_id):
            self._carve_frame=int(frame_id)
            self.aabb_boxes_override=self._make_carved_boxes() if o.strength > 0 and o.band > 0 else None
            torch.cuda.synchronize(); start=time.perf_counter()
            self.metrics.frame=int(frame_id)
            result=super().update_masks(frame_id)
            torch.cuda.synchronize(); self.aabb_update_times.append(dict(frame=int(frame_id),seconds=time.perf_counter()-start))
            if hasattr(self, 'pair_graph'):
                self.ablation_stats['pair_observations']=int(self.pair_graph.total_observations)
                self.ablation_stats['new_pairs']=int(self.pair_graph.new_pair_count)
                self.ablation_stats['pair_status_transitions']=dict(self.pair_graph.status_transitions)
            return result

    source.Scene_rep=CarvedScene
    raw=o.args[1:] if o.args[:1]==['--'] else o.args
    sys.argv=[str(repo/'main.py')]+raw
    text=(repo/'main.py').read_text()
    if o.limit:
        text=text.replace('max_frame_num = None','max_frame_num = %d'%o.limit,1)
    start=time.perf_counter(); status='failed'
    try:
        exec(compile(text,str(repo/'main.py'),'exec'),{'__name__':'__main__','__file__':str(repo/'main.py')})
        status='complete'
    finally:
        for scene in scenes:
            files=['main.py','Scene_rep.py','Metrics.py','MaskGraph.py','Instance.py','PC_extractor.py','voxelized_points.py','Dataset/dataset.py','tool/post_process.py']
            manifest=dict(status=status,seconds=time.perf_counter()-start,strength=o.strength,band=o.band,seed=o.seed,limit=o.limit,tag=o.tag,depth_shrink_enabled=bool(o.depth_shrink),repo=str(repo),argv=raw,
                          original_hashes={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},
                          experiment_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                          updates=scene.aabb_update_times,evidence=scene.metrics.records,carving=scene.carve_stats,
                          depth_shrink=getattr(scene, 'depth_shrink_stats', []),
                          ablation_stats=getattr(scene, 'ablation_stats', {}),
                          aabb_candidate_pairs=sum(int(x.get('eligible_pairs', 0)) for x in getattr(scene.metrics, 'records', [])),
                          note='Original dense MaskGraph baseline; strength=0 bypasses Carving-AABB computation and score adjustment. Depth shrink changes temporary backprojection masks only. No lineage, same-frame guard or spatial pair audit.')
            dest=Path(scene.seq_output_dir)/'carved_aabb_manifest.json'; dest.write_text(json.dumps(manifest,indent=2))
            print('CARVED_AABB_RUN',status,'seconds',manifest['seconds'],'manifest',dest,flush=True)


if __name__=='__main__': main()
