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
from scipy import ndimage

from aabb_soft import AABBMetrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--strength', type=float, default=.15)
    p.add_argument('--band', type=float, default=.10)
    p.add_argument('--split-guard', action='store_true')
    p.add_argument('--split-voxel', type=float, default=.025)
    p.add_argument('--split-ratio', type=float, default=.30)
    p.add_argument('--split-separation', type=float, default=.75)
    p.add_argument('--same-frame-guard', action='store_true')
    p.add_argument('--same-frame-overlap', type=float, default=.10)
    p.add_argument('--seed', type=int, default=27)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('args', nargs=argparse.REMAINDER)
    o=p.parse_args()
    if not 0 <= o.strength <= .25 or not 0 <= o.band <= .5:
        p.error('strength must be within [0,.25], band within [0,.5]')
    repo=Path(o.repo).resolve(); os.chdir(repo)
    sys.path.insert(0,str(repo)); sys.path.insert(1,str(repo/'third_party/FCGF'))
    random.seed(o.seed); np.random.seed(o.seed); torch.manual_seed(o.seed); torch.cuda.manual_seed_all(o.seed)
    import Scene_rep as source
    original=source.Scene_rep; scenes=[]

    class CarvedScene(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.metrics=AABBMetrics(self.metrics,self,o.strength,o.band)
            self.aabb_boxes_override=None; self.latest_depth=None; self.latest_pose=None
            self.latest_seg=None; self.aabb_update_times=[]; self.carve_stats=[]
            self.split_guard_cache={}; self.split_guard_stats=[]
            self.same_frame_guard_cache={}; self.same_frame_guard_stats=[]
            scenes.append(self)

        @torch.no_grad()
        def _split_guard_pair(self, i, j):
            key=(min(int(i),int(j)), max(int(i),int(j)))
            if key in self.split_guard_cache: return self.split_guard_cache[key]
            a=self.mask_voxels_coords.get(key[0]); b=self.mask_voxels_coords.get(key[1])
            if a is None or b is None or a.numel()==0 or b.numel()==0:
                self.split_guard_cache[key]=False; return False
            pts=torch.cat((a,b),dim=0).detach().float().cpu().numpy()
            voxel=float(o.split_voxel); org=pts.min(0)-voxel
            ijk=np.floor((pts-org)/voxel).astype(np.int32)
            shape=tuple((ijk.max(0)+2).tolist())
            # Avoid large dense allocations for pathological boxes.
            if np.prod(shape)>8_000_000:
                self.split_guard_cache[key]=False; return False
            occ=np.zeros(shape,np.uint8); occ[ijk[:,0],ijk[:,1],ijk[:,2]]=1
            lab,n=ndimage.label(occ,np.ones((3,3,3),np.uint8))
            pl=lab[ijk[:,0],ijk[:,1],ijk[:,2]]
            sizes=np.bincount(pl,minlength=n+1)[1:]
            order=np.argsort(sizes)[::-1]
            comps=[]
            for q in order:
                sel=pl==q+1; count=int(sel.sum())
                if count>=100: comps.append((count,pts[sel].mean(0)))
            suspicious=False; second_ratio=0.; separation=0.
            if len(comps)==2:
                second_ratio=comps[1][0]/max(1,len(pts))
                separation=float(np.linalg.norm(comps[0][1]-comps[1][1]))
                suspicious=second_ratio>=float(o.split_ratio) and separation>=float(o.split_separation)
            self.split_guard_cache[key]=suspicious
            if suspicious:
                self.split_guard_stats.append(dict(i=key[0],j=key[1],points=int(len(pts)),
                    components=len(comps),second_ratio=second_ratio,separation=separation))
            return suspicious

        @torch.no_grad()
        def _same_frame_guard_pair(self, i, j):
            key=(min(int(i),int(j)), max(int(i),int(j)))
            if key in self.same_frame_guard_cache: return self.same_frame_guard_cache[key]
            a=self.maskGraph.instance_dict.get(key[0]); b=self.maskGraph.instance_dict.get(key[1])
            if a is None or b is None:
                self.same_frame_guard_cache[key]=False; return False
            af={}; bf={}
            for f,m in a.ori_mask_list: af.setdefault(int(f),[]).append(int(m))
            for f,m in b.ori_mask_list: bf.setdefault(int(f),[]).append(int(m))
            common=sorted(set(af).intersection(bf)); conflict=False
            for frame in common:
                for ma in af[frame]:
                    for mb in bf[frame]:
                        pa=self.maskGraph.f_mask_voxels_clouds.get((frame,ma)); pb=self.maskGraph.f_mask_voxels_clouds.get((frame,mb))
                        if pa is None or pb is None or pa.shape[0] < 100 or pb.shape[0] < 100: continue
                        qa=np.round(pa.detach().float().cpu().numpy()/float(o.split_voxel)).astype(np.int64)
                        qb=np.round(pb.detach().float().cpu().numpy()/float(o.split_voxel)).astype(np.int64)
                        sa={tuple(x) for x in qa.tolist()}; sb={tuple(x) for x in qb.tolist()}
                        overlap=len(sa.intersection(sb))/max(1,min(len(sa),len(sb)))
                        separation=float(np.linalg.norm(qa.mean(0)-qb.mean(0))*float(o.split_voxel))
                        if overlap <= float(o.same_frame_overlap) and separation >= float(o.split_separation):
                            conflict=True
                            self.same_frame_guard_stats.append(dict(i=key[0],j=key[1],frame=frame,
                                overlap=overlap,separation=separation,points_a=len(sa),points_b=len(sb)))
                            break
                    if conflict: break
                if conflict: break
            self.same_frame_guard_cache[key]=conflict
            return conflict

        def merge_masks(self, frame_id, merge_mat, c_mat, count_merge=False, full_geo_merge=True):
            if o.split_guard and merge_mat is not None and torch.count_nonzero(merge_mat):
                guarded=merge_mat.clone()
                for pair in torch.nonzero(merge_mat, as_tuple=False).detach().cpu().tolist():
                    i,j=pair
                    blocked = o.split_guard and self._split_guard_pair(i,j)
                    blocked = blocked or (o.same_frame_guard and self._same_frame_guard_pair(i,j))
                    if blocked:
                        guarded[i,j]=False
                        if i < guarded.shape[0] and j < guarded.shape[1]: guarded[j,i]=False
                merge_mat=guarded
            return super().merge_masks(frame_id,merge_mat,c_mat,count_merge=count_merge,full_geo_merge=full_geo_merge)

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

        def insert_seg_frame(self, frame_id, color_img, depth_img, pose_c2w,
                             frustum_block_coords, seg_image, mask_features):
            result=super().insert_seg_frame(frame_id,color_img,depth_img,pose_c2w,
                                            frustum_block_coords,seg_image,mask_features)
            self._carve_frame=int(frame_id)
            self.latest_depth=depth_img.detach().clone()
            self.latest_pose=pose_c2w.detach().clone()
            self.latest_seg=seg_image.detach().clone() if torch.is_tensor(seg_image) else None
            return result

        def update_masks(self, frame_id):
            self._carve_frame=int(frame_id)
            self.split_guard_cache={}; self.split_guard_stats=[]
            self.same_frame_guard_cache={}; self.same_frame_guard_stats=[]
            self.aabb_boxes_override=self._make_carved_boxes()
            torch.cuda.synchronize(); start=time.perf_counter()
            self.metrics.frame=int(frame_id)
            result=super().update_masks(frame_id)
            torch.cuda.synchronize(); self.aabb_update_times.append(dict(frame=int(frame_id),seconds=time.perf_counter()-start))
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
            manifest=dict(status=status,seconds=time.perf_counter()-start,strength=o.strength,band=o.band,seed=o.seed,limit=o.limit,repo=str(repo),argv=raw,
                          original_hashes={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},
                          experiment_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                          updates=scene.aabb_update_times,evidence=scene.metrics.records,carving=scene.carve_stats,
                          split_guard=scene.split_guard_stats,
                          same_frame_guard=scene.same_frame_guard_stats,
                          note='Baseline candidates/tracks/merges/export retained; current-view depth carving supplies only the AABB soft-score box; no baseline voxels are deleted.')
            dest=Path(scene.seq_output_dir)/'carved_aabb_manifest.json'; dest.write_text(json.dumps(manifest,indent=2))
            print('CARVED_AABB_RUN',status,'seconds',manifest['seconds'],'manifest',dest,flush=True)


if __name__=='__main__': main()
