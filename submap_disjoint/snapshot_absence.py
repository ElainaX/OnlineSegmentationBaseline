"""Read-only disappearance proposals. Spatial retrieval, no identity association.

All coordinates are world meters / camera-Z meters. Raw IG mask lineage is
only used to follow existing merges; this module never changes a merge or TSDF.
"""
from collections import OrderedDict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SnapshotOptions:
    model_repo: str = '/root/models/dinov2_snapshot'
    weights: str = '/root/models/dinov2_snapshot_weights/dinov2_vits14_pretrain.pth'
    model_name: str = 'dinov2_vits14'
    input_long_edge: int = 448
    views_per_instance: int = 3
    retrieval_views: int = 3
    max_candidates_per_frame: int = 8
    max_pending: int = 128
    cache_mb: int = 256
    min_mask_pixels: int = 100
    min_patch_tokens: int = 4
    min_age_frames: int = 30
    min_view_translation_m: float = .05
    min_view_angle_deg: float = 5.
    min_vote_gap: int = 5
    bbox_margin_m: float = .10
    bbox_grid_m: float = 1.
    bbox_max_cells: int = 4096
    max_tokens_per_view: int = 256
    semantic_absence_below: float = .55
    min_confirmed_fraction: float = .60
    min_tested_fraction: float = .70
    max_support_fraction: float = .05
    min_current_free_fraction: float = .10
    min_points: int = 20
    confirm_views: int = 2
    max_reference_points: int = 4000

    def __post_init__(self):
        for k in ['views_per_instance','retrieval_views','max_candidates_per_frame','max_pending','cache_mb','min_patch_tokens','max_tokens_per_view','confirm_views','max_reference_points']:
            if getattr(self,k)<1: raise ValueError(k)
        for k in ['semantic_absence_below','min_confirmed_fraction','min_tested_fraction','max_support_fraction','min_current_free_fraction']:
            if not 0<=getattr(self,k)<=1: raise ValueError(k)
        if self.input_long_edge<56 or self.bbox_grid_m<=0: raise ValueError('Invalid spatial/image resolution')


def array(x):
    return x.detach().cpu().numpy() if isinstance(x,torch.Tensor) else np.asarray(x)


def rgb_bytes(x):
    a=array(x)
    if a.ndim!=3 or a.shape[-1]!=3: raise ValueError('Expected RGB HWC')
    if np.issubdtype(a.dtype,np.floating) and a.max(initial=0)<=1: a=a*255
    return np.clip(np.rint(a),0,255).astype(np.uint8)


def overlap(a,b,margin=0.):
    return bool(np.all(a[0]-margin<=b[1]) and np.all(b[0]-margin<=a[1]))


class BoxIndex:
    """Coarse spatial hash; large boxes use an explicit bounded fallback list."""
    def __init__(self,cell=1.,limit=4096): self.cell=cell;self.limit=limit;self.grid={};self.large=[];self.boxes={}
    def keys(self,box):
        lo=np.floor(box[0]/self.cell).astype(int); hi=np.floor(box[1]/self.cell).astype(int)
        if np.prod(hi-lo+1,dtype=np.float64)>self.limit:return None
        return [(x,y,z) for x in range(lo[0],hi[0]+1) for y in range(lo[1],hi[1]+1) for z in range(lo[2],hi[2]+1)]
    def rebuild(self,boxes):
        self.grid={};self.large=[];self.boxes=boxes
        for key,box in boxes.items():
            cells=self.keys(box)
            if cells is None:self.large.append(key)
            else:
                for cell in cells:self.grid.setdefault(cell,set()).add(key)
    def query(self,box):
        cells=self.keys(box)
        ids=set(self.boxes) if cells is None else set(self.large).union(*(self.grid.get(c,()) for c in cells))
        return [i for i in ids if overlap(self.boxes[i],box)]


def frustum_box(K,pose,shape,near,far):
    h,w=shape
    rays=np.array([[0,0,1],[w-1,0,1],[0,h-1,1],[w-1,h-1,1]],float)@np.linalg.inv(K).T
    pts=np.vstack([rays*near,rays*far])@pose[:3,:3].T+pose[:3,3]
    return np.array([pts.min(0),pts.max(0)])


def independent(points,old_camera,new_camera,old_frame,frame,opt):
    a=points-old_camera;b=points-new_camera
    den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    angle=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/np.maximum(den,1e-8),-1,1)))
    motion=np.linalg.norm(np.asarray(old_camera)-new_camera,axis=-1)
    return ((frame-old_frame>=opt.min_vote_gap)&(motion>=opt.min_view_translation_m)&(angle>=opt.min_view_angle_deg))


def update_evidence(state,points,codes,low_semantic,frame,pose,opt):
    """1=support, 2=free, 3=occluded, 0=unknown. Unknown never adds absence."""
    n=len(points)
    if 'votes' not in state:
        state.update(votes=np.zeros(n,np.uint16),last=np.full(n,-1,np.int64),camera=np.zeros((n,3)),tested=np.zeros(n,bool))
    if len(state['votes'])!=n:raise ValueError('Reference changed without resetting evidence')
    support=codes==1;free=codes==2
    state['tested']|=support|free
    state['votes'][support]=0;state['last'][support]=-1
    negative=free&low_semantic
    good=negative & ((state['last']<0)|independent(points,state['camera'],pose[:3,3],state['last'],frame,opt))
    state['votes'][good]=np.minimum(state['votes'][good].astype(np.int32)+1,opt.confirm_views)
    state['last'][good]=frame;state['camera'][good]=pose[:3,3]
    confirmed=state['votes']>=opt.confirm_views
    stats=dict(reference_points=n,tested_fraction=float(state['tested'].mean()),confirmed_fraction=float(confirmed.mean()),
               current_free_fraction=float(free.mean()),current_support_fraction=float(support.mean()),
               current_joint_absence_points=int(negative.sum()),new_votes=int(good.sum()),confirmed_points=int(confirmed.sum()),
               unknown_points=int((codes==0).sum()),occluded_points=int((codes==3).sum()))
    reasons=[]
    if n<opt.min_points:reasons.append('too_few_reference_points')
    if stats['tested_fraction']<opt.min_tested_fraction:reasons.append('insufficient_tested_coverage')
    if stats['confirmed_fraction']<opt.min_confirmed_fraction:reasons.append('insufficient_joint_absence')
    if stats['current_support_fraction']>opt.max_support_fraction:reasons.append('current_depth_support')
    if stats['current_free_fraction']<opt.min_current_free_fraction:reasons.append('not_currently_free')
    stats['reasons']=reasons
    return not reasons,stats


class DinoEncoder:
    def __init__(self,opt,device):
        self.opt=opt;self.device=torch.device(device);self.calls=0;self.seconds=0.
        # Loading a feature extractor must not perturb baseline RNG streams.
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
            self.model=torch.hub.load(opt.model_repo,opt.model_name,source='local',pretrained=False)
            weights=torch.load(opt.weights,map_location='cpu')
            self.model.load_state_dict(weights,strict=True)
            self.model=self.model.eval().to(device)
        self.patch=int(self.model.patch_size)
        self.sha256=hashlib.sha256(Path(opt.weights).read_bytes()).hexdigest()
    @torch.inference_mode()
    def encode(self,rgb):
        start=time.perf_counter();h,w=rgb.shape[:2];p=self.patch
        scale=self.opt.input_long_edge/max(h,w)
        hh=max(p,round(h*scale/p)*p);ww=max(p,round(w*scale/p)*p)
        x=torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device,dtype=torch.float32).permute(2,0,1)[None]/255
        x=F.interpolate(x,size=(hh,ww),mode='bilinear',align_corners=False)
        mean=x.new_tensor([.485,.456,.406])[None,:,None,None];std=x.new_tensor([.229,.224,.225])[None,:,None,None]
        with torch.autocast(device_type=self.device.type,enabled=self.device.type=='cuda',dtype=torch.float16):
            tokens=self.model.forward_features((x-mean)/std)['x_norm_patchtokens'][0]
        tokens=F.normalize(tokens.float(),dim=1).cpu().numpy().astype(np.float16)
        gh,gw=hh//p,ww//p
        yy,xx=np.meshgrid((np.arange(gh)+.5)*h/gh-.5,(np.arange(gw)+.5)*w/gw-.5,indexing='ij')
        uv=np.rint(np.c_[xx.ravel(),yy.ravel()]).astype(int)
        self.calls+=1;self.seconds+=time.perf_counter()-start
        return dict(features=tokens,uv=uv,grid=np.array([gh,gw]))


class FrameCache:
    def __init__(self,path,mb):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.limit=mb*1024**2
        self.cache=OrderedDict();self.bytes=0;self.peak=0;self.hits=0;self.misses=0
    def _put(self,frame,data):
        size=sum(x.nbytes for x in data.values())
        if frame in self.cache:self.bytes-=sum(x.nbytes for x in self.cache.pop(frame).values())
        while self.cache and self.bytes+size>self.limit:
            _,d=self.cache.popitem(last=False);self.bytes-=sum(x.nbytes for x in d.values())
        if size<=self.limit:self.cache[frame]=data;self.bytes+=size;self.peak=max(self.peak,self.bytes)
    def save(self,frame,data):
        np.savez_compressed(self.path/f'{frame:06d}.npz',**data);self._put(frame,data)
    def get(self,frame):
        if frame in self.cache:self.hits+=1;self.cache.move_to_end(frame);return self.cache[frame]
        self.misses+=1
        with np.load(self.path/f'{frame:06d}.npz') as f:data={k:f[k] for k in f.files}
        self._put(frame,data);return data


class SnapshotAbsenceInspector:
    """Independent diagnostic proposals. No GT, session boundary, or writes to owner."""
    def __init__(self,owner,cfg,output,opt=None):
        self.owner=owner;self.opt=opt or SnapshotOptions();self.cfg=cfg['remove_detection'];self.cam=cfg['cam']
        self.K=array(owner.intrinsics).astype(float);self.voxel=cfg['scene']['voxel_size']
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=True)
        self.encoder=DinoEncoder(self.opt,owner.device);self.frames=FrameCache(self.out/'frames',self.opt.cache_mb)
        self.records={};self.pending={};self.next_record=0;self.counter=Counter();self.seconds=0.
        self.last_test={};self.last_input=None;self.index=BoxIndex(self.opt.bbox_grid_m,self.opt.bbox_max_cells)

    def group(self):
        """Existing raw IDs follow IG merges. No semantic identity matching."""
        mapping=self.owner.global2merged_mask_id;groups={}
        for sid,s in list(self.records.items()):
            row=mapping[s['raw']] if s['raw']<len(mapping) else -1
            if row<0:self.records.pop(sid);self.counter['inactive_snapshots']+=1;continue
            groups.setdefault(int(row),[]).append(sid)
        for row,ids in groups.items():
            # Old diverse views are retained; incoming views only replace a
            # near-identical view with strictly more mask tokens in capture().
            ids.sort(key=lambda i:self.records[i]['frame'])
            keep=[]
            for sid in ids:
                s=self.records[sid]
                if not keep or all(self.view_different(s,self.records[j]) for j in keep):keep.append(sid)
                if len(keep)==self.opt.views_per_instance:break
            for sid in set(ids)-set(keep):self.records.pop(sid);self.counter['pruned_redundant_views']+=1
            groups[row]=keep
        return groups

    def view_different(self,a,b):
        center=(a['box'][0]+a['box'][1]+b['box'][0]+b['box'][1])/4
        return bool(independent(center[None],b['camera'][None],a['camera'],np.array([min(a['frame'],b['frame'])]),max(a['frame'],b['frame']),self.opt)[0])

    def features(self,frame,rgb):
        if self.last_input is not None and self.last_input[0]==frame:return self.last_input[1]
        d=self.encoder.encode(rgb);self.last_input=(frame,d);return d

    def capture(self,frame,rgb,depth,pose,seg):
        start=time.perf_counter();rgb=rgb_bytes(rgb);depth=array(depth);pose=array(pose);seg=array(seg)
        groups=self.group();mapping=self.owner.global2merged_mask_id;new=[];f=None
        from .readonly_detection_core import interior_mask
        valid=np.isfinite(depth)&(depth>self.cam['depth_near'])&(depth<self.cam['depth_far'])
        inner=interior_mask(depth,valid,self.cfg['edge_threshold_m'],self.cfg['boundary_margin_px'])
        for pair in self.owner.last_inserted_mask_pairs:
            raw=pair['global_mask_id'];row=mapping[raw]
            if row<0:continue
            mask=seg==pair['mask_id'];mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&inner
            if mask.sum()<self.opt.min_mask_pixels:self.counter['small_masks']+=1;continue
            y,x=np.nonzero(mask);sample=np.linspace(0,len(x)-1,min(len(x),2000)).astype(int);x,y=x[sample],y[sample]
            points=(np.c_[x,y,np.ones(len(x))]@np.linalg.inv(self.K).T)*depth[y,x,None]
            points=points@pose[:3,:3].T+pose[:3,3];box=np.array([points.min(0),points.max(0)])
            cand=dict(frame=frame,raw=raw,mask_id=int(pair['mask_id']),box=box,camera=pose[:3,3].copy())
            old=[self.records[i] for i in groups.get(row,[])]
            if len(old)>=self.opt.views_per_instance or any(not self.view_different(cand,s) for s in old):continue
            if f is None:f=self.features(frame,rgb)
            uv=f['uv'];take=np.flatnonzero(mask[uv[:,1],uv[:,0]])
            if len(take)<self.opt.min_patch_tokens:self.counter['small_patch_masks']+=1;continue
            take=take[np.linspace(0,len(take)-1,min(len(take),self.opt.max_tokens_per_view)).astype(int)]
            xy=uv[take];pts=(np.c_[xy,np.ones(len(xy))]@np.linalg.inv(self.K).T)*depth[xy[:,1],xy[:,0],None]
            cand.update(tokens=take,points=pts@pose[:3,:3].T+pose[:3,3])
            sid=self.next_record;self.next_record+=1;self.records[sid]=cand;groups.setdefault(row,[]).append(sid);new.append(sid)
        if new:
            self.frames.save(frame,dict(rgb=rgb,depth=depth.astype(np.float32),mask=seg.astype(np.int32),pose=pose,K=self.K,**f))
            for sid in new:
                s=self.records[sid];np.savez_compressed(self.out/f'snapshot_{sid:06d}.npz',**s)
        self.counter['snapshots_created']+=len(new);self.seconds+=time.perf_counter()-start

    def inspect(self,frame,rgb,depth,pose):
        start=time.perf_counter();pose=array(pose);depth=array(depth)
        if not np.isfinite(pose).all():return
        groups=self.group();mapping=np.asarray(self.owner.global2merged_mask_id)
        live={tuple(np.flatnonzero(mapping==row).tolist()) for row in groups}
        for source in list(self.pending):
            if source not in live:self.pending.pop(source);self.counter['lineage_reset']+=1
        boxes={row:np.array([np.min([self.records[i]['box'][0] for i in ids],axis=0)-self.opt.bbox_margin_m,
                             np.max([self.records[i]['box'][1] for i in ids],axis=0)+self.opt.bbox_margin_m]) for row,ids in groups.items()}
        self.index.rebuild(boxes)
        query=frustum_box(self.K,pose,depth.shape,self.cam['depth_near'],self.cam['depth_far'])
        rows=self.index.query(query);self.counter['bbox_candidates']+=len(rows)
        rows.sort(key=lambda r:(self.last_test.get(min(np.flatnonzero(mapping==r)),-1),r))
        from .instance_retirement import DepthContext
        context=None;current=None;logs=[];processed=0
        for row in rows:
            if processed>=self.opt.max_candidates_per_frame:break
            source=tuple(np.flatnonzero(mapping==row).tolist())
            ids=[i for i in groups[row] if frame-self.records[i]['frame']>=self.opt.min_age_frames]
            if not ids:continue
            ids.sort(key=lambda i:np.linalg.norm(self.records[i]['camera']-pose[:3,3]));ids=ids[:self.opt.retrieval_views]
            if source not in self.pending:
                if len(self.pending)>=self.opt.max_pending:self.counter['pending_budget_skips']+=1;continue
                # Archive whole owned IG volume support for geometry veto. It
                # is not used as DINO evidence or labelled observed surface.
                owned=array(self.owner.maskGraph.instance_dict[row].mask_voxel_coords)
                if not len(owned):continue
                ref=np.concatenate([self.records[i]['points'] for i in ids])
                cells=np.floor(ref/self.voxel).astype(np.int64);_,idx=np.unique(cells,axis=0,return_index=True)
                self.pending[source]=dict(ids=tuple(ids),points=ref[idx],created=frame,reported=False,
                    owned=owned[np.linspace(0,len(owned)-1,min(len(owned),self.opt.max_reference_points)).astype(int)].copy())
            st=self.pending[source]
            # Reference is frozen. Querying new views does not grow denominator
            # or overwrite the old appearance while disappearance is pending.
            ids=list(st['ids']);processed+=1;self.last_test[source[0]]=frame
            if context is None:context=DepthContext(depth,pose,self.K,self.cfg,self.cam['depth_near'],self.cam['depth_far'],2*self.voxel)
            codes=context.classify(st['points']);owned_codes=context.classify(st['owned'])
            if not np.any(codes==2):
                low=np.zeros(len(codes),bool);sim_details=[]
            else:
                if current is None:
                    current=self.features(frame,rgb_bytes(rgb));xy=current['uv']
                    rays=np.c_[xy,np.ones(len(xy))]@np.linalg.inv(self.K).T
                    current['rays_world']=rays@pose[:3,:3].T
                scores=[];all_points=[];sim_details=[]
                for sid in ids:
                    snap=self.records.get(sid)
                    if snap is None:raise RuntimeError('Frozen snapshot was pruned')
                    old=self.frames.get(snap['frame']);box=snap['box']+np.array([[-self.opt.bbox_margin_m]*3,[self.opt.bbox_margin_m]*3])
                    # Ray-AABB test includes the background behind an old object;
                    # selecting only current 3D surface points would miss absence.
                    rays=current['rays_world'];safe=np.where(np.abs(rays)<1e-10,1e-10,rays)
                    t0=(box[0]-pose[:3,3])/safe;t1=(box[1]-pose[:3,3])/safe
                    a=np.minimum(t0,t1).max(1);b=np.maximum(t0,t1).min(1)
                    selected=np.flatnonzero((b>=np.maximum(a,self.cam['depth_near']))&(a<=self.cam['depth_far']))
                    if not len(selected):best=np.full(len(snap['tokens']),np.nan)
                    else:
                        q=torch.from_numpy(old['features'][snap['tokens']].astype(np.float32)).to(self.owner.device)
                        k=torch.from_numpy(current['features'][selected].astype(np.float32)).to(self.owner.device)
                        with torch.inference_mode():best=(F.normalize(q,dim=1)@F.normalize(k,dim=1).T).max(1).values.cpu().numpy()
                    scores.append(best);all_points.append(snap['points'])
                    sim_details.append(dict(snapshot=sid,frame=snap['frame'],query_tokens=len(best),current_tokens=len(selected),
                                            mean_similarity=float(np.nanmean(best)) if np.isfinite(best).any() else None))
                rawpoints=np.concatenate(all_points);rawscores=np.concatenate(scores)
                cells=np.floor(rawpoints/self.voxel).astype(np.int64);_,idx,inv=np.unique(cells,axis=0,return_index=True,return_inverse=True)
                best=np.full(len(idx),-np.inf);np.maximum.at(best,inv,np.where(np.isfinite(rawscores),rawscores,-np.inf))
                low=np.isfinite(best)&(best<self.opt.semantic_absence_below)
                assert np.allclose(st['points'],rawpoints[idx])
            accept,stats=update_evidence(st,st['points'],codes,low,frame,pose,self.opt)
            # Small clean snapshot alone cannot authorize a big contaminated IG
            # track. All sampled old IG geometry must pass the broad depth veto.
            stats['owned_support_fraction']=float((owned_codes==1).mean())
            stats['owned_free_fraction']=float((owned_codes==2).mean())
            if stats['owned_support_fraction']>self.opt.max_support_fraction:stats['reasons'].append('whole_IG_track_has_support')
            if stats['owned_free_fraction']<self.opt.min_confirmed_fraction:stats['reasons'].append('whole_IG_track_not_observed_free')
            accept=accept and not stats['reasons']
            event=dict(frame=frame,row=int(row),raw_ids=list(source),status='would_remove' if accept else 'pending',**stats,views=sim_details)
            logs.append(event)
            if accept and not st['reported']:
                st['reported']=True;self.counter['unique_would_remove']+=1
                np.savez_compressed(self.out/f'would_remove_{frame:06d}_{source[0]:06d}.npz',points=st['points'],owned_geometry=st['owned'],raw_ids=source,votes=st['votes'],codes=codes)
            self.counter['would_remove_frames']+=int(accept)
        if logs:
            with (self.out/'events.jsonl').open('a') as f:
                for log in logs:f.write(json.dumps(log)+'\n')
        self.counter['tested_candidates']+=processed;self.counter['frames']+=1;self.seconds+=time.perf_counter()-start

    def save(self):
        report=dict(method='IG AABB-retrieved DINO snapshot disappearance v1',mode='read_only',GT_used=False,identity_association=False,
            options=asdict(self.opt),model_weight_sha256=self.encoder.sha256,model='DINOv2, not OASIS-Map DINOv3',
            counters=dict(self.counter),resident_snapshots=len(self.records),pending=len(self.pending),
            feature_extractions=self.encoder.calls,feature_seconds=self.encoder.seconds,total_seconds=self.seconds,
            cache_peak_bytes=self.frames.peak,cache_hits=self.frames.hits,cache_misses=self.frames.misses,
            reference='mask-interior DINO patch-centre depth samples; sampled full IG volume support is a separate veto, not surface coverage',
            threshold_status='initial engineering thresholds, not calibrated on GT',
            retrieval='world AABB overlap with current camera frustum; no global appearance/identity search')
        (self.out/'summary.json').write_text(json.dumps(report,indent=2));return report
