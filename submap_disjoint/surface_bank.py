"""Experimental causal surface-existence memory independent of instance slots.

Stored samples are measured surfaces (one per grid cell), not occupied cubes.
Past independent depth support establishes existence; later free-space rays
establish local absence. Never infer absence from unknown or occluded pixels.
No session boundary, GT, appearance model, or scene mutation is used here.
"""
import numpy as np
from .readonly_detection_core import classify_depth, independent_observation, SUPPORT, CONTRADICTION, rgb_to_lab, normalize_rgb


def voxel_keys(points, voxel_size):
    q = np.floor(np.asarray(points)/voxel_size).astype(np.int64) + (1 << 20)
    if np.any(q < 0) or np.any(q >= (1 << 21)):
        raise ValueError('Coordinates outside signed 21-bit voxel range')
    return (q[:,0] << 42) | (q[:,1] << 21) | q[:,2]


def independent_rays(points, old_centers, new_center, old_frames, frame, cfg):
    """An optical-axis rotation is not a new ray to a fixed world point."""
    old=points-old_centers;new=points-new_center
    denominator=np.linalg.norm(old,axis=1)*np.linalg.norm(new,axis=1)
    cosine=np.sum(old*new,axis=1)/np.maximum(denominator,1e-12)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    shift=np.linalg.norm(old_centers-new_center,axis=1)
    return ((frame-old_frames>=cfg['min_observation_gap'])
            &(angle>=cfg['min_rotation_deg'])&(shift>=cfg['min_translation_m']))


class SurfaceBank:
    def __init__(self, voxel_size, cfg, seed_stride=2, require_color=False, view_gate='camera'):
        self.voxel_size, self.cfg, self.seed_stride = voxel_size, cfg, seed_stride
        self.require_color = require_color
        if view_gate not in ('camera','ray'):raise ValueError(view_gate)
        self.view_gate=view_gate
        self.points = np.empty((0,3),np.float32)
        self.colors = np.empty((0,3),np.float32)
        self.keys = np.empty(0,np.int64)
        self.sorted_keys = self.keys.copy(); self.views = []
        self.support = np.empty(0,np.uint32)
        self.streak = np.empty(0,np.uint32)
        self.last_vote = np.empty(0,np.int32)
        self.last_conflict_frame = np.empty(0,np.int64)
        self.created_frame = np.empty(0,np.int64)
        self.event_frames=[]; self.event_indices=[]; self.event_states=[]
        self.last_frame = -1

    @property
    def active(self):
        return self.streak >= self.cfg['confirm_observations']

    def step(self, frame, pose, k, depth, interior, near, far, rgb=None):
        if frame <= self.last_frame: raise ValueError('Input must be causal and unique')
        self.last_frame=frame
        if self.require_color and rgb is None: raise ValueError('RGB evidence required')
        if rgb is not None:
            import cv2
            rgb=cv2.GaussianBlur(normalize_rgb(rgb),(3,3),0)
        n=len(self.points); before=self.active
        stats=dict(frame=int(frame),old_points=n,tested=0,free=0,support=0,occluded=0)
        view_index=len(self.views)
        if n:
            camera=(self.points-pose[:3,3]) @ pose[:3,:3]
            z=camera[:,2]
            ids=np.flatnonzero(np.isfinite(camera).all(1)&(z>near)&(z<far))
            p=camera[ids]@k.T
            uv=np.rint(p[:,:2]/p[:,2:3]).astype(np.int64)
            h,w=depth.shape
            inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
            ids,uv=ids[inside],uv[inside]
            y,x=uv[:,1],uv[:,0]
            # All old surfaces pierced by the measured ray are testable. An old
            # foreground surface is not allowed to hide another old surface.
            codes=classify_depth(depth[y,x],z[ids],interior[y,x],
                self.cfg['depth_epsilon_m'],self.cfg['depth_relative_epsilon'])
            if self.require_color:
                # Robust exposure offset/threshold use only currently depth-
                # consistent rays. Color is corroboration, never a free-space
                # observation by itself. Lack of color change means abstain.
                current=rgb[y,x];reference=self.colors[ids]
                anchors=np.flatnonzero(codes==SUPPORT)
                if len(anchors):
                    pixels=y[anchors]*w+x[anchors]
                    _,unique=np.unique(pixels,return_index=True);anchors=anchors[unique]
                    offset=np.median(reference[anchors]-current[anchors],axis=0)
                else:offset=np.zeros(3,np.float32)
                current=np.clip(current+offset,0,1)
                residual=np.linalg.norm(rgb_to_lab(current)-rgb_to_lab(reference),axis=1) if len(ids) else np.empty(0)
                if len(anchors):
                    median=np.median(residual[anchors]);mad=np.median(np.abs(residual[anchors]-median))
                    threshold=np.clip(median+self.cfg['rgb_mad_multiplier']*1.4826*mad,
                                      self.cfg['rgb_threshold_min'],self.cfg['rgb_threshold_max'])
                else:threshold=self.cfg['rgb_threshold_max']
                raw_free=codes==CONTRADICTION
                codes[raw_free&(residual<=threshold)]=0
                stats.update(color_threshold=float(threshold),depth_free_before_color=int(raw_free.sum()),
                             depth_free_after_color=int((codes==CONTRADICTION).sum()))
            good=(codes==SUPPORT)|(codes==CONTRADICTION)
            if self.view_gate=='camera':
                independent=np.array([independent_observation(pose,oldpose,frame,oldframe,self.cfg)
                                      for oldframe,oldpose in self.views],bool)
                accepted=good & independent[self.last_vote[ids]]
            else:
                centers=np.array([p[:3,3] for _,p in self.views])
                frames=np.array([f for f,_ in self.views])
                old=self.last_vote[ids]
                accepted=good&independent_rays(self.points[ids],centers[old],pose[:3,3],frames[old],frame,self.cfg)
            support_ids=ids[accepted & (codes==SUPPORT)]
            free_ids=ids[accepted & (codes==CONTRADICTION)]
            free_ids=free_ids[self.support[free_ids]>=self.cfg['confirm_observations']]
            self.support[support_ids]+=1
            self.streak[support_ids]=0
            self.streak[free_ids]+=1
            self.last_vote[support_ids]=view_index
            self.last_vote[free_ids]=view_index
            self.last_conflict_frame[free_ids]=frame
            after=self.active; changed=np.flatnonzero(before!=after)
            if len(changed):
                self.event_frames.append(np.full(len(changed),frame,np.int64))
                self.event_indices.append(changed)
                self.event_states.append(after[changed])
            stats.update(tested=int(good.sum()),free=len(free_ids),support=len(support_ids),
                         occluded=int((codes==3).sum()),confirmed_points=int(after.sum()))
        # Seed only measured, smooth depth surfaces; no completion or AABB fill.
        sampled=interior.copy()
        yy,xx=np.indices(depth.shape)
        sampled &= (yy%self.seed_stride==0)&(xx%self.seed_stride==0)
        y,x=np.nonzero(sampled)
        rays=np.column_stack((x,y,np.ones(len(x)))) @ np.linalg.inv(k).T
        camera=rays*depth[y,x,None]
        points=camera @ pose[:3,:3].T + pose[:3,3]
        colors=rgb[y,x] if rgb is not None else np.zeros((len(y),3),np.float32)
        keys=voxel_keys(points,self.voxel_size)
        keys,first=np.unique(keys,return_index=True);points=points[first];colors=colors[first]
        if len(self.sorted_keys):
            index=np.searchsorted(self.sorted_keys,keys)
            exists=(index<len(self.sorted_keys))&(self.sorted_keys[np.minimum(index,len(self.sorted_keys)-1)]==keys)
            keys,points,colors=keys[~exists],points[~exists],colors[~exists]
        m=len(points)
        if m:
            self.points=np.concatenate((self.points,points.astype(np.float32)))
            self.colors=np.concatenate((self.colors,colors.astype(np.float32)))
            self.keys=np.concatenate((self.keys,keys));self.sorted_keys=np.sort(self.keys)
            self.support=np.concatenate((self.support,np.ones(m,np.uint32)))
            self.streak=np.concatenate((self.streak,np.zeros(m,np.uint32)))
            self.last_vote=np.concatenate((self.last_vote,np.full(m,view_index,np.int32)))
            self.created_frame=np.concatenate((self.created_frame,np.full(m,frame,np.int64)))
            self.last_conflict_frame=np.concatenate((self.last_conflict_frame,np.full(m,-1,np.int64)))
        self.views.append((frame,np.asarray(pose).copy()))
        stats.update(new_points=m,total_points=len(self.points),confirmed_points=int(self.active.sum()))
        return stats

    def arrays(self):
        cat=lambda data,dtype:np.concatenate(data).astype(dtype) if data else np.empty(0,dtype)
        return dict(points=self.points,colors=self.colors,keys=self.keys,support=self.support,streak=self.streak,
            created_frame=self.created_frame,last_conflict_frame=self.last_conflict_frame,
            active=self.active,event_frames=cat(self.event_frames,np.int64),
            event_indices=cat(self.event_indices,np.int64),event_states=cat(self.event_states,bool))
