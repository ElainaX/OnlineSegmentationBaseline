"""Experimental causal surface-existence memory independent of instance slots.

Stored samples are measured surfaces (one per grid cell), not occupied cubes.
Past independent depth support establishes existence; later free-space rays
establish local absence. Never infer absence from unknown or occluded pixels.
No session boundary, GT, appearance model, or scene mutation is used here.
"""
import numpy as np
from .readonly_detection_core import classify_depth, independent_observation, SUPPORT, CONTRADICTION


def voxel_keys(points, voxel_size):
    q = np.floor(np.asarray(points)/voxel_size).astype(np.int64) + (1 << 20)
    if np.any(q < 0) or np.any(q >= (1 << 21)):
        raise ValueError('Coordinates outside signed 21-bit voxel range')
    return (q[:,0] << 42) | (q[:,1] << 21) | q[:,2]


class SurfaceBank:
    def __init__(self, voxel_size, cfg, seed_stride=2):
        self.voxel_size, self.cfg, self.seed_stride = voxel_size, cfg, seed_stride
        self.points = np.empty((0,3),np.float32)
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

    def step(self, frame, pose, k, depth, interior, near, far):
        if frame <= self.last_frame: raise ValueError('Input must be causal and unique')
        self.last_frame=frame
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
            good=(codes==SUPPORT)|(codes==CONTRADICTION)
            independent=np.array([independent_observation(pose,oldpose,frame,oldframe,self.cfg)
                                  for oldframe,oldpose in self.views],bool)
            accepted=good & independent[self.last_vote[ids]]
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
        keys=voxel_keys(points,self.voxel_size)
        keys,first=np.unique(keys,return_index=True);points=points[first]
        if len(self.sorted_keys):
            index=np.searchsorted(self.sorted_keys,keys)
            exists=(index<len(self.sorted_keys))&(self.sorted_keys[np.minimum(index,len(self.sorted_keys)-1)]==keys)
            keys,points=keys[~exists],points[~exists]
        m=len(points)
        if m:
            self.points=np.concatenate((self.points,points.astype(np.float32)))
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
        return dict(points=self.points,keys=self.keys,support=self.support,streak=self.streak,
            created_frame=self.created_frame,last_conflict_frame=self.last_conflict_frame,
            active=self.active,event_frames=cat(self.event_frames,np.int64),
            event_indices=cat(self.event_indices,np.int64),event_states=cat(self.event_states,bool))
