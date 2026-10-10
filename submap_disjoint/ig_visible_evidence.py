"""IG depth-only contradiction evidence; metres and camera-to-world poses.

One projected sample per pixel. Visible means valid, non-edge, non-occluded
current depth. Candidates and confirmed local regions use the SAME denominator.
Two independent observations of the SAME surface cell authorize local action.
"""
from dataclasses import dataclass
import numpy as np
from .contradiction import UNKNOWN, SUPPORT, FREE, OCCLUDED, validate_camera
from .change_frontend import interior


@dataclass(frozen=True)
class VisibleOptions:
    min_visible: int = 30
    conflict_ratio: float = .30
    depth_abs_m: float = .05
    depth_relative: float = .01
    edge_m: float = .08
    view_distance_m: float = .10
    view_angle_deg: float = 10.
    evidence_max_age: int = 64

    def __post_init__(self):
        if self.min_visible < 1 or self.evidence_max_age < 1: raise ValueError('Invalid counts')
        if not 0 < self.conflict_ratio <= 1: raise ValueError('Invalid ratio')
        for k in ('depth_abs_m','edge_m','view_distance_m','view_angle_deg'):
            if not np.isfinite(getattr(self,k)) or getattr(self,k)<=0: raise ValueError(k)
        if not np.isfinite(self.depth_relative) or self.depth_relative<0: raise ValueError('depth_relative')


def project_evidence(points, depth, K, pose, near, far, opt):
    validate_camera(K, pose)
    p=np.asarray(points); h,w=depth.shape
    codes=np.zeros(len(p),np.uint8)
    cam=(p-pose[:3,3]) @ pose[:3,:3]; z=cam[:,2]
    idx=np.flatnonzero(np.isfinite(cam).all(1)&(z>near)&(z<far))
    u=np.rint(K[0,0]*cam[idx,0]/z[idx]+K[0,2]).astype(int)
    v=np.rint(K[1,1]*cam[idx,1]/z[idx]+K[1,2]).astype(int)
    inside=(u>=0)&(u<w)&(v>=0)&(v<h)
    idx,u,v=idx[inside],u[inside],v[inside]
    projected=len(idx)
    # Only the closest reference surface contributes per pixel, independent
    # of point-array order. A denser cloud cannot inflate pixel evidence.
    order=np.lexsort((idx,z[idx],v*w+u))
    _,first=np.unique((v*w+u)[order],return_index=True); pick=order[first]
    idx,u,v=idx[pick],u[pick],v[pick]
    d=depth[v,u]
    valid=np.isfinite(depth)&(depth>near)&(depth<far)
    safe=interior(depth,valid,opt.edge_m,0)
    ok=safe[v,u]
    tau=opt.depth_abs_m+opt.depth_relative*np.maximum(d,z[idx])
    codes[idx[ok&(np.abs(d-z[idx])<=tau)]]=SUPPORT
    codes[idx[ok&(d-z[idx]>tau)]]=FREE
    codes[idx[ok&(d-z[idx]<-tau)]]=OCCLUDED
    visible=(codes==SUPPORT)|(codes==FREE)
    count=int(visible.sum());free=int((codes==FREE).sum())
    ratio=free/max(count,1)
    reason='not_projected' if not projected else ('insufficient_visible' if count<opt.min_visible else ('low_visible_conflict' if ratio<opt.conflict_ratio else None))
    return codes,dict(projected_cells=projected,unique_pixels=len(idx),
        invalid_or_edge_pixels=int((~ok).sum()),occluded_cells=int((codes==OCCLUDED).sum()),
        visible_cells=count,free_cells=free,visible_conflict_ratio=ratio,
        candidate=reason is None,candidate_failures=[] if reason is None else [reason])


class TwoViewEvidence:
    """Support retracts old FREE. Unknown/occlusion neither confirm nor refute."""
    def __init__(self,n,opt):
        self.opt=opt
        self.votes=np.zeros(n,np.uint8)
        self.last=np.full(n,-1,np.int64)
        self.anchor=np.zeros((n,4,4),float)

    def update(self,tick,pose,codes):
        expired=(self.last>=0)&(tick-self.last>=self.opt.evidence_max_age)
        self.votes[expired|(codes==SUPPORT)]=0
        ids=np.flatnonzero(codes==FREE)
        old=self.votes[ids]>0
        moved=np.linalg.norm(self.anchor[ids,:3,3]-pose[:3,3],axis=1)>=self.opt.view_distance_m
        traces=np.einsum('nij,ij->n',self.anchor[ids,:3,:3],pose[:3,:3])
        angle=np.degrees(np.arccos(np.clip((traces-1)/2,-1,1)))
        independent=~old|moved|(angle>=self.opt.view_angle_deg)
        take=ids[independent]
        self.votes[take]=np.minimum(2,self.votes[take].astype(int)+1)
        self.anchor[take]=pose
        self.last[ids]=tick
        # Deletion never uses historical FREE alone: this observation must also
        # report FREE, and unseen surface cells are excluded from the ratio.
        return (self.votes>=2)&(codes==FREE)
