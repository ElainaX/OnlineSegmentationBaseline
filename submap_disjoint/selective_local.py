"""Select instance-local reconstruction using held-out RGB-D, never GT.

Points and depth are metres; poses are camera-to-world. Unobserved and occluded
surface points cannot count as precision failures. Identity/order/confidence
remain owned by IG; this module chooses geometry for existing valid instances.
"""
import numpy as np
from scipy.spatial import cKDTree


def uniform_points(xyz, spacing=.025):
    xyz = np.asarray(xyz, dtype=np.float64)
    if len(xyz) == 0:
        return xyz.reshape(0, 3)
    _, first = np.unique(np.floor(xyz / spacing).astype(np.int64), axis=0,
                         return_index=True)
    return xyz[np.sort(first)]


def split_views(views, maximum_validation=3):
    """Deterministic distinct timestamps; hold-out frames never enter training."""
    by_frame = {int(p['frame']): (p, m) for p, m in views}
    views = [by_frame[f] for f in sorted(by_frame)]
    if len(views) < 8:
        return views, []
    selected = set(np.linspace(1, len(views)-2, maximum_validation,
                              dtype=int).tolist())
    return ([v for i, v in enumerate(views) if i not in selected],
            [v for i, v in enumerate(views) if i in selected])


def score_surface(xyz, views, K, tolerance_m=.025, near=.1, far=5.):
    """Per-view surface F1 at a common spatial sampling density.

    Recall measures coverage of observed mask depth; precision checks visible
    model points against depth and mask. This is a noisy observation proxy,
    not a guarantee of complete objects or improved GT AP.
    """
    xyz = uniform_points(xyz)
    scores = []
    for packet, mask in views:
        depth = packet['depth']; pose = packet['pose']
        valid = np.isfinite(depth) & (depth > near) & (depth < far)
        vv, uu = np.nonzero(valid & mask)
        if len(uu) == 0 or len(xyz) == 0:
            scores.append(0.); continue
        z = depth[vv, uu]
        observed = np.column_stack(((uu-K[0,2])*z/K[0,0],
                                    (vv-K[1,2])*z/K[1,1], z))
        observed = uniform_points(observed @ pose[:3,:3].T + pose[:3,3])
        camera = (xyz-pose[:3,3]) @ pose[:3,:3]
        keep = np.isfinite(camera).all(axis=1) & (camera[:,2] > near) & (camera[:,2] < far)
        indices = np.flatnonzero(keep); camera = camera[keep]
        u = np.rint(K[0,0]*camera[:,0]/camera[:,2]+K[0,2]).astype(int)
        v = np.rint(K[1,1]*camera[:,1]/camera[:,2]+K[1,2]).astype(int)
        inside = (u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
        u=u[inside]; v=v[inside]; indices=indices[inside]; z=camera[inside,2]
        # A farther model surface is occluded by the measured depth.
        visible = valid[v,u] & (z <= depth[v,u] + tolerance_m)
        indices=indices[visible]; u=u[visible]; v=v[visible]; z=z[visible]
        if len(indices) == 0:
            scores.append(0.); continue
        precision = np.mean(mask[v,u] & (np.abs(z-depth[v,u]) <= tolerance_m))
        recall = np.mean(cKDTree(xyz[indices]).query(observed, workers=1)[0] <= tolerance_m)
        scores.append(float(2*precision*recall/max(precision+recall,1e-12)))
    return scores


def improves(candidate, reference, margin=.02):
    a=np.asarray(candidate); b=np.asarray(reference)
    return (len(a)>=2 and len(a)==len(b) and np.isfinite(a).all()
            and np.isfinite(b).all() and float(np.mean(a-b))>=margin
            and np.count_nonzero(a>b)>=2)


def choose_resolution(base, coarse, fine=None):
    """Coarser wins ties. Fine must beat both original IG and local coarse."""
    choice='ig'
    if improves(coarse, base): choice='local25'
    if fine is not None and improves(fine, base) and improves(fine, coarse):
        choice='local20'
    return choice


def assemble_outputs(base_xyz, base_rgb, base_masks, replacements):
    """Keep global coordinates and non-target masks; append target surfaces.

    No refiltering, instance creation, score resorting or cross-object point
    averaging. The caller copies feature/class/score arrays unchanged.
    """
    if not replacements:
        return base_xyz, base_rgb, base_masks
    columns=base_masks.shape[1]; total=len(base_xyz)+sum(len(v[0]) for v in replacements.values())
    masks=np.zeros((total,columns),bool); masks[:len(base_xyz)]=base_masks
    xyz=[base_xyz]; rgb=[base_rgb]; offset=len(base_xyz)
    for col,(points,colors) in sorted(replacements.items()):
        if not 0<=col<columns or len(points)==0 or len(points)!=len(colors):
            raise ValueError('Invalid replacement')
        masks[:len(base_xyz),col]=False
        masks[offset:offset+len(points),col]=True
        xyz.append(points);rgb.append(colors);offset+=len(points)
    return np.concatenate(xyz),np.concatenate(rgb),masks


def observation_votes(xyz, views, K, tolerance_m=.025, near=.1, far=5.):
    """Count direct supporting/opposing observations; unknown is neither."""
    xyz=np.asarray(xyz);positive=np.zeros(len(xyz),np.int32);negative=positive.copy()
    seen=set()
    for packet,mask in views:
        if packet['frame'] in seen:continue
        seen.add(packet['frame']);depth=packet['depth'];pose=packet['pose']
        q=(xyz-pose[:3,3])@pose[:3,:3]
        ids=np.flatnonzero(np.isfinite(q).all(axis=1)&(q[:,2]>near)&(q[:,2]<far))
        q=q[ids];u=np.rint(K[0,0]*q[:,0]/q[:,2]+K[0,2]).astype(int)
        v=np.rint(K[1,1]*q[:,1]/q[:,2]+K[1,2]).astype(int)
        inside=(u>=0)&(u<depth.shape[1])&(v>=0)&(v<depth.shape[0])
        ids=ids[inside];q=q[inside];u=u[inside];v=v[inside];d=depth[v,u]
        visible=np.isfinite(d)&(d>near)&(d<far)&(q[:,2]<=d+tolerance_m)
        supported=mask[v,u]&(np.abs(q[:,2]-d)<=tolerance_m)
        positive[ids[visible&supported]]+=1
        negative[ids[visible&~supported]]+=1
    return positive,negative


def refine_mask(base_mask, occupied_other, near_local, positive, negative):
    """Change only consistently evidenced points; do not steal other labels."""
    result=base_mask.copy()
    remove=base_mask&(negative>=3)&(positive==0)
    add=(~base_mask)&(~occupied_other)&near_local&(positive>=3)&(negative==0)
    result[remove]=False;result[add]=True
    return result,dict(removed_points=int(remove.sum()),added_points=int(add.sum()))


def refine_instances(xyz, original, ids, tracks, packet_for_frame, K):
    """Complete scene mask refinement; independent of TSDF and GPU state."""
    result=original.copy();label_count=original.sum(axis=1);changes=[]
    for col,mid in enumerate(ids):
        observations={}
        for frame,raw in tracks[mid]['observations']:
            observations.setdefault(frame,[]).append(raw)
        if len(observations)<3:continue
        views=[]
        for frame,raw in sorted(observations.items()):
            packet=packet_for_frame(frame)
            views.append((packet,np.isin(packet['labels'],raw)))
        target=xyz[original[:,col]]
        if not len(target):continue
        roi=np.all((xyz>=target.min(axis=0)-.05)&(xyz<=target.max(axis=0)+.05),axis=1)
        idx=np.flatnonzero(roi);pos,neg=observation_votes(xyz[idx],views,K)
        other=(label_count[idx]-original[idx,col])>0
        refined,audit=refine_mask(original[idx,col],other,np.ones(len(idx),bool),pos,neg)
        if not refined.any():
            audit=dict(removed_points=0,added_points=0,empty_fallback=True)
            refined=original[idx,col]
        result[idx,col]=refined;changes.append(dict(track_id=mid,views=len(views),**audit))
    ambiguous=(label_count==0)&(result.sum(axis=1)>1);result[ambiguous]=False
    return result,dict(changes=changes,ambiguous_additions_abstained=int(ambiguous.sum()),
        removed_memberships=int(np.count_nonzero(original&~result)),
        added_memberships=int(np.count_nonzero(~original&result)))
