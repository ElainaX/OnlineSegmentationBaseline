"""Offline GAP-v3: observed removed/moved old surfaces and archived physical erasures.

Does not rerun mapping or change AP. CPU only. Lengths are metres; scores 0..1.
Current evaluation requires full incident-edge .evidence.npz archives; missing
evidence is an error. Historical two-scene runner has been removed.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, hashlib, time, traceback
from pathlib import Path, PurePosixPath
import numpy as np
import cv2
from scipy.spatial import cKDTree

V = .025
TOL = .05
ACTION_RADIUS = np.sqrt(3)*V
MIN_GAP = 5
MIN_TRANSLATION = .02
MIN_ROTATION = np.deg2rad(3)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text())

def nearest(a,b):
    if not len(b): return np.full(len(a),np.inf),np.full(len(a),-1,int)
    return cKDTree(b).query(a)

def cloud(p):
    import open3d as o3d
    a=np.asarray(o3d.io.read_point_cloud(str(p)).points)
    assert len(a) and np.isfinite(a).all()
    return a

def reference(p):
    with open(p) as f:
        assert f.readline().strip()=='ply'
        assert f.readline().strip()=='format ascii 1.0'
        props=[]; vertex=False
        for line in f:
            x=line.split()
            if x[:2]==['element','vertex']: n=int(x[2]);vertex=True
            elif x[:1]==['element']:vertex=False
            elif vertex and x[0]=='property':props.append(x[-1])
            if x[0]=='end_header':break
        a=np.loadtxt(f,max_rows=n)
    xyz=a[:,[props.index(x) for x in ['x','y','z']]]
    labels=a[:,props.index('objectId')].astype(int)
    return xyz,labels

def atoms(xyz,labels=None):
    keys,first,inv=np.unique(np.floor(xyz/V).astype(np.int64),axis=0,return_index=True,return_inverse=True)
    n=np.bincount(inv)
    pts=np.column_stack([np.bincount(inv,weights=xyz[:,i])/n for i in range(3)])
    # One spatial atom, one label: closest original point to atom centroid.
    if labels is not None: lab=labels[cKDTree(xyz).query(pts)[1]]
    else:lab=None
    return pts,keys,lab

def project_evidence(points,depth,pose,K):
    q=(points-pose[:3,3])@pose[:3,:3]
    z=q[:,2]; safe=np.where(z>0,z,1.)
    u=np.rint(K[0,0]*q[:,0]/safe+K[0,2]).astype(int)
    v=np.rint(K[1,1]*q[:,1]/safe+K[1,2]).astype(int)
    inside=(z>.1)&(z<10)&(u>=1)&(u<depth.shape[1]-1)&(v>=1)&(v<depth.shape[0]-1)
    ids=np.flatnonzero(inside);seen=np.zeros(len(points),bool);free=seen.copy()
    if not len(ids):return seen,free
    patch=np.stack([depth[v[ids]+dy,u[ids]+dx] for dy in [-1,0,1] for dx in [-1,0,1]])
    valid=(patch>.1).all(axis=0)&(patch<10).all(axis=0)
    # Reference surface visibility uses central depth; free-space needs all 9
    # neighbours behind the old point, guarding holes and foreground edges.
    seen[ids]=valid&(np.abs(patch[4]-z[ids])<=TOL)
    free[ids]=valid&((patch.min(axis=0)-z[ids])>np.maximum(TOL,.01*patch[4]))
    return seen,free

def observations(points,frames,cut):
    K=np.loadtxt(frames/'intrinsic/intrinsic_depth.txt')[:3,:3]
    posepaths=sorted((frames/'pose').glob('*.txt'),key=lambda p:int(p.stem))
    poses={int(p.stem):np.loadtxt(p) for p in posepaths}
    visible=np.zeros(len(points),bool);free2=visible.copy();first=np.full(len(points),-1,int)
    for frame,pose in poses.items():
        if not np.isfinite(pose).all():continue
        depth=cv2.imread(str(frames/'depth'/f'{frame}.png'),cv2.IMREAD_UNCHANGED)
        assert depth is not None and depth.ndim==2
        seen,free=project_evidence(points,depth.astype(float)/1000.,pose,K)
        if frame<cut:visible|=seen;continue
        prior=np.unique(first[free&(first>=0)&~free2])
        for f0 in prior:
            if frame-f0<MIN_GAP:continue
            p0=poses[int(f0)]
            angle=np.arccos(np.clip((np.trace(p0[:3,:3].T@pose[:3,:3])-1)/2,-1,1))
            if np.linalg.norm(p0[:3,3]-pose[:3,3])>=MIN_TRANSLATION or angle>=MIN_ROTATION:
                free2|=free&(first==f0)
        first[free&(first<0)]=frame
    return visible,free2

def archived_surface(z):
    """Actual pre-erasure zero-crossings; occupied volume nodes are NOT surfaces."""
    xyz=z['xyz'];s=np.asarray(z['tsdf']).reshape(-1);w=np.asarray(z['weight']).reshape(-1)
    good=np.isfinite(xyz).all(axis=1)&np.isfinite(s)&(w>0)
    erased=np.asarray(z['erased'],bool)[good] if 'erased' in z else np.ones(int(good.sum()),bool)
    xyz=xyz[good];s=s[good];k=np.rint(xyz/V).astype(np.int64)
    assert np.max(np.abs(xyz-k*V),initial=0)<1e-5
    lookup={tuple(v):i for i,v in enumerate(k)};out=[]
    used=set()
    for i,ki in enumerate(k):
        if erased[i] and abs(s[i])<1e-8:out.append(xyz[i]);used.add(i)
        for ax in range(3):
            other=ki.copy();other[ax]+=1;j=lookup.get(tuple(other))
            if j is None or s[i]*s[j]>=0 or not (erased[i] or erased[j]):continue
            t=s[i]/(s[i]-s[j]);out.append(xyz[i]+t*(xyz[j]-xyz[i]));used.update([i,j])
    pts=np.asarray(out,float).reshape(-1,3)
    if len(pts):pts=atoms(pts)[0]
    return pts,dict(archived_valid_nodes=len(xyz),zero_crossing_surface_atoms=len(pts),nodes_supporting_zero_crossings=len(used))

def set_score(target,after,before,extras=0):
    # Union across all actions. A later valid erasure can establish the final
    # outcome at an earlier-erased location, but never produces duplicate credit.
    a=after|before
    tp=int((after&target).sum());fp=int((a&~target).sum()+(before&target&~after).sum())+int(extras)
    fn=int(target.sum())-tp;den=tp+fp+fn
    return dict(D=tp/den if den else 0.,TP=tp,FP=fp,FN=fn,target_atoms=int(target.sum()),union_atoms=den,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None)

def relocated_archive(correction_root, archive):
    """Preserve candidate subdirectories: multiple actions can share a frame ID."""
    parts=PurePosixPath(str(archive).replace('\\','/')).parts
    if 'correction' not in parts:
        raise ValueError('Archive path must contain the correction directory')
    i=len(parts)-1-list(reversed(parts)).index('correction')
    tail=parts[i+1:]
    if not tail or tail[0] not in ('actions','erasure_archive') or any(x in ('.','..') for x in tail):
        raise ValueError('Unexpected correction archive layout')
    root=Path(correction_root).resolve()
    path=root.joinpath(*tail).resolve()
    if not path.is_relative_to(root):raise ValueError('Archive escapes correction root')
    return path

def evaluate_method(root,ref,keys,labels,target,removed,moved,neutral,cut,final,archive_root=None):
    summary=read(root/'correction/summary.json') if (root/'correction/summary.json').exists() else {'actions':[]}
    before=np.zeros(len(ref),bool);after=before.copy();extra=set();audits=[];hashes={}
    absent=nearest(ref,final)[0]>TOL
    for action in summary['actions']:
        if not action.get('archive'):continue
        p=Path(action['archive'])
        if archive_root is not None: p=relocated_archive(archive_root,p)
        hashes[str(p)]=digest(p)
        ep=p.with_suffix('.evidence.npz')
        assert ep.is_file(),f'Missing full incident-edge evidence: {ep}'
        hashes[str(ep)]=digest(ep)
        z=np.load(ep);surf,audit=archived_surface(z);frame=int(action['frame'])
        audit.update(frame=frame,archive=str(p))
        if not len(surf):audits.append(audit);continue
        gone=surf[nearest(surf,final)[0]>TOL]
        hit=(nearest(ref,gone)[0]<=ACTION_RADIUS)&absent&~neutral
        if frame>=cut:after|=hit
        else:before|=hit
        far=gone[nearest(gone,ref)[0]>ACTION_RADIUS]
        for k in np.floor(far/V).astype(np.int64):extra.add(tuple(k))
        audit.update(absent_surface_atoms=len(gone),reference_atoms_affected=int(hit.sum()),after_rescan=frame>=cut)
        audits.append(audit)
    score=set_score(target,after,before,len(extra))
    groups={}
    for name,ids in [('removed',removed),('moved',moved)]:
        region=np.isin(labels,list(ids));t=target&region
        # Other class TPs are not FPs in this class. Unassigned errors are counted
        # in BOTH class diagnostics. Aggregate is computed independently above.
        other=target&~region
        groups[name]=set_score(t,after&~other,before&~other,len(extra)) if t.any() else None
    objects=[]
    for gid in sorted(removed|moved):
        t=target&(labels==gid)
        objects.append(dict(object_id=gid,kind='removed' if gid in removed else 'moved',target_atoms=int(t.sum()),correctly_cleared_atoms=int((t&after).sum())))
    return dict(**score,by_type=groups,per_object=objects,actions=audits,archive_sha256=hashes,
                off_reference_false_atoms=len(extra),affected_reference_atoms=int((after|before).sum()),early_only_atoms=int((before&~after).sum()))

def tests():
    t=np.array([1,1,0],bool);no=np.zeros(3,bool)
    assert set_score(t,no,no)['D']==0
    assert set_score(t,t,no)['D']==1
    assert set_score(t,np.ones(3,bool),no)['D']==2/3
    assert set_score(t,t,t)['D']==1
    assert set_score(t,no,t)['D']==0
    assert set_score(no,no,no)['D']==0
    z=dict(xyz=np.array([[0,0,0],[V,0,0]]),tsdf=np.array([-.5,.5]),weight=np.ones(2))
    p,_=archived_surface(z);assert np.allclose(p,[[V/2,0,0]])
    z['weight'][0]=0;assert len(archived_surface(z)[0])==0
    for ap in [0,.2,1]:assert ap+(1-ap)*set_score(t,no,no)['D']==ap
    print('TESTS_OK: no action, perfect/false/premature/deduplicated erasure, empty target, zero-crossing proof',flush=True)
