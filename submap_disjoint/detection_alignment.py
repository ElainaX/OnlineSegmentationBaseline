"""Bounded trimmed ICP for diagnostic geometry, with no GT or instance labels.

This is not a SLAM pose replacement. Return identity unless stable 3D anchors
and a held-out distance check support a small rigid correction.
"""
import numpy as np
from scipy.spatial import cKDTree


def fit_rigid(source, target):
    a=source.mean(0);b=target.mean(0)
    u,_,vt=np.linalg.svd((source-a).T@(target-b))
    r=vt.T@u.T
    if np.linalg.det(r)<0:
        vt[-1]*=-1;r=vt.T@u.T
    t=b-r@a
    matrix=np.eye(4);matrix[:3,:3]=r;matrix[:3,3]=t
    return matrix


def refine_detection_pose(points, target, pose, max_distance=.15, max_center_shift=.2,
                          max_rotation_deg=5., min_points=200, iterations=12):
    """Every second source point is held out of ICP fitting, for acceptance."""
    identity=np.eye(4)
    meta=dict(accepted=False,reason='insufficient_anchors')
    if len(points)<2*min_points or len(target)<min_points:return pose.copy(),meta
    train=points[::2];test=points[1::2];tree=cKDTree(target)
    initial,_=tree.query(test);before=float(np.mean(np.minimum(initial,max_distance)**2))
    delta=identity.copy();fit_count=0
    for _ in range(iterations):
        transformed=train@delta[:3,:3].T+delta[:3,3]
        distance,index=tree.query(transformed)
        valid=np.flatnonzero(distance<max_distance)
        if len(valid)<min_points:return pose.copy(),meta
        selected=valid[np.argsort(distance[valid])[:max(min_points,int(len(valid)*.7))]]
        destinations=target[index[selected]]
        eigen=np.linalg.eigvalsh(np.cov(destinations.T))
        if eigen[0]<.01*max(eigen[-1],1e-9):
            return pose.copy(),dict(accepted=False,reason='degenerate_anchor_geometry')
        correction=fit_rigid(transformed[selected],destinations)
        delta=correction@delta;fit_count=len(selected)
    proposed=delta@pose
    center_shift=float(np.linalg.norm(proposed[:3,3]-pose[:3,3]))
    angle=float(np.degrees(np.arccos(np.clip((np.trace(delta[:3,:3])-1)/2,-1,1))))
    corrected=test@delta[:3,:3].T+delta[:3,3]
    distances,_=tree.query(corrected)
    after=float(np.mean(np.minimum(distances,max_distance)**2))
    gain=(before-after)/max(before,1e-12)
    accepted=(center_shift<=max_center_shift and angle<=max_rotation_deg
              and gain>=.15 and (distances<max_distance).mean()>=.5)
    meta=dict(accepted=bool(accepted),reason='accepted' if accepted else 'bounded_or_validation_gate',
        center_shift_m=center_shift,rotation_deg=angle,relative_mse_gain=gain,
        held_out_before_rmse_m=before**.5,held_out_after_rmse_m=after**.5,
        fit_points=fit_count,transform=delta.tolist())
    return proposed if accepted else pose.copy(),meta
