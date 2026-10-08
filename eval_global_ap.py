"""G-AP v1: size-weighted retention/deletion IoU, with extra prediction mass.

Custom Global AP name requested by the user; NOT precision-recall area.
All quantities are counts of 25 mm surface samples, not occupied object volume.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def downsample(points, size, labels=None):
    points=np.asarray(points,float).reshape(-1,3)
    if not len(points): return points.copy(),np.empty(0,int),np.empty(0,int)
    keys=np.floor(points/size).astype(np.int64)
    # Keep different GT instances separate even when they share a grid cell.
    if labels is not None: keys=np.c_[np.asarray(labels,dtype=np.int64),keys]
    _,first,inverse=np.unique(keys,axis=0,return_index=True,return_inverse=True)
    count=np.bincount(inverse)
    centers=np.column_stack([np.bincount(inverse,weights=points[:,i])/count for i in range(3)])
    return centers,first,inverse


def nearest(source,target):
    if not len(target): return np.full(len(source),np.inf),np.full(len(source),-1,int)
    return cKDTree(target).query(source)


def spatial_components(points,radius):
    if not len(points): return np.empty(0,int)
    edges=cKDTree(points).query_pairs(radius,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges),bool),(edges[:,0],edges[:,1])),shape=(len(points),len(points)))
    _,labels=connected_components(graph,directed=False)
    return labels


def aggregate_iou(iou,gt_sizes,pred_sizes,gt_ids=None):
    """One-to-one Hungarian match maximizing GT-size-weighted IoU.

    No positive-overlap prediction is filtered, no IoU acceptance threshold.
    Unmatched GT contributes zero. Extra prediction support adds to denominator.
    """
    iou=np.asarray(iou,float);gt_sizes=np.asarray(gt_sizes,float);pred_sizes=np.asarray(pred_sizes,float)
    if iou.shape!=(len(pred_sizes),len(gt_sizes)): raise ValueError('Inconsistent shapes')
    if not np.isfinite(iou).all() or np.any(iou<0) or np.any(iou>1): raise ValueError('Invalid IoU')
    if np.any(gt_sizes<=0) or np.any(pred_sizes<=0): raise ValueError('Empty instances must be listed separately')
    matched={}
    if iou.size:
        rows,cols=linear_sum_assignment(-(iou*gt_sizes[None,:]))
        matched={int(g):int(p) for p,g in zip(rows,cols) if iou[p,g]>0}
    used=set(matched.values());fp=[i for i in range(len(pred_sizes)) if i not in used]
    numerator=sum(gt_sizes[g]*iou[p,g] for g,p in matched.items())
    weight=float(gt_sizes.sum());fp_weight=float(pred_sizes[fp].sum())
    denom=weight+fp_weight
    ids=list(range(len(gt_sizes))) if gt_ids is None else list(map(int,gt_ids))
    return dict(score=float(numerator/denom) if denom else None,numerator=float(numerator),
        GT_weight=weight,extra_prediction_weight=fp_weight,denominator=denom,
        GT_only_weighted_IoU=float(numerator/weight) if weight else None,
        GT_instances=len(gt_sizes),predicted_regions=len(pred_sizes),extra_predictions=len(fp),
        unmatched_prediction_indices=fp,prediction_sizes=pred_sizes.tolist(),
        per_GT=[dict(GT_id=ids[g],GT_size=float(gt_sizes[g]),prediction=matched.get(g),
                     IoU=float(iou[matched[g],g]) if g in matched else 0.) for g in range(len(gt_sizes))])


def score_sets(gt_labels,gt_ids,pred_domain,extra_counts):
    """Prediction sets consist of mapped GT-domain atoms plus off-domain atoms."""
    gt_ids=np.asarray(gt_ids,dtype=int);pred_domain=np.asarray(pred_domain,bool)
    extra_counts=np.asarray(extra_counts,int)
    pred_sizes=pred_domain.sum(axis=1)+extra_counts
    keep=pred_sizes>0
    pred_domain=pred_domain[keep];pred_sizes=pred_sizes[keep]
    sizes=np.array([(gt_labels==g).sum() for g in gt_ids],float)
    iou=np.zeros((len(pred_sizes),len(gt_ids)))
    for j,g in enumerate(gt_ids):
        inter=pred_domain[:,gt_labels==g].sum(axis=1)
        iou[:,j]=inter/(pred_sizes+sizes[j]-inter)
    result=aggregate_iou(iou,sizes,pred_sizes,gt_ids)
    result['retained_prediction_input_indices']=np.flatnonzero(keep).tolist()
    result['iou_matrix']=iou.tolist()
    return result


def retention(points,masks,gt_points,gt_labels,voxel,tolerance):
    valid=gt_labels>0
    gt,first,_=downsample(gt_points[valid],voxel,gt_labels[valid]);labels=gt_labels[valid][first]
    ids=np.unique(labels)
    pred,_,inverse=downsample(points,voxel)
    vmasks=np.zeros((len(pred),masks.shape[1]),bool)
    for i in range(masks.shape[1]): np.logical_or.at(vmasks[:,i],inverse,masks[:,i])
    assigned=vmasks.any(axis=1);mapped=np.zeros((masks.shape[1],len(gt)),bool)
    d,idx=nearest(gt,pred[assigned]);ok=d<=tolerance
    if ok.any(): mapped[:,ok]=vmasks[assigned][idx[ok]].T
    d_pred,_=nearest(pred,gt)
    # Every off-GT surface cell is part of its predicted instance's union.
    extras=vmasks[d_pred>tolerance].sum(axis=0)
    # A nonempty but completely shadowed prediction still counts as a FP.
    shadowed=(mapped.sum(axis=1)+extras==0)&(vmasks.sum(axis=0)>0)
    extras[shadowed]=vmasks[:,shadowed].sum(axis=0)
    return score_sets(labels,ids,mapped,extras)


def deletion(initial,final,reference_gt,reference_labels,final_gt,removed_ids,ignore_old_ids,voxel,tolerance):
    """Actual geometric disappearance, including natural TSDF clearing.

    GT/reference is fixed for every method. Unbuilt surfaces and old surfaces
    occupied by valid final GT are not deletion targets. Known motion/deformation
    old regions are neutral in this removed-object-only v1 protocol.
    """
    initial,_,_=downsample(initial,voxel)
    ref,first,_=downsample(reference_gt,voxel,reference_labels)
    labels=reference_labels[first];objects=labels%1000
    built_dist,witness=nearest(ref,initial);built=built_dist<=tolerance
    curr_dist,_=nearest(ref,final_gt)
    reusable=curr_dist<=tolerance
    ignored=(labels>0)&np.isin(objects,list(ignore_old_ids))&~np.isin(objects,list(removed_ids))
    target=(labels>0)&np.isin(objects,list(removed_ids))&built&~reusable
    # Keep unchanged and unannotated initial GT atoms as possible false erasure.
    domain=built&~reusable&~ignored
    domain_points=ref[domain];domain_objects=np.where(target[domain],objects[domain],-1)
    target_ids=np.unique(objects[target])
    distances,_=nearest(initial,final)
    missing=distances>tolerance
    # Neutral regions are recognized against ALL reference geometry, not just
    # the closest annotated object, to avoid bleeding labels through a wall.
    d_ref,near_ref=nearest(initial,ref)
    neutral=np.zeros(len(initial),bool)
    hit=d_ref<=tolerance
    neutral[hit]=ignored[near_ref[hit]]
    d_current,_=nearest(initial,final_gt)
    neutral|=d_current<=tolerance
    chosen=np.flatnonzero(missing&~neutral)
    components=spatial_components(initial[chosen],3*voxel)
    group=np.full(len(initial),-1,int);group[chosen]=components
    n=int(components.max())+1 if len(components) else 0
    projected=group[witness[domain]]
    mapped=np.array([projected==k for k in range(n)],bool).reshape(n,len(domain_points))
    d_extra,_=nearest(initial[chosen],domain_points)
    extras=np.bincount(components[d_extra>tolerance],minlength=n)
    physical_sizes=np.bincount(components,minlength=n)
    shadowed=(mapped.sum(axis=1)+extras==0)&(physical_sizes>0)
    extras[shadowed]=physical_sizes[shadowed]
    score=score_sets(domain_objects,target_ids,mapped,extras)
    per=[]
    for g in sorted(removed_ids):
        mask=(labels>0)&(objects==g)
        per.append(dict(GT_id=g,total_reference_samples=int(mask.sum()),initially_built=int((mask&built).sum()),
                        reusable_current_surface=int((mask&built&reusable).sum()),
                        deletion_target_samples=int((mask&target).sum()),
                        absent_final_samples=int((group[witness[target&(objects==g)]]>=0).sum())))
    score.update(per_removed_GT_domain=per,initial_samples=len(initial),missing_initial_samples=int(missing.sum()),
        neutral_missing_samples=int((missing&neutral).sum()),scored_deleted_samples=len(chosen),
        deletion_component_radius_m=3*voxel,no_min_component_size=True)
    return score


def combine(keep,remove):
    denominator=keep['denominator']+remove['denominator']
    return (keep['numerator']+remove['numerator'])/denominator if denominator else None


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def cloud(path):
    import open3d as o3d
    points=np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if not np.isfinite(points).all():raise ValueError('Invalid geometry')
    return points


def run_pair01():
    ROOT=Path('/root/autodl-tmp/output/gap_20261009');ROOT.mkdir(exist_ok=True)
    OLD=Path('/root/autodl-tmp/output/ig_surface_deletion_20261008')
    WHOLE=Path('/root/autodl-tmp/output/ig_whole_instance_20261008')
    DATA=Path('/root/autodl-tmp/3RScan/3RScan_Dynamic_OnlineAnySeg');SEQ='pair_01_0cac767c_19f1a892'
    ref_snapshot=Path('/root/autodl-tmp/output/ig_correction_20261008/delete')/SEQ/'correction/reference_surfaces.npz'
    reference_parent=json.loads((ref_snapshot.parent.parent/'run_metadata.json').read_text())
    freeze=json.loads((ref_snapshot.parent/'summary.json').read_text())
    boundary=json.loads((DATA/SEQ/'session_boundary.json').read_text())
    assert freeze['freeze_frame']==boundary['rescan_start_frame']==170
    assert reference_parent['tsdf_voxel_size_mm']==25 and reference_parent['mapping_frame_interval']==2
    initial=np.load(ref_snapshot)['all_points']
    refpath=DATA/'gt_reference'/SEQ/(SEQ+'_vh_clean_2.ply');curpath=DATA/SEQ/(SEQ+'_vh_clean_2.ply')
    reflabelpath=DATA/'gt_seg_reference_all_things'/(SEQ+'.txt');curlabelpath=DATA/'gt_seg_all_things'/(SEQ+'.txt')
    ref=cloud(refpath);current=cloud(curpath)
    labels=np.loadtxt(reflabelpath,dtype=int);current_labels=np.loadtxt(curlabelpath,dtype=int)
    assert len(ref)==len(labels) and len(current)==len(current_labels)
    changes=boundary['changed_instances'];removed=set(map(int,changes['removed']))
    ignored=set(map(int,changes['nonrigid']))|{int(r['instance_reference']) for r in changes['rigid']}
    variants=dict(no_deletion=OLD/'off'/SEQ,local_historical=OLD/'on'/SEQ,
                  local_control=WHOLE/'local_control_run/hybrid'/SEQ,hybrid=WHOLE/'hybrid'/SEQ)
    report=dict(name='G-AP (Global AP)',version='v1-20261009',standard_AP=False,
        definition='sum(GT_size * matched_IoU) / (sum(GT_size) + sum(unmatched_prediction_support_size)); summed over retention and deletion tasks',
        protocol=dict(voxel_size_m=.025,tolerance_m=.05,current_GT='all annotated instances; no 100-point or semantic-class exclusion',
            GT_weights='per-instance occupied surface sample count at common 25 mm resolution',
            coordinates='existing reference-world dataset alignment; no per-method ICP',
            extra_prediction_penalty_weight=1,matching='one-to-one Hungarian maximizing GT-size-weighted IoU, no acceptance threshold',
            shadowed_predictions='nonempty regions with zero projected support count as extra predictions using their physical surface sample count',
            deleted_prediction='spatial components of actual missing initial geometry, radius 75mm, no minimum size',
            initial_reference='shared actual pre-rescan reconstructed surface snapshot; not final no-deletion cloud',
            unknown_geometry='strict diagnostic: not ignored, may include unannotated genuine surfaces',
            rigid_nonrigid='old regions neutral in deletion task; final instances retained in normal segmentation task',
            unbuilt_old_surface='excluded from deletion target, no free reward',
            initial_reused_surface='old reference surface within 5cm of final GT is not a deletion target'),
        shared_input_sha256={str(p):digest(p) for p in [ref_snapshot,refpath,curpath,reflabelpath,curlabelpath,DATA/SEQ/'session_boundary.json']},variants={})
    for mode,path in variants.items():
        points=cloud(path/'final.ply');data=np.load(path/'ckpt_final.npz');masks=data['pred_masks']
        assert masks.shape[0]==len(points)
        keep=retention(points,masks,current,current_labels,.025,.05)
        remove=deletion(initial,points,ref,labels,current,removed,ignored,.025,.05)
        row=dict(G_AP=combine(keep,remove),retention=keep,deletion=remove,
                 input_sha256={str(p):digest(p) for p in [path/'final.ply',path/'ckpt_final.npz']})
        report['variants'][mode]=row
        print(mode,dict(G_AP=row['G_AP'],retention=keep['score'],deletion=remove['score'],
                       deletion_target_weight=remove['GT_weight'],deleted_regions=remove['predicted_regions']),flush=True)
        (ROOT/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=='__main__':run_pair01()
