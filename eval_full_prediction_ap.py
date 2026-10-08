"""Diagnostic class-agnostic AP that never drops a nonempty prediction.

FullPred-AP keeps GT-domain IoU but removes prediction/void filtering.
VoxelUnion-AP measures IoU on a shared world voxel grid, including extra geometry.
Neither is an official ScanNet score. Original evaluator remains untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial import cKDTree


THRESHOLDS = [0.25] + [x / 100 for x in range(50, 95, 5)]


def strict_ap(ious, scores, thresholds=THRESHOLDS):
    """One-to-one, maximum-cardinality matching at each confidence prefix.

    Equal scores enter together, so file order does not break ties. Integration
    follows the existing ScanNet-style centered-recall-step convention, including
    (recall=0, precision=1). Every included prediction is exactly TP or FP.
    """
    ious = np.asarray(ious, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if ious.ndim != 2 or ious.shape[0] != len(scores):
        raise ValueError("IoUs must have shape (predictions, GT instances)")
    if not np.all(np.isfinite(ious)) or not np.all(np.isfinite(scores)):
        raise ValueError("Non-finite input")
    if np.any(ious < 0) or np.any(ious > 1):
        raise ValueError("IoU outside [0,1]")
    n_pred, n_gt = ious.shape
    rows = []
    for threshold in thresholds:
        curve = []
        for score in np.unique(scores)[::-1]:
            selected = scores >= score
            graph = csr_matrix(ious[selected] > threshold)
            tp = int(np.count_nonzero(maximum_bipartite_matching(graph, perm_type="column") >= 0)) if n_gt else 0
            count = int(selected.sum())
            curve.append(dict(score=float(score),TP=tp,FP=count-tp,FN=n_gt-tp,
                              precision=tp/count,recall=tp/n_gt if n_gt else 0.))
        if not n_gt:
            ap = None  # No positive GT: report FP, not a fictitious perfect AP.
        elif not n_pred:
            ap = 0.
        else:
            precision = np.array([r['precision'] for r in curve][::-1] + [1.])
            recall = np.array([r['recall'] for r in curve][::-1] + [0.])
            widths = np.convolve(np.r_[recall[0], recall, 0.], [-.5, 0., .5], 'valid')
            ap = float(precision @ widths)
        operating = curve[-1] if curve else dict(TP=0,FP=0,FN=n_gt,precision=0.,recall=0.)
        denom=2*operating['TP']+operating['FP']+operating['FN']
        rows.append(dict(iou_threshold=threshold,AP=ap,**operating,
                         F1=2*operating['TP']/denom if denom else None,curve=curve))
    lookup = {r['iou_threshold']:r['AP'] for r in rows}
    avg = [r['AP'] for r in rows if r['iou_threshold'] >= .5 and r['AP'] is not None]
    return dict(AP=float(np.mean(avg)) if avg else None,AP50=lookup.get(.5),AP25=lookup.get(.25),
                F1_50=next((r['F1'] for r in rows if r['iou_threshold']==.5),None),
                F1_25=next((r['F1'] for r in rows if r['iou_threshold']==.25),None),
                predictions=n_pred,GT_instances=n_gt,thresholds=rows)


def projected_iou(points, masks, gt_points, labels, gt_ids, radius):
    """Keep ALL nonempty raw masks, even when GT projection is empty."""
    assigned = np.any(masks, axis=1)
    mapped = np.zeros((len(gt_points), masks.shape[1]), dtype=bool)
    if assigned.any() and len(gt_points):
        distances, nearest = cKDTree(points[assigned]).query(gt_points)
        valid = distances <= radius
        mapped[valid] = masks[assigned][nearest[valid]]
    size = mapped.sum(axis=0)
    ious = np.zeros((masks.shape[1],len(gt_ids)))
    for j, gt_id in enumerate(gt_ids):
        gt = labels == gt_id
        intersection = mapped[gt].sum(axis=0)
        ious[:,j] = intersection / (size + gt.sum() - intersection)
    return ious, size


def voxel_union_iou(points, masks, gt_points, labels, gt_ids, voxel_size):
    """Exact set IoU on one fixed grid; each occupied cell has unit weight.

    Extra predicted cells stay in the union. No dilation, nearest-neighbor
    tolerance, per-method bounding-box origin, or prediction-dependent alignment.
    """
    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    grid = np.floor(np.vstack((points,gt_points)) / voxel_size).astype(np.int64)
    _, codes = np.unique(grid, axis=0, return_inverse=True)
    pred_codes, gt_codes = codes[:len(points)], codes[len(points):]
    pred_sets = [np.unique(pred_codes[masks[:,i]]) for i in range(masks.shape[1])]
    gt_sets = [np.unique(gt_codes[labels==g]) for g in gt_ids]
    ious = np.zeros((len(pred_sets),len(gt_sets)))
    for i,pred in enumerate(pred_sets):
        for j,gt in enumerate(gt_sets):
            intersection = np.intersect1d(pred,gt,assume_unique=True).size
            ious[i,j] = intersection / (len(pred)+len(gt)-intersection)
    return ious, [len(p) for p in pred_sets], [len(g) for g in gt_sets]


def geometry_diagnostics(points, gt_points, voxel_size, tolerances=(.025,.05,.10,.15)):
    """Unmatched does NOT prove disappearance: GT can be incomplete."""
    pred = (np.unique(np.floor(points/voxel_size).astype(np.int64),axis=0)+.5)*voxel_size
    gt = (np.unique(np.floor(gt_points/voxel_size).astype(np.int64),axis=0)+.5)*voxel_size
    d_pred = cKDTree(gt).query(pred)[0] if len(gt) else np.full(len(pred),np.inf)
    d_gt = cKDTree(pred).query(gt)[0] if len(pred) else np.full(len(gt),np.inf)
    return {str(t):dict(predicted_voxels=len(pred),GT_voxels=len(gt),
                        unsupported_predicted_voxels=int((d_pred>t).sum()),
                        supported_prediction_fraction=float((d_pred<=t).mean()) if len(pred) else None,
                        GT_coverage=float((d_gt<=t).mean()) if len(gt) else None) for t in tolerances}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def load_cloud(path):
    import open3d as o3d
    points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    if not np.all(np.isfinite(points)): raise ValueError('Non-finite point cloud')
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir',type=Path,required=True)
    parser.add_argument('--gt-cloud',type=Path,required=True)
    parser.add_argument('--gt-labels',type=Path,required=True)
    parser.add_argument('--gt-to-prediction-transform',type=Path,required=True,
                        help='Fixed 4x4 transform JSON; reuse SAME file for all compared methods')
    parser.add_argument('--valid-class-ids',required=True,help='Comma-separated existing evaluator class IDs')
    parser.add_argument('--min-gt-points',type=int,default=100)
    parser.add_argument('--voxel-size',type=float,default=.025)
    parser.add_argument('--projection-radius',type=float,default=.15)
    parser.add_argument('--with-experimental-voxel-ap',action='store_true',
                        help='Diagnostic only: exact voxel overlap is sensitive to surface sampling/alignment')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.projection_radius<=0 or args.min_gt_points<1: raise ValueError('Invalid threshold')
    pred_path=args.prediction_dir/'final.ply'; ckpt=args.prediction_dir/'ckpt_final.npz'
    points=load_cloud(pred_path); gt_points=load_cloud(args.gt_cloud)
    labels=np.loadtxt(args.gt_labels,dtype=np.int64).reshape(-1)
    data=np.load(ckpt); masks=np.asarray(data['pred_masks'],dtype=bool); scores=data['pred_score']
    if masks.shape!=(len(points),len(scores)) or len(gt_points)!=len(labels):
        raise ValueError('Point/mask/label length mismatch')
    transform=np.asarray(json.loads(args.gt_to_prediction_transform.read_text())['T_prediction_GT'],dtype=float)
    if transform.shape!=(4,4) or not np.all(np.isfinite(transform)) or not np.allclose(transform[3],[0,0,0,1]):
        raise ValueError('Invalid homogeneous transform')
    if not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(transform[:3,:3]),1,atol=1e-5):
        raise ValueError('Transform must be rigid in meters')
    gt_points=gt_points@transform[:3,:3].T+transform[:3,3]
    classes=list(map(int,args.valid_class_ids.split(',')))
    ids,counts=np.unique(labels,return_counts=True)
    gt_ids=ids[(ids>=1000)&np.isin(ids//1000,classes)&(counts>=args.min_gt_points)]
    nonempty=masks.any(axis=0);raw_ids=np.flatnonzero(nonempty);masks=masks[:,nonempty];scores=scores[nonempty]
    projected,mapped_sizes=projected_iou(points,masks,gt_points,labels,gt_ids,args.projection_radius)
    voxel,pred_sizes,gt_sizes=voxel_union_iou(points,masks,gt_points,labels,gt_ids,args.voxel_size) if args.with_experimental_voxel_ap else (None,None,None)
    report=dict(protocol='strict-full-prediction-v1',official_ScanNet_metric=False,
                note='All nonempty predictions count. No void/class/GT-overlap prediction exclusion. Missing GT annotation may create apparent FP; unmatched geometry is not proven disappearance.',
                config=dict(voxel_size_m=args.voxel_size,grid_origin=[0,0,0],projection_radius_m=args.projection_radius,
                            GT_min_raw_points=args.min_gt_points,GT_valid_classes=classes,class_agnostic=True,
                            matching='maximum one-to-one matching at each tied confidence prefix',
                            integration='existing centered-recall-step convention',strict_iou_greater_than=True),
                inputs={str(p):digest(p) for p in [pred_path,ckpt,args.gt_cloud,args.gt_labels,args.gt_to_prediction_transform]},
                raw_predictions=len(nonempty),empty_raw_predictions=int((~nonempty).sum()),
                GT_ids=gt_ids.tolist(),
                FullPred_AP=strict_ap(projected,scores),
                geometry=geometry_diagnostics(points,gt_points,args.voxel_size),
                instances=[dict(raw_id=int(raw),score=float(scores[i]),raw_points=int(masks[:,i].sum()),
                                projected_GT_points=int(mapped_sizes[i]),
                                best_projected_iou=float(projected[i].max()) if len(gt_ids) else None,
                                best_voxel_iou=float(voxel[i].max()) if voxel is not None and len(gt_ids) else None) for i,raw in enumerate(raw_ids)],
                projected_iou_matrix=projected.tolist())
    if voxel is not None:
        report.update(VoxelUnion_AP=strict_ap(voxel,scores),voxel_iou_matrix=voxel.tolist(),GT_voxel_counts=gt_sizes,
                      voxel_AP_status='EXPERIMENTAL: exact grid-cell overlap; not adopted as main metric',prediction_voxel_counts=pred_sizes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({key:{k:report[key][k] for k in ['AP','AP50','AP25','predictions','GT_instances']} for key in ['FullPred_AP','VoxelUnion_AP'] if key in report},indent=2))


if __name__=='__main__': main()
