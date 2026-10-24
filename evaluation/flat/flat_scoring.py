"""CPU scoring primitives; GT is used here only, never by mapping or segmentation."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from scipy.spatial import cKDTree
from types import SimpleNamespace
import reference_ap_function as reference


def correspondences(points, reconstruction, radius=.05):
    """Preserve every GT point; -1 denotes a missing reconstruction correspondence."""
    points, reconstruction = np.asarray(points), np.asarray(reconstruction)
    assert np.isfinite(points).all() and np.isfinite(reconstruction).all()
    if not len(reconstruction):
        return np.full(len(points), -1, int)
    tree = cKDTree(reconstruction)
    distance, index = tree.query(points, k=2)
    result = np.where(distance[:, 0] <= radius, index[:, 0], -1)
    # Resolve exact-distance ambiguity independently of KD-tree traversal order.
    ties = np.flatnonzero((result >= 0) & (distance[:, 0] == distance[:, 1]))
    for i in ties:
        candidates = np.array(tree.query_ball_point(points[i], np.nextafter(distance[i, 0], np.inf)))
        squared = np.sum((reconstruction[candidates] - points[i]) ** 2, axis=1)
        result[i] = candidates[squared == squared.min()].min()
    return result


def matches_from_masks(labels, thing_ids, projected_masks, scores, minimum):
    """Only accepted GT points enter; stuff stays in prediction sizes as negatives."""
    labels = np.asarray(labels)
    masks = np.asarray(projected_masks, dtype=bool)
    scores = np.asarray(scores)
    assert masks.shape == (len(labels), len(scores))
    assert np.all(labels >= 0) and np.isfinite(scores).all()
    gt = []
    for ident in sorted(set(labels) & set(thing_ids)):
        gt.append(dict(instance_id=int(ident) + 1001, label_id=1,
            vert_count=int(np.sum(labels == ident)), med_dist=0., dist_conf=1.,
            matched_pred=[]))
    pred = []
    for i, score in enumerate(scores):
        size = int(masks[:, i].sum())
        if size < minimum:
            continue
        p = dict(filename=f"flat_prediction_{i}", pred_id=i, label_id=1,
                 vert_count=size, confidence=float(score), void_intersection=0, matched_gt=[])
        for g in gt:
            count = int(np.sum(masks[:, i] & (labels == g["instance_id"] - 1001)))
            if count:
                # No recursive matched lists: evaluator needs scalar fields only.
                p["matched_gt"].append({k: v for k, v in g.items() if k != "matched_pred"} | {"intersection": count})
                g["matched_pred"].append({k: v for k, v in p.items() if k != "matched_gt"} | {"intersection": count})
        pred.append(p)
    return {"flat_run1_run2": {"gt": {"object": gt}, "pred": {"object": pred}}}


def evaluate_projected(labels, thing_ids, projected_masks, scores, minimum=100):
    reference.CLASS_LABELS = ["object"]
    reference.opt = SimpleNamespace(overlaps=np.append(np.arange(.5, .95, .05), .25),
        min_region_sizes=np.array([minimum]), distance_threshes=np.array([np.inf]),
        distance_confs=np.array([-np.inf]))
    matches = matches_from_masks(labels, thing_ids, projected_masks, scores, minimum)
    aps = reference.evaluate_matches(matches)[0, 0]
    gt = matches["flat_run1_run2"]["gt"]["object"]
    eligible = [g["instance_id"] - 1001 for g in gt if g["vert_count"] >= minimum]
    return dict(AP=float(aps[:-1].mean()) if eligible else None,
        AP50=float(aps[0]) if eligible else None, AP25=float(aps[-1]) if eligible else None,
        threshold_AP=[float(x) if np.isfinite(x) else None for x in aps],
        minimum_points=minimum, eligible_GT_ids=eligible,
        excluded_GT={str(g["instance_id"] - 1001): g["vert_count"] for g in gt if g["vert_count"] < minimum},
        evaluated_predictions=len(matches["flat_run1_run2"]["pred"]["object"]))


def score_reconstruction(domain, reconstruction, masks, scores):
    reconstruction, masks = np.asarray(reconstruction), np.asarray(masks, bool)
    scores = np.asarray(scores)
    assert masks.shape == (len(reconstruction), len(scores)) and np.isfinite(scores).all()
    accepted = domain["instance_id"] >= 0
    points, labels = domain["xyz"][accepted], domain["instance_id"][accepted]
    thing_ids = np.unique(domain["instance_id"][domain["is_thing"]])
    indices = correspondences(points, reconstruction)
    projected = np.zeros((len(points), len(scores)), bool)
    valid = indices >= 0
    projected[valid] = masks[indices[valid]]
    # Reverse support is disclosure only, not used to crop GT or tune the score.
    reverse = correspondences(reconstruction, points)
    no_support = reverse < 0
    return dict(primary=evaluate_projected(labels, thing_ids, projected, scores, 100),
        secondary_all_sizes=evaluate_projected(labels, thing_ids, projected, scores, 1),
        labelled_GT_points=len(points), GT_points_without_reconstruction=int((~valid).sum()),
        reconstruction_vertices_without_labelled_surface_support=int(no_support.sum()),
        prediction_vertex_counts_without_labelled_surface_support=masks[no_support].sum(axis=0).tolist(),
        projected_prediction_sizes=projected.sum(axis=0).tolist(),
        predictions_with_zero_domain_support=np.flatnonzero(~projected.any(axis=0)).tolist())
