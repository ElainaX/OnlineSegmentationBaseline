# CPU-only copy of evaluate_matches; evaluator globals injected by flat_scoring.py.
import numpy as np

def evaluate_matches(matches):
    overlaps = opt.overlaps  # different overlap thresholds to compute AP
    min_region_sizes = [opt.min_region_sizes[0]]
    dist_threshes = [opt.distance_threshes[0]]
    dist_confs = [opt.distance_confs[0]]

    # results: class x overlap
    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):

            pred_visited = {}
            for m in matches:
                for p in matches[m]['pred']:
                    for label_name in CLASS_LABELS:
                        for p in matches[m]['pred'][label_name]:
                            if 'filename' in p:
                                pred_visited[p['filename']] = False

            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                # go through each seq
                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]  # *** predicted instances, dict
                    gt_instances = matches[m]['gt'][label_name]  # *** GT instances, dict

                    # filter groups in ground truth
                    gt_instances = [gt for gt in gt_instances if gt['instance_id'] >= 1000 and gt['vert_count'] >= min_region_size and gt['med_dist'] <= distance_thresh and gt['dist_conf'] >= distance_conf]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)  # whether each GT instance have matching predicted instance

                    # Step 1: collect matches (for each GT instance, find its corresponding predicted instance)
                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False  # whether this GT instance has found corresponding predicted instance
                        num_pred = len(gt['matched_pred'])  # number of predicted instances that have overlap with it

                        for pred in gt['matched_pred']:
                            # greedy assignments
                            if pred_visited[pred['filename']]:
                                continue

                            # 1.1: compute IoU of this pred instance and current GT instance
                            pred_instance_id = pred['pred_id']
                            intersect_num = pred['intersection']
                            gt_instance_num = gt['vert_count']
                            pred_instance_num = pred['vert_count']
                            overlap = float(intersect_num) / (gt_instance_num + pred_instance_num - intersect_num)  # IoU

                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                # if already have a prediction for this gt,
                                # the prediction with the lower score is automatically a false positive
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    # append false positive
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                # otherwise set score
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['filename']] = True
                        # END for each matched pred instance

                        if not found_match:
                            hard_false_negatives += 1
                    # END for each GT instance

                    # remove non-matched GT instances
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    # Step 2: collect non-matched predicted instances as False Positive
                    pred_instance_unmatched = []  # to record those pred instances having no corresponding GT instance
                    for (pred_idx, pred) in enumerate(pred_instances):
                        found_gt = False
                        pred_instance_id = pred['pred_id']

                        for gt in pred['matched_gt']:
                            gt_instance_id = gt["instance_id"] % 1000  # ID of this GT instance (starting from 1)
                            intersect_num = gt['intersection']
                            gt_instance_num = gt['vert_count']
                            pred_instance_num = pred['vert_count']
                            overlap = float(intersect_num) / (gt_instance_num + pred_instance_num - intersect_num)

                            if overlap > overlap_th:
                                found_gt = True
                                break
                        # END for each matched GT instance

                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                # group?
                                if gt['instance_id'] < 1000:
                                    num_ignore += gt['intersection']
                                # small ground truth instances
                                if gt['vert_count'] < min_region_size or gt['med_dist'] > distance_thresh or gt['dist_conf'] < distance_conf:
                                    num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            # if not ignored append false positive
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                confidence = pred["confidence"]
                                cur_score = np.append(cur_score, confidence)
                                pred_instance_unmatched.append(pred_idx)  # only for testing
                    # END for each pred instance

                    # append to overall results
                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)
                # END for seq

                # Step 3: compute average precision for given GT instances and predicted instances
                if has_gt and has_pred:
                    if len(y_score) == 0:
                        ap_current = 0.0
                    else:
                        # 3.1: compute precision recall curve first
                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)  # number of predicted instance
                        num_true_examples = y_true_sorted_cumsum[-1]  # number of predicted instance that has corresponding GT instance
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives  # number of GT instance that has no corresponding pred instance
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.
                        recall[-1] = 0.

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.)

                        stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')

                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')

                ap[di, li, oi] = ap_current
            # END for
        # END for
    return ap
