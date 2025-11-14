import logging

import numpy as np
from cellpose.metrics import average_precision
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# Jaccard index (IoU) calculation
def jaccard_index_3d(gt_mask, pred_mask):
    """Computes Jaccard index (IoU) for two 3D binary masks."""
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    return intersection / union if union > 0 else 0.0

def jaccard(ground_truth, masks):
    """Computes the mean Jaccard score for matched instances."""
    unique_gt = np.unique(ground_truth[ground_truth > 0])
    unique_masks = np.unique(masks[masks > 0])
    if len(unique_gt) == 0 or len(unique_masks) == 0:
        return 0.0
    aji = np.zeros((len(unique_gt), len(unique_masks)))
    for i, label_gt in enumerate(unique_gt):
        gt_mask_instance = ground_truth == label_gt
        for j, label_mask in enumerate(unique_masks):
            pred_mask_instance = masks == label_mask
            aji[i, j] = jaccard_index_3d(gt_mask_instance, pred_mask_instance)
    row_ind, col_ind = linear_sum_assignment(-aji)
    matched_scores = aji[row_ind, col_ind]
    return matched_scores.mean()


def calculate_segmentation_stats(true_masks, pred_masks, iou_threshold=0.5):
    """
    Calculates precision, recall, and F1-score based on instance segmentation.

    Args:
        true_masks (np.ndarray): The ground truth label mask.
        pred_masks (np.ndarray): The predicted label mask.
        iou_threshold (float): The IoU threshold to consider a prediction a True Positive.

    Returns:
        dict: A dictionary containing precision, recall, f1_score, TP, FP, and FN counts.
    """
    # Use cellpose's average_precision function which returns matching info
    # The first return value (ap) is not what we need, but tp, fp, fn are.
    # It returns these values for a range of IoU thresholds. We'll use the one for our desired threshold.
    ap, tp, fp, fn = average_precision(true_masks, pred_masks, threshold=[iou_threshold])
    jaccard_val = jaccard(true_masks, pred_masks)

    # The results are arrays, but since we use a single threshold, we take the first element.
    true_positives = tp[0]
    false_positives = fp[0]
    false_negatives = fn[0]

    # Calculate precision, recall, and F1 score
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0.0

    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    n_instances_true = np.unique(true_masks[true_masks > 0]).size
    n_instances_pred = np.unique(pred_masks[pred_masks > 0]).size

    return {
        'jaccard': jaccard_val,
        f'f1@{iou_threshold:.2f}': f1_score,
        # 'precision': precision,
        # 'recall': recall,
        # 'tp': true_positives,
        # 'fp': false_positives,
        # 'fn': false_negatives,
        'n_instances_true': n_instances_true,
        'n_instances_pred': n_instances_pred,
    }