"""
Metrics for tamper localization evaluation:
  - pixel_iou / pixel_f1: standard segmentation overlap metrics,
    computed per-image then averaged (macro), which is the fairer
    choice here since forged region size varies a lot across
    manipulation types.
  - image_level_auc: treats "max predicted probability in the image"
    as an image-level tamper score and computes ROC-AUC against the
    binary real/fake label — mirrors how these detectors are actually
    used as a first-pass screen before someone looks at the mask.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    """pred_mask, gt_mask: binary (0/1) arrays, same shape."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float((intersection + eps) / (union + eps))


def pixel_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, eps: float = 1e-6) -> float:
    tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return float(2 * precision * recall / (precision + recall + eps))


def batch_pixel_metrics(pred_probs: np.ndarray, gt_masks: np.ndarray, threshold: float = 0.5) -> dict:
    """pred_probs, gt_masks: [N, H, W] float arrays (probs in [0,1] and
    binary 0/1 respectively). Returns macro-averaged IoU/F1 over N images."""
    preds_bin = (pred_probs >= threshold).astype(np.uint8)
    gts_bin = (gt_masks >= 0.5).astype(np.uint8)

    ious, f1s = [], []
    for p, g in zip(preds_bin, gts_bin):
        ious.append(pixel_iou(p, g))
        f1s.append(pixel_f1(p, g))
    return {"iou": float(np.mean(ious)), "f1": float(np.mean(f1s))}


def image_level_auc(pred_probs: np.ndarray, image_labels: np.ndarray) -> float:
    """pred_probs: [N, H, W] predicted probability maps.
    image_labels: [N] binary (1 = tampered image, 0 = clean image).
    Image-level score = max predicted probability within the image."""
    scores = pred_probs.reshape(pred_probs.shape[0], -1).max(axis=1)
    if len(np.unique(image_labels)) < 2:
        # AUC undefined with a single class present — signal this clearly
        # rather than silently returning a meaningless number.
        return float("nan")
    return float(roc_auc_score(image_labels, scores))
