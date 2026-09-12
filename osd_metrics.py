"""The OSD L4-global metric, kept independent of MMSeg for fair comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


CURRENT_CLASSES = ("background", "oil", "others", "water")
REPORT_CLASSES = ("oil", "water", "others")


def _iou(confusion: np.ndarray, class_index: int, gt_indices: tuple[int, ...]) -> float:
    true_positive = float(confusion[class_index, class_index])
    false_positive = float(confusion[list(gt_indices), class_index].sum() - true_positive)
    false_negative = float(confusion[class_index, :].sum() - true_positive)
    denominator = true_positive + false_positive + false_negative
    return true_positive / denominator if denominator else float("nan")


def confusion_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    ignore_index: int = 255,
) -> np.ndarray:
    """Return a GT-row/prediction-column confusion matrix for one batch."""
    if logits.ndim != 4 or logits.shape[1] != num_classes:
        raise ValueError(f"Expected logits [B,{num_classes},H,W], got {tuple(logits.shape)}")
    predictions = logits.argmax(dim=1)
    if targets.ndim == 4 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if targets.ndim != 3 or predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: {tuple(predictions.shape)} vs {tuple(targets.shape)}"
        )

    pred = predictions.detach().long().cpu()
    gt = targets.detach().long().cpu()
    valid = gt != ignore_index
    pred, gt = pred[valid], gt[valid]
    if pred.numel() == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)
    if int(pred.min()) < 0 or int(pred.max()) >= num_classes:
        raise ValueError("Prediction contains an out-of-range class")
    if int(gt.min()) < 0 or int(gt.max()) >= num_classes:
        raise ValueError("Ground truth contains an out-of-range class")
    packed = (gt * num_classes + pred).numpy()
    return np.bincount(packed, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).astype(np.int64)


def l4_global_metrics(confusion: np.ndarray) -> dict[str, float]:
    """Match ``OSDL4GlobalMetric`` exactly, including background error terms."""
    confusion = np.asarray(confusion, dtype=np.int64)
    expected = (len(CURRENT_CLASSES), len(CURRENT_CLASSES))
    if confusion.shape != expected:
        raise ValueError(f"Expected a {expected[0]}x{expected[1]} confusion matrix, got {confusion.shape}")

    iou_all, iou_ignore_background, miou4, miou3_report, miou3_ignore_background = _raw_l4_values(confusion)
    precision_all, recall_all, f1_all, pixel_accuracy = _raw_classification_values(confusion)
    report_f1 = [f1_all[name] for name in REPORT_CLASSES]

    def pct(value: float) -> float:
        return round(value * 100.0, 6)

    metrics = {
        "mIoU3_report_only_global": pct(miou3_report),
        "mIoU4_all_classes_global": pct(miou4),
        "mIoU3_ignore_gt_background_global": pct(miou3_ignore_background),
        "IoU_background_global": pct(iou_all["background"]),
        "IoU_oil_global": pct(iou_all["oil"]),
        "IoU_water_global": pct(iou_all["water"]),
        "IoU_others_global": pct(iou_all["others"]),
        "mF1_3_report_only_global": pct(float(np.nanmean(report_f1))),
        "mF1_4_all_classes_global": pct(float(np.nanmean(list(f1_all.values())))),
        "PixelAccuracy_global": pct(pixel_accuracy),
    }
    for name in CURRENT_CLASSES:
        metrics[f"Precision_{name}_global"] = pct(precision_all[name])
        metrics[f"Recall_{name}_global"] = pct(recall_all[name])
        metrics[f"F1_{name}_global"] = pct(f1_all[name])
    return metrics


def _raw_l4_values(
    confusion: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], float, float, float]:
    all_indices = tuple(range(len(CURRENT_CLASSES)))
    semantic_indices = tuple(CURRENT_CLASSES.index(name) for name in REPORT_CLASSES)
    iou_all = {
        name: _iou(confusion, index, all_indices)
        for index, name in enumerate(CURRENT_CLASSES)
    }
    iou_ignore_background = {
        name: _iou(confusion, CURRENT_CLASSES.index(name), semantic_indices)
        for name in REPORT_CLASSES
    }
    miou4 = float(np.nanmean(list(iou_all.values())))
    miou3_report = float(np.nanmean([iou_all[name] for name in REPORT_CLASSES]))
    miou3_ignore_background = float(np.nanmean(list(iou_ignore_background.values())))
    return iou_all, iou_ignore_background, miou4, miou3_report, miou3_ignore_background


def _raw_classification_values(
    confusion: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    """Return global one-vs-rest precision, recall, F1/Dice and pixel accuracy."""
    gt_counts = confusion.sum(axis=1).astype(np.float64)
    predicted_counts = confusion.sum(axis=0).astype(np.float64)
    true_positive = np.diag(confusion).astype(np.float64)

    def safe_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    precision = {
        name: safe_ratio(true_positive[index], predicted_counts[index])
        for index, name in enumerate(CURRENT_CLASSES)
    }
    recall = {
        name: safe_ratio(true_positive[index], gt_counts[index])
        for index, name in enumerate(CURRENT_CLASSES)
    }
    f1 = {
        name: safe_ratio(2.0 * true_positive[index], gt_counts[index] + predicted_counts[index])
        for index, name in enumerate(CURRENT_CLASSES)
    }
    total = float(confusion.sum())
    pixel_accuracy = safe_ratio(float(true_positive.sum()), total)
    return precision, recall, f1, pixel_accuracy


def make_l4_payload(
    confusion: np.ndarray,
    split: str,
    input_size: int | Sequence[int],
    protocol_id: str = "OSD-EXP-v1.0/L4-global-512",
    model_name: Optional[str] = None,
    evaluated_images: Optional[int] = None,
    selection_split: Optional[str] = None,
) -> dict:
    confusion = np.asarray(confusion, dtype=np.int64)
    metrics = l4_global_metrics(confusion)
    iou_all, iou_ignore_background, miou4, miou3_report, miou3_ignore_background = _raw_l4_values(confusion)
    precision_all, recall_all, f1_all, pixel_accuracy = _raw_classification_values(confusion)
    payload = {
        "protocol_id": protocol_id,
        "model": model_name,
        "split": split,
        "selection_split": selection_split,
        "evaluated_images": evaluated_images,
        "input_size": (
            int(input_size)
            if isinstance(input_size, (int, np.integer))
            else [int(value) for value in input_size]
        ),
        "metric": "global 4x4 confusion matrix over all valid pixels",
        "class_order_current": list(CURRENT_CLASSES),
        "class_order_report": list(REPORT_CLASSES),
        "confusion_matrix_current_order_gt_rows_pred_columns": confusion.tolist(),
        "iou_global_current_order": iou_all,
        "iou_report_order": {name: iou_all[name] for name in REPORT_CLASSES},
        "iou_ignore_gt_background": iou_ignore_background,
        "precision_global_current_order": precision_all,
        "recall_global_current_order": recall_all,
        "f1_global_current_order": f1_all,
        "f1_report_order": {name: f1_all[name] for name in REPORT_CLASSES},
        "mF1_3_report_only_global": float(np.nanmean([f1_all[name] for name in REPORT_CLASSES])),
        "mF1_4_all_classes_global": float(np.nanmean(list(f1_all.values()))),
        "PixelAccuracy_global": pixel_accuracy,
        "mIoU4_all_classes_global": miou4,
        "mIoU3_report_only_global": miou3_report,
        "mIoU3_ignore_gt_background_global": miou3_ignore_background,
        "mIoU_report_three": miou3_report,
        "metrics_percent": metrics,
        "background_excluded_from_reported_mean": True,
        "background_retained_in_confusion": True,
    }
    return payload


def write_l4_payload(payload: dict, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
