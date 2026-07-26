"""
Evaluation metrics for multi-label PTB-XL classification.

Change vs. the original submission: F1 was computed at a hard 0.5 threshold on
sigmoid outputs, which for an imbalanced multi-label problem systematically
understates F1 (the submitted macro F1 of ~0.62 alongside a macro AUC of ~0.90
is largely a thresholding artefact). Thresholds are now tuned per class on the
*validation* split and then frozen for the test split.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             roc_auc_score)


def per_class_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[int, float]:
    out = {}
    for c in range(y_true.shape[1]):
        col = y_true[:, c]
        out[c] = float(roc_auc_score(col, y_prob[:, c])) if 0 < col.sum() < len(col) \
            else float("nan")
    return out


def macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals = [v for v in per_class_auc(y_true, y_prob).values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def tune_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                    grid: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-class threshold maximising F1 on the given (validation) split."""
    grid = grid if grid is not None else np.linspace(0.05, 0.95, 91)
    thr = np.full(y_true.shape[1], 0.5, dtype=float)
    for c in range(y_true.shape[1]):
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            f1 = f1_score(y_true[:, c], (y_prob[:, c] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thr[c] = best_t
    return thr


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                        class_names: Optional[List[str]] = None,
                        thresholds: Optional[np.ndarray] = None) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    thr = thresholds if thresholds is not None else np.full(y_true.shape[1], 0.5)
    y_pred = (y_prob >= thr[None, :]).astype(int)

    pc_auc = per_class_auc(y_true, y_prob)
    vals = [v for v in pc_auc.values() if np.isfinite(v)]

    try:
        m_ap = float(average_precision_score(y_true, y_prob, average="macro"))
    except ValueError:
        m_ap = float("nan")

    out = {
        "macro_auc": float(np.mean(vals)) if vals else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_ap": m_ap,
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "per_class_auc": pc_auc,
        "thresholds": thr.tolist(),
    }
    if class_names:
        for i, name in enumerate(class_names):
            out[f"auc_{name}"] = pc_auc.get(i, float("nan"))
    return out


def print_metrics(metrics: dict, dim=None, class_names=None) -> None:
    tag = f"[d={dim}] " if dim is not None else ""
    print(f"  {tag}macro AUC : {metrics['macro_auc']:.4f}")
    print(f"  {tag}macro F1  : {metrics['macro_f1']:.4f}")
    print(f"  {tag}macro AP  : {metrics['macro_ap']:.4f}")
    if class_names:
        for i, name in enumerate(class_names):
            print(f"      {name:6s}: {metrics['per_class_auc'].get(i, float('nan')):.4f}")
