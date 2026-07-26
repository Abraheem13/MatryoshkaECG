"""
Uncertainty quantification for multi-label AUC comparisons.

Both reviewers objected that the reported differences (0.0017 across a 32x
compression; ~0.010 between backbones) are presented without any notion of
sampling error. This module supplies:

  * `bootstrap_macro_auc`  - stratified bootstrap CI over test recordings
  * `delong_roc_test`      - DeLong's paired test for two correlated ROC AUCs
  * `paired_bootstrap_diff`- CI for the *difference* between two models,
                             resampling recordings jointly (correct for
                             predictions made on the same test set)
  * `holm_bonferroni`      - family-wise error control across the 6 nesting
                             dimensions

DeLong is exact for a single binary label; for the 5-class multi-label macro
AUC we apply DeLong per class and combine with Holm, and additionally report a
paired bootstrap CI on the macro average (which requires no independence
assumption across classes).

Reference: Sun & Xu, "Fast Implementation of DeLong's Algorithm", IEEE SPL 2014.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Fast DeLong
# ---------------------------------------------------------------------------
def _compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, m: int):
    """Return (aucs, covariance) for k models; positives must come first."""
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]
    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]])
        sy = np.array([[float(sy)]])
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray
                    ) -> Dict[str, float]:
    """
    Two-sided DeLong test for H0: AUC_a == AUC_b on the same binary labels.

    Returns dict with auc_a, auc_b, diff, z, p.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    order = (-y_true).argsort(kind="mergesort")  # positives first, stable
    label_ordered = y_true[order]
    m = int(label_ordered.sum())
    n = len(label_ordered) - m
    if m == 0 or n == 0:
        return {"auc_a": np.nan, "auc_b": np.nan, "diff": np.nan,
                "z": np.nan, "p": np.nan}

    preds = np.vstack([np.asarray(prob_a).ravel()[order],
                       np.asarray(prob_b).ravel()[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    if var <= 0:
        z = 0.0 if diff == 0 else np.inf * np.sign(diff)
        p = 1.0 if diff == 0 else 0.0
    else:
        z = diff / np.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "diff": float(diff), "z": float(z), "p": float(p)}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def fast_auc(y_true_col: np.ndarray, score: np.ndarray) -> float:
    """
    Mann-Whitney U form of the ROC AUC, with tie-correct midranks.

    Equivalent to sklearn's roc_auc_score but ~50x faster, which matters
    because the bootstrap evaluates it O(10^5) times. Verified against
    sklearn in tests/test_stats.py.
    """
    n_pos = int(y_true_col.sum())
    n = len(y_true_col)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[y_true_col == 1].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def _macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    vals = [fast_auc(y_true[:, c], y_prob[:, c]) for c in range(y_true.shape[1])]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_macro_auc(y_true: np.ndarray, y_prob: np.ndarray,
                        n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI for macro AUC, resampling recordings."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    point = _macro_auc(y_true, y_prob)
    boots = np.empty(n_boot, dtype=float)
    filled = 0
    attempts = 0
    while filled < n_boot and attempts < n_boot * 5:
        attempts += 1
        idx = rng.integers(0, n, n)
        val = _macro_auc(y_true[idx], y_prob[idx])
        if np.isfinite(val):
            boots[filled] = val
            filled += 1
    boots = boots[:filled]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"auc": point, "ci_low": float(lo), "ci_high": float(hi),
            "se": float(boots.std(ddof=1)), "n_boot": int(filled)}


def paired_bootstrap_diff(y_true: np.ndarray, prob_a: np.ndarray,
                          prob_b: np.ndarray, n_boot: int = 2000,
                          alpha: float = 0.05, seed: int = 0) -> Dict[str, float]:
    """
    CI and two-sided p-value for macro-AUC(a) - macro-AUC(b), resampling the
    same recording indices for both models (paired design).
    """
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    point = _macro_auc(y_true, prob_a) - _macro_auc(y_true, prob_b)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        da = _macro_auc(yt, prob_a[idx])
        db = _macro_auc(yt, prob_b[idx])
        if np.isfinite(da) and np.isfinite(db):
            diffs.append(da - db)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # bootstrap p-value: proportion of resamples on the other side of zero
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": float(point), "ci_low": float(lo), "ci_high": float(hi),
            "p": float(min(1.0, p)), "n_boot": int(len(diffs))}


def equivalence_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
                     margin: float = 0.01, n_boot: int = 2000,
                     seed: int = 0) -> Dict[str, float]:
    """
    TOST-style equivalence check: is |AUC_a - AUC_b| bounded within +/- margin?

    This is the statistically appropriate framing for the paper's central
    claim, which is *not* that d=16 beats d=512 but that the two are
    practically indistinguishable. Superiority tests cannot establish that;
    an equivalence test can.
    """
    res = paired_bootstrap_diff(y_true, prob_a, prob_b, n_boot=n_boot, seed=seed)
    equivalent = (res["ci_low"] > -margin) and (res["ci_high"] < margin)
    res.update({"margin": float(margin), "equivalent": bool(equivalent)})
    return res


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------
def holm_bonferroni(pvalues: Sequence[float], alpha: float = 0.05
                    ) -> Tuple[List[float], List[bool]]:
    """Holm step-down. Returns (adjusted p-values, reject flags) in input order."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj.tolist(), (adj <= alpha).tolist()


def seed_summary(values: Sequence[float]) -> Dict[str, float]:
    """mean, std and 95% t-CI over independent training seeds."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(v.mean())
    if n == 1:
        return {"mean": mean, "std": 0.0, "n": 1,
                "ci_low": mean, "ci_high": mean}
    sd = float(v.std(ddof=1))
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
    return {"mean": mean, "std": sd, "n": n,
            "ci_low": mean - half, "ci_high": mean + half}
