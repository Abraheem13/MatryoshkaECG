"""
Representation probing: does the embedding actually organise coarse-to-fine?

Three complementary analyses, all fitted on the TRAIN split and evaluated on
the TEST split (no leakage -- unlike the SVD baseline in the original
submission, which fitted and evaluated on the same partition):

1. `prefix_probe`  - ridge regression from z[:m] to each physiological
   descriptor. Saturation point = smallest m reaching 95% of the best R^2
   achieved at any m. Compare against Table II's claim.

2. `slab_probe`    - probe restricted to the *new* coordinates
   z[m_{i-1}:m_i]. Answers "what does each shell add?" rather than "what does
   each prefix contain", which is the sharper question.

3. `linear_cka`    - similarity between prefix representations, quantifying
   how much genuinely new geometry each shell introduces.

A supporting result looks like: saturation_dim(rr) < saturation_dim(qrs) <
saturation_dim(st) < ... A null result (all descriptors saturating at d=16)
means the hierarchy is not a real property of the representation and Table II
must be withdrawn or restated as a design intention.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, HIERARCHY_CLAIM


MIN_PROBE_SAMPLES = 50


def _fit_eval_ridge(Ztr, ytr, Zte, yte, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0),
                    min_samples: int = MIN_PROBE_SAMPLES):
    """
    Out-of-sample R^2 for a ridge probe, or NaN if too few records survive
    delineation. Returning NaN rather than a noisy estimate is deliberate:
    a probe fitted on a handful of records would produce an R^2 that looks
    like evidence about the hierarchy but is not.
    """
    ok_tr = np.isfinite(ytr)
    ok_te = np.isfinite(yte)
    if ok_tr.sum() < min_samples or ok_te.sum() < min_samples:
        return float("nan")
    sx = StandardScaler().fit(Ztr[ok_tr])
    ytr_c = ytr[ok_tr]
    mu, sd = ytr_c.mean(), ytr_c.std() + 1e-8
    model = RidgeCV(alphas=list(alphas))
    model.fit(sx.transform(Ztr[ok_tr]), (ytr_c - mu) / sd)
    pred = model.predict(sx.transform(Zte[ok_te])) * sd + mu
    resid = np.sum((yte[ok_te] - pred) ** 2)
    total = np.sum((yte[ok_te] - yte[ok_te].mean()) ** 2)
    return float(1.0 - resid / total) if total > 0 else float("nan")


def prefix_probe(Z_train: np.ndarray, F_train: np.ndarray,
                 Z_test: np.ndarray, F_test: np.ndarray,
                 dims: List[int],
                 feature_names: Optional[List[str]] = None) -> Dict[str, dict]:
    """
    R^2 of each descriptor from each prefix, plus the saturation dimension.
    """
    feature_names = feature_names or FEATURE_NAMES
    results: Dict[str, dict] = {}
    for fi, fname in enumerate(feature_names):
        n_tr = int(np.isfinite(F_train[:, fi]).sum())
        n_te = int(np.isfinite(F_test[:, fi]).sum())
        if min(n_tr, n_te) < MIN_PROBE_SAMPLES:
            print(f"    [skip] {fname}: only {n_tr} train / {n_te} test records "
                  f"survived delineation (need {MIN_PROBE_SAMPLES}); probe "
                  f"would be too noisy to interpret")
        r2_by_dim = {}
        for m in dims:
            r2_by_dim[m] = _fit_eval_ridge(
                Z_train[:, :m], F_train[:, fi], Z_test[:, :m], F_test[:, fi]
            )
        finite = {m: v for m, v in r2_by_dim.items() if np.isfinite(v)}
        if finite:
            best = max(finite.values())
            thresh = 0.95 * best if best > 0 else best
            sat = min([m for m, v in finite.items() if v >= thresh])
        else:
            best, sat = float("nan"), None
        results[fname] = {
            "r2_by_dim": r2_by_dim,
            "best_r2": best,
            "saturation_dim": sat,
            "claimed_dim": HIERARCHY_CLAIM.get(fname),
        }
    return results


def slab_probe(Z_train: np.ndarray, F_train: np.ndarray,
               Z_test: np.ndarray, F_test: np.ndarray,
               dims: List[int],
               feature_names: Optional[List[str]] = None) -> Dict[str, dict]:
    """R^2 from the incremental coordinate block [prev_dim : dim)."""
    feature_names = feature_names or FEATURE_NAMES
    bounds = [(0, dims[0])] + [(dims[i - 1], dims[i]) for i in range(1, len(dims))]
    results: Dict[str, dict] = {}
    for fi, fname in enumerate(feature_names):
        per_slab = {}
        for lo, hi in bounds:
            per_slab[f"{lo}-{hi}"] = _fit_eval_ridge(
                Z_train[:, lo:hi], F_train[:, fi],
                Z_test[:, lo:hi], F_test[:, fi]
            )
        results[fname] = {"r2_by_slab": per_slab}
    return results


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two representations of the same samples."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xty = np.linalg.norm(X.T @ Y, ord="fro") ** 2
    xx = np.linalg.norm(X.T @ X, ord="fro")
    yy = np.linalg.norm(Y.T @ Y, ord="fro")
    return float(xty / (xx * yy)) if xx > 0 and yy > 0 else float("nan")


def prefix_cka_matrix(Z: np.ndarray, dims: List[int]) -> np.ndarray:
    """CKA between every pair of prefixes."""
    n = len(dims)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = linear_cka(Z[:, :dims[i]], Z[:, :dims[j]])
    return M


def effective_rank(Z: np.ndarray) -> float:
    """Entropy-based effective rank of the embedding covariance spectrum."""
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def variance_profile(Z: np.ndarray) -> np.ndarray:
    """Per-coordinate variance -- shows whether leading dims carry more signal."""
    return Z.var(axis=0)


def summarise_hierarchy(prefix_results: Dict[str, dict]) -> dict:
    """
    Turn probe output into a verdict on Table II.

    Reports Spearman correlation between the *claimed* dimension ordering and
    the *measured* saturation ordering. A high positive correlation supports
    the hierarchy; near-zero or negative refutes it.
    """
    from scipy.stats import spearmanr

    claimed, measured, names = [], [], []
    for fname, res in prefix_results.items():
        c, s = res.get("claimed_dim"), res.get("saturation_dim")
        if c is not None and s is not None:
            claimed.append(c)
            measured.append(s)
            names.append(fname)
    if len(claimed) < 3:
        return {"n": len(claimed), "rho": float("nan"), "p": float("nan"),
                "supported": False, "features": names}
    rho, p = spearmanr(claimed, measured)
    return {"n": len(claimed), "rho": float(rho), "p": float(p),
            "supported": bool(np.isfinite(rho) and rho > 0.5 and p < 0.05),
            "features": names, "claimed": claimed, "measured": measured}
