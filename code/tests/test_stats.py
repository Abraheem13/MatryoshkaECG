"""Correctness tests for the statistical machinery."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.metrics import roc_auc_score
from mecg.analysis.stats import (fast_auc, delong_roc_test, holm_bonferroni,
                                 bootstrap_macro_auc, paired_bootstrap_diff,
                                 equivalence_test)

def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = rng.integers(50, 2000)
        y = (rng.random(n) < 0.3).astype(int)
        if y.sum() in (0, n): continue
        s = rng.random(n)
        assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-9

def test_fast_auc_with_ties():
    y = np.array([0,0,1,1,0,1]); s = np.array([.5,.5,.5,.9,.1,.9])
    assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-9

def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    n = 1000
    y = (rng.random(n) < 0.4).astype(int)
    a = y * 0.6 + rng.random(n) * 0.7
    b = y * 0.3 + rng.random(n) * 0.9
    r = delong_roc_test(y, a, b)
    assert abs(r["auc_a"] - roc_auc_score(y, a)) < 1e-6
    assert abs(r["auc_b"] - roc_auc_score(y, b)) < 1e-6
    assert r["p"] < 0.05          # a is genuinely better

def test_delong_identical_predictors():
    rng = np.random.default_rng(2)
    y = (rng.random(500) < 0.5).astype(int); s = rng.random(500)
    r = delong_roc_test(y, s, s)
    assert abs(r["diff"]) < 1e-12 and r["p"] == 1.0

def test_holm_monotone_and_bounded():
    adj, rej = holm_bonferroni([0.001, 0.02, 0.04, 0.5])
    assert all(0 <= a <= 1 for a in adj)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]
    assert rej[0] is True and rej[3] is False

def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(3)
    y = (rng.random((800, 5)) < 0.3).astype(float)
    p = y * 0.5 + rng.random((800, 5)) * 0.6
    r = bootstrap_macro_auc(y, p, n_boot=200, seed=0)
    assert r["ci_low"] <= r["auc"] <= r["ci_high"]

def test_equivalence_detects_identical():
    rng = np.random.default_rng(4)
    y = (rng.random((600, 5)) < 0.3).astype(float)
    p = y * 0.5 + rng.random((600, 5)) * 0.6
    r = equivalence_test(y, p, p.copy(), margin=0.01, n_boot=200)
    assert r["equivalent"] is True and abs(r["diff"]) < 1e-12

def test_equivalence_rejects_large_gap():
    rng = np.random.default_rng(5)
    y = (rng.random((600, 5)) < 0.3).astype(float)
    good = y * 1.5 + rng.random((600, 5)) * 0.5
    bad = rng.random((600, 5))
    r = equivalence_test(y, good, bad, margin=0.01, n_boot=200)
    assert r["equivalent"] is False

if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for f in fns:
        try:
            f(); print(f"  PASS  {f.__name__}")
        except Exception as e:
            fails += 1; print(f"  FAIL  {f.__name__}: {e}"); traceback.print_exc()
    print(f"\n{len(fns)-fails}/{len(fns)} passed")
    sys.exit(1 if fails else 0)
