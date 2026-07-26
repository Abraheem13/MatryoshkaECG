"""
Test the Table II hierarchy claim empirically.

Pipeline:
  1. Extract physiological descriptors (RR, QRS duration, ST level, T
     amplitude, R amplitude, P amplitude, axis) from the raw waveforms.
  2. Fit ridge probes from each embedding prefix z[:m] on the TRAIN split.
  3. Evaluate out-of-sample R^2 on the TEST split.
  4. Record where each descriptor saturates and correlate the measured
     saturation order against the order Table II claims.
  5. Compute prefix CKA and per-coordinate variance for geometry context.

Whatever the outcome, it goes in the paper. If the Spearman correlation
between claimed and measured ordering is not significantly positive, Table II
must be restated as a design intention rather than a validated property.

Usage:
    python scripts/run_probing.py \
        --checkpoint results/checkpoints/inception1d_mrl_s0_best.pt \
        --out results/probing.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.analysis.features import (FEATURE_NAMES, extract_features_batch)
from mecg.analysis.probing import (effective_rank, prefix_cka_matrix,
                                   prefix_probe, slab_probe,
                                   summarise_hierarchy, variance_profile)
from mecg.data.dataset import ECGDataModule
from mecg.models.model import ECGModel


@torch.no_grad()
def embed(model, loader, device):
    Z, Y = [], []
    for x, y in loader:
        Z.append(model.embed(x.to(device, non_blocking=True)).float().cpu())
        Y.append(y)
    return torch.cat(Z).numpy(), torch.cat(Y).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="results/probing.json")
    ap.add_argument("--feature-cache", default="results/ecg_descriptors.npz")
    ap.add_argument("--max-train", type=int, default=8000,
                    help="subsample train for descriptor extraction (slow step)")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg, dims = ck["config"], ck["dims"]
    model = ECGModel(cfg, verbose=False).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])

    dm = ECGDataModule(args.data_dir, cfg)
    # MUST be the eval loaders: the training loader shuffles and drops the last
    # partial batch, so row i of its embedding matrix is not record i of
    # X_train.npy -- which would silently misalign every probe against the
    # physiological descriptors extracted below.
    train_loader, _, test_loader = dm.get_eval_loaders()

    print("[1/4] embeddings")
    Ztr, _ = embed(model, train_loader, device)
    Zte, _ = embed(model, test_loader, device)
    print(f"    train {Ztr.shape}  test {Zte.shape}")

    print("[2/4] physiological descriptors")
    if os.path.exists(args.feature_cache):
        cache = np.load(args.feature_cache)
        Ftr, Fte, idx_tr = cache["Ftr"], cache["Fte"], cache["idx_tr"]
        print("    loaded cache")
    else:
        Xtr = np.load(os.path.join(args.data_dir, "X_train.npy"), mmap_mode="r")
        Xte = np.load(os.path.join(args.data_dir, "X_test.npy"), mmap_mode="r")
        n = min(args.max_train, Xtr.shape[0])
        rng = np.random.default_rng(0)
        idx_tr = np.sort(rng.choice(Xtr.shape[0], n, replace=False))
        Ftr = extract_features_batch(np.asarray(Xtr[idx_tr]), n_jobs=args.n_jobs)
        Fte = extract_features_batch(np.asarray(Xte), n_jobs=args.n_jobs)
        os.makedirs(os.path.dirname(args.feature_cache) or ".", exist_ok=True)
        np.savez_compressed(args.feature_cache, Ftr=Ftr, Fte=Fte, idx_tr=idx_tr,
                            names=np.array(FEATURE_NAMES))
    Ztr_sub = Ztr[idx_tr]
    valid = {n: int(np.isfinite(Fte[:, i]).sum())
             for i, n in enumerate(FEATURE_NAMES)}
    print(f"    usable test descriptors: {valid}")

    print("[3/4] prefix and slab probes")
    prefix = prefix_probe(Ztr_sub, Ftr, Zte, Fte, dims)
    slab = slab_probe(Ztr_sub, Ftr, Zte, Fte, dims)
    verdict = summarise_hierarchy(prefix)

    for name in FEATURE_NAMES:
        r = prefix[name]
        row = "  ".join(f"d{d}={r['r2_by_dim'][d]:.3f}"
                        if np.isfinite(r["r2_by_dim"][d]) else f"d{d}=  n/a"
                        for d in dims)
        print(f"    {name:14s} {row}   sat={r['saturation_dim']} "
              f"claim={r['claimed_dim']}")
    print(f"    hierarchy Spearman rho={verdict['rho']:.3f} "
          f"p={verdict['p']:.4f} supported={verdict['supported']}")

    print("[4/4] geometry")
    cka = prefix_cka_matrix(Zte, dims)
    geom = {
        "cka_matrix": cka.tolist(),
        "cka_dims": dims,
        "effective_rank_full": effective_rank(Zte),
        "effective_rank_by_prefix": {str(d): effective_rank(Zte[:, :d])
                                     for d in dims},
        "variance_profile": variance_profile(Zte).tolist(),
        "variance_cumfrac_by_dim": {
            str(d): float(variance_profile(Zte)[:d].sum() /
                          variance_profile(Zte).sum()) for d in dims},
    }
    print(f"    effective rank (d=512): {geom['effective_rank_full']:.1f}")
    print(f"    variance captured by first 16 dims: "
          f"{geom['variance_cumfrac_by_dim'][str(dims[0])]:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint, "dims": dims,
            "feature_names": FEATURE_NAMES,
            "n_valid_test_descriptors": valid,
            "prefix_probe": {k: {"r2_by_dim": {str(d): v["r2_by_dim"][d]
                                               for d in dims},
                                 "best_r2": v["best_r2"],
                                 "saturation_dim": v["saturation_dim"],
                                 "claimed_dim": v["claimed_dim"]}
                             for k, v in prefix.items()},
            "slab_probe": slab,
            "hierarchy_verdict": verdict,
            "geometry": geom,
        }, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
