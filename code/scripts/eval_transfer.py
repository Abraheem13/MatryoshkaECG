"""
Evaluation beyond the PTB-XL test fold:

  --mode svd         leakage-free SVD/PCA compression baseline
  --mode external    zero-shot transfer to CPSC2018 / Chapman / Georgia
  --mode robustness  wearable-like artefact sweep at controlled SNR
  --mode leads       reduced-lead (including single-lead) evaluation

The SVD baseline in the original submission fitted the PCA basis *and* the
logistic-regression classifier on the test partition and then evaluated on the
same partition. The manuscript acknowledged this as "optimistically biased",
but it was still plotted as an upper bound and compared against honestly
evaluated models. Here the basis and classifier are fitted on the training
split and applied unchanged to the test split.

Usage:
  python scripts/eval_transfer.py --mode svd \
      --checkpoint results/checkpoints/inception1d_mrl_s0_best.pt
  python scripts/eval_transfer.py --mode external \
      --checkpoint ... --external-dir data/processed_cpsc2018 --tag cpsc2018
  python scripts/eval_transfer.py --mode robustness --checkpoint ...
  python scripts/eval_transfer.py --mode leads --checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.analysis.metrics import compute_all_metrics, macro_auc
from mecg.analysis.stats import bootstrap_macro_auc
from mecg.data.dataset import (ECGDataModule, ECGDataset, LEAD_SUBSETS,
                               compute_norm_stats)
from mecg.models.model import ECGModel


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ECGModel(ck["config"], verbose=False).to(device).eval()
    model.load_state_dict(ck["model_state_dict"])
    return model, ck["config"], ck["dims"], ck.get("class_names")


@torch.no_grad()
def embed_loader(model, loader, device):
    Z, Y = [], []
    for x, y in tqdm(loader, desc="    embedding", leave=False):
        Z.append(model.embed(x.to(device, non_blocking=True)).float().cpu())
        Y.append(y)
    return torch.cat(Z).numpy(), torch.cat(Y).numpy()


@torch.no_grad()
def score(model, Z, device, dims, chunk=8192):
    out = {}
    for d in dims:
        parts = []
        for i in range(0, Z.shape[0], chunk):
            zz = torch.from_numpy(Z[i:i + chunk]).to(device)
            parts.append(torch.sigmoid(model.head.logits_at(zz, d)).float().cpu())
        out[d] = torch.cat(parts).numpy()
    return out


# ---------------------------------------------------------------------------
def mode_svd(model, cfg, dims, args, device):
    """Fit PCA + logistic regression on TRAIN, evaluate on TEST."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier

    dm = ECGDataModule(args.data_dir, cfg)
    train_loader, _, test_loader = dm.get_eval_loaders()
    Ztr, Ytr = embed_loader(model, train_loader, device)
    Zte, Yte = embed_loader(model, test_loader, device)

    mu = Ztr.mean(0, keepdims=True)
    pca = PCA(n_components=max(dims), svd_solver="randomized",
              random_state=0).fit(Ztr - mu)

    results = {}
    for d in dims:
        Ptr = (Ztr - mu) @ pca.components_[:d].T
        Pte = (Zte - mu) @ pca.components_[:d].T
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=1.0), n_jobs=-1).fit(Ptr, Ytr)
        probs = clf.predict_proba(Pte)
        if isinstance(probs, list):
            probs = np.column_stack([p[:, 1] for p in probs])
        m = compute_all_metrics(Yte, probs)
        boot = bootstrap_macro_auc(Yte, probs, seed=0)
        results[str(d)] = {"macro_auc": m["macro_auc"], "macro_f1": m["macro_f1"],
                           "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
                           "explained_variance":
                               float(pca.explained_variance_ratio_[:d].sum())}
        print(f"    SVD d={d:4d}: AUC={m['macro_auc']:.4f} "
              f"[{boot['ci_low']:.4f},{boot['ci_high']:.4f}] "
              f"EV={results[str(d)]['explained_variance']:.3f}")
    return {"svd_no_leakage": results}


def mode_external(model, cfg, dims, args, device):
    # normalisation statistics come from the ORIGINAL training corpus, as they
    # would at deployment time -- not recomputed on the target corpus
    dm = ECGDataModule(args.data_dir, cfg,
                       lead_subset=cfg.get("data", {}).get("lead_subset"))
    loader = dm.get_external_loader(args.external_dir)
    Z, Y = embed_loader(model, loader, device)
    probs = score(model, Z, device, dims)
    out = {}
    for d in dims:
        m = compute_all_metrics(Y, probs[d])
        b = bootstrap_macro_auc(Y, probs[d], seed=0)
        out[str(d)] = {"macro_auc": m["macro_auc"], "macro_f1": m["macro_f1"],
                       "ci_low": b["ci_low"], "ci_high": b["ci_high"],
                       "per_class_auc": {str(k): v
                                         for k, v in m["per_class_auc"].items()}}
        print(f"    {args.tag} d={d:4d}: AUC={m['macro_auc']:.4f} "
              f"[{b['ci_low']:.4f},{b['ci_high']:.4f}]")
    return {f"external_{args.tag}": {"n_records": int(len(Y)), "by_dim": out}}


def mode_robustness(model, cfg, dims, args, device):
    out = {}
    # clean reference first, so degradation is reported relative to it
    dm0 = ECGDataModule(args.data_dir, cfg)
    _, _, clean_loader = dm0.get_eval_loaders()
    Z, Y = embed_loader(model, clean_loader, device)
    probs = score(model, Z, device, dims)
    out["clean"] = {str(d): macro_auc(Y, probs[d]) for d in dims}
    print(f"    {'clean':17s}           d{dims[0]}={out['clean'][str(dims[0])]:.4f}  "
          f"d{dims[-1]}={out['clean'][str(dims[-1])]:.4f}")

    for kind in ["baseline_wander", "muscle", "electrode_motion", "powerline",
                 "mixed"]:
        for snr in [20, 15, 10, 5, 0]:
            dm = ECGDataModule(args.data_dir, cfg,
                               test_corruption={"kind": kind, "snr_db": snr,
                                                "fs": 100, "seed": 1234})
            _, _, test_loader = dm.get_eval_loaders()
            Z, Y = embed_loader(model, test_loader, device)
            probs = score(model, Z, device, dims)
            key = f"{kind}_snr{snr}"
            out[key] = {str(d): macro_auc(Y, probs[d]) for d in dims}
            print(f"    {kind:17s} SNR={snr:3d}dB  "
                  f"d{dims[0]}={out[key][str(dims[0])]:.4f}  "
                  f"d{dims[-1]}={out[key][str(dims[-1])]:.4f}")
    return {"robustness": out}


def mode_leads(model, cfg, dims, args, device):
    """
    Zero-shot lead ablation: zero out the missing leads at inference.
    (Models trained *natively* on reduced-lead input are produced separately by
    train.py --lead-subset; this measures graceful degradation of the 12-lead
    model, which is the realistic fallback when a wearable streams one lead.)
    """
    out = {}
    dm = ECGDataModule(args.data_dir, cfg)
    _, _, test_loader = dm.get_eval_loaders()
    for name, idx in LEAD_SUBSETS.items():
        Z, Y = [], []
        keep = torch.zeros(12)
        keep[idx] = 1.0
        with torch.no_grad():
            for x, y in test_loader:
                x = (x * keep[None, :, None]).to(device, non_blocking=True)
                Z.append(model.embed(x).float().cpu())
                Y.append(y)
        Z = torch.cat(Z).numpy()
        Y = torch.cat(Y).numpy()
        probs = score(model, Z, device, dims)
        out[name] = {str(d): macro_auc(Y, probs[d]) for d in dims}
        print(f"    {name:8s} ({len(idx):2d} leads): "
              f"d{dims[0]}={out[name][str(dims[0])]:.4f}  "
              f"d{dims[-1]}={out[name][str(dims[-1])]:.4f}")
    return {"lead_ablation": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["svd", "external", "robustness", "leads"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--external-dir", default=None)
    ap.add_argument("--tag", default="external")
    ap.add_argument("--out", default="results/transfer.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, dims, _ = load_model(args.checkpoint, device)
    print(f"  {args.mode}: {args.checkpoint} on {device}")

    fn = {"svd": mode_svd, "external": mode_external,
          "robustness": mode_robustness, "leads": mode_leads}[args.mode]
    result = fn(model, cfg, dims, args, device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    existing = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
    existing.update(result)
    with open(args.out, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
