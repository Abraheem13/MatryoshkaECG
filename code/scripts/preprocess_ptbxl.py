"""
PTB-XL preprocessing.

Key change: signals are saved in RAW physical units (mV). Normalisation is
applied at load time by `ECGDataModule` according to `data.norm_mode`, so that
per-record and dataset-level standardisation can be compared without
re-running preprocessing. The original pipeline baked per-record z-scoring
into the .npy files, which made that ablation impossible and discarded
absolute voltage information needed for hypertrophy.

Usage:
    python scripts/preprocess_ptbxl.py --raw data/ptbxl_raw --out data/processed
"""

from __future__ import annotations

import argparse
import ast
import os
import pickle
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPERCLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]


def load_raw(df, sampling_rate, path):
    import wfdb
    col = df.filename_lr if sampling_rate == 100 else df.filename_hr
    sigs = []
    for f in tqdm(col, desc=f"  reading {sampling_rate} Hz"):
        sig, _ = wfdb.rdsamp(os.path.join(path, str(f).replace(".dat", "")))
        sigs.append(sig)
    return np.asarray(sigs, dtype=np.float32)


def aggregate_labels(df, agg_df, task="superdiagnostic"):
    def _agg(codes):
        out = set()
        for key in codes:
            if key in agg_df.index:
                if task == "superdiagnostic":
                    cat = agg_df.loc[key].diagnostic_class
                elif task == "subdiagnostic":
                    cat = agg_df.loc[key].diagnostic_subclass
                else:
                    cat = key
                if isinstance(cat, str):
                    out.add(cat)
        return sorted(out)
    return df.scp_codes.apply(_agg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/ptbxl_raw")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--sampling-rate", type=int, default=100)
    ap.add_argument("--task", default="superdiagnostic",
                    choices=["superdiagnostic", "subdiagnostic", "diagnostic"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("[1/5] metadata")
    df = pd.read_csv(os.path.join(args.raw, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    agg_df = pd.read_csv(os.path.join(args.raw, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]
    print(f"    records: {len(df)}")

    print("[2/5] waveforms (raw mV, no normalisation)")
    X = load_raw(df, args.sampling_rate, args.raw)
    print(f"    shape: {X.shape}")

    print("[3/5] labels")
    df["labels"] = aggregate_labels(df, agg_df, args.task)
    keep = df["labels"].apply(len) > 0
    df, X = df[keep], X[keep.values]

    classes = sorted({c for row in df["labels"] for c in row})
    cls_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(df), len(classes)), dtype=np.float32)
    for i, row in enumerate(df["labels"]):
        for c in row:
            Y[i, cls_idx[c]] = 1.0
    print(f"    kept {len(df)} recordings, {len(classes)} classes: {classes}")
    for c, i in cls_idx.items():
        n = int(Y[:, i].sum())
        print(f"      {c:6s}: {n:6d} ({100*n/len(Y):5.1f}%)")

    print("[4/5] official fold split")
    folds = df.strat_fold.values
    masks = {"train": np.isin(folds, range(1, 9)), "val": folds == 9,
             "test": folds == 10}

    # (N, T, 12) -> (N, 12, T)
    X = np.transpose(X, (0, 2, 1)).astype(np.float32)

    print("[5/5] saving")
    for split, m in masks.items():
        np.save(os.path.join(args.out, f"X_{split}.npy"), X[m])
        np.save(os.path.join(args.out, f"y_{split}.npy"), Y[m])
        print(f"    {split:5s}: {int(m.sum()):6d}")
        # patient ids enable patient-level bootstrap if required
        np.save(os.path.join(args.out, f"pid_{split}.npy"),
                df.patient_id.values[m])

    meta = {
        "task": args.task,
        "sampling_rate": args.sampling_rate,
        "num_classes": len(classes),
        "class_to_idx": cls_idx,
        "idx_to_class": {i: c for c, i in cls_idx.items()},
        "signal_length": X.shape[2],
        "units": "mV (raw, unnormalised)",
        "n_patients": int(df.patient_id.nunique()),
    }
    with open(os.path.join(args.out, "metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print("  done ->", args.out)


if __name__ == "__main__":
    main()
