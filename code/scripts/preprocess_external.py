"""
External dataset preprocessing for cross-dataset evaluation.

Reviewers 1 and 2 both required evaluation beyond PTB-XL. This script converts
PhysioNet/CinC-2020-style WFDB corpora into the same (N, 12, T) tensor layout
and maps their SNOMED-CT diagnosis codes onto the five PTB-XL superclasses.

Supported (all freely downloadable, no credentialing):
    cpsc2018   - China Physiological Signal Challenge 2018   (6,877 records)
    chapman    - Chapman-Shaoxing / Ningbo                   (45,152 records)
    georgia    - Georgia 12-lead ECG Challenge database      (10,344 records)

MIMIC-IV-ECG (800k records) and CODE-15% are also supported by
`--layout mimic`, but require PhysioNet credentialed access; obtain the files
yourself and point --raw at the extracted directory.

Label mapping caveat, to be stated in the paper: the source corpora are
annotated with a different (finer, rhythm-oriented) vocabulary than PTB-XL's
diagnostic superclasses. The mapping below is many-to-one and necessarily
lossy, so cross-dataset numbers measure *transfer of the superclass concept*,
not identical-task performance. Records whose codes map to nothing are dropped.

Usage:
    python scripts/preprocess_external.py --dataset cpsc2018 \
        --raw data/external/cpsc2018 --out data/processed_cpsc2018
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List

import numpy as np
from scipy.signal import resample_poly
from tqdm import tqdm

SUPERCLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]

# SNOMED-CT -> PTB-XL superclass. Codes follow the PhysioNet/CinC 2020 table.
SNOMED_TO_SUPER: Dict[str, List[str]] = {
    "426783006": ["NORM"],                       # sinus rhythm
    "164889003": ["CD"],   "164890007": ["CD"],  # AF / atrial flutter
    "270492004": ["CD"],   "195042002": ["CD"],  # I / II AV block
    "54016002":  ["CD"],   "28189009":  ["CD"],
    "27885002":  ["CD"],   "251170000": ["CD"],
    "59118001":  ["CD"],   "713427006": ["CD"],  # RBBB / incomplete RBBB
    "59931005":  ["STTC"], "164934002": ["STTC"],  # T-wave inversion / abnormal
    "429622005": ["STTC"], "164930006": ["STTC"],  # ST depression / ST changes
    "164931005": ["STTC"], "164917005": ["STTC"],
    "426177001": ["NORM"],                        # sinus bradycardia
    "427084000": ["NORM"], "427393009": ["NORM"], # sinus tachy / arrhythmia
    "164865005": ["MI"],   "164861001": ["MI"],   # MI / ischaemia
    "57054005":  ["MI"],   "413444003": ["MI"],
    "164873001": ["HYP"],  "164877002": ["HYP"],  # LVH / hypertrophy
    "266249003": ["HYP"],  "89792004":  ["HYP"],
    "39732003":  ["HYP"],  "446358003": ["HYP"],
    "164909002": ["CD"],   "445118002": ["CD"],   # LBBB / LAFB
    "251146004": ["STTC"], "365413008": ["STTC"],
}


def parse_header(path: str):
    """Read a WFDB .hea file for sampling rate, gain and Dx codes."""
    fs, gains, dx = None, [], []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if i == 0:
                parts = line.split()
                if len(parts) >= 3:
                    fs = float(parts[2])
            elif line.startswith("#Dx:"):
                dx = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
            elif not line.startswith("#") and i > 0:
                parts = line.split()
                if len(parts) >= 3 and "/" in parts[2]:
                    try:
                        gains.append(float(parts[2].split("/")[0]))
                    except ValueError:
                        gains.append(1000.0)
    return fs, gains, dx


def load_record(base: str, target_fs: int = 100, target_len: int = 1000):
    import wfdb
    sig, meta = wfdb.rdsamp(base)          # (T, 12) in physical units (mV)
    fs = int(meta["fs"])
    x = np.asarray(sig, dtype=np.float32).T  # (12, T)
    if x.shape[0] < 12:
        return None
    x = x[:12]
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if fs != target_fs:
        from math import gcd
        g = gcd(int(fs), int(target_fs))
        x = resample_poly(x, target_fs // g, fs // g, axis=1).astype(np.float32)

    T = x.shape[1]
    if T >= target_len:                    # centre crop
        s = (T - target_len) // 2
        x = x[:, s:s + target_len]
    else:                                  # zero pad
        pad = target_len - T
        x = np.pad(x, ((0, 0), (pad // 2, pad - pad // 2)))
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["cpsc2018", "chapman", "georgia", "custom"])
    ap.add_argument("--raw", required=True, help="directory containing .hea/.mat")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fs", type=int, default=100)
    ap.add_argument("--target-len", type=int, default=1000)
    ap.add_argument("--max-records", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    headers = []
    for root, _, files in os.walk(args.raw):
        for fn in files:
            if fn.endswith(".hea"):
                headers.append(os.path.join(root, fn))
    headers.sort()
    if args.max_records:
        headers = headers[:args.max_records]
    print(f"  found {len(headers)} records in {args.raw}")

    X_list, Y_list, kept, dropped = [], [], 0, 0
    for hea in tqdm(headers, desc="  converting"):
        base = hea[:-4]
        _, _, dx = parse_header(hea)
        supers = set()
        for code in dx:
            supers.update(SNOMED_TO_SUPER.get(code, []))
        if not supers:
            dropped += 1
            continue
        try:
            x = load_record(base, args.target_fs, args.target_len)
        except Exception:
            dropped += 1
            continue
        if x is None:
            dropped += 1
            continue
        y = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for s in supers:
            y[SUPERCLASSES.index(s)] = 1.0
        X_list.append(x)
        Y_list.append(y)
        kept += 1

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    print(f"  kept {kept}, dropped {dropped} (unmappable or unreadable)")
    for i, c in enumerate(SUPERCLASSES):
        n = int(Y[:, i].sum())
        print(f"    {c:5s}: {n:6d} ({100*n/len(Y):5.1f}%)")

    # external corpora are used purely as held-out test sets
    np.save(os.path.join(args.out, "X_test.npy"), X)
    np.save(os.path.join(args.out, "y_test.npy"), Y)
    with open(os.path.join(args.out, "metadata.pkl"), "wb") as f:
        pickle.dump({"dataset": args.dataset, "num_classes": len(SUPERCLASSES),
                     "idx_to_class": {i: c for i, c in enumerate(SUPERCLASSES)},
                     "class_to_idx": {c: i for i, c in enumerate(SUPERCLASSES)},
                     "n_records": int(len(X)), "units": "mV (raw)",
                     "mapping": "SNOMED-CT -> PTB-XL superclass (lossy)"}, f)
    print("  done ->", args.out)


if __name__ == "__main__":
    main()
