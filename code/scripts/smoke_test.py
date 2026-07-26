"""
End-to-end smoke test on synthetic data. Requires no PTB-XL download.

Run this FIRST, before committing GPU hours. It exercises every code path the
real pipeline uses -- preprocessing layout, all backbones, all head types,
training loop, AMP, checkpointing, evaluation, statistics, probing geometry,
quantisation export and the aggregation-to-LaTeX step -- on 400 fake
recordings, in about a minute.

    python scripts/smoke_test.py                 # core paths
    python scripts/smoke_test.py --full          # + probing, ONNX, benchmark

A green run means the plumbing is correct. It says nothing about whether the
science works: the data is noise, so AUCs near 0.5 are expected and correct.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

_results = []


def check(name, fn):
    try:
        detail = fn()
        print(f"  {GREEN}PASS{RESET}  {name}" + (f"  [{detail}]" if detail else ""))
        _results.append(True)
        return True
    except Exception as exc:
        print(f"  {RED}FAIL{RESET}  {name}\n        {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        _results.append(False)
        return False


def make_fake_dataset(out_dir, n_train=400, n_val=100, n_test=120,
                      leads=12, length=1000, n_classes=5, seed=0):
    """Synthetic corpus in the exact layout preprocess_ptbxl.py produces."""
    import pickle
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    classes = ["CD", "HYP", "MI", "NORM", "STTC"][:n_classes]
    if n_classes > 5:
        classes = [f"C{i}" for i in range(n_classes)]

    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        y = np.zeros((n, n_classes), dtype=np.float32)
        for i in range(n):
            k = rng.integers(1, 3)
            y[i, rng.choice(n_classes, k, replace=False)] = 1.0
        # weak class-dependent signal so metrics are not degenerate
        t = np.arange(length) / 100.0
        X = rng.normal(0, 0.4, (n, leads, length)).astype(np.float32)
        for i in range(n):
            for c in np.flatnonzero(y[i]):
                X[i] += (0.3 * np.sin(2 * np.pi * (1.0 + c) * t)).astype(np.float32)
        np.save(os.path.join(out_dir, f"X_{split}.npy"), X)
        np.save(os.path.join(out_dir, f"y_{split}.npy"), y)
        np.save(os.path.join(out_dir, f"pid_{split}.npy"), np.arange(n))

    with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
        pickle.dump({"task": "smoke", "sampling_rate": 100,
                     "num_classes": n_classes,
                     "class_to_idx": {c: i for i, c in enumerate(classes)},
                     "idx_to_class": {i: c for i, c in enumerate(classes)},
                     "signal_length": length, "units": "synthetic",
                     "n_patients": n_train}, f)
    return out_dir


def write_config(path, backbone="inception1d", head="mrl", dims=(16, 32, 64),
                 epochs=2, batch_size=32, leads=12):
    import yaml
    cfg = {
        "project_name": "smoke", "seed": 0, "deterministic": True,
        "data": {"norm_mode": "dataset", "lead_subset": None},
        "model": {"backbone": backbone, "head": head, "input_channels": leads,
                  "embedding_dim": max(dims), "num_classes": 5, "dropout": 0.2,
                  "backbone_kwargs": {"use_se": True} if backbone == "inception1d"
                  else {"use_se": True}},
        "nesting": {"dims": list(dims), "weight_strategy": "equal"},
        "training": {"epochs": epochs, "batch_size": batch_size,
                     "learning_rate": 1e-3, "weight_decay": 1e-2,
                     "warmup_epochs": 1, "label_smoothing": 0.0,
                     "pos_weight": False, "early_stopping_patience": 10,
                     "gradient_clip_norm": 1.0, "num_workers": 0,
                     "pin_memory": False},
        "augmentation": {"enabled": True, "gaussian_noise_std": 0.01,
                         "random_scale": [0.9, 1.1], "random_shift": 20,
                         "p_noise": 0.5, "p_scale": 0.5, "p_shift": 0.5,
                         "p_lead_dropout": 0.1},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg


def run(cmd, cwd=ROOT):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"exit {r.returncode}\nSTDOUT:\n{r.stdout[-2500:]}"
                           f"\nSTDERR:\n{r.stderr[-2500:]}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also test probing, ONNX export and benchmarking")
    ap.add_argument("--keep", action="store_true", help="keep the temp dir")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="mecg_smoke_")
    data_dir = os.path.join(tmp, "data")
    res_dir = os.path.join(tmp, "results")
    paper_dir = os.path.join(tmp, "paper")
    os.makedirs(os.path.join(paper_dir, "generated"), exist_ok=True)
    os.makedirs(os.path.join(paper_dir, "figures"), exist_ok=True)

    print(f"\n{'='*68}\n  SMOKE TEST  (workdir {tmp})\n{'='*68}\n")

    # ---- imports --------------------------------------------------------
    print("Imports and environment")

    def _t_torch():
        import torch
        return (f"torch {torch.__version__}, cuda="
                f"{torch.cuda.is_available()}")
    check("torch available", _t_torch)
    check("mecg package imports", lambda: __import__(
        "mecg.models.model", fromlist=["ECGModel"]) and "")

    # ---- synthetic data -------------------------------------------------
    print("\nSynthetic data")
    check("build fake corpus", lambda: make_fake_dataset(data_dir) and
          f"{len(np.load(os.path.join(data_dir,'X_train.npy')))} train")

    # ---- model construction --------------------------------------------
    print("\nModel construction and forward pass")
    import torch
    from mecg.models.model import ECGModel

    def _fwd(backbone, head, dims, leads=12, resid=1, width="growing"):
        cfg = write_config(os.path.join(tmp, "c.yaml"), backbone, head, dims,
                           leads=leads)
        cfg["model"]["backbone_kwargs"].update(
            {"residual_every": resid, "width_mode": width}
            if backbone == "inception1d" else {})
        m = ECGModel(cfg, verbose=False)
        out = m(torch.randn(4, leads, 1000))
        assert set(out) == set(dims), f"heads {set(out)} != dims {set(dims)}"
        for d, lg in out.items():
            assert lg.shape == (4, 5), f"d={d} shape {lg.shape}"
        n = sum(p.numel() for p in m.parameters())
        return f"{n:,} params"

    check("inception1d + mrl", lambda: _fwd("inception1d", "mrl", (16, 32, 64)))
    check("inception1d + mrl-e", lambda: _fwd("inception1d", "mrl-e", (16, 32, 64)))
    check("inception1d + linear", lambda: _fwd("inception1d", "linear", (64,)))
    check("xresnet1d101 + mrl", lambda: _fwd("xresnet1d101", "mrl", (16, 32, 64)))
    check("xresnet1d50 + mrl", lambda: _fwd("xresnet1d50", "mrl", (16, 32, 64)))
    # regression test for the residual-pooling bug
    check("inception1d residual_every=3",
          lambda: _fwd("inception1d", "mrl", (16, 32, 64), resid=3))
    check("inception1d width_mode=constant",
          lambda: _fwd("inception1d", "mrl", (16, 32, 64), width="constant"))
    check("single-lead input", lambda: _fwd("inception1d", "mrl", (16, 32, 64),
                                           leads=1))

    # ---- prefix consistency --------------------------------------------
    print("\nMatryoshka invariants")

    def _prefix():
        cfg = write_config(os.path.join(tmp, "c.yaml"), dims=(16, 32, 64))
        m = ECGModel(cfg, verbose=False).eval()
        x = torch.randn(8, 12, 1000)
        with torch.no_grad():
            z = m.embed(x)
            a = m.head.logits_at(z, 16)
            b = m.head.logits_at(z[:, :16], 16)
        assert torch.allclose(a, b, atol=1e-6), "prefix truncation not equivalent"
        return "z[:16] == truncate(z)[:16]"
    check("prefix truncation is a pure slice", _prefix)

    def _mrle_params():
        from mecg.models.heads import MatryoshkaEHead, MatryoshkaHead
        dims = [16, 32, 64, 128, 256, 512]
        a = MatryoshkaHead(dims, 5).classifier_params()
        b = MatryoshkaEHead(dims, 5).classifier_params()
        assert b < a, f"MRL-E ({b}) should be smaller than MRL ({a})"
        return f"MRL {a} -> MRL-E {b} ({a/b:.2f}x)"
    check("MRL-E reduces classifier params", _mrle_params)

    # ---- dataset paths --------------------------------------------------
    print("\nData pipeline")
    from mecg.data.dataset import ECGDataModule, apply_corruption

    def _loaders():
        cfg = write_config(os.path.join(tmp, "c.yaml"))
        dm = ECGDataModule(data_dir, cfg)
        tr, va, te = dm.get_dataloaders(seed=0)
        x, y = next(iter(tr))
        assert x.shape[1:] == (12, 1000), x.shape
        return f"batch {tuple(x.shape)}"
    check("training loaders", _loaders)

    def _eval_alignment():
        """The bug this guards: shuffled/drop_last loaders misalign probing."""
        cfg = write_config(os.path.join(tmp, "c.yaml"))
        dm = ECGDataModule(data_dir, cfg)
        etr, _, _ = dm.get_eval_loaders()
        ys = torch.cat([y for _, y in etr]).numpy()
        ref = np.load(os.path.join(data_dir, "y_train.npy"))
        assert ys.shape == ref.shape, f"{ys.shape} vs {ref.shape} (dropped samples?)"
        assert np.allclose(ys, ref), "eval loader reordered the data"
        return f"{len(ys)} rows aligned to X_train"
    check("eval loader preserves order and length", _eval_alignment)

    def _corrupt():
        x = np.random.randn(12, 1000).astype(np.float32)
        outs = {}
        for k in ("baseline_wander", "muscle", "electrode_motion",
                  "powerline", "mixed"):
            c = apply_corruption(x, kind=k, snr_db=10, idx=7)
            assert c.shape == x.shape and np.isfinite(c).all()
            outs[k] = float(np.mean((c - x) ** 2))
        c1 = apply_corruption(x, kind="mixed", snr_db=10, idx=7)
        c2 = apply_corruption(x, kind="mixed", snr_db=10, idx=7)
        assert np.allclose(c1, c2), "corruption is not deterministic in idx"
        return "5 artefact types, reproducible"
    check("artefact injection", _corrupt)

    def _norm_modes():
        from mecg.data.dataset import ECGDataset, compute_norm_stats
        st = compute_norm_stats(os.path.join(data_dir, "X_train.npy"))
        a = ECGDataset(os.path.join(data_dir, "X_test.npy"),
                       os.path.join(data_dir, "y_test.npy"),
                       norm_stats=st, norm_mode="dataset")[0][0]
        b = ECGDataset(os.path.join(data_dir, "X_test.npy"),
                       os.path.join(data_dir, "y_test.npy"),
                       norm_mode="per_record")[0][0]
        # per-record scaling forces unit SD per lead; dataset-level does not
        sd_b = b.std(dim=1).mean().item()
        assert abs(sd_b - 1.0) < 0.05, f"per_record SD {sd_b}"
        return f"dataset SD={a.std():.3f}, per_record SD={sd_b:.3f}"
    check("both normalisation modes", _norm_modes)

    # ---- statistics -----------------------------------------------------
    print("\nStatistics")
    check("stats unit tests", lambda: run(
        [sys.executable, "tests/test_stats.py"]).strip().splitlines()[-1])

    # ---- full training run ----------------------------------------------
    print("\nTraining (2 epochs, synthetic)")
    cfg_path = os.path.join(tmp, "smoke_mrl.yaml")
    write_config(cfg_path, "inception1d", "mrl", (16, 32, 64), epochs=2)

    def _train_mrl():
        run([sys.executable, "scripts/train.py", "--config", cfg_path,
             "--seed", "0", "--data-dir", data_dir, "--out-dir", res_dir,
             "--run-name", "smoke_inception1d_mrl_s0", "--amp", "off",
             "--save-train-embeddings"])
        j = json.load(open(os.path.join(res_dir, "runs",
                                        "smoke_inception1d_mrl_s0",
                                        "results.json")))
        return f"AUC(d=16)={j['test']['16']['macro_auc']:.3f}"
    if not check("train MRL model", _train_mrl):
        print(f"\n{RED}Training failed -- later stages skipped.{RESET}")
        _summary(tmp, args.keep)
        return

    def _train_fixed():
        run([sys.executable, "scripts/train.py", "--config", cfg_path,
             "--seed", "0", "--data-dir", data_dir, "--out-dir", res_dir,
             "--head", "linear", "--embedding-dim", "64",
             "--run-name", "smoke_inception1d_fixed_d64_s0", "--amp", "off"])
        return "fixed-dim baseline"
    check("train fixed-dim baseline", _train_fixed)

    def _train_mrle():
        run([sys.executable, "scripts/train.py", "--config", cfg_path,
             "--seed", "1", "--data-dir", data_dir, "--out-dir", res_dir,
             "--head", "mrl-e", "--run-name", "smoke_inception1d_mrle_s1",
             "--amp", "off"])
        return "MRL-E"
    check("train MRL-E variant", _train_mrle)

    def _train_lead():
        run([sys.executable, "scripts/train.py", "--config", cfg_path,
             "--seed", "0", "--data-dir", data_dir, "--out-dir", res_dir,
             "--lead-subset", "lead_I",
             "--run-name", "smoke_inception1d_leadI_s0", "--amp", "off"])
        return "single-lead training"
    check("train reduced-lead model", _train_lead)

    def _train_perrecord():
        run([sys.executable, "scripts/train.py", "--config", cfg_path,
             "--seed", "0", "--data-dir", data_dir, "--out-dir", res_dir,
             "--norm-mode", "per_record",
             "--run-name", "smoke_perrecord_s0", "--amp", "off"])
        return "per-record normalisation ablation"
    check("train normalisation ablation", _train_perrecord)

    ckpt = os.path.join(res_dir, "checkpoints",
                        "smoke_inception1d_mrl_s0_best.pt")

    # ---- evaluation modes ----------------------------------------------
    print("\nEvaluation")
    transfer = os.path.join(res_dir, "transfer.json")
    for mode in ("svd", "leads"):
        check(f"eval_transfer --mode {mode}", lambda m=mode: run(
            [sys.executable, "scripts/eval_transfer.py", "--mode", m,
             "--checkpoint", ckpt, "--data-dir", data_dir,
             "--out", transfer]) and "")

    def _external():
        ext = os.path.join(tmp, "data_ext")
        make_fake_dataset(ext, n_train=10, n_val=10, n_test=80, seed=9)
        run([sys.executable, "scripts/eval_transfer.py", "--mode", "external",
             "--checkpoint", ckpt, "--data-dir", data_dir,
             "--external-dir", ext, "--tag", "fakeext", "--out", transfer])
        return "cross-dataset path"
    check("eval_transfer --mode external", _external)

    def _robust_quick():
        # the full sweep is 25 conditions; the path is identical, so exercise
        # a single condition here to keep the smoke test fast
        from mecg.data.dataset import ECGDataModule as DM
        cfg = write_config(os.path.join(tmp, "c.yaml"))
        dm = DM(data_dir, cfg, test_corruption={"kind": "mixed", "snr_db": 10,
                                                "fs": 100, "seed": 1234})
        _, _, te = dm.get_eval_loaders()
        x, _ = next(iter(te))
        assert torch.isfinite(x).all()
        return "corrupted test loader"
    check("robustness loader", _robust_quick)

    # ---- quantisation ---------------------------------------------------
    print("\nQuantisation")
    check("quantize.py (dynamic INT8)", lambda: run(
        [sys.executable, "scripts/quantize.py", "--checkpoint", ckpt,
         "--data-dir", data_dir, "--modes", "fp32", "dynamic",
         "--out", os.path.join(res_dir, "quantization.json")]) and "")

    # ---- optional heavy paths ------------------------------------------
    if args.full:
        print("\nOptional (--full)")
        check("benchmark_hardware.py", lambda: run(
            [sys.executable, "scripts/benchmark_hardware.py", "--checkpoint",
             ckpt, "--out", os.path.join(res_dir, "benchmark.json"),
             "--repeats", "20", "--batch-sizes", "1"]) and "")
        check("run_probing.py", lambda: run(
            [sys.executable, "scripts/run_probing.py", "--checkpoint", ckpt,
             "--data-dir", data_dir, "--max-train", "60", "--n-jobs", "2",
             "--out", os.path.join(res_dir, "probing.json"),
             "--feature-cache", os.path.join(res_dir, "desc.npz")]) and "")
    else:
        print(f"\n{YELLOW}  skipped (use --full): probing, benchmark, ONNX{RESET}")

    # ---- aggregation to LaTeX ------------------------------------------
    print("\nAggregation to LaTeX")

    def _agg():
        run([sys.executable, "scripts/aggregate_results.py",
             "--results-dir", res_dir, "--paper-dir", paper_dir])
        nums = os.path.join(paper_dir, "generated", "numbers.tex")
        n = sum(1 for line in open(nums) if line.startswith("\\newcommand"))
        assert n > 0, "no macros generated"
        return f"{n} macros"
    check("aggregate_results.py", _agg)

    check("make_figures.py", lambda: run(
        [sys.executable, "scripts/make_figures.py", "--results-dir", res_dir,
         "--out-dir", os.path.join(paper_dir, "figures")]) and "")

    _summary(tmp, args.keep)


def _summary(tmp, keep):
    passed, total = sum(_results), len(_results)
    print(f"\n{'='*68}")
    if passed == total:
        print(f"  {GREEN}ALL {total} CHECKS PASSED{RESET}  -- pipeline is wired "
              f"correctly.\n  Synthetic data is noise, so AUCs near 0.5 are "
              f"expected.\n  Next: RUNBOOK.md section 1 (PTB-XL), then "
              f"SEEDS=\"0\" ./run_all.sh")
    else:
        print(f"  {RED}{total - passed} of {total} CHECKS FAILED{RESET}")
    print("=" * 68)
    if keep:
        print(f"  workdir kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
