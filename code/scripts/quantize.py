"""
Quantisation: the compute axis that nesting does NOT address.

The paper argues that MRL reduces embedding storage and artefact count but not
inference compute, and that quantisation is the complementary lever. That
argument is stronger if we measure it rather than assert it, so this script
quantises the trained MRL encoder and reports accuracy at every nesting
dimension alongside model size and latency.

The interesting question is an interaction: does INT8 quantisation degrade the
SMALL prefixes more than the large ones? If it does, the d=16 operating point
is worse than the fp32 numbers suggest at exactly the tier where quantisation
is most likely to be used -- which would matter for deployment and is not
something either reviewer thought to ask.

Modes
-----
fp32       reference
dynamic    post-training dynamic INT8 (weights only; Linear/Conv)
static     post-training static INT8 with calibration on the train split
qat        quantisation-aware training (fine-tune, few epochs)
fp16/bf16  half precision on GPU

Usage
-----
    python scripts/quantize.py \
        --checkpoint results/checkpoints/inception1d_mrl_s0_best.pt \
        --modes fp32 dynamic static fp16 \
        --out results/quantization.json
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.analysis.metrics import compute_all_metrics, macro_auc, tune_thresholds
from mecg.analysis.stats import bootstrap_macro_auc
from mecg.data.dataset import ECGDataModule
from mecg.models.model import ECGModel


def model_size_mb(model) -> float:
    tmp = "/tmp/_mecg_size_probe.pt"
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)
    return mb


@torch.no_grad()
def evaluate(model, loader, dims, device, amp_dtype=None, chunk=4096):
    model.eval()
    Z, Y = [], []
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None else contextlib.nullcontext())
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with ctx:
            Z.append(model.embed(x).float().cpu())
        Y.append(y)
    Z = torch.cat(Z)
    Y = torch.cat(Y).numpy()
    probs = {}
    for d in dims:
        parts = []
        for i in range(0, Z.shape[0], chunk):
            zz = Z[i:i + chunk].to(device)
            parts.append(torch.sigmoid(model.head.logits_at(zz, d)).float().cpu())
        probs[d] = torch.cat(parts).numpy()
    return Y, probs


@torch.no_grad()
def latency_ms(model, device, leads=12, length=1000, repeats=100, warmup=20):
    x = torch.randn(1, leads, length, device=device)
    for _ in range(warmup):
        model.embed(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.embed(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def prepare_static(model, calib_loader, device, n_batches=16):
    """Post-training static quantisation with calibration (CPU/fbgemm)."""
    import torch.ao.quantization as tq
    m = copy.deepcopy(model).cpu().eval()
    m.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(m, inplace=True)
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            m.embed(x.cpu())
            if i + 1 >= n_batches:
                break
    tq.convert(m, inplace=True)
    return m


def run_qat(model, cfg, dm, device, epochs=2, lr=1e-4):
    """Quantisation-aware fine-tuning, then convert."""
    import torch.ao.quantization as tq
    from mecg.losses import MatryoshkaObjective

    m = copy.deepcopy(model).cpu().train()
    m.qconfig = tq.get_default_qat_qconfig("fbgemm")
    tq.prepare_qat(m, inplace=True)
    m.to(device)

    dims = m.head.nesting_dims
    obj = MatryoshkaObjective(dims, m.num_classes,
                              weight_strategy=cfg["nesting"].get(
                                  "weight_strategy", "equal")).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    train_loader, _, _ = dm.get_dataloaders(seed=0)
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = obj(m(x), y)["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"      QAT epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    m.eval().cpu()
    return tq.convert(m, inplace=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out", default="results/quantization.json")
    ap.add_argument("--modes", nargs="+",
                    default=["fp32", "dynamic", "static", "fp16"],
                    choices=["fp32", "dynamic", "static", "qat", "fp16", "bf16"])
    ap.add_argument("--qat-epochs", type=int, default=2)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device) if args.device else gpu
    cpu = torch.device("cpu")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, dims = ck["config"], ck["dims"]
    class_names = ck.get("class_names")
    leads = cfg["model"].get("input_channels", 12)

    base = ECGModel(cfg, verbose=False)
    base.load_state_dict(ck["model_state_dict"])
    base.eval()

    dm = ECGDataModule(args.data_dir, cfg,
                       lead_subset=cfg.get("data", {}).get("lead_subset"))
    train_loader, val_loader, test_loader = dm.get_eval_loaders()

    report = {"checkpoint": args.checkpoint, "dims": dims, "modes": {}}

    for mode in args.modes:
        print(f"\n  [{mode}]")
        try:
            if mode == "fp32":
                m, dev, amp = copy.deepcopy(base).to(device), device, None
            elif mode == "fp16":
                if gpu.type != "cuda":
                    raise RuntimeError("fp16 path needs CUDA")
                m, dev, amp = copy.deepcopy(base).to(gpu), gpu, torch.float16
            elif mode == "bf16":
                if gpu.type != "cuda":
                    raise RuntimeError("bf16 path needs CUDA")
                m, dev, amp = copy.deepcopy(base).to(gpu), gpu, torch.bfloat16
            elif mode == "dynamic":
                m = torch.ao.quantization.quantize_dynamic(
                    copy.deepcopy(base).cpu().eval(),
                    {nn.Linear, nn.Conv1d}, dtype=torch.qint8)
                dev, amp = cpu, None
            elif mode == "static":
                m, dev, amp = prepare_static(base, train_loader, cpu), cpu, None
            elif mode == "qat":
                m = run_qat(base, cfg, dm, device, epochs=args.qat_epochs)
                dev, amp = cpu, None
            else:
                raise ValueError(mode)

            Yv, pv = evaluate(m, val_loader, dims, dev, amp)
            Yt, pt = evaluate(m, test_loader, dims, dev, amp)

            per_dim = {}
            for d in dims:
                thr = tune_thresholds(Yv, pv[d])
                met = compute_all_metrics(Yt, pt[d], class_names, thresholds=thr)
                boot = bootstrap_macro_auc(Yt, pt[d], n_boot=1000, seed=0)
                per_dim[str(d)] = {"macro_auc": met["macro_auc"],
                                   "macro_f1": met["macro_f1"],
                                   "ci_low": boot["ci_low"],
                                   "ci_high": boot["ci_high"]}
                print(f"      d={d:4d}  AUC={met['macro_auc']:.4f} "
                      f"[{boot['ci_low']:.4f},{boot['ci_high']:.4f}]")

            report["modes"][mode] = {
                "per_dim": per_dim,
                "size_mb": model_size_mb(m),
                "latency_ms": latency_ms(m, dev, leads=leads),
                "device": dev.type,
                "amp": str(amp) if amp else None,
            }
            print(f"      size={report['modes'][mode]['size_mb']:.2f} MB  "
                  f"latency={report['modes'][mode]['latency_ms']:.2f} ms "
                  f"({dev.type})")
        except Exception as exc:
            print(f"      SKIPPED: {type(exc).__name__}: {exc}")
            report["modes"][mode] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- the interaction the paper cares about ---------------------------
    ref = report["modes"].get("fp32", {}).get("per_dim")
    if ref:
        for mode, entry in report["modes"].items():
            if mode == "fp32" or "per_dim" not in entry:
                continue
            delta = {d: entry["per_dim"][d]["macro_auc"] - ref[d]["macro_auc"]
                     for d in ref if d in entry["per_dim"]}
            entry["delta_vs_fp32"] = delta
            small, large = delta.get(str(dims[0])), delta.get(str(dims[-1]))
            if small is not None and large is not None:
                entry["small_prefix_penalty"] = float(small - large)
                print(f"\n  {mode}: AUC change at d={dims[0]} is {small:+.4f}, "
                      f"at d={dims[-1]} is {large:+.4f} "
                      f"(differential {small - large:+.4f})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
