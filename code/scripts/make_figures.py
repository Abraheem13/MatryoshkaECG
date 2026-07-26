"""
Figures for the revised manuscript.

Changes over the original figure set:
  * Fig. 1 (Pareto) now shows confidence bands over seeds and uses a y-axis
    that is not zoomed to a 0.02 AUC window -- the original plot magnified
    differences of ~0.005 AUC into apparently large visual gaps.
  * The SVD curve is the leakage-free version and is no longer labelled an
    "upper bound".
  * New: latency-vs-dimension figure showing the flat profile (the honest
    computational story), robustness curves, and probing saturation.

Usage:
    python scripts/make_figures.py --results-dir results --out-dir ../paper/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

C = {"mrl_inc": "#3B4CC0", "mrl_xres": "#B40426", "fixed": "#E08214",
     "svd": "#1B9E77", "mrle": "#7570B3", "grey": "#666666"}
DIMS = [16, 32, 64, 128, 256, 512]


def load_runs(results_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(results_dir, "runs", "*", "results.json"))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs


def collect(runs, backbone, head, leads="12lead"):
    """dim -> list of AUCs over seeds."""
    acc = defaultdict(list)
    for r in runs:
        if (r["backbone"] == backbone and r["head"] == head
                and r.get("lead_subset", "12lead") == leads):
            for d in r["dims"]:
                acc[d].append(r["test"][str(d)]["macro_auc"])
    return acc


def band(ax, dims, acc, color, label, marker="o", ls="-"):
    xs = [d for d in dims if d in acc and acc[d]]
    if not xs:
        return
    mu = np.array([np.mean(acc[d]) for d in xs])
    sd = np.array([np.std(acc[d], ddof=1) if len(acc[d]) > 1 else 0.0 for d in xs])
    ax.plot(xs, mu, marker=marker, ls=ls, color=color, lw=1.8, ms=5, label=label)
    ax.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.18, lw=0)


def fig_pareto(runs, transfer, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    band(ax, DIMS, collect(runs, "inception1d", "mrl"), C["mrl_inc"],
         "Inception1D + MRL")
    band(ax, DIMS, collect(runs, "xresnet1d101", "mrl"), C["mrl_xres"],
         "XResNet1D-101 + MRL")

    fixed = defaultdict(list)
    for r in runs:
        if r["backbone"] == "inception1d" and r["head"] == "linear":
            fixed[r["dims"][0]].append(r["test"][str(r["dims"][0])]["macro_auc"])
    band(ax, DIMS, fixed, C["fixed"], "Inception1D fixed-$d$", marker="s", ls="--")

    if transfer and "svd_no_leakage" in transfer:
        sv = transfer["svd_no_leakage"]
        xs = [d for d in DIMS if str(d) in sv]
        ax.plot(xs, [sv[str(d)]["macro_auc"] for d in xs], "^:", color=C["svd"],
                lw=1.5, ms=5, label="SVD (no leakage)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(DIMS)
    ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("Macro AUC-ROC")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="lower left", frameon=False)
    fig.savefig(os.path.join(out, "fig_pareto.pdf"))
    fig.savefig(os.path.join(out, "fig_pareto.png"))
    plt.close(fig)


def fig_latency(bench, out):
    if not bench:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for dev, color, lbl in (("cuda", C["mrl_inc"], "GPU"),
                            ("cpu", C["fixed"], "CPU (1 thread)")):
        pd_ = bench["devices"].get(dev, {}).get("per_dim", {})
        xs = [d for d in bench["dims"] if str(d) in pd_]
        if not xs:
            continue
        ys = [pd_[str(d)]["end_to_end"]["median_ms"] for d in xs]
        er = [pd_[str(d)]["end_to_end"]["iqr_ms"] / 2 for d in xs]
        ax.errorbar(xs, ys, yerr=er, marker="o", ms=4, lw=1.6, capsize=2,
                    color=color, label=lbl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(bench["dims"])
    ax.set_xticklabels([str(d) for d in bench["dims"]])
    ax.set_xlabel("Embedding dimension $d$")
    ax.set_ylabel("End-to-end latency (ms)")
    ax.set_title("Latency is invariant to $d$", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out, "fig_latency.pdf"))
    plt.close(fig)


def _dim_keys(d):
    """Sorted string keys of a {dim: value} dict. Nesting dims are
    configurable, so never index them with a literal."""
    return sorted(d, key=int)


def fig_robustness(transfer, out):
    if not transfer or "robustness" not in transfer:
        return
    rob = transfer["robustness"]
    kinds = sorted({k.rsplit("_snr", 1)[0] for k in rob if k != "clean"})
    if not kinds:
        return
    dk = _dim_keys(next(iter(rob.values())))
    lo = dk[0]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    cmap = plt.get_cmap("viridis")
    for i, kind in enumerate(kinds):
        snrs, vals = [], []
        for k, v in rob.items():
            if k != "clean" and k.rsplit("_snr", 1)[0] == kind:
                snrs.append(int(k.rsplit("_snr", 1)[1]))
                vals.append(v[lo])
        o = np.argsort(snrs)
        ax.plot(np.array(snrs)[o], np.array(vals)[o], "o-", ms=4, lw=1.5,
                color=cmap(i / max(len(kinds) - 1, 1)),
                label=kind.replace("_", " "))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel(f"Macro AUC at $d$={lo}")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(out, "fig_robustness.pdf"))
    plt.close(fig)


def fig_probing(probe, out):
    if not probe:
        return
    names = [n for n, r in probe["prefix_probe"].items()
             if r.get("best_r2") is not None and np.isfinite(r["best_r2"])
             and r["best_r2"] > 0.05]
    if not names:
        return
    dims = probe["dims"]
    M = np.array([[probe["prefix_probe"][n]["r2_by_dim"].get(str(d), np.nan)
                   for d in dims] for n in names], dtype=float)
    fig, ax = plt.subplots(figsize=(3.5, 0.28 * len(names) + 1.2))
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0,
                   vmax=np.nanmax(M))
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.set_xlabel("Prefix dimension $d$")
    for i, n in enumerate(names):
        s = probe["prefix_probe"][n]["saturation_dim"]
        if s in dims:
            ax.plot(dims.index(s), i, "w*", ms=7)
    fig.colorbar(im, ax=ax, label="out-of-sample $R^2$", fraction=0.046)
    ax.set_title("Descriptor recoverability by prefix", fontsize=9)
    fig.savefig(os.path.join(out, "fig_probing.pdf"))
    plt.close(fig)


def fig_leads(transfer, out):
    if not transfer or "lead_ablation" not in transfer:
        return
    la = transfer["lead_ablation"]
    order = ["12lead", "8lead", "6lead", "3lead", "2lead", "lead_II", "lead_I"]
    order = [o for o in order if o in la]
    if not order:
        return
    dk = _dim_keys(la[order[0]])
    lo, hi = dk[0], dk[-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    w = 0.38
    x = np.arange(len(order))
    ax.bar(x - w / 2, [la[o][lo] for o in order], w, color=C["mrl_inc"],
           label=f"$d$={lo}")
    ax.bar(x + w / 2, [la[o][hi] for o in order], w, color=C["fixed"],
           label=f"$d$={hi}")
    ax.set_xticks(x)
    ax.set_xticklabels([o.replace("_", " ") for o in order], rotation=25,
                       ha="right")
    ax.set_ylabel("Macro AUC-ROC")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color=C["grey"], lw=0.8, ls=":")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(out, "fig_leads.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-dir", default="../paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    runs = load_runs(args.results_dir)

    def _load(n):
        p = os.path.join(args.results_dir, n)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    transfer, bench, probe = _load("transfer.json"), _load("benchmark.json"), \
        _load("probing.json")

    if runs:
        fig_pareto(runs, transfer, args.out_dir)
    fig_latency(bench, args.out_dir)
    fig_robustness(transfer, args.out_dir)
    fig_probing(probe, args.out_dir)
    fig_leads(transfer, args.out_dir)
    print(f"  figures -> {args.out_dir}")


if __name__ == "__main__":
    main()
