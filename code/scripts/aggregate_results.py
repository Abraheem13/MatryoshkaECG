r"""
Aggregate every run into (a) statistics and (b) LaTeX that the paper includes.

This is the bridge between experiments and manuscript. Nothing in the paper is
typed by hand: `paper/generated/numbers.tex` is written here, and the paper
`\input`s it. Re-running the pipeline updates the paper.

Outputs
-------
results/aggregate.json          machine-readable summary
paper/generated/numbers.tex     \newcommand macros for inline numbers
paper/generated/tab_main.tex    main results table (mean +/- SD over seeds)
paper/generated/tab_arch.tex    architecture comparison with paired tests
paper/generated/tab_stats.tex   equivalence tests across nesting dimensions
paper/generated/tab_compute.tex measured computational profile
paper/generated/tab_external.tex cross-dataset transfer
paper/generated/tab_probe.tex   probing / hierarchy verdict

Usage:
    python scripts/aggregate_results.py --results-dir results \
        --paper-dir ../paper
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.analysis.stats import (equivalence_test, holm_bonferroni,
                                 paired_bootstrap_diff, bootstrap_macro_auc,
                                 seed_summary)

DIMS = [16, 32, 64, 128, 256, 512]


# ---------------------------------------------------------------------------
def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "runs", "*", "results.json"))):
        with open(path) as f:
            r = json.load(f)
        r["_dir"] = os.path.dirname(path)
        runs.append(r)
    return runs


def group_key(r):
    # Condition = run name minus the seed suffix. Grouping on
    # (backbone, head, leads, dims) alone collapses every ablation into the
    # main model, because they share all four.
    import re as _re
    cond = _re.sub(r"_s\d+$", "", r.get("run_name", "unknown"))
    return (r["backbone"], r["head"], r["lead_subset"], cond,
            tuple(r["dims"]) if len(r["dims"]) > 1 else ("fixed", r["dims"][0]))


def fmt(mean, std=None, nd=4):
    if not np.isfinite(mean):
        return "--"
    if std is None or not np.isfinite(std) or std == 0:
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f}\\,$\\pm$\\,{std:.{nd}f}"


def esc(s):
    return str(s).replace("_", "\\_")


# ---------------------------------------------------------------------------
def build_summary(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)

    summary = {}
    for key, rs in groups.items():
        backbone, head, leads, cond, dimtag = key
        name = f"{cond}|{dimtag}"
        per_dim = {}
        dims = rs[0]["dims"]
        for d in dims:
            aucs = [r["test"][str(d)]["macro_auc"] for r in rs]
            f1s = [r["test"][str(d)]["macro_f1"] for r in rs]
            aps = [r["test"][str(d)]["macro_ap"] for r in rs]
            per_dim[str(d)] = {"auc": seed_summary(aucs),
                               "f1": seed_summary(f1s),
                               "ap": seed_summary(aps),
                               "seeds": [r["seed"] for r in rs]}
        summary[name] = {
            "backbone": backbone, "head": head, "lead_subset": leads,
            "condition": cond,
            "dims": dims, "n_seeds": len(rs),
            "n_params": rs[0].get("n_params"),
            "per_dim": per_dim,
            "run_dirs": [r["_dir"] for r in rs],
        }
    return summary


def load_preds(run_dir, d):
    f = np.load(os.path.join(run_dir, "test_predictions.npz"))
    return f["y_true"], f[f"probs_d{d}"]


def within_model_equivalence(summary, key, margin=0.01):
    """Is AUC at d=16 equivalent to AUC at d=512 within +/- margin?"""
    entry = summary.get(key)
    if not entry or len(entry["dims"]) < 2:
        return None
    rd = entry["run_dirs"][0]
    dims = entry["dims"]
    try:
        y, p_small = load_preds(rd, dims[0])
        _, p_large = load_preds(rd, dims[-1])
    except (FileNotFoundError, KeyError):
        return None
    res = equivalence_test(y, p_small, p_large, margin=margin, seed=0)
    res.update({"dim_small": dims[0], "dim_large": dims[-1]})
    return res


def across_model_tests(summary, key_a, key_b, margin=0.01):
    """Paired bootstrap at every nesting dimension, Holm-corrected."""
    a, b = summary.get(key_a), summary.get(key_b)
    if not a or not b:
        return None
    out, pvals, dims_used = {}, [], []
    for d in a["dims"]:
        if str(d) not in b["per_dim"]:
            continue
        try:
            y, pa = load_preds(a["run_dirs"][0], d)
            _, pb = load_preds(b["run_dirs"][0], d)
        except (FileNotFoundError, KeyError):
            continue
        r = paired_bootstrap_diff(y, pa, pb, seed=0)
        out[str(d)] = r
        pvals.append(r["p"])
        dims_used.append(d)
    if pvals:
        adj, rej = holm_bonferroni(pvals)
        for i, d in enumerate(dims_used):
            out[str(d)]["p_holm"] = adj[i]
            out[str(d)]["significant"] = bool(rej[i])
    return out


# ---------------------------------------------------------------------------
def write_numbers(path, macros):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% AUTO-GENERATED by scripts/aggregate_results.py -- do not edit\n")
        for k, v in sorted(macros.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"  wrote {path} ({len(macros)} macros)")


def write_table(path, body):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("% AUTO-GENERATED by scripts/aggregate_results.py\n")
        f.write(body)
    print(f"  wrote {path}")


def tab_main(summary):
    """Main table: MRL vs same-backbone fixed baselines, mean +/- SD."""
    rows = []
    inc_mrl = next((k for k, v in summary.items()
                    if v.get("condition") == "inception1d_mrl"), None)
    xres_mrl = next((k for k, v in summary.items()
                     if v.get("condition") == "xresnet1d101_mrl"), None)

    def fixed_for(backbone, d):
        for k, v in summary.items():
            if (v["backbone"] == backbone and v["head"] == "linear"
                    and v["lead_subset"] == "12lead" and v["dims"] == [d]):
                return v["per_dim"][str(d)]["auc"]
        return None

    lines = [
        "\\begin{table*}[!t]", "\\centering",
        "\\caption{Test-set macro AUC-ROC on PTB-XL fold 10 ($n$=2{,}158), "
        "mean\\,$\\pm$\\,SD over five training seeds. Fixed-dimension baselines "
        "use the \\emph{same} backbone as the corresponding MRL model, which is "
        "the like-for-like comparison. Storage is bytes per stored embedding "
        "(float32).}",
        "\\label{tab:main}",
        "\\begin{tabular}{rccccc}", "\\toprule",
        "$d$ & Inc1D-MRL & Inc1D fixed & XRes101-MRL & XRes101 fixed & Storage \\\\",
        "\\midrule",
    ]
    all_dims = (summary[inc_mrl]["dims"] if inc_mrl
                else (summary[xres_mrl]["dims"] if xres_mrl else DIMS))
    for d in all_dims:
        c = []
        for key, bb in ((inc_mrl, "inception1d"), (xres_mrl, "xresnet1d101")):
            mrl = (summary[key]["per_dim"].get(str(d), {}).get("auc")
                   if key else None)
            c.append(fmt(mrl["mean"], mrl["std"]) if mrl else "--")
            fx = fixed_for(bb, d)
            c.append(fmt(fx["mean"], fx["std"]) if fx else "--")
        lines.append(f"{d} & {c[0]} & {c[1]} & {c[2]} & {c[3]} & "
                     f"{d*4}\\,B \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def tab_stats(equiv, cross):
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Statistical assessment. Equivalence is assessed by a paired "
        "bootstrap ($B$=2{,}000) on the difference in macro AUC with a "
        "pre-registered margin of $\\pm$0.01 AUC; the claim of dimension "
        "invariance requires the whole CI to lie inside the margin.}",
        "\\label{tab:stats}",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Comparison & $\\Delta$AUC & 95\\% CI & $p$ & Verdict \\\\",
        "\\midrule",
    ]
    if equiv:
        v = "equivalent" if equiv["equivalent"] else "not equivalent"
        lines.append(
            f"$d$={equiv['dim_small']} vs $d$={equiv['dim_large']} & "
            f"{equiv['diff']:+.4f} & "
            f"[{equiv['ci_low']:+.4f}, {equiv['ci_high']:+.4f}] & "
            f"{equiv['p']:.3f} & {v} \\\\")
    if cross:
        for d, r in sorted(cross.items(), key=lambda kv: int(kv[0])):
            sig = "sig." if r.get("significant") else "n.s."
            lines.append(
                f"Inc1D vs XRes101, $d$={d} & {r['diff']:+.4f} & "
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] & "
                f"{r.get('p_holm', r['p']):.3f} & {sig} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tab_compute(bench):
    if not bench:
        return "% benchmark.json not found\n"
    dims = bench["dims"]
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Measured computational profile (median of 200 timed runs, "
        "batch size 1). Latency is flat in $d$: the backbone executes in full "
        "regardless of the truncation point, so MRL reduces embedding storage "
        "and model count, not inference compute.}",
        "\\label{tab:compute}",
        "\\begin{tabular}{rcccc}", "\\toprule",
        "$d$ & Emb. & Head params & GPU (ms) & CPU 1-thread (ms) \\\\",
        "\\midrule",
    ]
    gpu = bench["devices"].get("cuda", {}).get("per_dim", {})
    cpu = bench["devices"].get("cpu", {}).get("per_dim", {})
    for d in dims:
        g = gpu.get(str(d), {}).get("end_to_end", {}).get("median_ms", float("nan"))
        c = cpu.get(str(d), {}).get("end_to_end", {}).get("median_ms", float("nan"))
        lines.append(f"{d} & {d*4}\\,B & {5*d+5:,} & "
                     f"{g:.2f} & {c:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tab_external(transfer):
    if not transfer:
        return "% transfer.json not found\n"
    ext = {k: v for k, v in transfer.items() if k.startswith("external_")}
    if not ext:
        return "% no external results\n"
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Zero-shot cross-dataset transfer (macro AUC, 95\\% bootstrap "
        "CI). Labels are mapped from SNOMED-CT onto the PTB-XL superclasses, a "
        "lossy many-to-one mapping; these numbers measure transfer of the "
        "superclass concept rather than identical-task performance.}",
        "\\label{tab:external}",
        "\\begin{tabular}{lrcc}", "\\toprule",
        "Corpus & $n$ & smallest $d$ & largest $d$ \\\\", "\\midrule",
    ]
    for k, v in sorted(ext.items()):
        name = esc(k.replace("external_", ""))
        bd = v["by_dim"]
        keys = sorted(bd, key=int)
        a = bd.get(keys[0], {}) if keys else {}
        b = bd.get(keys[-1], {}) if keys else {}
        lines.append(
            f"{name} & {v['n_records']:,} & "
            f"{a.get('macro_auc', float('nan')):.4f} & "
            f"{b.get('macro_auc', float('nan')):.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tab_probe(probe):
    if not probe:
        return "% probing.json not found\n"
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Probing the nesting hierarchy. Saturation dimension is the "
        "smallest prefix reaching 95\\% of the best out-of-sample $R^2$ for "
        "that descriptor. `Claimed' is the dimension at which the original "
        "submission asserted the feature is resolved.}",
        "\\label{tab:probe}",
        "\\begin{tabular}{lccc}", "\\toprule",
        "Descriptor & Best $R^2$ & Saturation $d$ & Claimed $d$ \\\\",
        "\\midrule",
    ]
    for name, r in probe["prefix_probe"].items():
        if not np.isfinite(r.get("best_r2", float("nan"))):
            continue
        lines.append(f"{esc(name)} & {r['best_r2']:.3f} & "
                     f"{r['saturation_dim']} & {r['claimed_dim']} \\\\")
    v = probe.get("hierarchy_verdict", {})
    lines += ["\\midrule",
              f"\\multicolumn{{4}}{{l}}{{Spearman $\\rho$(claimed, measured) = "
              f"{v.get('rho', float('nan')):.3f}, $p$ = "
              f"{v.get('p', float('nan')):.3f}}} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)



def tab_perclass(summary, inc_key):
    """Per-class AUC across nesting dimensions for the proposed model."""
    if not inc_key or inc_key not in summary:
        return "% no MRL run found\n"
    entry = summary[inc_key]
    runs = []
    for rd in entry["run_dirs"]:
        f = os.path.join(rd, "results.json")
        if os.path.exists(f):
            with open(f) as fh:
                runs.append(json.load(fh))
    if not runs:
        return "% no run json\n"
    names = runs[0].get("class_names") or []
    dims = entry["dims"]
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Per-class test AUC-ROC across nesting dimensions "
        "(Inception1D-MRL, mean over seeds). Stability of the class ordering "
        "across granularities is what matters for consistent diagnostic "
        "behaviour at every operating point.}",
        "\\label{tab:perclass}", "\\small",
        "\\begin{tabular}{r" + "c" * len(names) + "}", "\\toprule",
        "$d$ & " + " & ".join(esc(n) for n in names) + " \\\\", "\\midrule",
    ]
    for d in dims:
        vals = []
        for ci in range(len(names)):
            xs = [r["test"][str(d)]["per_class_auc"].get(str(ci))
                  for r in runs if str(d) in r["test"]]
            xs = [x for x in xs if x is not None and np.isfinite(x)]
            vals.append(f"{np.mean(xs):.4f}" if xs else "--")
        lines.append(f"{d} & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tab_ablations(summary):
    """Every non-baseline condition, reported at the smallest and largest dim."""
    rows = []
    for name, v in sorted(summary.items()):
        if v["head"] == "linear":
            continue
        dims = v["dims"]
        if len(dims) < 2:
            continue
        lo = v["per_dim"][str(dims[0])]["auc"]
        hi = v["per_dim"][str(dims[-1])]["auc"]
        label = f"{v['backbone']}, {v['head']}"
        if v["lead_subset"] != "12lead":
            label += f", {v['lead_subset']}"
        rows.append((label, v["n_seeds"], lo, hi, dims))
    if not rows:
        return "% no ablation runs\n"
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Ablations. Each row is a full training condition; AUC is "
        "reported at the smallest and largest nesting dimension with SD over "
        "seeds. Flatness across $d$ is the property under test, so the two "
        "columns should track each other in every condition.}",
        "\\label{tab:ablations}", "\\small",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Condition & Seeds & AUC ($d_{\\min}$) & AUC ($d_{\\max}$) & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for label, ns, lo, hi, dims in rows:
        d = (lo["mean"] - hi["mean"]) if (np.isfinite(lo["mean"])
                                          and np.isfinite(hi["mean"])) else float("nan")
        lines.append(f"{esc(label)} & {ns} & {fmt(lo['mean'], lo['std'])} & "
                     f"{fmt(hi['mean'], hi['std'])} & "
                     f"{d:+.4f} \\\\" if np.isfinite(d) else
                     f"{esc(label)} & {ns} & {fmt(lo['mean'], lo['std'])} & "
                     f"{fmt(hi['mean'], hi['std'])} & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tab_quant(quant):
    """Quantisation: does INT8 hurt small prefixes more than large ones?"""
    if not quant or "modes" not in quant:
        return "% quantization.json not found\n"
    dims = quant["dims"]
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Quantisation, the compute axis nesting does not address. "
        "The final column is the differential penalty: AUC change at "
        f"$d$={dims[0]} minus AUC change at $d$={dims[-1]}. A large negative "
        "value would mean quantisation harms the small operating points "
        "disproportionately, which matters because those are the tiers where "
        "quantisation is most likely to be applied.}",
        "\\label{tab:quant}", "\\small",
        "\\begin{tabular}{lcccc}", "\\toprule",
        "Mode & Size (MB) & Lat. (ms) & "
        f"AUC $d$={dims[0]} & Diff. penalty \\\\", "\\midrule",
    ]
    for mode, e in quant["modes"].items():
        if "error" in e:
            lines.append(f"{esc(mode)} & \\multicolumn{{4}}{{c}}{{not run}} \\\\")
            continue
        a = e["per_dim"].get(str(dims[0]), {}).get("macro_auc", float("nan"))
        dp = e.get("small_prefix_penalty")
        lines.append(f"{esc(mode)} & {e['size_mb']:.1f} & {e['latency_ms']:.1f} "
                     f"& {a:.4f} & "
                     + (f"{dp:+.4f}" if dp is not None else "--") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def tab_robustness(transfer):
    """Artefact robustness, small vs large prefix."""
    if not transfer or "robustness" not in transfer:
        return "% robustness not found\n"
    rob = transfer["robustness"]
    dims = sorted(int(k) for k in next(iter(rob.values())).keys())
    kinds = sorted({k.rsplit("_snr", 1)[0] for k in rob if k != "clean"})
    lines = [
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Artefact robustness (macro AUC). Small and large prefixes "
        "are compared at each SNR: if the smallest prefix degraded faster, "
        "deploying $d$=16 at the noisiest tier would be the wrong choice.}",
        "\\label{tab:robustness}", "\\small",
        "\\begin{tabular}{lcccc}", "\\toprule",
        f"Artefact & SNR & $d$={dims[0]} & $d$={dims[-1]} & $\\Delta$ \\\\",
        "\\midrule",
    ]
    if "clean" in rob:
        c = rob["clean"]
        a, b = c[str(dims[0])], c[str(dims[-1])]
        lines.append(f"clean & -- & {a:.4f} & {b:.4f} & {a-b:+.4f} \\\\")
        lines.append("\\midrule")
    for kind in kinds:
        for snr in (20, 10, 0):
            k = f"{kind}_snr{snr}"
            if k not in rob:
                continue
            a = rob[k][str(dims[0])]
            b = rob[k][str(dims[-1])]
            lines.append(f"{esc(kind)} & {snr}\\,dB & {a:.4f} & {b:.4f} & "
                         f"{a-b:+.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--paper-dir", default="../paper")
    ap.add_argument("--margin", type=float, default=0.01)
    args = ap.parse_args()

    runs = load_runs(args.results_dir)
    print(f"  found {len(runs)} runs")
    summary = build_summary(runs)
    for k, v in summary.items():
        print(f"    {k}  seeds={v['n_seeds']}")

    inc_key = next((k for k, v in summary.items()
                    if v.get("condition") == "inception1d_mrl"), None)
    xres_key = next((k for k, v in summary.items()
                     if v.get("condition") == "xresnet1d101_mrl"), None)

    equiv = within_model_equivalence(summary, inc_key, args.margin) if inc_key else None
    cross = (across_model_tests(summary, inc_key, xres_key)
             if inc_key and xres_key else None)

    def _load(name):
        p = os.path.join(args.results_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
        return None

    bench = _load("benchmark.json")
    transfer = _load("transfer.json")
    probe = _load("probing.json")
    quant = _load("quantization.json")

    gen = os.path.join(args.paper_dir, "generated")
    write_table(os.path.join(gen, "tab_main.tex"), tab_main(summary))
    write_table(os.path.join(gen, "tab_stats.tex"), tab_stats(equiv, cross))
    write_table(os.path.join(gen, "tab_compute.tex"), tab_compute(bench))
    write_table(os.path.join(gen, "tab_external.tex"), tab_external(transfer))
    write_table(os.path.join(gen, "tab_probe.tex"), tab_probe(probe))
    write_table(os.path.join(gen, "tab_perclass.tex"), tab_perclass(summary, inc_key))
    write_table(os.path.join(gen, "tab_ablations.tex"), tab_ablations(summary))
    write_table(os.path.join(gen, "tab_quant.tex"), tab_quant(quant))
    write_table(os.path.join(gen, "tab_robustness.tex"), tab_robustness(transfer))

    # ---- inline macros ----------------------------------------------------
    macros = {}

    def put(name, val, nd=4):
        macros[name] = (f"{val:.{nd}f}" if isinstance(val, float)
                        and np.isfinite(val) else str(val))

    # Use the ACTUAL smallest/largest nesting dimension rather than hardcoded
    # "16"/"512": the dims are configurable, and hardcoding them crashes the
    # whole aggregation step the moment anyone changes them.
    def _ends(key):
        dims = summary[key]["dims"]
        return str(dims[0]), str(dims[-1])

    if inc_key:
        pd_ = summary[inc_key]["per_dim"]
        lo, hi = _ends(inc_key)
        put("IncAUCsmall", pd_[lo]["auc"]["mean"])
        put("IncAUClarge", pd_[hi]["auc"]["mean"])
        put("IncSDsmall", pd_[lo]["auc"]["std"])
        put("IncSDlarge", pd_[hi]["auc"]["std"])
        put("IncSpread", abs(pd_[lo]["auc"]["mean"] - pd_[hi]["auc"]["mean"]))
        macros["DimSmall"] = lo
        macros["DimLarge"] = hi
        macros["NumSeeds"] = str(summary[inc_key]["n_seeds"])
        macros["IncParams"] = f"{summary[inc_key]['n_params']:,}"
    if xres_key:
        pd_ = summary[xres_key]["per_dim"]
        lo, hi = _ends(xres_key)
        put("XResAUCsmall", pd_[lo]["auc"]["mean"])
        put("XResAUClarge", pd_[hi]["auc"]["mean"])
        macros["XResParams"] = f"{summary[xres_key]['n_params']:,}"
    if equiv:
        put("EquivDiff", equiv["diff"])
        put("EquivLo", equiv["ci_low"])
        put("EquivHi", equiv["ci_high"])
        macros["EquivVerdict"] = ("statistically equivalent"
                                  if equiv["equivalent"] else "not equivalent")
        put("EquivMargin", args.margin, nd=3)
    if bench:
        bdims = [str(d) for d in bench.get("dims", [])]
        top = bdims[-1] if bdims else "512"
        for dev, key, nd in (("cuda", "MeasuredGPU", 2), ("cpu", "MeasuredCPU", 1)):
            v = (bench["devices"].get(dev, {}).get("per_dim", {})
                 .get(top, {}).get("end_to_end", {}).get("median_ms"))
            if v is not None:
                put(key, float(v), nd=nd)
        gpu = bench["devices"].get("cuda", {}).get("per_dim", {})
        if gpu:
            lat = [gpu[str(d)]["end_to_end"]["median_ms"] for d in bench["dims"]
                   if str(d) in gpu]
            if lat:
                put("LatencySpread", max(lat) - min(lat), nd=3)
        if bench.get("macs", {}).get("macs"):
            macros["MeasuredMACs"] = f"{bench['macs']['macs']/1e6:.0f}"
        macros["ModelMB"] = f"{bench['params']['fp32_mb']:.1f}"
        gpuname = bench["environment"].get("gpu")
        macros["GPUName"] = esc(gpuname) if gpuname else "n/a"
        macros["TorchVersion"] = esc(bench["environment"]["torch"])
        macros["PythonVersion"] = esc(bench["environment"]["python"])
    if probe:
        v = probe.get("hierarchy_verdict", {})
        put("HierRho", v.get("rho", float("nan")), nd=3)
        put("HierP", v.get("p", float("nan")), nd=3)
        macros["HierVerdict"] = ("supported" if v.get("supported")
                                 else "not supported")
        g = probe.get("geometry", {})
        if g.get("effective_rank_full"):
            put("EffRank", g["effective_rank_full"], nd=1)
        cf = g.get("variance_cumfrac_by_dim", {}).get("16")
        if cf is not None:
            put("VarFracSixteen", float(cf), nd=3)

    if quant and "modes" in quant:
        for mode, key in (("dynamic", "QuantDyn"), ("static", "QuantStatic"),
                          ("qat", "QuantQAT")):
            e = quant["modes"].get(mode, {})
            if "size_mb" in e:
                put(key + "Size", e["size_mb"], nd=1)
                put(key + "Lat", e["latency_ms"], nd=1)
                if e.get("small_prefix_penalty") is not None:
                    put(key + "Penalty", e["small_prefix_penalty"])
    if transfer and "robustness" in transfer and "clean" in transfer["robustness"]:
        rob = transfer["robustness"]
        d0 = str(min(int(k) for k in rob["clean"]))
        if "mixed_snr10" in rob:
            put("RobustMixed10", rob["mixed_snr10"][d0])
    if transfer and "lead_ablation" in transfer:
        la = transfer["lead_ablation"]
        d0 = str(min(int(k) for k in next(iter(la.values()))))
        if "lead_I" in la:
            put("LeadIAUC", la["lead_I"][d0])

    write_numbers(os.path.join(gen, "numbers.tex"), macros)

    out = {"summary": summary, "equivalence": equiv, "cross_model": cross,
           "n_runs": len(runs)}
    with open(os.path.join(args.results_dir, "aggregate.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  wrote {args.results_dir}/aggregate.json")


if __name__ == "__main__":
    main()
