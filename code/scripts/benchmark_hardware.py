"""
Measured (not analytical) computational profiling.

Reviewer 1: "claims about latency, memory, and energy advantages require real-
device benchmarking or a more careful limitation of the claims."
Reviewer 2: "latency and memory results are mostly analytical."

The original submission scaled a single MPS measurement by a ratio of nominal
device MFLOPS to produce Cortex-M4 latency figures. That is not a measurement
and the resulting table has been removed. This script instead measures:

  * GPU latency        - CUDA events, warmup, N repeats, median + IQR
  * GPU energy         - NVML integration over the measured window
  * CPU latency        - single- and multi-threaded torch
  * INT8 CPU latency   - dynamic quantisation
  * ONNX Runtime       - portable runtime, closest available proxy for an
                         embedded deployment stack
  * peak memory        - torch.cuda.max_memory_allocated / resident set size
  * MACs and params    - counted with thop/fvcore, not estimated

It also measures the quantity that actually matters for the paper's argument:
latency as a function of nesting dimension d. The expected (and honest) result
is that latency is *flat* in d because the backbone runs in full regardless.
MRL saves embedding storage/transmission and model-artifact count, not compute.

Usage:
    python scripts/benchmark_hardware.py \
        --checkpoint results/checkpoints/inception1d_mrl_s0_best.pt \
        --out results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.models.model import ECGModel


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_callable(fn, device, warmup=30, repeats=200):
    """Median and IQR latency in ms."""
    for _ in range(warmup):
        fn()
    sync(device)
    times = []
    if device.type == "cuda":
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            sync(device)
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    q1 = times[len(times) // 4]
    q3 = times[(3 * len(times)) // 4]
    return {"median_ms": statistics.median(times), "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)), "iqr_ms": q3 - q1,
            "p95_ms": times[int(0.95 * len(times)) - 1], "n": repeats}


def measure_energy_nvml(fn, seconds=5.0):
    """Integrate GPU power over a fixed window. Returns mJ per inference."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        idle = np.mean([pynvml.nvmlDeviceGetPowerUsage(h) for _ in range(20)])
        samples, count, t0 = [], 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            fn()
            count += 1
            if count % 10 == 0:
                samples.append(pynvml.nvmlDeviceGetPowerUsage(h))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        avg_mw = float(np.mean(samples)) if samples else float("nan")
        total_mj = avg_mw * elapsed                      # mW * s = mJ
        idle_mj = float(idle) * elapsed
        pynvml.nvmlShutdown()
        return {"avg_power_w": avg_mw / 1000.0,
                "idle_power_w": float(idle) / 1000.0,
                "inferences": count,
                "energy_per_inference_mj": (total_mj - idle_mj) / max(count, 1),
                "total_energy_per_inference_mj": total_mj / max(count, 1)}
    except Exception as exc:
        return {"error": str(exc)}


def count_macs(model, x):
    try:
        from thop import profile
        macs, params = profile(model, inputs=(x,), verbose=False)
        return {"macs": float(macs), "params": float(params), "tool": "thop"}
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        f = FlopCountAnalysis(model, x)
        f.unsupported_ops_warnings(False)
        return {"macs": float(f.total()), "tool": "fvcore"}
    except Exception as exc:
        return {"error": str(exc)}


def bench_onnx(model, dims, batch=1, leads=12, length=1000, opset=17):
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed"}
    path = "/tmp/_mecg_backbone.onnx"
    model_cpu = model.cpu().eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m.embed(x)

    dummy = torch.randn(batch, leads, length)
    torch.onnx.export(Wrap(model_cpu), dummy, path, opset_version=opset,
                      input_names=["ecg"], output_names=["embedding"],
                      dynamic_axes={"ecg": {0: "batch"}})
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = {"ecg": dummy.numpy()}
    for _ in range(20):
        sess.run(None, inp)
    times = []
    for _ in range(100):
        t0 = time.perf_counter()
        sess.run(None, inp)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {"median_ms": float(np.median(times)), "std_ms": float(np.std(times)),
            "threads": 1, "size_mb": os.path.getsize(path) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 64])
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, dims = ck["config"], ck["dims"]
    leads = cfg["model"].get("input_channels", 12)
    length = 1000

    report = {
        "checkpoint": args.checkpoint,
        "backbone": cfg["model"]["backbone"],
        "head": cfg["model"].get("head", "mrl"),
        "dims": dims,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cuda": torch.version.cuda,
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
        },
        "devices": {},
    }

    model = ECGModel(cfg, verbose=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    report["params"] = {"total": n_params, "backbone": n_backbone, "head": n_head,
                        "fp32_mb": n_params * 4 / 1e6, "int8_mb": n_params / 1e6}
    report["macs"] = count_macs(ECGModel(cfg, verbose=False).eval(),
                                torch.randn(1, leads, length))

    for dev_name in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
        device = torch.device(dev_name)
        m = model.to(device).eval()
        entry = {"batch": {}, "per_dim": {}}

        for bs in args.batch_sizes:
            x = torch.randn(bs, leads, length, device=device)
            with torch.no_grad():
                entry["batch"][str(bs)] = {
                    "backbone": time_callable(lambda: m.embed(x), device,
                                              repeats=args.repeats),
                    "per_sample_ms": None,
                }
            entry["batch"][str(bs)]["per_sample_ms"] = (
                entry["batch"][str(bs)]["backbone"]["median_ms"] / bs)

        # THE key measurement: latency vs nesting dimension at batch 1
        x1 = torch.randn(1, leads, length, device=device)
        with torch.no_grad():
            z = m.embed(x1)
            for d in dims:
                entry["per_dim"][str(d)] = {
                    "head_only": time_callable(
                        lambda d=d: m.head.logits_at(z, d), device,
                        repeats=args.repeats),
                    "end_to_end": time_callable(
                        lambda d=d: m.head.logits_at(m.embed(x1), d), device,
                        repeats=args.repeats),
                    "embedding_bytes": d * 4,
                }

        if dev_name == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                m.embed(x1)
            entry["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 1e6
            with torch.no_grad():
                entry["energy"] = measure_energy_nvml(lambda: m.embed(x1))

        if dev_name == "cpu":
            torch.set_num_threads(1)
            with torch.no_grad():
                entry["single_thread"] = time_callable(
                    lambda: m.embed(x1), device, repeats=max(30, args.repeats // 4))
            try:
                qm = torch.ao.quantization.quantize_dynamic(
                    ECGModel(cfg, verbose=False).eval(),
                    {torch.nn.Linear}, dtype=torch.qint8)
                with torch.no_grad():
                    entry["int8_dynamic"] = time_callable(
                        lambda: qm.embed(x1.cpu()), torch.device("cpu"),
                        repeats=50)
            except Exception as exc:
                entry["int8_dynamic"] = {"error": str(exc)}
            torch.set_num_threads(os.cpu_count() or 1)

        report["devices"][dev_name] = entry

    if not args.skip_onnx:
        report["onnxruntime_cpu"] = bench_onnx(
            ECGModel(cfg, verbose=False).eval(), dims, leads=leads, length=length)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"params": report["params"], "macs": report["macs"]}, indent=2))
    for dev, e in report["devices"].items():
        lat = [e["per_dim"][str(d)]["end_to_end"]["median_ms"] for d in dims]
        print(f"  {dev}: end-to-end latency across d={dims}: "
              f"{[round(v, 3) for v in lat]} ms "
              f"(spread {max(lat) - min(lat):.3f} ms)")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
