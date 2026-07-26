"""
Unified training entry point.

Covers every condition the revision needs:

    # proposed MRL model
    python scripts/train.py --config configs/mrl_inception1d.yaml --seed 0

    # SAME-BACKBONE fixed-dimension baseline (the comparison Reviewer 2 asked for)
    python scripts/train.py --config configs/mrl_inception1d.yaml \
        --head linear --embedding-dim 64 --run-name fixed_inc1d_d64 --seed 0

    # MRL-E shared-weight variant
    python scripts/train.py --config configs/mrl_inception1d.yaml --head mrl-e

    # reduced-lead (wearable proxy)
    python scripts/train.py --config configs/mrl_inception1d.yaml --lead-subset lead_I

Bug fixes over the original trainer
-----------------------------------
* Weight-decay grouping used a substring test (`'bn' in name`), which missed
  every BatchNorm nested inside `ConvBlock1d` (named `block.1.*`) and therefore
  applied weight decay to normalisation parameters. Now grouped by
  `param.ndim <= 1`, the standard criterion.
* `torch.cuda.amp.autocast` is deprecated; uses `torch.amp` with bf16 where
  supported (no GradScaler needed, more stable than fp16).
* Validation embeddings are scored in chunks instead of moving the entire
  matrix to the GPU at once.
* Full determinism: seeds torch/numpy/python, dataloader workers, and cuDNN.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import contextlib
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mecg.analysis.metrics import compute_all_metrics, macro_auc, tune_thresholds
from mecg.data.dataset import ECGDataModule, LEAD_SUBSETS
from mecg.losses import MatryoshkaObjective
from mecg.models.model import ECGModel


# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(arg=None):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model, cfg):
    """No weight decay on biases and 1-D (normalisation) parameters."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": float(cfg.get("weight_decay", 1e-2))},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=float(cfg.get("learning_rate", 1e-3)))


def build_scheduler(optimizer, cfg, steps_per_epoch):
    epochs = int(cfg.get("epochs", 50))
    warmup = int(cfg.get("warmup_epochs", 5)) * steps_per_epoch
    total = epochs * steps_per_epoch

    def fn(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def amp_ctx(device, amp_dtype):
    """autocast context that is a no-op when AMP is disabled.

    torch.autocast rejects dtype=None even with enabled=False, so the
    disabled case needs its own branch rather than a single call."""
    if amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


@torch.no_grad()
def collect_embeddings(model, loader, device, amp_dtype):
    model.eval()
    Z, Y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with amp_ctx(device, amp_dtype):
            z = model.embed(x)
        Z.append(z.float().cpu())
        Y.append(y)
    return torch.cat(Z), torch.cat(Y).numpy()


@torch.no_grad()
def score_dims(model, Z, device, dims, chunk=8192):
    out = {}
    for d in dims:
        parts = []
        for i in range(0, Z.shape[0], chunk):
            zz = Z[i:i + chunk].to(device)
            parts.append(torch.sigmoid(model.head.logits_at(zz, d)).float().cpu())
        out[d] = torch.cat(parts).numpy()
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head", default=None, choices=["mrl", "mrl-e", "linear"])
    ap.add_argument("--backbone", default=None)
    ap.add_argument("--backbone-kwargs", default=None,
                    help='JSON dict merged into model.backbone_kwargs, e.g. '
                         '\'{"width_mode":"constant","residual_every":3}\'')
    ap.add_argument("--embedding-dim", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--norm-mode", default=None, choices=["dataset", "per_record"])
    ap.add_argument("--lead-subset", default=None, choices=list(LEAD_SUBSETS))
    ap.add_argument("--weight-strategy", default=None)
    ap.add_argument("--data-dir", default="data/processed")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--save-train-embeddings", action="store_true",
                    help="also store train-split embeddings (needed for probing "
                         "and the SVD baseline; ~35 MB per run)")
    ap.add_argument("--tag", default=None, help="free-form label recorded in results.json")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- CLI overrides -----------------------------------------------------
    if args.head:            cfg["model"]["head"] = args.head
    if args.backbone:        cfg["model"]["backbone"] = args.backbone
    if args.backbone_kwargs:
        cfg["model"].setdefault("backbone_kwargs", {}).update(
            json.loads(args.backbone_kwargs))
    if args.embedding_dim:   cfg["model"]["embedding_dim"] = args.embedding_dim
    if args.epochs:          cfg["training"]["epochs"] = args.epochs
    if args.batch_size:      cfg["training"]["batch_size"] = args.batch_size
    if args.lr:              cfg["training"]["learning_rate"] = args.lr
    if args.label_smoothing is not None:
        cfg["training"]["label_smoothing"] = args.label_smoothing
    if args.norm_mode:       cfg.setdefault("data", {})["norm_mode"] = args.norm_mode
    if args.weight_strategy: cfg["nesting"]["weight_strategy"] = args.weight_strategy

    lead_subset = args.lead_subset or cfg.get("data", {}).get("lead_subset")
    if lead_subset:
        cfg["model"]["input_channels"] = len(LEAD_SUBSETS[lead_subset])

    run_name = args.run_name or (
        f"{cfg['model']['backbone']}_{cfg['model'].get('head','mrl')}"
        f"{'_' + lead_subset if lead_subset else ''}_s{args.seed}"
    )
    ckpt_dir = os.path.join(args.out_dir, "checkpoints")
    run_dir = os.path.join(args.out_dir, "runs", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed, deterministic=cfg.get("deterministic", True))
    device = pick_device(args.device)
    amp_dtype = None
    if args.amp != "off" and device.type == "cuda":
        amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

    print("=" * 72)
    print(f"  RUN {run_name}")
    print(f"  device={device}  amp={args.amp}  seed={args.seed}")
    print(f"  leads={lead_subset or '12lead'}  "
          f"norm={cfg.get('data', {}).get('norm_mode', 'dataset')}")
    print("=" * 72)

    dm = ECGDataModule(data_dir=args.data_dir, config=cfg, lead_subset=lead_subset)
    train_loader, val_loader, test_loader = dm.get_dataloaders(seed=args.seed)

    with open(os.path.join(args.data_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    class_names = [meta["idx_to_class"][i] for i in range(meta["num_classes"])]
    cfg["model"]["num_classes"] = meta["num_classes"]

    model = ECGModel(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    dims = model.head.nesting_dims

    pos_weight = None
    if cfg["training"].get("pos_weight", False):
        y_tr = np.load(os.path.join(args.data_dir, "y_train.npy"))
        pw = (len(y_tr) - y_tr.sum(0)) / np.maximum(y_tr.sum(0), 1)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        print(f"  pos_weight={np.round(pw, 2).tolist()}")

    objective = MatryoshkaObjective(
        nesting_dims=dims,
        num_classes=meta["num_classes"],
        weight_strategy=cfg["nesting"].get("weight_strategy", "equal"),
        label_smoothing=float(cfg["training"].get("label_smoothing", 0.0)),
        pos_weight=pos_weight,
    ).to(device)

    optimizer = build_optimizer(model, cfg["training"])
    scheduler = build_scheduler(optimizer, cfg["training"], len(train_loader))
    scaler = torch.amp.GradScaler(device=device.type,
                                  enabled=(amp_dtype == torch.float16))

    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"].get("early_stopping_patience", 10))
    clip = float(cfg["training"].get("gradient_clip_norm", 1.0))
    best_auc, best_epoch, wait = -1.0, -1, 0
    history = []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        run_loss, per_dim, nb = 0.0, defaultdict(float), 0

        for x, y in tqdm(train_loader, desc=f"  ep{epoch+1:03d}", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx(device, amp_dtype):
                logits = model(x)
                out = objective(logits, y)
                loss = out["total_loss"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            scheduler.step()

            run_loss += float(loss.detach())
            for d, v in out["per_dim_loss"].items():
                per_dim[d] += float(v)
            nb += 1

        Zv, Yv = collect_embeddings(model, val_loader, device, amp_dtype)
        probs_v = score_dims(model, Zv, device, dims)
        val_auc = macro_auc(Yv, probs_v[dims[-1]])
        # model selection uses the MEAN AUC over granularities for MRL, so that
        # early stopping is not driven by the largest head alone
        sel_auc = float(np.mean([macro_auc(Yv, probs_v[d]) for d in dims]))

        history.append({"epoch": epoch + 1, "train_loss": run_loss / max(nb, 1),
                        "val_auc_max_dim": val_auc, "val_auc_mean_dims": sel_auc,
                        "lr": optimizer.param_groups[0]["lr"],
                        "secs": time.time() - t0})

        flag = ""
        if sel_auc > best_auc:
            best_auc, best_epoch, wait = sel_auc, epoch, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "config": cfg, "dims": dims, "seed": args.seed,
                        "val_auc": sel_auc, "class_names": class_names},
                       os.path.join(ckpt_dir, f"{run_name}_best.pt"))
            flag = "  *best*"
        else:
            wait += 1

        print(f"  ep{epoch+1:03d}/{epochs} loss={run_loss/max(nb,1):.4f} "
              f"val_auc(max_d)={val_auc:.4f} val_auc(mean_d)={sel_auc:.4f} "
              f"{time.time()-t0:.0f}s{flag}")

        if wait >= patience:
            print(f"  early stop at epoch {epoch+1} (patience {patience})")
            break

    # ---- test evaluation with validation-tuned thresholds ------------------
    ck = torch.load(os.path.join(ckpt_dir, f"{run_name}_best.pt"),
                    map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    print(f"  restored best epoch {ck['epoch']+1}")

    # deterministic, unaugmented, no dropped samples -- required so that
    # saved embeddings align row-for-row with X_<split>.npy for downstream
    # probing and patient-level analysis
    eval_train, eval_val, eval_test = dm.get_eval_loaders()
    Zv, Yv = collect_embeddings(model, eval_val, device, amp_dtype)
    Zt, Yt = collect_embeddings(model, eval_test, device, amp_dtype)
    probs_v = score_dims(model, Zv, device, dims)
    probs_t = score_dims(model, Zt, device, dims)

    results = {}
    for d in dims:
        thr = tune_thresholds(Yv, probs_v[d])
        results[d] = compute_all_metrics(Yt, probs_t[d], class_names, thresholds=thr)
        print(f"  d={d:4d}  AUC={results[d]['macro_auc']:.4f}  "
              f"F1={results[d]['macro_f1']:.4f}  AP={results[d]['macro_ap']:.4f}")

    np.savez_compressed(os.path.join(run_dir, "test_predictions.npz"),
                        y_true=Yt, **{f"probs_d{d}": probs_t[d] for d in dims})
    if args.save_train_embeddings:
        Ztr, Ytr = collect_embeddings(model, eval_train, device, amp_dtype)
        np.savez_compressed(os.path.join(run_dir, "embeddings.npz"),
                            Z_test=Zt.numpy(), y_test=Yt,
                            Z_train=Ztr.numpy(), y_train=Ytr)
    else:
        np.savez_compressed(os.path.join(run_dir, "embeddings.npz"),
                            Z_test=Zt.numpy(), y_test=Yt)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump({"run_name": run_name, "seed": args.seed, "dims": dims,
                   "backbone": cfg["model"]["backbone"],
                   "head": cfg["model"].get("head", "mrl"),
                   "lead_subset": lead_subset or "12lead",
                   "norm_mode": cfg.get("data", {}).get("norm_mode", "dataset"),
                   "tag": args.tag,
                   "num_classes": meta["num_classes"],
                   "best_epoch": int(ck["epoch"]) + 1,
                   "best_val_auc": float(best_auc),
                   "n_params": int(sum(p.numel() for p in model.parameters())),
                   "class_names": class_names,
                   "history": history,
                   "test": {str(d): {k: v for k, v in results[d].items()
                                     if k != "per_class_auc"} |
                            {"per_class_auc": {str(k): v for k, v in
                                               results[d]["per_class_auc"].items()}}
                            for d in dims}}, f, indent=2)
    print(f"  wrote {run_dir}/results.json")


if __name__ == "__main__":
    main()
