# RUNBOOK — Matryoshka-ECG revision

Everything needed to regenerate the revised manuscript from scratch.

**Important:** the manuscript ships with **no results in it**. Every number is
a macro that renders as a red **[TBD]** until you run the pipeline. This is
deliberate — see [Why the paper has no numbers yet](#why-the-paper-has-no-numbers-yet).

---

## 0. Validate before you burn GPU hours

```bash
cd code
pip install -r requirements.txt
python scripts/smoke_test.py --full        # ~2 min, no PTB-XL needed
```

This builds 400 synthetic recordings and exercises every path the real
pipeline uses: all backbones, all head types, both normalisation modes,
training, checkpointing, evaluation, SVD, robustness, quantisation, probing,
benchmarking, and aggregation to LaTeX. Green means the plumbing is correct.
It says nothing about the science — the data is noise.

Everything in this package was verified by running it, not only by reading it.
The smoke test caught three real bugs during development; see
[What changed in the code](#what-changed-in-the-code).

---

## 0b. Environment

```bash
conda env create -f environment.yml && conda activate mecg
# or:
conda create -n mecg python=3.11 -y && conda activate mecg
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r code/requirements.txt
```

Verify:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 1. PTB-XL

```bash
cd code
./scripts/download_data.sh                 # PTB-XL only
WITH_EXTERNAL=1 ./scripts/download_data.sh # + CPSC-2018, Chapman, Georgia

python scripts/preprocess_ptbxl.py --raw data/ptbxl_raw --out data/processed
```

Expect `21,388` recordings and splits `17,084 / 2,146 / 2,158`.

> Signals are stored **raw, in mV**. Normalisation happens at load time so
> `--norm-mode dataset` vs `per_record` can be ablated without re-preprocessing.
> This is one of the substantive fixes: the old pipeline baked per-record
> z-scoring into the `.npy` files, discarding the absolute voltage that
> hypertrophy diagnosis depends on.

---

## 2. Run everything

```bash
cd code
SEEDS="0" ./run_all.sh      # one seed, whole matrix, ~4 h — do this first
./run_all.sh                # full matrix
```

Idempotent: a run with an existing `results.json` is skipped, so an
interrupted job resumes cleanly.

### Knobs

| Variable | Default | Notes |
|---|---|---|
| `SEEDS` | `0 1 2 3 4` | training seeds |
| `STAGES` | `1..10` | e.g. `STAGES="3"` to run only the baselines |
| `BS` | `128` | raise it — you have the VRAM. 256–512 is fine |
| `AMP` | `bf16` | `bf16` \| `fp16` \| `off` |
| `COMPILE` | `0` | `1` enables `torch.compile` |
| `TASK` | `super` | `sub` for the 23-class diagnostic task |

### Time estimates (single A100/4090-class GPU)

| Stage | What | Runs | Est. |
|---|---|---|---|
| 2 | MRL models, 2 backbones × 5 seeds | 10 | 6–9 h |
| 3 | Fixed-dim baselines, 2 × 6 dims × 5 seeds | 60 | 18–24 h |
| 4 | Ablations (MRL-E, norm, smoothing, weights, widths, XResNet50) | ~20 | 6 h |
| 5 | Native reduced-lead models, 6 subsets × 5 seeds | 30 | 8 h |
| 6–9 | SVD, robustness, transfer, probing, benchmark, quantisation | — | 3 h |

**~45–55 GPU-hours total.** Stage 3 dominates. If time-constrained, run it
with `SEEDS="0 1 2"` — three seeds still supports the statistics, and having
*some* matched baselines (Reviewer 2's central request) matters far more than
five seeds of everything.

Shard across GPUs by seed:
```bash
CUDA_VISIBLE_DEVICES=0 SEEDS="0 1" ./run_all.sh &
CUDA_VISIBLE_DEVICES=1 SEEDS="2 3" ./run_all.sh &
CUDA_VISIBLE_DEVICES=2 SEEDS="4"   ./run_all.sh &
```

### Since compute is not your constraint

Three extensions are wired up and worth the GPU time:

1. **The 23-class diagnostic task** (`TASK=sub`). This is the sharpest open
   prediction in the paper: if the flat accuracy–dimension profile is caused
   by low intrinsic dimensionality rather than by MRL regularising, the
   sufficient prefix should *grow* with task granularity. A positive result
   there is a genuine finding, not a routine extension.
2. **Native reduced-lead models** (Stage 5) — not just masking leads on a
   12-lead model, but training on single-lead input directly. This is the
   closest available proxy to the wearable claim Reviewer 1 rejected.
3. **Quantisation** (Stage 9) — measures the compute axis nesting leaves
   untouched, and specifically whether INT8 degrades small prefixes more than
   large ones.

---

## 3. External datasets (optional but addresses R1-1 / R2-5)

Free, no credentialing — from the PhysioNet/CinC 2020 collection:

```bash
cd code/data/external
wget https://physionet.org/files/challenge-2020/1.0.2/training/cpsc_2018.tar.gz
wget https://physionet.org/files/challenge-2020/1.0.2/training/georgia.tar.gz
wget https://physionet.org/files/challenge-2020/1.0.2/training/chapman_shaoxing.tar.gz
for f in *.tar.gz; do tar xzf "$f"; done
cd ../..

for D in cpsc2018 chapman georgia; do
  python scripts/preprocess_external.py --dataset $D \
    --raw data/external/$D --out data/processed_$D
done
```

Then re-run Stage 7 (or just `./run_all.sh` again — earlier stages skip).

**MIMIC-IV-ECG / CODE-15%** need credentialed PhysioNet access. Once obtained,
point `--raw` at the extracted directory; the loader handles them.

---

## 4. Individual commands

```bash
# proposed model
python scripts/train.py --config configs/mrl_inception1d.yaml --seed 0

# matched fixed-dimension baseline  ← the comparison Reviewer 2 asked for
python scripts/train.py --config configs/mrl_inception1d.yaml \
    --head linear --embedding-dim 64 --run-name inception1d_fixed_d64_s0 --seed 0

# MRL-E shared-weight head
python scripts/train.py --config configs/mrl_inception1d.yaml --head mrl-e

# normalisation ablation (tests the HYP/voltage hypothesis)
python scripts/train.py --config configs/mrl_inception1d.yaml --norm-mode per_record

# natively trained single-lead model
python scripts/train.py --config configs/mrl_inception1d.yaml --lead-subset lead_I

# measured hardware profile
python scripts/benchmark_hardware.py \
    --checkpoint results/checkpoints/inception1d_mrl_s0_best.pt

# probing the hierarchy claim
python scripts/run_probing.py \
    --checkpoint results/checkpoints/inception1d_mrl_s0_best.pt

# leakage-free SVD / robustness / leads
python scripts/eval_transfer.py --mode svd        --checkpoint <ckpt>
python scripts/eval_transfer.py --mode robustness --checkpoint <ckpt>
python scripts/eval_transfer.py --mode leads      --checkpoint <ckpt>
```

---

## 5. Build the paper

```bash
python scripts/aggregate_results.py --results-dir results --paper-dir ../paper
python scripts/make_figures.py --results-dir results --out-dir ../paper/figures

cd ../paper && make            # or: latexmk -pdf main.tex
```

`aggregate_results.py` writes:

| File | Contents |
|---|---|
| `generated/numbers.tex` | `\newcommand` macros for every inline number |
| `generated/tab_main.tex` | main results, mean ± SD over seeds |
| `generated/tab_stats.tex` | equivalence + paired tests, Holm-corrected |
| `generated/tab_compute.tex` | measured latency / params / MACs |
| `generated/tab_external.tex` | cross-dataset transfer |
| `generated/tab_probe.tex` | probing saturation vs claimed hierarchy |
| `generated/tab_perclass.tex` | per-class AUC across dimensions |
| `generated/tab_ablations.tex` | every training condition |
| `generated/tab_quant.tex` | quantisation + differential penalty |
| `generated/tab_robustness.tex` | artefact robustness, small vs large prefix |

**Verify no placeholders survive before submitting:**
```bash
pdftotext main.pdf - | grep -c "\[TBD\]"     # must be 0
```

---

## Why the paper has no numbers yet

I could not run these experiments — no GPU, no PTB-XL, no checkpoints in the
environment where this was written. Rather than write plausible-looking
numbers into the manuscript for you to overwrite later, every result is a
macro that renders as a conspicuous red **[TBD]** until the pipeline fills it.

This matters more than it might seem. A revision responding to reviewers who
specifically objected to unsupported claims cannot contain invented figures,
even as placeholders — the failure mode is that one survives to submission.
The `\R{}` macro in `main.tex` makes that impossible to miss, and the grep
above is a hard gate.

Prose that depends on an outcome (chiefly the probing verdict, Section V-D) is
written to be honest under **either** result, and says so explicitly. If the
probes refute the hierarchy, the paper as written already commits to
withdrawing Table II rather than reinterpreting it. Passages marked `$\dagger$`
in the response letter need a final pass once you know the answer.

---

## What changed in the code

| Fix | Where | Why |
|---|---|---|
| Dataset-level normalisation | `data/dataset.py` | per-record z-scoring destroyed absolute voltage → likely cause of poor HYP |
| Weight decay grouping | `scripts/train.py` | `'bn' in name` missed BatchNorm inside `ConvBlock1d` |
| SVD leakage | `scripts/eval_transfer.py` | basis + classifier were fitted on the test split |
| Threshold tuning | `analysis/metrics.py` | F1 at fixed 0.5 understates imbalanced multi-label F1 |
| Model selection | `scripts/train.py` | selecting on d=512 biases toward one operating point |
| SE in Inception1D | `models/backbones.py` | paper claimed SE; code had none |
| Head/loss separation | `models/heads.py`, `losses.py` | `MatryoshkaLoss` held inference parameters |
| Chunked scoring | `models/model.py` | whole test embedding matrix was moved to GPU at once |
| MRL-E | `models/heads.py` | was listed as future work; now evaluated |
| Label smoothing off | `losses.py` | unusual on multi-label BCE; now an ablation, not a default |
| **ModuleDict key lookup** | `models/heads.py` | `dim not in self.classifiers` compared an `int` against string keys, so it **always** raised — every MRL forward pass was broken. Caught by the smoke test. |
| **Residual pooling** | `models/backbones.py` | with `residual_every > 1` the shortcut tensor was not pooled, so the skip addition failed on shape |
| **Eval loader alignment** | `data/dataset.py` | the training loader shuffles and drops the last batch; probing indexed into `X_train` order, silently misaligning every probe. `get_eval_loaders()` added |
| Hardcoded dim keys | `aggregate_results.py`, `make_figures.py` | `"16"`/`"512"` literals crashed aggregation whenever nesting dims were changed |
| Probe sample guard | `analysis/probing.py` | now reports when a descriptor is skipped rather than silently returning `n/a` |

---

## Repository layout

```
code/
  mecg/
    data/dataset.py            normalisation, lead subsets, artefact injection
    models/backbones.py        XResNet1D, Inception1D (+SE)
    models/heads.py            MRL, MRL-E, linear
    models/model.py            unified wrapper
    losses.py                  multi-granularity objective
    analysis/stats.py          DeLong, bootstrap, equivalence, Holm
    analysis/probing.py        prefix/slab probes, CKA, effective rank
    analysis/features.py       physiological descriptor extraction
    analysis/metrics.py        AUC/F1/AP with tuned thresholds
  scripts/                     preprocess, train, eval, probe, benchmark, aggregate
  configs/                     experiment configurations
  run_all.sh                   full matrix
paper/
  main.tex                     revised manuscript
  sections/                    per-section sources
  generated/                   auto-written tables + macros (do not hand-edit)
  response_to_reviewers.tex    point-by-point response
  IEEEtran.cls                 bundled for self-contained builds
```
