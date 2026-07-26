#!/usr/bin/env bash
# ===========================================================================
#  Matryoshka-ECG : full revision experiment matrix
# ===========================================================================
#  Every experiment the reviewers asked for, plus the extensions that a strong
#  GPU makes affordable. Idempotent: completed runs are skipped, so an
#  interrupted job resumes cleanly.
#
#  QUICK START
#    python scripts/smoke_test.py --full     # validate plumbing, ~2 min
#    SEEDS="0" ./run_all.sh                  # one seed end to end, ~3 h
#    ./run_all.sh                            # full matrix
#
#  ENV KNOBS
#    SEEDS="0 1 2 3 4"    training seeds
#    STAGES="2 3 9"       run only these stages
#    BS=256               batch size override (raise it, you have the VRAM)
#    AMP=bf16             bf16 | fp16 | off
#    COMPILE=1            enable torch.compile (slower first epoch, faster after)
#    TASK=super           super | sub  (5-class or 23-class diagnostic)
# ===========================================================================
set -euo pipefail

SEEDS="${SEEDS:-0 1 2 3 4}"
DIMS="${DIMS:-16 32 64 128 256 512}"
STAGES="${STAGES:-1 2 3 4 5 6 7 8 9 10}"
DATA_DIR="${DATA_DIR:-data/processed}"
RESULTS="${RESULTS:-results}"
PAPER="${PAPER:-../paper}"
BS="${BS:-128}"
AMP="${AMP:-bf16}"
TASK="${TASK:-super}"
INC_CFG=configs/mrl_inception1d.yaml
XRES_CFG=configs/mrl_xresnet1d101.yaml

EXTRA=(--batch-size "$BS" --amp "$AMP" --data-dir "$DATA_DIR" --out-dir "$RESULTS")
[ "${COMPILE:-0}" = "1" ] && EXTRA+=(--compile)

mkdir -p "$RESULTS"/{checkpoints,runs,logs}

have ()  { [ -f "$RESULTS/runs/$1/results.json" ]; }
log ()   { echo -e "\n\033[1;36m>>> $*\033[0m"; }
stage () { [[ " $STAGES " == *" $1 "* ]]; }

train () {  # train <run-name> <config> <seed> [extra args...]
  local name=$1 cfg=$2 seed=$3; shift 3
  have "$name" && { echo "    skip $name"; return 0; }
  python scripts/train.py --config "$cfg" --seed "$seed" --run-name "$name" \
    "${EXTRA[@]}" "$@" 2>&1 | tee "$RESULTS/logs/${name}.log"
}

# ---------------------------------------------------------------------------
if stage 1; then
log "STAGE 1  preprocessing"
if [ ! -f "$DATA_DIR/X_train.npy" ]; then
  if [ "$TASK" = "sub" ]; then
    python scripts/preprocess_ptbxl.py --raw data/ptbxl_raw --out "$DATA_DIR" \
      --task subdiagnostic
  else
    python scripts/preprocess_ptbxl.py --raw data/ptbxl_raw --out "$DATA_DIR"
  fi
else
  echo "    already preprocessed"
fi
fi

# ---------------------------------------------------------------------------
if stage 2; then
log "STAGE 2  MRL models, both backbones, multi-seed"
for S in $SEEDS; do
  train "inception1d_mrl_s${S}"   "$INC_CFG"  "$S" --save-train-embeddings
  train "xresnet1d101_mrl_s${S}"  "$XRES_CFG" "$S"
done
fi

# ---------------------------------------------------------------------------
if stage 3; then
log "STAGE 3  fixed-dimension baselines, SAME backbone (R1-2b / R2-2)"
# This is the comparison that isolates nesting from architecture. It is the
# single most important addition in the revision, and also the most expensive:
# 2 backbones x 6 dims x |SEEDS| full training runs.
for S in $SEEDS; do
  for D in $DIMS; do
    train "inception1d_fixed_d${D}_s${S}"  "$INC_CFG"  "$S" \
      --head linear --embedding-dim "$D"
    train "xresnet1d101_fixed_d${D}_s${S}" "$XRES_CFG" "$S" \
      --head linear --embedding-dim "$D"
  done
done
fi

# ---------------------------------------------------------------------------
if stage 4; then
log "STAGE 4  ablations"
for S in $SEEDS; do
  train "inception1d_mrle_s${S}" "$INC_CFG" "$S" --head mrl-e
done
# normalisation: tests the amplitude/HYP hypothesis
train "inception1d_mrl_perrecord_s0" "$INC_CFG" 0 --norm-mode per_record
# label smoothing on multi-label BCE
train "inception1d_mrl_ls01_s0" "$INC_CFG" 0 --label-smoothing 0.1
# nesting weight strategies
for W in linear exponential inverse; do
  train "inception1d_mrl_w${W}_s0" "$INC_CFG" 0 --weight-strategy "$W"
done
# faithful InceptionTime width/residual schedule
train "inception1d_constant_s0" "$INC_CFG" 0 \
  --backbone-kwargs '{"width_mode":"constant","residual_every":3}'
# smaller backbone for the compute-vs-accuracy frontier
for S in $SEEDS; do
  train "xresnet1d50_mrl_s${S}" "$XRES_CFG" "$S" --backbone xresnet1d50
done
fi

# ---------------------------------------------------------------------------
if stage 5; then
log "STAGE 5  natively trained reduced-lead models (wearable proxy, R1-1)"
for L in 8lead 6lead 3lead 2lead lead_I lead_II; do
  for S in $SEEDS; do
    train "inception1d_mrl_${L}_s${S}" "$INC_CFG" "$S" --lead-subset "$L"
  done
done
fi

# ---------------------------------------------------------------------------
BEST="$RESULTS/checkpoints/inception1d_mrl_s0_best.pt"

if stage 6; then
log "STAGE 6  leakage-free SVD baseline"
python scripts/eval_transfer.py --mode svd --checkpoint "$BEST" \
  --data-dir "$DATA_DIR" --out "$RESULTS/transfer.json"
fi

if stage 7; then
log "STAGE 7  robustness sweep + zero-shot lead ablation"
python scripts/eval_transfer.py --mode robustness --checkpoint "$BEST" \
  --data-dir "$DATA_DIR" --out "$RESULTS/transfer.json"
python scripts/eval_transfer.py --mode leads --checkpoint "$BEST" \
  --data-dir "$DATA_DIR" --out "$RESULTS/transfer.json"
fi

if stage 8; then
log "STAGE 8  cross-dataset transfer (R1-1 / R2-5)"
for TAG in cpsc2018 chapman georgia; do
  if [ -f "data/processed_${TAG}/X_test.npy" ]; then
    python scripts/eval_transfer.py --mode external --checkpoint "$BEST" \
      --external-dir "data/processed_${TAG}" --tag "$TAG" \
      --data-dir "$DATA_DIR" --out "$RESULTS/transfer.json"
  else
    echo "    data/processed_${TAG} missing -- see RUNBOOK.md section 3"
  fi
done
fi

if stage 9; then
log "STAGE 9  probing, measured hardware profile, quantisation"
python scripts/run_probing.py --checkpoint "$BEST" \
  --data-dir "$DATA_DIR" --out "$RESULTS/probing.json"
python scripts/benchmark_hardware.py --checkpoint "$BEST" \
  --out "$RESULTS/benchmark.json"
python scripts/quantize.py --checkpoint "$BEST" --data-dir "$DATA_DIR" \
  --modes fp32 dynamic static fp16 --out "$RESULTS/quantization.json"
fi

# ---------------------------------------------------------------------------
if stage 10; then
log "STAGE 10  aggregate -> paper"
python scripts/aggregate_results.py --results-dir "$RESULTS" --paper-dir "$PAPER"
python scripts/make_figures.py --results-dir "$RESULTS" --out-dir "$PAPER/figures"
log "Now: cd $PAPER && make check"
fi
