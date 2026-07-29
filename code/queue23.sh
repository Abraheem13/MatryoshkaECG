#!/usr/bin/env bash
cd ~/MatryoshkaECG/code
while kill -0 963938 2>/dev/null || kill -0 963939 2>/dev/null; do sleep 60; done
echo ">>> PHASE 2 $(date)"
CUDA_VISIBLE_DEVICES=0 SEEDS="0" STAGES="4 5" BS=256 AMP=bf16 ./run_all.sh > /tmp/C.log 2>&1
echo ">>> PHASE 3 $(date)"
STAGES="6 7 9 10" ./run_all.sh > /tmp/D.log 2>&1
echo "=== ALL DONE $(date) ==="
