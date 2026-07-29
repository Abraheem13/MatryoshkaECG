#!/usr/bin/env bash
cd ~/MatryoshkaECG/code
echo "=== training now ==="
pgrep -af train.py | grep -o "run-name [a-z0-9_]*" | sort -u
echo "=== stream A ==="; tail -n 2 /tmp/A.log
echo "=== stream B ==="; tail -n 2 /tmp/B.log
echo "=== queue ==="; cat /tmp/QUEUE.log 2>/dev/null
echo "=== completed: $(ls results/runs 2>/dev/null | wc -l) ==="
ls results/runs 2>/dev/null | tr '\n' ' '
echo; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
