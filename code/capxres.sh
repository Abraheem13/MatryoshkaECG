#!/usr/bin/env bash
# kill XResNet seeds 2,3,4 whenever they appear; keep seeds 0 and 1
for i in $(seq 1 2880); do
  pkill -f "run-name xresnet1d101_mrl_s[234]" 2>/dev/null && \
    echo "capped an xresnet run $(date)"
  sleep 60
done
