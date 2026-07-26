#!/usr/bin/env bash
# Fetch PTB-XL and (optionally) the external corpora used for cross-dataset
# evaluation. All are freely downloadable; none require credentialing.
set -euo pipefail
mkdir -p data && cd data

if [ ! -d ptbxl_raw ]; then
  echo ">>> PTB-XL v1.0.3 (~1.7 GB)"
  wget -q --show-progress -r -N -c -np \
    https://physionet.org/files/ptb-xl/1.0.3/
  mv physionet.org/files/ptb-xl/1.0.3 ptbxl_raw
  rm -rf physionet.org
else
  echo ">>> PTB-XL already present"
fi

if [ "${WITH_EXTERNAL:-0}" = "1" ]; then
  mkdir -p external && cd external
  BASE=https://physionet.org/files/challenge-2020/1.0.2/training
  for f in cpsc_2018 georgia chapman_shaoxing; do
    [ -d "$f" ] && { echo ">>> $f present"; continue; }
    echo ">>> $f"
    wget -q --show-progress "$BASE/$f.tar.gz"
    tar xzf "$f.tar.gz" && rm "$f.tar.gz"
  done
  cd ..
fi
echo ">>> done. Next: python scripts/preprocess_ptbxl.py --raw data/ptbxl_raw --out data/processed"
