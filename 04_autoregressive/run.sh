#!/bin/bash
# All runs behind the 04_autoregressive results. Run from the repo root: bash 04_autoregressive/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 04_autoregressive/runs

$PY 04_autoregressive/train.py > 04_autoregressive/runs/pixelcnn_ch64_s0.out 2>&1
$PY 04_autoregressive/sample.py --run 04_autoregressive/runs/pixelcnn_ch64_s0
