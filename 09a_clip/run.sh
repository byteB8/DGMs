#!/bin/bash
# All runs behind the 09a_clip results. Run from the repo root: bash 09a_clip/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 09a_clip/runs

$PY 09a_clip/train.py > 09a_clip/runs/clip_s0.out 2>&1 &
$PY 09a_clip/train.py --hard-negatives > 09a_clip/runs/clip_hardneg_s0.out 2>&1 &
wait
