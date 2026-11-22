#!/bin/bash
# All runs behind the 03_flow results. Run from the repo root: bash 03_flow/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 03_flow/runs

for data in 8gauss swissroll moons; do
    $PY 03_flow/train.py --data $data > "03_flow/runs/${data}_s0.out" 2>&1 &
done
wait
