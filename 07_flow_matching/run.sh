#!/bin/bash
# All runs behind the 07_flow_matching results. Run from the repo root: bash 07_flow_matching/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
D=07_flow_matching
mkdir -p $D/runs

train_and_sweep() {  # <name> <train.py args...>
    local name=$1
    shift
    $PY $D/train.py --name "$name" "$@" > "$D/runs/$name.out" 2>&1
    $PY $D/sample.py --run "$D/runs/$name" >> "$D/runs/$name.out" 2>&1
}

# 8 Gaussians, then one reflow on its pairs (needs the first model)
( train_and_sweep 8gauss_s0 --data 8gauss &&
  train_and_sweep 8gauss_reflow_s0 --data 8gauss --reflow-from $D/runs/8gauss_s0 ) &
train_and_sweep swissroll_s0 --data swissroll &
train_and_sweep fmnist_cond_s0 --data fmnist --cond &
wait
