#!/bin/bash
# All runs behind the 06_ddpm results. Run from the repo root: bash 06_ddpm/run.sh
# One conditional Fashion-MNIST model serves both unconditional sampling ("no label") and guidance.
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 06_ddpm/runs

run() {  # run <name> <train.py args...>: train, then the sampler / guidance sweep
    local name=$1
    shift
    ( $PY 06_ddpm/train.py --name "$name" "$@" > "06_ddpm/runs/$name.out" 2>&1 &&
      $PY 06_ddpm/sample.py --run "06_ddpm/runs/$name" >> "06_ddpm/runs/$name.out" 2>&1 ) &
}

run swissroll_s0 --data swissroll
run 8gauss_s0 --data 8gauss
run fmnist_cond_s0 --data fmnist --cond
wait
