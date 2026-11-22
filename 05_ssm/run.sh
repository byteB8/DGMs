#!/bin/bash
# All runs behind the 05_ssm results. Run from the repo root: bash 05_ssm/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 05_ssm/runs

for kind in ssm transformer; do
    ( $PY 05_ssm/train.py --kind $kind > "05_ssm/runs/${kind}_s0.out" 2>&1 &&
      $PY 05_ssm/sample.py --run "05_ssm/runs/${kind}_s0" >> "05_ssm/runs/${kind}_s0.out" 2>&1 ) &
done
wait
