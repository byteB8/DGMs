#!/bin/bash
# All runs behind the 09c_text2image results. Run from the repo root after 09a and 09b: bash 09c_text2image/run.sh
set -e
export OMP_NUM_THREADS=2 TRANSFORMERS_VERBOSITY=error
PY=${PY:-python}
D=09c_text2image
mkdir -p $D/runs

$PY $D/train.py > $D/runs/t2i_s0.out 2>&1
$PY $D/sample.py --run $D/runs/t2i_s0 >> $D/runs/t2i_s0.out 2>&1
