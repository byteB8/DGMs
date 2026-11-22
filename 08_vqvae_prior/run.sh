#!/bin/bash
# All runs behind the 08_vqvae_prior results. Run from the repo root: bash 08_vqvae_prior/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
D=08_vqvae_prior
mkdir -p $D/runs

$PY $D/train.py --stage vqvae > $D/runs/vqvae_s0.out 2>&1
$PY $D/train.py --stage prior --vqvae-run $D/runs/vqvae_s0 > $D/runs/prior_s0.out 2>&1
$PY $D/sample.py --vqvae-run $D/runs/vqvae_s0 --prior-run $D/runs/prior_s0 >> $D/runs/prior_s0.out 2>&1
