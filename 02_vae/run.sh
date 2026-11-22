#!/bin/bash
# All runs behind the 02_vae results. Run from the repo root: bash 02_vae/run.sh
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 02_vae/runs

# latent size 2 (to look at) and 16 (better fit), three KL weights each
for z in 2 16; do
    for beta in 0.5 1 4; do
        $PY 02_vae/train.py --z-dim $z --beta $beta > "02_vae/runs/z${z}_beta${beta}.out" 2>&1 &
    done
done
wait
