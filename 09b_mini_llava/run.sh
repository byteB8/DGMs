#!/bin/bash
# All runs behind the 09b_mini_llava results. Run from the repo root: bash 09b_mini_llava/run.sh
# The first run downloads SigLIP-base and SmolLM2-135M-Instruct and caches SigLIP features in data/shapes/.
set -e
export OMP_NUM_THREADS=2 TRANSFORMERS_VERBOSITY=error
PY=${PY:-python}
D=09b_mini_llava
mkdir -p $D/runs

$PY $D/train.py --stage projector > $D/runs/projector_s0.out 2>&1
$PY $D/sample.py --run $D/runs/projector_s0 >> $D/runs/projector_s0.out 2>&1
$PY $D/train.py --stage lora --init-from $D/runs/projector_s0 > $D/runs/lora_s0.out 2>&1
$PY $D/sample.py --run $D/runs/lora_s0 >> $D/runs/lora_s0.out 2>&1
