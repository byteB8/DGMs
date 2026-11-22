#!/bin/bash
# All runs behind the 01_gan results. Run from the repo root: bash 01_gan/run.sh
# Jobs are small, so they run side by side on GPU 0 and GPU 1.
set -e
export OMP_NUM_THREADS=2
PY=${PY:-python}
mkdir -p 01_gan/runs

job() {  # job <gpu> <run name> <train.py args...>
    local gpu=$1 name=$2
    shift 2
    CUDA_VISIBLE_DEVICES=$gpu $PY 01_gan/train.py --name "$name" "$@" > "01_gan/runs/$name.out" 2>&1 &
}

# 8 Gaussians: three generator losses, three seeds each
for seed in 0 1 2; do
    job 0 "8gauss_minimax_s$seed" --data 8gauss --loss minimax --seed $seed
    job 0 "8gauss_ns_s$seed" --data 8gauss --loss ns --seed $seed
    job 1 "8gauss_wgan-gp_s$seed" --data 8gauss --loss wgan-gp --seed $seed
done

# Fashion-MNIST: DCGAN, conditional DCGAN, WGAN-GP
job 0 fmnist_ns_s0 --data fmnist --loss ns
job 1 fmnist_ns_cond_s0 --data fmnist --loss ns --cond
job 1 fmnist_wgan-gp_s0 --data fmnist --loss wgan-gp

wait
