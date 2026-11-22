# DGMs

Small PyTorch projects on deep generative models.

## Roadmap

| # | Project | What we build | Key result |
|---|---|---|---|
| 01 | [gan](01_gan) | Toy GAN on 8 Gaussians (watch mode collapse), fix with WGAN-GP; DCGAN and conditional GAN on Fashion-MNIST | non-saturating GAN keeps 4 of 8 modes, WGAN-GP 8 of 8 (penalty 0.1 or 1); Fashion-MNIST FID-lite 14.8 (WGAN-GP) vs 114 |
| 02 | [vae](02_vae) | VAE on MNIST with a 2-D latent: latent map, interpolation, beta sweep (0.5, 1, 4) | -log p(x) 75.2 nats (16 latents) vs 129.4 (2); ELBO gap 4-12 nats |
| 03 | [flow](03_flow) | RealNVP on 2-D toys: learned density and how the Gaussian is warped | 8 of 8 modes, KL(p \|\| q) = 0.215 nats; thin bridges between modes |
| 04 | [autoregressive](04_autoregressive) | Gated PixelCNN on binarized MNIST: exact test NLL, pixel-by-pixel sampling time | exact -log p(x) 57.7 nats (VAE: 75.2); 64 samples in 7.1 s |
| 05 | [ssm](05_ssm) | Same task as 04 as a 784-step sequence: diagonal SSM (S4D) vs a causal transformer of equal size | S4D 55.2 nats, transformer 56.1 (PixelCNN 57.7); S4D step 3x cheaper to train, flat sampling cost |
| 06 | [ddpm](06_ddpm) | DDPM on a swiss roll, then a small U-Net on Fashion-MNIST; DDIM at 10/50/1000 steps; classifier-free guidance | DDIM 50 steps = DDPM 1000 (FID-lite 7.2 vs 7.1) at 1/20 the time; guidance: label accuracy 89 to 100 %, FID-lite 3.7 to 42 |
| 07 | [flow_matching](07_flow_matching) | Same U-Net as 06 with the rectified-flow loss; quality vs number of steps | |
| 08 | [vqvae_prior](08_vqvae_prior) | VQ-VAE on Fashion-MNIST, then a small transformer prior over the codes | samples FID-lite 20.8 (reconstructions 19.0); 144 bits per image vs 343 |
| 09a | [clip](09a_clip) | Tiny CLIP on synthetic shape scenes; text-to-image retrieval | retrieval top-1 97.3 % (100 % with swapped-caption negatives); binding learned after a bag-of-words phase |
| 09b | [mini_llava](09b_mini_llava) | Frozen SigLIP + frozen SmolLM2-135M + trained projector, then LoRA; answers questions about shapes | 91.9 % of questions with the projector alone, 99.1 % with LoRA; 37 % with another image |
| 09c | [text2image](09c_text2image) | U-Net from 06/07 conditioned on the 09a text encoder; 09b checks the generated images | |

## Data

Downloads go to `data/` (not tracked).

| Data | Size | Used in |
|---|---|---|
| 2-D toys: 8 Gaussians, swiss roll, two moons (generated) | none | 01, 03, 06, 07 |
| MNIST, binarized MNIST | 11 MB | 02, 04, 05 |
| Fashion-MNIST | 30 MB | 01, 06, 07, 08 |
| Shape scenes with captions and Q&A (generated) | none | 09a, 09b, 09c |
| Pretrained SigLIP-base and SmolLM2-135M-Instruct | ~1 GB | 09b, 09c |

## Evaluation

| Model type | Metric |
|---|---|
| Likelihood models (VAE bound, flow, PixelCNN, SSM) | test NLL in nats, or bits per dim |
| Sample-only models (GAN, DDPM, flow matching) | FID-lite, class coverage |
| 2-D toys | samples over the true density; modes covered out of 8 |
| VLM | retrieval top-1 (09a), answer accuracy per question type (09b), share of correct images (09c) |
| All | sample grid at fixed seeds; time to draw 64 samples |

## Layout

```
common/            shared code: data, toy sets, shape scenes, networks, metrics, plots
NN_name/           one project
  README.md        notes and results
  objective.py     losses
  sample.py        sampling and evaluation
  model.py         networks
  train.py         training loop
  figs/            figures shown in the README
```

Runs write checkpoints and logs to `runs/` (not tracked).

## Setup

```
pip install -r requirements.txt
```
