# 08 VQ-VAE with a transformer prior

Two stages. A VQ-VAE compresses each Fashion-MNIST image into a 7 x 7 grid of discrete codes;
a transformer then learns which code grids are likely, one code at a time. Sampling draws a code grid
from the transformer and decodes it.

## Math

**Quantisation.** The encoder gives a vector $z_e$ at each of the 7 x 7 positions. Each is replaced by its nearest
codebook entry:

$$k = \arg\min_j \lVert z_e - e_j \rVert^2, \qquad z_q = e_k$$

**Straight-through gradient.** $\arg\min$ has no gradient, so the forward pass uses $e_k$ and the backward pass copies
the gradient of $z_q$ onto $z_e$: in code, $z_q = z_e + \mathrm{sg}(e_k - z_e)$, where sg stops gradients.

**Loss.**

$$L = \lVert x - \mathrm{dec}(z_q) \rVert^2 + \lVert \mathrm{sg}(z_e) - e_k \rVert^2 + \beta\, \lVert z_e - \mathrm{sg}(e_k) \rVert^2$$

reconstruction; codebook term (moves the code towards the encoder output); commitment term (keeps the encoder near
its code), with $\beta = 0.25$.

**Link to the VAE (02).** The posterior $q(z \mid x)$ puts all its mass on one code per position. With a uniform prior
over $K$ codes, the KL term is the constant $\log K$ per position, so it drops out of training.
The prior is learned afterwards instead.

**Prior.** Over the 49 codes in raster order, conditioned on the class $y$, by the chain rule (as in 04):

$$\log p(c \mid y) = \sum_{t=1}^{49} \log p(c_t \mid y, c_{<t})$$

**Compression.** One image becomes $49 \log_2 128 = 343$ bits as codes. A good prior needs fewer bits on average:
$-\log_2 p(c \mid y)$.

**Codebook usage.** Perplexity $= \exp(H)$ of the code histogram: the effective number of codes in use.
Unused codes are restarted at random encoder outputs after each epoch.

## Code map

| Math | Code |
|---|---|
| $k = \arg\min_j \lVert z_e - e_j \rVert^2$ | `model.py: VectorQuantizer.indices` |
| $z_q = z_e + \mathrm{sg}(e_k - z_e)$ | `model.py: VectorQuantizer.forward` |
| reconstruction + codebook + $\beta$ commitment | `objective.py: vqvae_loss` |
| $-\sum_t \log p(c_t \mid y, c_{<t})$ | `objective.py: prior_nll` |
| $\exp(H)$ of code usage | `objective.py: perplexity` |
| restart unused codes | `train.py: train_vqvae` |
| $c_t \sim p(c_t \mid y, c_{<t})$, then decode | `sample.py: sample_codes`, `model.py: VQVAE.decode` |

## Setup

| | |
|---|---|
| Data | Fashion-MNIST, images in [-1, 1] |
| VQ-VAE | conv encoder 28 -> 14 -> 7 with residual blocks; 128 codes of dimension 64; Adam, lr 2e-4, batch 128, 20 epochs |
| Prior | 6 transformer blocks, width 256, 8 heads, class token first; AdamW, lr 3e-4 (one-cycle), batch 256, 30 epochs |
| Metrics | test MSE and PSNR, code usage, prior NLL, FID-lite and label accuracy of samples (2000 images) |

## Run

```
bash 08_vqvae_prior/run.sh          # VQ-VAE, then prior, then sampling
python 08_vqvae_prior/figures.py    # figures and tables
```

## Checklist

- [ ] VQ-VAE: reconstructions and code usage
- [ ] Transformer prior over codes, class-conditional
- [ ] Samples: quality, label accuracy and sampling time

## Results
