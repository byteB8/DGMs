# 02 VAE

Learn a latent variable model $p(x) = \int p(x \mid z)\, p(z)\, dz$ of binarized MNIST digits,
together with an encoder $q(z \mid x)$ that maps an image to a distribution over latents.

## Math

**Model.** $p(z) = \mathcal N(0, I)$, $p(x \mid z) = \prod_i \mathrm{Bernoulli}(x_i;\, \sigma(\text{dec}(z))_i)$,
$q(z \mid x) = \mathcal N(\mu(x), \mathrm{diag}\, \sigma^2(x))$.

**ELBO.** For any $q$,

$$\log p(x) = \underbrace{\mathbb E_{q(z \mid x)}[\log p(x \mid z)] - \mathrm{KL}(q(z \mid x) \,\|\, p(z))}_{\mathrm{ELBO}(x)} + \mathrm{KL}(q(z \mid x) \,\|\, p(z \mid x))$$

The last KL is $\ge 0$, so the ELBO is a lower bound, and the gap is how far $q$ is from the true posterior.

**KL in closed form.** For a diagonal Gaussian against $\mathcal N(0, I)$:

$$\mathrm{KL} = \tfrac12 \sum_j \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

Example: $\mu = 1, \sigma = 1$ in one dimension gives $\tfrac12(1 + 1 - 0 - 1) = 0.5$ nats.

**Reparameterisation.** $z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal N(0, I)$.
The randomness moves into $\epsilon$, so gradients flow through $\mu$ and $\sigma$.

**Estimating $\log p(x)$.** Importance sampling with $q$ as proposal:

$$\log p(x) \approx \log \frac1k \sum_{j=1}^k \frac{p(x \mid z_j)\, p(z_j)}{q(z_j \mid x)}, \qquad z_j \sim q(z \mid x)$$

This is a lower bound that tightens as $k$ grows ($k = 1$ is the ELBO). We use $k = 500$.

**beta-VAE.** Minimise $D + \beta R$ with distortion $D = -\mathbb E_q[\log p(x \mid z)]$ and rate
$R = \mathrm{KL}(q(z \mid x) \,\|\, p(z))$. $\beta = 1$ is the negative ELBO; other $\beta$ pick another point on the
rate-distortion curve.

## Code map

| Math | Code |
|---|---|
| $\log p(x \mid z)$ for Bernoulli pixels | `objective.py: bernoulli_log_likelihood` |
| $\mathrm{KL}(\mathcal N(\mu, \sigma^2) \,\|\, \mathcal N(0, I))$ | `objective.py: gaussian_kl` |
| $D + \beta R$ | `objective.py: negative_elbo` |
| $\log \frac1k \sum_j p(x \mid z_j) p(z_j) / q(z_j \mid x)$ | `objective.py: importance_log_likelihood` |
| $z = \mu + \sigma \odot \epsilon$ | `model.py: VAE.reparameterize` |
| $q(z \mid x)$, $p(x \mid z)$ | `model.py: VAE.encode`, `VAE.decode` |

## Setup

| | |
|---|---|
| Data | MNIST thresholded at 0.5 (60k train, 10k test) |
| Networks | MLP encoder and decoder, 2 hidden layers of 512 |
| Latent size | 2 (to look at) and 16 |
| beta | 0.5, 1, 4 |
| Optimiser | Adam, lr 1e-3, batch 128, 30 epochs |
| Metrics | test reconstruction, KL, -ELBO, -log p(x) with $k = 500$, all in nats per image |

## Run

```
bash 02_vae/run.sh           # all runs
python 02_vae/figures.py     # figures and table
python 02_vae/train.py --z-dim 2 --beta 1    # a single run
```

## Checklist

- [ ] 2-D latent: latent map, encoder means, interpolation
- [ ] ELBO vs importance-sampled log p(x)
- [ ] beta sweep: rate vs distortion

## Results
