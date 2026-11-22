# 01 GAN

Learn to sample from a data distribution $p$ using only its samples.
A generator $G$ maps noise $z \sim \mathcal N(0, I)$ to $x = G(z)$; these samples follow a distribution $q$.
A discriminator $D$ tries to tell real samples from generated ones.

## Math

**The game.**

$$\min_G \max_D \; V(D, G) = \mathbb E_{x \sim p}[\log D(x)] + \mathbb E_{z}[\log(1 - D(G(z)))]$$

**Best discriminator.** For a fixed $G$, maximise $p(x) \log D(x) + q(x) \log(1 - D(x))$ at every $x$:

$$D^*(x) = \frac{p(x)}{p(x) + q(x)}, \qquad V(D^*, G) = 2\,\mathrm{JS}(p \,\|\, q) - \log 4$$

So the generator minimises the Jensen-Shannon divergence. Example: where $p(x) = 0.3$ and $q(x) = 0.1$, $D^*(x) = 0.75$.
At the optimum $q = p$, $D^* = 1/2$ everywhere and the discriminator loss $-V$ equals $\log 4 \approx 1.386$.

**Saturation.** Write $D = \sigma(l)$ with logit $l$. Then

$$\frac{\partial}{\partial l} \log(1 - \sigma(l)) = -\sigma(l), \qquad \frac{\partial}{\partial l} \left(-\log \sigma(l)\right) = -(1 - \sigma(l))$$

When $D$ confidently rejects a fake ($\sigma(l) \approx 0$), the minimax generator loss gives almost no gradient,
while the non-saturating loss $-\mathbb E_q[\log D(x)]$ gives a gradient near $-1$. Both have the same fixed point.

**WGAN.** By Kantorovich-Rubinstein duality,

$$W_1(p, q) = \sup_{\lVert f \rVert_L \le 1} \mathbb E_p[f(x)] - \mathbb E_q[f(x)]$$

A critic $f$ estimates this sup. The gradient penalty $\lambda\, \mathbb E_{\hat x}(\lVert \nabla f(\hat x) \rVert - 1)^2$,
on points $\hat x$ between real and fake samples, keeps $f$ close to 1-Lipschitz.

**Conditional GAN.** Both networks see the label $y$: $G(z, y)$ and $D(x, y)$.
The critic uses projection: $D(x, y) = w^\top h(x) + e_y^\top h(x)$.

## Code map

| Math | Code |
|---|---|
| $-\mathbb E_p[\log D] - \mathbb E_q[\log(1 - D)]$ | `objective.py: d_loss_gan` |
| $\mathbb E_q[\log(1 - D)]$ | `objective.py: g_loss_minimax` |
| $-\mathbb E_q[\log D]$ | `objective.py: g_loss_nonsaturating` |
| $\mathbb E_q[f] - \mathbb E_p[f]$ | `objective.py: critic_loss_wgan` |
| $-\mathbb E_q[f]$ | `objective.py: g_loss_wgan` |
| $\lambda\, \mathbb E(\lVert \nabla f(\hat x) \rVert - 1)^2$ | `objective.py: gradient_penalty` |
| $w^\top h(x) + e_y^\top h(x)$ | `model.py: ConvCritic.forward` |
| $z \sim \mathcal N(0, I),\ x = G(z)$ | `sample.py: generate` |

## Setup

| | 8 Gaussians | Fashion-MNIST |
|---|---|---|
| Data | 8 Gaussians on a circle of radius 2, std 0.05 | 60k train images, 28 x 28 |
| Networks | MLPs, 3 hidden layers of 512 | DCGAN-style conv nets |
| Noise | $z \in \mathbb R^2$ | $z \in \mathbb R^{64}$ |
| Optimiser | Adam, lr 1e-4, betas (0.5, 0.9), batch 256 | Adam, lr 2e-4, betas (0.5, 0.9), batch 128 |
| Length | 20k generator steps | 15k generator steps |
| WGAN-GP | 5 critic steps per generator step, $\lambda = 10$ | same |
| Metrics | modes covered, share of high-quality samples | FID-lite, class balance, label accuracy |

A sample is high quality if it lies within 3 std of a mode; a mode is covered if it holds at least 1 % of all samples.
Metrics use 10k samples.

## Run

```
bash 01_gan/run.sh           # all runs
python 01_gan/figures.py     # figures and tables
python 01_gan/train.py --data 8gauss --loss ns    # a single run
```

## Checklist

- [ ] 8 Gaussians: minimax vs non-saturating vs WGAN-GP, 3 seeds
- [ ] Fashion-MNIST: DCGAN and WGAN-GP
- [ ] Fashion-MNIST: conditional DCGAN

## Results
