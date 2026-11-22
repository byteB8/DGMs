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
| WGAN-GP | 5 critic steps per generator step, $\lambda \in \{0.1, 1, 10\}$ | 5 critic steps, $\lambda = 10$ |
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

- [x] 8 Gaussians: minimax vs non-saturating vs WGAN-GP (penalty weight 0.1, 1, 10)
- [x] Fashion-MNIST: DCGAN and WGAN-GP
- [x] Fashion-MNIST: conditional DCGAN

## Results

**8 Gaussians** (10k samples at the end of training).

| Model | Modes covered (of 8) | High-quality samples |
|---|---|---|
| minimax | 8 | 0.96 |
| non-saturating | 4 | 0.96 |
| WGAN-GP, $\lambda = 10$ | 0 | 0.00 |
| WGAN-GP, $\lambda = 1$ | 8 | 0.95 |
| WGAN-GP, $\lambda = 0.1$ | 8 | 0.97 |

**Fashion-MNIST** (10k samples; real training images score FID-lite 0.4 against the test set).

| Model | FID-lite | Class KL | Rarest class share | Label accuracy |
|---|---|---|---|---|
| DCGAN, non-saturating | 114.3 | 0.295 | 0.008 | - |
| DCGAN, WGAN-GP ($\lambda = 10$) | 14.8 | 0.013 | 0.075 | - |
| conditional DCGAN, non-saturating | 56.6 | 0.005 | 0.085 | 0.94 |

- **The non-saturating loss drops modes.** It keeps 4 of 8 Gaussians, losing them one by one during training,
  and on Fashion-MNIST the rarest class gets 0.8 % of the samples instead of 10 %.
  With the best discriminator, this loss follows the gradient of $\mathrm{KL}(q \,\|\, p) - 2\,\mathrm{JS}(p \,\|\, q)$
  (Arjovsky and Bottou, 2017). $\mathrm{KL}(q \,\|\, p)$ barely punishes missing modes: if $q$ sits evenly on 4 of the
  8 equal modes, $\mathrm{KL}(q \,\|\, p) = \log 2 \approx 0.69$ nats, while $\mathrm{KL}(p \,\|\, q)$ would be infinite.
- **The minimax loss did not saturate here.** Its discriminator loss stayed at about 1.38, close to $\log 4$, so
  $D(x) \approx 1/2$: the discriminator never became confident, which is the case where $\log(1 - D)$ has no gradient.
  Its generator gradients were about 10 times smaller than the non-saturating ones, and it kept all 8 modes.
- **The WGAN-GP penalty weight decides the outcome.** $\lambda = 0.1$ and $1$ cover all 8 modes. With $\lambda = 10$
  the toy run never settles: the samples swing across the plane and the critic's estimate of $W_1$ keeps changing sign.
  The same $\lambda = 10$ gives the best Fashion-MNIST model (FID-lite 14.8, balanced classes).
- **Labels fix class balance.** The conditional model hits the requested class 94 % of the time.
  Its FID-lite was lowest (about 30) at 5k-7k steps and then rose to 56.6; the table reports the final model.
