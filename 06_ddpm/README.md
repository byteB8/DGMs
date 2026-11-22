# 06 DDPM

Turn data into noise with a fixed process, then learn to undo it one small step at a time.
The network only has to predict the noise that was added.

## Math

**Forward process.** With a schedule $\beta_1, \dots, \beta_T$ and $\alpha_t = 1 - \beta_t$:

$$q(x_t \mid x_{t-1}) = \mathcal N\!\left(\sqrt{\alpha_t}\, x_{t-1},\ \beta_t I\right)$$

Composing Gaussians gives a closed form for any $t$, with $\bar\alpha_t = \prod_{s \le t} \alpha_s$:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal N(0, I)$$

Example: $T = 2$, $\beta_1 = \beta_2 = 0.5$ gives $\bar\alpha_2 = 0.25$, so $x_2 = 0.5\, x_0 + \sqrt{0.75}\, \epsilon$.
With $T = 1000$ and $\beta$ from $10^{-4}$ to $0.02$, $\bar\alpha_T \approx 4 \times 10^{-5}$: $x_T$ is almost pure noise.

**Reverse step.** Given $x_0$, the step back is Gaussian too: $q(x_{t-1} \mid x_t, x_0) = \mathcal N(\tilde\mu_t, \tilde\beta_t I)$.
Writing $x_0$ through the noise, $x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon) / \sqrt{\bar\alpha_t}$, the mean becomes

$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon\right)$$

So the model $p_\theta(x_{t-1} \mid x_t)$ only needs a noise estimate $\epsilon_\theta(x_t, t)$.

**Loss.** The ELBO is a sum of KLs between these Gaussians; each is a weighted $\lVert \epsilon - \epsilon_\theta \rVert^2$.
Dropping the weights gives the simple loss

$$L = \mathbb E_{t, x_0, \epsilon} \left\lVert \epsilon - \epsilon_\theta\!\left(\sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon,\ t\right) \right\rVert^2$$

**Link to scores.** $\epsilon_\theta(x_t, t) \approx -\sqrt{1 - \bar\alpha_t}\, \nabla_x \log p_t(x_t)$: predicting noise is
estimating the score of the noised data.

**DDIM.** A deterministic map with the same marginals $q(x_t \mid x_0)$. From $x_t$, estimate
$\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\, \epsilon_\theta) / \sqrt{\bar\alpha_t}$, then jump to any earlier time $s$:
$x_s = \sqrt{\bar\alpha_s}\, \hat x_0 + \sqrt{1 - \bar\alpha_s}\, \epsilon_\theta$. This allows far fewer than $T$ steps.

**Classifier-free guidance.** Train one network with labels, replacing 10 % of them by "no label". At sampling time

$$\tilde\epsilon = (1 + w)\, \epsilon_\theta(x_t, t, y) - w\, \epsilon_\theta(x_t, t, \varnothing)$$

In score terms this follows $\nabla \log p(x \mid y) + w\, \nabla \log p(y \mid x)$: the label is pushed harder as $w$ grows.

## Code map

| Math | Code |
|---|---|
| $\beta_t$, $\alpha_t$, $\bar\alpha_t$ | `objective.py: NoiseSchedule` |
| $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$ | `objective.py: q_sample` |
| $\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$ | `objective.py: ddpm_loss` |
| $x_{t-1} = \tilde\mu_t(x_t, \epsilon_\theta) + \sqrt{\beta_t}\, z$ | `sample.py: ddpm_sample` |
| $\hat x_0$, then $x_s$ | `sample.py: ddim_sample` |
| $(1 + w)\, \epsilon(x, y) - w\, \epsilon(x, \varnothing)$ | `sample.py: predict_eps` |
| $\epsilon_\theta$ | `model.py: ToyDenoiser`, `ImageDenoiser` (shared U-Net in `common/nets.py`) |

## Setup

| | 2-D | Fashion-MNIST |
|---|---|---|
| Data | swiss roll, 8 Gaussians | 60k train images, 28 x 28, labels dropped 10 % of the time |
| Network | MLP, 3 hidden layers of 256, sinusoidal time embedding | U-Net, 6.5 M parameters, 28 -> 14 -> 7, attention at 7 x 7 |
| Schedule | $T = 1000$, linear $\beta$ from $10^{-4}$ to $0.02$ | same |
| Optimiser | Adam, lr 1e-3, batch 512, 20k steps | Adam, lr 2e-4, batch 128, 30k steps |
| Weights used for sampling | EMA, decay 0.999 | same |
| Metrics | modes covered (8 Gaussians, 5000 samples) | FID-lite, class balance, label accuracy (2000 samples) |

## Run

```
bash 06_ddpm/run.sh          # train, then the sampler and guidance sweeps
python 06_ddpm/figures.py    # figures and tables
```

## Checklist

- [ ] 2-D: swiss roll and 8 Gaussians
- [ ] Fashion-MNIST: DDPM with 1000 steps vs DDIM with 10, 50, 100, 1000 steps
- [ ] Classifier-free guidance, w from 0 to 4

## Results
