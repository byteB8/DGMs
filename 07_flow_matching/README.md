# 07 Flow matching

Learn a velocity field that carries Gaussian noise to data along (nearly) straight paths,
then generate by solving an ODE with a handful of steps. Same U-Net and data as 06.

## Math

**Straight paths.** Pair noise $x_0 \sim \mathcal N(0, I)$ with data $x_1$ and interpolate

$$x_t = (1 - t)\, x_0 + t\, x_1, \qquad \frac{d x_t}{dt} = x_1 - x_0, \qquad t \in [0, 1]$$

**Marginal velocity.** Many pairs pass through the same point $x$ at time $t$. Their average direction

$$v^*(x, t) = \mathbb E[\, x_1 - x_0 \mid x_t = x \,]$$

moves the whole distribution $p_t$ along the path (it satisfies the continuity equation
$\partial_t p_t + \nabla \cdot (p_t v^*) = 0$). Following $dx/dt = v^*(x, t)$ from noise at $t = 0$ ends at the data at $t = 1$.

**Loss.** $v^*$ is unknown, but regressing on the per-pair velocity gives the same minimiser, because
$\mathbb E \lVert v - (x_1 - x_0) \rVert^2 = \mathbb E \lVert v - v^* \rVert^2 + \text{const}$:

$$L = \mathbb E_{t, x_0, x_1} \left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2$$

**Sampling.** Euler steps $x \leftarrow x + v_\theta(x, t)\, \Delta t$ with $\Delta t = 1/K$.

**Why paths bend, and reflow.** Example in 1-D with a single data point $x_1 = 3$: $v^*(x, t) = (3 - x)/(1 - t)$,
every path is a straight line to 3, and one Euler step from $t = 0$ lands exactly on 3.
With many data points, independent pairs cross, $v^*$ averages the crossing directions, and the paths curve,
so one step is no longer enough. Reflow retrains on pairs $(x_0, \mathrm{ODE}(x_0))$ produced by the first model.
These pairs do not cross, so the new paths are straighter.

**Straightness**, as measured here: the mean distance between the 1-step and the 100-step endpoints from the same noise.
It is 0 for straight paths travelled at constant speed.

**Classifier-free guidance** works as in 06, on the velocity: $\tilde v = (1 + w)\, v_\theta(x, t, y) - w\, v_\theta(x, t, \varnothing)$.

## Code map

| Math | Code |
|---|---|
| $x_t = (1 - t)\, x_0 + t\, x_1$ | `objective.py: interpolate` |
| $\lVert v_\theta(x_t, t) - (x_1 - x_0) \rVert^2$ | `objective.py: flow_matching_loss` |
| $x \leftarrow x + v_\theta(x, t) / K$ | `sample.py: euler_sample` |
| $(1 + w)\, v(x, y) - w\, v(x, \varnothing)$ | `sample.py: velocity` |
| 1-step vs 100-step endpoint distance | `sample.py: straightness` |
| pairs $(x_0, \mathrm{ODE}(x_0))$ for reflow | `train.py: reflow_pairs` |
| $v_\theta$ | `model.py: ToyVelocity`, `ImageVelocity` (shared U-Net in `common/nets.py`) |

## Setup

As in 06: same networks (time $t \in [0, 1]$ scaled by 1000 before the sinusoidal embedding), optimiser, EMA,
label dropout and metrics.

| | 2-D | Fashion-MNIST |
|---|---|---|
| Data | 8 Gaussians, swiss roll | 60k train images, labels dropped 10 % of the time |
| Training | 20k steps, batch 512, lr 1e-3 | 30k steps, batch 128, lr 2e-4 |
| Reflow | 8 Gaussians: 100k pairs from the first model (100 Euler steps), then 20k steps | - |
| Sampling | Euler, 1 to 100 steps | Euler, 1 to 100 steps |

## Run

```
bash 07_flow_matching/run.sh          # train, then the step and guidance sweeps
python 07_flow_matching/figures.py    # figures and tables
```

## Checklist

- [ ] 2-D: 8 Gaussians and swiss roll, quality vs number of steps
- [ ] Reflow on 8 Gaussians: straighter paths, fewer steps
- [ ] Fashion-MNIST: quality vs number of steps, against DDIM from 06
- [ ] Classifier-free guidance

## Results
