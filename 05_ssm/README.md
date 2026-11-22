# 05 State space model

The same task as 04: binarized MNIST, $\log p(x) = \sum_t \log p(x_t \mid x_{<t})$, but each image is read as a
sequence of 784 pixels. Two sequence models of similar size predict the next pixel:
a diagonal state space model (S4D) and a causal transformer.

## Math

**Continuous-time SSM**, one per channel, with input $u$, state $s \in \mathbb C^{N/2}$, output $y$:

$$s'(t) = A\, s(t) + B\, u(t), \qquad y(t) = \mathrm{Re}(C\, s(t)) + D\, u(t)$$

**Discretise** with step $\Delta$ (zero-order hold):

$$\bar A = e^{\Delta A}, \qquad \bar B = (\bar A - I) A^{-1} B, \qquad s_k = \bar A s_{k-1} + \bar B u_k, \qquad y_k = \mathrm{Re}(C s_k) + D u_k$$

**Recurrence = convolution.** Unrolling the recurrence from $s_{-1} = 0$:

$$y_k = \sum_{j \le k} K_{k-j}\, u_j + D u_k, \qquad K_m = \mathrm{Re}(C \bar A^m \bar B)$$

Example: $y_2 = \mathrm{Re}(C\bar B)\, u_2 + \mathrm{Re}(C \bar A \bar B)\, u_1 + \mathrm{Re}(C \bar A^2 \bar B)\, u_0 + D u_2$.

With a diagonal $A$, $K_m = \mathrm{Re}\sum_n C_n \bar B_n \bar A_n^m$ is cheap to build.
Training uses the convolution (parallel over all 784 steps, FFT in $O(L \log L)$);
sampling uses the recurrence (constant cost per step).

**Memory.** $\mathrm{Re}(A_n) < 0$ gives $|\bar A_n| < 1$, so old inputs fade at rate $e^{\Delta\, \mathrm{Re}(A_n)}$ per step;
$\mathrm{Im}(A_n) = \pi n$ gives each mode its own oscillation frequency.

**Causal attention**, for comparison: $\mathrm{softmax}(Q K^\top / \sqrt{d_h} + M)\, V$ with $M_{ts} = -\infty$ for $s > t$.

| | S4D | Causal transformer |
|---|---|---|
| Training, one sequence of length $L$ | $O(L \log L)$ | $O(L^2)$ time and memory |
| Sampling, step $t$ | $O(1)$ (update the state) | $O(t)$ (attend to the KV cache) |
| Knows the order of positions | by construction | needs position embeddings |

## Code map

| Math | Code |
|---|---|
| $\bar A = e^{\Delta A}$, $\bar B = (\bar A - 1)/A$ | `model.py: S4DLayer.discrete` |
| $K_m = 2\,\mathrm{Re}\sum_n C_n \bar B_n \bar A_n^m$ | `model.py: S4DLayer.kernel` |
| $y = K * u + D u$ (FFT) | `model.py: S4DLayer.forward` |
| $s_k = \bar A s_{k-1} + \bar B u_k$ | `model.py: S4DLayer.step` |
| $\mathrm{softmax}(QK^\top/\sqrt{d_h} + M)V$, KV cache | `model.py: CausalSelfAttention` |
| $-\sum_t \log p(x_t \mid x_{<t})$ | `objective.py: nll_per_sequence` |
| parallel mode = step mode | `sample.py: step_matches_parallel` |

The factor 2 in the kernel accounts for the complex-conjugate modes, which are not stored.

## Setup

| | |
|---|---|
| Data | MNIST thresholded at 0.5 (same as 02 and 04), read in raster order |
| Models | 4 blocks, width 128: `x + Mixer(LN(x))`, then `x + MLP(LN(x))` |
| Mixer | S4D with 32 complex modes per channel, or 4-head causal attention + learned position embeddings |
| Optimiser | AdamW, lr 1e-3 (one-cycle), batch 64, 10 epochs, gradient clipping at 1 |
| Metrics | exact test -log p(x) in nats, bits per pixel, time per training step, time to draw 64 samples |

## Run

```
bash 05_ssm/run.sh           # train and sample both models
python 05_ssm/figures.py     # figures and table
```

## Checklist

- [x] Recurrent and convolution modes give the same outputs
- [x] Test log-likelihood: S4D vs transformer (and the PixelCNN from 04)
- [x] Training and sampling cost

## Results

| Model | Parameters | Test -log p(x) | Bits per pixel | Train step (batch 64) | 64 samples | Max gap, parallel vs step mode |
|---|---|---|---|---|---|---|
| S4D | 0.73 M | 55.17 nats | 0.1015 | 109 ms | 1.4 s | 3.2e-4 |
| causal transformer | 0.89 M | 56.14 nats | 0.1033 | 360 ms | 2.3 s | 2.7e-5 |
| Gated PixelCNN (04) | 0.84 M | 57.66 nats | 0.1061 | - | 7.1 s | - |

Timings were taken on a shared GPU, so compare them within this table only.

- **Both sequence models beat the PixelCNN.** S4D reaches 55.2 nats and the transformer 56.1, against 57.7 for the
  PixelCNN in 04 (same data, similar size). Unlike the PixelCNN's finite window, both can use every earlier pixel.
- **S4D learns faster per epoch and per second.** It is ahead at every epoch, and a training step costs about a third
  of the transformer's: $O(L \log L)$ convolution against $O(L^2)$ attention over 784 steps.
- **One layer, two modes.** The FFT convolution and the recurrence give the same logits up to $3 \times 10^{-4}$
  (float32 rounding grows as the trained kernels get longer memory).
- **Sampling cost per step: flat vs growing.** S4D spends about 2.7 ms on every step; the transformer's step grows
  from 2.6 to 3.6 ms as its KV cache grows. At 784 steps and this model size the gap is small; it widens with length.
