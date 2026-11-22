"""DDPM: the fixed forward noising process and the noise-prediction loss.

Forward process, t = 1..T:   q(x_t | x_{t-1}) = N(sqrt(1 - beta_t) x_{t-1}, beta_t I)
Closed form:                 q(x_t | x_0) = N(sqrt(abar_t) x_0, (1 - abar_t) I),   abar_t = prod_{s <= t} (1 - beta_s)
so                           x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps,   eps ~ N(0, I)
Simplified ELBO:             L = E_{t, x_0, eps} || eps - eps_theta(x_t, t) ||^2

Code index i = 0..T-1 stands for t = i + 1.
"""
import torch
import torch.nn.functional as F


class NoiseSchedule:
    """Linear beta schedule and the quantities derived from it."""

    def __init__(self, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, device="cpu"):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)


def expand(v, x):
    """Reshape a (B,) tensor so it broadcasts against x."""
    return v.view(-1, *([1] * (x.dim() - 1)))


def q_sample(sched: NoiseSchedule, x0, i, eps):
    """x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
    ab = expand(sched.alpha_bars[i], x0)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps


def ddpm_loss(model, sched: NoiseSchedule, x0, y=None):
    """|| eps - eps_theta(x_t, t, y) ||^2 with t uniform over 1..T (a fresh t and eps per example)."""
    i = torch.randint(0, sched.T, (len(x0),), device=x0.device)
    eps = torch.randn_like(x0)
    return F.mse_loss(model(q_sample(sched, x0, i, eps), i, y), eps)
