"""2-D toy distributions. Every sampler returns a float tensor of shape (n, 2)."""
import math

import torch


def eight_gaussian_centers(radius: float = 2.0) -> torch.Tensor:
    angles = torch.arange(8) * (2 * math.pi / 8)
    return radius * torch.stack([angles.cos(), angles.sin()], dim=1)


def eight_gaussians(n: int, radius: float = 2.0, std: float = 0.05) -> torch.Tensor:
    """Mixture of 8 equal Gaussians on a circle."""
    centers = eight_gaussian_centers(radius)
    idx = torch.randint(0, 8, (n,))
    return centers[idx] + std * torch.randn(n, 2)


def eight_gaussians_log_prob(x: torch.Tensor, radius: float = 2.0, std: float = 0.05) -> torch.Tensor:
    """Exact log density of the 8-Gaussian mixture: log (1/8) sum_k N(x; c_k, std^2 I)."""
    centers = eight_gaussian_centers(radius).to(x)
    sq_dist = ((x[:, None, :] - centers[None]) ** 2).sum(-1)                  # (n, 8)
    log_comp = -sq_dist / (2 * std ** 2) - math.log(2 * math.pi * std ** 2)
    return torch.logsumexp(log_comp, dim=1) - math.log(8)


def swiss_roll(n: int, noise: float = 0.08) -> torch.Tensor:
    """Spiral x = t (cos t, sin t), t in [1.5 pi, 4.5 pi], scaled to about [-2.5, 2.5]."""
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n))
    x = torch.stack([t * t.cos(), t * t.sin()], dim=1) / 6
    return x + noise * torch.randn(n, 2)


def two_moons(n: int, noise: float = 0.05) -> torch.Tensor:
    """Two interleaved half circles, centred at the origin."""
    n_top = n // 2
    t_top = math.pi * torch.rand(n_top)
    t_bot = math.pi * torch.rand(n - n_top)
    top = torch.stack([t_top.cos(), t_top.sin()], dim=1)
    bot = torch.stack([1 - t_bot.cos(), 0.5 - t_bot.sin()], dim=1)
    x = torch.cat([top, bot])[torch.randperm(n)] - torch.tensor([0.5, 0.25])
    return x + noise * torch.randn(n, 2)


SAMPLERS = {"8gauss": eight_gaussians, "swissroll": swiss_roll, "moons": two_moons}


def sample(name: str, n: int, device="cpu") -> torch.Tensor:
    return SAMPLERS[name](n).to(device)
