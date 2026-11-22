"""RealNVP for 2-D data: a stack of affine coupling layers.

Direction convention: forward maps data x to latent z, inverse maps z back to x.
"""
import torch
from torch import nn

from common.nets import mlp


class AffineCoupling(nn.Module):
    """Keep coordinates a (mask = 1) and transform coordinates b (mask = 0) using a and only a:

        z_a = x_a,    z_b = x_b * exp(s(x_a)) + t(x_a)

    The Jacobian is triangular, so log |det dz/dx| = sum_b s(x_a), and the inverse is exact:
        x_b = (z_b - t(z_a)) * exp(-s(z_a)).
    """

    def __init__(self, mask: torch.Tensor, hidden: int = 256):
        super().__init__()
        self.register_buffer("mask", mask)
        self.net = mlp(2, 4, hidden, depth=3)                 # -> (raw scale, shift) for both coordinates
        self.scale = nn.Parameter(torch.ones(1))

    def _scale_shift(self, kept):
        raw, shift = self.net(kept).chunk(2, dim=1)
        free = 1 - self.mask
        return self.scale * torch.tanh(raw) * free, shift * free   # tanh keeps exp(s) bounded

    def forward(self, x):
        s, t = self._scale_shift(x * self.mask)
        z = x * self.mask + (1 - self.mask) * (x * s.exp() + t)
        return z, s.sum(dim=1)

    def inverse(self, z):
        s, t = self._scale_shift(z * self.mask)
        return z * self.mask + (1 - self.mask) * ((z - t) * (-s).exp())


class RealNVP(nn.Module):
    def __init__(self, n_layers: int = 8, hidden: int = 256):
        super().__init__()
        masks = [torch.tensor([1.0, 0.0]) if i % 2 == 0 else torch.tensor([0.0, 1.0]) for i in range(n_layers)]
        self.layers = nn.ModuleList(AffineCoupling(m, hidden) for m in masks)

    def forward(self, x):
        """x -> z and log |det dz/dx|, summed over layers."""
        logdet = torch.zeros(len(x), device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, z, return_all: bool = False):
        """z -> x. With return_all, also return the points after every layer (z first, x last)."""
        points = [z]
        for layer in reversed(self.layers):
            z = layer.inverse(z)
            points.append(z)
        return points if return_all else z
