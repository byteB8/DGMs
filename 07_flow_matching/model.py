"""Velocity networks v_theta(x_t, t, y), t in [0, 1]: an MLP for 2-D points and the shared U-Net for images.

Same architectures as 06_ddpm; only the time range (continuous [0, 1] instead of 0..T-1) and the target change.
y is a class label in 0..n_classes-1, or n_classes for "no label" (used for classifier-free guidance).
"""
import torch
from torch import nn

from common.nets import SinusoidalEmbedding, UNet, mlp

TIME_SCALE = 1000.0  # t in [0, 1] is scaled before the sinusoidal embedding, as for DDPM steps 0..999


class ToyVelocity(nn.Module):
    def __init__(self, hidden: int = 256, t_dim: int = 64):
        super().__init__()
        self.t_embed = SinusoidalEmbedding(t_dim)
        self.net = mlp(2 + t_dim, 2, hidden, depth=3, act=nn.SiLU)

    def forward(self, x, t, y=None):
        return self.net(torch.cat([x, self.t_embed(t * TIME_SCALE)], dim=1))


class ImageVelocity(nn.Module):
    def __init__(self, n_classes: int = 10, ch: int = 64, in_ch: int = 1):
        super().__init__()
        self.null_label = n_classes
        self.label = nn.Embedding(n_classes + 1, 4 * ch) if n_classes else None
        self.unet = UNet(in_ch, ch, cond_dim=4 * ch if n_classes else 0, time_scale=TIME_SCALE)

    def forward(self, x, t, y=None):
        cond = self.label(y) if (self.label is not None and y is not None) else None
        return self.unet(x, t, cond)


def build(data: str, cond: bool, ch: int = 64):
    if data in ("mnist", "fmnist"):
        return ImageVelocity(10 if cond else 0, ch)
    return ToyVelocity()
