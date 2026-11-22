"""Generators and critics.

All modules share one signature, so the training loop does not care which pair it gets:
    G(z, y=None) -> samples        D(x, y=None) -> one score (logit) per sample
y is a class label, used only by the conditional image models.
"""
import torch
from torch import nn

from common.nets import mlp


class ToyGenerator(nn.Module):
    """z in R^z_dim -> point in R^2."""

    def __init__(self, z_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.net = mlp(z_dim, 2, hidden)

    def forward(self, z, y=None):
        return self.net(z)


class ToyCritic(nn.Module):
    """Point in R^2 -> score."""

    def __init__(self, hidden: int = 512):
        super().__init__()
        self.net = mlp(2, 1, hidden)

    def forward(self, x, y=None):
        return self.net(x).squeeze(1)


class ConvGenerator(nn.Module):
    """DCGAN-style generator: z (+ optional label) -> 1 x 28 x 28 image in [-1, 1]."""

    def __init__(self, z_dim: int = 64, n_classes: int = 0, ch: int = 64):
        super().__init__()
        self.ch = ch
        self.embed = nn.Embedding(n_classes, z_dim) if n_classes else None
        in_dim = 2 * z_dim if n_classes else z_dim
        self.fc = nn.Sequential(nn.Linear(in_dim, 2 * ch * 7 * 7), nn.BatchNorm1d(2 * ch * 7 * 7), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(),  # 7 -> 14
            nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Tanh(),                           # 14 -> 28
        )

    def forward(self, z, y=None):
        if self.embed is not None:
            z = torch.cat([z, self.embed(y)], dim=1)
        return self.net(self.fc(z).view(-1, 2 * self.ch, 7, 7))


class ConvCritic(nn.Module):
    """1 x 28 x 28 image (+ optional label) -> score.

    No batch norm, so the same network works for the GAN and the WGAN-GP losses
    (the gradient penalty is defined per sample).
    Labels enter by projection: score(x, y) = w . h(x) + e_y . h(x).
    """

    def __init__(self, n_classes: int = 0, ch: int = 64, feat: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, ch, 4, 2, 1), nn.LeakyReLU(0.2),           # 28 -> 14
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.LeakyReLU(0.2),      # 14 -> 7
            nn.Flatten(), nn.Linear(2 * ch * 7 * 7, feat), nn.LeakyReLU(0.2),
        )
        self.out = nn.Linear(feat, 1)
        self.embed = nn.Embedding(n_classes, feat) if n_classes else None

    def forward(self, x, y=None):
        h = self.body(x)
        score = self.out(h).squeeze(1)
        if self.embed is not None:
            score = score + (self.embed(y) * h).sum(dim=1)
        return score


def build(data: str, z_dim: int, cond: bool):
    """Return (G, D) for a dataset name."""
    if data in ("mnist", "fmnist"):
        n_classes = 10 if cond else 0
        return ConvGenerator(z_dim, n_classes), ConvCritic(n_classes)
    return ToyGenerator(z_dim), ToyCritic()
