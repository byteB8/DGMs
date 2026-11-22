"""Building blocks reused across projects."""
from torch import nn


def mlp(in_dim: int, out_dim: int, hidden: int = 512, depth: int = 3, act=nn.ReLU) -> nn.Sequential:
    """depth hidden layers of width hidden, then a linear output layer."""
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)
