"""VAE with MLP encoder and decoder for 28 x 28 binarized images."""
import torch
from torch import nn

from common.nets import mlp


class VAE(nn.Module):
    def __init__(self, z_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = mlp(784, 2 * z_dim, hidden, depth=2)   # -> (mu, logvar)
        self.decoder = mlp(z_dim, 784, hidden, depth=2)       # -> Bernoulli logits

    def encode(self, x):
        """q(z | x): mean and log variance, each (B, z_dim)."""
        mu, logvar = self.encoder(x.flatten(1)).chunk(2, dim=1)
        return mu, logvar

    def decode(self, z):
        """p(x | z): pixel logits shaped like the images."""
        return self.decoder(z).view(-1, 1, 28, 28)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I): a sample of q(z | x) that gradients can flow through."""
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar
