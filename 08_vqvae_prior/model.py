"""VQ-VAE for 28 x 28 images (7 x 7 grid of discrete codes) and a causal transformer prior over the codes."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from common.nets import TransformerBlock


class ResidualStack(nn.Module):
    def __init__(self, ch: int, n: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(nn.Sequential(nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
                                                  nn.Conv2d(ch, ch, 1)) for _ in range(n))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return F.relu(x)


class VectorQuantizer(nn.Module):
    """Replace each D-dim vector z_e by its nearest codebook entry e_k (k = argmin ||z_e - e_k||^2).

    The forward value is e_k; the backward pass copies the gradient straight through to z_e:
        z_q = z_e + sg(e_k - z_e)
    """

    def __init__(self, n_codes: int = 128, dim: int = 64):
        super().__init__()
        self.codebook = nn.Embedding(n_codes, dim)
        self.codebook.weight.data.uniform_(-1 / n_codes, 1 / n_codes)

    def indices(self, z_e):
        """z_e: (B, D, H, W) -> code indices (B, H, W)."""
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, z_e.shape[1])
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.weight.t() + \
            self.codebook.weight.pow(2).sum(1)[None]
        return dist.argmin(1).view(z_e.shape[0], *z_e.shape[2:])

    def lookup(self, idx):
        """Code indices (B, H, W) -> code vectors (B, D, H, W)."""
        return self.codebook(idx).permute(0, 3, 1, 2)

    def forward(self, z_e):
        idx = self.indices(z_e)
        e = self.lookup(idx)
        z_q = z_e + (e - z_e).detach()   # straight-through estimator
        return z_q, e, idx


class VQVAE(nn.Module):
    def __init__(self, n_codes: int = 128, dim: int = 64, ch: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch // 2, 4, 2, 1), nn.ReLU(),        # 28 -> 14
            nn.Conv2d(ch // 2, ch, 4, 2, 1), nn.ReLU(),       # 14 -> 7
            nn.Conv2d(ch, ch, 3, padding=1), ResidualStack(ch),
            nn.Conv2d(ch, dim, 1),
        )
        self.quantizer = VectorQuantizer(n_codes, dim)
        self.decoder = nn.Sequential(
            nn.Conv2d(dim, ch, 3, padding=1), ResidualStack(ch),
            nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1), nn.ReLU(),   # 7 -> 14
            nn.ConvTranspose2d(ch // 2, 1, 4, 2, 1),               # 14 -> 28
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, e, idx = self.quantizer(z_e)
        return self.decoder(z_q), z_e, e, idx

    @torch.no_grad()
    def encode(self, x):
        return self.quantizer.indices(self.encoder(x))

    @torch.no_grad()
    def decode(self, idx):
        """Codes (B, 7, 7), or (B, 49) in raster order -> images."""
        if idx.dim() == 2:
            side = int(idx.shape[1] ** 0.5)
            idx = idx.view(len(idx), side, side)
        return self.decoder(self.quantizer.lookup(idx))


class PriorTransformer(nn.Module):
    """Causal transformer over the 49 codes of an image in raster order, conditioned on the class.

    Input sequence: [class token, c_1, ..., c_48]; output t gives p(c_{t+1} | class, c_1..c_t).
    """

    def __init__(self, n_codes: int = 128, n_classes: int = 10, length: int = 49, d: int = 256,
                 n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.length = length
        self.code_embed = nn.Embedding(n_codes, d)
        self.class_embed = nn.Embedding(n_classes, d)
        self.pos = nn.Parameter(torch.zeros(length, d))
        self.blocks = nn.ModuleList(TransformerBlock(d, n_heads, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_codes)

    def forward(self, codes, y):
        """codes: (B, L) with L <= 49 (the codes seen so far, may be empty), y: (B,) -> logits (B, L + 1, n_codes)
        for the next code at every position. Only the first 49 outputs are used."""
        h = self.class_embed(y)[:, None]
        if codes.shape[1]:
            h = torch.cat([h, self.code_embed(codes)], dim=1)
        h = h[:, :self.length] + self.pos[:min(h.shape[1], self.length)]
        for block in self.blocks:
            h = block(h, causal=True)
        return self.head(self.norm(h))


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def bits(n_codes: int, length: int = 49) -> float:
    """Size of one image as codes: length * log2(n_codes) bits."""
    return length * math.log2(n_codes)
