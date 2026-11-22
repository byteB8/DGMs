"""Tiny CLIP: a CNN image encoder and a transformer text encoder that map into one shared space."""
import math

import torch
import torch.nn.functional as F
from torch import nn

from common.nets import TextEncoder
from common.shapes import VOCAB


class ImageEncoder(nn.Module):
    """3 x 32 x 32 image -> vector."""

    def __init__(self, d: int = 128, ch: int = 64):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
                                 nn.Conv2d(c_out, c_out, 3, padding=1), nn.GroupNorm(8, c_out), nn.GELU())

        self.net = nn.Sequential(block(3, ch), nn.MaxPool2d(2),            # 32 -> 16
                                 block(ch, 2 * ch), nn.MaxPool2d(2),       # 16 -> 8
                                 block(2 * ch, 4 * ch), nn.MaxPool2d(2),   # 8 -> 4
                                 nn.Flatten(), nn.Linear(4 * ch * 16, d))

    def forward(self, x):
        return self.net(x)


class CLIP(nn.Module):
    def __init__(self, d: int = 128, embed_dim: int = 128):
        super().__init__()
        self.image_encoder = ImageEncoder(d)
        self.text_encoder = TextEncoder(len(VOCAB), d)
        self.image_proj = nn.Linear(d, embed_dim)
        self.text_proj = nn.Linear(d, embed_dim)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))   # logits = exp(log_scale) * cosine

    def embed_images(self, x):
        """Unit-length image embeddings."""
        return F.normalize(self.image_proj(self.image_encoder(x)), dim=-1)

    def embed_texts(self, tokens):
        """Unit-length text embeddings."""
        return F.normalize(self.text_proj(self.text_encoder(tokens)), dim=-1)
