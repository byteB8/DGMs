"""Building blocks reused across projects."""
import copy
import math

import torch
import torch.nn.functional as F
from torch import nn


def mlp(in_dim: int, out_dim: int, hidden: int = 512, depth: int = 3, act=nn.ReLU) -> nn.Sequential:
    """depth hidden layers of width hidden, then a linear output layer."""
    layers, d = [], in_dim
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), act()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SinusoidalEmbedding(nn.Module):
    """Embed a scalar (e.g. a diffusion time) as [sin(t w_k), cos(t w_k)] with geometric frequencies w_k."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        half = dim // 2
        self.register_buffer("freqs", torch.exp(-math.log(max_period) * torch.arange(half) / half))

    def forward(self, t):
        args = t.float()[:, None] * self.freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class EMA:
    """Exponential moving average of model weights: w_ema <- decay * w_ema + (1 - decay) * w."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p, 1 - self.decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)


# ------------------------------------------------------------------------------------------- U-Net
class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, add the (time + condition) embedding, GroupNorm -> SiLU -> dropout -> conv, skip."""

    def __init__(self, c_in: int, c_out: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out)
        self.norm2, self.conv2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class SelfAttention2d(nn.Module):
    """Single-head self-attention over the H * W positions of a feature map, with a residual connection."""

    def __init__(self, ch: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).flatten(2).chunk(3, dim=1)            # (B, C, HW) each
        attn = (q.transpose(1, 2) @ k / math.sqrt(C)).softmax(dim=-1)        # (B, HW, HW)
        return x + self.proj((v @ attn.transpose(1, 2)).view(B, C, H, W))


class UNet(nn.Module):
    """Small U-Net for 28 x 28 or 32 x 32 images: predicts a tensor shaped like its input.

    t is a (B,) tensor of times (any scale; it goes through a sinusoidal embedding).
    cond is an optional (B, cond_dim) vector (class or text embedding), added to the time embedding.
    Resolutions for 28 x 28 with mults (1, 2, 2): 28 -> 14 -> 7, self-attention at 7 x 7.
    """

    def __init__(self, in_ch: int = 1, ch: int = 64, mults=(1, 2, 2), n_res: int = 2, attn_levels=(2,),
                 cond_dim: int = 0, dropout: float = 0.1, time_scale: float = 1.0):
        super().__init__()
        emb_dim = 4 * ch
        self.time_scale = time_scale
        self.time_mlp = nn.Sequential(SinusoidalEmbedding(ch), nn.Linear(ch, emb_dim), nn.SiLU(),
                                      nn.Linear(emb_dim, emb_dim))
        self.cond_proj = nn.Linear(cond_dim, emb_dim) if cond_dim else None
        self.inc = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down, chans, c = nn.ModuleList(), [ch], ch
        for level, mult in enumerate(mults):
            for _ in range(n_res):
                self.down.append(nn.ModuleList([ResBlock(c, ch * mult, emb_dim, dropout),
                                                SelfAttention2d(ch * mult) if level in attn_levels else nn.Identity()]))
                c = ch * mult
                chans.append(c)
            if level < len(mults) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))
                chans.append(c)

        self.mid = nn.ModuleList([ResBlock(c, c, emb_dim, dropout), SelfAttention2d(c), ResBlock(c, c, emb_dim, dropout)])

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(mults))):
            for _ in range(n_res + 1):
                self.up.append(nn.ModuleList([ResBlock(c + chans.pop(), ch * mult, emb_dim, dropout),
                                              SelfAttention2d(ch * mult) if level in attn_levels else nn.Identity()]))
                c = ch * mult
            if level > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)))

        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))

    def forward(self, x, t, cond=None):
        emb = self.time_mlp(t * self.time_scale)
        if self.cond_proj is not None and cond is not None:
            emb = emb + self.cond_proj(cond)
        h = self.inc(x)
        skips = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](h, emb))
            else:
                h = layer(h)
            skips.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, skips.pop()], dim=1), emb))
            else:
                h = layer(h)
        return self.out(h)
