"""Gated PixelCNN for binary 28 x 28 images.

Pixels are ordered row by row (raster order). Output (i, j) may only depend on pixels that come before (i, j).
Two stacks of masked convolutions enforce this without a blind spot:
  vertical stack   at (i, j) sees all pixels in rows above i,
  horizontal stack at (i, j) sees pixels left of j in row i, plus the vertical stack.
"""
import torch
import torch.nn.functional as F
from torch import nn


class MaskedConv2d(nn.Conv2d):
    """Convolution whose kernel is multiplied by a fixed 0/1 mask, so some inputs are never seen."""

    def __init__(self, c_in: int, c_out: int, mask: torch.Tensor, dilation: int = 1):
        kh, kw = mask.shape
        super().__init__(c_in, c_out, (kh, kw), dilation=dilation,
                         padding=(dilation * (kh - 1) // 2, dilation * (kw - 1) // 2))
        self.register_buffer("mask", mask[None, None])

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding, dilation=self.dilation)


def vertical_mask(k: int, mask_center: bool) -> torch.Tensor:
    """k x k kernel that keeps rows above the centre (and the centre row unless mask_center)."""
    mask = torch.ones(k, k)
    mask[k // 2 + 1:] = 0
    if mask_center:
        mask[k // 2] = 0
    return mask


def horizontal_mask(k: int, mask_center: bool) -> torch.Tensor:
    """1 x k kernel that keeps pixels left of the centre (and the centre unless mask_center)."""
    mask = torch.ones(1, k)
    mask[0, k // 2 + 1:] = 0
    if mask_center:
        mask[0, k // 2] = 0
    return mask


class GatedLayer(nn.Module):
    """One gated layer: out = tanh(a) * sigmoid(b) on both stacks, vertical feeds horizontal,
    residual connection on the horizontal stack."""

    def __init__(self, ch: int, k: int = 3, dilation: int = 1):
        super().__init__()
        self.v_conv = MaskedConv2d(ch, 2 * ch, vertical_mask(k, False), dilation)
        self.h_conv = MaskedConv2d(ch, 2 * ch, horizontal_mask(k, False), dilation)
        self.v_to_h = nn.Conv2d(2 * ch, 2 * ch, 1)
        self.h_out = nn.Conv2d(ch, ch, 1)

    @staticmethod
    def gate(x):
        a, b = x.chunk(2, dim=1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(self, v, h):
        v_feat = self.v_conv(v)
        h_feat = self.h_conv(h) + self.v_to_h(v_feat)
        return self.gate(v_feat), h + self.h_out(self.gate(h_feat))


class GatedPixelCNN(nn.Module):
    def __init__(self, ch: int = 64, dilations=(1, 2, 1, 4, 1, 2, 1)):
        super().__init__()
        # first layer masks the centre: output (i, j) must not see input (i, j)
        self.v_in = MaskedConv2d(1, ch, vertical_mask(3, True))
        self.h_in = MaskedConv2d(1, ch, horizontal_mask(3, True))
        self.layers = nn.ModuleList(GatedLayer(ch, 3, d) for d in dilations)
        self.out = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        """x in {0, 1}, shape (B, 1, 28, 28) -> logits of p(x_ij = 1 | pixels before ij), same shape."""
        x = 2 * x - 1
        v, h = self.v_in(x), self.h_in(x)
        for layer in self.layers:
            v, h = layer(v, h)
        return self.out(F.elu(h))
