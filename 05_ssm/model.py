"""Sequence models for binary images read as 784-step sequences: a diagonal SSM (S4D) and a causal transformer.

Both see the shifted sequence (start token, x_1, ..., x_783) and predict x_t at step t,
so output t only uses x_<t. Both have a parallel mode (training) and a step mode (sampling).
Blocks share one layout:  x = x + Mixer(LN(x));  x = x + MLP(LN(x)),  where Mixer is an SSM or attention.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

START = 2  # token id of the start symbol (pixels are 0 and 1)


# ----------------------------------------------------------------------------------------------- S4D
class S4DLayer(nn.Module):
    """One diagonal state space model per channel (H channels, N/2 complex modes each).

    Continuous time:  s'(t) = A s(t) + B u(t),   y(t) = Re(C s(t)) + D u(t),   A diagonal, B = 1.
    Zero-order hold with step dt:   A_bar = exp(dt A),   B_bar = (A_bar - 1) / A.
    Recurrent mode:   s_k = A_bar s_{k-1} + B_bar u_k,   y_k = 2 Re(C s_k) + D u_k
    Convolution mode: y = K * u + D u,   K_m = 2 Re(sum_n C_n B_bar_n A_bar_n^m)
    (the factor 2 accounts for the complex-conjugate modes that are not stored).
    """

    def __init__(self, H: int, N: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        self.log_dt = nn.Parameter(torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, N // 2)))          # Re(A) = -0.5
        self.A_imag = nn.Parameter(math.pi * torch.arange(N // 2).float().repeat(H, 1))  # Im(A) = pi n
        self.C = nn.Parameter(torch.view_as_real(torch.randn(H, N // 2, dtype=torch.cfloat) * 0.5 ** 0.5))
        self.D = nn.Parameter(torch.randn(H))

    def discrete(self):
        """A_bar, B_bar and C, each (H, N/2) complex."""
        A = -self.log_A_real.exp() + 1j * self.A_imag
        dtA = A * self.log_dt.exp()[:, None]
        A_bar = dtA.exp()
        return A_bar, (A_bar - 1) / A, torch.view_as_complex(self.C)

    def kernel(self, L: int):
        """K_m for m = 0..L-1, shape (H, L)."""
        A = -self.log_A_real.exp() + 1j * self.A_imag
        dtA = A * self.log_dt.exp()[:, None]
        powers = (dtA[..., None] * torch.arange(L, device=dtA.device)).exp()           # A_bar^m, (H, N/2, L)
        CB = torch.view_as_complex(self.C) * (dtA.exp() - 1) / A
        return 2 * torch.einsum("hn,hnl->hl", CB, powers).real

    def forward(self, u):
        """Convolution mode. u: (B, L, H) -> y: (B, L, H). The causal convolution uses FFTs of length 2L."""
        L = u.shape[1]
        u = u.transpose(1, 2)
        y = torch.fft.irfft(torch.fft.rfft(u, n=2 * L) * torch.fft.rfft(self.kernel(L), n=2 * L), n=2 * L)[..., :L]
        return (y + self.D[:, None] * u).transpose(1, 2)

    def step(self, u, state):
        """Recurrent mode for one time step. u: (B, H); state: (B, H, N/2) complex."""
        A_bar, B_bar, C = self.discrete()
        state = A_bar * state + B_bar * u[..., None]
        return 2 * (C * state).sum(-1).real + self.D * u, state

    def initial_state(self, batch: int, device):
        return torch.zeros(batch, *self.A_imag.shape, dtype=torch.cfloat, device=device)


class SSMMixer(nn.Module):
    """S4D layer, GELU, then a gated linear output: GLU(W gelu(y))."""

    def __init__(self, d: int, N: int):
        super().__init__()
        self.ssm = S4DLayer(d, N)
        self.out = nn.Linear(d, 2 * d)

    def forward(self, x):
        return F.glu(self.out(F.gelu(self.ssm(x))), dim=-1)

    def step(self, x, state):
        y, state = self.ssm.step(x, state)
        return F.glu(self.out(F.gelu(y)), dim=-1), state


# --------------------------------------------------------------------------------------- attention
class CausalSelfAttention(nn.Module):
    """softmax(Q K^T / sqrt(d_head) + M) V with M = -inf above the diagonal (no peeking at later steps).
    In step mode, keys and values of earlier steps are kept in a cache instead of being recomputed."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def _heads(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        return [t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2) for t in (q, k, v)]

    @staticmethod
    def _attend(q, k, v, mask=None):
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        return scores.softmax(dim=-1) @ v

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self._heads(x)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        return self.proj(self._attend(q, k, v, mask).transpose(1, 2).reshape(B, T, d))

    def step(self, x, cache):
        """x: (B, d) for one new step; cache: (keys, values) of all earlier steps, or None."""
        B, d = x.shape
        q, k, v = self._heads(x[:, None])
        if cache is not None:
            k, v = torch.cat([cache[0], k], dim=2), torch.cat([cache[1], v], dim=2)
        y = self._attend(q, k, v).transpose(1, 2).reshape(B, d)  # the newest step may see every cached step
        return self.proj(y), (k, v)


# ---------------------------------------------------------------------------------------- the model
class Block(nn.Module):
    def __init__(self, d: int, mixer: nn.Module):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mixer = mixer
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.mixer(self.norm1(x))
        return x + self.mlp(self.norm2(x))

    def step(self, x, state):
        h, state = self.mixer.step(self.norm1(x), state)
        x = x + h
        return x + self.mlp(self.norm2(x)), state


class SequenceModel(nn.Module):
    """kind = "ssm" or "transformer". Outputs one Bernoulli logit per step."""

    def __init__(self, kind: str, d: int = 128, n_layers: int = 4, length: int = 784, N: int = 64, n_heads: int = 4):
        super().__init__()
        self.kind = kind
        self.embed = nn.Embedding(3, d)
        # attention has no notion of order, so it needs position embeddings; the SSM is ordered by construction
        self.pos = nn.Parameter(torch.zeros(length, d)) if kind == "transformer" else None
        mixer = (lambda: SSMMixer(d, N)) if kind == "ssm" else (lambda: CausalSelfAttention(d, n_heads))
        self.blocks = nn.ModuleList(Block(d, mixer()) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, x):
        """Parallel mode. x: (B, 784) pixels in {0, 1} -> logits (B, 784); logit t predicts x_t from x_<t."""
        tokens = torch.cat([torch.full_like(x[:, :1], START), x[:, :-1]], dim=1).long()
        h = self.embed(tokens)
        if self.pos is not None:
            h = h + self.pos
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h)).squeeze(-1)

    def initial_states(self, batch: int, device):
        if self.kind == "ssm":
            return [b.mixer.ssm.initial_state(batch, device) for b in self.blocks]
        return [None] * len(self.blocks)

    def step(self, token, t: int, states):
        """Step mode: token (B,) at position t -> logit (B,) for x_t, and the new states."""
        h = self.embed(token.long())
        if self.pos is not None:
            h = h + self.pos[t]
        new_states = []
        for block, state in zip(self.blocks, states):
            h, state = block.step(h, state)
            new_states.append(state)
        return self.head(self.norm(h)).squeeze(-1), new_states
