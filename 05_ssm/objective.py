"""Exact log-likelihood of a binary sequence by the chain rule:

    log p(x) = sum_t log p(x_t | x_<t),   p(x_t = 1 | x_<t) = sigmoid(logit_t)
"""
import torch.nn.functional as F


def nll_per_sequence(logits, x):
    """-log p(x) in nats, one value per sequence. logits and x: (B, L)."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(1)
