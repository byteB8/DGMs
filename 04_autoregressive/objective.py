"""Exact log-likelihood of an autoregressive model of binary images.

Chain rule over pixels in raster order:

    log p(x) = sum_i log p(x_i | x_1, ..., x_{i-1})

Each factor is a Bernoulli whose logit the network outputs at pixel i, so one forward pass
gives all 784 conditionals at once (training is parallel; sampling is not).
"""
import torch.nn.functional as F


def nll_per_image(logits, x):
    """-log p(x) in nats, one value per image."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)
