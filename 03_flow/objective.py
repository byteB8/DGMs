"""Exact log-likelihood of a normalizing flow by change of variables.

If z = f(x) is invertible and z ~ N(0, I), then

    log q(x) = log N(f(x); 0, I) + log |det df/dx|

Training maximises the average log q(x) over data, which is the same as minimising
KL(p || q) = E_p[log p(x)] - E_p[log q(x)], since E_p[log p(x)] does not depend on the model.
"""
import math

import torch


def standard_normal_log_prob(z):
    """log N(z; 0, I), summed over coordinates."""
    return -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=1)


def log_prob(flow, x):
    """log q(x) for each row of x."""
    z, logdet = flow(x)
    return standard_normal_log_prob(z) + logdet


def nll(flow, x):
    """Mean negative log-likelihood in nats."""
    return -log_prob(flow, x).mean()
