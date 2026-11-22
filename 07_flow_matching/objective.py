"""Flow matching with straight paths (rectified flow).

Pair noise x0 ~ N(0, I) with data x1 and move along the straight line
    x_t = (1 - t) x0 + t x1,    t in [0, 1],    whose velocity is  dx_t/dt = x1 - x0.
Train a network v_theta(x_t, t) to predict that velocity:
    L = E_{t, x0, x1} || v_theta(x_t, t) - (x1 - x0) ||^2
The minimiser is v*(x, t) = E[x1 - x0 | x_t = x], and the ODE dx/dt = v*(x, t) started from x0 ~ N(0, I)
ends at the data distribution at t = 1.
"""
import torch
import torch.nn.functional as F


def interpolate(x0, x1, t):
    """x_t = (1 - t) x0 + t x1, with t of shape (B,)."""
    t = t.view(-1, *([1] * (x0.dim() - 1)))
    return (1 - t) * x0 + t * x1


def flow_matching_loss(model, x1, y=None, x0=None):
    """|| v_theta(x_t, t, y) - (x1 - x0) ||^2 with t ~ U(0, 1).

    x0 defaults to fresh Gaussian noise (independent pairs). Passing x0 lets reflow train on fixed pairs.
    """
    if x0 is None:
        x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), device=x1.device)
    return F.mse_loss(model(interpolate(x0, x1, t), t, y), x1 - x0)
