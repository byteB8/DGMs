"""Text-conditioned flow matching (as in 07, with a caption embedding c instead of a class label).

    x_t = (1 - t) x0 + t x1,   x0 ~ N(0, I),   x1 = image with caption embedding c
    L = E || v_theta(x_t, t, c) - (x1 - x0) ||^2,   with c replaced by "no caption" 10 % of the time
Classifier-free guidance at sampling time:  v = (1 + w) v(x, t, c) - w v(x, t, no caption)
"""
import torch
import torch.nn.functional as F


def flow_matching_loss(model, x1, cond, p_uncond: float = 0.1):
    x0 = torch.randn_like(x1)
    t = torch.rand(len(x1), device=x1.device)
    tt = t.view(-1, 1, 1, 1)
    drop = torch.rand(len(x1), device=x1.device) < p_uncond
    return F.mse_loss(model((1 - tt) * x0 + tt * x1, t, cond, drop), x1 - x0)
