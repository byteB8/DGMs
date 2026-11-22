"""VQ-VAE loss and the prior's likelihood.

VQ-VAE, with sg = stop-gradient and e the nearest code to the encoder output z_e:
    L = ||x - dec(z_q)||^2  +  ||sg(z_e) - e||^2  +  beta ||z_e - sg(e)||^2
        reconstruction       codebook: move e to z_e   commitment: keep z_e near its code
Prior over the 49 codes c of an image with class y (chain rule, as in 04):
    log p(c | y) = sum_t log p(c_t | y, c_<t)
"""
import torch
import torch.nn.functional as F


def vqvae_loss(x, x_hat, z_e, e, beta: float = 0.25):
    """Returns (total, reconstruction, codebook, commitment), each a batch mean."""
    rec = F.mse_loss(x_hat, x)
    codebook = F.mse_loss(e, z_e.detach())
    commit = F.mse_loss(z_e, e.detach())
    return rec + codebook + beta * commit, rec, codebook, commit


def prior_nll(prior, codes, y):
    """Mean -log p(c_t | y, c_<t) in nats per code. codes: (B, 49)."""
    logits = prior(codes[:, :-1], y)                    # input [y, c_1..c_48] -> predictions for c_1..c_49
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), codes.reshape(-1))


def perplexity(idx, n_codes: int) -> float:
    """exp(entropy) of code usage: the effective number of codes in use (n_codes if all are used equally)."""
    p = torch.bincount(idx.flatten(), minlength=n_codes).float()
    p = p / p.sum()
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()).item()
