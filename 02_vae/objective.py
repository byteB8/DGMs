"""VAE objective: the ELBO and the importance-weighted bound.

Model:   p(z) = N(0, I),  p(x | z) = Bernoulli(sigmoid(decoder(z)))  (x is a binarized image)
Encoder: q(z | x) = N(mu(x), diag(sigma(x)^2)), parameterised by mu and logvar = log sigma^2.

    log p(x) >= ELBO(x) = E_q[log p(x | z)] - KL(q(z | x) || p(z))
    log p(x) -  ELBO(x) = KL(q(z | x) || p(z | x))      (the gap)
"""
import math

import torch
import torch.nn.functional as F


def bernoulli_log_likelihood(logits, x):
    """log p(x | z) = sum_i [x_i log s_i + (1 - x_i) log(1 - s_i)], s = sigmoid(logits). One value per image."""
    return -F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1)


def gaussian_kl(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) = 1/2 sum_j (mu_j^2 + sigma_j^2 - log sigma_j^2 - 1). One value per image."""
    return 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(1)


def negative_elbo(x, logits, mu, logvar, beta: float = 1.0):
    """Loss = -E_q[log p(x | z)] + beta * KL, averaged over the batch (beta = 1 is the true ELBO).

    E_q[log p(x | z)] is estimated with the single reparameterised sample that produced logits.
    Returns (loss, reconstruction term, KL term), the last two as batch means.
    """
    rec = -bernoulli_log_likelihood(logits, x)
    kl = gaussian_kl(mu, logvar)
    return (rec + beta * kl).mean(), rec.mean(), kl.mean()


def gaussian_log_density(z, mu, logvar):
    """log N(z; mu, diag(exp(logvar))), summed over the last dimension."""
    return -0.5 * (math.log(2 * math.pi) + logvar + (z - mu).pow(2) / logvar.exp()).sum(-1)


@torch.no_grad()
def importance_log_likelihood(vae, x, k: int = 500):
    """Importance-sampling estimate of log p(x) with q(z | x) as proposal (the IWAE bound with k samples):

        log p(x) ~= log (1/k) sum_j p(x | z_j) p(z_j) / q(z_j | x),   z_j ~ q(z | x)

    It is a lower bound that tightens as k grows; k = 1 gives the ELBO. One value per image.
    """
    mu, logvar = vae.encode(x)
    z = mu + (0.5 * logvar).exp() * torch.randn(k, *mu.shape, device=x.device)    # (k, B, d)
    logits = vae.decode(z.flatten(0, 1)).view(k, *x.shape)
    log_px_z = bernoulli_log_likelihood(logits.flatten(0, 1), x.expand(k, *x.shape).flatten(0, 1)).view(k, -1)
    log_pz = gaussian_log_density(z, torch.zeros_like(z), torch.zeros_like(z))
    log_qz = gaussian_log_density(z, mu, logvar)
    return torch.logsumexp(log_px_z + log_pz - log_qz, dim=0) - math.log(k)
