"""GAN and WGAN losses.

Notation: p is the data distribution, q the generator distribution (x = G(z), z ~ N(0, I)).
The GAN discriminator outputs a logit l, and D(x) = sigmoid(l).
Two identities keep the code stable:
    -log D(x)       = -log sigmoid(l)     = softplus(-l)
    -log(1 - D(x))  = -log sigmoid(-l)    = softplus(l)
The WGAN critic f outputs an unbounded score.
"""
import torch
import torch.nn.functional as F


def d_loss_gan(real_logits, fake_logits):
    """Discriminator: maximise E_p[log D(x)] + E_q[log(1 - D(x))], i.e. minimise its negative."""
    return F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()


def g_loss_minimax(fake_logits):
    """Generator, original game: minimise E_q[log(1 - D(x))].

    d/dl log(1 - sigmoid(l)) = -sigmoid(l), which is close to 0 when D confidently rejects fakes.
    """
    return -F.softplus(fake_logits).mean()


def g_loss_nonsaturating(fake_logits):
    """Generator, non-saturating: minimise -E_q[log D(x)].

    d/dl -log sigmoid(l) = -(1 - sigmoid(l)), which is close to -1 when D confidently rejects fakes.
    """
    return F.softplus(-fake_logits).mean()


def critic_loss_wgan(real_scores, fake_scores):
    """Critic: maximise E_p[f(x)] - E_q[f(x)] (the Wasserstein-1 dual), i.e. minimise its negative."""
    return fake_scores.mean() - real_scores.mean()


def g_loss_wgan(fake_scores):
    """Generator: minimise -E_q[f(x)]."""
    return -fake_scores.mean()


def gradient_penalty(critic, real, fake, y=None):
    """E[(||grad f(x_hat)|| - 1)^2] with x_hat = eps * real + (1 - eps) * fake, eps ~ U(0, 1).

    Pushes the critic towards a unit gradient norm, i.e. towards a 1-Lipschitz function.
    """
    eps = torch.rand(real.size(0), *([1] * (real.dim() - 1)), device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    (grad,) = torch.autograd.grad(critic(x_hat, y).sum(), x_hat, create_graph=True)
    return ((grad.flatten(1).norm(dim=1) - 1) ** 2).mean()
