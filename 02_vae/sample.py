"""Evaluate a trained VAE and draw its pictures.

    python 02_vae/sample.py --run 02_vae/runs/<name>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.data import binarize, load_images  # noqa: E402
from common.utils import get_device  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import VAE  # noqa: E402
from objective import bernoulli_log_likelihood, gaussian_kl, importance_log_likelihood  # noqa: E402


@torch.no_grad()
def evaluate(vae, x, k: int = 500, batch: int = 500) -> dict:
    """Test-set averages in nats per image.

    rec = -E_q[log p(x|z)], kl = KL(q(z|x) || p(z)), neg_elbo = rec + kl.
    With k > 0 also nll = -log p(x) (importance sampling with k samples) and gap = neg_elbo - nll.
    """
    was_training = vae.training
    vae.eval()
    rec = kl = nll = 0.0
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        mu, logvar = vae.encode(xb)
        logits = vae.decode(vae.reparameterize(mu, logvar))
        rec -= bernoulli_log_likelihood(logits, xb).sum().item()
        kl += gaussian_kl(mu, logvar).sum().item()
        if k:
            nll -= importance_log_likelihood(vae, xb, k).sum().item()
    vae.train(was_training)
    n = len(x)
    out = {"rec": rec / n, "kl": kl / n, "neg_elbo": (rec + kl) / n}
    if k:
        out["nll"] = nll / n
        out["gap"] = out["neg_elbo"] - out["nll"]
    return out


@torch.no_grad()
def decode_mean(vae, z):
    """Pixel probabilities sigmoid(decoder(z)), mapped to [-1, 1] for plotting."""
    return torch.sigmoid(vae.decode(z)) * 2 - 1


@torch.no_grad()
def latent_grid(vae, n: int = 20, span: float = 2.5):
    """Decode an n x n grid of z values in [-span, span]^2 (2-D latents only). Row 0 is the top (largest z2)."""
    ticks = torch.linspace(-span, span, n)
    z2, z1 = torch.meshgrid(ticks.flip(0), ticks, indexing="ij")
    z = torch.stack([z1, z2], dim=-1).view(-1, 2).to(next(vae.parameters()).device)
    return decode_mean(vae, z)


@torch.no_grad()
def interpolate(vae, x_a, x_b, steps: int = 10):
    """Decode points on the straight line between the encoder means of two images."""
    mu_a, _ = vae.encode(x_a[None])
    mu_b, _ = vae.encode(x_b[None])
    t = torch.linspace(0, 1, steps, device=mu_a.device)[:, None]
    return decode_mean(vae, (1 - t) * mu_a + t * mu_b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()

    run = Path(args.run)
    device = get_device()
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    vae = VAE(ckpt["args"]["z_dim"]).to(device)
    vae.load_state_dict(ckpt["model"])
    vae.eval()

    torch.manual_seed(0)
    save_image_grid(decode_mean(vae, torch.randn(64, vae.z_dim, device=device)), run / "prior_samples.png")
    x_test, _ = load_images("mnist", train=False, device=device)
    x_test = binarize(x_test)
    logits, _, _ = vae(x_test[:32])
    pairs = torch.stack([x_test[:32] * 2 - 1, torch.sigmoid(logits) * 2 - 1], dim=1).flatten(0, 1)
    save_image_grid(pairs, run / "reconstructions.png", nrow=16)
    if vae.z_dim == 2:
        save_image_grid(latent_grid(vae), run / "latent_grid.png", nrow=20)


if __name__ == "__main__":
    main()
