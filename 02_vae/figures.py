"""Build the README figures and print the results table from the runs in run.sh.

    python 02_vae/figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from common.data import binarize, load_images  # noqa: E402
from common.utils import get_device, load_json  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import VAE  # noqa: E402
from sample import decode_mean, interpolate, latent_grid  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
Z_DIMS, BETAS = [2, 16], [0.5, 1, 4]
COLORS = {2: "#2a78d6", 16: "#eb6834"}


def load_vae(name, device):
    ckpt = torch.load(RUNS / name / "ckpt.pt", map_location=device)
    vae = VAE(ckpt["args"]["z_dim"]).to(device)
    vae.load_state_dict(ckpt["model"])
    return vae.eval()


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@torch.no_grad()
def latent_scatter(vae, x, y, path):
    """Encoder means of test images in grey; each digit's label sits at the median of its points."""
    mu, _ = vae.encode(x)
    mu, y = mu.cpu(), y.cpu()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(mu[:, 0], mu[:, 1], s=1, c="0.6", alpha=0.4, linewidths=0)
    for digit in range(10):
        cx, cy = mu[y == digit].median(dim=0).values.tolist()
        ax.text(cx, cy, str(digit), fontsize=9, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="0.3", lw=0.6))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    style(ax, "$z_1$", "$z_2$")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    x_test, y_test = load_images("mnist", train=False, device=device)
    x_test = binarize(x_test)

    print("| Latent size | beta | Reconstruction | KL | -ELBO | -log p(x) (k = 500) | Gap |")
    print("|---|---|---|---|---|---|---|")
    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    for z in Z_DIMS:
        rates, distortions = [], []
        for beta in BETAS:
            m = load_json(RUNS / f"z{z}_beta{beta:g}" / "metrics.json")
            rates.append(m["kl"])
            distortions.append(m["rec"])
            print(f"| {z} | {beta:g} | {m['rec']:.1f} | {m['kl']:.1f} | {m['neg_elbo']:.1f} "
                  f"| {m['nll']:.1f} | {m['gap']:.1f} |")
            offset = (5, -11) if beta == 0.5 else (5, 4)
            ax.annotate(f"beta={beta:g}", (m["kl"], m["rec"]), textcoords="offset points", xytext=offset, fontsize=7)
        ax.plot(rates, distortions, "o-", color=COLORS[z], lw=1.6, ms=6, label=f"latent size {z}")
    style(ax, "rate: KL(q(z|x) || p(z))  [nats]", "distortion: -E log p(x|z)  [nats]")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "rate_distortion.png", dpi=150)
    plt.close(fig)

    for beta in BETAS:
        vae = load_vae(f"z2_beta{beta:g}", device)
        save_image_grid(latent_grid(vae), FIGS / f"latent_grid_beta{beta:g}.png", nrow=20)
        if beta == 1:
            latent_scatter(vae, x_test, y_test, FIGS / "latent_scatter.png")
            # interpolation between the first test images of digits (1, 7), (3, 8), (4, 9)
            first = {d: (y_test == d).nonzero()[0].item() for d in range(10)}
            rows = [interpolate(vae, x_test[first[a]], x_test[first[b]]) for a, b in [(1, 7), (3, 8), (4, 9)]]
            save_image_grid(torch.cat(rows), FIGS / "interpolation.png", nrow=10)

    torch.manual_seed(0)
    for z in Z_DIMS:
        vae = load_vae(f"z{z}_beta1", device)
        save_image_grid(decode_mean(vae, torch.randn(64, z, device=device)), FIGS / f"prior_samples_z{z}.png")
        logits, _, _ = vae(x_test[:16])
        pairs = torch.stack([x_test[:16] * 2 - 1, torch.sigmoid(logits) * 2 - 1], dim=0).flatten(0, 1)
        save_image_grid(pairs, FIGS / f"reconstructions_z{z}.png", nrow=16)


if __name__ == "__main__":
    main()
