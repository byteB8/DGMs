"""Build the README figures and print the results table from the runs in run.sh.

    python 03_flow/figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.utils import get_device, load_json  # noqa: E402
from model import RealNVP  # noqa: E402
from sample import density_grid, sample  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
DATASETS = ["8gauss", "swissroll", "moons"]
QUADRANT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LIM = 3.0


def load_flow(name, device):
    ckpt = torch.load(RUNS / name / "ckpt.pt", map_location=device)
    flow = RealNVP(ckpt["args"]["layers"], ckpt["args"]["hidden"]).to(device)
    flow.load_state_dict(ckpt["model"])
    return flow.eval()


def clean(ax, title=None, lim=LIM):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)


def density_figure(device):
    """Top row: learned density q(x). Bottom row: flow samples (blue) over data (grey)."""
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(2.8 * len(DATASETS), 5.6))
    torch.manual_seed(0)
    for j, data in enumerate(DATASETS):
        flow = load_flow(f"{data}_s0", device)
        q = density_grid(flow, device, LIM).cpu()
        axes[0, j].imshow(q, origin="lower", extent=(-LIM, LIM, -LIM, LIM), cmap="Blues")
        clean(axes[0, j], f"{data}: q(x)")
        real = toy2d.sample(data, 3000)
        fake = sample(flow, 3000, device).cpu()
        axes[1, j].scatter(real[:, 0], real[:, 1], s=1, c="0.75", alpha=0.5, linewidths=0)
        axes[1, j].scatter(fake[:, 0], fake[:, 1], s=1, c="#2a78d6", alpha=0.6, linewidths=0)
        clean(axes[1, j], f"{data}: samples")
    fig.tight_layout()
    fig.savefig(FIGS / "density.png", dpi=150)
    plt.close(fig)


def layers_figure(device):
    """Gaussian samples pushed through the flow, coloured by the quadrant of z they start in."""
    flow = load_flow("8gauss_s0", device)
    torch.manual_seed(0)
    points = sample(flow, 4000, device, return_all=True)
    z = points[0].cpu()
    quadrant = (z[:, 0] < 0).long() + 2 * (z[:, 1] < 0).long()
    picks = list(range(0, len(points), 2))
    if picks[-1] != len(points) - 1:
        picks.append(len(points) - 1)
    fig, axes = plt.subplots(1, len(picks), figsize=(2.4 * len(picks), 2.6))
    for ax, i in zip(axes, picks):
        p = points[i].cpu()
        for q, color in enumerate(QUADRANT_COLORS):
            sel = quadrant == q
            ax.scatter(p[sel, 0], p[sel, 1], s=1, c=color, alpha=0.6, linewidths=0)
        lim = max(LIM, p.abs().quantile(0.995).item() * 1.05)   # middle layers can leave the data range
        clean(ax, "z ~ N(0, I)" if i == 0 else ("x" if i == len(points) - 1 else f"after {i} layers"), lim)
    fig.tight_layout()
    fig.savefig(FIGS / "layers.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    density_figure(device)
    layers_figure(device)
    print("| Data | Test NLL | True NLL | KL(p \\|\\| q) |")
    print("|---|---|---|---|")
    for data in DATASETS:
        m = load_json(RUNS / f"{data}_s0" / "metrics.json")
        true_nll = f"{m['true_nll']:.3f}" if "true_nll" in m else "-"
        kl = f"{m['kl']:.3f}" if "kl" in m else "-"
        print(f"| {data} | {m['nll']:.3f} | {true_nll} | {kl} |")


if __name__ == "__main__":
    main()
