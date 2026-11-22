"""Build the README figures and print the results tables from the runs in run.sh.

    python 01_gan/figures.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402
import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.data import load_images  # noqa: E402
from common.utils import get_device, read_log  # noqa: E402
from common.viz import plot_2d_panels, save_image_grid  # noqa: E402
from model import build  # noqa: E402
from sample import generate  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
TOY_RUNS = {"minimax": "8gauss_minimax_s0", "ns": "8gauss_ns_s0", "wgan-gp": "8gauss_wgan-gp_s0",
            "gp1": "8gauss_wgan-gp_gp1_s0", "gp0.1": "8gauss_wgan-gp_gp0.1_s0"}
IMAGE_RUNS = {"ns": "fmnist_ns_s0", "wgan-gp": "fmnist_wgan-gp_s0", "ns_cond": "fmnist_ns_cond_s0"}
COLORS = {"minimax": "#2a78d6", "ns": "#eb6834", "wgan-gp": "#1baf7a", "gp1": "#eda100", "gp0.1": "#e87ba4",
          "ns_cond": "#008300"}
DASHES = {"minimax": "-", "gp1": "--", "gp0.1": ":"}   # three runs overlap at 8 modes; dashes keep them visible
THOUSANDS = FuncFormatter(lambda v, _: f"{v / 1000:g}k" if v else "0")
LABELS = {"minimax": "minimax", "ns": "non-saturating", "wgan-gp": "WGAN-GP, $\\lambda$=10",
          "gp1": "WGAN-GP, $\\lambda$=1", "gp0.1": "WGAN-GP, $\\lambda$=0.1", "ns_cond": "conditional"}
TABLE_LABELS = {"minimax": "minimax", "ns": "non-saturating", "wgan-gp": "WGAN-GP, lambda = 10",
                "gp1": "WGAN-GP, lambda = 1", "gp0.1": "WGAN-GP, lambda = 0.1"}


def load_generator(run, device):
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    cfg = ckpt["args"]
    G, _ = build(cfg["data"], cfg["z_dim"], cfg["cond"])
    G.load_state_dict(ckpt["G"])
    return G.to(device), cfg


def style(ax, xlabel, ylabel):
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def toy_figures(device):
    torch.manual_seed(0)
    panels = []
    for key, name in TOY_RUNS.items():
        G, cfg = load_generator(RUNS / name, device)
        panels.append((LABELS[key], generate(G, 3000, cfg["z_dim"], device)))
        shutil.copy(RUNS / name / "training.png", FIGS / f"8gauss_training_{key}.png")
    plot_2d_panels(panels, FIGS / "8gauss_samples.png", real=toy2d.sample("8gauss", 3000))

    fig, (ax_modes, ax_grad) = plt.subplots(1, 2, figsize=(9, 3.2))
    print("| Model | Modes covered (of 8) | High-quality samples |")
    print("|---|---|---|")
    for key, name in TOY_RUNS.items():
        log = read_log(RUNS / name)
        steps = [r["step"] for r in log]
        ax_modes.plot(steps, [r["modes"] for r in log], color=COLORS[key], lw=1.8, ls=DASHES.get(key, "-"),
                      label=LABELS[key])
        if key in ("minimax", "ns"):  # a WGAN critic has another scale, so its gradients are not comparable
            ax_grad.plot(steps, [r["g_grad"] for r in log], color=COLORS[key], lw=1.6, label=LABELS[key])
        print(f"| {TABLE_LABELS[key]} | {log[-1]['modes']} | {log[-1]['hq']:.2f} |")
    style(ax_modes, "generator step", "modes covered (of 8)")
    ax_modes.set_ylim(-0.3, 8.6)
    ax_modes.set_yticks(range(0, 9, 2))
    style(ax_grad, "generator step", "generator gradient norm")
    ax_grad.set_yscale("log")
    ax_grad.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    handles, labels = ax_modes.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGS / "8gauss_curves.png", dpi=150)
    plt.close(fig)


def image_figures(device):
    x_test, _ = load_images("fmnist", train=False)
    save_image_grid(x_test[:64], FIGS / "fmnist_real.png")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    print("\n| Model | FID-lite | Class KL | Rarest class share | Label accuracy |")
    print("|---|---|---|---|---|")
    for key, name in IMAGE_RUNS.items():
        G, cfg = load_generator(RUNS / name, device)
        torch.manual_seed(0)
        if cfg["cond"]:  # 10 x 10 grid, column j = class j
            labels = torch.arange(100, device=device) % 10
            save_image_grid(generate(G, 100, cfg["z_dim"], device, labels), FIGS / f"fmnist_{key}.png", nrow=10)
        else:
            save_image_grid(generate(G, 64, cfg["z_dim"], device), FIGS / f"fmnist_{key}.png")
        log = read_log(RUNS / name)
        ax.plot([r["step"] for r in log], [r["fid_lite"] for r in log], color=COLORS[key], lw=1.6,
                label=LABELS[key])
        f = log[-1]
        acc = f"{f['label_acc']:.2f}" if "label_acc" in f else "-"
        print(f"| {TABLE_LABELS.get(key, LABELS[key])} | {f['fid_lite']:.1f} | {f['class_kl']:.3f} | {f['min_class']:.3f} | {acc} |")
    style(ax, "generator step", "FID-lite (lower is better)")
    ax.set_yscale("log")
    ax.set_yticks([15, 25, 50, 100, 200])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fmnist_fid.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    toy_figures(device)
    image_figures(device)


if __name__ == "__main__":
    main()
