"""Build the README figures and print the results tables from the runs in run.sh.
The Fashion-MNIST comparison also reads 06_ddpm's sweep, so run 06 first.

    python 07_flow_matching/figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402
import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.utils import get_device, load_json  # noqa: E402
from common.viz import plot_2d_panels, save_image_grid  # noqa: E402
from model import build  # noqa: E402
from sample import euler_sample  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
DDPM_SWEEP = HERE.parent / "06_ddpm" / "runs" / "fmnist_cond_s0" / "sweep.json"
COLORS = {"fm": "#2a78d6", "reflow": "#eb6834", "ddim": "#1baf7a"}


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load_model(name, device):
    ckpt = torch.load(RUNS / name / "ckpt.pt", map_location=device)
    model = build(ckpt["args"]["data"], ckpt["args"]["cond"], ckpt["args"]["ch"]).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval()


def toy_figures(device):
    print("**8 Gaussians** (straightness: 1-step vs 100-step endpoint distance)\n")
    print("| Model | Straightness | 1 step | 2 | 5 | 10 | 20 | 100 |\n|---|---|---|---|---|---|---|---|")
    for name, label in [("8gauss_s0", "flow matching"), ("8gauss_reflow_s0", "after reflow")]:
        sweep = load_json(RUNS / name / "sweep.json")
        cells = {r["steps"]: f"{r['modes']} / {r['hq']:.2f}" for r in sweep["steps"]}
        print(f"| {label} | {sweep['straightness']:.2f} | " + " | ".join(cells[k] for k in [1, 2, 5, 10, 20, 100]) + " |")
    print("\nCells: modes covered (of 8) / share of high-quality samples.")

    for name in ["8gauss_s0", "8gauss_reflow_s0", "swissroll_s0"]:
        samples = torch.load(RUNS / name / "sweep_samples.pt")
        data = "swissroll" if "swissroll" in name else "8gauss"
        plot_2d_panels([(f"{k} step{'s' if k > 1 else ''}", samples[f"euler_{k}"]) for k in [1, 2, 5, 10, 100]],
                       FIGS / f"{name.replace('_s0', '')}_steps.png", real=toy2d.sample(data, 3000))

    # sampling paths from the same noise, before and after reflow
    torch.manual_seed(0)
    x0 = torch.randn(300, 2, device=device)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    for ax, (name, title) in zip(axes, [("8gauss_s0", "flow matching"), ("8gauss_reflow_s0", "after reflow")]):
        path = torch.stack(euler_sample(load_model(name, device), x0, 100, return_path=True)).cpu()  # (101, n, 2)
        for i in range(path.shape[1]):
            ax.plot(path[:, i, 0], path[:, i, 1], color=COLORS["fm" if name == "8gauss_s0" else "reflow"],
                    lw=0.5, alpha=0.5)
        ax.scatter(path[0, :, 0], path[0, :, 1], s=3, c="0.4", zorder=3, linewidths=0, label="noise (t = 0)")
        ax.scatter(path[-1, :, 0], path[-1, :, 1], s=3, c="black", zorder=3, linewidths=0, label="sample (t = 1)")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGS / "8gauss_paths.png", dpi=150)
    plt.close(fig)


def image_figures():
    run = RUNS / "fmnist_cond_s0"
    sweep, ddpm = load_json(run / "sweep.json"), load_json(DDPM_SWEEP)
    samples = torch.load(run / "sweep_samples.pt")
    ddim = {r["steps"]: r for r in ddpm["samplers"] if r["sampler"] == "ddim"}

    print(f"\n**Fashion-MNIST, steps** (unconditional; straightness {sweep['straightness']:.1f})\n")
    print("| Steps | Flow matching FID-lite | DDIM (06) FID-lite | Flow matching class KL |\n|---|---|---|---|")
    for r in sweep["steps"]:
        d = f"{ddim[r['steps']]['fid_lite']:.1f}" if r["steps"] in ddim else "-"
        print(f"| {r['steps']} | {r['fid_lite']:.1f} | {d} | {r['class_kl']:.3f} |")
    print("\n**Fashion-MNIST, guidance** (50 Euler steps)\n")
    print("| w | FID-lite | Label accuracy | Class KL |\n|---|---|---|---|")
    for r in sweep["guidance"]:
        print(f"| {r['w']:g} | {r['fid_lite']:.1f} | {r['label_acc']:.3f} | {r['class_kl']:.3f} |")

    for k in [1, 5, 10, 50]:
        save_image_grid(samples[f"euler_{k}"][:64].clamp(-1, 1), FIGS / f"fmnist_euler_{k}.png")
    for w in [0, 1, 4]:
        save_image_grid(samples[f"guided_w{w}"].clamp(-1, 1), FIGS / f"fmnist_guided_w{w}.png", nrow=10)

    fig, (ax_steps, ax_acc) = plt.subplots(1, 2, figsize=(9, 3.2))
    ax_steps.plot([r["steps"] for r in sweep["steps"]], [r["fid_lite"] for r in sweep["steps"]], "o-",
                  color=COLORS["fm"], lw=1.6, label="flow matching (Euler)")
    ax_steps.plot(sorted(ddim), [ddim[k]["fid_lite"] for k in sorted(ddim)], "s-", color=COLORS["ddim"], lw=1.6,
                  label="DDPM model, DDIM sampler (06)")
    ax_steps.set_xscale("log")
    ax_steps.set_yscale("log")
    for axis in (ax_steps.xaxis, ax_steps.yaxis):
        axis.set_major_formatter(ScalarFormatter())
    ax_steps.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    style(ax_steps, "sampling steps", "FID-lite (lower is better)")
    ax_steps.legend(frameon=False, fontsize=8)
    ws = [r["w"] for r in sweep["guidance"]]
    ax_acc.plot(ws, [r["label_acc"] for r in sweep["guidance"]], "o-", color=COLORS["fm"], lw=1.6,
                label="flow matching")
    ax_acc.plot([r["w"] for r in ddpm["guidance"]], [r["label_acc"] for r in ddpm["guidance"]], "s-",
                color=COLORS["ddim"], lw=1.6, label="DDPM (06)")
    ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    style(ax_acc, "guidance weight w", "label accuracy")
    ax_acc.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fmnist_sweeps.png", dpi=150)
    plt.close(fig)


def same_noise_figure():
    """06 (DDIM, 50 steps) and 07 (Euler, 50 steps) sampled from the same noise (both sweeps use seed 0).
    Both are deterministic maps from noise to images; compare paired samples with random pairs."""
    fm = torch.load(RUNS / "fmnist_cond_s0" / "sweep_samples.pt")["euler_50"].clamp(-1, 1)
    dd = torch.load(DDPM_SWEEP.parent / "sweep_samples.pt")["ddim_50"].clamp(-1, 1)
    paired = (fm - dd).flatten(1).norm(dim=1).mean().item()
    shuffled = (fm - dd.roll(1, dims=0)).flatten(1).norm(dim=1).mean().item()
    print(f"\nSame noise, 06 DDIM-50 vs 07 Euler-50: mean L2 distance {paired:.2f} for pairs, "
          f"{shuffled:.2f} for random pairs")
    save_image_grid(torch.stack([dd[:16], fm[:16]]).flatten(0, 1), FIGS / "same_noise.png", nrow=16)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    toy_figures(device)
    image_figures()
    same_noise_figure()


if __name__ == "__main__":
    main()
