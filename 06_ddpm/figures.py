"""Build the README figures and print the results tables from the runs in run.sh.

    python 06_ddpm/figures.py
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
from common.data import load_images  # noqa: E402
from common.utils import load_json  # noqa: E402
from common.viz import plot_2d_panels, save_image_grid  # noqa: E402
from objective import NoiseSchedule, q_sample  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
SAMPLER_LABELS = {"ddpm_1000": "DDPM, 1000 steps", "ddim_10": "DDIM, 10", "ddim_50": "DDIM, 50",
                  "ddim_100": "DDIM, 100", "ddim_1000": "DDIM, 1000"}


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def forward_figures():
    """q(x_t | x_0) at a few times, for the swiss roll and for one Fashion-MNIST image."""
    sched = NoiseSchedule()
    times = [0, 50, 150, 300, 600, 999]
    torch.manual_seed(0)
    x0 = toy2d.sample("swissroll", 3000)
    panels = [(f"t = {i + 1}", q_sample(sched, x0, torch.full((len(x0),), i), torch.randn_like(x0))) for i in times]
    plot_2d_panels(panels, FIGS / "forward_swissroll.png", lim=3.5)
    img = load_images("fmnist", train=False)[0][:1].repeat(len(times), 1, 1, 1)
    noised = q_sample(sched, img, torch.tensor(times), torch.randn_like(img))
    save_image_grid(noised, FIGS / "forward_fmnist.png", nrow=len(times))


def toy_figures():
    print("**8 Gaussians**\n\n| Sampler | Steps | Modes covered | High-quality samples |\n|---|---|---|---|")
    for r in load_json(RUNS / "8gauss_s0" / "sweep.json")["samplers"]:
        print(f"| {r['sampler'].upper()} | {r['steps']} | {r['modes']} | {r['hq']:.2f} |")
    for data in ["swissroll", "8gauss"]:
        samples = torch.load(RUNS / f"{data}_s0" / "sweep_samples.pt")
        keys = ["ddpm_1000", "ddim_10", "ddim_50", "ddim_1000"]
        plot_2d_panels([(SAMPLER_LABELS[k], samples[k]) for k in keys], FIGS / f"{data}_samplers.png",
                       real=toy2d.sample(data, 3000))


def image_figures():
    run = RUNS / "fmnist_cond_s0"
    sweep = load_json(run / "sweep.json")
    samples = torch.load(run / "sweep_samples.pt")

    print("\n**Fashion-MNIST, samplers** (unconditional: the 'no label' input)\n")
    print("| Sampler | Steps | FID-lite | Class KL | Rarest class share | Seconds per 1k samples |")
    print("|---|---|---|---|---|---|")
    for r in sweep["samplers"]:
        print(f"| {r['sampler'].upper()} | {r['steps']} | {r['fid_lite']:.1f} | {r['class_kl']:.3f} "
              f"| {r['min_class']:.3f} | {r['seconds_per_1k']:.0f} |")
    print("\n**Fashion-MNIST, guidance** (DDIM, 50 steps)\n")
    print("| w | FID-lite | Label accuracy | Class KL |\n|---|---|---|---|")
    for r in sweep["guidance"]:
        print(f"| {r['w']:g} | {r['fid_lite']:.1f} | {r['label_acc']:.3f} | {r['class_kl']:.3f} |")

    for key in ["ddpm_1000", "ddim_10", "ddim_50"]:
        save_image_grid(samples[key][:64], FIGS / f"fmnist_{key}.png")
    for w in [0, 1, 4]:
        save_image_grid(samples[f"guided_w{w}"], FIGS / f"fmnist_guided_w{w}.png", nrow=10)

    ddim = [r for r in sweep["samplers"] if r["sampler"] == "ddim"]
    ddpm = [r for r in sweep["samplers"] if r["sampler"] == "ddpm"][0]
    fig, (ax_steps, ax_fid, ax_acc) = plt.subplots(1, 3, figsize=(12, 3.2))
    ax_steps.plot([r["steps"] for r in ddim], [r["fid_lite"] for r in ddim], "o-", color="#2a78d6", lw=1.6,
                  label="DDIM")
    ax_steps.plot([ddpm["steps"]], [ddpm["fid_lite"]], "s", color="#eb6834", ms=8, label="DDPM")
    ax_steps.set_xscale("log")
    ax_steps.xaxis.set_major_formatter(ScalarFormatter())
    style(ax_steps, "sampling steps", "FID-lite (lower is better)")
    ax_steps.legend(frameon=False, fontsize=8)
    ws = [r["w"] for r in sweep["guidance"]]
    ax_fid.plot(ws, [r["fid_lite"] for r in sweep["guidance"]], "o-", color="#2a78d6", lw=1.6)
    style(ax_fid, "guidance weight w", "FID-lite")
    ax_acc.plot(ws, [r["label_acc"] for r in sweep["guidance"]], "o-", color="#2a78d6", lw=1.6)
    ax_acc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
    style(ax_acc, "guidance weight w", "label accuracy")
    fig.tight_layout()
    fig.savefig(FIGS / "fmnist_sweeps.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    forward_figures()
    toy_figures()
    image_figures()


if __name__ == "__main__":
    main()
