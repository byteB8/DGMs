"""Build the README figures and print the results tables from the runs in run.sh.

    python 08_vqvae_prior/figures.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from common.data import load_images  # noqa: E402
from common.utils import get_device, load_json, read_log  # noqa: E402
from model import bits  # noqa: E402
from sample import load  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
VQ, PRIOR = RUNS / "vqvae_s0", RUNS / "prior_s0"


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


@torch.no_grad()
def code_map_figure(vqvae, device, n: int = 6):
    """Test images next to their 7 x 7 code grids (each code drawn as its index)."""
    x, _ = load_images("fmnist", train=False, device=device)
    codes = vqvae.encode(x[:n]).cpu()
    fig, axes = plt.subplots(2, n, figsize=(1.6 * n, 3.4))
    for k in range(n):
        axes[0, k].imshow(x[k, 0].cpu(), cmap="gray")
        axes[1, k].imshow(codes[k], cmap="Blues", vmin=0, vmax=vqvae.quantizer.codebook.num_embeddings)
        for i in range(codes.shape[1]):
            for j in range(codes.shape[2]):
                axes[1, k].text(j, i, str(codes[k, i, j].item()), ha="center", va="center", fontsize=4.5)
        for ax in axes[:, k]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("image", fontsize=8)
    axes[1, 0].set_ylabel("codes", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "codes.png", dpi=150)
    plt.close(fig)


def curves_figure():
    vq, prior = read_log(VQ), read_log(PRIOR)
    fig, (ax_vq, ax_use, ax_pr) = plt.subplots(1, 3, figsize=(12, 3.1))
    ax_vq.plot([r["step"] for r in vq], [r["mse"] for r in vq], color="#2a78d6", lw=1.6)
    style(ax_vq, "epoch", "test reconstruction MSE")
    ax_use.plot([r["step"] for r in vq], [r["perplexity"] for r in vq], color="#2a78d6", lw=1.6)
    style(ax_use, "epoch", "code perplexity (of 128)")
    ax_pr.plot([r["step"] for r in prior], [r["train_nll_per_code"] for r in prior], color="#2a78d6", lw=1.6,
               label="train")
    ax_pr.plot([r["step"] for r in prior], [r["test_nll_per_code"] for r in prior], color="#eb6834", lw=1.6,
               label="test")
    style(ax_pr, "prior epoch", "-log p(code)  [nats]")
    ax_pr.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "curves.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    vqvae, _ = load(VQ, PRIOR, device)
    code_map_figure(vqvae, device)
    curves_figure()
    shutil.copy(PRIOR / "samples.png", FIGS / "samples.png")
    shutil.copy(PRIOR / "reconstructions.png", FIGS / "reconstructions.png")

    vq, pr, s = load_json(VQ / "metrics.json"), load_json(PRIOR / "metrics.json"), load_json(PRIOR / "sample_metrics.json")
    n_codes = load_json(VQ / "args.json")["n_codes"]
    print("| VQ-VAE | Value |\n|---|---|")
    print(f"| Test MSE / PSNR | {vq['mse']:.4f} / {vq['psnr']:.1f} dB |")
    print(f"| Codes used / perplexity (of {n_codes}) | {vq['codes_used']} / {vq['perplexity']:.1f} |")
    print(f"| Size of one image as codes | 49 x log2({n_codes}) = {bits(n_codes):.0f} bits |")
    print(f"| Parameters | {vq['params'] / 1e6:.2f} M |")
    print("\n| Prior | Value |\n|---|---|")
    print(f"| Test -log p(code) | {pr['nll_per_code']:.3f} nats per code |")
    print(f"| Test -log p(codes) per image | {pr['bits_per_image']:.0f} bits (vs {bits(n_codes):.0f} for uniform codes) |")
    print(f"| Parameters | {pr['params'] / 1e6:.2f} M |")
    print("\n| Images | FID-lite | Class KL | Label accuracy |\n|---|---|---|---|")
    r = s["reconstructions"]
    print(f"| Reconstructions | {r['fid_lite']:.1f} | {r['class_kl']:.3f} | - |")
    g = s["samples"]
    print(f"| Prior samples (class-conditional) | {g['fid_lite']:.1f} | {g['class_kl']:.3f} | {g['label_acc']:.2f} |")
    print(f"\nSampling: {s['sample_seconds_per_1k']:.1f} s per 1000 images (49 prior steps + one decoder pass)")


if __name__ == "__main__":
    main()
