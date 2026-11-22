"""Build the README figures and print the results table from the run in run.sh.

    python 04_autoregressive/figures.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import torch  # noqa: E402

from common.utils import get_device, load_json, read_log  # noqa: E402
from model import GatedPixelCNN  # noqa: E402
from sample import dependency_map  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN, FIGS = HERE / "runs" / "pixelcnn_ch64_s0", HERE / "figs"


def dependency_figure(model, device):
    """Grey = pixels the output at the outlined pixel depends on (non-zero gradient)."""
    positions = [(3, 20), (14, 14), (27, 27)]
    fig, axes = plt.subplots(1, len(positions), figsize=(2.6 * len(positions), 2.8))
    for ax, (i, j) in zip(axes, positions):
        seen = (dependency_map(model, i, j, device) > 0).float().cpu()
        ax.imshow(seen, cmap="Greys", vmin=0, vmax=2.5)
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=1.5))
        ax.set_title(f"output pixel ({i}, {j})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGS / "dependencies.png", dpi=150)
    plt.close(fig)


def curve_figure():
    log = read_log(RUN)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot([r["step"] for r in log], [r["train_nll"] for r in log], color="#2a78d6", lw=1.6, label="train")
    ax.plot([r["step"] for r in log], [r["test_nll"] for r in log], color="#eb6834", lw=1.6, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("-log p(x)  [nats per image]")
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "nll_curve.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    ckpt = torch.load(RUN / "ckpt.pt", map_location=device)
    model = GatedPixelCNN(ckpt["args"]["ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dependency_figure(model, device)
    curve_figure()
    shutil.copy(RUN / "samples.png", FIGS / "samples.png")

    m, s = load_json(RUN / "metrics.json"), load_json(RUN / "sampling.json")
    print("| Test -log p(x) | Bits per pixel | Parameters | Time for 64 samples |")
    print("|---|---|---|---|")
    print(f"| {m['nll']:.2f} nats | {m['bits_per_pixel']:.4f} | {m['params'] / 1e6:.2f} M | {s['sample_seconds']:.1f} s |")


if __name__ == "__main__":
    main()
