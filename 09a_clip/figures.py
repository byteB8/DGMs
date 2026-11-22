"""Build the README figures and print the results table from the runs in run.sh.

    python 09a_clip/figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import get_device, load_json, read_log  # noqa: E402
from sample import embed, load_clip  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
RUN_NAMES = {"clip_s0": "in-batch negatives", "clip_hardneg_s0": "+ swapped-caption negatives"}
COLORS = {"clip_s0": "#2a78d6", "clip_hardneg_s0": "#eb6834"}
PERCENT = FuncFormatter(lambda v, _: f"{v:.0%}")


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def curves():
    fig, (ax_ret, ax_bind) = plt.subplots(1, 2, figsize=(9, 3.3))
    for name, label in RUN_NAMES.items():
        log = read_log(RUNS / name)
        ep = [r["step"] for r in log]
        ax_ret.plot(ep, [r["text_to_image_top1"] for r in log], color=COLORS[name], lw=1.6, label=label)
        ax_bind.plot(ep, [r["binding_color"] for r in log], color=COLORS[name], lw=1.6, label=f"{label}, color")
        ax_bind.plot(ep, [r["binding_shape"] for r in log], color=COLORS[name], lw=1.6, ls="--",
                     label=f"{label}, shape")
    style(ax_ret, "epoch", "text-to-image top-1")
    ax_ret.legend(frameon=False, fontsize=8)
    ax_bind.axhline(0.5, color="0.5", lw=0.8, ls=":")
    ax_bind.text(0.5, 0.51, "chance", fontsize=7, color="0.4", ha="left", va="bottom")
    style(ax_bind, "epoch", "binding accuracy (true vs swapped)")
    ax_bind.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGS / "curves.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def similarity_figure(model, data, device, n: int = 8):
    """Cosine similarity between n test images and their n captions."""
    img, txt = embed(model, data["images"][:n], data["captions"][:n], device)
    sim = (img @ txt.t()).cpu()
    fig = plt.figure(figsize=(9.5, 5.2))
    ax_img = fig.add_axes([0.30, 0.83, 0.55, 0.14])
    strip = shapes.to_float(data["images"][:n]).add(1).div(2).permute(0, 2, 3, 1)
    ax_img.imshow(torch.cat(list(strip), dim=1))
    ax_img.axis("off")
    ax = fig.add_axes([0.30, 0.05, 0.55, 0.76])
    im = ax.imshow(sim.t(), cmap="Blues", vmin=sim.min().item(), vmax=1)
    ax.set_xticks([])
    ax.set_yticks(range(n))
    ax.set_yticklabels([c if len(c) < 60 else c[:57] + "..." for c in data["captions"][:n]], fontsize=6.5)
    for i in range(n):
        for j in range(n):
            ax.text(i, j, f"{sim[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if sim[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, label="cosine similarity")
    fig.savefig(FIGS / "similarity.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def retrieval_figure(model, data, device, queries=(0, 3, 7), n: int = 1000, k: int = 5):
    """For a few test captions, the top-k test images."""
    img, txt = embed(model, data["images"][:n], data["captions"][:n], device)
    fig, axes = plt.subplots(len(queries), k, figsize=(1.3 * k + 3.5, 1.4 * len(queries)))
    for row, q in enumerate(queries):
        top = (txt[q] @ img.t()).topk(k).indices.tolist()
        for col, i in enumerate(top):
            ax = axes[row, col]
            ax.imshow(shapes.to_float(data["images"][i]).add(1).div(2).permute(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            if data["captions"][i] == data["captions"][q]:
                for side in ax.spines.values():
                    side.set_edgecolor("#1baf7a")
                    side.set_linewidth(2)
        axes[row, 0].set_ylabel(data["captions"][q].replace(" and ", "\nand ").replace(", ", ",\n"), fontsize=6,
                                rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(FIGS / "retrieval.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    data = shapes.load("test")
    curves()
    model = load_clip(RUNS / "clip_hardneg_s0", device)
    similarity_figure(model, data, device)
    retrieval_figure(model, data, device)

    keys = [("text_to_image_top1", "Text to image, top-1"), ("image_to_text_top1", "Image to text, top-1"),
            ("zero_shot_shape", "Zero-shot shape"), ("zero_shot_color", "Zero-shot color"),
            ("zero_shot_size", "Zero-shot size"), ("zero_shot_position", "Zero-shot position"),
            ("binding_color", "Binding, colors swapped"), ("binding_shape", "Binding, shapes swapped")]
    m = {name: load_json(RUNS / name / "metrics.json") for name in RUN_NAMES}
    print("| Metric (test set) | " + " | ".join(RUN_NAMES.values()) + " |")
    print("|---|" + "---|" * len(RUN_NAMES))
    for key, label in keys:
        print(f"| {label} | " + " | ".join(f"{m[name][key]:.3f}" for name in RUN_NAMES) + " |")
    for name in RUN_NAMES:
        log = read_log(RUNS / name)
        first = next((r["step"] for r in log if r["binding_shape"] >= 0.95), None)
        print(f"{name}: shape binding first >= 95 % at epoch {first}")


if __name__ == "__main__":
    main()
