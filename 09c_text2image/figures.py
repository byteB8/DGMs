"""Build the README figures and print the results table from the run in run.sh.

    python 09c_text2image/figures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import torch  # noqa: E402

from common.utils import load_json, read_log  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN, FIGS = HERE / "runs" / "t2i_s0", HERE / "figs"
TYPES = ["count", "shape", "color", "size", "exists"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PERCENT = FuncFormatter(lambda v, _: f"{v:.0%}")


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def samples_figure(saved, n: int = 8):
    """Rows: guidance w = 0, 1, 3. Columns: the first n test captions."""
    keys = [k for k in ["w0", "w1", "w3"] if k in saved["samples"]]
    fig, axes = plt.subplots(len(keys), n, figsize=(1.55 * n, 1.75 * len(keys) + 1.6))
    for r, key in enumerate(keys):
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(saved["samples"][key][c].add(1).div(2).permute(1, 2, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"w = {key[1:]}", fontsize=8)
            if r == len(keys) - 1:
                ax.set_xlabel(saved["captions"][c].replace(" and ", "\nand ").replace(", ", ",\n")
                              .replace(" on the", "\non the").replace(" in the", "\nin the"), fontsize=5)
    fig.tight_layout()
    fig.savefig(FIGS / "samples.png", dpi=150)
    plt.close(fig)


def metrics_figure(ev):
    """Left: share of 09b answers that match the caption, per question type (all captions).
    Right: share of images with every answer right, all captions vs captions not seen in training."""
    gen, real = ev["generated"], ev["real"]
    ws = [r["w"] for r in gen]
    fig, (ax_q, ax_f) = plt.subplots(1, 2, figsize=(9, 3.3))
    for t, color in zip(TYPES, COLORS):
        ax_q.plot(ws, [r[f"all_vlm_{t}"] for r in gen], "o-", color=color, lw=1.6, label=t)
    ax_q.yaxis.set_major_formatter(PERCENT)
    style(ax_q, "guidance weight w", "09b answers that match the caption")
    ax_q.legend(frameon=False, fontsize=7, ncol=2)
    for subset, color in [("all", "#2a78d6"), ("unseen", "#eb6834")]:
        label = "all captions" if subset == "all" else "captions not seen in training"
        ax_f.plot(ws, [r[f"{subset}_vlm_image_fully_correct"] for r in gen], "o-", color=color, lw=1.6,
                  label=f"generated, {label}")
        ax_f.axhline(real[f"{subset}_vlm_image_fully_correct"], color=color, lw=1, ls=":")
    ax_f.text(ws[-1], real["all_vlm_image_fully_correct"], "real images", fontsize=7, color="0.4",
              ha="right", va="bottom")
    ax_f.yaxis.set_major_formatter(PERCENT)
    style(ax_f, "guidance weight w", "images with every answer right")
    ax_f.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "metrics.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    ev = load_json(RUN / "eval.json")
    saved = torch.load(RUN / "eval_samples.pt")
    samples_figure(saved)
    metrics_figure(ev)
    steps = [r["step"] for r in read_log(RUN)]
    print(f"training snapshots at steps {steps}")

    sizes = ev["subset_sizes"]
    print(f"captions: {sizes['all']} ({sizes['unseen']} not seen in training, all of them three-object scenes)")
    print("| Images | Answers right, all | Every answer right, all | Answers right, unseen | "
          "Every answer right, unseen | Yes / no, unseen | CLIP score, all |")
    print("|---|---|---|---|---|---|---|")
    rows = [("real test images", ev["real"])] + [(f"generated, w = {r['w']:g}", r) for r in ev["generated"]]
    for label, r in rows:
        cells = [r["all_vlm_all"], r["all_vlm_image_fully_correct"], r["unseen_vlm_all"],
                 r["unseen_vlm_image_fully_correct"], r["unseen_vlm_exists"], r["all_clip_score"]]
        print(f"| {label} | " + " | ".join(f"{c:.3f}" for c in cells) + " |")


if __name__ == "__main__":
    main()
