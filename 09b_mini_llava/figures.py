"""Build the README figures and print the results table from the runs in run.sh.

    python 09b_mini_llava/figures.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import get_device, load_json, read_log  # noqa: E402
from sample import answer, cached_features, load_run  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
TYPES = ["count", "shape", "color", "size", "exists"]
BARS = {"projector": ("projector only", "#2a78d6"), "lora": ("projector + LoRA", "#eb6834"),
        "shuffled": ("other image's features", "#1baf7a")}
PERCENT = FuncFormatter(lambda v, _: f"{v:.0%}")


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(PERCENT)
    ax.grid(alpha=0.25, linewidth=0.6, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def accuracy_figure(m_proj, m_lora):
    fig, (ax_bar, ax_curve) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    values = {"projector": [m_proj[f"acc_{t}"] for t in TYPES], "lora": [m_lora[f"acc_{t}"] for t in TYPES],
              "shuffled": [m_lora[f"shuffled_acc_{t}"] for t in TYPES]}
    x = np.arange(len(TYPES))
    for k, (key, (label, color)) in enumerate(BARS.items()):
        ax_bar.bar(x + (k - 1) * 0.27, values[key], 0.25, color=color, label=label)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(TYPES)
    ax_bar.set_ylim(0, 1.05)
    style(ax_bar, "question type", "test accuracy")
    ax_bar.legend(frameon=False, fontsize=7, loc="lower left")

    log = read_log(RUNS / "projector_s0") + [{**r, "step": r["step"] + 3000} for r in read_log(RUNS / "lora_s0")]
    for t, color in zip(TYPES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]):
        ax_curve.plot([r["step"] for r in log], [r[f"acc_{t}"] for r in log], color=color, lw=1.6, label=t)
    ax_curve.axvline(3000, color="0.5", lw=0.8, ls=":")
    ax_curve.text(3050, 0.42, "LoRA on", fontsize=7, color="0.4")
    ax_curve.set_ylim(0.4, 1.02)
    style(ax_curve, "training step", "accuracy (1000 test scenes)")
    ax_curve.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "accuracy.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def examples_figure(device, n: int = 4):
    """A few test images with questions, the model's answers and the true answers."""
    lm, tok, projector = load_run(RUNS / "lora_s0", device)
    test = shapes.load("test")
    feats = cached_features("test", device)
    rng = random.Random(3)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.6))
    for ax, i in zip(axes, [1, 2, 6, 9][:n]):
        qa = shapes.questions(test["scenes"][i], rng)
        qa = [qa[0]] + rng.sample(qa[1:], min(2, len(qa) - 1))
        preds = answer(lm, tok, projector, feats[[i] * len(qa)].to(device), [q for q, _, _ in qa])
        ax.imshow(shapes.to_float(test["images"][i]).add(1).div(2).permute(1, 2, 0))
        ax.set_xticks([])
        ax.set_yticks([])
        lines = [f"{q}\n  -> {p}" + ("" if p == a else f"  (true: {a})") for (q, a, _), p in zip(qa, preds)]
        ax.set_xlabel("\n".join(lines), fontsize=7, loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "examples.png", dpi=150)
    plt.close(fig)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    m_proj, m_lora = load_json(RUNS / "projector_s0" / "metrics.json"), load_json(RUNS / "lora_s0" / "metrics.json")
    accuracy_figure(m_proj, m_lora)
    examples_figure(device)
    print("| Question type | Projector only | Projector + LoRA | Other image's features (LoRA model) |")
    print("|---|---|---|---|")
    for t in TYPES + ["all"]:
        print(f"| {t} | {m_proj[f'acc_{t}']:.3f} | {m_lora[f'acc_{t}']:.3f} | {m_lora[f'shuffled_acc_{t}']:.3f} |")
    for name in ["projector_s0", "lora_s0"]:
        print(name, "trained parameters:", load_json(RUNS / name / "params.json")["trained_params"])


if __name__ == "__main__":
    main()
