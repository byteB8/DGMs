"""Build the README figures and print the results table from the runs in run.sh.

    python 05_ssm/figures.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from common.utils import get_device, load_json, read_log  # noqa: E402
from model import START, SequenceModel  # noqa: E402
from objective import nll_per_sequence  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE / "runs", HERE / "figs"
KINDS = {"ssm": "S4D", "transformer": "transformer"}
COLORS = {"ssm": "#2a78d6", "transformer": "#eb6834"}


def load_model(kind, device):
    ckpt = torch.load(RUNS / f"{kind}_s0" / "ckpt.pt", map_location=device)
    model = SequenceModel(kind, ckpt["args"]["d"], ckpt["args"]["layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def train_step_ms(model, device, batch: int = 64, repeats: int = 10) -> float:
    """Time of one forward + backward pass on a batch of 784-step sequences."""
    model.train()
    x = (torch.rand(batch, 784, device=device) > 0.5).float()
    for _ in range(2):
        nll_per_sequence(model(x), x).mean().backward()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        nll_per_sequence(model(x), x).mean().backward()
    torch.cuda.synchronize()
    return (time.time() - start) / repeats * 1000


@torch.no_grad()
def step_times(model, device, batch: int = 64, length: int = 784):
    """Wall-clock time of each sampling step, in ms (step mode), after a short warm-up."""
    model.eval()
    states = model.initial_states(batch, device)
    token = torch.full((batch,), START, device=device)
    for t in range(20):  # warm-up: the first calls include one-off CUDA set-up
        _, states = model.step(token, t, states)
    states = model.initial_states(batch, device)
    token = torch.full((batch,), START, device=device)
    times = []
    for t in range(length):
        torch.cuda.synchronize()
        start = time.time()
        logit, states = model.step(token, t, states)
        token = torch.bernoulli(torch.sigmoid(logit))
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    return times


def style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    FIGS.mkdir(exist_ok=True)
    device = get_device()
    fig, (ax_nll, ax_step) = plt.subplots(1, 2, figsize=(9, 3.2))
    rows = []
    for kind, label in KINDS.items():
        run = RUNS / f"{kind}_s0"
        log, m, s = read_log(run), load_json(run / "metrics.json"), load_json(run / "sampling.json")
        ax_nll.plot([r["step"] for r in log], [r["test_nll"] for r in log], color=COLORS[kind], lw=1.6, label=label)
        model = load_model(kind, device)
        times = step_times(model, device)
        window = 49
        smooth = [sum(times[i:i + window]) / window for i in range(0, len(times) - window + 1, window)]
        ax_step.plot([i + window // 2 for i in range(0, len(times) - window + 1, window)], smooth,
                     color=COLORS[kind], lw=1.6, label=label)
        rows.append((label, m, s, train_step_ms(model, device)))
        shutil.copy(run / "samples.png", FIGS / f"samples_{kind}.png")
    style(ax_nll, "epoch", "test -log p(x)  [nats]")
    ax_nll.legend(frameon=False, fontsize=8)
    style(ax_step, "position t being sampled", "time per step  [ms]")
    ax_step.set_ylim(bottom=0)
    ax_step.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "curves.png", dpi=150)
    plt.close(fig)

    print("| Model | Parameters | Test -log p(x) | Bits per pixel | Train step (batch 64) | 64 samples | "
          "Max gap, parallel vs step mode |")
    print("|---|---|---|---|---|---|---|")
    for label, m, s, ms in rows:
        print(f"| {label} | {m['params'] / 1e6:.2f} M | {m['nll']:.2f} nats | {m['bits_per_pixel']:.4f} | "
              f"{ms:.0f} ms | {s['sample_seconds']:.1f} s | {m['step_vs_parallel_max_diff']:.1e} |")


if __name__ == "__main__":
    main()
