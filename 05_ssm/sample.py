"""Sampling, evaluation and mode checks for the sequence models.

    python 05_ssm/sample.py --run 05_ssm/runs/<name>
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.utils import get_device, save_json  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import START, SequenceModel  # noqa: E402
from objective import nll_per_sequence  # noqa: E402


@torch.no_grad()
def sample(model, n: int, device, length: int = 784):
    """Step mode: x_t ~ Bernoulli(sigmoid(logit_t)), one step at a time, carrying SSM states or the KV cache."""
    states = model.initial_states(n, device)
    token = torch.full((n,), START, device=device)
    out = torch.zeros(n, length, device=device)
    for t in range(length):
        logit, states = model.step(token, t, states)
        token = torch.bernoulli(torch.sigmoid(logit))
        out[:, t] = token
    return out


@torch.no_grad()
def evaluate(model, x, batch: int = 500) -> dict:
    """Exact test NLL in nats per image and bits per pixel. x: (N, 784)."""
    total = sum(nll_per_sequence(model(x[i:i + batch]), x[i:i + batch]).sum().item() for i in range(0, len(x), batch))
    nll = total / len(x)
    return {"nll": nll, "bits_per_pixel": nll / (x.shape[1] * math.log(2))}


@torch.no_grad()
def step_matches_parallel(model, x) -> float:
    """Largest gap between logits from parallel mode and from step mode on the same sequences.
    For the SSM this compares the FFT convolution with the recurrence."""
    parallel = model(x)
    states = model.initial_states(len(x), x.device)
    token = torch.full((len(x),), START, device=x.device)
    stepped = []
    for t in range(x.shape[1]):
        logit, states = model.step(token, t, states)
        stepped.append(logit)
        token = x[:, t]
    return (parallel - torch.stack(stepped, dim=1)).abs().max().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()
    run = Path(args.run)
    device = get_device()
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    model = SequenceModel(ckpt["args"]["kind"], ckpt["args"]["d"], ckpt["args"]["layers"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    torch.manual_seed(0)
    start = time.time()
    x = sample(model, args.n, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    save_image_grid(x.view(-1, 1, 28, 28) * 2 - 1, run / "samples.png")
    save_json({"sample_seconds": seconds, "n": args.n}, run / "sampling.json")
    print(f"{args.n} samples in {seconds:.1f} s")


if __name__ == "__main__":
    main()
