"""Sampling, evaluation and a causality check for the PixelCNN.

    python 04_autoregressive/sample.py --run 04_autoregressive/runs/<name>
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.utils import get_device, save_json  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import GatedPixelCNN  # noqa: E402
from objective import nll_per_image  # noqa: E402


@torch.no_grad()
def sample(model, n: int, device, size: int = 28):
    """Draw pixels one at a time in raster order: x_i ~ Bernoulli(sigmoid(logit_i(x_<i))).
    Needs size * size forward passes."""
    x = torch.zeros(n, 1, size, size, device=device)
    for i in range(size):
        for j in range(size):
            logits = model(x)[:, :, i, j]
            x[:, :, i, j] = torch.bernoulli(torch.sigmoid(logits))
    return x


@torch.no_grad()
def evaluate(model, x, batch: int = 1000) -> dict:
    """Exact test NLL in nats per image and in bits per pixel."""
    total = sum(nll_per_image(model(x[i:i + batch]), x[i:i + batch]).sum().item() for i in range(0, len(x), batch))
    nll = total / len(x)
    return {"nll": nll, "bits_per_pixel": nll / (x[0].numel() * torch.log(torch.tensor(2.0)).item())}


def dependency_map(model, i: int, j: int, device, size: int = 28):
    """|d logit_ij / d x| over all input pixels: which pixels output (i, j) depends on.
    For a correct model it is zero at (i, j) and at every pixel after it."""
    x = torch.rand(1, 1, size, size, device=device, requires_grad=True)
    model(x)[0, 0, i, j].backward()
    return x.grad[0, 0].abs()


def check_causality(model, device, size: int = 28) -> bool:
    """True if no output pixel depends on itself or on a later pixel (tested at a few positions)."""
    for i, j in [(0, 0), (5, 20), (14, 14), (27, 27)]:
        grad = dependency_map(model, i, j, device, size).flatten()
        if grad[i * size + j:].abs().max() > 0:
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=64)
    args = p.parse_args()
    run = Path(args.run)
    device = get_device()
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    model = GatedPixelCNN(ckpt["args"]["ch"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    torch.manual_seed(0)
    start = time.time()
    x = sample(model, args.n, device)
    torch.cuda.synchronize() if device.type == "cuda" else None
    seconds = time.time() - start
    save_image_grid(x * 2 - 1, run / "samples.png")
    save_json({"sample_seconds": seconds, "n": args.n}, run / "sampling.json")
    print(f"{args.n} samples in {seconds:.1f} s")


if __name__ == "__main__":
    main()
