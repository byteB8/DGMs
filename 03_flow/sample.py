"""Sampling, density maps and evaluation for a trained flow.

    python 03_flow/sample.py --run 03_flow/runs/<name>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.utils import get_device, save_json  # noqa: E402
from model import RealNVP  # noqa: E402
from objective import log_prob  # noqa: E402


@torch.no_grad()
def sample(flow, n: int, device, return_all: bool = False):
    """x = f^{-1}(z), z ~ N(0, I)."""
    return flow.inverse(torch.randn(n, 2, device=device), return_all=return_all)


@torch.no_grad()
def density_grid(flow, device, lim: float = 3.0, res: int = 300):
    """q(x) on a res x res grid over [-lim, lim]^2. Row 0 is the bottom (smallest y)."""
    ticks = torch.linspace(-lim, lim, res, device=device)
    yy, xx = torch.meshgrid(ticks, ticks, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1).view(-1, 2)
    return log_prob(flow, grid).exp().view(res, res)


@torch.no_grad()
def evaluate(flow, data: str, device, n: int = 20000) -> dict:
    """Test NLL on fresh samples of the toy set. For 8 Gaussians the true density is known, so we also
    report the true NLL (the entropy of p) and KL(p || q) = E_p[log p(x) - log q(x)]."""
    x = toy2d.sample(data, n, device)
    log_q = log_prob(flow, x)
    out = {"nll": -log_q.mean().item()}
    if data == "8gauss":
        log_p = toy2d.eight_gaussians_log_prob(x)
        out["true_nll"] = -log_p.mean().item()
        out["kl"] = (log_p - log_q).mean().item()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    run = Path(args.run)
    device = get_device()
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    flow = RealNVP(ckpt["args"]["layers"], ckpt["args"]["hidden"]).to(device)
    flow.load_state_dict(ckpt["model"])
    metrics = evaluate(flow, ckpt["args"]["data"], device)
    save_json(metrics, run / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
