"""Sampling codes from the prior, decoding them, and evaluation.

    python 08_vqvae_prior/sample.py --vqvae-run 08_vqvae_prior/runs/vqvae_s0 --prior-run 08_vqvae_prior/runs/prior_s0
"""
import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.classifier import load_classifier  # noqa: E402
from common.data import load_images  # noqa: E402
from common.metrics import classifier_outputs, image_metrics  # noqa: E402
from common.utils import get_device, save_json  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import VQVAE, PriorTransformer  # noqa: E402
from objective import perplexity, prior_nll  # noqa: E402


@torch.no_grad()
def sample_codes(prior, y, length: int = 49, temperature: float = 1.0):
    """c_t ~ softmax(logits_t / temperature), one code at a time in raster order. y: (B,) class labels."""
    codes = torch.zeros(len(y), 0, dtype=torch.long, device=y.device)
    for _ in range(length):
        logits = prior(codes, y)[:, -1] / temperature
        codes = torch.cat([codes, torch.multinomial(logits.softmax(-1), 1)], dim=1)
    return codes


@torch.no_grad()
def evaluate_vqvae(vqvae, x, batch: int = 1000) -> dict:
    """Reconstruction error on images in [-1, 1] and code usage."""
    mse, idx = 0.0, []
    for i in range(0, len(x), batch):
        x_hat, _, _, codes = vqvae(x[i:i + batch])
        mse += (x_hat.clamp(-1, 1) - x[i:i + batch]).pow(2).mean(dim=(1, 2, 3)).sum().item()
        idx.append(codes)
    mse /= len(x)
    idx = torch.cat(idx)
    n_codes = vqvae.quantizer.codebook.num_embeddings
    return {"mse": mse, "psnr": 10 * math.log10(4 / mse),  # peak-to-peak range is 2, so peak^2 = 4
            "perplexity": perplexity(idx, n_codes), "codes_used": int(idx.unique().numel())}


@torch.no_grad()
def evaluate_prior(prior, codes, y, batch: int = 1000) -> dict:
    """Test NLL of the codes, per code and per image."""
    total = sum(prior_nll(prior, codes[i:i + batch], y[i:i + batch]).item() * len(codes[i:i + batch])
                for i in range(0, len(codes), batch))
    nats = total / len(codes)
    return {"nll_per_code": nats, "bits_per_image": nats * codes.shape[1] / math.log(2)}


def load(vqvae_run: Path, prior_run: Path, device):
    v = torch.load(vqvae_run / "ckpt.pt", map_location=device)
    vqvae = VQVAE(v["args"]["n_codes"], v["args"]["dim"]).to(device)
    vqvae.load_state_dict(v["model"])
    p = torch.load(prior_run / "ckpt.pt", map_location=device)
    prior = PriorTransformer(v["args"]["n_codes"], d=p["args"]["d"], n_layers=p["args"]["layers"]).to(device)
    prior.load_state_dict(p["model"])
    return vqvae.eval(), prior.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vqvae-run", required=True)
    p.add_argument("--prior-run", required=True)
    p.add_argument("--n", type=int, default=2000)
    args = p.parse_args()
    device = get_device()
    vqvae, prior = load(Path(args.vqvae_run), Path(args.prior_run), device)

    clf = load_classifier("fmnist", device)
    x_test, y_test = load_images("fmnist", train=False, device=device)
    real_feats, _ = classifier_outputs(clf, x_test[:args.n])
    out = {}

    x_rec = torch.cat([vqvae(x_test[i:i + 1000])[0] for i in range(0, args.n, 1000)]).clamp(-1, 1)
    out["reconstructions"] = image_metrics(clf, x_rec, real_feats)

    labels = torch.arange(args.n, device=device) % 10
    torch.manual_seed(0)
    start = time.time()
    codes = torch.cat([sample_codes(prior, labels[i:i + 1000]) for i in range(0, args.n, 1000)])
    x_gen = torch.cat([vqvae.decode(codes[i:i + 1000]) for i in range(0, args.n, 1000)]).clamp(-1, 1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    out["samples"] = image_metrics(clf, x_gen, real_feats, labels)
    out["sample_seconds_per_1k"] = (time.time() - start) * 1000 / args.n

    run = Path(args.prior_run)
    save_json(out, run / "sample_metrics.json")
    grid_labels = torch.arange(100, device=device) % 10           # column j = class j
    torch.manual_seed(1)
    save_image_grid(vqvae.decode(sample_codes(prior, grid_labels)).clamp(-1, 1), run / "samples.png", nrow=10)
    pairs = torch.stack([x_test[:16], vqvae(x_test[:16])[0].clamp(-1, 1)]).flatten(0, 1)
    save_image_grid(pairs, run / "reconstructions.png", nrow=16)
    print(out)


if __name__ == "__main__":
    main()
