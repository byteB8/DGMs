"""Train a Gated PixelCNN on binarized MNIST by exact maximum likelihood.

    python 04_autoregressive/train.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.data import binarize, epoch_batches, load_images  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import GatedPixelCNN  # noqa: E402
from objective import nll_per_image  # noqa: E402
from sample import check_causality, evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.name is None:
        args.name = f"pixelcnn_ch{args.ch}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")

    x, y = load_images("mnist", train=True, device=device)
    x_test, _ = load_images("mnist", train=False, device=device)
    x, x_test = binarize(x), binarize(x_test)

    model = GatedPixelCNN(args.ch).to(device)
    assert check_causality(model, device), "an output pixel sees itself or a later pixel"
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    logger = Logger(run_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, _ in epoch_batches(x, y, args.batch_size):
            loss = nll_per_image(model(xb), xb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
        sched.step()
        model.eval()
        logger.log(epoch, train_nll=total / n, **{f"test_{k}": v for k, v in evaluate(model, x_test).items()})

    metrics = evaluate(model, x_test)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
