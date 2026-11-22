"""Train an S4D model or a causal transformer on binarized MNIST read as 784-step sequences.

    python 05_ssm/train.py --kind ssm
    python 05_ssm/train.py --kind transformer
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.data import binarize, epoch_batches, load_images  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import SequenceModel  # noqa: E402
from objective import nll_per_sequence  # noqa: E402
from sample import evaluate, step_matches_parallel  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", default="ssm", choices=["ssm", "transformer"])
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.name is None:
        args.name = f"{args.kind}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")

    x, y = load_images("mnist", train=True, device=device)
    x_test, _ = load_images("mnist", train=False, device=device)
    x, x_test = binarize(x).flatten(1), binarize(x_test).flatten(1)

    model = SequenceModel(args.kind, args.d, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(x) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    logger = Logger(run_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, _ in epoch_batches(x, y, args.batch_size):
            loss = nll_per_sequence(model(xb), xb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(xb)
            n += len(xb)
        model.eval()
        logger.log(epoch, train_nll=total / n, **{f"test_{k}": v for k, v in evaluate(model, x_test).items()})

    metrics = evaluate(model, x_test)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    metrics["step_vs_parallel_max_diff"] = step_matches_parallel(model, x_test[:8])
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
