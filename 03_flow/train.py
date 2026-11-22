"""Fit a RealNVP flow to a 2-D toy set by maximum likelihood.

    python 03_flow/train.py --data 8gauss
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import RealNVP  # noqa: E402
from objective import nll  # noqa: E402
from sample import evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="8gauss", choices=list(toy2d.SAMPLERS))
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.name is None:
        args.name = f"{args.data}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")

    flow = RealNVP(args.layers, args.hidden).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    logger = Logger(run_dir)

    for step in range(1, args.steps + 1):
        loss = nll(flow, toy2d.sample(args.data, args.batch_size, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            logger.log(step, train_nll=loss.item(), **evaluate(flow, args.data, device))

    metrics = evaluate(flow, args.data, device, n=100000)
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": flow.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
