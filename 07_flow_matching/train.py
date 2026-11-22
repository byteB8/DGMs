"""Train a flow-matching velocity field on a 2-D toy set or on Fashion-MNIST.

    python 07_flow_matching/train.py --data 8gauss
    python 07_flow_matching/train.py --data 8gauss --reflow-from 07_flow_matching/runs/8gauss_s0
    python 07_flow_matching/train.py --data fmnist --cond

Reflow: draw pairs (x0, x1 = ODE(x0)) from a trained model and train a new model on those fixed pairs.
The new paths cross less, so they are straighter and need fewer steps.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.data import infinite_batches, load_images  # noqa: E402
from common.nets import EMA  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from common.viz import plot_2d_panels, save_image_grid  # noqa: E402
from model import build  # noqa: E402
from objective import flow_matching_loss  # noqa: E402
from sample import draw, euler_sample  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="8gauss", choices=[*toy2d.SAMPLERS, "mnist", "fmnist"])
    p.add_argument("--cond", action="store_true", help="class labels with dropout (enables guidance)")
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--reflow-from", default=None, help="run folder of a trained toy model to draw pairs from")
    p.add_argument("--reflow-pairs", type=int, default=100000)
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    toy = args.data in toy2d.SAMPLERS
    defaults = dict(steps=20000 if toy else 30000, batch_size=512 if toy else 128, lr=1e-3 if toy else 2e-4,
                    eval_every=2000 if toy else 5000)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.reflow_from and not toy:
        p.error("reflow is set up for the toy sets only")
    if args.name is None:
        args.name = f"{args.data}{'_cond' if args.cond else ''}{'_reflow' if args.reflow_from else ''}_s{args.seed}"
    return args


@torch.no_grad()
def reflow_pairs(run: str, n: int, device):
    """(x0, x1) with x1 = ODE solution from x0 under a trained toy model (100 Euler steps)."""
    ckpt = torch.load(Path(run) / "ckpt.pt", map_location=device)
    teacher = build(ckpt["args"]["data"], False).to(device)
    teacher.load_state_dict(ckpt["ema"])
    teacher.eval()
    x0 = torch.randn(n, 2, device=device)
    return x0, euler_sample(teacher, x0, 100)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")
    toy = args.data in toy2d.SAMPLERS

    if args.reflow_from:
        x0_all, x1_all = reflow_pairs(args.reflow_from, args.reflow_pairs, device)

        def batch():
            idx = torch.randint(0, len(x0_all), (args.batch_size,), device=device)
            return x1_all[idx], None, x0_all[idx]
    elif toy:
        def batch():
            return toy2d.sample(args.data, args.batch_size, device), None, None
    else:
        x, y = load_images(args.data, train=True, device=device)
        batches = infinite_batches(x, y, args.batch_size)

        def batch():
            xb, yb = next(batches)
            if not args.cond:
                return xb, None, None
            drop = torch.rand(len(yb), device=device) < args.p_uncond
            return xb, torch.where(drop, torch.full_like(yb, 10), yb), None

    model = build(args.data, args.cond, args.ch).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = Logger(run_dir)
    running = 0.0

    for step in range(1, args.steps + 1):
        x1, y, x0 = batch()
        loss = flow_matching_loss(model, x1, y, x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        running += loss.item()

        if step % args.eval_every == 0 or step == args.steps:
            logger.log(step, loss=running / args.eval_every)
            running = 0.0
            torch.manual_seed(step)
            if toy:
                s = draw(ema.model, 2000, (2,), device, 100)
                plot_2d_panels([(f"step {step}", s)], run_dir / f"samples_{step:06d}.png",
                               real=toy2d.sample(args.data, 2000))
            else:
                y_show = torch.arange(64, device=device) % 10 if args.cond else None
                s = draw(ema.model, 64, (1, 28, 28), device, 50, y_show, 0.0, 10)
                save_image_grid(s.clamp(-1, 1), run_dir / f"samples_{step:06d}.png")
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "args": vars(args)},
                       run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
