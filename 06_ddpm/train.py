"""Train a DDPM noise predictor on a 2-D toy set or on Fashion-MNIST.

    python 06_ddpm/train.py --data swissroll
    python 06_ddpm/train.py --data fmnist --cond
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
from objective import NoiseSchedule, ddpm_loss  # noqa: E402
from sample import draw  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="swissroll", choices=[*toy2d.SAMPLERS, "mnist", "fmnist"])
    p.add_argument("--cond", action="store_true", help="class labels with dropout (enables guidance)")
    p.add_argument("--p-uncond", type=float, default=0.1, help="share of labels replaced by 'no label'")
    p.add_argument("--T", type=int, default=1000)
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
    if args.name is None:
        args.name = f"{args.data}{'_cond' if args.cond else ''}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")
    toy = args.data in toy2d.SAMPLERS

    if toy:
        def batch():
            return toy2d.sample(args.data, args.batch_size, device), None
    else:
        x, y = load_images(args.data, train=True, device=device)
        batches = infinite_batches(x, y, args.batch_size)

        def batch():
            xb, yb = next(batches)
            if not args.cond:
                return xb, None
            drop = torch.rand(len(yb), device=device) < args.p_uncond
            return xb, torch.where(drop, torch.full_like(yb, 10), yb)

    model = build(args.data, args.cond, args.ch).to(device)
    ema = EMA(model, args.ema)
    sched = NoiseSchedule(args.T, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = Logger(run_dir)
    running = 0.0

    for step in range(1, args.steps + 1):
        x0, y = batch()
        loss = ddpm_loss(model, sched, x0, y)
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
                s = draw(ema.model, sched, 2000, (2,), device, "ddim", 100)
                plot_2d_panels([(f"step {step}", s)], run_dir / f"samples_{step:06d}.png",
                               real=toy2d.sample(args.data, 2000))
            else:
                y_show = torch.arange(64, device=device) % 10 if args.cond else None
                s = draw(ema.model, sched, 64, (1, 28, 28), device, "ddim", 50, y_show, 0.0, 10)
                save_image_grid(s, run_dir / f"samples_{step:06d}.png")
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "args": vars(args)},
                       run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
