"""Train a tiny CLIP on shape scenes and their captions.

    python 09a_clip/train.py
    python 09a_clip/train.py --hard-negatives
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import CLIP  # noqa: E402
from objective import clip_loss  # noqa: E402
from sample import evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hard-negatives", action="store_true", help="add attribute-swapped captions as negatives")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.name is None:
        args.name = f"clip{'_hardneg' if args.hard_negatives else ''}_s{args.seed}"
    return args


def swapped_caption(scene: list, rng: random.Random, fallback: str) -> str:
    """Caption with one attribute (color, shape or size) swapped between two objects that differ in it.
    Same words, different meaning. Scenes where no swap changes the caption return `fallback`."""
    options = [(i, j, attr) for i in range(len(scene)) for j in range(i + 1, len(scene))
               for attr in ("color", "shape", "size") if scene[i][attr] != scene[j][attr]]
    if not options:
        return fallback
    i, j, attr = rng.choice(options)
    new = [dict(o) for o in scene]
    new[i][attr], new[j][attr] = scene[j][attr], scene[i][attr]
    return shapes.caption(new)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")

    train, test = shapes.load("train"), shapes.load("test")
    images = shapes.to_float(train["images"]).to(device)
    tokens = shapes.tokenize(train["captions"]).to(device)
    negatives = None
    if args.hard_negatives:
        rng = random.Random(args.seed)
        n = len(train["captions"])
        negatives = shapes.tokenize([swapped_caption(s, rng, train["captions"][rng.randrange(n)])
                                     for s in train["scenes"]]).to(device)

    model = CLIP(args.d, args.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * (len(images) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    logger = Logger(run_dir)

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(images), device=device)
        total, n = 0.0, 0
        for i in range(0, len(images) - args.batch_size + 1, args.batch_size):
            idx = order[i:i + args.batch_size]
            neg = model.embed_texts(negatives[idx]) if negatives is not None else None
            loss = clip_loss(model.embed_images(images[idx]), model.embed_texts(tokens[idx]), model.log_scale, neg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                model.log_scale.clamp_(0, 4.6052)   # temperature >= 0.01, as in CLIP
            total += loss.item()
            n += 1
        logger.log(epoch, loss=total / n, temperature=float((-model.log_scale).exp()),
                   **evaluate(model, test, device))

    metrics = evaluate(model, test, device)
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
