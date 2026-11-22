"""Train a text-to-image flow-matching model on shape scenes, conditioned on frozen 09a caption embeddings.

    python 09c_text2image/train.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.nets import EMA  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import TextToImage, load_clip  # noqa: E402
from objective import flow_matching_loss  # noqa: E402
from sample import generate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ch", type=int, default=64)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--p-uncond", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.name is None:
        args.name = f"t2i_s{args.seed}"
    return args


@torch.no_grad()
def caption_embeddings(clip, captions, device, batch: int = 2000):
    return torch.cat([clip.embed_texts(shapes.tokenize(captions[i:i + batch]).to(device))
                      for i in range(0, len(captions), batch)])


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")

    train, test = shapes.load("train"), shapes.load("test")
    images = shapes.to_float(train["images"]).to(device)
    clip = load_clip(device)
    cond = caption_embeddings(clip, train["captions"], device)
    show_cond = caption_embeddings(clip, test["captions"][:64], device)
    del clip

    model = TextToImage(cond.shape[1], args.ch).to(device)
    ema = EMA(model, args.ema)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = Logger(run_dir)
    running = 0.0
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(images), (args.batch_size,), device=device)
        loss = flow_matching_loss(model, images[idx], cond[idx], args.p_uncond)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        running += loss.item()
        if step % args.eval_every == 0 or step == args.steps:
            logger.log(step, loss=running / args.eval_every)
            running = 0.0
            torch.manual_seed(0)
            save_image_grid(generate(ema.model, show_cond, 50, 1.0), run_dir / f"samples_{step:06d}.png")
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "args": vars(args)},
                       run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
