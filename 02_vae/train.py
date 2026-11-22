"""Train a VAE on binarized MNIST.

    python 02_vae/train.py --z-dim 2 --beta 1
    python 02_vae/train.py --z-dim 16 --beta 1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.data import binarize, epoch_batches, load_images  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import VAE  # noqa: E402
from objective import negative_elbo  # noqa: E402
from sample import evaluate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--z-dim", type=int, default=2)
    p.add_argument("--beta", type=float, default=1.0, help="weight on the KL term (1 = ELBO)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    if args.name is None:
        args.name = f"z{args.z_dim}_beta{args.beta:g}"
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

    vae = VAE(args.z_dim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=args.lr)
    logger = Logger(run_dir)

    for epoch in range(1, args.epochs + 1):
        totals = torch.zeros(3, device=device)
        n_batches = 0
        for xb, _ in epoch_batches(x, y, args.batch_size):
            logits, mu, logvar = vae(xb)
            loss, rec, kl = negative_elbo(xb, logits, mu, logvar, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals += torch.stack([loss.detach(), rec.detach(), kl.detach()])
            n_batches += 1
        loss, rec, kl = (totals / n_batches).tolist()
        test = evaluate(vae, x_test, k=0)  # ELBO terms only; the costly log p(x) estimate runs at the end
        logger.log(epoch, train_loss=loss, train_rec=rec, train_kl=kl, **test)

    test = evaluate(vae, x_test, k=500)
    save_json(test, run_dir / "metrics.json")
    print({k: round(v, 2) for k, v in test.items()})
    torch.save({"model": vae.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")


if __name__ == "__main__":
    main()
