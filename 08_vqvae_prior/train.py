"""Two stages on Fashion-MNIST: train the VQ-VAE, then a transformer prior on its codes.

    python 08_vqvae_prior/train.py --stage vqvae
    python 08_vqvae_prior/train.py --stage prior --vqvae-run 08_vqvae_prior/runs/vqvae_s0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common.data import epoch_batches, load_images  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, save_json, seed_everything  # noqa: E402
from model import VQVAE, PriorTransformer, count_params  # noqa: E402
from objective import perplexity, prior_nll, vqvae_loss  # noqa: E402
from sample import evaluate_prior, evaluate_vqvae  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="vqvae", choices=["vqvae", "prior"])
    p.add_argument("--vqvae-run", default=None, help="stage prior: folder of the trained VQ-VAE")
    p.add_argument("--n-codes", type=int, default=128)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    vq = args.stage == "vqvae"
    defaults = dict(epochs=20 if vq else 30, batch_size=128 if vq else 256, lr=2e-4 if vq else 3e-4)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not vq and args.vqvae_run is None:
        p.error("--stage prior needs --vqvae-run")
    if args.name is None:
        args.name = f"{args.stage}_s{args.seed}"
    return args


def train_vqvae(args, device, run_dir):
    x, y = load_images("fmnist", train=True, device=device)
    x_test, y_test = load_images("fmnist", train=False, device=device)
    model = VQVAE(args.n_codes, args.dim).to(device)
    with torch.no_grad():  # start the codebook at encoder outputs, so no code begins far from the data
        z = model.encoder(x[:1000]).permute(0, 2, 3, 1).reshape(-1, args.dim)
        model.quantizer.codebook.weight.copy_(z[torch.randperm(len(z), device=device)[:args.n_codes]])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    logger = Logger(run_dir)
    for epoch in range(1, args.epochs + 1):
        model.train()
        usage = torch.zeros(args.n_codes, device=device)
        for xb, _ in epoch_batches(x, y, args.batch_size):
            x_hat, z_e, e, idx = model(xb)
            loss, rec, cb, commit = vqvae_loss(xb, x_hat, z_e, e, args.beta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            usage += torch.bincount(idx.flatten(), minlength=args.n_codes)
        dead = (usage == 0).nonzero().flatten()
        if len(dead):  # restart unused codes at random encoder outputs from the last batch
            with torch.no_grad():
                z = z_e.permute(0, 2, 3, 1).reshape(-1, args.dim)
                model.quantizer.codebook.weight[dead] = z[torch.randint(0, len(z), (len(dead),), device=device)]
        model.eval()
        logger.log(epoch, rec=rec.item(), codebook=cb.item(), commit=commit.item(), restarted=len(dead),
                   train_perplexity=perplexity(idx, args.n_codes), **evaluate_vqvae(model, x_test))
    metrics = evaluate_vqvae(model, x_test)
    metrics["params"] = count_params(model)
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")
    # the prior trains on these codes
    codes = {"train": torch.cat([model.encode(x[i:i + 1000]) for i in range(0, len(x), 1000)]).flatten(1),
             "test": torch.cat([model.encode(x_test[i:i + 1000]) for i in range(0, len(x_test), 1000)]).flatten(1),
             "y_train": y, "y_test": y_test}
    torch.save({k: v.cpu() for k, v in codes.items()}, run_dir / "codes.pt")


def train_prior(args, device, run_dir):
    vq_args = torch.load(Path(args.vqvae_run) / "ckpt.pt", map_location="cpu")["args"]
    data = {k: v.to(device) for k, v in torch.load(Path(args.vqvae_run) / "codes.pt").items()}
    model = PriorTransformer(vq_args["n_codes"], d=args.d, n_layers=args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(data["train"]) // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    logger = Logger(run_dir)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for cb, yb in epoch_batches(data["train"], data["y_train"], args.batch_size):
            loss = prior_nll(model, cb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(cb)
            n += len(cb)
        model.eval()
        logger.log(epoch, train_nll_per_code=total / n,
                   **{f"test_{k}": v for k, v in evaluate_prior(model, data["test"], data["y_test"]).items()})
    metrics = evaluate_prior(model, data["test"], data["y_test"])
    metrics["params"] = count_params(model)
    save_json(metrics, run_dir / "metrics.json")
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")
    (train_vqvae if args.stage == "vqvae" else train_prior)(args, device, run_dir)


if __name__ == "__main__":
    main()
