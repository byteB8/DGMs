"""Train a GAN or WGAN-GP on a 2-D toy set or on (Fashion-)MNIST.

    python 01_gan/train.py --data 8gauss --loss ns
    python 01_gan/train.py --data 8gauss --loss wgan-gp
    python 01_gan/train.py --data fmnist --loss ns
    python 01_gan/train.py --data fmnist --loss ns --cond
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import objective as obj  # noqa: E402
from common import toy2d  # noqa: E402
from common.data import infinite_batches, load_images  # noqa: E402
from common.utils import Logger, get_device, make_run_dir, read_log, save_json, seed_everything  # noqa: E402
from common.viz import plot_2d_panels, plot_curves, save_image_grid  # noqa: E402
from model import build  # noqa: E402
from sample import Evaluator  # noqa: E402

G_LOSSES = {"minimax": obj.g_loss_minimax, "ns": obj.g_loss_nonsaturating, "wgan-gp": obj.g_loss_wgan}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="8gauss", choices=[*toy2d.SAMPLERS, "mnist", "fmnist"])
    p.add_argument("--loss", default="ns", choices=list(G_LOSSES))
    p.add_argument("--cond", action="store_true", help="class-conditional (image data only)")
    p.add_argument("--steps", type=int, default=None, help="generator updates")
    p.add_argument("--n-critic", type=int, default=None, help="critic updates per generator update")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gp-weight", type=float, default=10.0)
    p.add_argument("--z-dim", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    args = p.parse_args()

    toy = args.data in toy2d.SAMPLERS
    defaults = dict(steps=20000 if toy else 15000, batch_size=256 if toy else 128, lr=1e-4 if toy else 2e-4,
                    z_dim=2 if toy else 64, eval_every=500 if toy else 1000,
                    n_critic=5 if args.loss == "wgan-gp" else 1)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.cond and toy:
        p.error("--cond needs an image dataset")
    if args.name is None:
        args.name = f"{args.data}_{args.loss}{'_cond' if args.cond else ''}_s{args.seed}"
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    run_dir = make_run_dir(__file__, args.name)
    save_json(vars(args), run_dir / "args.json")
    toy = args.data in toy2d.SAMPLERS

    if toy:
        def real_batch():
            return toy2d.sample(args.data, args.batch_size, device), None
    else:
        x, y = load_images(args.data, train=True, device=device)
        batches = infinite_batches(x, y, args.batch_size)

        def real_batch():
            xb, yb = next(batches)
            return xb, (yb if args.cond else None)

    G, D = build(args.data, args.z_dim, args.cond)
    G, D = G.to(device), D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))
    g_loss_fn = G_LOSSES[args.loss]
    evaluate = Evaluator(args.data, device)
    logger = Logger(run_dir)
    snapshots = []

    for step in range(1, args.steps + 1):
        # critic / discriminator updates
        for _ in range(args.n_critic):
            real, y = real_batch()
            with torch.no_grad():
                fake = G(torch.randn(len(real), args.z_dim, device=device), y)
            real_s, fake_s = D(real, y), D(fake, y)
            if args.loss == "wgan-gp":
                d_loss = obj.critic_loss_wgan(real_s, fake_s)
                d_total = d_loss + args.gp_weight * obj.gradient_penalty(D, real, fake, y)
            else:
                d_loss = d_total = obj.d_loss_gan(real_s, fake_s)
            opt_d.zero_grad(set_to_none=True)
            d_total.backward()
            opt_d.step()

        # generator update
        y_g = torch.randint(0, 10, (args.batch_size,), device=device) if args.cond else None
        fake = G(torch.randn(args.batch_size, args.z_dim, device=device), y_g)
        g_loss = g_loss_fn(D(fake, y_g))
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        g_grad = torch.nn.utils.clip_grad_norm_(G.parameters(), float("inf")).item()  # norm only, no clipping
        opt_g.step()

        if step % args.eval_every == 0 or step == args.steps:
            metrics, samples = evaluate(G, args.z_dim, args.cond)
            # for wgan-gp, -d_loss = E_p f - E_q f is the critic's estimate of W1(p, q)
            logger.log(step, d_loss=d_loss.item(), g_loss=g_loss.item(), g_grad=g_grad, **metrics)
            snapshots.append((step, samples[:3000].cpu()))
            if not toy:
                save_image_grid(samples[:64], run_dir / f"samples_{step:06d}.png")

    torch.save({"G": G.state_dict(), "D": D.state_dict(), "args": vars(args)}, run_dir / "ckpt.pt")

    if toy:  # samples at six points of training
        picks = [snapshots[round(i * (len(snapshots) - 1) / 5)] for i in range(6)]
        plot_2d_panels([(f"step {s}", x) for s, x in picks], run_dir / "training.png",
                       real=toy2d.sample(args.data, 3000))
    records = read_log(run_dir)
    plot_curves({"g_grad": ([r["step"] for r in records], [r["g_grad"] for r in records])},
                run_dir / "g_grad.png", ylabel="generator gradient norm", logy=True)


if __name__ == "__main__":
    main()
