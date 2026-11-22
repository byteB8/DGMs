"""Samplers for a trained noise predictor, and the evaluation sweeps behind the README.

    python 06_ddpm/sample.py --run 06_ddpm/runs/<name>
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.classifier import load_classifier  # noqa: E402
from common.data import load_images  # noqa: E402
from common.metrics import classifier_outputs, image_metrics, mode_coverage  # noqa: E402
from common.utils import get_device, save_json  # noqa: E402
from model import build  # noqa: E402
from objective import NoiseSchedule  # noqa: E402


def predict_eps(model, x, i, y=None, w: float = 0.0, null=None):
    """Classifier-free guidance: eps = (1 + w) eps(x_t, t, y) - w eps(x_t, t, no label). w = 0 is plain conditional."""
    t = torch.full((len(x),), i, device=x.device, dtype=torch.long)
    if y is None or w == 0:
        return model(x, t, y)
    eps = model(torch.cat([x, x]), torch.cat([t, t]), torch.cat([y, torch.full_like(y, null)]))
    eps_cond, eps_uncond = eps.chunk(2)
    return (1 + w) * eps_cond - w * eps_uncond


@torch.no_grad()
def ddpm_sample(model, sched: NoiseSchedule, shape, device, y=None, w: float = 0.0, null=None):
    """Ancestral sampling, T steps:
        x_{t-1} = (x_t - beta_t / sqrt(1 - abar_t) * eps) / sqrt(alpha_t) + sigma_t z,   sigma_t^2 = beta_t
    """
    x = torch.randn(shape, device=device)
    for i in reversed(range(sched.T)):
        eps = predict_eps(model, x, i, y, w, null)
        x = (x - sched.betas[i] / (1 - sched.alpha_bars[i]).sqrt() * eps) / sched.alphas[i].sqrt()
        if i > 0:
            x = x + sched.betas[i].sqrt() * torch.randn_like(x)
    return x


@torch.no_grad()
def ddim_sample(model, sched: NoiseSchedule, shape, device, steps: int, y=None, w: float = 0.0, null=None,
                clip: bool = False):
    """Deterministic DDIM (eta = 0) on `steps` evenly spaced times t > s:
        x0_hat = (x_t - sqrt(1 - abar_t) eps) / sqrt(abar_t)
        x_s    = sqrt(abar_s) x0_hat + sqrt(1 - abar_s) eps
    With steps = T it uses every time step.
    """
    times = torch.linspace(sched.T - 1, 0, steps).round().long().tolist()
    x = torch.randn(shape, device=device)
    for k, i in enumerate(times):
        eps = predict_eps(model, x, i, y, w, null)
        ab = sched.alpha_bars[i]
        ab_prev = sched.alpha_bars[times[k + 1]] if k + 1 < len(times) else torch.tensor(1.0, device=device)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        if clip:
            x0 = x0.clamp(-1, 1)
            eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
    return x


def draw(model, sched, n, shape, device, sampler: str, steps: int, y=None, w=0.0, null=None, batch: int = 1000):
    """n samples in batches. sampler = "ddpm" (always T steps) or "ddim"."""
    out = []
    for k in range(0, n, batch):
        m = min(batch, n - k)
        yb = None if y is None else y[k:k + m]
        if sampler == "ddpm":
            out.append(ddpm_sample(model, sched, (m, *shape), device, yb, w, null))
        else:
            out.append(ddim_sample(model, sched, (m, *shape), device, steps, yb, w, null, clip=len(shape) > 1))
    return torch.cat(out)


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    out = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.time() - start


def sweep(run: Path, device, n=None):
    """Sampler sweep (DDPM vs DDIM steps) and, for conditional models, a guidance sweep."""
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    cfg = ckpt["args"]
    model = build(cfg["data"], cfg["cond"], cfg["ch"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    sched = NoiseSchedule(cfg["T"], device=device)
    toy = cfg["data"] in toy2d.SAMPLERS
    n = n or (5000 if toy else 2000)
    results = {"samplers": [], "guidance": []}
    samples = {}

    if toy:
        shape, null = (2,), None
        score = (lambda x: mode_coverage(x, toy2d.eight_gaussian_centers(), 0.05)) if cfg["data"] == "8gauss" \
            else (lambda x: {})
    else:
        shape, null = (1, 28, 28), 10 if cfg["cond"] else None
        clf = load_classifier(cfg["data"], device)
        x_test, _ = load_images(cfg["data"], train=False, device=device)
        real_feats, _ = classifier_outputs(clf, x_test[:n])

        def score(x, labels=None):
            return image_metrics(clf, x, real_feats, labels)

    # unconditional samples (for a conditional model: the "no label" input)
    y_none = torch.full((n,), null, device=device) if null is not None else None
    for sampler, steps in [("ddpm", sched.T), ("ddim", 10), ("ddim", 50), ("ddim", 100), ("ddim", sched.T)]:
        torch.manual_seed(0)
        x, seconds = timed(lambda: draw(model, sched, n, shape, device, sampler, steps, y_none))
        results["samplers"].append({"sampler": sampler, "steps": steps, "seconds_per_1k": seconds * 1000 / n,
                                    **score(x)})
        samples[f"{sampler}_{steps}"] = x[:3000 if toy else 100].cpu()
        print(results["samplers"][-1])

    if null is not None:  # classifier-free guidance, DDIM 50 steps, labels 0..9 in turn
        labels = torch.arange(n, device=device) % 10
        for w in [0.0, 0.5, 1.0, 2.0, 4.0]:
            torch.manual_seed(0)
            x = draw(model, sched, n, shape, device, "ddim", 50, labels, w, null)
            results["guidance"].append({"w": w, **score(x, labels)})
            samples[f"guided_w{w:g}"] = x[:100].cpu()
            print(results["guidance"][-1])

    save_json(results, run / "sweep.json")
    torch.save(samples, run / "sweep_samples.pt")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=None, help="samples per setting (default 5000 toy, 2000 images)")
    args = p.parse_args()
    sweep(Path(args.run), get_device(), args.n)


if __name__ == "__main__":
    main()
