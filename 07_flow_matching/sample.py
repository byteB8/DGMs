"""ODE sampling for a trained velocity field, and the evaluation sweeps behind the README.

    python 07_flow_matching/sample.py --run 07_flow_matching/runs/<name>
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


def velocity(model, x, t: float, y=None, w: float = 0.0, null=None):
    """Classifier-free guidance on the velocity: v = (1 + w) v(x, t, y) - w v(x, t, no label)."""
    tt = torch.full((len(x),), t, device=x.device)
    if y is None or w == 0:
        return model(x, tt, y)
    v = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([y, torch.full_like(y, null)]))
    v_cond, v_uncond = v.chunk(2)
    return (1 + w) * v_cond - w * v_uncond


@torch.no_grad()
def euler_sample(model, x0, steps: int, y=None, w: float = 0.0, null=None, return_path: bool = False):
    """Solve dx/dt = v(x, t) from t = 0 (noise x0) to t = 1 with `steps` Euler steps: x <- x + v(x, t) / steps."""
    x, path = x0, [x0]
    for k in range(steps):
        x = x + velocity(model, x, k / steps, y, w, null) / steps
        if return_path:
            path.append(x)
    return path if return_path else x


def draw(model, n, shape, device, steps: int, y=None, w=0.0, null=None, batch: int = 1000):
    out = []
    for k in range(0, n, batch):
        m = min(batch, n - k)
        yb = None if y is None else y[k:k + m]
        out.append(euler_sample(model, torch.randn(m, *shape, device=device), steps, yb, w, null))
    return torch.cat(out)


@torch.no_grad()
def straightness(model, shape, device, n: int = 1000, fine: int = 100, null=None) -> float:
    """Mean distance between the 1-step and the `fine`-step endpoints from the same noise
    (unconditional, i.e. "no label" for a conditional model).
    It is 0 when every path is a straight line travelled at constant speed."""
    x0 = torch.randn(n, *shape, device=device)
    y = None if null is None else torch.full((n,), null, device=device)
    one = euler_sample(model, x0, 1, y, 0.0, null)
    many = euler_sample(model, x0, fine, y, 0.0, null)
    return (one - many).flatten(1).norm(dim=1).mean().item()


def sweep(run: Path, device, n=None):
    """Quality vs number of Euler steps and, for conditional models, a guidance sweep."""
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    cfg = ckpt["args"]
    model = build(cfg["data"], cfg["cond"], cfg["ch"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    toy = cfg["data"] in toy2d.SAMPLERS
    n = n or (5000 if toy else 2000)
    results = {"steps": [], "guidance": []}
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
            return image_metrics(clf, x.clamp(-1, 1), real_feats, labels)

    y_none = torch.full((n,), null, device=device) if null is not None else None
    results["straightness"] = straightness(model, shape, device, null=null)
    for steps in [1, 2, 5, 10, 20, 50, 100]:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.time()
        x = draw(model, n, shape, device, steps, y_none)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        results["steps"].append({"steps": steps, "seconds_per_1k": (time.time() - start) * 1000 / n, **score(x)})
        samples[f"euler_{steps}"] = x[:3000 if toy else 100].cpu()
        print(results["steps"][-1])

    if null is not None:  # guidance with 50 Euler steps, labels 0..9 in turn
        labels = torch.arange(n, device=device) % 10
        for w in [0.0, 0.5, 1.0, 2.0, 4.0]:
            torch.manual_seed(0)
            x = draw(model, n, shape, device, 50, labels, w, null)
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
