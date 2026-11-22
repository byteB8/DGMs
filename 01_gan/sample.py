"""Sample from a trained generator and score the samples.

    python 01_gan/sample.py --run 01_gan/runs/<name>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import toy2d  # noqa: E402
from common.classifier import load_classifier  # noqa: E402
from common.data import load_images  # noqa: E402
from common.metrics import classifier_outputs, image_metrics, mode_coverage  # noqa: E402
from common.utils import get_device, save_json  # noqa: E402
from common.viz import plot_2d_panels, save_image_grid  # noqa: E402
from model import build  # noqa: E402


@torch.no_grad()
def generate(G, n: int, z_dim: int, device, labels=None, batch: int = 2000):
    """Draw n samples x = G(z), z ~ N(0, I). labels (optional) fixes the class of each sample."""
    was_training = G.training
    G.eval()
    out = []
    for i in range(0, n, batch):
        z = torch.randn(min(batch, n - i), z_dim, device=device)
        y = None if labels is None else labels[i:i + len(z)]
        out.append(G(z, y))
    G.train(was_training)
    return torch.cat(out)


class Evaluator:
    """Scores a generator on its dataset; real-data statistics are computed once."""

    def __init__(self, data: str, device, n: int = 10000):
        self.data, self.device, self.n = data, device, n
        self.toy = data in toy2d.SAMPLERS
        if not self.toy:
            self.clf = load_classifier(data, device)
            x_test, _ = load_images(data, train=False, device=device)
            self.real_feats, _ = classifier_outputs(self.clf, x_test[:n])

    def __call__(self, G, z_dim: int, cond: bool):
        """Return (metrics dict, samples)."""
        labels = torch.arange(self.n, device=self.device) % 10 if cond else None
        x = generate(G, self.n, z_dim, self.device, labels)
        if self.toy:
            metrics = {}
            if self.data == "8gauss":
                metrics = mode_coverage(x, toy2d.eight_gaussian_centers(), std=0.05)
            return metrics, x
        return image_metrics(self.clf, x, self.real_feats, labels), x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run folder with ckpt.pt")
    p.add_argument("--n", type=int, default=10000)
    args = p.parse_args()

    run = Path(args.run)
    device = get_device()
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    cfg = ckpt["args"]
    G, _ = build(cfg["data"], cfg["z_dim"], cfg["cond"])
    G.load_state_dict(ckpt["G"])
    G.to(device)

    metrics, x = Evaluator(cfg["data"], device, args.n)(G, cfg["z_dim"], cfg["cond"])
    save_json(metrics, run / "metrics.json")
    print(metrics)
    if cfg["data"] in toy2d.SAMPLERS:
        plot_2d_panels([(cfg["name"], x[:5000])], run / "samples.png", real=toy2d.sample(cfg["data"], 5000))
    elif cfg["cond"]:
        # rows = samples, columns = classes 0..9
        labels = torch.arange(100, device=device) % 10
        save_image_grid(generate(G, 100, cfg["z_dim"], device, labels), run / "samples.png", nrow=10)
    else:
        save_image_grid(x[:64], run / "samples.png")


if __name__ == "__main__":
    main()
