"""Evaluation of a trained CLIP: retrieval, zero-shot attribute recognition and attribute binding.

    python 09a_clip/sample.py --run 09a_clip/runs/clip_s0
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import get_device, save_json  # noqa: E402
from model import CLIP  # noqa: E402


@torch.no_grad()
def embed(model, images, captions, device, batch: int = 1000):
    img = torch.cat([model.embed_images(shapes.to_float(images[i:i + batch]).to(device))
                     for i in range(0, len(images), batch)])
    txt = torch.cat([model.embed_texts(shapes.tokenize(captions[i:i + batch]).to(device))
                     for i in range(0, len(captions), batch)])
    return img, txt


@torch.no_grad()
def retrieval(model, data, device, n: int = 1000) -> dict:
    """Text -> image and image -> text among n test pairs. Captions repeat, so a hit is any item with the
    correct caption (top-1 and top-5)."""
    caps = data["captions"][:n]
    img, txt = embed(model, data["images"][:n], caps, device)
    same = torch.tensor([[a == b for b in caps] for a in caps], device=device)       # same[i, j]: caption i == j
    out = {}
    for name, sim in [("text_to_image", txt @ img.t()), ("image_to_text", img @ txt.t())]:
        top = sim.topk(5, dim=1).indices
        hits = same.gather(1, top)
        out[f"{name}_top1"] = hits[:, 0].float().mean().item()
        out[f"{name}_top5"] = hits.any(1).float().mean().item()
    return out


@torch.no_grad()
def zero_shot(model, data, device) -> dict:
    """Single-object test images: match each image against every possible one-object caption and read the
    attributes off the best caption. Accuracy per attribute."""
    combos = list(itertools.product(shapes.SIZES, shapes.COLORS, shapes.SHAPES, shapes.POSITIONS))
    objects = [{"size": s, "color": c, "shape": sh, "position": p} for s, c, sh, p in combos]
    _, txt = embed(model, data["images"][:1], [shapes.describe(o) for o in objects], device)
    idx = [i for i, s in enumerate(data["scenes"]) if len(s) == 1]
    img, _ = embed(model, data["images"][idx], ["a"] * len(idx), device)
    best = (img @ txt.t()).argmax(1).tolist()
    out = {}
    for attr in ["shape", "color", "size", "position"]:
        correct = [objects[b][attr] == data["scenes"][i][0][attr] for b, i in zip(best, idx)]
        out[f"zero_shot_{attr}"] = sum(correct) / len(correct)
    return out


@torch.no_grad()
def binding(model, data, device) -> dict:
    """Two-object test scenes with two different colors: does the true caption score higher than the caption
    with the two colors swapped? (A bag-of-words model cannot tell them apart and scores 50 %.)
    Same test for swapping the two shapes."""
    out = {}
    for attr in ["color", "shape"]:
        idx = [i for i, s in enumerate(data["scenes"]) if len(s) == 2 and s[0][attr] != s[1][attr]]
        swapped = []
        for i in idx:
            a, b = [dict(o) for o in data["scenes"][i]]
            a[attr], b[attr] = b[attr], a[attr]
            swapped.append(shapes.caption([a, b]))
        img, txt_true = embed(model, data["images"][idx], [data["captions"][i] for i in idx], device)
        _, txt_swap = embed(model, data["images"][:1], swapped, device)
        out[f"binding_{attr}"] = ((img * txt_true).sum(1) > (img * txt_swap).sum(1)).float().mean().item()
    return out


def evaluate(model, data, device) -> dict:
    model.eval()
    return {**retrieval(model, data, device), **zero_shot(model, data, device), **binding(model, data, device)}


def load_clip(run: Path, device) -> CLIP:
    ckpt = torch.load(Path(run) / "ckpt.pt", map_location=device)
    model = CLIP(ckpt["args"]["d"], ckpt["args"]["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    device = get_device()
    metrics = evaluate(load_clip(Path(args.run), device), shapes.load("test"), device)
    save_json(metrics, Path(args.run) / "metrics.json")
    print({k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
