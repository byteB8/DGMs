"""Generate images from captions, and check them with the 09b VLM and the 09a CLIP model.

    python 09c_text2image/sample.py --run 09c_text2image/runs/t2i_s0
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import get_device, save_json  # noqa: E402
from common.viz import save_image_grid  # noqa: E402
from model import TextToImage, load_clip, load_llava  # noqa: E402


@torch.no_grad()
def generate(model, cond, steps: int = 50, w: float = 0.0, batch: int = 500):
    """Euler steps for dx/dt = (1 + w) v(x, t, c) - w v(x, t, no caption), from noise at t = 0 to t = 1."""
    out = []
    for i in range(0, len(cond), batch):
        c = cond[i:i + batch]
        x = torch.randn(len(c), 3, 32, 32, device=c.device)
        for k in range(steps):
            t = torch.full((len(c),), k / steps, device=c.device)
            v = model(x, t, c)
            if w:
                v = (1 + w) * v - w * model(x, t, None)
            x = x + v / steps
        out.append(x.clamp(-1, 1))
    return torch.cat(out)


@torch.no_grad()
def vlm_check(checker, images, scenes) -> dict:
    """Ask the 09b model every question about the scene each image should show.
    Returns accuracy per question type and the share of images with every answer right."""
    llava, lm, tok, projector, vision = checker
    feats = llava.image_features(vision, images)
    rng = random.Random(0)
    items = [(i, q, a, t) for i, s in enumerate(scenes) for q, a, t in shapes.questions(s, rng)]
    preds = []
    for k in range(0, len(items), 256):
        chunk = items[k:k + 256]
        idx = torch.tensor([i for i, _, _, _ in chunk], device=images.device)
        inputs, mask, _, pos = llava.build_inputs(lm, tok, projector(feats[idx].float()), [q for _, q, _, _ in chunk])
        logits = lm(inputs_embeds=inputs, attention_mask=mask).logits
        preds += [tok.decode([t]).strip() for t in logits[torch.arange(len(pos)), pos].argmax(-1).tolist()]
    by_type, per_image = defaultdict(list), defaultdict(list)
    for p, (i, _, a, t) in zip(preds, items):
        by_type[t].append(p == a)
        per_image[i].append(p == a)
    out = {f"vlm_{t}": sum(v) / len(v) for t, v in sorted(by_type.items())}
    out["vlm_all"] = sum(sum(v) for v in by_type.values()) / len(items)
    out["vlm_image_fully_correct"] = sum(all(v) for v in per_image.values()) / len(per_image)
    return out


@torch.no_grad()
def clip_check(clip, images, captions) -> dict:
    """Cosine between each image and its caption (CLIP score), and caption -> image retrieval top-1
    among the generated images (a hit is any image generated from the same caption)."""
    img = clip.embed_images(images)
    txt = clip.embed_texts(shapes.tokenize(captions).to(images.device))
    best = (txt @ img.t()).argmax(1).tolist()
    return {"clip_score": (img * txt).sum(1).mean().item(),
            "clip_retrieval_top1": sum(captions[b] == captions[i] for i, b in enumerate(best)) / len(captions)}


def check(clip, checker, images, scenes, captions, subsets: dict) -> dict:
    """All checks on every subset of indices (e.g. captions seen / not seen in training)."""
    out = {}
    for name, idx in subsets.items():
        r = {**vlm_check(checker, images[idx], [scenes[i] for i in idx]),
             **clip_check(clip, images[idx], [captions[i] for i in idx])}
        out.update({f"{name}_{k}": v for k, v in r.items()})
    return out


def evaluate(model, clip, checker, scenes, captions, cond, subsets, guidance=(0.0, 1.0, 3.0), steps: int = 50):
    results, samples = [], {}
    for w in guidance:
        torch.manual_seed(0)
        x = generate(model, cond, steps, w)
        results.append({"w": w, **check(clip, checker, x, scenes, captions, subsets)})
        samples[f"w{w:g}"] = x[:64].cpu()
        print({k: round(v, 3) for k, v in results[-1].items() if k.startswith(("all_vlm_all", "all_vlm_image", "unseen"))})
    return results, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=2000, help="test captions to generate from")
    p.add_argument("--clip-run", default="09a_clip/runs/clip_hardneg_s0")
    p.add_argument("--llava-run", default="09b_mini_llava/runs/lora_s0")
    args = p.parse_args()
    run = Path(args.run)
    device = get_device()
    ckpt = torch.load(run / "ckpt.pt", map_location=device)
    model = TextToImage(ch=ckpt["args"]["ch"]).to(device)
    model.load_state_dict(ckpt["ema"])
    model.eval()
    clip, checker = load_clip(device, args.clip_run), load_llava(device, args.llava_run)

    test = shapes.load("test")
    scenes, captions = test["scenes"][:args.n], test["captions"][:args.n]
    cond = clip.embed_texts(shapes.tokenize(captions).to(device))
    seen = set(shapes.load("train")["captions"])
    subsets = {"all": list(range(len(captions))),
               "seen": [i for i, c in enumerate(captions) if c in seen],
               "unseen": [i for i, c in enumerate(captions) if c not in seen]}
    real = check(clip, checker, shapes.to_float(test["images"][:args.n]).to(device), scenes, captions, subsets)
    print("real images:", {k: round(v, 3) for k, v in real.items() if k.startswith(("all_vlm", "unseen_vlm"))})
    results, samples = evaluate(model, clip, checker, scenes, captions, cond, subsets)
    save_json({"subset_sizes": {k: len(v) for k, v in subsets.items()}, "real": real, "generated": results},
              run / "eval.json")
    torch.save({"samples": samples, "captions": captions[:64]}, run / "eval_samples.pt")
    save_image_grid(samples["w1"], run / "samples_w1.png")


if __name__ == "__main__":
    main()
