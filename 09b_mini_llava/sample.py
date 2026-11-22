"""Answering questions with a trained mini-LLaVA, and its evaluation.

    python 09b_mini_llava/sample.py --run 09b_mini_llava/runs/lora_s0
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from common import shapes  # noqa: E402
from common.utils import DATA_DIR, get_device, save_json  # noqa: E402
from model import Projector, add_lora, build_inputs, image_features, load_language_model, load_vision  # noqa: E402


def cached_features(split: str, device, n=None) -> torch.Tensor:
    """SigLIP features of the 32 x 32 shape images, computed once and stored in data/shapes/ (float16)."""
    path = DATA_DIR / "shapes" / f"siglip_{split}.pt"
    if not path.exists():
        data = shapes.load(split)
        k = n or len(data["images"])
        feats = image_features(load_vision(device), shapes.to_float(data["images"][:k]).to(device))
        torch.save(feats.half().cpu(), path)
    return torch.load(path)


def qa_items(scenes: list, seed: int = 0, per_scene=None) -> list:
    """(scene index, question, answer, type) for every question of every scene (or per_scene random ones)."""
    rng = random.Random(seed)
    items = []
    for i, scene in enumerate(scenes):
        qa = shapes.questions(scene, rng)
        if per_scene:
            qa = rng.sample(qa, min(per_scene, len(qa)))
        items += [(i, q, a, t) for q, a, t in qa]
    return items


@torch.no_grad()
def answer(lm, tok, projector, features, questions, batch: int = 256) -> list:
    """Greedy one-word answers: the most likely token after "Answer:". Every answer word is one token."""
    out = []
    for i in range(0, len(questions), batch):
        inputs, mask, _, pos = build_inputs(lm, tok, projector(features[i:i + batch].float()), questions[i:i + batch])
        logits = lm(inputs_embeds=inputs, attention_mask=mask).logits
        ids = logits[torch.arange(len(pos)), pos].argmax(-1)
        out += [tok.decode([t]).strip() for t in ids.tolist()]
    return out


def accuracy(predictions, items) -> dict:
    by_type = defaultdict(list)
    for pred, (_, _, ans, typ) in zip(predictions, items):
        by_type[typ].append(pred == ans)
    out = {f"acc_{t}": sum(v) / len(v) for t, v in sorted(by_type.items())}
    out["acc_all"] = sum(sum(v) for v in by_type.values()) / len(predictions)
    return out


def evaluate(lm, tok, projector, features, scenes, shuffle: bool = False) -> dict:
    """Accuracy per question type on all questions of the given scenes.
    shuffle=True pairs each question with another image's features (a check that the image is used)."""
    items = qa_items(scenes, seed=1)
    idx = torch.tensor([i for i, _, _, _ in items])
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(0))]
    preds = answer(lm, tok, projector, features[idx].to(projector.net[0].weight.device), [q for _, q, _, _ in items])
    return accuracy(preds, items)


def load_run(run: Path, device):
    """LM (with LoRA weights if the run trained them), tokenizer and projector."""
    ckpt = torch.load(Path(run) / "ckpt.pt", map_location=device)
    lm, tok = load_language_model(device)
    if ckpt["lora"] is not None:
        lm = add_lora(lm)
        lm.load_state_dict(ckpt["lora"], strict=False)
        lm.eval()
    projector = Projector().to(device)
    projector.load_state_dict(ckpt["projector"])
    return lm, tok, projector.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    args = p.parse_args()
    device = get_device()
    lm, tok, projector = load_run(Path(args.run), device)
    feats = cached_features("test", device)
    scenes = shapes.load("test")["scenes"]
    metrics = evaluate(lm, tok, projector, feats, scenes)
    metrics.update({f"shuffled_{k}": v for k, v in evaluate(lm, tok, projector, feats, scenes, shuffle=True).items()})
    save_json(metrics, Path(args.run) / "metrics.json")
    print({k: round(v, 3) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
