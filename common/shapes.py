"""Synthetic shape scenes with captions and question-answer pairs (used by 09a, 09b, 09c).

A scene is a list of 1-3 objects, each a dict with shape, color, size and position (left / middle / right);
no two objects share a position. Scenes are plain data; render() draws them at any resolution.

    a small red circle on the left and a large blue square on the right
    Q: What color is the square?   A: blue
"""
import random

import numpy as np
import torch
from PIL import Image, ImageDraw

from .utils import DATA_DIR

SHAPES = ["circle", "square", "triangle"]
COLORS = {"red": (230, 60, 50), "green": (60, 190, 80), "blue": (60, 110, 235), "yellow": (240, 210, 50)}
SIZES = {"small": 0.08, "large": 0.14}            # half-width as a fraction of the image side
POSITIONS = {"left": 1 / 6, "middle": 1 / 2, "right": 5 / 6}
PHRASE = {"left": "on the left", "middle": "in the middle", "right": "on the right"}
NUMBERS = {1: "one", 2: "two", 3: "three"}


def random_scene(rng: random.Random) -> list:
    positions = rng.sample(list(POSITIONS), rng.randint(1, 3))
    scene = [{"shape": rng.choice(SHAPES), "color": rng.choice(list(COLORS)), "size": rng.choice(list(SIZES)),
              "position": p, "dx": rng.uniform(-0.03, 0.03), "y": rng.uniform(0.3, 0.7)} for p in positions]
    return sorted(scene, key=lambda o: POSITIONS[o["position"]])


def render(scene: list, size: int = 32, supersample: int = 4) -> np.ndarray:
    """Draw a scene on a black background. Returns uint8 (size, size, 3). Drawn large, then downsampled."""
    big = size * supersample
    img = Image.new("RGB", (big, big))
    draw = ImageDraw.Draw(img)
    for o in scene:
        cx, cy, r = (POSITIONS[o["position"]] + o["dx"]) * big, o["y"] * big, SIZES[o["size"]] * big
        color = COLORS[o["color"]]
        if o["shape"] == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        elif o["shape"] == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
        else:
            draw.polygon([(cx, cy - r), (cx - r, cy + r * 0.8), (cx + r, cy + r * 0.8)], fill=color)
    return np.asarray(img.resize((size, size), Image.LANCZOS))


def describe(o: dict) -> str:
    return f"a {o['size']} {o['color']} {o['shape']} {PHRASE[o['position']]}"


def caption(scene: list) -> str:
    parts = [describe(o) for o in scene]
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


def questions(scene: list, rng: random.Random) -> list:
    """(question, one-word answer, type) triples about a scene."""
    qa = [("How many shapes are there?", NUMBERS[len(scene)], "count")]
    for o in scene:
        qa.append((f"What shape is {PHRASE[o['position']]}?", o["shape"], "shape"))
        qa.append((f"What color is the shape {PHRASE[o['position']]}?", o["color"], "color"))
        qa.append((f"Is the shape {PHRASE[o['position']]} small or large?", o["size"], "size"))
    # yes / no: a real object half of the time, otherwise one that is not in the scene
    present = {(o["color"], o["shape"]) for o in scene}
    if rng.random() < 0.5:
        color, shape = rng.choice(sorted(present))
    else:
        absent = [(c, s) for c in COLORS for s in SHAPES if (c, s) not in present]
        color, shape = rng.choice(absent)
    qa.append((f"Is there a {color} {shape}?", "yes" if (color, shape) in present else "no", "exists"))
    return qa


def make_split(split: str, n: int, size: int = 32, seed: int = 0) -> dict:
    """Scenes, images (uint8, N x 3 x size x size) and captions, cached in data/shapes/<split>_<size>.pt."""
    path = DATA_DIR / "shapes" / f"{split}_{size}.pt"
    if path.exists():
        return torch.load(path)
    rng = random.Random(f"{split}-{seed}")
    scenes = [random_scene(rng) for _ in range(n)]
    images = torch.from_numpy(np.stack([render(s, size) for s in scenes])).permute(0, 3, 1, 2).contiguous()
    data = {"scenes": scenes, "images": images, "captions": [caption(s) for s in scenes]}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def load(split: str = "train", size: int = 32) -> dict:
    """50k training scenes or 5k test scenes."""
    return make_split(split, 50000 if split == "train" else 5000, size)


def to_float(images: torch.Tensor) -> torch.Tensor:
    """uint8 images -> float in [-1, 1]."""
    return images.float().div(127.5).sub(1)


# ------------------------------------------------------------------------------------- tokenizer
VOCAB = ["<pad>", "a", *SIZES, *COLORS, *SHAPES, "on", "in", "the", "left", "middle", "right", "and", ","]
WORD_ID = {w: i for i, w in enumerate(VOCAB)}


def tokenize(texts: list, max_len: int = 32) -> torch.Tensor:
    """Word-level token ids, (B, max_len), 0 = padding. Commas are separate tokens."""
    out = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, text in enumerate(texts):
        ids = [WORD_ID[w] for w in text.replace(",", " ,").split()][:max_len]
        out[i, :len(ids)] = torch.tensor(ids)
    return out
