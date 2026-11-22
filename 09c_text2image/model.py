"""Text-conditioned velocity network, plus loaders for the 09a CLIP model and the 09b mini-LLaVA."""
import importlib.util
from pathlib import Path

import torch
from torch import nn

from common.nets import UNet

ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    """Import a project's model.py under its own name (every project has a model.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TextToImage(nn.Module):
    """v_theta(x_t, t, c): U-Net on 3 x 32 x 32 images, conditioned on a caption embedding c (128-d, from 09a).
    c = None means "no caption": a learned null embedding takes its place (for classifier-free guidance)."""

    def __init__(self, cond_dim: int = 128, ch: int = 64):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(cond_dim))
        self.unet = UNet(in_ch=3, ch=ch, cond_dim=cond_dim, time_scale=1000.0)

    def forward(self, x, t, cond=None, drop=None):
        """drop (B,) bool: replace those captions by the null embedding."""
        if cond is None:
            cond = self.null.expand(len(x), -1)
        elif drop is not None:
            cond = torch.where(drop[:, None], self.null.expand(len(x), -1), cond)
        return self.unet(x, t, cond)


def load_clip(device, run: str = "09a_clip/runs/clip_hardneg_s0"):
    clip = load_module(ROOT / "09a_clip" / "model.py", "clip_model")
    ckpt = torch.load(ROOT / run / "ckpt.pt", map_location=device)
    model = clip.CLIP(ckpt["args"]["d"], ckpt["args"]["embed_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval().requires_grad_(False)


def load_llava(device, run: str = "09b_mini_llava/runs/lora_s0"):
    """Returns (module, lm, tokenizer, projector, vision) for the trained 09b model."""
    llava = load_module(ROOT / "09b_mini_llava" / "model.py", "llava_model")
    ckpt = torch.load(ROOT / run / "ckpt.pt", map_location=device)
    lm, tok = llava.load_language_model(device)
    if ckpt["lora"] is not None:
        lm = llava.add_lora(lm)
        lm.load_state_dict(ckpt["lora"], strict=False)
    projector = llava.Projector().to(device)
    projector.load_state_dict(ckpt["projector"])
    return llava, lm.eval(), tok, projector.eval(), llava.load_vision(device)
