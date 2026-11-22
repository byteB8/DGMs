"""Mini-LLaVA: a frozen SigLIP image encoder, a frozen SmolLM2 language model and a small trained projector.

Image (3 x 32 x 32) -> upsample to 224 -> SigLIP -> 196 patch vectors (768-d) -> 2 x 2 average pool -> 49 vectors
-> projector (MLP, 768 -> 576 -> 576) -> 49 "visual tokens" in the LM's embedding space.
LM input: [49 visual tokens] "Question: ...\\nAnswer:"  ->  the LM predicts " red" (then end of text).
"""
import torch
import torch.nn.functional as F
from torch import nn

VISION_MODEL = "google/siglip-base-patch16-224"
LANGUAGE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def load_vision(device):
    from transformers import SiglipVisionModel
    return SiglipVisionModel.from_pretrained(VISION_MODEL).to(device).eval().requires_grad_(False)


def load_language_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
    lm = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL).to(device).eval().requires_grad_(False)
    return lm, tok


@torch.no_grad()
def image_features(vision, images, batch: int = 250):
    """images: float in [-1, 1], (N, 3, H, W) -> pooled SigLIP patch features (N, 49, 768).
    SigLIP expects 224 x 224 inputs scaled to [-1, 1], which is our image range."""
    out = []
    for i in range(0, len(images), batch):
        x = F.interpolate(images[i:i + batch], size=224, mode="bilinear", align_corners=False)
        h = vision(pixel_values=x).last_hidden_state                       # (B, 196, 768)
        B, T, D = h.shape
        side = int(T ** 0.5)
        h = F.avg_pool2d(h.transpose(1, 2).reshape(B, D, side, side), 2)   # 14 x 14 -> 7 x 7
        out.append(h.flatten(2).transpose(1, 2))
    return torch.cat(out)


class Projector(nn.Module):
    """Maps vision features into the LM's word-embedding space."""

    def __init__(self, vision_dim: int = 768, lm_dim: int = 576):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vision_dim, lm_dim), nn.GELU(), nn.Linear(lm_dim, lm_dim))

    def forward(self, features):
        return self.net(features)


def build_inputs(lm, tok, visual_tokens, questions, answers=None):
    """Assemble [visual tokens][prompt][answer][end] as input embeddings, right-padded.

    Returns (inputs_embeds, attention_mask, targets, answer_pos):
      targets[b, t] is the token that position t should predict (-100 where there is no loss),
      answer_pos[b] is the position whose prediction is the first answer token.
    """
    device = visual_tokens.device
    n_vis = visual_tokens.shape[1]
    embed = lm.get_input_embeddings()
    seqs, preds = [], []
    for k, q in enumerate(questions):
        prompt = tok(f"Question: {q}\nAnswer:", add_special_tokens=False).input_ids
        answer = [] if answers is None else tok(" " + answers[k], add_special_tokens=False).input_ids + [tok.eos_token_id]
        seqs.append((prompt, answer))
    length = max(len(p) + len(a) for p, a in seqs)
    ids = torch.full((len(seqs), length), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros(len(seqs), n_vis + length, dtype=torch.long)
    targets = torch.full((len(seqs), n_vis + length), -100, dtype=torch.long)
    for b, (p, a) in enumerate(seqs):
        ids[b, :len(p) + len(a)] = torch.tensor(p + a)
        mask[b, :n_vis + len(p) + len(a)] = 1
        start = n_vis + len(p) - 1                        # the last prompt position predicts the first answer token
        targets[b, start:start + len(a)] = torch.tensor(a, dtype=torch.long)
        preds.append(start)
    inputs = torch.cat([visual_tokens.to(embed.weight.dtype), embed(ids.to(device))], dim=1)
    return inputs, mask.to(device), targets.to(device), torch.tensor(preds, device=device)


def add_lora(lm, r: int = 8, alpha: int = 16):
    """Low-rank adapters on the attention query and value projections: W x + (alpha / r) B A x, A and B trained."""
    from peft import LoraConfig, get_peft_model
    return get_peft_model(lm, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05,
                                         target_modules=["q_proj", "v_proj"]))
