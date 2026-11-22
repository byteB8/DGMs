"""Contrastive loss (InfoNCE) for paired images and texts.

For a batch of B pairs with unit embeddings u_i (images) and v_j (texts), the logits are
    s_ij = tau^{-1} u_i . v_j     (cosine similarity divided by a learned temperature tau)
Each image must pick out its own text among the B texts, and each text its own image:
    L = 1/2 [ CE(s, diag) + CE(s^T, diag) ]
Hard negatives: extra texts that each image must also rank below its own caption
(here: the caption with two objects' attributes swapped).
"""
import torch
import torch.nn.functional as F


def clip_loss(image_emb, text_emb, log_scale, negative_text_emb=None):
    scale = log_scale.exp()
    logits = scale * image_emb @ text_emb.t()
    target = torch.arange(len(logits), device=logits.device)
    image_to_text = logits
    if negative_text_emb is not None:
        image_to_text = torch.cat([logits, scale * image_emb @ negative_text_emb.t()], dim=1)
    return 0.5 * (F.cross_entropy(image_to_text, target) + F.cross_entropy(logits.t(), target))
