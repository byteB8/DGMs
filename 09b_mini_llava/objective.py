"""Answer-only next-token loss.

The LM models p(w_t | w_<t, visual tokens). Only the answer tokens (and the end token) are scored:
    L = - sum_{t in answer} log p(w_t | visual tokens, question, w_<t)
so the model learns to answer, not to repeat the question.
"""
import torch.nn.functional as F


def answer_loss(logits, targets):
    """logits: (B, T, V); targets: (B, T) with -100 where there is no loss."""
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=-100)
