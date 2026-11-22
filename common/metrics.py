"""Evaluation metrics shared by projects."""
import numpy as np
import torch
from scipy import linalg


def mode_coverage(samples, centers, std: float, min_frac: float = 0.01) -> dict:
    """Mode metrics for Gaussian mixtures such as 8 Gaussians.

    A sample is high quality (hq) if it lies within 3 std of its nearest center.
    A mode is covered if at least min_frac of all samples are hq samples of that mode.
    """
    dist, nearest = torch.cdist(samples, centers.to(samples)).min(dim=1)
    good = dist < 3 * std
    counts = torch.bincount(nearest[good], minlength=len(centers))
    return {"modes": int((counts >= min_frac * len(samples)).sum()), "hq": good.float().mean().item()}


@torch.no_grad()
def classifier_outputs(clf, x, batch_size: int = 2000):
    """Penultimate features (float64 numpy) and predicted labels for images x."""
    feats, preds = [], []
    for i in range(0, len(x), batch_size):
        f = clf.features(x[i:i + batch_size])
        feats.append(f)
        preds.append(clf.head(f).argmax(1))
    return torch.cat(feats).double().cpu().numpy(), torch.cat(preds).cpu()


def frechet_distance(f1: np.ndarray, f2: np.ndarray, eps: float = 1e-6) -> float:
    """||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)) between Gaussians fitted to two feature sets.

    eps * I is added to both covariances so the matrix square root stays well defined.
    """
    mu1, mu2 = f1.mean(0), f2.mean(0)
    offset = eps * np.eye(f1.shape[1])
    s1, s2 = np.cov(f1, rowvar=False) + offset, np.cov(f2, rowvar=False) + offset
    covmean = linalg.sqrtm(s1 @ s2).real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def image_metrics(clf, fake, real_feats: np.ndarray, labels=None, n_classes: int = 10) -> dict:
    """Scores for generated images.

    fid_lite:  Frechet distance in the classifier's feature space (lower is better).
    class_kl:  KL(predicted class histogram || uniform); 0 means all classes equally often.
    min_class: share of the rarest predicted class (0.1 is perfect balance for 10 classes).
    label_acc: for conditional models, how often the classifier agrees with the requested label.
    """
    feats, preds = classifier_outputs(clf, fake)
    hist = np.bincount(preds.numpy(), minlength=n_classes) / len(preds)
    out = {
        "fid_lite": frechet_distance(feats, real_feats),
        "class_kl": float(np.sum(hist * np.log(np.maximum(hist, 1e-12) * n_classes))),
        "min_class": float(hist.min()),
    }
    if labels is not None:
        out["label_acc"] = (preds == labels.cpu()).float().mean().item()
    return out
