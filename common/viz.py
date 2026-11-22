"""Plotting helpers. Figures are written to disk, never shown."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from torchvision.utils import save_image  # noqa: E402


def save_image_grid(x, path, nrow: int = 8) -> None:
    """Save images in [-1, 1] with shape (N, C, H, W) as one grid."""
    save_image(x.detach().cpu().clamp(-1, 1).add(1).div(2), path, nrow=nrow, padding=2, pad_value=1)


def plot_2d_panels(panels, path, real=None, lim: float = 3.0, size: float = 2.6) -> None:
    """One scatter panel per (title, samples) pair; optional real samples drawn underneath in grey."""
    fig, axes = plt.subplots(1, len(panels), figsize=(size * len(panels), size), squeeze=False)
    for ax, (title, x) in zip(axes[0], panels):
        if real is not None:
            r = real.detach().cpu()
            ax.scatter(r[:, 0], r[:, 1], s=1, c="0.75", alpha=0.5, linewidths=0)
        x = x.detach().cpu()
        ax.scatter(x[:, 0], x[:, 1], s=1, c="C0", alpha=0.6, linewidths=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(series, path, xlabel: str = "step", ylabel: str = "", logy: bool = False) -> None:
    """series: dict label -> (xs, ys)."""
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for label, (xs, ys) in series.items():
        ax.plot(xs, ys, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
