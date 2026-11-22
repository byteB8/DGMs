"""Image datasets as in-memory tensors. They are small enough to sit on the GPU."""
import torch
from torchvision import datasets

from .utils import DATA_DIR

_DATASETS = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}

CLASS_NAMES = {
    "mnist": [str(i) for i in range(10)],
    "fmnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "cifar10": ["plane", "car", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"],
}


def load_images(name: str, train: bool = True, device="cpu"):
    """Return images x in [-1, 1] with shape (N, C, H, W) and integer labels y."""
    ds = _DATASETS[name](DATA_DIR, train=train, download=True)
    x = torch.as_tensor(ds.data)
    x = x[:, None] if x.ndim == 3 else x.permute(0, 3, 1, 2)
    x = x.float().div(127.5).sub(1)
    y = torch.as_tensor(ds.targets, dtype=torch.long)
    return x.to(device), y.to(device)


def binarize(x: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] images to {0, 1} by thresholding at mid-grey."""
    return (x > 0).float()


def epoch_batches(x, y, batch_size: int, shuffle: bool = True):
    """One pass over (x, y) in batches. The last partial batch is dropped when shuffling."""
    n = len(x)
    order = torch.randperm(n, device=x.device) if shuffle else torch.arange(n, device=x.device)
    stop = n - batch_size + 1 if shuffle else n
    for i in range(0, stop, batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], y[idx]


def infinite_batches(x, y, batch_size: int):
    """Endless shuffled batches, for step-based training loops."""
    while True:
        yield from epoch_batches(x, y, batch_size)
