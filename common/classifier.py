"""Small CNN classifier used to score generated images (FID-lite, class coverage).

Trained once per dataset and cached in data/classifiers/<name>.pt.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .data import epoch_batches, load_images
from .utils import DATA_DIR


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, img_size: int = 28, n_classes: int = 10, feat_dim: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (img_size // 4) ** 2, feat_dim), nn.ReLU(),
        )
        self.head = nn.Linear(feat_dim, n_classes)

    def features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))


@torch.no_grad()
def accuracy(model, x, y, batch_size: int = 2000) -> float:
    correct = sum((model(xb).argmax(1) == yb).sum().item()
                  for xb, yb in epoch_batches(x, y, batch_size, shuffle=False))
    return correct / len(x)


def train_classifier(name: str, device, epochs: int = 5, batch_size: int = 128) -> SmallCNN:
    x, y = load_images(name, train=True, device=device)
    x_test, y_test = load_images(name, train=False, device=device)
    model = SmallCNN(in_ch=x.shape[1], img_size=x.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        for xb, yb in epoch_batches(x, y, batch_size):
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        print(f"classifier {name} epoch {epoch + 1}: test acc {accuracy(model, x_test, y_test):.4f}")
    path = DATA_DIR / "classifiers" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "in_ch": x.shape[1], "img_size": x.shape[-1],
                "test_acc": accuracy(model, x_test, y_test)}, path)
    return model


def load_classifier(name: str, device) -> SmallCNN:
    """Load the cached classifier for a dataset, training it first if needed."""
    path = DATA_DIR / "classifiers" / f"{name}.pt"
    if not path.exists():
        train_classifier(name, device)
    ckpt = torch.load(path, map_location=device)
    model = SmallCNN(in_ch=ckpt["in_ch"], img_size=ckpt["img_size"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval().requires_grad_(False)
    return model
