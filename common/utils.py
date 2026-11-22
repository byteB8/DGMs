"""Small helpers shared by all projects."""
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_run_dir(script_file: str, name: str) -> Path:
    """Create <project>/runs/<name>/ next to the calling script."""
    path = Path(script_file).resolve().parent / "runs" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2))


def load_json(path):
    return json.loads(Path(path).read_text())


class Logger:
    """Print metrics and write them to <run_dir>/log.jsonl, one line per call."""

    def __init__(self, run_dir):
        self.file = open(Path(run_dir) / "log.jsonl", "w")
        self.start = time.time()

    def log(self, step: int, **metrics) -> None:
        record = {"step": step, "time": round(time.time() - self.start, 1), **metrics}
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                        for k, v in record.items()), flush=True)


def read_log(run_dir) -> list:
    with open(Path(run_dir) / "log.jsonl") as f:
        return [json.loads(line) for line in f]
