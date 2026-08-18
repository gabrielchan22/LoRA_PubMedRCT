"""
Measurement harness — shared by every experiment so results are directly
comparable.
"""

import os
import time
import torch
from sklearn.metrics import f1_score


def count_trainable_parameters(model) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": 100 * trainable / total,
    }


class TrainingRunTracker:
    """Tracks wall-clock time and peak GPU memory for one training run."""

    def __init__(self):
        self._start_time = None
        self._end_time = None
        self._peak_memory_mb = None

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.time()

    def stop(self):
        self._end_time = time.time()
        if torch.cuda.is_available():
            self._peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            # Peak memory isn't meaningfully measurable this way on CPU;
            # record NaN rather than a misleading number.
            self._peak_memory_mb = float("nan")

    def summary(self) -> dict:
        return {
            "wall_clock_sec": self._end_time - self._start_time,
            "peak_memory_mb": self._peak_memory_mb,
        }


def compute_classification_metrics(all_preds, all_labels) -> dict:
    all_preds = torch.as_tensor(all_preds)
    all_labels = torch.as_tensor(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
    macro_f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def get_checkpoint_size_mb(path: str) -> float:
    if os.path.isdir(path):
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                total += os.path.getsize(os.path.join(dirpath, f))
        return total / (1024 ** 2)
    return os.path.getsize(path) / (1024 ** 2)
