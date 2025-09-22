"""Utilities to log metrics to JSON and TensorBoard + rich console."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from torch.utils.tensorboard import SummaryWriter
from rich.console import Console


console = Console()


def save_metrics_json(path: str, metrics: Dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        json.dump(metrics, f, indent=2)


def make_writer(logdir: str):
    p = Path(logdir)
    p.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(p))


def log_epoch(writer: SummaryWriter, metrics: Dict, step: int):
    for k, v in metrics.items():
        try:
            writer.add_scalar(k, float(v), step)
        except Exception:
            pass
    writer.flush()
