from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .metrics import compute_metrics, confusion


def load_yaml(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class TrainState:
    epoch: int = 0
    best_acc: float = 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str):
    model.train()
    ce = nn.CrossEntropyLoss()
    losses = []
    all_y, all_p = [], []
    for batch in tqdm(loader, leave=False):
        x = batch['x'].to(device)
        m = batch['mask'].to(device)
        y = batch['y'].to(device)
        optimizer.zero_grad()
        logits = model(x, m)
        loss = ce(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        all_y.append(y.detach())
        all_p.append(logits.argmax(dim=1).detach())
    all_y = torch.cat(all_y)
    all_p = torch.cat(all_p)
    mets = compute_metrics(all_y, all_p)
    mets['loss'] = float(sum(losses) / max(1, len(losses)))
    return mets


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[Dict[str, float], torch.Tensor]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    losses = []
    all_y, all_p = [], []
    for batch in loader:
        x = batch['x'].to(device)
        m = batch['mask'].to(device)
        y = batch['y'].to(device)
        logits = model(x, m)
        loss = ce(logits, y)
        losses.append(loss.item())
        all_y.append(y.detach())
        all_p.append(logits.argmax(dim=1).detach())
    all_y = torch.cat(all_y)
    all_p = torch.cat(all_p)
    mets = compute_metrics(all_y, all_p)
    mets['loss'] = float(sum(losses) / max(1, len(losses)))
    cm = confusion(all_y, all_p)
    return mets, torch.from_numpy(cm)


def save_checkpoint(path: str, model: nn.Module, state: TrainState, extra: Dict = None):
    ensure_dir(os.path.dirname(path))
    torch.save({
        'model': model.state_dict(),
        'state': state.__dict__,
        'extra': extra or {}
    }, path)


def load_checkpoint(path: str, model: nn.Module):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    st = TrainState(**ckpt.get('state', {}))
    return st, ckpt.get('extra', {})
