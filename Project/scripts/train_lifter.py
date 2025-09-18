#!/usr/bin/env python
"""
Train a minimal 2D->3D lifter on synthetic pairs or a provided NPZ dataset.

Synthetic mode (default): generates random 3D poses, projects to 2D with noise.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

# Ensure we can import src.* when called as a script
import sys
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.lifter import LifterMLP, mpjpe, procrustes_align, root_center


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_dataset(n: int, num_joints: int, noise_std: float = 0.01):
    """Generate simple synthetic 3D poses and 2D projections w/ noise."""
    # 3D GT centered around origin
    pose3d = np.random.randn(n, num_joints, 3).astype(np.float32)
    # Simple weak-perspective projection: drop z and add noise
    pose2d = pose3d[..., :2] + np.random.randn(n, num_joints, 2).astype(np.float32) * noise_std
    return pose2d, pose3d


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='Project/configs/lifter.yaml')
    ap.add_argument('--synthetic', action='store_true', help='Use synthetic data (default)')
    ap.add_argument('--n-train', type=int, default=6000)
    ap.add_argument('--n-val', type=int, default=1000)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))

    num_joints = int(cfg.get('num_joints', 17))
    root_index = int(cfg.get('root_index', 0))

    # Data
    if args.synthetic or True:
        xtr, ytr = synthetic_dataset(args.n_train, num_joints)
        xva, yva = synthetic_dataset(args.n_val, num_joints)
    else:
        raise NotImplementedError('Non-synthetic loaders not yet implemented')

    xtr = torch.from_numpy(xtr)
    ytr = torch.from_numpy(ytr)
    xva = torch.from_numpy(xva)
    yva = torch.from_numpy(yva)

    # Optional root-centering
    if cfg.get('normalize', {}).get('root_center', True):
        xtr = root_center(xtr, root_index)
        xva = root_center(xva, root_index)
        ytr = root_center(ytr, root_index)
        yva = root_center(yva, root_index)

    device = torch.device(args.device)

    # Model
    hidden = int(cfg['model'].get('hidden', 512))
    depth = int(cfg['model'].get('depth', 4))
    dropout = float(cfg['model'].get('dropout', 0.2))
    model = LifterMLP(num_joints=num_joints, hidden=hidden, depth=depth, dropout=dropout).to(device)

    # Optim
    bs = int(cfg['optim'].get('batch_size', 256))
    epochs = int(cfg['optim'].get('epochs', 5))
    lr = float(cfg['optim'].get('lr', 3e-4))
    wd = float(cfg['optim'].get('weight_decay', 1e-4))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    work_dir = Path(cfg.get('work_dir', 'Project/data/lifter_runs'))
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / 'lifter_best.pt'

    # Dataloaders (simple tensor batches)
    def batches(x, y, batch_size, shuffle=True):
        idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
        for i in range(0, len(x), batch_size):
            j = idx[i:i + batch_size]
            yield x[j], y[j]

    best_val = float('inf')

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in batches(xtr, ytr, bs, shuffle=True):
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = mpjpe(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(xtr)

        model.eval()
        with torch.no_grad():
            pred = model(xva.to(device))
            val_mpjpe = mpjpe(pred, yva.to(device)).item()
            pa_pred = procrustes_align(pred, yva.to(device))
            val_pa = mpjpe(pa_pred, yva.to(device)).item()
        if val_mpjpe < best_val:
            best_val = val_mpjpe
            torch.save({'model': model.state_dict(), 'cfg': cfg}, ckpt_path)
        print(f"Epoch {ep}/{epochs} | train_mpjpe={tr_loss:.4f} val_mpjpe={val_mpjpe:.4f} pa_mpjpe={val_pa:.4f}")

    print(f"Best MPJPE: {best_val:.4f} | saved to {ckpt_path}")


if __name__ == '__main__':
    main()
