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

from src.models.lifter import LifterMLP, mpjpe, procrustes_align, root_center, build_lifter, TemporalTCNLifter
from src.utils.preprocess import fill_nans_temporal, to_tensor_and_normalize_mean_bone
from src.utils.metrics_logging import make_writer, save_metrics_json, log_epoch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synthetic_dataset(n: int, num_joints: int, noise_std: float = 0.01, seq_len: int = 1):
    """Generate synthetic 3D poses (random walk per sequence) and 2D projections."""
    if seq_len == 1:
        pose3d = np.random.randn(n, num_joints, 3).astype(np.float32)
        pose2d = pose3d[..., :2] + np.random.randn(n, num_joints, 2).astype(np.float32) * noise_std
        return pose2d, pose3d
    # temporal: (N, T, J, 3)
    pose3d = np.cumsum(np.random.randn(n, seq_len, num_joints, 3).astype(np.float32) * 0.05, axis=1)
    pose3d += np.random.randn(n, 1, num_joints, 3).astype(np.float32)
    pose2d = pose3d[..., :2] + np.random.randn(n, seq_len, num_joints, 2).astype(np.float32) * noise_std
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
    ap.add_argument('--seq-len', type=int, default=None, help='Override seq_len in config')
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get('seed', 42))

    num_joints = int(cfg.get('num_joints', 17))
    seq_len = args.seq_len if args.seq_len is not None else int(cfg.get('seq_len', 1))
    root_index = int(cfg.get('root_index', 0))

    # Data
    if args.synthetic or True:
        xtr, ytr = synthetic_dataset(args.n_train, num_joints, seq_len=seq_len)
        xva, yva = synthetic_dataset(args.n_val, num_joints, seq_len=seq_len)
    else:
        raise NotImplementedError('Non-synthetic loaders not yet implemented')

    xtr = torch.from_numpy(xtr)
    ytr = torch.from_numpy(ytr)
    xva = torch.from_numpy(xva)
    yva = torch.from_numpy(yva)
    # Fill NaNs
    if seq_len > 1:
        xtr = torch.from_numpy(fill_nans_temporal(xtr.numpy()))
        xva = torch.from_numpy(fill_nans_temporal(xva.numpy()))
    else:
        xtr = torch.from_numpy(np.nan_to_num(xtr.numpy(), nan=0.0))
        xva = torch.from_numpy(np.nan_to_num(xva.numpy(), nan=0.0))

    # Optional root-centering
    if cfg.get('normalize', {}).get('root_center', True):
        if seq_len == 1:
            xtr = root_center(xtr, root_index)
            xva = root_center(xva, root_index)
            ytr = root_center(ytr, root_index)
            yva = root_center(yva, root_index)
        else:
            xtr = root_center(xtr, root_index)
            xva = root_center(xva, root_index)
            ytr = root_center(ytr, root_index)
            yva = root_center(yva, root_index)

    device = torch.device(args.device)

    # Model
    model_cfg = cfg['model']
    model_type = model_cfg.get('type', 'mlp')
    if seq_len > 1 and model_type == 'mlp':
        print('[warn] seq_len > 1 com MLP (processa frame a frame). Para temporal, use model.type: tcn.')
    model = build_lifter(model_type, num_joints,
                         hidden=model_cfg.get('hidden', 512),
                         depth=model_cfg.get('depth', 4),
                         dropout=model_cfg.get('dropout', 0.2),
                         d_model=model_cfg.get('d_model', 256),
                         levels=model_cfg.get('levels', 4),
                         k=model_cfg.get('k', 3)).to(device)

    # Optim
    bs = int(cfg['optim'].get('batch_size', 256))
    epochs = int(cfg['optim'].get('epochs', 5))
    lr = float(cfg['optim'].get('lr', 3e-4))
    wd = float(cfg['optim'].get('weight_decay', 1e-4))
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    work_dir = Path(cfg.get('work_dir', 'Project/data/lifter_runs'))
    work_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = work_dir / 'lifter_best.pt'
    writer = make_writer(str(work_dir / 'tb'))

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
            pred = model(xb) if seq_len == 1 else model(xb)
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
            pa_pred = procrustes_align(pred.reshape(-1, num_joints, 3) if seq_len > 1 else pred,
                                       yva.to(device).reshape(-1, num_joints, 3) if seq_len > 1 else yva.to(device))
            val_pa = mpjpe(pa_pred, yva.to(device)).item()
        if val_mpjpe < best_val:
            best_val = val_mpjpe
            torch.save({'model': model.state_dict(), 'cfg': cfg}, ckpt_path)
        print(f"Epoch {ep}/{epochs} | train_mpjpe={tr_loss:.4f} val_mpjpe={val_mpjpe:.4f} pa_mpjpe={val_pa:.4f}")

        # Logging
        metrics = {'train_mpjpe': tr_loss, 'val_mpjpe': val_mpjpe, 'pa_mpjpe': val_pa}
        log_epoch(writer, metrics, ep)

    # Save metrics JSON
    save_metrics_json(str(work_dir / 'metrics.json'), {'best_val_mpjpe': best_val})

    print(f"Best MPJPE: {best_val:.4f} | saved to {ckpt_path}")


if __name__ == '__main__':
    main()
