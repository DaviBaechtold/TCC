#!/usr/bin/env python
"""Evaluate a trained lifter on synthetic validation data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

# Add Project/ to sys.path so we can import src.*
THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.lifter import LifterMLP, mpjpe, procrustes_align, root_center  # noqa: E402


def synthetic_dataset(n: int, num_joints: int, noise_std: float = 0.01):
    pose3d = np.random.randn(n, num_joints, 3).astype(np.float32)
    pose2d = pose3d[..., :2] + np.random.randn(n, num_joints, 2).astype(np.float32) * noise_std
    return pose2d, pose3d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--num-joints', type=int, default=17)
    ap.add_argument('--n-val', type=int, default=1000)
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    device = torch.device(args.device)

    xva, yva = synthetic_dataset(args.n_val, args.num_joints)
    xva = torch.from_numpy(xva)
    yva = torch.from_numpy(yva)

    xva = root_center(xva, 0)
    yva = root_center(yva, 0)

    model = LifterMLP(num_joints=args.num_joints)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    with torch.no_grad():
        pred = model(xva.to(device))
        val_mpjpe = mpjpe(pred, yva.to(device)).item()
        pa_pred = procrustes_align(pred, yva.to(device))
        val_pa = mpjpe(pa_pred, yva.to(device)).item()
    print(f"MPJPE={val_mpjpe:.4f} | PA-MPJPE={val_pa:.4f}")


if __name__ == '__main__':
    main()
