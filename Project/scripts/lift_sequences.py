#!/usr/bin/env python
"""Convert 2D sequences (.npz) to 3D using a trained lifter model.

Input NPZ is expected to contain an array with shape (N, J, 2). We try keys in order:
- 'pose2d', 'keypoints', 'x'
Fallback: first array with shape (_, J, 2).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.lifter import LifterMLP, root_center, build_lifter  # noqa: E402


CANDIDATE_KEYS = ['pose2d', 'keypoints', 'x']


def find_2d_array(npz: np.lib.npyio.NpzFile):
    for k in CANDIDATE_KEYS:
        if k in npz:
            arr = npz[k]
            if arr.ndim == 3 and arr.shape[-1] == 2:
                return arr
    # fallback: first array with last dim == 2
    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 2:
            return arr
    raise ValueError('No (N,J,2) array found in NPZ')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=str, required=True, help='Input .npz with 2D sequences')
    ap.add_argument('--output', type=str, required=True, help='Output .npz path for 3D sequences')
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--root-center', action='store_true', help='Apply root-centering (idx=0) before lifting')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    data = np.load(args.input)
    x2d = find_2d_array(data).astype(np.float32)
    N, J, _ = x2d.shape

    # Load checkpoint first to determine expected joint count / model type
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_cfg = ckpt.get('cfg', {}) or {}
    ckpt_num_joints = int(ckpt_cfg.get('num_joints', 0)) if ckpt_cfg.get('num_joints') is not None else 0

    if ckpt_num_joints and ckpt_num_joints != J:
        if ckpt_num_joints < J:
            print(f"[warn] checkpoint expects {ckpt_num_joints} joints but input has {J}; subsetting first {ckpt_num_joints} joints")
            x2d = x2d[:, :ckpt_num_joints, :]
            J = ckpt_num_joints
        else:
            raise ValueError(f"Checkpoint expects {ckpt_num_joints} joints but input has only {J}; please provide matching topology or a mapping")

    x = torch.from_numpy(x2d)
    if args.root_center:
        x = root_center(x, 0)

    # Build model matching checkpoint if config present
    model_type = ckpt_cfg.get('model', {}).get('type', 'mlp')
    try:
        model = build_lifter(model_type, num_joints=J)
    except Exception:
        model = LifterMLP(num_joints=J)

    model.load_state_dict(ckpt['model'])
    model.to(torch.device(args.device))
    model.eval()

    with torch.no_grad():
        y3d = model(x.to(args.device)).cpu().numpy()

    np.savez(args.output, pose3d=y3d)
    print(f"Saved 3D sequences to {args.output} | shape={y3d.shape}")


if __name__ == '__main__':
    main()
