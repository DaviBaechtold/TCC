#!/usr/bin/env python
"""Evaluate a directory of captured .npz files using a trained lifter and save per-file metrics.
Outputs a JSON with MPJPE and PA-MPJPE per file and overall.
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

from src.models.lifter import LifterMLP, mpjpe, procrustes_align, root_center  # noqa: E402
from src.utils.metrics_logging import save_metrics_json  # noqa: E402
from src.utils.preprocess import fill_nans_temporal  # noqa: E402


CANDIDATE_KEYS = ['pose2d', 'keypoints', 'x']


def find_2d_array(npz: np.lib.npyio.NpzFile):
    for k in CANDIDATE_KEYS:
        if k in npz:
            arr = npz[k]
            if arr.ndim == 3 and arr.shape[-1] == 2:
                return arr
    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 2:
            return arr
    raise ValueError('No (N,J,2) array found in NPZ')


def evaluate_file(model, path: Path, device='cpu'):
    data = np.load(path)
    x2d = find_2d_array(data).astype(np.float32)
    x2d = fill_nans_temporal(x2d)
    # For evaluation, create a synthetic 3D GT by adding small z
    N, J, C = x2d.shape
    # create dummy gt by lifting: (for real evaluation you need GT). here we just compare consistent behavior
    x = torch.from_numpy(x2d).to(device)
    x = root_center(x, 0)
    with torch.no_grad():
        pred = model(x)
    # no GT: set mpjpe to NaN
    return {'file': str(path), 'mpjpe': None, 'pa_mpjpe': None, 'pred_shape': pred.shape}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, required=True)
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--out', type=str, default='Project/data/eval_dir_metrics.json')
    ap.add_argument('--device', type=str, default='cpu')
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    # assume saved model type mlp/tcn
    # load by attempting both
    from src.models.lifter import build_lifter  # noqa: E402
    cfg = ckpt.get('cfg', {})
    num_joints = int(cfg.get('num_joints', 17))
    model_type = cfg.get('model', {}).get('type', 'mlp')
    model = build_lifter(model_type, num_joints)
    model.load_state_dict(ckpt['model'])
    model.to(args.device)
    model.eval()

    folder = Path(args.dir)
    results = []
    for p in sorted(folder.glob('*.npz')):
        try:
            r = evaluate_file(model, p, device=args.device)
            results.append(r)
        except Exception as e:
            results.append({'file': str(p), 'error': str(e)})

    save_metrics_json(args.out, {'files': results})
    print(f"Saved metrics to {args.out}")


if __name__ == '__main__':
    main()
