#!/usr/bin/env python
"""Estimate a linear mapping A s.t. A X = Y in least-squares sense.
Used to adapt 2D keypoints from source (synthetic) to target (real) domain.
Usage:
  python domain_adapt_linear.py --src src.npz --tgt tgt.npz --out A.npy

Expect src/tgt to provide arrays (N,J,2) with matching N and J.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

THIS = Path(__file__).resolve()
PROJECT_ROOT = THIS.parents[1]
sys.path.append(str(PROJECT_ROOT))


def load_xy(path: Path):
    npz = np.load(path)
    # pick first matching array
    for k in npz.files:
        arr = npz[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 3 and arr.shape[-1] == 2:
            return arr.reshape(arr.shape[0], -1)  # (N, 2J)
    raise ValueError('No (N,J,2) array found')


def fit_linear(src: np.ndarray, tgt: np.ndarray):
    # solve for A: (N,2J) -> we will find A (2J+1, 2J) including bias using least squares
    N, D = src.shape
    X = np.concatenate([src, np.ones((N, 1), dtype=src.dtype)], axis=1)  # (N, D+1)
    # solve A^T = pinv(X) @ Y ; returns (D+1, D)
    A, *_ = np.linalg.lstsq(X, tgt, rcond=None)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=str, required=True)
    ap.add_argument('--tgt', type=str, required=True)
    ap.add_argument('--out', type=str, required=True)
    args = ap.parse_args()

    src = load_xy(Path(args.src))
    tgt = load_xy(Path(args.tgt))
    if src.shape != tgt.shape:
        raise ValueError('src and tgt must have same shape (N, J, 2)')
    A = fit_linear(src, tgt)
    np.save(args.out, A)
    print(f"Saved linear mapping to {args.out} (shape={A.shape})")


if __name__ == '__main__':
    main()
