"""Preprocessing helpers: NaN handling and mean-bone normalization helpers."""
from __future__ import annotations

import numpy as np
import torch
from typing import List, Tuple


def fill_nans_temporal(arr: np.ndarray) -> np.ndarray:
    """Interpolate NaNs along the time axis for arrays with shape (T,J,2) or (N,T,J,2) or (N,J,2).
    If no temporal axis present (2D sample), forward-fill NaNs with zeros.
    """
    a = arr.copy()
    # Normalize to shape (N, T, J, 2)
    # Handle ambiguous 3D shapes: could be (T,J,2) (a single sequence) or (N,J,2) (batch of frames)
    # Heuristic: if first dim (arr.shape[0]) is small (<16) treat as temporal length T, else treat as batch N
    if a.ndim == 3 and a.shape[-1] == 2:
        # Could be (T,J,2) or (N,J,2)
        if a.shape[0] < 16:
            # treat as (T,J,2) -> convert to (1, T, J, 2)
            a = a[None, ...]
            _orig_was_T = True
        else:
            # treat as (N,J,2) -> convert to (N, 1, J, 2)
            a = a[:, None, ...]
            _orig_was_T = False
    elif a.ndim == 2 and a.shape[-1] == 2:
        a = a[None, None, ...]
    elif a.ndim == 4:
        pass
    else:
        raise ValueError(f"Unsupported shape for fill_nans_temporal: {arr.shape}")

    N, T, J, C = a.shape
    for n in range(N):
        for j in range(J):
            for c in range(C):
                series = a[n, :, j, c]
                mask = np.isfinite(series)
                if mask.all():
                    continue
                if not mask.any():
                    # no valid values: fill with 0
                    a[n, :, j, c] = 0.0
                    continue
                idx = np.arange(T)
                valid_x = idx[mask]
                valid_y = series[mask]
                # linear interpolation
                interp = np.interp(idx, valid_x, valid_y)
                a[n, :, j, c] = interp

    # squeeze back
    # Return to original dimensionality where possible
    if arr.ndim == 3:
        # If we converted an original (T,J,2) to (1,T,J,2) return (T,J,2)
        if ' _orig_was_T' in locals() and _orig_was_T:
            return a[0]
        # If we converted an original (N,J,2) to (N,1,J,2) return (N,J,2)
        if ' _orig_was_T' in locals() and not _orig_was_T:
            return a[:, 0]
        # Fallback: if first dim matches original first dim, attempt to squeeze
        if a.shape[0] == arr.shape[0]:
            return a[0]
        return a
    if arr.ndim == 2:
        return a[0, 0]
    return a


def to_tensor_and_normalize_mean_bone(x: torch.Tensor, edges: List[Tuple[int, int]] = None, eps: float = 1e-8):
    """If edges provided, scale x by mean bone length.
    x: (B,J,D) or (B,T,J,D)
    Returns scaled x and scale tensor.
    """
    if edges is None or len(edges) == 0:
        return x, torch.ones((x.shape[0], 1, 1), device=x.device)

    # Convert edges to tensor
    device = x.device
    edges_t = torch.tensor(edges, dtype=torch.long, device=device)

    if x.ndim == 4:
        # (B,T,J,D) -> collapse T into B*T for bone computation
        B, T, J, D = x.shape
        xt = x.reshape(B * T, J, D)
        b = xt[:, edges_t[:, 0], :] - xt[:, edges_t[:, 1], :]
        lens = torch.norm(b, dim=-1)
        scale = (lens.mean(dim=1, keepdim=True).clamp_min(eps))[:, None, :]
        scale = scale.reshape(B, T, 1, 1)
        return x / scale, scale
    else:
        # (B,J,D)
        b = x[:, edges_t[:, 0], :] - x[:, edges_t[:, 1], :]
        lens = torch.norm(b, dim=-1)
        scale = (lens.mean(dim=1, keepdim=True).clamp_min(eps))[:, None, :]
        return x / scale, scale
