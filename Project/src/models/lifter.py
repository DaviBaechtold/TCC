"""
Minimal 2D->3D lifter models and utilities.

Contracts:
- Input 2D: (B, J, 2) float32
- Output 3D: (B, J, 3) float32 (root-centered, scale-normalized if configured)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NormConfig:
    root_index: int = 0
    eps: float = 1e-8


def root_center(x: torch.Tensor, root_index: int = 0) -> torch.Tensor:
    """Subtract root joint from all joints.
    x: (B, J, D)
    """
    root = x[:, root_index:root_index + 1, :]
    return x - root


def scale_by_mean_bone(x: torch.Tensor, edges: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scale skeleton by mean bone length.
    x: (B, J, D), edges: (E, 2) long indices
    Returns: (x_scaled, scale) where scale: (B, 1, 1)
    """
    b = x[:, edges[:, 0], :] - x[:, edges[:, 1], :]
    lens = torch.norm(b, dim=-1)  # (B, E)
    scale = (lens.mean(dim=1, keepdim=True).clamp_min(eps))[:, None, :]  # (B,1,1)
    return x / scale, scale


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.act(h)
        return x + self.dropout(h)


class LifterMLP(nn.Module):
    """Per-frame lifter: flattens joints and maps to 3D joints.
    Input:  (B, J, 2)
    Output: (B, J, 3)
    """

    def __init__(self, num_joints: int, hidden: int = 512, depth: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_joints = num_joints
        in_dim = num_joints * 2
        out_dim = num_joints * 3
        self.inp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden, dropout) for _ in range(max(0, depth))])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x2d: torch.Tensor) -> torch.Tensor:
        B, J, D = x2d.shape
        assert D == 2 and J == self.num_joints, f"expected (B,{self.num_joints},2), got {tuple(x2d.shape)}"
        h = x2d.reshape(B, -1)
        h = self.inp(h)
        h = self.blocks(h)
        y = self.out(h).reshape(B, self.num_joints, 3)
        return y


def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Mean Per Joint Position Error.
    pred, gt: (B, J, 3)
    Returns scalar tensor.
    """
    return torch.norm(pred - gt, dim=-1).mean()


def procrustes_align(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Procrustes alignment (scale+rotation) per-sample.
    Returns aligned pred with same shape as input.
    """
    B, J, D = pred.shape
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    gt_c = gt - gt.mean(dim=1, keepdim=True)
    pred_norm = torch.norm(pred_c, dim=(1, 2), keepdim=True).clamp_min(eps)
    gt_norm = torch.norm(gt_c, dim=(1, 2), keepdim=True).clamp_min(eps)
    pred_c = pred_c / pred_norm
    gt_c = gt_c / gt_norm

    # Compute optimal rotation via SVD for each batch
    # Using batched matmul: (B,3,J) @ (B,J,3) -> (B,3,3)
    H = torch.matmul(pred_c.transpose(1, 2), gt_c)
    U, S, Vt = torch.linalg.svd(H)  # U:(B,3,3), Vt:(B,3,3)
    R = torch.matmul(U, Vt)
    # Correct improper rotation (reflection)
    det = torch.linalg.det(R)
    mask = (det < 0).view(B, 1, 1)
    if mask.any():
        Vt_fix = Vt.clone()
        Vt_fix[mask.view(-1), :, -1] *= -1
        R = torch.matmul(U, Vt_fix)
    aligned = torch.matmul(pred_c, R)
    # Scale to gt
    scale = (gt_c * aligned).sum(dim=(1, 2), keepdim=True) / (aligned.pow(2).sum(dim=(1, 2), keepdim=True).clamp_min(eps))
    aligned = aligned * scale
    # Add gt mean back (optional for PA-MPJPE we usually keep centered)
    aligned = aligned + gt.mean(dim=1, keepdim=True)
    return aligned


__all__ = [
    "NormConfig",
    "root_center",
    "scale_by_mean_bone",
    "ResidualBlock",
    "LifterMLP",
    "mpjpe",
    "procrustes_align",
]
