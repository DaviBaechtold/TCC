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


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, d: int = 1, dropout: float = 0.1):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):  # x: (B,C,T)
        h = self.conv1(x)
        h = h[..., : x.shape[-1]]  # causal trim
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = h[..., : x.shape[-1]]
        h = self.act(h)
        return self.downsample(x) + self.dropout(h)


class TemporalTCNLifter(nn.Module):
    """Temporal lifter using 1D dilated convolutions.
    Input: (B, T, J, 2)
    Output: (B, T, J, 3)
    We flatten joint dims into channels per time step.
    """

    def __init__(self, num_joints: int, d_model: int = 256, levels: int = 4, k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_joints = num_joints
        in_ch = num_joints * 2
        self.stem = nn.Linear(in_ch, d_model)
        blocks = []
        for i in range(levels):
            d = 2 ** i
            blocks.append(TemporalBlock(d_model, d_model, k=k, d=d, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, num_joints * 3)

    def forward(self, x):  # x: (B,T,J,2)
        B, T, J, D = x.shape
        assert D == 2 and J == self.num_joints
        h = x.reshape(B, T, J * D)  # (B,T,Cin)
        h = self.stem(h)  # (B,T,d_model)
        h = h.transpose(1, 2)  # (B,d_model,T)
        h = self.blocks(h)
        h = h.transpose(1, 2)  # (B,T,d_model)
        out = self.head(h).reshape(B, T, self.num_joints, 3)
        return out


def build_lifter(model_type: str, num_joints: int, **kwargs):
    if model_type == 'mlp':
        return LifterMLP(num_joints=num_joints,
                         hidden=kwargs.get('hidden', 512),
                         depth=kwargs.get('depth', 4),
                         dropout=kwargs.get('dropout', 0.2))
    if model_type == 'tcn':
        return TemporalTCNLifter(num_joints=num_joints,
                                 d_model=kwargs.get('d_model', 256),
                                 levels=kwargs.get('levels', 4),
                                 k=kwargs.get('k', 3),
                                 dropout=kwargs.get('dropout', 0.1))
    raise ValueError(f"Unknown model_type={model_type}")

