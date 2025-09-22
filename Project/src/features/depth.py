from __future__ import annotations
"""Depth feature extraction wrappers.

This module provides lightweight depth encoders as placeholders for integrating
monocular depth models like Depth Anything v2 or Depth Pro. For portability we
avoid bundling heavy weights; users can plug in their own model by subclassing
`BaseDepthModel`.

Contracts:
  Input depth tensor: (B, T, H, W) float32 normalized to [0,1] or arbitrary scale.
  Output embedding per frame: (B, T, D)
"""
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DepthEncoderConfig:
    in_channels: int = 1
    hidden: int = 64
    out_dim: int = 128


class CNNDepthEncoder(nn.Module):
    def __init__(self, cfg: DepthEncoderConfig):
        super().__init__()
        c = cfg.in_channels
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(128, cfg.out_dim)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # depth: (B,T,H,W) -> merge B,T
        B, T, H, W = depth.shape
        x = depth.reshape(B*T, 1, H, W)
        h = self.net(x).reshape(B*T, -1)
        emb = self.proj(h).reshape(B, T, -1)
        return emb


def build_depth_encoder(cfg: DepthEncoderConfig) -> nn.Module:
    return CNNDepthEncoder(cfg)

__all__ = ["DepthEncoderConfig", "build_depth_encoder", "CNNDepthEncoder"]
