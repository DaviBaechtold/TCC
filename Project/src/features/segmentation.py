from __future__ import annotations
"""Segmentation feature encoder.

Takes a binary person mask (B,T,H,W) in {0,1} and produces per-frame embeddings
(B,T,D). Simple CNN reduces spatial dimension.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class SegmentationEncoderConfig:
    in_channels: int = 1
    out_dim: int = 64


class MaskEncoder(nn.Module):
    def __init__(self, cfg: SegmentationEncoderConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cfg.in_channels, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.proj = nn.Linear(64, cfg.out_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        B, T, H, W = mask.shape
        x = mask.reshape(B*T, 1, H, W).float()
        h = self.net(x).reshape(B*T, -1)
        return self.proj(h).reshape(B, T, -1)


def build_segmentation_encoder(cfg: SegmentationEncoderConfig) -> nn.Module:
    return MaskEncoder(cfg)

__all__ = ["SegmentationEncoderConfig", "build_segmentation_encoder", "MaskEncoder"]
