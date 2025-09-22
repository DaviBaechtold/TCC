from __future__ import annotations
"""Video embedding encoder (placeholder).

For simplicity we implement a lightweight 3D CNN to produce clip-level features
and then broadcast or keep temporal dimension. Input expected as (B,T,3,H,W).
Output: (B,T,D) embeddings (frame-aligned). We use causal temporal conv.
"""
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class VideoEncoderConfig:
    in_channels: int = 3
    hidden: int = 64
    out_dim: int = 256
    kernel_t: int = 3


class Simple3DConvEncoder(nn.Module):
    def __init__(self, cfg: VideoEncoderConfig):
        super().__init__()
        kt = cfg.kernel_t
        self.conv1 = nn.Conv3d(cfg.in_channels, 64, (kt,5,5), stride=(1,2,2), padding=(kt//2,2,2))
        self.conv2 = nn.Conv3d(64, 128, (kt,3,3), stride=(1,2,2), padding=(kt//2,1,1))
        self.conv3 = nn.Conv3d(128, 128, (kt,3,3), stride=(1,2,2), padding=(kt//2,1,1))
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool3d((None,1,1))  # keep temporal dim
        self.proj = nn.Linear(128, cfg.out_dim)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (B,T,3,H,W) -> (B,3,T,H,W)
        B,T,C,H,W = video.shape
        x = video.permute(0,2,1,3,4)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))  # (B,128,T,?,?)
        x = self.pool(x)             # (B,128,T,1,1)
        x = x.squeeze(-1).squeeze(-1).permute(0,2,1)  # (B,T,128)
        return self.proj(x)


def build_video_encoder(cfg: VideoEncoderConfig) -> nn.Module:
    return Simple3DConvEncoder(cfg)

__all__ = ["VideoEncoderConfig", "build_video_encoder", "Simple3DConvEncoder"]
