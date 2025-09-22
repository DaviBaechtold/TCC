from __future__ import annotations
"""Multimodal 2D->3D lifting model.

Fuses keypoint 2D sequences with optional depth, segmentation and video embeddings
into a latent space and regresses 3D joints.

Input expected (all float32 tensors):
  keypoints: (B,T,J,2)
  depth:     (B,T,Hd,Wd) optional
  mask:      (B,T,Hs,Ws) optional (binary person)
  video:     (B,T,3,Hv,Wv) optional

Output:
  pred_3d: (B,T,J,3)

Simplified latent fusion:
  - Keypoints encoded per joint -> (B,T,J,Dk)
  - Other modalities encoded per frame -> broadcast across joints or concatenated to joint tokens
  - Tokens flattened as sequence [frame * joint] and passed through temporal Transformer encoder.

Config dictionary keys (see configs/multimodal.yaml) controlling construction.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features.depth import DepthEncoderConfig, build_depth_encoder
from ..features.segmentation import SegmentationEncoderConfig, build_segmentation_encoder
from ..features.video_embeddings import VideoEncoderConfig, build_video_encoder


class KeypointEncoder(nn.Module):
    def __init__(self, num_joints: int, in_dim: int = 2, hidden: int = 128, out_dim: int = 256):
        super().__init__()
        self.num_joints = num_joints
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        B,T,J,C = kpts.shape
        h = self.net(kpts)  # (B,T,J,out)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,N,d)
        n = x.shape[1]
        return x + self.pe[:n].unsqueeze(0)


class FusionTransformer(nn.Module):
    def __init__(self, d_model: int, depth: int, heads: int, dropout: float):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model*4,
                                               dropout=dropout, activation='gelu', batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.pos = PositionalEncoding(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B,N,d)
        x = self.pos(tokens)
        return self.enc(x)


class MultiModalLifter(nn.Module):
    def __init__(self, cfg: Dict[str,Any]):
        super().__init__()
        self.cfg = cfg
        J = int(cfg['num_joints'])
        fusion_cfg = cfg['fusion']
        d_model = fusion_cfg['d_model']

        # Keypoints encoder -> produce d_model_k
        self.kp_encoder = KeypointEncoder(J, 2, hidden=min(256,d_model), out_dim=d_model)

        # Optional encoders
        self.depth_encoder = None
        if cfg['modalities']['depth']['enabled']:
            dcfg = DepthEncoderConfig(in_channels=cfg['modalities']['depth'].get('in_channels',1),
                                      out_dim=cfg['modalities']['depth'].get('out_dim',128))
            self.depth_encoder = build_depth_encoder(dcfg)
        self.seg_encoder = None
        if cfg['modalities']['segmentation']['enabled']:
            scfg = SegmentationEncoderConfig(out_dim=cfg['modalities']['segmentation'].get('out_dim',64))
            self.seg_encoder = build_segmentation_encoder(scfg)
        self.video_encoder = None
        if cfg['modalities']['video']['enabled']:
            vcfg = VideoEncoderConfig(out_dim=cfg['modalities']['video'].get('out_dim',256))
            self.video_encoder = build_video_encoder(vcfg)

        # Projection of frame-level embeddings to joint tokens (broadcast + linear)
        total_frame_dim = 0
        if self.depth_encoder: total_frame_dim += cfg['modalities']['depth'].get('out_dim',128)
        if self.seg_encoder: total_frame_dim += cfg['modalities']['segmentation'].get('out_dim',64)
        if self.video_encoder: total_frame_dim += cfg['modalities']['video'].get('out_dim',256)
        self.frame_proj = None
        if total_frame_dim > 0:
            self.frame_proj = nn.Sequential(
                nn.Linear(total_frame_dim, d_model), nn.ReLU(inplace=True),
            )

        self.fusion = FusionTransformer(d_model=d_model, depth=fusion_cfg['depth'], heads=fusion_cfg['heads'], dropout=fusion_cfg['dropout'])
        self.head = nn.Linear(d_model, 3)  # per joint output

    def forward(self, keypoints: torch.Tensor, depth: Optional[torch.Tensor]=None,
                mask: Optional[torch.Tensor]=None, video: Optional[torch.Tensor]=None) -> torch.Tensor:
        # keypoints: (B,T,J,2)
        B,T,J,_ = keypoints.shape
        kp_tokens = self.kp_encoder(keypoints)  # (B,T,J,d)

        frame_feats = []
        if self.depth_encoder and depth is not None:
            frame_feats.append(self.depth_encoder(depth))  # (B,T,Dd)
        if self.seg_encoder and mask is not None:
            frame_feats.append(self.seg_encoder(mask))    # (B,T,Ds)
        if self.video_encoder and video is not None:
            frame_feats.append(self.video_encoder(video)) # (B,T,Dv)
        if len(frame_feats) > 0:
            frame_cat = torch.cat(frame_feats, dim=-1)
            frame_emb = self.frame_proj(frame_cat)  # (B,T,d)
            # broadcast to joints
            frame_emb = frame_emb.unsqueeze(2).expand(B,T,J,frame_emb.shape[-1])
            tokens = kp_tokens + frame_emb
        else:
            tokens = kp_tokens

        # Flatten temporal * joints sequence
        tokens = tokens.reshape(B, T*J, -1)
        fused = self.fusion(tokens)  # (B,T*J,d)
        out = self.head(fused)       # (B,T*J,3)
        return out.reshape(B,T,J,3)


def build_multimodal_lifter(cfg: Dict[str,Any]) -> MultiModalLifter:
    return MultiModalLifter(cfg)

__all__ = ["MultiModalLifter", "build_multimodal_lifter"]
