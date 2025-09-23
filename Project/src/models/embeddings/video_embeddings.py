"""
Extração de embeddings de vídeo para análise temporal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


class VideoEmbeddingExtractor(nn.Module):
    """
    Extrator de embeddings de vídeo para capturar informações temporais.
    Usa uma arquitetura 3D CNN ou transformer para análise temporal.
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 temporal_depth: int = 16,
                 spatial_size: int = 112,
                 architecture: str = "3dcnn"):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.temporal_depth = temporal_depth
        self.spatial_size = spatial_size
        self.architecture = architecture
        
        if architecture == "3dcnn":
            self.backbone = self._build_3dcnn_backbone()
        elif architecture == "transformer":
            self.backbone = self._build_transformer_backbone()
        else:
            raise ValueError(f"Arquitetura não suportada: {architecture}")
    
    def _build_3dcnn_backbone(self) -> nn.Module:
        """Constrói backbone 3D CNN para extração de features temporais."""
        return nn.Sequential(
            # Primeira camada 3D
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Bloco 1
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            # Bloco 2
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            # Bloco 3
            nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, self.embedding_dim)
        )
    
    def _build_transformer_backbone(self) -> nn.Module:
        """Constrói backbone Transformer para análise temporal."""
        # Encoder CNN para features espaciais
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Transformer para análise temporal
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        
        return nn.Sequential(
            nn.TransformerEncoder(encoder_layer, num_layers=3),
            nn.Linear(256, self.embedding_dim)
        )
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extrai embeddings de uma sequência de vídeo.
        
        Args:
            video: Tensor de vídeo (B, T, 3, H, W)
            
        Returns:
            embeddings: Video embeddings (B, embedding_dim)
        """
        B, T, C, H, W = video.shape
        
        if self.architecture == "3dcnn":
            # Redimensionar para entrada 3D CNN
            video_resized = F.interpolate(
                video.view(B * T, C, H, W),
                size=(self.spatial_size, self.spatial_size),
                mode='bilinear'
            ).view(B, T, C, self.spatial_size, self.spatial_size)
            
            # Ajustar profundidade temporal
            if T != self.temporal_depth:
                indices = torch.linspace(0, T-1, self.temporal_depth).long()
                video_resized = video_resized[:, indices]
            
            # Reordenar para (B, C, T, H, W) para 3D CNN
            video_3d = video_resized.permute(0, 2, 1, 3, 4)
            
            embeddings = self.backbone(video_3d)
            
        else:  # transformer
            # Extrair features espaciais para cada frame
            video_flat = video.view(B * T, C, H, W)
            spatial_features = self.spatial_encoder(video_flat)  # (B*T, 256)
            spatial_features = spatial_features.view(B, T, -1)  # (B, T, 256)
            
            # Análise temporal com transformer
            temporal_features = self.backbone[0](spatial_features)  # (B, T, 256)
            
            # Pool temporal (média ou último token)
            pooled_features = temporal_features.mean(dim=1)  # (B, 256)
            
            # Projeção final
            embeddings = self.backbone[1](pooled_features)
        
        return embeddings
    
    def extract_frame_features(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extrai features para cada frame individualmente.
        
        Args:
            video: Tensor de vídeo (B, T, 3, H, W)
            
        Returns:
            frame_features: Features por frame (B, T, embedding_dim)
        """
        if self.architecture != "transformer":
            raise NotImplementedError("Extração por frame disponível apenas para arquitetura transformer")
        
        B, T, C, H, W = video.shape
        
        # Extrair features espaciais
        video_flat = video.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(video_flat)  # (B*T, 256)
        spatial_features = spatial_features.view(B, T, -1)  # (B, T, 256)
        
        # Aplicar transformer
        temporal_features = self.backbone[0](spatial_features)  # (B, T, 256)
        
        # Projeção para cada frame
        frame_features = self.backbone[1](temporal_features)  # (B, T, embedding_dim)
        
        return frame_features


class MotionEmbedding(nn.Module):
    """
    Embedding especializado para captura de movimento entre frames.
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 flow_method: str = "optical_flow"):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.flow_method = flow_method
        
        # Encoder para optical flow
        self.flow_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim)
        )
    
    def compute_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Computa optical flow entre dois frames.
        
        Args:
            frame1, frame2: Frames consecutivos
            
        Returns:
            flow: Optical flow (H, W, 2)
        """
        if not HAS_OPENCV:
            raise ImportError("opencv-python is required for optical flow computation")
        
        # Converter para grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # Calcular optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
        
        return flow
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Extrai embeddings de movimento de uma sequência de frames.
        
        Args:
            frames: Sequência de frames (B, T, 3, H, W)
            
        Returns:
            motion_embeddings: Embeddings de movimento (B, T-1, embedding_dim)
        """
        B, T, C, H, W = frames.shape
        motion_embeddings = []
        
        for t in range(T - 1):
            # Placeholder para optical flow
            # Em implementação completa, computar optical flow real
            flow = torch.randn(B, 2, H, W).to(frames.device)
            
            # Redimensionar se necessário
            flow_resized = F.interpolate(flow, size=(224, 224), mode='bilinear')
            
            # Extrair embedding
            motion_emb = self.flow_encoder(flow_resized)
            motion_embeddings.append(motion_emb)
        
        return torch.stack(motion_embeddings, dim=1)


class MultiScaleVideoEmbedding(nn.Module):
    """
    Extração de embeddings em múltiplas escalas temporais.
    """
    
    def __init__(self, 
                 embedding_dim: int = 256,
                 temporal_scales: List[int] = [4, 8, 16]):
        super().__init__()
        
        self.temporal_scales = temporal_scales
        self.scale_extractors = nn.ModuleList([
            VideoEmbeddingExtractor(
                embedding_dim=embedding_dim // len(temporal_scales),
                temporal_depth=scale
            ) for scale in temporal_scales
        ])
        
        self.fusion = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Extrai embeddings em múltiplas escalas temporais.
        
        Args:
            video: Tensor de vídeo (B, T, 3, H, W)
            
        Returns:
            multi_scale_embeddings: Embeddings multi-escala (B, embedding_dim)
        """
        scale_embeddings = []
        
        for extractor in self.scale_extractors:
            emb = extractor(video)
            scale_embeddings.append(emb)
        
        # Concatenar embeddings de diferentes escalas
        combined = torch.cat(scale_embeddings, dim=1)
        
        # Fusão final
        fused_embeddings = self.fusion(combined)
        
        return fused_embeddings