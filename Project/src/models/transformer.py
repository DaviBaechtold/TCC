from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, T, D]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model)
        self.use_cls = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B, T, F], mask: [B, T] (1 valid, 0 pad)
        x = self.proj(x)
        x = self.pos(x)
        if self.use_cls:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # [B, 1+T, D]
            if mask is not None:
                mask = torch.cat([torch.ones(B, 1, device=x.device), mask], dim=1)
        # Transformer expects True for pads in src_key_padding_mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = (mask == 0)
        z = self.encoder(x, src_key_padding_mask=key_padding_mask)
        if self.use_cls:
            pooled = z[:, 0]
        else:
            # mean over valid tokens
            if mask is None:
                pooled = z.mean(dim=1)
            else:
                denom = mask.sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.head(self.norm(pooled))
        return logits
