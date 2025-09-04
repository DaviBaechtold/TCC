import numpy as np
import torch
from src.models.transformer import TransformerClassifier


def test_model_shapes():
    B, T, F, C = 2, 16, 21, 2
    x = torch.randn(B, T, F*C)
    m = torch.ones(B, T)
    model = TransformerClassifier(input_dim=F*C, num_classes=5)
    y = model(x, m)
    assert y.shape == (B, 5)
