import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 256, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # x: [B, T, F]
        out, (hn, cn) = self.lstm(x)
        if mask is None:
            feat = out[:, -1]
        else:
            lengths = mask.sum(dim=1).long().clamp(min=1)
            idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(2))
            feat = out.gather(1, idx).squeeze(1)
        return self.head(feat)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # flatten time
        B, T, F = x.shape
        return self.net(x.reshape(B, T * F))
