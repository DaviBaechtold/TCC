from typing import Dict
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    out = {
        "acc": float(accuracy_score(y_true_np, y_pred_np)),
        "f1": float(f1_score(y_true_np, y_pred_np, average='macro')),
    }
    return out


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor):
    return confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
