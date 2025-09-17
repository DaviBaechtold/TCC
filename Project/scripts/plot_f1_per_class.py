#!/usr/bin/env python
"""
Compute and plot per-class F1 scores for a given checkpoint and split.

Example:
  CUDA_VISIBLE_DEVICES="" python Project/scripts/plot_f1_per_class.py \
    --config Project/configs/transformer.yaml \
    --checkpoint "/media/davs/SSD/TCC - Database/processed/runs/transformer/best.pt" \
    --split val --out Doc/Slides/f1_val.png
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.train_utils import load_yaml, load_checkpoint  # noqa: E402
from src.data.dataset import DataConfig, SkeletonSequenceDataset  # noqa: E402
from src.models.transformer import TransformerClassifier  # noqa: E402
from src.models.baselines import LSTMClassifier, MLPClassifier  # noqa: E402


def build_model(name: str, input_dim: int, num_classes: int, cfg: dict):
    if name == 'transformer':
        return TransformerClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            d_model=cfg.get('d_model', 192),
            nhead=cfg.get('nhead', 6),
            num_layers=cfg.get('num_layers', 4),
            dim_feedforward=cfg.get('dim_feedforward', 512),
            dropout=cfg.get('dropout', 0.1),
        )
    if name == 'lstm':
        return LSTMClassifier(input_dim=input_dim, num_classes=num_classes)
    if name == 'mlp':
        return MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def derive_split_paths(base_manifest: str):
    p = Path(base_manifest)
    stem = p.stem
    return (
        str(p.with_name(stem + '_train.csv')),
        str(p.with_name(stem + '_val.csv')),
        str(p.with_name(stem + '_test.csv')),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    ap.add_argument('--out', default='f1.png')
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = 'cpu'
    if torch.cuda.is_available():
        try:
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 7:
                device = 'cuda'
        except Exception:
            pass

    base_manifest = cfg['data']['manifest']
    tr_csv, va_csv, te_csv = derive_split_paths(base_manifest)
    if args.split == 'train':
        eval_csv = tr_csv
        train_csv = tr_csv
    elif args.split == 'val':
        eval_csv = va_csv
        train_csv = tr_csv
    else:
        eval_csv = te_csv
        train_csv = tr_csv

    seq_len = int(cfg['data']['seq_len'])
    normalize = bool(cfg['data'].get('normalize', True))

    train_ds = SkeletonSequenceDataset(DataConfig(train_csv, seq_len, normalize=normalize))
    eval_ds = SkeletonSequenceDataset(DataConfig(eval_csv, seq_len, normalize=normalize), class_to_idx=train_ds.class_to_idx)

    input_dim = eval_ds[0]['x'].shape[-1]
    num_classes = train_ds.num_classes
    model_cfg = cfg['model']
    model = build_model(model_cfg.get('type', 'transformer'), input_dim, num_classes, model_cfg)
    _state, _extra = load_checkpoint(args.checkpoint, model)
    model.to(device)

    loader = DataLoader(eval_ds, batch_size=int(cfg['optim'].get('batch_size', 64)), shuffle=False, num_workers=2,
                        pin_memory=(device == 'cuda'))

    model.eval()
    all_y = []
    all_p = []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            m = batch['mask'].to(device)
            y = batch['y'].to(device)
            logits = model(x, m)
            pred = logits.argmax(dim=1)
            all_y.append(y.cpu().numpy())
            all_p.append(pred.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    f1s = f1_score(y_true, y_pred, average=None, labels=list(range(num_classes)))
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(num_classes)]

    plt.figure(figsize=(10, 4))
    order = np.argsort(f1s)[::-1]
    plt.bar(np.arange(num_classes), f1s[order])
    plt.xticks(np.arange(num_classes), [labels[i] for i in order], rotation=60, ha='right')
    plt.ylim(0, 1)
    plt.ylabel('F1 por classe')
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved per-class F1 to {out_path}")


if __name__ == '__main__':
    main()
