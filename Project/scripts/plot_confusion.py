#!/usr/bin/env python
"""
Plot and save a confusion matrix for a given checkpoint and dataset split/manifest.

Examples:
  python Project/scripts/plot_confusion.py \
    --config Project/configs/transformer.yaml \
    --checkpoint "/media/davs/SSD/TCC - Database/processed/runs/transformer/best.pt" \
    --split val --out cm_val.png

  # Or provide a manifest explicitly
  python Project/scripts/plot_confusion.py \
    --config Project/configs/transformer.yaml \
    --checkpoint "/media/davs/SSD/TCC - Database/processed/runs/transformer/best.pt" \
    --manifest "/media/davs/SSD/TCC - Database/processed/manifest_full_val.csv" --out cm_val.png
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'Project/src' is on sys.path when executing as a script
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.train_utils import load_yaml, load_checkpoint, evaluate  # noqa: E402
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
    ap.add_argument('--manifest', default='', help='Optional explicit manifest CSV to evaluate on')
    ap.add_argument('--out', default='cm.png')
    ap.add_argument('--normalize', choices=['none', 'true', 'pred'], default='none',
                    help='Normalize confusion by rows (true) or columns (pred)')
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

    # pick eval manifest
    if args.manifest:
        eval_csv = args.manifest
        # attempt to get a train csv alongside for class mapping
        p = Path(eval_csv)
        candidate_train = str(p.with_name(p.stem.replace('_val', '').replace('_test', '') + '_train.csv'))
        train_csv = candidate_train if Path(candidate_train).exists() else eval_csv
    else:
        if args.split == 'train':
            eval_csv = tr_csv
        elif args.split == 'val':
            eval_csv = va_csv
        else:
            eval_csv = te_csv
        train_csv = tr_csv

    seq_len = int(cfg['data']['seq_len'])
    normalize = bool(cfg['data'].get('normalize', True))

    # datasets with consistent class mapping
    train_ds = SkeletonSequenceDataset(DataConfig(train_csv, seq_len, normalize=normalize))
    eval_ds = SkeletonSequenceDataset(DataConfig(eval_csv, seq_len, normalize=normalize), class_to_idx=train_ds.class_to_idx)

    input_dim = eval_ds[0]['x'].shape[-1]
    num_classes = train_ds.num_classes

    model_cfg = cfg['model']
    model = build_model(model_cfg.get('type', 'transformer'), input_dim, num_classes, model_cfg)
    _state, extra = load_checkpoint(args.checkpoint, model)
    model.to(device)

    loader = DataLoader(eval_ds, batch_size=int(cfg['optim'].get('batch_size', 64)), shuffle=False, num_workers=2,
                        pin_memory=(device == 'cuda'))

    mets, cm_t = evaluate(model, loader, device)
    cm = cm_t.numpy()

    # normalization
    if args.normalize == 'true':
        cm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1e-9, None)
    elif args.normalize == 'pred':
        cm = cm / np.clip(cm.sum(axis=0, keepdims=True), 1e-9, None)

    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    labels = [idx_to_class[i] for i in range(num_classes)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels, square=True)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f"Confusion ({args.split}) | acc={mets['acc']:.3f} f1={mets['f1']:.3f}")
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == '__main__':
    main()
