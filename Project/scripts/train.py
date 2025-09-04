#!/usr/bin/env python
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import DataConfig, SkeletonSequenceDataset, build_splits
from src.models.transformer import TransformerClassifier
from src.models.baselines import LSTMClassifier, MLPClassifier
from src.utils.train_utils import load_yaml, ensure_dir, TrainState, train_one_epoch, evaluate, save_checkpoint


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    torch.manual_seed(cfg.get('seed', 42))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_cfg_raw = cfg['data']
    manifest = data_cfg_raw['manifest']
    # split manifests
    tr_csv, va_csv, te_csv = build_splits(manifest, cfg['eval']['val_split'], cfg['eval']['test_split'], seed=cfg.get('seed', 42))

    seq_len = data_cfg_raw['seq_len']
    data_cfg = DataConfig(
        manifest=tr_csv, seq_len=seq_len, normalize=data_cfg_raw.get('normalize', True),
        noise_std=cfg['data']['augment'].get('noise_std', 0.0),
        time_mask_prob=cfg['data']['augment'].get('time_mask_prob', 0.0),
        time_mask_len=cfg['data']['augment'].get('time_mask_len', 0),
    )

    # Learn classes from train set
    train_ds = SkeletonSequenceDataset(data_cfg)
    data_cfg_val = DataConfig(manifest=va_csv, seq_len=seq_len, normalize=data_cfg_raw.get('normalize', True))
    data_cfg_test = DataConfig(manifest=te_csv, seq_len=seq_len, normalize=data_cfg_raw.get('normalize', True))
    val_ds = SkeletonSequenceDataset(data_cfg_val, class_to_idx=train_ds.class_to_idx)
    test_ds = SkeletonSequenceDataset(data_cfg_test, class_to_idx=train_ds.class_to_idx)

    input_dim = train_ds[0]['x'].shape[-1]
    num_classes = train_ds.num_classes

    model_cfg = cfg['model']
    model = build_model(model_cfg.get('type', 'transformer'), input_dim, num_classes, model_cfg)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg['optim']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['optim']['batch_size'], shuffle=False, num_workers=2, pin_memory=True)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg['optim']['lr'], weight_decay=cfg['optim'].get('weight_decay', 0.0))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, cfg['optim']['epochs'] - cfg['optim'].get('warmup_epochs', 0)))

    work_dir = cfg.get('work_dir', 'runs/transformer')
    ensure_dir(work_dir)
    best_path = Path(work_dir) / 'best.pt'

    state = TrainState()
    for epoch in range(cfg['optim']['epochs']):
        print(f"Epoch {epoch+1}/{cfg['optim']['epochs']}")
        tr_m = train_one_epoch(model, train_loader, optim, device)
        va_m, _ = evaluate(model, val_loader, device)
        print(f"train: {tr_m} val: {va_m}")
        sched.step()
        state.epoch = epoch + 1
        if va_m['acc'] > state.best_acc:
            state.best_acc = va_m['acc']
            save_checkpoint(str(best_path), model, state, extra={"class_to_idx": train_ds.class_to_idx})
            print(f"Saved new best to {best_path}")

    # final eval on test
    test_loader = DataLoader(test_ds, batch_size=cfg['optim']['batch_size'], shuffle=False)
    te_m, cm = evaluate(model, test_loader, device)
    print(f"test: {te_m}\nconfusion:\n{cm}")


if __name__ == '__main__':
    main()
