#!/usr/bin/env python
import argparse
from pathlib import Path
import sys
import os

import torch
from torch.utils.data import DataLoader

# Ensure 'Project/src' is on sys.path when executing as a script
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import DataConfig, SkeletonSequenceDataset, build_splits
from src.models.transformer import TransformerClassifier
from src.models.baselines import LSTMClassifier, MLPClassifier
from src.utils.train_utils import load_yaml, ensure_dir, TrainState, train_one_epoch, evaluate, save_checkpoint, load_checkpoint


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
    ap.add_argument('--resume', default='', help="Path to checkpoint to resume from, or 'best'/'last' to use from work_dir")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    torch.manual_seed(int(cfg.get('seed', 42)))
    # Prefer CPU if GPU arch is unsupported by installed torch build
    device = 'cpu'
    if torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability(0)
            if major >= 7:
                device = 'cuda'
        except Exception:
            pass

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

    bs = int(cfg['optim']['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=(device=='cuda'))

    lr = float(cfg['optim']['lr'])
    wd = float(cfg['optim'].get('weight_decay', 0.0))
    epochs = int(cfg['optim']['epochs'])
    warmup = int(cfg['optim'].get('warmup_epochs', 0))
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, epochs - warmup))

    work_dir = cfg.get('work_dir', 'runs/transformer')
    ensure_dir(work_dir)
    best_path = Path(work_dir) / 'best.pt'
    last_path = Path(work_dir) / 'last.pt'

    # Optionally resume
    state = TrainState()
    start_epoch = 0
    if args.resume:
        if args.resume.lower() == 'best':
            resume_path = best_path
        elif args.resume.lower() == 'last':
            resume_path = last_path
        else:
            resume_path = Path(args.resume)
        if resume_path.exists():
            try:
                state, extra = load_checkpoint(str(resume_path), model)
                # Best acc is stored in state; use it to keep tracking improvements
                start_epoch = int(state.epoch)
                # Optional: sanity-check class mapping
                ckpt_cti = extra.get('class_to_idx') if isinstance(extra, dict) else None
                if ckpt_cti and ckpt_cti != train_ds.class_to_idx:
                    print('WARN: class_to_idx in checkpoint differs from current dataset. Continuing anyway.')
                print(f"Resumed from {resume_path} at epoch {start_epoch} (best_acc={state.best_acc:.4f})")
            except Exception as e:
                print(f"WARN: failed to resume from {resume_path}: {e}")
        else:
            print(f"WARN: resume checkpoint not found: {resume_path}")

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        tr_m = train_one_epoch(model, train_loader, optim, device)
        va_m, _ = evaluate(model, val_loader, device)
        print(f"train: {tr_m} val: {va_m}")
        sched.step()
        state.epoch = epoch + 1
        # Always save rolling last checkpoint
        save_checkpoint(str(last_path), model, state, extra={"class_to_idx": train_ds.class_to_idx})
        if va_m['acc'] > state.best_acc:
            state.best_acc = va_m['acc']
            save_checkpoint(str(best_path), model, state, extra={"class_to_idx": train_ds.class_to_idx})
            print(f"Saved new best to {best_path}")

    # final eval on test (skip if no samples)
    if len(test_ds) > 0:
        test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)
        te_m, cm = evaluate(model, test_loader, device)
        print(f"test: {te_m}\nconfusion:\n{cm}")
    else:
        print("No test samples found; skipping final evaluation.")


if __name__ == '__main__':
    main()
