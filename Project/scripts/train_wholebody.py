#!/usr/bin/env python
"""Treina RTMPose WholeBody, com suporte a reescalar o cronograma de épocas.

O `--epochs` reescala de forma consistente o max_epochs, o ponto de início do
cosine annealing, a época de troca de pipeline e o intervalo de validação. Isso
permite rodar um ensaio curto que é uma versão comprimida — e não truncada — do
treino longo, mantendo a comparação honesta.

Exemplos:
    # Ensaio curto (~1h)
    python scripts/train_wholebody.py \
        --config configs/rtmpose_m_wholebody_gray_ft.py \
        --epochs 10 --work-dir work_dirs/ft_smoke

    # Treino completo
    python scripts/train_wholebody.py \
        --config configs/rtmpose_m_wholebody_gray_ft.py
"""

import argparse
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--work-dir', default=None)
    p.add_argument('--epochs', type=int, default=None,
                   help='Reescala todo o cronograma para este total de épocas')
    p.add_argument('--val-interval', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--load-from', default=None)
    return p.parse_args()


def rescale_schedule(cfg, epochs):
    """Comprime o cronograma de treino para `epochs` mantendo as proporções."""
    original = cfg.train_cfg.max_epochs
    ratio = epochs / original

    cfg.train_cfg.max_epochs = epochs

    for sched in cfg.param_scheduler:
        if sched.get('type') != 'CosineAnnealingLR':
            continue
        begin = max(0, int(round(sched.get('begin', 0) * ratio)))
        sched['begin'] = begin
        sched['end'] = epochs
        sched['T_max'] = max(1, epochs - begin)

    for hook in cfg.get('custom_hooks', []):
        if 'PipelineSwitchHook' in str(hook.get('type', '')):
            # Mantém a proporção de épocas no estágio 2, com no mínimo 1.
            stage2 = max(1, int(round((original - hook['switch_epoch']) * ratio)))
            hook['switch_epoch'] = max(1, epochs - stage2)

    # Não faz sentido validar a cada 5 épocas num treino de 10.
    cfg.train_cfg.val_interval = max(1, min(cfg.train_cfg.val_interval,
                                            epochs // 2))
    if 'checkpoint' in cfg.default_hooks:
        cfg.default_hooks.checkpoint.interval = cfg.train_cfg.val_interval

    return cfg


def main():
    args = parse_args()

    import torch
    import numpy as np

    # Checkpoints do OpenMMLab carregam objetos numpy; PyTorch >= 2.6 usa
    # weights_only=True por padrão e os rejeita.
    _orig_load = torch.load

    def _load_trusted(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig_load(*a, **kw)

    torch.load = _load_trusted

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.config)

    if args.work_dir:
        cfg.work_dir = args.work_dir
    if args.load_from:
        cfg.load_from = args.load_from
    if args.resume:
        cfg.resume = True
    if args.batch_size:
        cfg.train_dataloader.batch_size = args.batch_size
    if args.lr:
        cfg.optim_wrapper.optimizer.lr = args.lr
        for sched in cfg.param_scheduler:
            if sched.get('type') == 'CosineAnnealingLR':
                sched['eta_min'] = args.lr * 0.02
    if args.epochs:
        cfg = rescale_schedule(cfg, args.epochs)
    if args.val_interval:
        cfg.train_cfg.val_interval = args.val_interval

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    switch = next((h['switch_epoch'] for h in cfg.get('custom_hooks', [])
                   if 'PipelineSwitchHook' in str(h.get('type', ''))), None)
    cosine = next((s for s in cfg.param_scheduler
                   if s.get('type') == 'CosineAnnealingLR'), {})

    print('=' * 72)
    print(f'  config        {args.config}')
    print(f'  work_dir      {cfg.work_dir}')
    print(f'  load_from     {cfg.get("load_from")}')
    print(f'  epochs        {cfg.train_cfg.max_epochs}'
          f'  (val a cada {cfg.train_cfg.val_interval})')
    print(f'  batch / lr    {cfg.train_dataloader.batch_size}'
          f' / {cfg.optim_wrapper.optimizer.lr}')
    print(f'  cosine        épocas {cosine.get("begin")}–{cosine.get("end")}'
          f'  eta_min={cosine.get("eta_min")}')
    print(f'  stage2 aug    a partir da época {switch}')
    print('=' * 72)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
