#!/usr/bin/env python
"""Avalia um checkpoint RTMPose em um conjunto de validação COCO-WholeBody.

Permite trocar o diretório de imagens sem editar o config, o que torna possível
medir o mesmo checkpoint em RGB e em grayscale e isolar o domain gap.

Exemplo:
    python scripts/eval_checkpoint.py \
        --ckpt checkpoints/rtmpose-m_wholebody_official_256x192.pth \
        --data-root data/processed/grayscale/ \
        --tag official_gray
"""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cfg', default='configs/eval/rtmpose_m_wholebody_eval.py')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data-root', default='data/processed/grayscale/')
    p.add_argument('--img-prefix', default='val2017/')
    p.add_argument('--ann-file', default='annotations/coco_wholebody_val_v1.0.json')
    p.add_argument('--tag', required=True, help='Nome curto para o arquivo de resultado')
    # Resultados de medição são evidência das afirmações do documento, então
    # ficam versionados em results/, e não em work_dirs/, que é descartável.
    p.add_argument('--out-dir', default='results/baselines')
    p.add_argument('--batch-size', type=int, default=None,
                   help='Sobrescreve o batch do config (modelos maiores pedem menos)')
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    import numpy as np

    # Checkpoints do OpenMMLab (2023) carregam objetos numpy; PyTorch >= 2.6
    # usa weights_only=True por padrão e os rejeita.
    _orig_load = torch.load

    def _load_trusted(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig_load(*a, **kw)

    torch.load = _load_trusted

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.cfg)
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.data_prefix = dict(img=args.img_prefix)
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    cfg.test_evaluator.ann_file = os.path.join(args.data_root, args.ann_file)
    if args.batch_size:
        cfg.test_dataloader.batch_size = args.batch_size
    cfg.val_dataloader = cfg.test_dataloader
    cfg.val_evaluator = cfg.test_evaluator
    cfg.load_from = args.ckpt
    cfg.work_dir = os.path.join(args.out_dir, args.tag)

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()

    out = Path(args.out_dir) / f'{args.tag}.json'
    payload = {
        'tag': args.tag,
        'checkpoint': args.ckpt,
        'data_root': args.data_root,
        'metrics': {k: float(v) for k, v in metrics.items()},
    }
    out.write_text(json.dumps(payload, indent=2))

    print(f'\n{"=" * 70}')
    print(f'  {args.tag}   ({args.data_root})')
    print(f'{"=" * 70}')
    for k, v in sorted(metrics.items()):
        print(f'  {k:<32} {v:.4f}')
    print(f'\nSalvo em: {out}')


if __name__ == '__main__':
    main()
