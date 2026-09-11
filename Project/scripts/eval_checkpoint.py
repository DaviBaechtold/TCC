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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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

    from src.models import torch_compat  # noqa: F401

    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(args.cfg)
    cfg.test_dataloader.dataset.data_root = args.data_root
    cfg.test_dataloader.dataset.data_prefix = dict(img=args.img_prefix)
    cfg.test_dataloader.dataset.ann_file = args.ann_file
    # O avaliador pode ser um dicionário ou uma lista deles, quando mais de uma
    # métrica é reportada lado a lado. Só as que leem anotação têm `ann_file`.
    annotation_path = os.path.join(args.data_root, args.ann_file)
    evaluators = (cfg.test_evaluator if isinstance(cfg.test_evaluator, list)
                  else [cfg.test_evaluator])
    for evaluator in evaluators:
        if 'ann_file' in evaluator:
            evaluator['ann_file'] = annotation_path
    if args.batch_size:
        cfg.test_dataloader.batch_size = args.batch_size
    cfg.val_dataloader = cfg.test_dataloader
    cfg.val_evaluator = cfg.test_evaluator
    cfg.work_dir = os.path.join(args.out_dir, args.tag)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    cfg.load_from = _merged_if_lora(args.ckpt)

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


def _merged_if_lora(checkpoint: str) -> str:
    """Funde adaptadores, se houver, e devolve o caminho a carregar.

    Um checkpoint treinado com LoRA tem nomes de camada diferentes dos do
    modelo comum descrito no config de avaliação. O MMEngine trata chave
    ausente como aviso, de modo que avaliá-lo sem fundir rodaria até o fim e
    reportaria a métrica de um modelo aleatório, sem nada indicando o erro.
    Fundir aqui torna a avaliação correta por construção, e não por o operador
    lembrar de usar o script certo.
    """
    import torch

    from src.models.lora import has_lora_adapters, merge_lora_state_dict

    loaded = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state_dict = loaded.get('state_dict', loaded)
    if not has_lora_adapters(state_dict):
        return checkpoint

    # Ao lado do checkpoint de origem, e não em results/, que é versionado:
    # um modelo fundido tem centenas de megabytes e não é evidência de medição.
    source = Path(checkpoint)
    merged_path = source.with_name(f'{source.stem}_merged.pth')
    torch.save({'state_dict': merge_lora_state_dict(state_dict),
                'meta': loaded.get('meta', {})}, merged_path)
    print(f'  checkpoint com LoRA detectado; fundido em {merged_path}')
    return str(merged_path)


if __name__ == '__main__':
    main()
