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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    p.add_argument('--lora-rank', type=int, default=None,
                   help='Sobrescreve o posto declarado no config')
    p.add_argument('--no-lora', action='store_true',
                   help='Ignora a seção `lora` do config e treina tudo')
    p.add_argument('--log-interval', type=int, default=None,
                   help='Iterações entre registros. Use 1 para diagnosticar '
                        'em que passo exato a perda diverge')
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
    if args.log_interval:
        cfg.default_hooks.logger.interval = args.log_interval
        # A janela de suavização precisa acompanhar: com janela 50 e registro a
        # cada iteração, um único NaN contamina os cinquenta registros seguintes
        # e esconde em qual passo ele apareceu.
        cfg.log_processor.window_size = args.log_interval

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
    _apply_lora_if_requested(cfg, args, runner)
    runner.train()


def _apply_lora_if_requested(cfg, args, runner):
    """Carrega o checkpoint e injeta adaptadores de posto baixo, nesta ordem.

    A ordem é o ponto crítico. `Runner.from_cfg` constrói o modelo mas **não**
    carrega `load_from`: quem carrega é `train()`, mais tarde. Injetar os
    adaptadores antes disso renomeia os parâmetros — `backbone.stem.0.conv`
    passa a ser `backbone.stem.0.conv.base` — e o carregamento posterior falha
    silenciosamente em quase todas as chaves, deixando o modelo treinar a partir
    de inicialização aleatória. O sintoma é discreto: o treino roda, a perda cai,
    e apenas a acurácia denuncia (0,05 em vez de 0,98).

    Por isso o checkpoint é carregado aqui, explicitamente, antes da injeção, e
    o Runner é informado de que não há mais nada a carregar.
    """
    lora_cfg = cfg.get('lora')
    if args.no_lora or not lora_cfg:
        return

    # O MMPose 1.3.2 não converte bfloat16 para NumPy ao medir acurácia.
    from src.models import bf16_compat  # noqa: F401
    from src.models.lora import (adapter_output_magnitude, freeze_except_lora,
                                 inject_lora, parameter_summary)

    model = runner.model
    device = next(model.parameters()).device

    if runner._load_from:
        runner.load_checkpoint(runner._load_from, map_location='cpu')
        runner._load_from = None  # já carregado; evita recarga pós-injeção

    reference = _weight_fingerprint(model)

    rank = args.lora_rank or lora_cfg.get('rank', 16)
    adapted = inject_lora(model, rank=rank,
                          include=tuple(lora_cfg.get('include', ('backbone',))))
    frozen_norms = freeze_except_lora(
        model, trainable_prefixes=tuple(lora_cfg.get('trainable', ('head',))))
    model.to(device)

    # Verifica que os pesos pré-treinados sobreviveram ao envelopamento. É o
    # teste que teria apanhado a inversão de ordem descrita acima.
    after = _weight_fingerprint(model)
    if reference is None or after is None or abs(reference - after) > 1e-6:
        raise RuntimeError(
            'Os pesos do backbone mudaram durante a injeção do LoRA '
            f'({reference} -> {after}). O checkpoint provavelmente não foi '
            'carregado antes da injeção.')

    # `Runner.train()` chama `_init_model_weights()` depois desta função, o que
    # reinicializaria tanto os pesos vindos do checkpoint quanto a projeção de
    # saída dos adaptadores, que precisa começar em zero. Como o modelo já está
    # inicializado, a chamada é neutralizada em vez de tolerada.
    model.init_weights = _weights_already_loaded

    runner.register_hook(_build_adapters_pristine_hook(skip=args.resume))

    summary = parameter_summary(model)
    print(f'  checkpoint    carregado antes da injeção, pesos preservados '
          f'(norma {after:.4f})')
    print(f'  LoRA          posto {rank}, {sum(adapted.values())} camadas '
          f'adaptadas {dict(adapted)}')
    print(f'  normalização  {frozen_norms} camadas congeladas em modo eval')
    print(f'  init_weights  neutralizado (pesos vêm do checkpoint)')
    print(f'  parâmetros    {summary["trainable_M"]:.1f}M treináveis de '
          f'{summary["total_M"]:.1f}M ({summary["trainable_pct"]:.1f}%)')
    print('=' * 72)


def _weights_already_loaded(*_args, **_kwargs):
    """Substitui `init_weights` num modelo cujos pesos já vieram do checkpoint."""
    return None


def _build_adapters_pristine_hook(skip: bool):
    """Hook que verifica, ao iniciar o treino, se os adaptadores seguem íntegros.

    Entre a injeção e o primeiro passo o MMEngine executa uma sequência que já
    corrompeu os adaptadores duas vezes durante o desenvolvimento: o
    carregamento do checkpoint, que renomeia chaves, e `init_weights`, que
    sobrescreve pesos. Ambos falhavam em silêncio — o treino rodava, a perda
    caía, e só a acurácia denunciava. Esta verificação converte esse modo de
    falha em erro imediato, e vale mantê-la mesmo com a ordem hoje correta,
    porque ela depende de detalhes internos do framework.
    """
    from mmengine.hooks import Hook

    class AdaptersPristineHook(Hook):
        priority = 'HIGHEST'

        def before_train(self, runner):
            if skip:
                return
            from src.models.lora import adapter_output_magnitude

            magnitude = adapter_output_magnitude(runner.model)
            if magnitude != 0.0:
                raise RuntimeError(
                    'Os adaptadores LoRA foram reinicializados entre a injeção '
                    'e o início do treino (soma das projeções de saída = '
                    f'{magnitude:.1f}, esperado 0). O modelo adaptado deixou de '
                    'ser idêntico ao checkpoint e o treino divergiria.')
            runner.logger.info(
                'Adaptadores LoRA íntegros: projeções de saída ainda em zero.')

    return AdaptersPristineHook()


def _weight_fingerprint(model):
    """Norma de um peso profundo do backbone, usada como assinatura.

    Uma camada profunda é preferível à primeira: ela é sensível a qualquer
    reinicialização e não é afetada por adaptações de canal de entrada.
    """
    import torch

    for name, parameter in model.backbone.named_parameters():
        if name.endswith('stage4.0.conv.weight') or name.endswith(
                'stage4.0.conv.base.weight'):
            return float(torch.linalg.vector_norm(parameter.detach()).cpu())
    return None


if __name__ == '__main__':
    main()
