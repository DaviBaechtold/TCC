#!/usr/bin/env python
"""Verifica que fundir os adaptadores preserva exatamente a saída do modelo.

A fusão é a etapa que torna um checkpoint treinado com LoRA avaliável por um
config comum. Se ela estiver errada, nada quebra: o modelo carrega, a avaliação
roda e reporta um número plausível porém falso. Este teste é a única barreira
contra esse modo de falha, e por isso exercita as três formas de camada que a
injeção produz — convolução com passo, convolução ponto-a-ponto e linear.

Executar:  python tests/test_lora_merge.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.lora import LoRAConv2d, LoRALinear, merge_lora_state_dict

# Erro aceitável para float32 acumulado sobre algumas centenas de produtos.
TOLERANCE = 1e-5


def _rebuild_plain(base: nn.Module) -> nn.Module:
    """Cria a camada original, sem adaptador, com a mesma geometria."""
    if isinstance(base, nn.Linear):
        return nn.Linear(base.in_features, base.out_features)
    return nn.Conv2d(base.in_channels, base.out_channels, base.kernel_size,
                     stride=base.stride, padding=base.padding)


def _assert_merge_preserves_output(base: nn.Module, sample: torch.Tensor,
                                   rank: int = 4) -> float:
    wrapper_type = LoRALinear if isinstance(base, nn.Linear) else LoRAConv2d
    wrapper = wrapper_type(base, rank, alpha=float(rank))

    # `up` nasce em zero para que o modelo adaptado seja idêntico ao original
    # no passo zero. Testar nesse estado não exercitaria a fusão coisa alguma.
    nn.init.normal_(wrapper.up.weight, std=0.4)
    wrapper.eval()

    with torch.no_grad():
        expected = wrapper(sample)

    merged = merge_lora_state_dict(
        {f'layer.{name}': value for name, value in wrapper.state_dict().items()})

    plain = _rebuild_plain(base)
    plain.load_state_dict({name.removeprefix('layer.'): value
                           for name, value in merged.items()})
    plain.eval()
    with torch.no_grad():
        actual = plain(sample)

    error = (expected - actual).abs().max().item()
    assert error < TOLERANCE, f'{type(base).__name__}: erro {error:.2e}'
    return error


def main():
    torch.manual_seed(0)
    cases = [
        ('conv com passo', nn.Conv2d(8, 12, 3, stride=2, padding=1),
         torch.randn(2, 8, 16, 16)),
        ('conv ponto-a-ponto', nn.Conv2d(6, 6, 1), torch.randn(2, 6, 5, 5)),
        ('linear', nn.Linear(10, 7), torch.randn(3, 10)),
    ]
    for label, base, sample in cases:
        error = _assert_merge_preserves_output(base, sample)
        print(f'  {label:20s} erro máximo {error:.2e}  OK')
    print('\nfusão de adaptadores preserva a saída em todas as formas de camada')


if __name__ == '__main__':
    main()
