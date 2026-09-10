"""Compatibilidade do MMPose com tensores em bfloat16.

O MMPose 1.3.2 é anterior ao uso corrente de bfloat16 e converte tensores para
NumPy com `Tensor.numpy()` ao calcular a acurácia de treino. O NumPy não tem um
tipo correspondente ao bfloat16, então a chamada levanta
`TypeError: Got unsupported ScalarType BFloat16` e derruba o treino na primeira
iteração.

A conversão para float32 antes de sair do PyTorch resolve sem efeito colateral:
o valor só é usado para uma métrica de acompanhamento, nunca para o gradiente.

Importar este módulo instala a correção. A operação é idempotente.
"""

from __future__ import annotations

import importlib

import torch

_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    tensor_utils = importlib.import_module('mmpose.utils.tensor_utils')
    original_to_numpy = tensor_utils.to_numpy

    def to_numpy(x, return_device: bool = False, unzip: bool = False):
        if isinstance(x, torch.Tensor) and x.dtype is torch.bfloat16:
            x = x.float()
        elif isinstance(x, (list, tuple)):
            x = type(x)(t.float() if isinstance(t, torch.Tensor)
                        and t.dtype is torch.bfloat16 else t for t in x)
        return original_to_numpy(x, return_device=return_device, unzip=unzip)

    tensor_utils.to_numpy = to_numpy

    # O módulo da cabeça importa o símbolo diretamente, de modo que substituir
    # apenas em `tensor_utils` não alcançaria a chamada real.
    for module_path in ('mmpose.models.heads.coord_cls_heads.rtmw_head',
                        'mmpose.models.heads.coord_cls_heads.rtmcc_head',
                        'mmpose.models.heads.coord_cls_heads.simcc_head'):
        try:
            head_module = importlib.import_module(module_path)
        except ImportError:
            continue
        if hasattr(head_module, 'to_numpy'):
            head_module.to_numpy = to_numpy

    _installed = True


install()
