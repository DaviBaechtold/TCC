"""Substitui operadores compilados do MMCV por equivalentes do torchvision.

O MMCV distribui parte de seus operadores como extensão C++/CUDA compilada. A
instalação deste projeto usa uma combinação de PyTorch e CUDA para a qual não
existe wheel pré-compilada, então `mmcv._ext` carrega como stub: o import passa,
mas qualquer chamada a um operador levanta erro em tempo de execução. É por isso
que o caminho com detector de pessoas falhava apenas na primeira inferência, e
não no carregamento do modelo.

Compilar o MMCV localmente resolveria, mas custa dezenas de minutos e volta a
quebrar a cada atualização do PyTorch. Como o único operador de que o pipeline
top-down precisa é o NMS, e o torchvision oferece uma implementação CUDA
equivalente e amplamente testada, a substituição é preferível.

Importar este módulo instala o substituto. A operação é idempotente.
"""

from __future__ import annotations

import importlib

import torch
from torchvision.ops import nms as torchvision_nms

_installed = False


def _nms(boxes: torch.Tensor,
         scores: torch.Tensor,
         iou_threshold: float,
         offset: int = 0,
         score_threshold: float = 0,
         max_num: int = -1) -> torch.Tensor:
    """Reimplementa `mmcv.ops.nms.NMSop.forward` sobre o torchvision.

    Args:
        offset: 0 quando a largura da caixa é x2 - x1, e 1 quando é x2 - x1 + 1.
            O torchvision assume sempre a primeira convenção, então o segundo
            caso é acomodado dilatando as caixas antes da comparação.
        max_num: número máximo de caixas mantidas; negativo significa sem limite.

    Returns:
        Índices das caixas mantidas, ordenados por score decrescente.
    """
    valid_mask = scores > score_threshold
    if score_threshold > 0:
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        boxes, scores = boxes[valid_mask], scores[valid_mask]
    else:
        valid_indices = None

    if boxes.numel() == 0:
        return boxes.new_zeros((0, ), dtype=torch.long)

    if offset:
        boxes = boxes.clone()
        boxes[:, 2:] += offset

    keep = torchvision_nms(boxes.float(), scores.float(), iou_threshold)
    if max_num > 0:
        keep = keep[:max_num]

    return valid_indices[keep] if valid_indices is not None else keep


def install() -> None:
    """Aponta o NMS do MMCV para a implementação do torchvision."""
    global _installed
    if _installed:
        return

    # `import mmcv.ops.nms as x` devolveria a *função* nms, e não o módulo:
    # mmcv/ops/__init__.py reexporta o nome `nms`, sombreando o submódulo
    # homônimo. Atribuir a essa função seria um no-op silencioso.
    mmcv_nms = importlib.import_module('mmcv.ops.nms')

    class _TorchvisionNMSop(torch.autograd.Function):
        @staticmethod
        def forward(ctx, bboxes, scores, iou_threshold, offset,
                    score_threshold, max_num):
            return _nms(bboxes, scores, iou_threshold, offset,
                        score_threshold, max_num)

    mmcv_nms.NMSop = _TorchvisionNMSop
    _installed = True


install()
