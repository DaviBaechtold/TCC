"""Preparação do H3WB para o formato de anotação que o MMPose consome.

O H3WB (Human3.6M 3D WholeBody, ICCV 2023) distribui as anotações reformatadas
como dois arquivos `.npy` independentes, `train_data` e `metadata`. O
`H36MWholeBodyDataset` do MMPose espera um único `.npz` contendo três arrays de
objeto: `train_data`, `metadata` e `bbox`.

As duas primeiras podem ser reempacotadas sem alteração. A terceira não é
distribuída e precisa ser derivada: o dataset indexa `bbox` por
`(sujeito, ação, câmera, frame_id)` e lê `x_min`, `y_min`, `x_max` e `y_max`.

Camada Model: não imprime nem lê argumentos de linha de comando. O controller
correspondente é `scripts/convert_h3wb.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Chaves de `train_data[sujeito][ação]` que descrevem a sequência em vez de uma
# câmera específica. Tudo o mais naquele nível é um identificador de câmera.
NON_CAMERA_KEYS = frozenset({'global_3d', 'frame_id'})

# Margem relativa aplicada ao envelope dos keypoints ao derivar a caixa. O H3WB
# não distribui bounding boxes; inferi-las do envelope sem margem produziria
# caixas que tangenciam a silhueta, destoando da convenção do Human3.6M, em que
# a caixa envolve o corpo com folga.
BBOX_PADDING_RATIO = 0.1


@dataclass(frozen=True)
class ConversionSummary:
    subjects: tuple[str, ...]
    num_sequences: int
    num_boxes: int
    num_keypoints: int


def _camera_ids(sequence: dict) -> Iterator[str]:
    for key in sequence:
        if key not in NON_CAMERA_KEYS:
            yield key


def bounding_box_from_keypoints(keypoints_2d: np.ndarray) -> dict[str, float]:
    """Caixa envolvente dos keypoints de um frame, com margem relativa.

    Args:
        keypoints_2d: [N, 2] em pixels.
    """
    top_left = keypoints_2d.min(axis=0)
    bottom_right = keypoints_2d.max(axis=0)
    padding = (bottom_right - top_left) * BBOX_PADDING_RATIO
    top_left = top_left - padding
    bottom_right = bottom_right + padding

    return {
        'x_min': float(top_left[0]),
        'y_min': float(top_left[1]),
        'x_max': float(bottom_right[0]),
        'y_max': float(bottom_right[1]),
    }


def build_bounding_boxes(train_data: dict) -> dict[tuple, dict[str, float]]:
    """Deriva o índice de caixas exigido pelo `H36MWholeBodyDataset`.

    A chave reproduz exatamente a que o dataset consulta, incluindo o
    `frame_id` como string — é assim que ele vem no arquivo original, e
    converter para inteiro faria toda consulta falhar com `KeyError`.
    """
    boxes: dict[tuple, dict[str, float]] = {}

    for subject, actions in train_data.items():
        for action, sequence in actions.items():
            frame_ids = sequence['frame_id']
            for camera in _camera_ids(sequence):
                keypoints_2d = sequence[camera]['pose_2d']
                for index, frame_id in enumerate(frame_ids):
                    boxes[(subject, action, camera, frame_id)] = (
                        bounding_box_from_keypoints(keypoints_2d[index]))

    return boxes


def summarize(train_data: dict, boxes: dict) -> ConversionSummary:
    sequences = sum(len(actions) for actions in train_data.values())
    first_subject = next(iter(train_data.values()))
    first_sequence = next(iter(first_subject.values()))
    first_camera = next(_camera_ids(first_sequence))
    num_keypoints = first_sequence[first_camera]['pose_2d'].shape[1]

    return ConversionSummary(
        subjects=tuple(sorted(train_data)),
        num_sequences=sequences,
        num_boxes=len(boxes),
        num_keypoints=num_keypoints)


def convert(train_data_path: Path,
            metadata_path: Path,
            output_path: Path) -> ConversionSummary:
    """Reempacota os dois `.npy` do H3WB no `.npz` esperado pelo MMPose."""
    train_data = np.load(train_data_path, allow_pickle=True).item()
    metadata = np.load(metadata_path, allow_pickle=True).item()
    boxes = build_bounding_boxes(train_data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path,
             train_data=np.array(train_data, dtype=object),
             metadata=np.array(metadata, dtype=object),
             bbox=np.array(boxes, dtype=object))

    return summarize(train_data, boxes)
