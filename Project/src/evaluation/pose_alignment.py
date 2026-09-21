"""Alinhamento por similaridade entre duas poses, sem deformá-las.

Camada Model. Vive aqui, e não dentro de um script, porque dois consumidores
precisam exatamente do mesmo alinhamento: a validação no domínio veicular
(`scripts/validate_lifting_driveact.py`) e o protocolo de corte
(`src/evaluation/truncation_protocol.py`). Duas cópias divergiriam em silêncio,
e o PA-MPJPE que elas alimentam só é comparável se o alinhamento for o mesmo.
"""

from __future__ import annotations

import numpy as np


def procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Alinha `predicted` a `target` por similaridade, sem deformar a pose.

    Rotação, escala e translação são livres; a forma não. É o alinhamento do
    PA-MPJPE, e remove exatamente as três grandezas que uma câmera monocular
    deixa indeterminadas --- razão pela qual ele é **cego** a um erro de escala,
    como a Subseção da calibração do Projeto Físico registra: um erro de 3,8
    vezes no tamanho da pose passou por ele sem alterar o número.

    Args:
        predicted: [K, 3] pose predita.
        target: [K, 3] referência, na mesma ordem de keypoints.
    """
    predicted_center = predicted - predicted.mean(axis=0)
    target_center = target - target.mean(axis=0)

    covariance = predicted_center.T @ target_center
    left, singular, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0:          # evita reflexão
        right[-1] *= -1
        rotation = right.T @ left.T
        singular[-1] *= -1

    scale = singular.sum() / (predicted_center ** 2).sum()
    return scale * (predicted_center @ rotation.T) + target.mean(axis=0)
