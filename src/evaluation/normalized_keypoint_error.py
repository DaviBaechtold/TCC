"""Erro de keypoint normalizado por tronco, para quando a OKS satura.

Camada Model. Existe por um defeito medido, não por preferência metodológica.

No Drive&Act o AP do protocolo COCO satura: o modelo sem nenhum treino no
domínio mede 0,9351 e o modelo adaptado mede 0,9330 — a métrica não distingue os
dois, e a meta do projeto já estaria cumprida antes de qualquer trabalho. Três
causas se somam, e nenhuma é do modelo:

1. A caixa de validação é derivada dos próprios keypoints, porque o dataset não
   anota caixa de pessoa. Ela entrega parte da resposta.
2. O ocupante ocupa cerca de 28% do frame, e a OKS normaliza pela área do
   objeto: objeto grande significa tolerância grande. Um erro de 30 px num ombro
   ainda pontua 0,82 de OKS.
3. Só cerca de doze keypoints por instância estão anotados, todos de tronco e
   cabeça, que são os mais fáceis.

A métrica aqui troca o normalizador de área por **comprimento de tronco**, a
distância entre o ponto médio dos ombros e o ponto médio dos quadris. Ele é
invariante à distância da câmera, como a área, mas cresce linearmente com a
escala em vez de quadraticamente, e não é inflado por um enquadramento generoso.
É o normalizador usado pelo PCK clássico.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from mmengine.evaluator import BaseMetric

from mmpose.registry import METRICS

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

# Fração do comprimento do tronco abaixo da qual o keypoint conta como correto.
# 0,2 é a convenção do PCK@0.2; 0,1 é a variante estrita, que separa métodos que
# o limiar frouxo empata.
PCK_THRESHOLDS = (0.1, 0.2)

# Um tronco degenerado produz normalizador próximo de zero e erro infinito.
MIN_TORSO_PIXELS = 1.0


@METRICS.register_module()
class TorsoNormalizedError(BaseMetric):
    """Erro médio em pixels e PCK normalizados pelo comprimento do tronco.

    Reporta também o erro em pixels sem normalizar, que é o número que um
    operador consegue interpretar olhando para a imagem.
    """

    default_prefix = 'torso'

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            predicted = np.asarray(data_sample['pred_instances']['keypoints'])
            ground_truth = data_sample['gt_instances']
            target = np.asarray(ground_truth['keypoints'])
            visible = np.asarray(ground_truth['keypoints_visible']) > 0

            torso = torso_length(target[0], visible[0])
            if torso is None:
                continue  # sem tronco não há normalizador; a amostra é omitida

            self.results.append({
                'distances': np.linalg.norm(predicted[0] - target[0], axis=-1),
                'visible': visible[0],
                'torso': torso,
            })

    def compute_metrics(self, results: list) -> dict[str, float]:
        distances = np.concatenate([r['distances'][r['visible']]
                                    for r in results])
        normalized = np.concatenate([
            r['distances'][r['visible']] / r['torso'] for r in results])

        # A média e a mediana discordam sobre qual época é melhor, e a razão
        # entre elas diz por quê: medida na adaptação ao Drive&Act, 10,0 contra
        # 7,5 pixels, ou 1,34. Numa distribuição simétrica seria 1,0. A média é
        # puxada por uma cauda de quadros ruins, de modo que escolher o melhor
        # checkpoint por ela é escolher pelos piores casos. O percentil 90
        # dimensiona essa cauda em vez de deixá-la implícita.
        metrics = {
            'px_mean': float(distances.mean()),
            'px_median': float(np.median(distances)),
            'px_p90': float(np.percentile(distances, 90)),
            'normalized_mean': float(normalized.mean()),
            'normalized_median': float(np.median(normalized)),
            'samples': float(len(results)),
        }
        for threshold in PCK_THRESHOLDS:
            metrics[f'PCK@{threshold}'] = float((normalized < threshold).mean())
        return metrics


def torso_length(keypoints: np.ndarray, visible: np.ndarray) -> float | None:
    """Distância entre o centro dos ombros e o centro dos quadris, em pixels."""
    shoulders = (LEFT_SHOULDER, RIGHT_SHOULDER)
    hips = (LEFT_HIP, RIGHT_HIP)
    if not (visible[list(shoulders)].any() and visible[list(hips)].any()):
        return None

    shoulder_center = _visible_center(keypoints, visible, shoulders)
    hip_center = _visible_center(keypoints, visible, hips)
    length = float(np.linalg.norm(shoulder_center - hip_center))
    return length if length >= MIN_TORSO_PIXELS else None


def _visible_center(keypoints: np.ndarray, visible: np.ndarray,
                    indices: tuple[int, ...]) -> np.ndarray:
    """Centro dos keypoints visíveis dentre `indices`.

    Aceitar um único lado visível importa: da posição de retrovisor um dos
    ombros é frequentemente ocluído pelo encosto, e exigir ambos descartaria
    boa parte das amostras.
    """
    chosen = [index for index in indices if visible[index]]
    return keypoints[chosen].mean(axis=0)


def instance_error(predicted: np.ndarray,
                   reference: np.ndarray) -> tuple[float, int] | None:
    """Erro normalizado por tronco de uma instância, e quantas juntas entraram.

    Args:
        predicted: [K, 2] keypoints preditos, em pixels do quadro original.
        reference: [K, 3] anotação no formato COCO, com o terceiro canal
            indicando visibilidade.

    Returns:
        `(erro, juntas_visiveis)`, ou `None` quando o tronco é indeterminado ---
        caso em que não há normalizador e medir em pixels crus compararia poses a
        distâncias diferentes da câmera.

    Vive aqui, e não nos controllers, porque três medições diferentes precisam
    exatamente desta conta: a comparação de detectores, a bateria de validação e
    a estratificação. Copiada, ela divergiria em silêncio --- e o erro de cada
    cópia seria indistinguível de um efeito do que se está medindo.
    """
    visible = reference[:, 2] > 0
    if not visible.any():
        return None

    scale = torso_length(reference[:, :2], visible)
    if scale is None:
        return None

    distances = np.linalg.norm(
        predicted[:len(reference)][visible] - reference[visible, :2], axis=-1)
    return float(distances.mean() / scale), int(visible.sum())


# Acima disto a pose não descreve a pessoa anotada: 0,2 comprimento de tronco é
# o limiar do PCK@0.2, e o erro típico do estimador adaptado no Drive&Act é de
# 0,022. Medido no conjunto de treino veicular extraído pela primeira caixa,
# 1,03% dos quadros passavam deste limiar --- e 0,91% passavam de 1,0, isto é,
# a caixa estava em outro lugar do quadro.
MAX_ASSOCIATION_ERROR = 0.2


def occupant_index(keypoints: np.ndarray, reference: np.ndarray) -> int | None:
    """Qual das pessoas detectadas é a pessoa da anotação.

    Args:
        keypoints: [N, K, 2] poses preditas, uma por caixa detectada.
        reference: [K, 3] anotação no formato COCO.

    Returns:
        O índice da pose mais próxima da anotação, ou `None` quando nenhuma fica
        abaixo de `MAX_ASSOCIATION_ERROR`.

    Existe porque "a primeira caixa é a do ocupante" é falso no infravermelho:
    o RTMDet-nano devolve mais de uma caixa em 58% dos quadros do Drive&Act,
    que tem uma pessoa só, e a primeira nem sempre é a dela. Parear o 2D de uma
    caixa espúria com o 3D do ocupante ensina a rede uma correspondência que
    não existe.
    """
    erros = [instance_error(pose, reference) for pose in keypoints]
    candidatos = [(medido[0], indice) for indice, medido in enumerate(erros)
                  if medido is not None]
    if not candidatos:
        return None
    erro, indice = min(candidatos)
    return indice if erro <= MAX_ASSOCIATION_ERROR else None
