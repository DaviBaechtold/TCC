"""Erro por região anatômica, cada uma normalizada pela sua própria escala.

Camada Model. Existe porque o whole-body AP é um número só para 133 keypoints e
**não diz o que quebrou**: um modelo pode perder a face inteira e o agregado cair
poucos pontos, diluído pelo corpo que continua certo. Foi essa cegueira que
deixou passar, por uma sessão inteira, a suspeita de que a adaptação ao domínio
veicular tivesse degradado face e mãos --- justamente as regiões sem anotação
naquele conjunto, e por isso sem peso na perda.

Cada região é normalizada pela escala que lhe é própria, que é o que torna os
números comparáveis entre si e entre imagens:

    corpo    pelo comprimento do tronco (ombro ao quadril)
    face     pela distância interocular, convenção da literatura de landmarks
    mãos     pela diagonal do envelope da própria mão

A normalização da face merece nota: a distância interocular é pequena (dezenas
de pixels), de modo que um erro de poucos pixels vira uma fração grande. Isso é
desejado --- é a escala em que um desvio da face importa --- mas impede comparar
o número da face com o do corpo. Compare o mesmo modelo entre si, ou dois
modelos na mesma região.
"""

from __future__ import annotations

import numpy as np

# Blocos do layout COCO-WholeBody.
BODY = list(range(17))
FACE = list(range(23, 91))
LEFT_HAND = list(range(91, 112))
RIGHT_HAND = list(range(112, 133))

# Índices dos cantos externos dos olhos no bloco de 68 landmarks, já deslocados
# para a numeração de 133: é a distância interocular da convenção de landmarks.
LEFT_EYE_CORNER = 23 + 45
RIGHT_EYE_CORNER = 23 + 36

# Abaixo disto a escala de normalização é degenerada e o quociente explode em
# vez de medir --- pessoa minúscula, mão de perfil, olhos colados.
MIN_SCALE_PIXELS = 2.0


def _distance(keypoints: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(keypoints[first] - keypoints[second]))


def _diagonal(keypoints: np.ndarray) -> float:
    lower, upper = keypoints.min(axis=0), keypoints.max(axis=0)
    return float(np.linalg.norm(upper - lower))


def region_scales(truth: np.ndarray) -> dict[str, float]:
    """Escala de normalização de cada região, a partir do ground truth.

    O tronco toma o maior dos dois lados porque um deles pode estar ocluído e
    anotado em (0, 0); o maior é o que tem chance de ser o real.
    """
    torso = max(_distance(truth, 5, 11), _distance(truth, 6, 12))
    return {
        'corpo': torso,
        'face': _distance(truth, LEFT_EYE_CORNER, RIGHT_EYE_CORNER),
        'mao esquerda': _diagonal(truth[LEFT_HAND]),
        'mao direita': _diagonal(truth[RIGHT_HAND]),
    }


REGION_INDICES = {
    'corpo': BODY,
    'face': FACE,
    'mao esquerda': LEFT_HAND,
    'mao direita': RIGHT_HAND,
}


def normalized_errors(predicted: np.ndarray, truth: np.ndarray,
                      visible: np.ndarray) -> dict[str, float]:
    """Erro médio por região, em frações da escala da região.

    Args:
        predicted, truth: [133, 2] em pixels da mesma imagem.
        visible: [133] booleano da anotação. **Obrigatório**: no COCO um
            keypoint não anotado vem como (0, 0), e incluí-lo mede a distância
            até a origem da imagem em vez do erro --- o que inflou o erro
            corporal para 0,45 do tronco nos dois modelos na primeira execução
            desta medição, escondendo a diferença entre eles.

    Returns:
        Uma entrada por região, omitindo as de escala degenerada ou sem
        keypoint anotado algum.
    """
    scales = region_scales(truth)
    errors = {}
    for region, indices in REGION_INDICES.items():
        scale = scales[region]
        anotados = [k for k in indices if visible[k]]
        if scale < MIN_SCALE_PIXELS or not anotados:
            continue
        distance = np.linalg.norm(
            predicted[anotados] - truth[anotados], axis=-1)
        errors[region] = float(distance.mean() / scale)
    return errors


def summarize(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mediana, média e contagem por região sobre um conjunto de amostras.

    A mediana acompanha a média porque uma única detecção em pessoa errada
    desloca a média de uma região inteira, e a distinção entre as duas é o que
    denuncia esse caso.
    """
    resumo = {}
    for region in REGION_INDICES:
        valores = [s[region] for s in samples if region in s]
        if not valores:
            continue
        resumo[region] = {
            'mediana': round(float(np.median(valores)), 4),
            'media': round(float(np.mean(valores)), 4),
            'amostras': len(valores),
        }
    return resumo
