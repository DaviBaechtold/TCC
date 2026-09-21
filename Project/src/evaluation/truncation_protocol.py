"""O que o lifting faz quando o quadro corta o corpo, medido contra ground truth.

Camada Model. O H3WB é o único conjunto com referência tridimensional das juntas
que a câmera do projeto não enxerga: da posição de retrovisor os tornozelos não
aparecem em quadro algum, e numa webcam de mesa o quadril também fica fora. Como
a referência do Drive&Act cobre só os doze keypoints observáveis --- joelhos em
2,1% e 12,1% dos quadros, tornozelos em nenhum ---, é aqui que a previsão das
pernas pode ser cobrada.

Três condições, e o que muda entre elas é a montagem da câmera:

    completa    entrada íntegra, o regime em que o lifting foi treinado
    retrovisor  joelhos, tornozelos e pés ausentes, como no Drive&Act
    mesa        corte horizontal acima do quadril, como na webcam de mesa

A corrupção é a **mesma** que `src/data/estimator_noise.py` aplica no treino, e
isso é deliberado: medir com um mecanismo diferente do treinado já custou 25mm de
conclusão errada a este projeto (ver o docstring daquele módulo).

**Armadilha que acompanha obrigatoriamente todo número daqui.** A condição
`mesa` é o mecanismo que treina o v3. Para esse checkpoint ela mede aderência ao
próprio treino, não generalização. O que generaliza é medido onde a corrupção
vem do estimador real: no Drive&Act e na gravação da webcam.
"""

from __future__ import annotations

import numpy as np

from src.data.estimator_noise import SimulatedEstimatorNoise
from src.evaluation.pose_alignment import procrustes_align
from src.models.observability import MIRROR_VIEW_ABSENT

MILLIMETERS = 1000.0

CONDITIONS = ('completa', 'retrovisor', 'mesa')

# Altura do corte de mesa, em fração do vão entre a linha dos ombros e a dos
# joelhos. Medido sobre as janelas do próprio H3WB: o quadril cai em 0,55 desse
# vão em pé e em 0,78 sentado, de modo que 0,35 fica acima dele nas duas posturas
# --- que é o enquadramento da webcam sobre a mesa, onde o quadril está fora da
# imagem. Fixo, e não sorteado como no treino, para que a medição seja repetível.
MESA_CUT_LEVEL = 0.35

REGIONS = {
    'corpo': list(range(17)),
    'quadris': [11, 12],
    'pernas': [13, 14, 15, 16],
    'pes': list(range(17, 23)),
    'face': list(range(23, 91)),
    'maos': list(range(91, 133)),
}

# Ossos que revelam o colapso do tronco, que é o defeito visível no painel: o
# lifting sobe o quadril até a altura do ombro e as coxas nascem no pescoço.
BONES = {
    'ombro-quadril esquerdo': (5, 11),
    'ombro-quadril direito': (6, 12),
    'coxa esquerda': (11, 13),
    'coxa direita': (12, 14),
    'canela esquerda': (13, 15),
    'canela direita': (14, 16),
}


def corrupt(labels: np.ndarray, condition: str,
            noise: SimulatedEstimatorNoise,
            cut_level: float = MESA_CUT_LEVEL
            ) -> tuple[np.ndarray, np.ndarray]:
    """Corrompe uma janela e devolve o que a câmera daquela montagem não vê.

    Args:
        labels: [T, K, 3] no espaço normalizado do codec, com confiança.
        condition: uma de `CONDITIONS`.
        noise: a transformação do treino, que carrega o teto de confiança do
            checkpoint sob medição --- treino e inferência precisam concordar.

    Returns:
        A janela corrompida e a máscara [T, K] das juntas ocultas.
    """
    if condition not in CONDITIONS:
        raise ValueError(f'condição desconhecida: {condition}')

    labels = labels.copy()
    hidden = np.zeros(labels.shape[:-1], dtype=bool)

    if condition == 'completa':
        return labels, hidden

    if condition == 'retrovisor':
        affected = list(MIRROR_VIEW_ABSENT)
        retained = [k for k in range(labels.shape[-2]) if k not in affected]
        noise.extrapolate(labels, affected, retained)
        hidden[..., affected] = True
        return labels, hidden

    results, below = noise.apply_frame_cut(
        {'keypoint_labels': labels}, labels, level=cut_level)
    return results['keypoint_labels'], (below if below is not None else hidden)


def decompose_error(predicted: np.ndarray, target: np.ndarray,
                    hidden: np.ndarray) -> dict[str, float | None]:
    """Erro em milímetros, separado entre o que foi visto e o que foi previsto.

    A separação é o ponto: um modelo pode melhorar a média global piorando
    justamente as juntas que a câmera não enxerga, que são as que o sistema
    precisa inventar.

    Args:
        predicted, target: [N, K, 3] em metros, ancorados na raiz.
        hidden: [N, K] booleano das juntas ocultas em cada janela.
    """
    distance = np.linalg.norm(predicted - target, axis=-1) * MILLIMETERS
    visible = ~hidden

    report: dict[str, float | None] = {
        'mpjpe_mm': float(distance.mean()),
        'mpjpe_visivel_mm': (float(distance[visible].mean())
                             if visible.any() else None),
        'mpjpe_oculto_mm': (float(distance[hidden].mean())
                            if hidden.any() else None),
    }
    for name, indices in REGIONS.items():
        report[f'mpjpe_{name}_mm'] = float(distance[:, indices].mean())

    aligned = np.stack([procrustes_align(p, t)
                        for p, t in zip(predicted, target)])
    report['pa_mpjpe_mm'] = float(
        (np.linalg.norm(aligned - target, axis=-1) * MILLIMETERS).mean())
    return report


def bone_geometry(predicted: np.ndarray,
                  target: np.ndarray) -> dict[str, dict[str, float]]:
    """Comprimento mediano de cada osso, predito contra verdadeiro, em mm."""
    geometry = {}
    for name, (start, end) in BONES.items():
        def length(poses):
            return float(np.median(np.linalg.norm(
                poses[:, start] - poses[:, end], axis=-1)) * MILLIMETERS)

        predito, verdadeiro = length(predicted), length(target)
        geometry[name] = {
            'predito_mm': round(predito, 1),
            'verdadeiro_mm': round(verdadeiro, 1),
            'erro_mm': round(predito - verdadeiro, 1),
        }
    return geometry
