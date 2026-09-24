"""A janela temporal recupera um ponto que some por alguns quadros?

Camada Model. Responde à segunda metade da QP2. As outras medições de oclusão
deste projeto tratam de pontos **permanentemente** ausentes --- a perna que a
câmera do retrovisor nunca vê. Aqui o ponto estava visível e deixou de estar: a
mão que entra sob o volante. É o caso em que o contexto temporal poderia ajudar,
porque a janela ainda guarda onde a mão estava.

O protocolo oculta um grupo de pontos nos últimos `k` quadros da janela, que é
exatamente o que o sistema vê no instante de leitura causal quando a mão sumiu
há `k` quadros. O ponto oculto fica congelado na última posição vista --- o
estimador perdeu a mão --- com a confiança no teto que o treino usa para
ausência. O controle é a mesma entrada sem contexto temporal: a janela preenchida
com cópias do quadro corrente, como no teste 6 da bateria de validação. Se a
janela recupera o ponto, o erro com histórico fica abaixo do erro sem ele, e
cresce com `k`.

A ressalva: não se mediu o que o estimador 2D real faz com uma mão sob o volante.
Congelar é uma hipótese de comportamento, a mais favorável ao contexto temporal
--- a posição congelada é a última verdadeira.
"""

from __future__ import annotations

import numpy as np

from src.data.estimator_noise import UNOBSERVED_CONFIDENCE_CAP

# Punho e mão direitos: a mão que sai do campo sob o volante na posição de
# motorista do Drive&Act, que senta à esquerda da câmera central.
RIGHT_WRIST = 10
RIGHT_HAND = tuple(range(112, 133))
OCCLUDED_GROUP = (RIGHT_WRIST,) + RIGHT_HAND

# De nenhum quadro oculto até a janela inteira. Com 16, o ponto sumiu antes de a
# janela começar, e o contexto não tem mais o que oferecer.
DURATIONS = (0, 1, 2, 4, 8, 12, 16)

CONFIDENCE_CHANNEL = 2


def occlude_last_frames(window: np.ndarray, joints: tuple[int, ...],
                        frames: int,
                        confidence: float = UNOBSERVED_CONFIDENCE_CAP
                        ) -> np.ndarray:
    """Oculta `joints` nos últimos `frames` quadros de uma janela [T, K, 3].

    O ponto oculto fica na última posição vista antes da oclusão, e a
    confiança cai ao teto. Com `frames` igual ao tamanho da janela, a última
    posição vista é a do primeiro quadro --- o ponto sumiu no instante em que a
    janela começou.
    """
    corrupted = window.copy()
    if frames <= 0:
        return corrupted
    total = window.shape[0]
    frames = min(frames, total)
    onset = total - frames
    last_seen = max(onset - 1, 0)
    indices = list(joints)
    corrupted[onset:, indices, :CONFIDENCE_CHANNEL] = \
        window[last_seen, indices, :CONFIDENCE_CHANNEL]
    corrupted[onset:, indices, CONFIDENCE_CHANNEL] = np.minimum(
        window[onset:, indices, CONFIDENCE_CHANNEL], confidence)
    return corrupted


def without_context(window: np.ndarray) -> np.ndarray:
    """A mesma rede sem contexto temporal: dezesseis cópias do quadro atual.

    Mantém arquitetura e pesos idênticos, de modo que a diferença para a janela
    real mede só o contexto.
    """
    return np.repeat(window[-1:], window.shape[0], axis=0)


def joint_error_mm(predicted: np.ndarray, target: np.ndarray,
                   joints: tuple[int, ...]) -> float:
    """MPJPE dos `joints`, em mm, com as poses já ancoradas na raiz."""
    indices = list(joints)
    return float(np.linalg.norm(predicted[:, indices] - target[:, indices],
                                axis=-1).mean() * 1000.0)
