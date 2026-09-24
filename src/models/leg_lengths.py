"""Comprimento estável para os ossos da perna prevista --- testado e NÃO adotado.

Camada Model. **Não está em operação**: ao vivo a restrição piorou o tremor da
perna (10,62 para 18,44mm na gravação com tabuleiro, 8,01 para 11,38mm na de
24/09), e o painel não a usa. O módulo fica para que as medições se reproduzam
(`scripts/measure_leg_lengths.py` e `measure_live_quality.py --comprimento-perna`).
 Na webcam de mesa só 3,5% dos pontos de perna estão na imagem, e o
lifting os prevê a partir do resto do corpo. A previsão muda de ideia de um quadro
para outro: na gravação com tabuleiro a coxa varia 90mm de desvio ao longo do
vídeo e a canela 57mm, contra 18mm da largura do quadril, e o olho lê isso como a
perna "respirando". O filtro temporal suaviza a posição de cada junta, mas não
impõe que a distância entre duas delas fique fixa.

A restrição mantém a direção de cada segmento que o modelo prevê e fixa o
comprimento na **mediana do que o próprio modelo previu para aquela pessoa** nos
últimos segundos. O pé acompanha o tornozelo sem se deformar. Só a perna não
observada é restringida.

Medido no sujeito retido S7 do H3WB, com o protocolo de corte de quadro
(`scripts/measure_leg_lengths.py`), MPJPE de pernas e pés em mm:

    condição     bruto          autoconsistente   âncora no braço
    mesa         117,5 / 140,2  117,3 / 139,6     126,6 / 152,5
    retrovisor   181,4 / 221,8  180,5 / 224,4     188,3 / 233,2
    completa      66,4 /  73,5   76,1 /  84,5      88,2 / 101,7

Com a perna prevista a autoconsistência não muda a precisão, e com a perna
visível ela piora --- por isso só se aplica à perna não observada. A âncora no
braço, que fixaria também a proporção, piora em toda condição, mesmo com o braço
verdadeiro: até o ground truth do H3WB varia cerca de 10% de proporção entre
quadros, e o erro de comprimento se propaga pela cadeia da perna. Ela fica aqui
só porque a medição a reproduz.

**Por que falhou ao vivo, medido quadro a quadro.** A oscilação do comprimento é
lenta: o comprimento restrito muda 0,2 a 0,7mm de um quadro para o seguinte, mas
o próprio modelo desloca a perna numa escala de segundos, e a mediana de três
segundos acompanha a deriva (desvio de 75 a 85mm ao longo do vídeo). E o tremor
quadro a quadro é de direção, não de comprimento: fixar o comprimento não reduz a
oscilação angular, e quando a coxa crua sai mais curta que a mediana o joelho é
empurrado mais longe na mesma direção ruidosa (10,7 para 14,2mm por quadro).
"""

from __future__ import annotations

from collections import deque

import numpy as np

# (quadril, joelho, tornozelo, pé) de cada lado, na ordem do COCO-WholeBody.
LEG_CHAINS = ((11, 13, 15, (17, 18, 19)),
              (12, 14, 16, (20, 21, 22)))

# (ombro, cotovelo) de cada lado.
UPPER_ARMS = ((5, 7), (6, 8))

# Razão entre perna e braço na convenção de pontos do H3WB, medida no ground truth
# 3D dos sujeitos de treino S1, S5 e S6 (15.000 quadros), mediana por quadro. As
# de Drillis e Contini (1,317 e 1,323) medem segmentos anatômicos, e o ombro de um
# esqueleto de keypoints não é a articulação: com elas a coxa saía 55mm curta.
# Usadas só pela variante descartada, a âncora no braço.
THIGH_PER_UPPER_ARM = 1.526
SHANK_PER_UPPER_ARM = 1.498

# Quadros sobre os quais o comprimento é a mediana: três segundos a 30 FPS, o
# bastante para um quadro ruim não mover a perna inteira.
LENGTH_WINDOW_FRAMES = 90

_EPSILON = 1e-9


def upper_arm_length(pose: np.ndarray) -> np.ndarray:
    """Média dos dois braços, em metros. `pose` é [..., 133, 3]."""
    return np.mean([np.linalg.norm(pose[..., s, :] - pose[..., e, :], axis=-1)
                    for s, e in UPPER_ARMS], axis=0)


def lengths_from_upper_arm(upper_arm_m: np.ndarray | float
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Coxa e canela que o braço prevê, pelas razões do H3WB (descartada)."""
    arm = np.asarray(upper_arm_m, dtype=np.float64)
    return arm * THIGH_PER_UPPER_ARM, arm * SHANK_PER_UPPER_ARM


def leg_lengths(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coxa e canela da própria pose, média dos dois lados, em metros."""
    def mean_of(pairs):
        return np.mean([np.linalg.norm(pose[..., a, :] - pose[..., b, :], axis=-1)
                        for a, b in pairs], axis=0)
    return (mean_of([(11, 13), (12, 14)]), mean_of([(13, 15), (14, 16)]))


def constrain_legs(pose: np.ndarray, thigh_m: np.ndarray | float,
                   shank_m: np.ndarray | float,
                   sides: tuple[bool, bool] = (True, True)) -> np.ndarray:
    """A mesma pose, com coxa e canela nos comprimentos dados.

    O quadril fica onde está. O joelho vai para a direção prevista, à distância
    da coxa; o tornozelo, para a direção prevista a partir do joelho novo, à
    distância da canela; o pé se desloca junto do tornozelo, rígido.

    Args:
        pose: [..., 133, 3].
        thigh_m, shank_m: escalares ou com a forma de `pose[..., 0, 0]`.
        sides: quais pernas restringir, esquerda e direita.
    """
    constrained = np.array(pose, dtype=np.float64, copy=True)
    thigh = np.asarray(thigh_m, dtype=np.float64)[..., None]
    shank = np.asarray(shank_m, dtype=np.float64)[..., None]

    def unit(vector):
        return vector / (np.linalg.norm(vector, axis=-1, keepdims=True) + _EPSILON)

    for (hip, knee, ankle, foot), apply in zip(LEG_CHAINS, sides):
        if not apply:
            continue
        new_knee = pose[..., hip, :] + unit(pose[..., knee, :] - pose[..., hip, :]) * thigh
        new_ankle = new_knee + unit(pose[..., ankle, :] - pose[..., knee, :]) * shank
        shift = new_ankle - pose[..., ankle, :]
        constrained[..., knee, :] = new_knee
        constrained[..., ankle, :] = new_ankle
        constrained[..., list(foot), :] = pose[..., list(foot), :] + shift[..., None, :]
    return constrained.astype(pose.dtype)


class LegLengthStabilizer:
    """Fixa ao vivo o comprimento da perna prevista na mediana recente do modelo."""

    def __init__(self, window_frames: int = LENGTH_WINDOW_FRAMES):
        self._thighs: deque[float] = deque(maxlen=window_frames)
        self._shanks: deque[float] = deque(maxlen=window_frames)

    def reset(self) -> None:
        self._thighs.clear()
        self._shanks.clear()

    def __call__(self, pose: np.ndarray, observed: np.ndarray) -> np.ndarray:
        """Restringe as pernas não observadas de uma pose [133, 3].

        Uma perna conta como observada quando joelho e tornozelo foram vistos;
        aí o modelo acerta mais sozinho, e a restrição só piora (medido).
        """
        thigh, shank = leg_lengths(pose)
        self._thighs.append(float(thigh))
        self._shanks.append(float(shank))
        sides = tuple(not (observed[knee] and observed[ankle])
                      for _, knee, ankle, _ in LEG_CHAINS)
        if not any(sides):
            return pose
        return constrain_legs(pose, float(np.median(self._thighs)),
                              float(np.median(self._shanks)), sides)
