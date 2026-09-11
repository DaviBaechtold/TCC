"""Quais keypoints a câmera de retrovisor de fato enxerga.

Camada Model. É uma propriedade da montagem da câmera, não do modelo: com a
câmera fixa no retrovisor interno, há juntas que não aparecem em quadro algum, e
isso é verificável antes de qualquer inferência.

Medido sobre as 20.288 instâncias do conjunto de validação do Drive&Act, fração
dos quadros em que cada junta corporal está anotada:

    nariz 97,2%   olhos 85,9 e 88,9%   orelha direita 88,1%
    ombros 98,6 e 99,1%   cotovelos 95,3 e 98,3%   pulsos 93,2 e 95,9%
    quadris 92,8 e 93,0%
    orelha esquerda 24,6%
    joelhos 0,2% e 3,7%   tornozelos 0,0% e 0,0%

Doze juntas passam de 80%; o restante do corpo não aparece. Os seis keypoints de
pé dependem dos tornozelos e seguem a mesma sorte.

**Por que isto existe como conhecimento explícito, e não como limiar de
confiança.** O estimador 2D adaptado ao Drive&Act ficou *mais* confiante sobre o
que não vê: a resposta nas juntas invisíveis subiu de 3,59 para 5,13 depois da
adaptação, e a fração delas que passa o limiar de 3,0 foi de 51% para 92%. A
causa é que essas juntas têm peso zero na função de perda, de modo que nada pune
uma predição confiante e errada ali. **Mascarar um keypoint da perda não faz o
modelo ficar calado sobre ele, faz o modelo ficar impune sobre ele.** Um limiar
não separa mais as duas populações; a geometria da montagem, sim.
"""

from __future__ import annotations

import numpy as np

NUM_WHOLEBODY_KEYPOINTS = 133

# Nariz, olhos, orelha direita, ombros, cotovelos, pulsos e quadris.
MIRROR_VIEW_OBSERVABLE = (0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)

# Joelhos, tornozelos e os seis keypoints de pé. Nunca aparecem.
MIRROR_VIEW_ABSENT = tuple(range(13, 23))

# Face e mãos aparecem na imagem e são estimadas; o que falta é anotação para
# medi-las, não observação. Não entram em `MIRROR_VIEW_ABSENT` — tratá-las como
# ausentes ensinaria o modelo a duvidar do que ele de fato vê.
FACE_AND_HANDS = tuple(range(23, NUM_WHOLEBODY_KEYPOINTS))


def mirror_view_mask(num_keypoints: int = NUM_WHOLEBODY_KEYPOINTS) -> np.ndarray:
    """Máscara booleana do que a vista de retrovisor observa.

    Face e mãos entram como observáveis: elas estão na imagem, ainda que o
    Drive&Act não as anote.
    """
    mask = np.ones(num_keypoints, dtype=bool)
    mask[list(MIRROR_VIEW_ABSENT)] = False
    return mask


def reliable_keypoints(scores: np.ndarray, min_score: float) -> np.ndarray:
    """Combina o limiar de resposta com o que a montagem permite observar.

    As duas condições são necessárias e nenhuma basta: o limiar sozinho deixa
    passar 92% das juntas invisíveis, e a máscara sozinha não filtra a junta que
    está em quadro mas foi mal localizada.
    """
    return (scores >= min_score) & mirror_view_mask(scores.shape[-1])
