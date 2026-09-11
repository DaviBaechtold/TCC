"""Supervisiona os keypoints que a câmera não enxerga, em vez de ignorá-los.

Camada Model. Corrige um efeito colateral medido da adaptação ao domínio
veicular.

**O problema.** No Drive&Act as juntas fora de quadro têm visibilidade zero, o
que lhes dá peso zero na função de perda. Nada pune uma predição confiante e
errada ali, e dez épocas de treino sem sinal deixam a resposta subir livremente:

    resposta nas juntas invisíveis   3,59 -> 5,13
    fração acima do limiar de 3,0      51% -> 92%

enquanto a resposta nas juntas observáveis não se move (8,26 -> 8,30).
**Mascarar um keypoint da perda não faz o modelo ficar calado sobre ele, faz o
modelo ficar impune sobre ele.** O efeito se propaga: o módulo de lifting passa a
receber pernas confiantemente erradas em vez de pernas duvidosas, e seu erro no
habitáculo sobe de 82,88mm para 98,33mm.

**A correção.** O alvo do SimCC para um keypoint ausente já é um vetor de zeros,
e a perda aplica softmax sobre o alvo (`label_softmax=True`), de modo que zeros
viram **distribuição uniforme**. Supervisionar com peso não nulo ensina a rede a
devolver resposta achatada onde não há observação --- que é exatamente o sinal
"não sei" que faltava. A máquina já existia; faltava dar peso a ela.

**A distinção que não pode ser perdida.** Só as juntas ausentes da *imagem*
recebem esse tratamento. Face e mãos também têm peso zero no Drive&Act, mas por
falta de *anotação*: elas estão em quadro e o modelo as estima bem, tendo
aprendido isso na etapa anterior. Ensiná-las a duvidar destruiria o que a
Etapa 2 construiu.
"""

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS

from src.models.observability import MIRROR_VIEW_ABSENT

# Peso da supervisão negativa. Menor que o das juntas observadas, porque o
# objetivo é ensinar incerteza e não competir com o sinal que de fato localiza.
# O config já usa 0,5 para atenuar a face; 0,3 é o primeiro valor tentado aqui e
# precisa ser confirmado por medição da separação de resposta.
DEFAULT_WEIGHT = 0.3


@TRANSFORMS.register_module()
class SuperviseAbsentKeypoints(BaseTransform):
    """Dá peso não nulo aos keypoints fora de quadro, cujo alvo é uniforme.

    Args:
        indices: quais keypoints estão ausentes da imagem. O padrão são os que
            a vista de retrovisor não observa, medidos em `observability`.
        weight: peso da supervisão negativa.
    """

    def __init__(self, indices: tuple[int, ...] = MIRROR_VIEW_ABSENT,
                 weight: float = DEFAULT_WEIGHT) -> None:
        super().__init__()
        self.indices = list(indices)
        self.weight = weight

    def transform(self, results: dict) -> dict:
        weights = results.get('keypoint_weights')
        if weights is None:
            return results

        weights = np.asarray(weights).copy()

        # Só onde o peso é zero: uma junta que por acaso foi anotada num quadro
        # continua com a supervisão positiva dela, que é mais informativa.
        absent = np.zeros(weights.shape[-1], dtype=bool)
        absent[self.indices] = True
        weights[..., absent] = np.where(weights[..., absent] == 0,
                                        self.weight, weights[..., absent])

        results['keypoint_weights'] = weights
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(indices={len(self.indices)} '
                f'keypoints, weight={self.weight})')
