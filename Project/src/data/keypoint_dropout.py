"""Remove keypoints da entrada durante o treino do lifting.

Camada Model. Existe por causa de uma medição, e não por analogia com dropout de
redes: ela diagnostica de onde vem o salto de domínio do Módulo 3.

O lifting treinado no H3WB mede 42,20mm de erro corporal no próprio domínio e
82,88mm no habitáculo. A hipótese natural seria a mudança de postura, de pessoas
em pé num laboratório para um ocupante sentado. **A medição diz outra coisa.**
Zerando, no próprio H3WB, os keypoints que a vista de retrovisor não observa
--- joelhos, tornozelos e pés --- o erro sobe para 80,75mm, praticamente o mesmo
valor do domínio veicular:

    entrada completa, H3WB      42,20 mm
    pernas zeradas, H3WB        80,75 mm
    Drive&Act real              82,88 mm
    pernas com ruído, H3WB     148,60 mm

Ou seja, **quase todo o salto de domínio é explicado por entrada incompleta**, e
não por postura ou ponto de vista. A rede nunca viu, no treino, uma janela em que
faltam keypoints, e por isso trata o zero como uma posição legítima.

A última linha também informa: o erro real fica junto da condição "zerado" e
longe da condição "ruído", o que indica que o estimador 2D, diante de uma junta
invisível, devolve uma posição consistente e não aleatória.

A correção é ensinar a rede a operar com entrada incompleta, que é o que esta
transformação faz.
"""

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS

# Grupos anatômicos que somem juntos numa cena real: um membro sai de quadro
# inteiro, não meia canela. Sortear keypoints isolados treinaria a rede para uma
# corrupção que não acontece.
KEYPOINT_GROUPS = {
    'pernas_e_pes': list(range(13, 23)),
    'perna_esquerda': [13, 15, 17, 18, 19],
    'perna_direita': [14, 16, 20, 21, 22],
    'pes': list(range(17, 23)),
    'braco_esquerdo': [7, 9],
    'braco_direito': [8, 10],
    'mao_esquerda': list(range(91, 112)),
    'mao_direita': list(range(112, 133)),
    'face': list(range(23, 91)),
}

# A vista de retrovisor perde pernas e pés em **todo** quadro, e é o caso de
# aplicação; por isso ele é sorteado com peso maior que os demais.
GROUP_WEIGHTS = {
    'pernas_e_pes': 4.0, 'perna_esquerda': 1.0, 'perna_direita': 1.0,
    'pes': 2.0, 'braco_esquerdo': 1.0, 'braco_direito': 1.0,
    'mao_esquerda': 1.0, 'mao_direita': 1.0, 'face': 1.0,
}


@TRANSFORMS.register_module()
class KeypointDropout(BaseTransform):
    """Zera grupos anatômicos da sequência de entrada, de forma consistente.

    Args:
        prob: probabilidade de aplicar a transformação a uma janela.
        max_groups: quantos grupos podem sair ao mesmo tempo.
    """

    def __init__(self, prob: float = 0.5, max_groups: int = 2) -> None:
        super().__init__()
        self.prob = prob
        self.max_groups = max_groups
        self._names = list(KEYPOINT_GROUPS)
        weights = np.array([GROUP_WEIGHTS[n] for n in self._names], float)
        self._probabilities = weights / weights.sum()

    def transform(self, results: dict) -> dict:
        if np.random.rand() >= self.prob:
            return results

        labels = results.get('keypoint_labels')
        if labels is None:
            return results

        count = np.random.randint(1, self.max_groups + 1)
        chosen = np.random.choice(self._names, size=count, replace=False,
                                  p=self._probabilities)
        missing = sorted({index for name in chosen
                          for index in KEYPOINT_GROUPS[name]})

        # O grupo some da janela inteira, e não de quadros avulsos: um membro
        # fora de quadro continua fora enquanto a câmera não se mexe. Sortear
        # por quadro ensinaria a rede a interpolar uma ausência intermitente,
        # que não é o regime da aplicação.
        labels = labels.copy()
        labels[..., missing, :] = 0.0
        results['keypoint_labels'] = labels

        if 'keypoint_labels_visible' in results:
            visible = results['keypoint_labels_visible'].copy()
            visible[..., missing] = 0.0
            results['keypoint_labels_visible'] = visible

        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'max_groups={self.max_groups})')
