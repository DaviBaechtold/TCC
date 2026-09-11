"""Simula um estimador 2D imperfeito na entrada do treino do lifting.

Camada Model. Substitui `KeypointDropout`, que apagava keypoints e **piorou** o
domínio veicular: de 82,88mm para 108,37mm sob o mesmo estimador 2D. O motivo do
fracasso é instrutivo e está registrado aqui para não se repetir.

**O erro de raciocínio.** Zerar as pernas no H3WB dava 80,75mm, e o domínio real
dava 82,88mm; disso concluí que a corrupção real *era* zeros. Não é. Coincidiram
as magnitudes, não os mecanismos. Medindo onde o estimador 2D de fato coloca uma
junta que não está na imagem, sobre 200 quadros do Drive&Act, em fração da caixa
da pessoa:

    joelho esquerdo   x 0,91 ± 0,09   y 0,92 ± 0,12
    tornozelo direito x 0,34 ± 0,15   y 1,07 ± 0,15
    dedão direito     x 0,40 ± 0,12   y 1,07 ± 0,13

Ele não espalha nada: coloca a junta **junto da borda inferior do recorte**, onde
a perna estaria se continuasse fora de quadro. É uma extrapolação plausível e
consistente, não ruído. Treinar contra zeros ensinou a rede a reconhecer um sinal
que nunca ocorre.

**O que a rede precisa aprender.** O terceiro canal da entrada carrega a
confiança, e no H3WB ele é **constante em exatamente 1,0** --- desvio padrão
zero, um único valor distinto em todo o conjunto. A rede aprendeu a ignorá-lo, e
a medição confirma: alimentar o canal com 1,0, com a resposta bruta ou com ela
normalizada muda o erro em menos de 2,5%. No habitáculo esse canal é justamente
o que separa junta observada de junta extrapolada --- a resposta média é 8,30
contra 5,13 --- e a rede é surda a ele.

Esta transformação torna o canal informativo: quem recebe confiança baixa recebe
também posição deslocada, de modo que a correlação entre os dois exista no treino
e possa ser aprendida.
"""

from __future__ import annotations

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS

# Grupos anatômicos que somem juntos numa cena real: um membro sai de quadro
# inteiro, não meia canela.
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

# A vista de retrovisor perde pernas e pés em todo quadro, e é o caso de
# aplicação; por isso ele é sorteado com peso maior.
GROUP_WEIGHTS = {
    'pernas_e_pes': 4.0, 'perna_esquerda': 1.0, 'perna_direita': 1.0,
    'pes': 2.0, 'braco_esquerdo': 1.0, 'braco_direito': 1.0,
    'mao_esquerda': 1.0, 'mao_direita': 1.0, 'face': 1.0,
}

# Onde a junta extrapolada aparece, em fração do envelope do corpo visível.
# Medido no Drive&Act: y junto da borda inferior, x espalhado pela largura.
EXTRAPOLATED_Y = (0.88, 1.12)
EXTRAPOLATED_X = (0.25, 0.95)
EXTRAPOLATED_JITTER = 0.13

# Confiança em [0, 1], obtida dividindo a resposta medida pela média dos
# keypoints observáveis (8,30). Observáveis ficam perto de 1; extrapolados, de
# 0,62, com a dispersão medida entre os percentis 10 e 90.
CONFIDENCE_OBSERVED = (0.85, 1.00)
CONFIDENCE_EXTRAPOLATED = (0.37, 0.92)


@TRANSFORMS.register_module()
class SimulatedEstimatorNoise(BaseTransform):
    """Desloca grupos anatômicos e rebaixa a confiança deles, em conjunto.

    Args:
        prob: probabilidade de aplicar a um exemplo.
        max_groups: quantos grupos podem ser afetados ao mesmo tempo.
    """

    def __init__(self, prob: float = 0.6, max_groups: int = 2) -> None:
        super().__init__()
        self.prob = prob
        self.max_groups = max_groups
        self._names = list(KEYPOINT_GROUPS)
        weights = np.array([GROUP_WEIGHTS[n] for n in self._names], float)
        self._probabilities = weights / weights.sum()

    def transform(self, results: dict) -> dict:
        labels = results.get('keypoint_labels')
        if labels is None:
            return results

        labels = labels.copy()

        # A confiança dos keypoints preservados também varia, ainda que pouco.
        # Sem isso o canal continuaria quase constante e a rede não teria motivo
        # para consultá-lo.
        labels[..., 2] = np.random.uniform(*CONFIDENCE_OBSERVED,
                                           size=labels.shape[:-1])

        if np.random.rand() >= self.prob:
            results['keypoint_labels'] = labels
            return results

        count = np.random.randint(1, self.max_groups + 1)
        chosen = np.random.choice(self._names, size=count, replace=False,
                                  p=self._probabilities)
        affected = sorted({index for name in chosen
                           for index in KEYPOINT_GROUPS[name]})

        retained = [k for k in range(labels.shape[-2]) if k not in affected]
        if not retained:
            results['keypoint_labels'] = labels
            return results

        self._extrapolate(labels, affected, retained)
        results['keypoint_labels'] = labels

        if 'keypoint_labels_visible' in results:
            visible = results['keypoint_labels_visible'].copy()
            visible[..., affected] = 0.0
            results['keypoint_labels_visible'] = visible

        return results

    def _extrapolate(self, labels: np.ndarray, affected: list[int],
                     retained: list[int]) -> None:
        """Move os keypoints afetados para a borda do corpo visível.

        O grupo é deslocado de forma consistente na janela inteira, e não quadro
        a quadro: um membro fora de quadro continua fora enquanto a câmera não se
        mexe. Sortear por quadro ensinaria a rede a esperar uma ausência
        intermitente, que não é o regime da aplicação.
        """
        visible = labels[..., retained, :2]
        lower = visible.min(axis=(-3, -2))
        upper = visible.max(axis=(-3, -2))
        extent = np.maximum(upper - lower, 1e-6)

        count = len(affected)
        fraction = np.stack([
            np.random.uniform(*EXTRAPOLATED_X, size=count),
            np.random.uniform(*EXTRAPOLATED_Y, size=count),
        ], axis=-1)
        placed = lower + fraction * extent
        placed += np.random.normal(0.0, EXTRAPOLATED_JITTER * extent.mean(),
                                   placed.shape)

        labels[..., affected, :2] = placed
        labels[..., affected, 2] = np.random.uniform(
            *CONFIDENCE_EXTRAPOLATED, size=labels.shape[:-2] + (count,))

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'max_groups={self.max_groups})')
