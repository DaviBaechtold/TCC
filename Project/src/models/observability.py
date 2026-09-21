"""Quais keypoints o sistema de fato observa num quadro.

Camada Model. Três condições independentes decidem, e nenhuma delas basta
sozinha.

**A montagem da câmera.** É propriedade da montagem, não do modelo: com a câmera
fixa no retrovisor interno há juntas que não aparecem em quadro algum, e isso é
verificável antes de qualquer inferência. Medido sobre as 20.288 instâncias do
conjunto de validação do Drive&Act, fração dos quadros em que cada junta
corporal está anotada:

    nariz 97,2%   olhos 85,9 e 88,9%   orelha direita 88,1%
    ombros 98,6 e 99,1%   cotovelos 95,3 e 98,3%   pulsos 93,2 e 95,9%
    quadris 92,8 e 93,0%
    orelha esquerda 24,6%
    joelhos 0,2% e 3,7%   tornozelos 0,0% e 0,0%

Doze juntas passam de 80%; o restante do corpo não aparece. Os seis keypoints de
pé dependem dos tornozelos e seguem a mesma sorte. Numa webcam de mesa essa lista
é vazia: o que a mesa esconde varia com o enquadramento, e quem trata disso é o
critério de borda.

**A borda do quadro.** O estimador 2D não tem como dizer "não sei": quando a
junta está fora do recorte ele a **encosta na borda** e continua confiante. Está
medido em `src/data/estimator_noise.py` sobre o Drive&Act (tornozelo direito em
y 1,07 da altura da caixa) e se repete na gravação de mesa: os dois quadris ficam
em y mediano 707 de 720, com resposta mediana 6,09, e 95,2% deles passam o limiar
de 3,0. Confiança alta em posição inventada é exatamente o caso que o limiar não
pega.

**O limiar de resposta.** Continua necessário para a junta que está em quadro mas
foi mal localizada — o que a borda não vê.

**Por que a montagem entra como conhecimento explícito, e não como limiar.** O
estimador 2D adaptado ao Drive&Act ficou *mais* confiante sobre o que não vê: a
resposta nas juntas invisíveis subiu de 3,59 para 5,13 depois da adaptação, e a
fração delas que passa o limiar de 3,0 foi de 51% para 92%. A causa é que essas
juntas têm peso zero na função de perda, de modo que nada pune uma predição
confiante e errada ali. **Mascarar um keypoint da perda não faz o modelo ficar
calado sobre ele, faz o modelo ficar impune sobre ele.**
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

# Keypoints que cada montagem de câmera nunca enxerga. É dado, não código: uma
# montagem nova entra como uma linha aqui.
MOUNTING_ABSENT = {
    # Webcam de mesa: nada é ausente por construção. O que sai de quadro sai por
    # enquadramento, e quem decide isso é a margem de borda.
    'mesa': (),
    'retrovisor': MIRROR_VIEW_ABSENT,
}

# Distância da borda abaixo da qual a posição é tratada como encostada nela, e
# não observada. Medida nos 290 quadros da gravação de mesa com a Etapa 2, em
# fração das observações de cada grupo que a margem sinaliza:
#
#     margem  quadris  ombros  cotovelos  punhos  face  mãos
#      14 px    61,0%    0,0%       0,5%    5,9%  0,0%  5,1%
#      20 px    92,1%    0,0%       3,4%    6,7%  0,0%  6,1%
#      28 px    99,3%    0,0%       5,0%    7,8%  0,0%  7,3%
#      50 px    99,5%    0,0%       7,6%    9,3%  0,0% 10,6%
#
# 28 px pega 99,3% dos quadris — que a gravação mostra encostados na borda
# inferior — cobrando 2,5% das observações de ombro e cotovelo, que é o preço
# mais baixo do intervalo em que o critério ainda funciona. Acima disso o ganho
# nos quadris é nulo e o custo nos demais grupos cresce.
EDGE_MARGIN_PX = 28

# A margem acima foi medida num quadro de 720 px de altura; o Drive&Act tem
# 1024. Ela escala com a altura porque o efeito é geométrico, não de sensor: a
# mesma fração da pessoa cai fora do quadro.
REFERENCE_FRAME_HEIGHT_PX = 720

# O critério vale para as quatro bordas, e não só a inferior. O estimador encosta
# na borda que a junta atravessa, qualquer uma delas; nesta gravação o ocupante
# está centrado e as duas variantes quase coincidem (mãos 7,3% contra 6,2% em
# 28 px, ombros e cotovelos idênticos), de modo que a medição não decide — decide
# o mecanismo.


def mirror_view_mask(num_keypoints: int = NUM_WHOLEBODY_KEYPOINTS) -> np.ndarray:
    """Máscara booleana do que a vista de retrovisor observa.

    Face e mãos entram como observáveis: elas estão na imagem, ainda que o
    Drive&Act não as anote.
    """
    return _mounting_mask('retrovisor', num_keypoints)


def _mounting_mask(mounting: str, num_keypoints: int) -> np.ndarray:
    if mounting not in MOUNTING_ABSENT:
        raise ValueError(f'montagem desconhecida: {mounting!r}; '
                         f'conhecidas: {sorted(MOUNTING_ABSENT)}')
    mask = np.ones(num_keypoints, dtype=bool)
    mask[list(MOUNTING_ABSENT[mounting])] = False
    return mask


def observed_keypoints(keypoints: np.ndarray,
                       scores: np.ndarray,
                       frame_size: tuple[int, int],
                       min_score: float,
                       mounting: str = 'mesa') -> np.ndarray:
    """Quais keypoints foram de fato observados neste quadro.

    Args:
        keypoints: [..., K, 2] em pixels do frame.
        scores: [..., K] resposta do estimador 2D.
        frame_size: (largura, altura) do frame em pixels.
        min_score: resposta mínima. Não é probabilidade: a saída do SimCC não
            tem teto em 1.
        mounting: chave de `MOUNTING_ABSENT`.

    Returns:
        [..., K] booleano, a forma de `scores`.
    """
    width, height = frame_size
    margin = EDGE_MARGIN_PX * height / REFERENCE_FRAME_HEIGHT_PX

    x, y = keypoints[..., 0], keypoints[..., 1]
    # A comparação com NaN é falsa nos dois sentidos, e é o resultado certo: um
    # quadro sem detecção chega com NaN e não observa nada.
    inside = ((x > margin) & (x < width - margin)
              & (y > margin) & (y < height - margin))

    return (scores >= min_score) & inside & _mounting_mask(mounting,
                                                           scores.shape[-1])
