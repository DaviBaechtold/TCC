#!/usr/bin/env python
"""Verifica as três condições que decidem se um keypoint foi observado.

O caso que motiva o teste: o estimador 2D coloca a junta fora de quadro
**encostada na borda** e continua confiante — os dois quadris da gravação de
mesa ficam em y 707 de 720 com resposta 6,09, bem acima do limiar de 3,0. Um
teste que só cobrisse o limiar passaria com o defeito presente.

Só CPU. Executar:  python tests/test_observability.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.observability import (EDGE_MARGIN_PX, MIRROR_VIEW_ABSENT,
                                      NUM_WHOLEBODY_KEYPOINTS,
                                      observed_keypoints)

FRAME_SIZE = (1280, 720)
MIN_SCORE = 3.0
CONFIDENT = 9.0

# Centro do quadro: longe de qualquer borda, para isolar a condição sob teste.
CENTER = (640.0, 360.0)


def _pose(positions: dict[int, tuple[float, float]],
          scores: np.ndarray | None = None):
    keypoints = np.tile(np.array(CENTER, np.float32),
                        (NUM_WHOLEBODY_KEYPOINTS, 1))
    for index, position in positions.items():
        keypoints[index] = position
    if scores is None:
        scores = np.full(NUM_WHOLEBODY_KEYPOINTS, CONFIDENT, np.float32)
    return keypoints, scores


def test_borda_inferior():
    """A junta encostada na borda inferior não é observada, por confiante que seja."""
    keypoints, scores = _pose({11: (591.0, 707.0), 12: (620.0, 708.0)})
    observed = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE)
    assert not observed[11] and not observed[12], 'quadril na borda passou'
    assert observed[5] and observed[6], 'ombro no centro foi reprovado'
    print('  borda inferior: quadris em y 707 reprovados, ombros aprovados  OK')


def test_todas_as_bordas():
    """O critério vale para as quatro bordas, não só a inferior."""
    width, height = FRAME_SIZE
    inside = EDGE_MARGIN_PX + 1.0
    outside = EDGE_MARGIN_PX - 1.0
    keypoints, scores = _pose({
        5: (outside, 360.0), 6: (width - outside, 360.0),
        7: (640.0, outside), 8: (640.0, height - outside),
        9: (inside, inside), 10: (width - inside, height - inside),
    })
    observed = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE)
    assert not observed[5:9].any(), 'junta dentro da margem passou'
    assert observed[9] and observed[10], 'junta fora da margem foi reprovada'
    print('  quatro bordas: 4 reprovadas dentro da margem, 2 aprovadas fora  OK')


def test_fora_do_quadro():
    """Coordenada negativa ou além da largura não é observação."""
    width, height = FRAME_SIZE
    keypoints, scores = _pose({
        13: (-40.0, 400.0), 14: (width + 90.0, 400.0),
        15: (640.0, height + 200.0), 16: (np.nan, np.nan),
    })
    observed = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE)
    assert not observed[13:17].any(), 'junta fora do quadro passou'
    print('  fora do quadro: 3 posições impossíveis e 1 NaN reprovadas  OK')


def test_limiar():
    """O limiar continua necessário: pega a junta em quadro e mal localizada."""
    scores = np.full(NUM_WHOLEBODY_KEYPOINTS, CONFIDENT, np.float32)
    scores[9] = MIN_SCORE - 0.01
    scores[10] = MIN_SCORE
    keypoints, scores = _pose({}, scores)
    observed = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE)
    assert not observed[9], 'resposta abaixo do limiar passou'
    assert observed[10], 'resposta exatamente no limiar foi reprovada'
    print('  limiar: reprova abaixo, aprova no valor exato  OK')


def test_montagem():
    """A ausência por montagem não depende de posição nem de resposta."""
    keypoints, scores = _pose({})
    mesa = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE, 'mesa')
    retrovisor = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE,
                                    'retrovisor')
    assert mesa.all(), 'a mesa não deveria declarar nenhuma junta ausente'
    assert not retrovisor[list(MIRROR_VIEW_ABSENT)].any()
    assert retrovisor.sum() == NUM_WHOLEBODY_KEYPOINTS - len(MIRROR_VIEW_ABSENT)
    try:
        observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE, 'capo')
    except ValueError:
        pass
    else:
        raise AssertionError('montagem desconhecida passou em silêncio')
    print(f'  montagem: mesa 133/133, retrovisor '
          f'{int(retrovisor.sum())}/133, montagem inválida recusada  OK')


def test_lote():
    """A forma de saída acompanha a de entrada, para N pessoas de uma vez."""
    keypoints = np.stack([_pose({11: (591.0, 707.0)})[0] for _ in range(3)])
    scores = np.full((3, NUM_WHOLEBODY_KEYPOINTS), CONFIDENT, np.float32)
    observed = observed_keypoints(keypoints, scores, FRAME_SIZE, MIN_SCORE)
    assert observed.shape == scores.shape, observed.shape
    assert not observed[:, 11].any()
    assert observed[:, 5].all()
    print(f'  lote: forma {observed.shape} preservada para 3 pessoas  OK')


def main():
    print('observabilidade:')
    test_borda_inferior()
    test_todas_as_bordas()
    test_fora_do_quadro()
    test_limiar()
    test_montagem()
    test_lote()
    print('\nas três condições decidem, e nenhuma sozinha')


if __name__ == '__main__':
    main()
