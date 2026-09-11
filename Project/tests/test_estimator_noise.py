#!/usr/bin/env python
"""Verifica que a simulação do estimador 2D produz o sinal que ela promete.

A transformação anterior, que apagava keypoints, não tinha teste e piorou o
domínio veicular em 25mm. O que ela errava não era código: era a hipótese. Este
teste não valida a hipótese --- só a medição faz isso --- mas trava as três
propriedades sem as quais a hipótese não chega a ser exercitada.

Executar:  python tests/test_estimator_noise.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.estimator_noise import (CONFIDENCE_EXTRAPOLATED,
                                      CONFIDENCE_OBSERVED, KEYPOINT_GROUPS,
                                      SimulatedEstimatorNoise)

SEQUENCE, KEYPOINTS = 16, 133


def uma_janela() -> dict:
    """Janela com posições distintas por keypoint e confiança constante em 1.

    Confiança constante é como o H3WB de fato chega: desvio padrão zero, um
    único valor distinto em todo o conjunto. É o que a transformação existe
    para quebrar.
    """
    labels = np.zeros((SEQUENCE, KEYPOINTS, 3), dtype=np.float32)
    labels[..., 0] = np.linspace(-1, 1, KEYPOINTS)
    labels[..., 1] = np.linspace(-1, 1, KEYPOINTS)
    labels[..., 2] = 1.0
    return {'keypoint_labels': labels}


def main():
    np.random.seed(0)

    # 1. A confiança deixa de ser constante mesmo quando nada é deslocado.
    intocada = SimulatedEstimatorNoise(prob=0.0).transform(uma_janela())
    confianca = intocada['keypoint_labels'][..., 2]
    assert confianca.std() > 0.01, 'a confiança continuou constante'
    assert CONFIDENCE_OBSERVED[0] <= confianca.min() <= confianca.max() <= 1.0
    print(f'  confiança varia mesmo sem deslocar    desvio {confianca.std():.3f}  OK')

    # 2. Confiança média mais baixa acompanha posição deslocada. A correlação
    #    é o ponto inteiro: sem ela a rede não tem por que consultar o canal.
    original = uma_janela()['keypoint_labels']
    ruidosa = SimulatedEstimatorNoise(prob=1.0).transform(uma_janela())
    labels = ruidosa['keypoint_labels']
    deslocamento = np.linalg.norm(labels[..., :2] - original[..., :2], axis=-1)

    # O marcador de quem foi extrapolado é o deslocamento, e não a confiança:
    # as duas faixas se sobrepõem de propósito, como se vê na verificação 3.
    afetados = deslocamento.max(axis=0) > 1e-6
    assert afetados.any(), 'nenhum keypoint foi deslocado'
    confianca = labels[..., 2]
    assert confianca[:, afetados].mean() < confianca[:, ~afetados].mean(), (
        'a confiança média não distingue extrapolado de observado')
    print(f'  confiança acompanha deslocamento      '
          f'{confianca[:, afetados].mean():.2f} contra '
          f'{confianca[:, ~afetados].mean():.2f}  OK')

    # 3. As faixas se sobrepõem, e isso é fiel e não descuido. Na medição real,
    #    92% das juntas invisíveis passam o limiar de detecção. Simular uma
    #    separação mais limpa que a realidade produziria um modelo que confia
    #    demais no canal justamente onde ele é ambíguo.
    assert CONFIDENCE_EXTRAPOLATED[1] > CONFIDENCE_OBSERVED[0], (
        'as faixas deixaram de se sobrepor; a simulação ficou otimista demais')
    sobreposicao = CONFIDENCE_EXTRAPOLATED[1] - CONFIDENCE_OBSERVED[0]
    print(f'  faixas se sobrepõem, como no real     {sobreposicao:.2f}  OK')

    # 4. O grupo some da janela inteira. Um membro fora de quadro continua fora
    #    enquanto a câmera não se mexe; ausência intermitente é outro regime.
    assert np.allclose(deslocamento[:, afetados],
                       deslocamento[0, afetados]), (
        'o deslocamento variou entre quadros da mesma janela')
    print(f'  ausência consistente na janela        '
          f'{int(afetados.sum())} keypoints  OK')

    # 5. Os grupos são anatômicos, não keypoints avulsos.
    indices = set(np.flatnonzero(afetados).tolist())
    assert any(indices >= set(g) for g in KEYPOINT_GROUPS.values()), (
        'o conjunto removido não corresponde a nenhum grupo anatômico')
    print('  remoção por grupo anatômico                          OK')

    print('\na simulação produz o canal de confiança informativo que faltava')


if __name__ == '__main__':
    main()
