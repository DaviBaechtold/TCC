#!/usr/bin/env python
"""Verifica o One Euro: tira tremor parado sem atrasar o movimento.

O compromisso do filtro é entre as duas coisas, e um teste que medisse só uma
passaria com a outra quebrada. Aqui o atraso é conferido contra a fórmula do
passa-baixa de primeira ordem, `rate/(2·pi·fc)` amostras, e depois contra si
mesmo com `beta` alto.

Só CPU. Executar:  python tests/test_temporal_filter.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.temporal_filter import (DEFAULT_BETA_PER_METRE,
                                        DEFAULT_MIN_CUTOFF_HZ, OneEuroFilter)

# Taxa da gravação sobre a qual o filtro foi ajustado.
RATE_HZ = 33.735

NUM_SAMPLES = 400

# Tremor da ordem do que o lifting devolve parado, em metros.
NOISE_M = 0.004

# O atraso assintótico só vale depois que o transitório passa.
SETTLED_FROM = 200

# Folga sobre a fórmula. O atraso medido bate na segunda casa (5,37 contra 5,37
# amostras); a tolerância protege contra erro de convenção — hertz trocado por
# radianos, taxa por período — e não contra variação numérica.
LAG_TOLERANCE = 0.15


def test_constante():
    """Entrada constante sai idêntica: o filtro não inventa transitório."""
    filtro = OneEuroFilter(RATE_HZ)
    values = np.full((133, 3), 0.25)
    outputs = [filtro(values) for _ in range(20)]
    for index, output in enumerate(outputs):
        assert np.allclose(output, values), f'quadro {index} desviou'
    print('  constante: 20 quadros de [133,3] inalterados  OK')


def test_reduz_ruido():
    """Ruído sobre sinal parado é atenuado."""
    rng = np.random.default_rng(0)
    truth = np.zeros((NUM_SAMPLES, 133, 3))
    noisy = truth + rng.normal(0.0, NOISE_M, truth.shape)

    filtro = OneEuroFilter(RATE_HZ)
    filtered = np.stack([filtro(sample) for sample in noisy])

    entrada = np.abs(np.diff(noisy[SETTLED_FROM:], axis=0)).mean()
    saida = np.abs(np.diff(filtered[SETTLED_FROM:], axis=0)).mean()
    reducao = 1.0 - saida / entrada
    assert reducao > 0.5, f'reducao de apenas {reducao:.1%}'
    print(f'  ruido parado: tremor cai {reducao:.1%}  OK')


def _lag_on_ramp(beta: float) -> float:
    """Atraso em amostras sobre uma rampa de velocidade constante."""
    speed_m_per_sample = 0.01
    ramp = np.arange(NUM_SAMPLES, dtype=float) * speed_m_per_sample
    filtro = OneEuroFilter(RATE_HZ, beta=beta)
    filtered = np.array([float(filtro(np.array([value]))[0]) for value in ramp])
    return float(np.mean((ramp - filtered)[SETTLED_FROM:]) / speed_m_per_sample)


def test_atraso_sem_beta():
    """Com beta zero o filtro é um passa-baixa fixo, e o atraso é o da fórmula."""
    esperado = RATE_HZ / (2.0 * np.pi * DEFAULT_MIN_CUTOFF_HZ)
    medido = _lag_on_ramp(0.0)
    erro = abs(medido - esperado) / esperado
    assert erro < LAG_TOLERANCE, f'{medido:.2f} contra {esperado:.2f} amostras'
    print(f'  rampa, beta 0: atraso {medido:.2f} amostras contra '
          f'{esperado:.2f} previstas  OK')


def test_beta_reduz_atraso():
    """É para isto que beta existe: o movimento rápido passa quase sem atraso."""
    sem_beta = _lag_on_ramp(0.0)
    com_beta = _lag_on_ramp(DEFAULT_BETA_PER_METRE)
    assert com_beta < sem_beta / 4.0, f'{com_beta:.2f} contra {sem_beta:.2f}'
    print(f'  rampa, beta {DEFAULT_BETA_PER_METRE:.0f}: atraso {com_beta:.2f} '
          f'contra {sem_beta:.2f} amostras  OK')


def test_reset():
    """Depois do reset o próximo quadro sai intacto, como o primeiro."""
    filtro = OneEuroFilter(RATE_HZ)
    for value in np.linspace(0.0, 1.0, 50):
        filtro(np.array([value]))
    filtro.reset()
    novo = np.array([7.0])
    assert filtro(novo) == novo, 'o estado antigo sobreviveu ao reset'
    print('  reset: o quadro seguinte sai intacto  OK')


def test_troca_de_forma():
    """Outra forma é outro sinal: o estado da pessoa anterior não vaza."""
    filtro = OneEuroFilter(RATE_HZ)
    for _ in range(10):
        filtro(np.zeros((133, 3)))
    outra = np.full((17, 3), 5.0)
    assert np.array_equal(filtro(outra), outra), 'estado vazou entre formas'
    print('  troca de forma: [133,3] para [17,3] recomeça limpo  OK')


def main():
    print('one euro:')
    test_constante()
    test_reduz_ruido()
    test_atraso_sem_beta()
    test_beta_reduz_atraso()
    test_reset()
    test_troca_de_forma()
    print('\ntira tremor parado sem atrasar o movimento')


if __name__ == '__main__':
    main()
