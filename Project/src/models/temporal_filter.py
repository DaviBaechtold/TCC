"""Filtro One Euro (Casiez, Roussel e Vogel, 2012) para a pose ao vivo.

Camada Model. Suaviza uma série temporal sem introduzir o atraso fixo de uma
média móvel: a frequência de corte sobe com a velocidade do sinal, de modo que
o que treme parado é filtrado e o que se move rápido passa.

**Por que na saída 3D e não na entrada 2D.** Medido sobre esta gravação, filtrar
os 133 keypoints 2D antes do lifting reduz menos o tremor da figura 3D do que
filtrar a própria saída 3D: a rede é temporal e redistribui o ruído de entrada
pelos dezesseis quadros da janela, de modo que suavizar antes ataca o sintoma no
lugar errado. Na saída, a medição deu 69% de redução de tremor a cerca de 0,14
quadro de atraso.

**A unidade de `beta` importa e não é adimensional.** `beta` está em hertz por
unidade de velocidade, isto é, 1/unidade: ele converte a velocidade do sinal em
acréscimo de frequência de corte. Trocar a unidade da coordenada sem trocar
`beta` muda o filtro em silêncio. A medição do diagnóstico deu 0,05 para
coordenadas em **milímetros**; o lifting devolve **metros**, e o mesmo filtro em
metros é 50.
"""

from __future__ import annotations

import numpy as np

# Corte com o sinal parado. Escolhido por varredura sobre a gravação da webcam,
# com a pose do v3 e a escala de entrada já corrigida, contra o controle de
# movimento: em 1 Hz o filtro tirava 80% do tremor da face mas cortava 61% do
# movimento e atrasava 5,4 quadros parado --- 180ms, o mesmo orçamento de
# latência que este projeto já recusou ao escolher o quadro causal no Módulo 3.
# Em 4 Hz o tremor da face cai 62% (4,39 para 1,67mm) e o atraso parado fica em
# 1,2 quadro, encolhendo ainda mais quando o corpo se move, que é quando o
# atraso seria percebido.
DEFAULT_MIN_CUTOFF_HZ = 4.0

# Hertz por metro por segundo. Ver a nota de unidade na docstring do módulo: a
# medição foi feita em milímetros e deu 0,05, que são 50 por metro.
DEFAULT_BETA_PER_METRE = 50.0

# Corte do filtro que suaviza a própria estimativa de velocidade. Fica em 1 Hz,
# e não acompanha o corte mínimo: uma velocidade ruidosa abriria o corte em
# quadros parados e devolveria o tremor que o filtro acabou de tirar.
DEFAULT_D_CUTOFF_HZ = 1.0


def _alpha(cutoff_hz: float | np.ndarray, rate_hz: float) -> float | np.ndarray:
    """Coeficiente do passa-baixa de primeira ordem para este corte."""
    time_constant = 1.0 / (2.0 * np.pi * cutoff_hz)
    return 1.0 / (1.0 + rate_hz * time_constant)


class OneEuroFilter:
    """Passa-baixa adaptativo, causal, vetorizado sobre qualquer forma.

    Guarda estado entre chamadas; `reset` o descarta.
    """

    def __init__(self,
                 rate_hz: float,
                 min_cutoff_hz: float = DEFAULT_MIN_CUTOFF_HZ,
                 beta: float = DEFAULT_BETA_PER_METRE,
                 d_cutoff_hz: float = DEFAULT_D_CUTOFF_HZ):
        """
        Args:
            rate_hz: taxa de amostragem. O filtro é de tempo discreto e assume
                intervalo constante; ao vivo isso é a taxa de quadros.
            min_cutoff_hz: corte com o sinal parado.
            beta: hertz por unidade de velocidade do sinal. Depende da unidade
                da coordenada — ver a docstring do módulo.
            d_cutoff_hz: corte aplicado à estimativa de velocidade.
        """
        if rate_hz <= 0:
            raise ValueError('a taxa de amostragem precisa ser positiva')
        self._rate_hz = rate_hz
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._d_cutoff_hz = d_cutoff_hz
        self._previous: np.ndarray | None = None
        self._speed: np.ndarray | None = None

    def reset(self) -> None:
        """Descarta o estado. Necessário ao trocar de fonte ou de pessoa."""
        self._previous = None
        self._speed = None

    def __call__(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)

        # Uma forma diferente é outro sinal — outra pessoa, outro conjunto de
        # juntas. Continuar com o estado antigo misturaria as duas séries.
        if self._previous is None or self._previous.shape != values.shape:
            self._previous = values
            self._speed = np.zeros_like(values)
            return values.copy()

        speed = (values - self._previous) * self._rate_hz
        alpha_speed = _alpha(self._d_cutoff_hz, self._rate_hz)
        self._speed = alpha_speed * speed + (1.0 - alpha_speed) * self._speed

        cutoff = self._min_cutoff_hz + self._beta * np.abs(self._speed)
        alpha = _alpha(cutoff, self._rate_hz)
        filtered = alpha * values + (1.0 - alpha) * self._previous

        self._previous = filtered
        return filtered
