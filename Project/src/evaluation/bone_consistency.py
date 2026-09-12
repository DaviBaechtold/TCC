"""Mede a qualidade de uma pose 3D sem precisar de referência alguma.

Camada Model. Existe porque a régua disponível não serve: a pose tridimensional
do Drive&Act, única referência do domínio de aplicação, tem ao menos 30,1mm de
incerteza por quadro --- mais que o dobro da diferença entre os modelos que se
quer comparar. Medir contra ela é medir o ruído dela.

O princípio aqui dispensa referência: **o comprimento de um osso é constante**.
A distância entre ombro e cotovelo da mesma pessoa não muda enquanto ela se
move, de modo que toda variação nela, na pose *predita*, é erro do modelo. Um
estimador perfeito produziria desvio zero.

O que a métrica capta e o que não capta, para que não seja superinterpretada:

**Capta** incoerência temporal e tremor --- a pose que oscila entre quadros, que
é exatamente o defeito que a fusão temporal do Módulo 3 existe para suprimir.

**Não capta** erro sistemático. Um modelo que encolha a pessoa inteira em 20%,
de forma consistente, tem desvio zero e está errado. A métrica é necessária e
não suficiente, e por isso acompanha o MPJPE em vez de substituí-lo.

Serve também de piso de comparação: a própria referência do Drive&Act mede
30,1mm nesta métrica, de modo que um modelo abaixo disso é mais coerente que a
anotação contra a qual está sendo avaliado.
"""

from __future__ import annotations

import numpy as np

MILLIMETERS = 1000.0

# Ossos cujos dois extremos a vista de retrovisor observa. Um osso com uma ponta
# extrapolada mediria a extrapolação, e não a coerência da pose.
OBSERVABLE_BONES = {
    'ombro a ombro': (5, 6),
    'braço esquerdo': (5, 7),
    'braço direito': (6, 8),
    'antebraço esquerdo': (7, 9),
    'antebraço direito': (8, 10),
    'quadril a quadril': (11, 12),
    'tronco esquerdo': (5, 11),
    'tronco direito': (6, 12),
}

# Abaixo disto a sequência é curta demais para que um desvio signifique algo.
MIN_FRAMES = 20


def bone_lengths(sequence: np.ndarray,
                 bones: dict[str, tuple[int, int]] = OBSERVABLE_BONES
                 ) -> dict[str, np.ndarray]:
    """Comprimento de cada osso, quadro a quadro.

    Args:
        sequence: [T, K, 3] em metros.

    Returns:
        Um vetor de T comprimentos por osso.
    """
    return {
        nome: np.linalg.norm(sequence[:, a] - sequence[:, b], axis=-1)
        for nome, (a, b) in bones.items()
    }


def consistency(sequence: np.ndarray,
                bones: dict[str, tuple[int, int]] = OBSERVABLE_BONES
                ) -> dict[str, float]:
    """Desvio do comprimento de cada osso ao longo da sequência, em milímetros.

    Returns:
        Desvio por osso, mais a chave `mediana` agregando todos. Vazio se a
        sequência for curta demais para o número significar algo.
    """
    if len(sequence) < MIN_FRAMES:
        return {}

    desvios = {nome: float(np.std(valores) * MILLIMETERS)
               for nome, valores in bone_lengths(sequence, bones).items()}
    desvios['mediana'] = float(np.median(list(desvios.values())))
    return desvios
