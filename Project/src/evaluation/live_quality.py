"""Qualidade da pose tridimensional ao vivo, sem referência alguma.

Camada Model. Numa captura ao vivo não existe ground truth 3D --- o painel
declara essa impossibilidade em vez de exibir um número aproximado --- e mesmo
assim o sistema precisa de um número, porque foi por inspeção visual que dois
defeitos apareceram: o tremor e o corpo que não fecha.

Três famílias de medida, e nenhuma delas usa anotação:

**Tremor.** A mediana da norma da segunda diferença temporal por região. É a
medida usual de suavidade, e capta exatamente a oscilação que a fusão temporal
do Módulo 3 existe para suprimir.

**Controle de movimento.** Obrigatório junto do tremor, pela mesma razão que
acompanha a coerência de osso em `bone_consistency`: uma pose congelada tem
tremor zero. Sem o controle, um modelo que amortece movimento vence sem estar
mais correto.

**Plausibilidade anatômica.** Comprimentos de osso contra faixas de adulto. Um
tronco de 15cm não precisa de referência para ser diagnosticado como errado, e
foi assim que o colapso do quadril apareceu antes de existir métrica para ele.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.bone_consistency import consistency, motion

MILLIMETERS = 1000.0

GROUPS = {
    'cabeca': list(range(5)),
    'face': list(range(23, 91)),
    'ombros': [5, 6],
    'cotovelos e pulsos': [7, 8, 9, 10],
    'quadris': [11, 12],
    'pernas e pes': list(range(13, 23)),
    'maos': list(range(91, 133)),
}

# Faixas de adulto, em milímetros, para distância entre centros articulares.
# Fonte: proporções antropométricas de Drillis e Contini, com a interpupilar da
# literatura oftalmológica. Servem de diagnóstico grosseiro --- um osso fora da
# faixa por um fator de dois é defeito, não variação individual.
ADULT_RANGES_MM = {
    'largura de ombros': (330.0, 420.0),
    'ombro-quadril': (450.0, 550.0),
    'largura de quadril': (180.0, 280.0),
    'coxa': (380.0, 470.0),
    'canela': (360.0, 440.0),
    'interpupilar': (58.0, 70.0),
}

_BONES = {
    'largura de ombros': (5, 6),
    'ombro-quadril': ((5, 6), (11, 12)),   # meio dos ombros ao meio do quadril
    'largura de quadril': (11, 12),
    'coxa': ((11, 13), (12, 14)),          # média dos dois lados
    'canela': ((13, 15), (14, 16)),
}

# Centros dos olhos no bloco de 68 landmarks de face, que começa no índice 23.
_RIGHT_EYE = list(range(23 + 36, 23 + 42))
_LEFT_EYE = list(range(23 + 42, 23 + 48))


def jitter_mm(poses: np.ndarray) -> dict[str, float]:
    """Mediana da norma da segunda diferença temporal, por região, em mm.

    Args:
        poses: [T, 133, 3] em metros, ancoradas na raiz.
    """
    if len(poses) < 3:
        return {}
    second = np.linalg.norm(np.diff(poses, n=2, axis=0), axis=-1)
    return {nome: round(float(np.median(second[:, indices]) * MILLIMETERS), 2)
            for nome, indices in GROUPS.items()}


def _pair_length(poses: np.ndarray, pair) -> np.ndarray:
    start, end = pair
    if isinstance(start, tuple):        # ponto médio contra ponto médio
        origem = poses[:, list(start)].mean(axis=1)
        destino = poses[:, list(end)].mean(axis=1)
        return np.linalg.norm(origem - destino, axis=-1)
    return np.linalg.norm(poses[:, start] - poses[:, end], axis=-1)


def body_geometry(poses: np.ndarray) -> dict[str, dict[str, float]]:
    """Comprimentos medianos, desvio temporal e se caem na faixa de adulto."""
    comprimentos: dict[str, np.ndarray] = {}
    for nome, definicao in _BONES.items():
        if nome in ('coxa', 'canela'):
            esquerda, direita = definicao
            comprimentos[nome] = (_pair_length(poses, esquerda) +
                                  _pair_length(poses, direita)) / 2.0
        else:
            comprimentos[nome] = _pair_length(poses, definicao)

    olhos_direito = poses[:, _RIGHT_EYE].mean(axis=1)
    olhos_esquerdo = poses[:, _LEFT_EYE].mean(axis=1)
    comprimentos['interpupilar'] = np.linalg.norm(
        olhos_direito - olhos_esquerdo, axis=-1)

    geometria = {}
    for nome, valores in comprimentos.items():
        mediana = float(np.median(valores) * MILLIMETERS)
        minimo, maximo = ADULT_RANGES_MM[nome]
        geometria[nome] = {
            'mediana_mm': round(mediana, 1),
            'desvio_mm': round(float(np.std(valores) * MILLIMETERS), 1),
            'faixa_adulto_mm': [minimo, maximo],
            'dentro_da_faixa': bool(minimo <= mediana <= maximo),
        }
    return geometria


def observed_fraction(observed: np.ndarray) -> dict[str, float]:
    """Fração dos quadros em que cada região foi de fato observada.

    É o que permite ao documento dizer quanto do corpo exibido é medição e
    quanto é previsão.
    """
    return {nome: round(float(observed[:, indices].mean()), 3)
            for nome, indices in GROUPS.items()}


def report(poses: np.ndarray, observed: np.ndarray) -> dict:
    """Relatório completo de uma sequência ao vivo."""
    coerencia = consistency(poses)
    return {
        'quadros': int(len(poses)),
        'tremor_mm': jitter_mm(poses),
        'movimento_mm': round(motion(poses), 2),
        'coerencia_osso_mm': (round(coerencia['mediana'], 2)
                              if 'mediana' in coerencia else None),
        'coerencia_forma': (round(coerencia['mediana_relativa'], 4)
                            if 'mediana_relativa' in coerencia else None),
        'geometria': body_geometry(poses),
        'fracao_observada': observed_fraction(observed),
    }
