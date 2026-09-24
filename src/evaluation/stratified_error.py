"""Estratificação do erro por condição de imagem, em vez de um agregado só.

Camada Model. Existe para cumprir uma exigência do Projeto Físico que o projeto
prometia e não media: a Fase 2 especifica "avaliação estratificada por iluminação
e oclusão", e até setembro de 2026 só havia o número agregado.

O motivo de estratificar não é completude burocrática. Um erro médio de 0,0282
pode descrever um sistema uniforme ou um sistema que acerta em quadro claro e
falha no escuro --- e são conclusões opostas sobre a aplicação, que roda à noite.
Um agregado não distingue as duas, e é justamente a distinção que interessa a
quem vai montar a câmera num carro.

Duas covariáveis, ambas medidas do próprio dado e não declaradas por metadado:

**Iluminação** pela intensidade média do quadro. O Drive&Act é todo
infravermelho, então não há partição dia/noite para herdar; o que varia, e varia
bastante, é quanto sinal o emissor devolve. Medido em 300 imagens de cada
conjunto, o NIR do Drive&Act tem intensidade média 29,6 contra 105,3 do COCO em
cinza --- três vezes e meia mais escuro.

**Oclusão** pelo número de juntas corporais que a referência marca visíveis. É
proxy, e a ressalva é obrigatória: da posição de retrovisor uma junta pode faltar
por estar ocluída pelo volante **ou** por estar fora do campo de visão, e a
anotação não separa os dois casos. O que o estrato mede é "quanto do corpo está
disponível", que é a grandeza de que o erro depende.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Terços. Mais estratos dariam granularidade que a amostra não sustenta: o
# conjunto de validação do Drive&Act tem dois participantes, e dividir em cinco
# deixaria cada faixa com amostra pequena e intervalo largo.
QUANTILES = (1 / 3, 2 / 3)

STRATUM_NAMES = ('baixo', 'médio', 'alto')


@dataclass(frozen=True)
class Observation:
    """Uma instância medida, com o erro e as covariáveis do quadro."""

    error: float
    brightness: float
    visible_joints: int


def frame_brightness(frame: np.ndarray) -> float:
    """Intensidade média do quadro, em [0, 255].

    Aceita quadro de um ou três canais. Num quadro infravermelho convertido para
    BGR os três canais são iguais, de modo que a média sobre todos eles é a
    mesma que sobre um --- e não custa ramificar por forma.
    """
    return float(np.mean(frame))


def stratify(values: list[float], names: tuple[str, ...] = STRATUM_NAMES
             ) -> tuple[list[str], dict[str, tuple[float, float]]]:
    """Rotula cada valor com seu terço, e devolve as fronteiras usadas.

    As fronteiras são quantis da própria amostra, não limiares absolutos: o que
    é "escuro" depende do conjunto, e fixar 50 de intensidade deixaria todos os
    quadros do Drive&Act num único estrato.

    Returns:
        `(rótulos, fronteiras)`, onde `fronteiras[nome]` é o intervalo
        `(mínimo, máximo)` observado naquele estrato --- é ele que torna o
        estrato interpretável por quem lê a tabela.
    """
    array = np.asarray(values, dtype=np.float64)
    cortes = np.quantile(array, QUANTILES)
    indices = np.searchsorted(cortes, array, side='right')

    rotulos = [names[min(int(i), len(names) - 1)] for i in indices]
    fronteiras = {}
    for nome in names:
        do_estrato = array[[r == nome for r in rotulos]]
        if do_estrato.size:
            fronteiras[nome] = (round(float(do_estrato.min()), 1),
                                round(float(do_estrato.max()), 1))
    return rotulos, fronteiras


def summarize(observations: list[Observation]) -> dict[str, dict]:
    """Erro por estrato de iluminação e de oclusão.

    Reporta mediana e média juntas de propósito. Onde as duas se separam, o
    estrato tem cauda --- alguns quadros muito ruins em vez de degradação
    uniforme ---, e essa é informação diferente sobre o sistema.
    """
    if not observations:
        raise ValueError('Nenhuma observação para estratificar.')

    por_criterio = {
        'iluminacao': [o.brightness for o in observations],
        'oclusao': [float(o.visible_joints) for o in observations],
    }
    erros = np.array([o.error for o in observations])

    relatorio: dict[str, dict] = {}
    for criterio, valores in por_criterio.items():
        rotulos, fronteiras = stratify(valores)
        estratos = {}
        for nome in STRATUM_NAMES:
            mascara = np.array([r == nome for r in rotulos])
            if not mascara.any():
                continue
            estratos[nome] = {
                'faixa': fronteiras[nome],
                'instancias': int(mascara.sum()),
                'erro_mediano': round(float(np.median(erros[mascara])), 4),
                'erro_medio': round(float(erros[mascara].mean()), 4),
            }
        relatorio[criterio] = estratos

    relatorio['agregado'] = {
        'instancias': len(observations),
        'erro_mediano': round(float(np.median(erros)), 4),
        'erro_medio': round(float(erros.mean()), 4),
    }
    return relatorio


def degradation(estratos: dict[str, dict], melhor: str, pior: str) -> float:
    """Razão entre o erro médio de dois estratos, em pontos percentuais.

    O número que a estratificação existe para produzir: quanto o sistema piora
    entre a condição favorável e a adversa. Um valor próximo de zero significa
    que a condição não é fator, o que também é resultado.
    """
    return round(100.0 * (estratos[pior]['erro_medio'] /
                          estratos[melhor]['erro_medio'] - 1.0), 1)
