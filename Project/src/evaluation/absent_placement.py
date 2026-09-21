"""Onde o estimador 2D coloca uma junta que não está na imagem.

Camada Model. O estimador nunca devolve coordenada fora do recorte: diante de
uma junta ausente ele devolve alguma posição dentro dele, e **qual** posição é
uma propriedade medível do modelo, não uma suposição. Foi medindo isso que o
projeto descobriu que o quadril é grudado na borda inferior, e foi por reproduzir
esse mecanismo no treino que o colapso do tronco foi corrigido.

Esta medição estende a caracterização às pernas, porque a mesma correção **não**
transferiu para elas: a simulação prende tudo na linha do corte, e a inspeção
mostrou joelho e tornozelo caindo sobre o peito e sobre o braço levantado.

Tudo é reportado em **larguras de ombro**, e não em pixels, para que a webcam de
mesa e a câmera de retrovisor --- que enxergam a pessoa em tamanhos diferentes
--- produzam números comparáveis e utilizáveis como parâmetro de simulação.
"""

from __future__ import annotations

import numpy as np

SHOULDERS = (5, 6)

# Juntas cuja posição se quer caracterizar, e as que servem de referência por
# estarem sempre em quadro nas duas montagens.
VISIBLE_REFERENCE = (0, 5, 6, 7, 8, 9, 10)

MIN_SHOULDER_WIDTH_PX = 10.0


def _shoulder_frame(keypoints: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Origem no meio dos ombros e escala pela largura deles."""
    left, right = keypoints[SHOULDERS[0]], keypoints[SHOULDERS[1]]
    width = float(np.linalg.norm(left - right))
    if width < MIN_SHOULDER_WIDTH_PX:
        return None
    return (left + right) / 2.0, width


def placement(keypoints: np.ndarray, scores: np.ndarray,
              frame_size: tuple[int, int], joints: tuple[int, ...]
              ) -> dict[int, dict[str, float]] | None:
    """Posição de cada junta pedida, no referencial dos ombros.

    Returns:
        Por junta: `x` e `y` em larguras de ombro a partir do meio dos ombros
        (y positivo para baixo), `borda` em larguras de ombro até a borda
        inferior do quadro (negativo se além dela), `resposta`, e a distância
        até o keypoint visível mais próximo, que denuncia a junta que grudou no
        braço ou no tronco em vez de na borda.
    """
    referencial = _shoulder_frame(keypoints)
    if referencial is None:
        return None
    origem, largura = referencial
    _, altura = frame_size

    medidas = {}
    for junta in joints:
        posicao = keypoints[junta]
        relativa = (posicao - origem) / largura
        distancias = np.linalg.norm(
            keypoints[list(VISIBLE_REFERENCE)] - posicao, axis=-1) / largura
        medidas[junta] = {
            'x': float(relativa[0]),
            'y': float(relativa[1]),
            'borda': float((altura - posicao[1]) / largura),
            'resposta': float(scores[junta]),
            'distancia_ao_visivel': float(distancias.min()),
            'vizinho': int(VISIBLE_REFERENCE[int(distancias.argmin())]),
        }
    return medidas


def summarize(amostras: list[dict[int, dict[str, float]]],
              joints: tuple[int, ...]) -> dict[str, dict[str, float]]:
    """Distribuição por junta, com os percentis que servem de parâmetro.

    A mediana sozinha esconderia o que interessa aqui: se a junta às vezes
    encosta na borda e às vezes cai no peito, o parâmetro de simulação não é a
    mediana, é a proporção entre os dois regimes.
    """
    resumo = {}
    for junta in joints:
        valores = [a[junta] for a in amostras if junta in a]
        if not valores:
            continue
        def coluna(nome):
            return np.array([v[nome] for v in valores], dtype=float)

        borda = coluna('borda')
        vizinho = coluna('distancia_ao_visivel')
        resumo[str(junta)] = {
            'x_mediano': round(float(np.median(coluna('x'))), 3),
            'y_mediano': round(float(np.median(coluna('y'))), 3),
            'y_p10': round(float(np.percentile(coluna('y'), 10)), 3),
            'y_p90': round(float(np.percentile(coluna('y'), 90)), 3),
            'borda_mediana': round(float(np.median(borda)), 3),
            'fracao_junto_da_borda': round(float((borda < 0.15).mean()), 3),
            'fracao_alem_da_borda': round(float((borda < 0.0).mean()), 3),
            'fracao_sobre_corpo_visivel': round(
                float((vizinho < 0.25).mean()), 3),
            'resposta_mediana': round(float(np.median(coluna('resposta'))), 2),
            'amostras': len(valores),
        }
    return resumo
