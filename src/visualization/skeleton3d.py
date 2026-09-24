"""Desenha o esqueleto tridimensional numa projeção ortográfica rotacionável.

Camada View. Recebe a pose 3D pronta do Módulo 3 e apenas desenha; nenhuma
regra de estimação vive aqui.

A projeção é ortográfica, e não em perspectiva, por dois motivos. O primeiro é
que a pose devolvida pelo lifting é relativa à raiz e sua escala absoluta
depende da calibração da câmera, que pode não existir; uma projeção em
perspectiva exigiria uma distância focal que seria arbitrada, e a arbitrariedade
apareceria na imagem como deformação. O segundo é de leitura: numa vista
ortográfica, comprimentos paralelos ao plano da tela são comparáveis a olho, o
que é justamente o que se quer inspecionar ao validar uma pose.

**O enquadramento é fixo, e isso corrige um defeito medido.** A versão anterior
recalculava escala e centro a cada quadro a partir do envelope da própria pose.
Como o envelope é dominado pelas extremidades — quadris encostados na borda
inferior do quadro, pontas de dedo — ele respirava a cada quadro: sobre a mesma
pose 3D dos 290 quadros da gravação de mesa, os pontos desenhados andavam 27,3 px
entre quadros consecutivos em mediana, contra 2,9 px aqui. **A figura tremia
porque a moldura tremia, não porque a pose tremesse.** Aqui a escala vem da
calibração, ou de um raio do próprio corpo que é invariante à rotação, e a
âncora é o meio dos ombros numa posição fixa da tela. Nada de enquadrar por um
subconjunto que muda de quadro para quadro, e nada de enquadrar depois de girar.

**Todo o corpo é desenhado, e o que foi predito se distingue do que foi
observado.** Esconder as juntas que a câmera não enxerga deixava a figura sem
pernas, que é o contrário do que se quer mostrar. Elas aparecem, em traço fino e
esmaecido contra o fundo do painel, de modo que predição nunca se confunda com
observação.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.panel import PANEL_FILL
from src.visualization.skeleton import SKELETON_LINKS

# Corpo e pés: o bloco que define o tamanho da figura. Face e mãos ficam de fora
# porque 110 dos 133 keypoints estão lá e eles deslocariam a escala para a
# cabeça.
BODY_AND_FEET = slice(0, 23)

# Ombros esquerdo e direito. O meio deles é a âncora porque é a junta mais
# estável da pose: o tronco não se alonga, e os quadris — o outro candidato —
# são justamente o que colapsa quando o estimador 2D os encosta na borda.
SHOULDERS = (5, 6)

# Altura útil do painel, em metros, e posição da âncora nele. Os dois saem da
# mesma conta: um homem de percentil 95 mede 1,87m em pé, com o acrômio a 1,54m,
# de modo que o corpo se estende 0,33m acima dos ombros e 1,46m abaixo. 1,8m de
# altura útil com a âncora a 0,18 do topo dão 0,32m acima e 1,48m abaixo — o
# mínimo que cabe o corpo inteiro em pé, e portanto a maior figura possível sem
# que a escala precise respirar.
#
# O preço da escala fixa é que o ocupante sentado, medido em 0,99m de extensão
# vertical nesta gravação, ocupa pouco mais da metade da altura — 229 px dos 404
# úteis. É barato perto do que custava enquadrar por quadro.
VIEW_HEIGHT_M = 1.8
ANCHOR_SCREEN = (0.5, 0.18)

# Distância 3D média das juntas 0..22 ao meio dos ombros num adulto, medida
# sobre os 290 quadros desta gravação com a escala de entrada já corrigida:
# média 0,424m, com a média por quadro entre 0,292 e 0,551. Serve de régua
# quando não há calibração — é grosseira, e por isso o rodapé do painel declara
# a escala como aproximada nesse caso.
MEAN_BODY_RADIUS_M = 0.424

# Elevação da vista, olhando de **cima**. Os eixos do H3WB seguem a convenção do
# Human3.6M: y cresce para baixo e z cresce afastando-se da câmera — verificado
# nesta gravação, em que o nariz de quem encara a câmera fica 0,170m à frente
# dos ombros. Com este sinal, y_tela = cos·y − sen·z, de modo que o que está
# mais longe sobe na tela, que é a vista natural. O sinal oposto, que estava
# aqui, olhava de baixo. A elevação é fixa porque uma vista muito de cima
# dificulta reconhecer a pose.
ELEVATION_RADIANS = np.deg2rad(12.0)

# Traço do que foi observado e do que foi predito pelo lifting.
OBSERVED_THICKNESS = 2
PREDICTED_THICKNESS = 1

# Quanto da cor original sobra depois de misturar com o fundo do painel. Ajustado
# olhando o painel renderizado: abaixo disto o membro predito some no fundo
# escuro, acima ele compete com o observado.
PREDICTED_BLEND = 0.45


def _rotation(azimuth: float) -> np.ndarray:
    """Rotação em torno do eixo vertical, seguida de uma elevação fixa."""
    cos_a, sin_a = np.cos(azimuth), np.sin(azimuth)
    around_y = np.array([[cos_a, 0.0, sin_a],
                         [0.0, 1.0, 0.0],
                         [-sin_a, 0.0, cos_a]])
    cos_e, sin_e = np.cos(ELEVATION_RADIANS), np.sin(ELEVATION_RADIANS)
    around_x = np.array([[1.0, 0.0, 0.0],
                         [0.0, cos_e, -sin_e],
                         [0.0, sin_e, cos_e]])
    return around_x @ around_y


def muted_color(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Mistura a cor com o fundo do painel, para o traço predito."""
    return tuple(int(round(fundo + (canal - fundo) * PREDICTED_BLEND))
                 for canal, fundo in zip(color, PANEL_FILL))


def _pixels_per_unit(keypoints_3d: np.ndarray, anchor: np.ndarray,
                     height: int, calibrated: bool) -> float:
    """Escala de desenho, em pixels por unidade da pose.

    Calibrada, a unidade é o metro e a escala é constante. Sem calibração ela
    sai do raio 3D do próprio corpo, que é invariante à rotação — ao contrário
    do envelope projetado, que muda com o ângulo e faz a figura pulsar.
    """
    pixels_per_metre = height / VIEW_HEIGHT_M
    if calibrated:
        return pixels_per_metre

    radius = float(np.linalg.norm(
        keypoints_3d[BODY_AND_FEET] - anchor, axis=-1).mean())
    if radius <= 0.0:
        return pixels_per_metre
    return pixels_per_metre * MEAN_BODY_RADIUS_M / radius


def project(keypoints_3d: np.ndarray, size: tuple[int, int], azimuth: float,
            calibrated: bool = False) -> np.ndarray:
    """Projeta a pose 3D em coordenadas da área de desenho.

    Args:
        keypoints_3d: [K, 3] relativo à raiz, em metros quando calibrado.
        size: (largura, altura) da área de desenho.
        azimuth: ângulo de rotação em radianos.
        calibrated: se a escala da pose é métrica.

    Returns:
        [K, 3]: duas colunas em pixels, com origem no canto superior esquerdo da
        área, e a terceira a profundidade depois da rotação — quem desenha
        precisa dela para ordenar, e recomputar a rotação lá seria duplicá-la.
    """
    width, height = size
    anchor = keypoints_3d[list(SHOULDERS)].mean(axis=0)
    scale = _pixels_per_unit(keypoints_3d, anchor, height, calibrated)

    rotated = (keypoints_3d - anchor) @ _rotation(azimuth).T
    origin = np.array([width, height]) * ANCHOR_SCREEN

    projected = np.empty_like(rotated)
    projected[:, :2] = rotated[:, :2] * scale + origin
    projected[:, 2] = rotated[:, 2]
    return projected


def draw_pose_3d(canvas: np.ndarray, keypoints_3d: np.ndarray,
                 origin: tuple[int, int], size: tuple[int, int],
                 azimuth: float, observed: np.ndarray | None = None,
                 calibrated: bool = False) -> None:
    """Desenha o esqueleto 3D dentro da região indicada de `canvas`.

    `observed` marca quais keypoints o sistema de fato observou neste quadro.
    Uma ligação só conta como observada quando as duas pontas são; as demais são
    desenhadas esmaecidas e finas. O lifting devolve posição para os 133
    keypoints sempre, inclusive para os que nunca apareceram na imagem, e a
    demonstração precisa mostrar o corpo inteiro **sem** apresentar predição
    como medida.
    """
    x, y = origin
    width, height = size
    # Desenhar numa fatia do canvas, e não no canvas inteiro com deslocamento:
    # a fatia é uma vista da mesma memória e o OpenCV recorta nela. Com escala
    # fixa uma junta pode cair fora do painel, e sem o recorte ela invadiria o
    # painel vizinho.
    region = canvas[y:y + height, x:x + width]

    projected = project(keypoints_3d, size, azimuth, calibrated)
    points = projected[:, :2].astype(np.int32)
    depth = projected[:, 2]

    # A profundidade ordena o desenho: o que está atrás sai primeiro e é
    # coberto pelo que está à frente, o que basta para dar oclusão correta sem
    # um buffer de profundidade.
    order = np.argsort([-(depth[a] + depth[b]) for a, b, _ in SKELETON_LINKS])

    for index in order:
        start, end, color = SKELETON_LINKS[index]
        seen = observed is None or (observed[start] and observed[end])
        cv2.line(region, tuple(points[start]), tuple(points[end]),
                 color if seen else muted_color(color),
                 OBSERVED_THICKNESS if seen else PREDICTED_THICKNESS,
                 cv2.LINE_AA)
