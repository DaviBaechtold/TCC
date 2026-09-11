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

O enquadramento é recalculado a cada quadro a partir da própria pose, de modo
que o esqueleto ocupe o painel independentemente da escala em que chegou. Fixar
a escala faria a figura encolher a nada sempre que o fator de calibração
estivesse ausente.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.visualization.skeleton import SKELETON_LINKS

# Fração do painel deixada como margem ao redor do esqueleto.
MARGIN_RATIO = 0.12

# Os eixos do H3WB seguem a convenção do Human3.6M: y cresce para baixo e z é a
# profundidade. Girar em torno de y é o que dá a sensação de volume; a elevação
# fica fixa porque uma vista de cima dificulta reconhecer a pose.
ELEVATION_RADIANS = np.deg2rad(-12.0)


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


def project(keypoints_3d: np.ndarray, size: tuple[int, int],
            azimuth: float) -> np.ndarray:
    """Projeta a pose 3D em coordenadas de tela, ajustando o enquadramento.

    Args:
        keypoints_3d: [K, 3] relativo à raiz.
        size: (largura, altura) da área de desenho.
        azimuth: ângulo de rotação em radianos.

    Returns:
        [K, 2] em pixels, com origem no canto superior esquerdo da área.
    """
    width, height = size
    rotated = keypoints_3d @ _rotation(azimuth).T
    plane = rotated[:, :2]

    lower, upper = plane.min(axis=0), plane.max(axis=0)
    extent = np.maximum(upper - lower, 1e-6)
    usable = np.array([width, height], dtype=np.float64) * (1 - 2 * MARGIN_RATIO)
    scale = float(np.min(usable / extent))

    # Centraliza pelo envelope, e não pela média: dos 133 keypoints, 110 estão
    # na face e nas mãos, e a média seria puxada para a cabeça, deixando o
    # esqueleto encostado na borda do painel.
    centered = (plane - (lower + upper) / 2.0) * scale
    return centered + np.array([width, height]) / 2.0


def draw_pose_3d(canvas: np.ndarray, keypoints_3d: np.ndarray,
                 origin: tuple[int, int], size: tuple[int, int],
                 azimuth: float) -> None:
    """Desenha o esqueleto 3D dentro da região indicada de `canvas`."""
    projected = project(keypoints_3d, size, azimuth) + np.array(origin)
    points = projected.astype(np.int32)

    # A profundidade ordena o desenho: o que está atrás sai primeiro e é
    # coberto pelo que está à frente, o que basta para dar oclusão correta sem
    # um buffer de profundidade.
    depth = (keypoints_3d @ _rotation(azimuth).T)[:, 2]
    order = np.argsort([-(depth[a] + depth[b]) for a, b, _ in SKELETON_LINKS])

    for index in order:
        start, end, color = SKELETON_LINKS[index]
        cv2.line(canvas, tuple(points[start]), tuple(points[end]),
                 color, 1, cv2.LINE_AA)
