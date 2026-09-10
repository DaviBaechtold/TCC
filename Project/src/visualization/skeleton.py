"""Definição e desenho do esqueleto COCO-WholeBody.

Camada View: recebe keypoints já estimados e desenha. Não decide o que é
válido — o limiar de confiança chega como argumento, vindo de quem chamou.
"""

from __future__ import annotations

import cv2
import numpy as np

# Índices de início de cada bloco no vetor de 133 keypoints do COCO-WholeBody.
BODY_START, FEET_START, FACE_START = 0, 17, 23
LEFT_HAND_START, RIGHT_HAND_START = 91, 112

# Cores em BGR, que é a ordem de canais do OpenCV.
COLOR_TORSO = (180, 190, 120)
COLOR_LEFT_ARM = (90, 200, 250)
COLOR_RIGHT_ARM = (250, 170, 90)
COLOR_LEFT_LEG = (120, 230, 160)
COLOR_RIGHT_LEG = (200, 140, 230)
COLOR_HEAD = (230, 230, 230)
COLOR_FACE = (150, 220, 255)
COLOR_LEFT_HAND = (110, 210, 255)
COLOR_RIGHT_HAND = (255, 190, 110)


def _chain(start: int, count: int) -> list[tuple[int, int]]:
    """Liga pontos consecutivos: usado em contornos abertos como o queixo."""
    return [(start + i, start + i + 1) for i in range(count - 1)]


def _loop(start: int, count: int) -> list[tuple[int, int]]:
    """Liga pontos consecutivos e fecha o ciclo: olhos e lábios."""
    return _chain(start, count) + [(start + count - 1, start)]


def _hand_links(base: int) -> list[tuple[int, int]]:
    """Cinco dedos partindo do punho, quatro segmentos cada."""
    links = []
    for finger in range(5):
        first_joint = base + 1 + finger * 4
        links.append((base, first_joint))
        links.extend(_chain(first_joint, 4))
    return links


def _build_links() -> list[tuple[int, int, tuple[int, int, int]]]:
    """Monta a lista de (início, fim, cor) do esqueleto completo."""
    links: list[tuple[int, int, tuple[int, int, int]]] = []

    def add(pairs, color):
        links.extend((a, b, color) for a, b in pairs)

    add([(5, 6), (5, 11), (6, 12), (11, 12)], COLOR_TORSO)
    add([(5, 7), (7, 9)], COLOR_LEFT_ARM)
    add([(6, 8), (8, 10)], COLOR_RIGHT_ARM)
    add([(11, 13), (13, 15)], COLOR_LEFT_LEG)
    add([(12, 14), (14, 16)], COLOR_RIGHT_LEG)
    add([(0, 1), (0, 2), (1, 3), (2, 4)], COLOR_HEAD)

    # Pés: tornozelo até dedões e calcanhar.
    add([(15, 17), (15, 18), (15, 19)], COLOR_LEFT_LEG)
    add([(16, 20), (16, 21), (16, 22)], COLOR_RIGHT_LEG)

    # Face, no layout padrão de 68 landmarks deslocado para a base do bloco.
    face = FACE_START
    add(_chain(face, 17), COLOR_FACE)          # contorno do queixo
    add(_chain(face + 17, 5), COLOR_FACE)      # sobrancelha direita
    add(_chain(face + 22, 5), COLOR_FACE)      # sobrancelha esquerda
    add(_chain(face + 27, 4), COLOR_FACE)      # dorso do nariz
    add(_chain(face + 31, 5), COLOR_FACE)      # narinas
    add(_loop(face + 36, 6), COLOR_FACE)       # olho direito
    add(_loop(face + 42, 6), COLOR_FACE)       # olho esquerdo
    add(_loop(face + 48, 12), COLOR_FACE)      # lábio externo
    add(_loop(face + 60, 8), COLOR_FACE)       # lábio interno

    add(_hand_links(LEFT_HAND_START), COLOR_LEFT_HAND)
    add(_hand_links(RIGHT_HAND_START), COLOR_RIGHT_HAND)

    return links


SKELETON_LINKS = _build_links()


def draw_pose(canvas: np.ndarray,
              keypoints: np.ndarray,
              scores: np.ndarray,
              min_score: float,
              point_radius: int = 2,
              line_thickness: int = 2) -> None:
    """Desenha um esqueleto sobre o canvas, no lugar.

    Args:
        canvas: [H, W, 3] BGR, modificado no lugar.
        keypoints: [133, 2] em pixels.
        scores: [133] confiança por keypoint.
        min_score: abaixo disto o keypoint é tratado como não detectado.
    """
    visible = scores >= min_score

    for start, end, color in SKELETON_LINKS:
        if not (visible[start] and visible[end]):
            continue
        cv2.line(canvas,
                 tuple(np.round(keypoints[start]).astype(int)),
                 tuple(np.round(keypoints[end]).astype(int)),
                 color, line_thickness, cv2.LINE_AA)

    # Keypoints de face e mãos são densos: um raio menor evita que virem um
    # borrão sólido que esconde a estrutura que as linhas já mostram.
    for index in np.flatnonzero(visible):
        dense = index >= FACE_START
        cv2.circle(canvas,
                   tuple(np.round(keypoints[index]).astype(int)),
                   1 if dense else point_radius,
                   COLOR_FACE if dense else (255, 255, 255),
                   -1, cv2.LINE_AA)


def draw_box(canvas: np.ndarray,
             box: np.ndarray,
             color: tuple[int, int, int] = (90, 90, 90),
             thickness: int = 1) -> None:
    """Desenha a caixa da pessoa detectada."""
    x1, y1, x2, y2 = np.round(box).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
