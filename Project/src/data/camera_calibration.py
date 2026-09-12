"""Calibração intrínseca de câmera a partir de um tabuleiro de xadrez.

Camada Model. Produz os mesmos campos que os arquivos de calibração do
Drive&Act, de modo que o resto do sistema não precise distinguir uma câmera
calibrada por nós de uma calibrada pelos autores do dataset.

**Por que isto importa neste projeto, e não é burocracia.** A escala que converte
a saída do Módulo 3 em metros vale `Z/fx`, com `fx` vindo da calibração. Herdar
o `fx` de outro dataset ampliou a pose em 3,8 vezes no domínio veicular, e o
erro absoluto ficou em 1,4 metro sem que a métrica adotada acusasse --- ela era
invariante a escala. Sem calibração da câmera própria, a demonstração ao vivo
carrega o mesmo defeito.

O tabuleiro é o alvo clássico porque seus cantos internos são detectáveis com
precisão subpixel e sua geometria é conhecida sem medição: os quadrados são
iguais entre si, e só o lado precisa ser informado.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Cantos **internos** do tabuleiro, que é o que a detecção encontra: um
# tabuleiro de 10 por 7 quadrados tem 9 por 6 cantos internos. Confundir os dois
# é o erro mais comum aqui, e faz a detecção falhar em todos os quadros.
DEFAULT_PATTERN = (9, 6)

# Abaixo disto o sistema de equações fica mal condicionado e os coeficientes de
# distorção saem instáveis, ainda que a calibração "funcione".
MIN_VIEWS = 10

# Critério de parada do refinamento subpixel dos cantos.
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CORNER_WINDOW = (11, 11)


@dataclass(frozen=True)
class Intrinsics:
    """Parâmetros intrínsecos, no formato dos arquivos do Drive&Act."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple[float, float, float, float, float]
    width: int
    height: int
    reprojection_error: float
    views: int

    def as_driveact_json(self) -> dict:
        """Mesma estrutura dos `.calibration.json` que acompanham o Drive&Act."""
        k1, k2, p1, p2, k3 = self.distortion
        return {
            'intrinsics': {
                'focallength': {'fx': self.fx, 'fy': self.fy},
                'principal_point': {'cx': self.cx, 'cy': self.cy},
                'img_size': {'width': self.width, 'height': self.height},
                'distortion': {'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2, 'k3': k3},
            },
            # Registrado junto porque um intrínseco sem o erro de reprojeção não
            # diz se a calibração prestou.
            'quality': {
                'reprojection_error_px': self.reprojection_error,
                'views': self.views,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_driveact_json(), indent=2))


def find_corners(image: np.ndarray,
                 pattern: tuple[int, int] = DEFAULT_PATTERN
                 ) -> np.ndarray | None:
    """Localiza os cantos internos do tabuleiro, com refinamento subpixel.

    Returns:
        [N, 1, 2] em pixels, ou `None` se o tabuleiro não aparece inteiro.
    """
    gray = (image if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    found, corners = cv2.findChessboardCorners(
        gray, pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, CORNER_WINDOW, (-1, -1),
                            CORNER_CRITERIA)


def calibrate(views: list[np.ndarray], image_size: tuple[int, int],
              pattern: tuple[int, int] = DEFAULT_PATTERN,
              square_size_m: float = 0.025) -> Intrinsics:
    """Resolve os intrínsecos a partir dos cantos detectados em várias vistas.

    Args:
        views: cantos de cada imagem, como `find_corners` devolve.
        image_size: (largura, altura) em pixels.
        square_size_m: lado do quadrado impresso, em metros. Ele fixa a escala
            do mundo; errá-lo não afeta `fx` nem a distorção, mas invalida
            qualquer distância que se meça depois.
    """
    if len(views) < MIN_VIEWS:
        raise ValueError(
            f'{len(views)} vistas é pouco para condicionar o sistema; '
            f'o mínimo é {MIN_VIEWS}')

    colunas, linhas = pattern
    modelo = np.zeros((linhas * colunas, 3), np.float32)
    modelo[:, :2] = np.mgrid[0:colunas, 0:linhas].T.reshape(-1, 2)
    modelo *= square_size_m

    erro, matriz, distorcao, _, _ = cv2.calibrateCamera(
        [modelo] * len(views), views, image_size, None, None)

    return Intrinsics(
        fx=float(matriz[0, 0]), fy=float(matriz[1, 1]),
        cx=float(matriz[0, 2]), cy=float(matriz[1, 2]),
        distortion=tuple(float(v) for v in distorcao.ravel()[:5]),
        width=image_size[0], height=image_size[1],
        reprojection_error=float(erro), views=len(views))


def chessboard_image(pattern: tuple[int, int] = DEFAULT_PATTERN,
                     square_px: int = 100, margin_px: int = 80) -> np.ndarray:
    """Gera o tabuleiro para impressão, com a contagem de cantos correta.

    Evita a busca por um PDF na internet e, mais importante, garante que o
    padrão impresso corresponda ao `pattern` que a detecção vai procurar.
    """
    colunas, linhas = pattern
    quadrados_x, quadrados_y = colunas + 1, linhas + 1
    tabuleiro = np.zeros((quadrados_y * square_px, quadrados_x * square_px),
                         np.uint8)
    for linha in range(quadrados_y):
        for coluna in range(quadrados_x):
            if (linha + coluna) % 2 == 0:
                tabuleiro[linha * square_px:(linha + 1) * square_px,
                          coluna * square_px:(coluna + 1) * square_px] = 255

    return cv2.copyMakeBorder(tabuleiro, margin_px, margin_px, margin_px,
                              margin_px, cv2.BORDER_CONSTANT, value=255)
